"""Run the Jetson video probe while recording tegrastats power and temperature."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

TIMESTAMP = re.compile(r"^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})")
POWER = re.compile(r"\bVDD_IN (\d+)mW/(\d+)mW\b")
GPU_TEMP = re.compile(r"\bgpu@([\d.]+)C\b")
GPU_UTIL = re.compile(r"\bGR3D_FREQ (\d+)%")


def parse_sample(line: str, local_tz: timezone) -> dict | None:
    stamp = TIMESTAMP.search(line)
    power = POWER.search(line)
    if not stamp or not power:
        return None
    local_time = datetime.strptime(stamp.group(1), "%m-%d-%Y %H:%M:%S")
    sample = {"timestamp_utc": local_time.replace(tzinfo=local_tz).astimezone(timezone.utc),
              "vdd_in_w": int(power.group(1)) / 1000}
    if match := GPU_TEMP.search(line):
        sample["gpu_temp_c"] = float(match.group(1))
    if match := GPU_UTIL.search(line):
        sample["gpu_util_pct"] = int(match.group(1))
    return sample


def summarize(samples: list[dict], key: str) -> dict | None:
    values = [sample[key] for sample in samples if key in sample]
    if not values:
        return None
    return {"mean": round(statistics.mean(values), 2),
            "min": round(min(values), 2), "max": round(max(values), 2)}


def within(samples: list[dict], start: datetime, end: datetime) -> list[dict]:
    # Tegrastats timestamps have second precision. Exclude boundary seconds.
    first_whole_second = start.replace(microsecond=0) + timedelta(seconds=1)
    last_whole_second = end.replace(microsecond=0)
    return [sample for sample in samples
            if first_whole_second <= sample["timestamp_utc"] < last_whole_second]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=280)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--confidence", type=float, default=.35)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--interval-ms", type=int, default=500)
    parser.add_argument("--idle-seconds", type=int, default=5)
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0 or args.interval_ms < 100 or args.idle_seconds < 3:
        parser.error("frames >= 1, warmup >= 0, interval-ms >= 100, idle-seconds >= 3 required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw_log = args.out.with_suffix(".tegrastats.log")
    video_result = args.out.with_suffix(".video.json")
    local_tz = timezone(datetime.now().astimezone().utcoffset())
    telemetry_start = datetime.now(timezone.utc)
    with raw_log.open("w") as log_file:
        telemetry = subprocess.Popen(["tegrastats", "--interval", str(args.interval_ms)],
                                     stdout=log_file, stderr=subprocess.STDOUT)
        try:
            time.sleep(args.idle_seconds)
            idle_end = datetime.now(timezone.utc)
            command = [sys.executable, str(Path(__file__).with_name("jetson_video_probe.py")),
                       "--model", str(args.model), "--video", str(args.video),
                       "--classes", str(args.classes), "--out", str(video_result),
                       "--frames", str(args.frames), "--warmup", str(args.warmup),
                       "--confidence", str(args.confidence)]
            if args.track:
                command.append("--track")
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        finally:
            telemetry.terminate()
            try:
                telemetry.wait(timeout=5)
            except subprocess.TimeoutExpired:
                telemetry.kill()
                telemetry.wait()

    samples = [sample for line in raw_log.read_text().splitlines()
               if (sample := parse_sample(line, local_tz)) is not None]
    video = json.loads(video_result.read_text())
    active_start = datetime.fromisoformat(video["measurement_start_utc"])
    active_end = datetime.fromisoformat(video["measurement_end_utc"])
    idle = within(samples, telemetry_start, idle_end)
    active = within(samples, active_start, active_end)
    if not idle or not active:
        raise RuntimeError(f"Too few aligned tegrastats samples; inspect {raw_log}")
    mode = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True,
                          check=True).stdout.splitlines()[0]
    video["power_mode"] = mode
    video["power"] = {
        "rail": "VDD_IN", "unit": "W", "sample_interval_ms": args.interval_ms,
        "idle_sample_count": len(idle), "active_sample_count": len(active),
        "idle_vdd_in_w": summarize(idle, "vdd_in_w"),
        "active_vdd_in_w": summarize(active, "vdd_in_w"),
        "active_gpu_temp_c": summarize(active, "gpu_temp_c"),
        "active_gpu_util_pct": summarize(active, "gpu_util_pct"),
        "scope": "Jetson input rail, not wall-plug power; one-second timestamp precision",
    }
    video["telemetry_raw_log"] = raw_log.name
    args.out.write_text(json.dumps(video, indent=2) + "\n")
    print(json.dumps({"result": str(args.out), "raw_log": str(raw_log),
                      "fps": video["end_to_end_fps"],
                      "idle_vdd_in_mean_w": video["power"]["idle_vdd_in_w"]["mean"],
                      "active_vdd_in_mean_w": video["power"]["active_vdd_in_w"]["mean"],
                      "active_gpu_temp_max_c": (video["power"]["active_gpu_temp_c"] or {}).get("max"),
                      "active_samples": len(active)}, indent=2))


if __name__ == "__main__":
    main()
