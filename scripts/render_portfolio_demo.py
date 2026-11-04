"""Assemble a silent, captioned portfolio draft from real evidence assets.

Requires ffmpeg. The source clip and final MP4 stay under ignored docs/CRM.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs/CRM"
SIZE = "1280:720"
FPS = "25"


def render_segment(source: Path, target: Path, seconds: float, caption: str | None,
                   *, still: bool = False, crop: str | None = None) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not font.is_file():
        raise FileNotFoundError(font)
    filters = []
    if crop:
        filters.append(f"crop={crop}")
    filters += [f"scale={SIZE}:force_original_aspect_ratio=decrease",
                f"pad={SIZE}:(ow-iw)/2:(oh-ih)/2:0x0b1612"]
    if caption:
        filters.append(f"drawtext=fontfile={font}:text='{caption}':"
                       "fontcolor=white:fontsize=22:x=24:y=684:"
                       "box=1:boxcolor=0x0b1612@0.88:boxborderw=8")
    filtergraph = ",".join(filters)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if still:
        command += ["-loop", "1"]
    command += ["-i", str(source), "-t", str(seconds), "-vf", filtergraph,
                "-r", FPS, "-an", "-c:v", "libx264", "-preset", "medium",
                "-crf", "22", "-pix_fmt", "yuv420p", str(target)]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=MEDIA / "videos/01-edge-sorter-demo.mp4")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    segments = [
        (MEDIA / "photos/01-dashboard-material-policy.png", 5.0,
         None, True, "1440:900:0:330"),
        (MEDIA / "videos/tracker_review/real_motion_default.mp4", 9.0,
         "Pexels footage by Peter Hanzo | tracker overlay | no actuator", False, None),
        (MEDIA / "photos/02-accuracy-parity.png", 5.0,
         None, True, None),
        (MEDIA / "photos/03-latency-scopes.png", 5.0,
         None, True, None),
    ]
    with tempfile.TemporaryDirectory(prefix="edge-demo-", dir=args.out.parent) as directory:
        folder = Path(directory)
        outputs = []
        for index, (source, seconds, caption, still, crop) in enumerate(segments, 1):
            target = folder / f"{index:02d}.mp4"
            render_segment(source, target, seconds, caption, still=still, crop=crop)
            outputs.append(target)
        list_file = folder / "segments.txt"
        list_file.write_text("".join(f"file '{path}'\n" for path in outputs))
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "concat", "-safe", "0", "-i", str(list_file),
                        "-c", "copy", "-movflags", "+faststart", str(args.out)], check=True)
    print(json.dumps({"out": str(args.out), "duration_s_approx": sum(x[1] for x in segments),
                      "source": "Pexels video by Peter Hanzo, edited with project overlays",
                      "scope": "Portfolio draft, not a live synchronized recording"}, indent=2))


if __name__ == "__main__":
    main()
