"""Render a plain-language, silent portfolio story from verified project media.

The edited video is a narrative montage. Its footage, tracker view, and final
dashboard screenshot are not a synchronized recording. Requires ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs/CRM"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
BG = "0x0b1612"
WHITE = "0xe9f1e9"
GREEN = "0xa8f15d"


def caption(folder: Path, name: str, words: str) -> Path:
    path = folder / f"{name}.txt"
    path.write_text(words)
    return path


def title(path: Path, *, size: int, x: int, y: int, colour: str = WHITE,
          box: bool = False) -> str:
    filters = []
    for index, words in enumerate(path.read_text().split("\n")):
        if not words:
            continue
        line = path.with_name(f"{path.stem}_{index}.txt")
        line.write_text(words)
        options = (f"drawtext=fontfile={FONT}:textfile={line}:fontcolor={colour}:"
                   f"fontsize={size}:x={x}:y={y + index * (size + 13)}")
        if box:
            options += f":box=1:boxcolor={BG}@0.88:boxborderw=12"
        filters.append(options)
    return ",".join(filters)


def encode(inputs: list[list[str]], graph: str, duration: int, out: Path) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for arguments in inputs:
        command.extend(arguments)
    command += ["-filter_complex", graph, "-map", "[v]", "-t", str(duration),
                "-r", "25", "-an", "-c:v", "libx264", "-preset", "medium",
                "-crf", "21", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(command, check=True)


def scene_one(folder: Path, raw_video: Path, out: Path) -> None:
    labels = [
        title(caption(folder, "one_kicker", "THE PROBLEM"), size=24, x=455, y=100, colour=GREEN),
        title(caption(folder, "one_head", "Waste moves fast.\nOne mistake matters."), size=58, x=455, y=175),
        title(caption(folder, "one_body", "On a busy recycling belt,\nitems overlap and move."),
              size=32, x=455, y=405),
        title(caption(folder, "one_credit", "Footage: Peter Hanzo / Pexels"),
              size=19, x=455, y=660, colour=GREEN),
    ]
    graph = ("[0:v]fps=25,scale=405:720,setsar=1[footage];"
             f"color=c={BG}:s=1280x720:r=25:d=6[base];"
             "[base][footage]overlay=0:0:shortest=1[screen];"
             f"[screen]{','.join(labels)}[v]")
    encode([["-ss", "0", "-t", "6", "-i", str(raw_video)]], graph, 6, out)


def scene_two(folder: Path, raw_video: Path, tracked_video: Path, out: Path) -> None:
    labels = [
        title(caption(folder, "two_camera", "CAMERA VIEW"), size=19, x=20, y=23, box=True),
        title(caption(folder, "two_model", "MODEL VIEW"), size=19, x=425, y=23, box=True),
        title(caption(folder, "two_kicker", "WHAT THE CAMERA SEES"),
              size=21, x=840, y=105, colour=GREEN),
        title(caption(folder, "two_head", "It spots items\nand follows them."),
              size=41, x=840, y=170),
        title(caption(folder, "two_body", "The box marks a bottle.\nThe software checks it\nagain as it moves."),
              size=27, x=840, y=365),
    ]
    graph = ("[0:v]fps=25,scale=405:720,setsar=1[raw];"
             "[1:v]fps=25,scale=405:720,setsar=1[tracked];"
             "[raw][tracked]hstack=inputs=2[duo];"
             f"color=c={BG}:s=1280x720:r=25:d=7[base];"
             "[base][duo]overlay=0:0:shortest=1[screen];"
             f"[screen]{','.join(labels)}[v]")
    inputs = [["-ss", "2", "-t", "7", "-i", str(raw_video)],
              ["-ss", "2", "-t", "7", "-i", str(tracked_video)]]
    encode(inputs, graph, 7, out)


def scene_three(folder: Path, screenshot: Path, out: Path) -> None:
    labels = [
        title(caption(folder, "three_kicker", "THE DECISION"),
              size=21, x=850, y=110, colour=GREEN),
        title(caption(folder, "three_head", "Plastic, or\nhuman review?"),
              size=43, x=850, y=170),
        title(caption(folder, "three_body", "The software checks\nmore than one view.\n\nIf unsure, it asks\na person to check."),
              size=28, x=850, y=330),
        title(caption(folder, "three_note", "Advice only.\nNo sorting machine is connected."),
              size=19, x=850, y=626, colour=GREEN),
    ]
    graph = ("[0:v]crop=1440:900:0:330,scale=780:488,setsar=1[shot];"
             f"color=c={BG}:s=1280x720:r=25:d=9[base];"
             "[base][shot]overlay=40:116:shortest=1[screen];"
             f"[screen]{','.join(labels)}[v]")
    encode([["-loop", "1", "-framerate", "25", "-i", str(screenshot)]], graph, 9, out)


def scene_four(folder: Path, failure_still: Path, out: Path) -> None:
    labels = [
        title(caption(folder, "four_kicker", "A FAILURE WE FOUND"),
              size=24, x=70, y=90, colour=GREEN),
        title(caption(folder, "four_head", "One bottle.\nCounted twice."),
              size=58, x=70, y=160),
        title(caption(folder, "four_body", "The software can make this mistake.\nWe found it in real footage and kept\nthe suggestions visible for review."),
              size=28, x=70, y=400),
        title(caption(folder, "four_next", "Next: test on longer, fully labelled footage."),
              size=21, x=70, y=650, colour=GREEN),
    ]
    graph = ("[0:v]scale=405:720,setsar=1[still];"
             f"color=c={BG}:s=1280x720:r=25:d=7[base];"
             "[base][still]overlay=875:0:shortest=1[screen];"
             f"[screen]{','.join(labels)}[v]")
    encode([["-loop", "1", "-framerate", "25", "-i", str(failure_still)]],
           graph, 7, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=MEDIA / "videos/01-edge-sorter-demo.mp4")
    args = parser.parse_args()
    if not FONT.is_file():
        raise FileNotFoundError(FONT)
    raw = ROOT / "data/sample_video/pexels-plastic-bottles-13662401.mp4"
    tracked = MEDIA / "videos/tracker_review/real_motion_default.mp4"
    screenshot = MEDIA / "photos/01-dashboard-material-policy.png"
    failure = MEDIA / "photos/04-duplicate-track-frame242.png"
    for path in (raw, tracked, screenshot, failure):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="edge-story-", dir=args.out.parent) as directory:
        folder = Path(directory)
        segments = [folder / f"{index:02d}.mp4" for index in range(1, 5)]
        scene_one(folder, raw, segments[0])
        scene_two(folder, raw, tracked, segments[1])
        scene_three(folder, screenshot, segments[2])
        scene_four(folder, failure, segments[3])
        listing = folder / "segments.txt"
        listing.write_text("".join(f"file '{path}'\n" for path in segments))
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", "-movflags", "+faststart", str(args.out)], check=True)
    print(json.dumps({"out": str(args.out), "duration_s": 29,
                      "story": "problem -> camera view -> route or review -> observed failure",
                      "scope": "Silent edited narrative; no synchronized dashboard recording or actuator"},
                     indent=2))


if __name__ == "__main__":
    main()
