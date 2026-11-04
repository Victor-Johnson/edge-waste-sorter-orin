"""Create an honest, repeatable video replay from selected WaRP-D images."""

import argparse
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/replay.yaml")
    parser.add_argument("--out", default="data/sample_video/warp-d-replay.mp4")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    fps, seconds = int(cfg["fps"]), int(cfg["seconds_per_image"])
    width, height = int(cfg["width"]), int(cfg["height"])
    if min(fps, seconds, width, height) < 1:
        raise ValueError("Replay dimensions, frame rate, and duration must be positive")
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {output}")
    count = 0
    try:
        for number, relative in enumerate(cfg["images"], 1):
            path = ROOT / relative
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            base = cv2.resize(image, (width, height))
            for frame_index in range(fps * seconds):
                progress = frame_index / max(fps * seconds - 1, 1)
                transform = cv2.getRotationMatrix2D((width / 2, height / 2), 0, 1 + .04 * progress)
                frame = cv2.warpAffine(base, transform, (width, height))
                cv2.rectangle(frame, (0, 0), (width, 44), (7, 17, 15), -1)
                cv2.putText(frame, f"WaRP-D DATASET REPLAY  |  example {number}/{len(cfg['images'])}",
                            (16, 29), cv2.FONT_HERSHEY_SIMPLEX, .65, (190, 240, 170), 2)
                writer.write(frame)
                count += 1
    finally:
        writer.release()
    print(f"Created {output} with {count} frames at {fps} FPS")


if __name__ == "__main__":
    main()
