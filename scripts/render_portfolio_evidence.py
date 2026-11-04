"""Render two portfolio figures from committed benchmark JSON (no inference)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/edge-waste-sorter-matplotlib")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BG = "#0b1612"
FG = "#e9f1e9"
MUTED = "#a9bfb0"
GREEN = "#a8f15d"
TEAL = "#65d2bd"
AMBER = "#f2c879"


def record(name: str) -> dict:
    return json.loads((ROOT / "benchmarks/results" / name).read_text())


def canvas():
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=180)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    return fig, ax


def accuracy(out: Path) -> None:
    pt = record("val_pt.json")
    onnx = record("val_onnx.json")
    fp16 = record("val_jetson_fp16.json")
    labels = ["PyTorch\nDesktop", "ONNX Runtime\nDesktop", "TensorRT FP16\nJetson"]
    values = [pt["mAP50_95"], onnx["mAP50_95"], fp16["mAP50_95"]]
    fig, ax = canvas()
    bars = ax.bar(labels, values, color=[GREEN, TEAL, AMBER], width=.55)
    ax.set_ylim(0, .55)
    ax.set_ylabel("mAP50–95", color=FG, fontsize=14, labelpad=15)
    ax.yaxis.grid(True, color="#2e4d3d", alpha=.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=FG, labelsize=13, pad=16)
    ax.tick_params(axis="y", colors=MUTED, labelsize=11)
    for bar, value in zip(bars, values):
        label = f"{value:.5f}" if value != values[-1] else f"{value:.3f}*"
        ax.text(bar.get_x() + bar.get_width()/2, value + .017, label,
                ha="center", color=FG, fontsize=16, fontweight="bold")
    fig.suptitle("Detector accuracy survives export", x=.09, y=.96,
                 ha="left", color=FG, fontsize=25, fontweight="bold")
    fig.text(.09, .875, "Same 522-image WaRP-D test split · 640 px · batch 1",
             color=MUTED, fontsize=13)
    fig.text(.09, .035, "* Jetson result was reported to three decimals. Desktop used Ultralytics 8.4.171; Jetson used 8.3.59.",
             color=MUTED, fontsize=10)
    fig.subplots_adjust(left=.13, right=.94, top=.78, bottom=.2)
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def latency(out: Path) -> None:
    engine = record("jetson_fp16_trtexec.json")
    predict = record("jetson_replay_fp16_predict_pair.json")
    track = record("jetson_replay_fp16_track.json")
    labels = ["TensorRT engine", "Video + prediction", "Video + tracking"]
    values = [engine["latency_mean_ms"], predict["pipeline_ms"]["mean"],
              track["pipeline_ms"]["mean"]]
    fig, ax = canvas()
    bars = ax.barh(labels, values, color=[GREEN, TEAL, AMBER], height=.52)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("Mean milliseconds per query or frame", color=FG, fontsize=14, labelpad=15)
    ax.xaxis.grid(True, color="#2e4d3d", alpha=.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", colors=FG, labelsize=13, pad=16)
    ax.tick_params(axis="x", colors=MUTED, labelsize=11)
    for bar, value in zip(bars, values):
        ax.text(value + 2, bar.get_y() + bar.get_height()/2, f"{value:.1f} ms",
                va="center", color=FG, fontsize=15, fontweight="bold")
    fig.suptitle("The edge system has more than one clock", x=.09, y=.96,
                 ha="left", color=FG, fontsize=25, fontweight="bold")
    fig.text(.09, .875, "Jetson Orin Nano · FP16 · 15 W mode", color=MUTED, fontsize=13)
    fig.text(.09, .035, "Engine uses generated inputs; video rows use a WaRP-D replay. Video includes decode, draw, JPEG; tracking row adds BoT-SORT. Network excluded.",
             color=MUTED, fontsize=10)
    fig.subplots_adjust(left=.30, right=.94, top=.78, bottom=.2)
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs/CRM/photos")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    first = args.out_dir / "02-accuracy-parity.png"
    second = args.out_dir / "03-latency-scopes.png"
    accuracy(first)
    latency(second)
    print(f"Rendered {first} and {second}")


if __name__ == "__main__":
    main()
