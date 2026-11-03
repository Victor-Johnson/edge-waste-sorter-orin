"""
host/export/export_onnx.py

Exports the trained YOLO checkpoint to a fixed-shape, simplified ONNX graph,
verifies that graph simplification did not change the model's outputs,
and writes an export record to benchmarks/results/export_onnx.json.

Run from the repo root:
    uv run python host/export/export_onnx.py
    uv run python host/export/export_onnx.py --weights exports/other.pt   # optional override
"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxslim

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── helpers ──────────────────────────────────────────────────────────────────
def load_config() -> dict:
    import yaml

    with open(REPO_ROOT / "configs" / "project.yaml") as f:
        return yaml.safe_load(f)


def repo_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def graph_summary(path: Path) -> dict:
    """Node count + operator histogram: the evidence for what slimming actually did."""
    proto = onnx.load(str(path))
    ops = Counter(node.op_type for node in proto.graph.node)
    return {
        "file": path.name,
        "sha256": sha256(path),
        "size_mb": round(path.stat().st_size / 1e6, 2),
        "opset": max(o.version for o in proto.opset_import if o.domain in ("", "ai.onnx")),
        "node_count": len(proto.graph.node),
        "op_histogram": dict(sorted(ops.items(), key=lambda kv: -kv[1])),
    }


def io_signature(session: ort.InferenceSession) -> dict:
    """All inputs and outputs, not just the first: YOLO heads can expose several."""
    describe = lambda nodes: [{"name": n.name, "shape": n.shape, "dtype": n.type} for n in nodes]
    return {"inputs": describe(session.get_inputs()), "outputs": describe(session.get_outputs())}


def run_onnx(path: Path, x: np.ndarray) -> list[np.ndarray]:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(None, {sess.get_inputs()[0].name: x})


def compare_graphs(raw: Path, slim: Path, imgsz: int, seed: int = 0) -> dict:
    """Same input through raw and slimmed graphs -> outputs must match.

    Input is image-like ([0, 1], fixed seed) rather than randn, so values stay in the
    range the model was trained on and the check is reproducible.
    """
    x = np.random.default_rng(seed).random((1, 3, imgsz, imgsz), dtype=np.float32)
    out_raw, out_slim = run_onnx(raw, x), run_onnx(slim, x)

    per_output = []
    for i, (a, b) in enumerate(zip(out_raw, out_slim)):
        per_output.append({
            "index": i,
            "shape": list(a.shape),
            "max_abs_diff": float(np.max(np.abs(a - b))) if a.size else 0.0,
            "all_finite": bool(np.isfinite(b).all()),
        })
    passed = (
        len(out_raw) == len(out_slim)
        and all(np.allclose(a, b, rtol=1e-3, atol=1e-3) for a, b in zip(out_raw, out_slim))
        and all(o["all_finite"] for o in per_output)
    )
    return {"input": f"uniform[0,1], seed={seed}", "passed": passed, "outputs": per_output}


# ── main pipeline ────────────────────────────────────────────────────────────
def export_and_optimize(weights: Path, out: Path, imgsz: int, opset: int, results_dir: Path) -> Path:
    from ultralytics import YOLO
    import torch
    import ultralytics

    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {weights}")
    out.mkdir(parents=True, exist_ok=True)

    raw_onnx = out / f"{weights.stem}_raw.onnx"
    final_onnx = out / f"{weights.stem}_static{imgsz}_opset{opset}.onnx"

    print(f"\n[1/5] Loading checkpoint: {weights.name}")
    model = YOLO(str(weights))

    print(f"[2/5] Exporting static ONNX (1x3x{imgsz}x{imgsz}, opset={opset})...")
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        batch=1,           # explicit: TensorRT gets exactly one input shape
        dynamic=False,     # static dims -> TensorRT can pick shape-specialised kernels
        half=False,        # keep FP32; precision is chosen at TensorRT engine build time
        simplify=False,    # slimming is done explicitly below so it can be measured
        opset=opset,
        device="cpu",      # deterministic CPU tracing; export needs no GPU
    )
    Path(exported).replace(raw_onnx)
    print(f"      Raw graph: {raw_onnx.name}")

    print("[3/5] Simplifying graph with onnxslim...")
    onnx.save(onnxslim.slim(str(raw_onnx)), str(final_onnx))
    onnx.checker.check_model(str(final_onnx))
    raw_info, slim_info = graph_summary(raw_onnx), graph_summary(final_onnx)
    print(f"      Nodes: {raw_info['node_count']} -> {slim_info['node_count']}")

    print("[4/5] Checking raw vs slimmed outputs match...")
    check = compare_graphs(raw_onnx, final_onnx, imgsz)
    for o in check["outputs"]:
        print(f"      output[{o['index']}] shape={o['shape']} max|diff|={o['max_abs_diff']:.2e}")
    print(f"      Slim parity: {'PASS' if check['passed'] else 'FAIL'}")

    sess = ort.InferenceSession(str(final_onnx), providers=["CPUExecutionProvider"])
    signature = io_signature(sess)
    for n in signature["inputs"]:
        print(f"      Input : '{n['name']}' {n['shape']} {n['dtype']}")
    for n in signature["outputs"]:
        print(f"      Output: '{n['name']}' {n['shape']} {n['dtype']}")

    print("[5/5] Writing export record...")
    record = {
        "stage": "onnx_export",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {"file": weights.name, "sha256": sha256(weights)},
        "export_settings": {"imgsz": imgsz, "batch": 1, "dynamic": False,
                            "half": False, "opset": opset, "device": "cpu"},
        "io_signature": signature,
        "raw_graph": raw_info,
        "slim_graph": slim_info,
        "slim_parity": check,
        "environment": {
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "onnxslim": onnxslim.__version__,
        },
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    record_path = results_dir / "export_onnx.json"
    record_path.write_text(json.dumps(record, indent=2))
    print(f"      Record: {record_path.relative_to(REPO_ROOT)}")

    if not check["passed"]:
        raise RuntimeError("Slimmed graph diverges from raw export; see export_onnx.json before using it.")

    print(f"\nExport successful: {final_onnx.relative_to(REPO_ROOT)}")
    print("Next: ONNX vs PyTorch accuracy parity (tests/test_onnx_parity.py).")
    return final_onnx


if __name__ == "__main__":
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Export YOLO checkpoint to static, simplified ONNX")
    parser.add_argument("--weights", default=cfg["checkpoint"], help="Checkpoint (default: configs/project.yaml)")
    parser.add_argument("--outdir", default=cfg["exports_dir"], help="Output directory (default: exports/)")
    parser.add_argument("--imgsz", type=int, default=cfg["imgsz"])
    parser.add_argument("--opset", type=int, default=cfg["opset"])
    args = parser.parse_args()

    export_and_optimize(
        weights=repo_path(args.weights),
        out=repo_path(args.outdir),
        imgsz=args.imgsz,
        opset=args.opset,
        results_dir=repo_path(cfg["results_dir"]),
    )