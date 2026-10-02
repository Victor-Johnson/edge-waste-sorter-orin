import hashlib
import json 
import platform
from datetime import datetime , timezone
from pathlib import Path

# ML-libs imports 
import torch
import ultralytics
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]


# Loading Configs from Yaml files
def load_config() -> dict:
    with open(REPO_ROOT / "configs" / "project.yaml") as f:
        return yaml.safe_load(f)

def repo_path(p) -> Path:
    """
    Absolute paths pass through , relative ones are held at the repo root 
    """
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT/ p 

def sha256(path, chunk:int=1 <<20)->str:
    """Fingerprint the checkpoint so every result is traceable to exact weights."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
 
 
def main() -> None:
    cfg = load_config()
    ckpt = repo_path(cfg["checkpoint"])
    v = cfg["val"]
 
    metrics = YOLO(str(ckpt)).val(
        data=str(repo_path(cfg["dataset_yaml"])),
        imgsz=cfg["imgsz"],
        batch=v["batch"],
        rect=v["rect"],
        split=v["split"],
        conf=v["conf"],
        iou=v["iou"],
        project=str(repo_path(cfg["runs_dir"])),
        name="baseline_pytorch_fp32",
        exist_ok=True,
        plots=False,
    )
 
    # Per-class mAP50-95 
    per_class = {
        metrics.names[int(c)]: round(float(metrics.box.maps[int(c)]), 5)
        for c in metrics.box.ap_class_index
    }
 
    result = {
        "model": "pytorch_fp32",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": ckpt.name,
        "checkpoint_sha256": sha256(ckpt),
        "eval_settings": {"imgsz": cfg["imgsz"], **v},
        "mAP50": round(float(metrics.box.map50), 5),
        "mAP50_95": round(float(metrics.box.map), 5),
        "precision": round(float(metrics.box.mp), 5),
        "recall": round(float(metrics.box.mr), 5),
        "per_class_mAP50_95": per_class,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
    }
 
    out_dir = repo_path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "baseline_pytorch_fp32.json"
    out_file.write_text(json.dumps(result, indent=2))
 
    print(f"mAP50     : {result['mAP50']:.4f}")
    print(f"mAP50-95  : {result['mAP50_95']:.4f}")
    print(f"Saved     : {out_file}")
 
 
if __name__ == "__main__":
    main() 