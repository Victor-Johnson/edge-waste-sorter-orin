"""Explicit MLflow logging for evaluation and benchmark JSON records."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "waste-sorter-edge"


def tracking_uri() -> str:
    """Use a user-supplied server URI or a private local SQLite database."""
    supplied = os.environ.get("MLFLOW_TRACKING_URI")
    if supplied:
        return supplied
    database = ROOT / ".mlflow" / "tracking.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database}"


def log_record(record: dict, result_file: str | Path, stage: str) -> str:
    """Log concise comparison metrics plus the complete JSON as an artifact."""
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("Install the optional tracking group: uv sync --group tracking") from exc

    if stage not in {"validation", "benchmark", "engine_benchmark", "decision_gate",
                     "material_policy"}:
        raise ValueError(f"Unknown stage: {stage}")
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(EXPERIMENT)
    model = Path(record.get("model", "material_policy"))
    run_name = (f"decision_gate:{Path(record['source']).stem}:{Path(record['tracker']).stem}"
                if stage == "decision_gate" else
                f"material_policy:{'replay' if 'policies' in record else 'benchmark'}:{record['tape_sha256'][:8]}"
                if stage == "material_policy" else f"{stage}:{model.stem}")
    if stage == "benchmark" and "track_enabled" in record:
        run_name += ":track" if record["track_enabled"] else ":predict"
        if "no_gmc" in record.get("tracker", ""):
            run_name += ":no-gmc"
        if record.get("profile_tracker"):
            run_name += ":profile"
        if record.get("power"):
            run_name += ":power"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("stage", stage)
        mlflow.set_tag("model_file", model.name)
        if record.get("sha256"):
            mlflow.set_tag("model_sha256", record["sha256"])
        if record.get("model_sha256"):
            mlflow.set_tag("model_sha256", record["model_sha256"])
        if record.get("source_sha256"):
            mlflow.set_tag("source_sha256", record["source_sha256"])
        for key, value in record.get("environment", {}).items():
            mlflow.set_tag(key, str(value))
        if record.get("source_note"):
            mlflow.set_tag("source_note", record["source_note"])
        if record.get("measurement_scope"):
            mlflow.set_tag("measurement_scope", record["measurement_scope"])
        mlflow.log_param("model", model.name)
        if stage == "validation":
            mlflow.log_param("dataset_yaml", Path(record["dataset_yaml"]).name)
            mlflow.log_params({key: value for key, value in record["settings"].items()
                               if isinstance(value, (str, int, float, bool))})
            mlflow.log_metrics({key: float(record[key]) for key in
                                ("mAP50", "mAP50_95", "precision", "recall")})
            for name, score in record.get("per_class_mAP50_95", {}).items():
                mlflow.log_metric(f"AP50_95/{name}", float(score))
        elif stage == "benchmark":
            if "track_enabled" in record:
                mlflow.set_tag("track_enabled", str(record["track_enabled"]).lower())
            if record.get("tracker"):
                mlflow.set_tag("tracker", record["tracker"])
            if record.get("tracker_config_sha256"):
                mlflow.set_tag("tracker_config_sha256", record["tracker_config_sha256"])
            mlflow.log_params({"source": Path(record["source"]).name,
                               "image_size": record["image_size"],
                               "confidence": record["confidence"],
                               "warmup_frames": record["warmup_frames"],
                               "measured_frames": record["measured_frames"]})
            metrics = {"end_to_end_fps": record["end_to_end_fps"],
                       "pipeline_p50_ms": record["pipeline_ms"]["p50"],
                       "pipeline_p95_ms": record["pipeline_ms"]["p95"],
                       "detections": record["detections"]}
            for key in ("unique_track_ids_observed", "untracked_detections"):
                if key in record:
                    metrics[key] = record[key]
            for stage_name in ("tracker_update", "gmc", "tracker_update_excluding_gmc"):
                if stage_name + "_ms" in record:
                    metrics[stage_name + "_mean_ms"] = record[stage_name + "_ms"]["mean"]
            if power := record.get("power"):
                mlflow.set_tag("power_mode", record.get("power_mode", "unknown"))
                mlflow.set_tag("power_rail", power["rail"])
                metrics["vdd_in_idle_mean_w"] = power["idle_vdd_in_w"]["mean"]
                metrics["vdd_in_active_mean_w"] = power["active_vdd_in_w"]["mean"]
                metrics["vdd_in_active_max_w"] = power["active_vdd_in_w"]["max"]
                metrics["input_energy_per_frame_j_est"] = round(
                    power["active_vdd_in_w"]["mean"] / record["end_to_end_fps"], 3)
                if power.get("active_gpu_temp_c"):
                    metrics["gpu_temp_active_max_c"] = power["active_gpu_temp_c"]["max"]
            mlflow.log_metrics(metrics)
            mlflow.set_tag("platform", record.get("platform", "unknown"))
        elif stage == "decision_gate":
            mlflow.set_tag("measurement_scope", record["scope"])
            mlflow.set_tag("trace_sha256", record["trace_sha256"])
            mlflow.set_tag("tracker", record["tracker"])
            if record.get("tracker_config_sha256"):
                mlflow.set_tag("tracker_config_sha256", record["tracker_config_sha256"])
            mlflow.log_params({"source": Path(record["source"]).name,
                               "frames": record["frames"],
                               "iterations": record["iterations"],
                               "min_iou": record["gate_config"]["min_iou"],
                               "max_track_gap_frames": record["gate_config"]["max_track_gap_frames"]})
            mlflow.log_metrics({"legacy_decisions": record["legacy"]["decision_count"],
                                "gated_decisions": record["gated"]["decision_count"],
                                "suppressed_candidates": record["gated"]["suppressed_count"],
                                "legacy_policy_p95_us": record["legacy"]["policy_latency"]["p95_us"],
                                "gated_policy_p95_us": record["gated"]["policy_latency"]["p95_us"]})
            mlflow.set_tag("platform", record["platform"])
        elif stage == "material_policy":
            mlflow.set_tag("measurement_scope", record["scope"])
            mlflow.set_tag("tape_sha256", record["tape_sha256"])
            mlflow.set_tag("policy_config_sha256", record["policy_config_sha256"])
            if record.get("classes_sha256"):
                mlflow.set_tag("classes_sha256", record["classes_sha256"])
            if "policies" in record:
                mlflow.set_tag("has_ground_truth_labels", str(record["label_sha256"] is not None).lower())
                for report in record["policies"]:
                    suffix = str(report["threshold"]).replace(".", "p")
                    mlflow.log_metric(f"routes_at_{suffix}", report["proposed_routes"])
                    mlflow.log_metric(f"reviews_at_{suffix}", report["proposed_reviews"])
                    if evaluation := report.get("labelled_evaluation"):
                        if evaluation["automatic_coverage"] is not None:
                            mlflow.log_metric(f"coverage_at_{suffix}", evaluation["automatic_coverage"])
                        if evaluation["cross_material_error_per_route"] is not None:
                            mlflow.log_metric(f"cross_material_error_at_{suffix}",
                                              evaluation["cross_material_error_per_route"])
            else:
                mlflow.log_params({"frames": record["frames"],
                                   "iterations": record["iterations"],
                                   "policy_version": record["config"]["version"],
                                   "min_mean_confidence": record["config"]["min_mean_confidence"]})
                mlflow.log_metrics({"policy_p50_us": record["policy_latency"]["p50_us"],
                                    "policy_p95_us": record["policy_latency"]["p95_us"],
                                    "proposed_routes": record["outcomes"]["route"],
                                    "proposed_reviews": record["outcomes"]["review"]})
                mlflow.set_tag("platform", record["platform"])
        else:
            mlflow.log_params(record["settings"])
            mlflow.log_metrics({key: float(record[key]) for key in (
                "throughput_qps", "latency_mean_ms", "latency_p95_ms",
                "gpu_compute_mean_ms", "gpu_compute_p95_ms", "enqueue_mean_ms",
                "h2d_mean_ms", "d2h_mean_ms")})
        mlflow.log_artifact(str(result_file), artifact_path="results")
        if record.get("telemetry_raw_log"):
            raw_log = Path(result_file).with_name(record["telemetry_raw_log"])
            if raw_log.is_file():
                mlflow.log_artifact(str(raw_log), artifact_path="results")
        return run.info.run_id
