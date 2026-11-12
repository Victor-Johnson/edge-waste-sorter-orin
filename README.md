# Waste Sorter // Edge Lab

A real-time waste detection and proposed sorting system for NVIDIA Jetson Orin Nano. A camera or video feeds one tracked inference pipeline. The browser shows annotated frames, live metrics, recent decisions, and a 2D conveyor twin driven by detection events. No actuator is connected.

## Current status

`data/Warp-D` contains 2,452 train and 522 test image/label pairs. The 28 class names match the checkpoint order. All accuracy runs below used the 522-image test split, batch 1, 640-pixel input, confidence 0.001, and IoU 0.7.

| Hardware and backend | Ultralytics | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: |
| Desktop PyTorch | 8.4.171 | 0.61235665 | 0.46245111 |
| Desktop ONNX | 8.4.171 | 0.61235666 | 0.46245112 |
| Jetson ONNX Runtime CPU | 8.3.59 | 0.616 | 0.470 |
| Jetson TensorRT FP16 | 8.3.59 | 0.617 | 0.470 |

The Jetson accuracy values are from CLI output rounded to three decimals. The Jetson ONNX and FP16 mAP50-95 values match at that precision; the desktop-to-Jetson change also includes an evaluator version change. Result JSON files are in `benchmarks/results/`, with separate validation and engine benchmark runs in MLflow. The Jetson FP16 engine is 42.7 MiB. A 15 W `trtexec` check with generated inputs reported 49.67 queries/s, 20.75 ms mean latency, and 20.09 ms mean GPU compute time. The video probes below measure application throughput with and without tracking; a separate pair records input-rail power. Live camera and network-streaming measurements remain outstanding.

Three stock recycling MP4s under `data/sample_video/` are unlabelled and cannot establish mAP. The Jetson video probe on `waste-sorting-4.mp4` reached 17.75 FPS over 200 measured frames, with mean frame time 55.97 ms (42.88 ms prediction, 9.10 ms JPEG encoding, 3.19 ms decoding, 0.80 ms drawing). It produced only five detections across those frames. A separate clip repeatedly showed one large bottle. For a useful sorting demonstration, `scripts/make_replay.py` creates a clearly marked WaRP-D dataset replay with plastic, metal, carton, and glass examples. The Jetson probe measured 16.57 FPS across 280 frames of that replay, with 1,159 frame-level detections across all four material groups. Repeated detections of the same object across frames are counted separately. Both probe results are saved in `benchmarks/results/` and MLflow. The class-to-lane mapping in `configs/demo.yaml` proposes material lanes from checkpoint class names; unknown classes route to `review`. In WaRP, `-full` means air-filled, not liquid-filled.

A paired replay check on the Jetson in 15 W power mode used the same FP16 engine, video, 20 warmup frames, 280 measured frames, 640-pixel input, and 0.35 confidence. The tracked path uses persistent BoT-SORT IDs and excludes browser/network streaming. The 15 W mode is a configured power limit, not a measurement of actual draw.

| Replay path | End-to-end FPS | Frame p50 | Frame p95 | Frame-level detections |
| --- | ---: | ---: | ---: | ---: |
| Prediction + draw + JPEG | 16.50 | 58.71 ms | 76.09 ms | 1,159 |
| Prediction + tracking + draw + JPEG | 10.99 | 91.01 ms | 94.67 ms | 1,145 |

This pair shows a 5.51 FPS drop when tracking is enabled. The tracked run observed 25 distinct track IDs and four detections without IDs in the measured frames. Track IDs are software identities, not verified unique physical objects; ID changes can overcount objects. These are single adjacent runs, so repeat them with power and temperature monitoring before claiming a sustained performance figure. Both JSON records are in `benchmarks/results/` and logged to MLflow.

To reproduce the tracked run on the Jetson, use the command below. Omit `--track` and choose another output filename for the prediction-only run.

```bash
cd ~/edge-waste-sorter
.venv/bin/python scripts/jetson_video_probe.py \
  --model exports/02_yolo26m_best_fp16.engine \
  --video data/sample_video/warp-d-replay.mp4 \
  --classes data/Warp-D/classes.txt \
  --out benchmarks/results/jetson_replay_fp16_track.json \
  --warmup 20 --frames 280 --track
```

The power probe uses `tegrastats` at 500 ms intervals and aligns its samples to the measured video frames. Its five-second idle sample precedes model loading. The active values exclude model startup and warmup; boundary seconds are omitted because `tegrastats` prints timestamps to one-second precision. NVIDIA identifies `VDD_IN` as **total module power** on Orin Nano, and describes the first rail value as current milliwatts ([power rail documentation](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html), [tegrastats format](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html)). It does not measure wall-plug power.

| Power probe, 15 W mode | FPS | Idle `VDD_IN` | Active `VDD_IN` | Peak GPU temp | Estimated module input energy/frame |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prediction only | 16.32 | 4.41 W | 7.00 W | 50.12 °C | 0.429 J |
| With BoT-SORT tracking | 11.20 | 4.38 W | 6.55 W | 49.53 °C | 0.585 J |

Energy per frame is estimated as active mean input power divided by measured FPS. The slower tracked path had lower average input power but used more module input energy per processed frame. These short, single runs include decoding and JPEG work, and the idle-to-active difference cannot be assigned to the model alone. Raw telemetry and summary JSON are in `benchmarks/results/` and attached to the two `:power` MLflow runs. Repeat over longer runs before claiming sustained thermal or efficiency performance.

Reproduce the tracked power measurement with the command below; omit `--track` and choose another output filename for prediction-only power.

```bash
cd ~/edge-waste-sorter
.venv/bin/python scripts/jetson_power_probe.py \
  --model exports/02_yolo26m_best_fp16.engine \
  --video data/sample_video/warp-d-replay.mp4 \
  --classes data/Warp-D/classes.txt \
  --out benchmarks/results/jetson_power_track.json \
  --warmup 20 --frames 280 --track
```

### Tracker profiling experiment

The installed Ultralytics 8.3.59 BoT-SORT uses sparse optical flow for camera-motion compensation. Timing its update separately on the 280-frame replay showed that this step dominated tracker time. `configs/trackers/botsort_no_gmc.yaml` matches the installed BoT-SORT settings except `gmc_method: none`; the alternate run used the same FP16 engine, replay, confidence, warmup, and frame count. These are instrumented runs, so use them to locate the hotspot and compare configurations, not as the final uninstrumented speed claim.

| Tracker configuration | FPS | Tracker update mean | Motion compensation mean | Track IDs observed | Detections without IDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default BoT-SORT | 10.91 | 31.84 ms | 28.55 ms | 25 | 4 |
| BoT-SORT without motion compensation | 15.13 | 4.05 ms | 0.03 ms | 26 | 5 |

The alternate configuration recovered 4.22 FPS on this replay. A separate 300-frame ID review found the reason for its extra global ID: at the hard cut between source images, default BoT-SORT reused ID 8 from a `bottle-transp` in frame 75 for `cans` in frame 76. No-GMC left the can untracked for frame 76, then assigned it fresh ID 24 in frame 77. The full runs had 27 versus 28 global IDs, but the same per-scene ID counts (21, 3, 1, 3) and the same frame-by-frame class counts. Since the dashboard proposes one decision per first-seen ID, the default tracker would suppress the can's new decision in this uninterrupted replay. This is a cross-scene cut artifact, not evidence of an ID switch within a continuous scene. The annotated videos and per-frame traces are in the gitignored `docs/CRM/videos/tracker_review/`; the compact review record is `benchmarks/results/jetson_tracker_identity_review.json`.

The replay has no ground-truth track identities or real object motion, so this result does not settle tracker quality on a conveyor. The profiling JSON files and MLflow runs preserve the speed comparison. Ultralytics documents `gmc_method: none` as the way to disable camera-motion compensation ([tracker configuration](https://docs.ultralytics.com/modes/track)).

### Real-motion tracker check

We also reviewed a 275-frame, 9.17-second [Pexels clip of workers sorting plastic bottles](https://www.pexels.com/video/people-working-in-sorting-plastic-bottles-13662401/) filmed from a fixed viewpoint. It is by Peter Hanzo under the [Pexels License](https://www.pexels.com/license/). The source video is gitignored at `data/sample_video/pexels-plastic-bottles-13662401.mp4`; its SHA-256 and the exact tracker/model settings are in `benchmarks/results/jetson_real_motion_tracker_review.json`.

Both tracker settings produced the same 246 frame-level detections and the same five track IDs, with identical IDs on every frame. A sparse [manual landmark audit](benchmarks/annotations/pexels_plastic_bottle_landmarks.json) follows two identifiable bottles through selected source frames. IDs 1 and 7 both cover the same large bottle's marked cap in frames 242, 257, and 264; their boxes overlap in 14 frames. This creates two first-ID proposed decisions for one observed object in either setting. A second bottle's marked neck is covered by a tracked box in only one of nine sampled frames, suggesting a detection/localization miss. The [audit result](benchmarks/results/jetson_real_motion_landmark_audit.json) and `scripts/audit_track_identity.py` make the review repeatable. This small sample does not establish a general ID-switch rate, detector recall, or speed result. The full annotated videos and per-frame traces are in the gitignored `docs/CRM/videos/tracker_review/` workspace. Keep no-GMC optional until a longer clip with more manually marked identities supports a deployment decision.

### Optional duplicate-decision experiment

`--duplicate-gate` enables an experimental gate after tracking. When a new ID first appears, it holds that proposed decision if its box overlaps a recently accepted ID of the same predicted class by at least 0.60 IoU. It keeps drawing and reporting the detection. The dashboard shows how many duplicate candidates were held and their matching IDs. The default remains the original one-decision-per-ID behavior.

Three hand-checked [same/different object pairs](benchmarks/annotations/decision_gate_pairs.json) anchor the rule. On the real-motion trace, the gate holds ID 7 against ID 1 at frame 242 (IoU 0.602), changing five first-ID proposals to four. It leaves the two labelled distinct replay pairs separate. A class-change safeguard prevents a reused bottle ID from suppressing a new can ID after the replay's hard scene cut; the replay keeps 27 proposals under both rules. These are narrow examples, not a false-merge rate. A distinct item entering behind another same-class item could still be held incorrectly. The gate remains opt-in until more overlapping *different* items are labelled.

The [Jetson policy replay records](benchmarks/results/jetson_decision_gate_real_motion.json) and [denser replay record](benchmarks/results/jetson_decision_gate_replay.json) measure only Python decision work from saved traces. Gated p95 was 4.99 microseconds/frame on the real clip and 15.74 microseconds/frame on the denser replay; these are not full video FPS measurements. A live Jetson dashboard smoke test with `--duplicate-gate` finished the real clip with four proposed decisions, one held candidate, and no error. Both saved-trace results are logged in MLflow as `decision_gate:pexels-plastic-bottles-13662401:botsort` and `decision_gate:warp-d-replay:botsort`.

### Multi-frame material policy and replay lab

`--material-policy` enables the advisory `material-v1` policy. It waits for three observations of one track, checks that their material lanes agree, then routes only if their **mean model confidence score** meets the configured threshold. Unknown classes, conflicting materials, missing track IDs, disappearing tracks, and low scores produce explicit `review` reasons. `configs/policy.yaml` sets the observation count, wait and gap limits, and threshold. Confidence is not calibrated probability. The original first-ID behavior remains the default. The spatial duplicate gate can also be enabled, but its false-merge risk remains unmeasured.

The [saved-tape replay](benchmarks/results/material_policy_real_motion_sweep.json) uses the unlabelled 275-frame real-motion clip. At score thresholds 0.40, 0.55, and 0.70, it produced 4/1, 1/4, and 0/5 route/review proposals respectively. These are **proposal counts, not accuracy, physical item counts, or a risk-versus-coverage curve**. The [Jetson policy-only benchmark](benchmarks/results/material_policy_jetson_real_motion.json) measured 6.945 microseconds p95 per frame over 200 replay iterations. A live TensorRT run completed 275 frames with no error and emitted one route plus four reviews under the 0.55 setting. The browser exposes the reason for each proposal.

The tape is a flat JSONL of tracker observations with model, source, class-list, tracker, and tape hashes in its manifest. It contains no raw frames. Make it from a saved `jetson_track_review.py` trace and compare policies without another inference run:

```bash
.venv/bin/python scripts/jetson_track_review.py \
  --model exports/02_yolo26m_best_fp16.engine \
  --video data/sample_video/pexels-plastic-bottles-13662401.mp4 \
  --classes data/Warp-D/classes.txt --tracker botsort.yaml \
  --scene-frames 300 \
  --out-prefix docs/CRM/videos/tracker_review/real_motion_default
.venv/bin/python scripts/record_prediction_tape.py \
  --trace docs/CRM/videos/tracker_review/real_motion_default.jsonl \
  --summary docs/CRM/videos/tracker_review/real_motion_default.summary.json \
  --classes data/Warp-D/classes.txt \
  --out benchmarks/runs/policy/real_motion.jsonl
.venv/bin/python scripts/evaluate_policy.py \
  --tape benchmarks/runs/policy/real_motion.jsonl \
  --manifest benchmarks/runs/policy/real_motion.manifest.json \
  --classes data/Warp-D/classes.txt \
  --out benchmarks/runs/policy/real_motion_sweep.json
```

To measure cross-material error, provide `--labels path/to/labels.json`. A simple mapping such as `{"1":"plastic"}` scores **track IDs**. For a real identity check, use the [physical-item example](benchmarks/annotations/real_motion_partial_physical_items.json): `physical-items-v1` groups all IDs belonging to one physical object, counts duplicate proposals, and requires a `complete` declaration. An incomplete annotation reports case counts but leaves coverage and error rates null. The [one-item partial audit](benchmarks/results/material_policy_partial_physical_audit.json) groups IDs 1 and 7 from the same bottle and finds two proposals; it does **not** estimate a population error rate. The current clip has too few manually identified objects to support threshold selection. The WaRP-D replay uses held-out **test** images, so it must not be used to tune the policy threshold.

### Hash-checked deployment and rollback

`configs/class_order.yaml` pins all 28 class IDs. Settings loading fails with the first incorrect index if `classes.txt` is reordered. The Jetson manifests in `configs/deployments/` tie the FP16 engine hash to the class list, lane map, tracker file, policy file, and validation/benchmark evidence hashes. Supplying `--manifest` makes dashboard startup reject a mismatch. The engine itself is ignored by Git and must still exist on the Jetson.

```bash
cd ~/edge-waste-sorter
.venv/bin/python scripts/run_dashboard.py \
  --model exports/02_yolo26m_best_fp16.engine \
  --source data/sample_video/pexels-plastic-bottles-13662401.mp4 \
  --material-policy --manifest configs/deployments/jetson_fp16_material_v1.json
```

To restore the validated first-ID behavior using the same engine, run the command without `--material-policy` and choose `configs/deployments/jetson_fp16_baseline.json`. This is a **policy rollback**, not a second engine or a production failover. `scripts/build_deployment_manifest.py` regenerates a manifest after a model or configuration change and checks that the supplied validation and benchmark JSONs reference the exact engine hash.

```bash
cd ~/edge-waste-sorter
.venv/bin/python scripts/run_dashboard.py \
  --model exports/02_yolo26m_best_fp16.engine \
  --source data/sample_video/pexels-plastic-bottles-13662401.mp4 \
  --duplicate-gate
```

Reproduce the visual review on Jetson from the repository root; the script saves an annotated MP4, a per-frame JSONL trace, and a summary JSON for each run:

```bash
.venv/bin/python scripts/jetson_track_review.py --model exports/02_yolo26m_best_fp16.engine --video data/sample_video/warp-d-replay.mp4 --classes data/Warp-D/classes.txt --tracker botsort.yaml --out-prefix benchmarks/runs/tracker_review/default
.venv/bin/python scripts/jetson_track_review.py --model exports/02_yolo26m_best_fp16.engine --video data/sample_video/warp-d-replay.mp4 --classes data/Warp-D/classes.txt --tracker configs/trackers/botsort_no_gmc.yaml --out-prefix benchmarks/runs/tracker_review/no_gmc
```

To inspect the alternate tracker in the dashboard, pass `--tracker configs/trackers/botsort_no_gmc.yaml` alongside the usual Jetson `--model` argument. Remove that flag to return to default BoT-SORT.

## Architecture

```text
camera / video → YOLO tracking → class-to-lane observations → optional policy → browser twin
                           ↘ annotated JPEG frames → browser video panel
```

The observation event shape is `{timestamp, frame_id, track_id, class, confidence, bbox_xyxy, lane}`. The dashboard shows every current-frame detection. By default its decision history records only the first sighting of each confirmed track ID. With the material policy enabled, a track yields one later route or review proposal; an untracked observation yields review. An ID switch can still make one physical item appear as a new track. These are visual proposals, not actuator commands. `GET /api/status` returns current metrics and events; `GET /api/events` streams status snapshots with Server-Sent Events; `GET /stream.mjpg` serves annotated frames. Pipeline latency includes tracking and annotation, but excludes video decoding and JPEG encoding; measured throughput includes all of those stages.

`scripts/jetson_video_probe.py` measures decoding, model prediction, annotation, and JPEG encoding separately. With `--track`, its model stage includes persistent BoT-SORT tracking. It always excludes network streaming. Copy the JSON result back to the host before importing it into MLflow with `scripts/log_result.py --stage benchmark`.

## Run locally

Use the existing environment or `uv sync`. Generate the labelled dataset replay and run the dashboard:

```bash
uv run python scripts/make_replay.py
uv run python scripts/run_dashboard.py
```

Open `http://127.0.0.1:8000`. For a camera, use `--source 0`; for one of the stock clips, use `--source data/sample_video/waste-sorting-4.mp4`. The default model is the PyTorch checkpoint. The stock clips have watermarks; verify publication rights before using them in a public portfolio. The generated replay is visibly labelled as dataset footage. Set `lanes` in `configs/demo.yaml` using exact checkpoint class names.

## Run on Jetson

The Jetson `.venv` uses its system PyTorch and TensorRT. From a Jetson terminal, run the FP16 engine against the labelled replay:

```bash
cd ~/edge-waste-sorter
.venv/bin/python scripts/run_dashboard.py --model exports/02_yolo26m_best_fp16.engine
```

From the desktop, forward port 8000 with `ssh -F /dev/null -L 8000:127.0.0.1:8000 jetson@100.84.38.20`, then open `http://127.0.0.1:8000`. Start the tunnel before the dashboard so you can watch the 300-frame replay from the beginning. The default replay ends after its last frame. The dashboard reports one proposed decision per track ID; it does not count every frame-level detection as a new item. Its p50/p95 values cover tracking, class routing, and annotation, while the displayed FPS includes decoding and JPEG encoding too.

## Track experiments with MLflow

The optional `tracking` group adds MLflow. Runs use the `waste-sorter-edge` experiment in a local SQLite database under `.mlflow/`; artifacts are stored under `mlruns/`. Both are ignored by Git. Each run logs the model name, evaluation or benchmark settings, summary metrics, and the complete JSON result artifact. Validation runs also log per-class AP. We do not upload weights or dataset images.

```bash
uv sync --group tracking
# The completed PyTorch validation has already been imported once.
.venv/bin/mlflow ui --backend-store-uri sqlite:///.mlflow/tracking.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`, select `waste-sorter-edge`, and inspect the `validation:02_yolo26m_best` run. Compare its `mAP50_95` and per-class AP metrics with later runs. To log a new validation or video benchmark, append `--mlflow` to the normal command. To import an existing JSON result without rerunning inference, use `python scripts/log_result.py --stage validation --file benchmarks/results/val_pt.json`. Avoid importing the same result twice. Set `MLFLOW_TRACKING_URI` to a tracking server URL when a shared server is available; otherwise the local database is used.

For the decision gate experiment, compare the two `decision_gate:*` runs. They log legacy and gated proposal counts, held candidates, and policy p95 in microseconds; their JSON artifacts contain the exact suppressed IDs. The real-motion run ID is `aa2a4b4e70704356aa92f79378138b9c`; the replay run ID is `dc8d6351867f48139dd31ac3ab88d657`. These run IDs refer to this workspace's local MLflow database.

The material policy has two separate MLflow runs: `material_policy:replay:ecb78949` (`efbc07fe34de4929a1270b2ab1bb2d8c`) stores route/review counts at each threshold, and `material_policy:benchmark:ecb78949` (`893f33213c244df1b53baae27d0de128`) stores Jetson policy-only latency. The replay run is tagged `has_ground_truth_labels=false`; its metrics must not be read as routing accuracy. Reimport a new JSON using `scripts/log_result.py --stage material_policy --file path/to/result.json`.
The partial one-item audit is a third run, `material_policy:replay:ecb78949:partial` (`8070b75905f64a0eab42966a269cae2a`). It is tagged `labelled_evaluation_complete=false` and logs no coverage or error-rate metric.

## Jetson milestone

1. The reachable Jetson reports Jetson Linux R36.4.7, Python 3.10.12, NVIDIA PyTorch 2.5.0a0, TensorRT 10.7.0, Ultralytics 8.3.59, and 15 W power mode. No `/dev/video*` device is currently exposed. Its `.venv` inherits the NVIDIA PyTorch/TensorRT installation and now has `lap`, FastAPI, and Uvicorn. The desktop `uv.lock` targets Python 3.12 and is not a Jetson environment. The FP16 engine, WaRP-D class list, replay video, and dashboard source are present on the Jetson.
2. Use the configured WaRP-D test split to validate each model with identical settings:

```bash
python scripts/validate_model.py --model weights/detection/02_yolo26m_best.pt --out benchmarks/results/val_pt.json
python scripts/validate_model.py --model exports/02_yolo26m_best_static640_opset17.onnx --out benchmarks/results/val_onnx.json
python edge/trt/build_engine.py --precision fp16
python edge/trt/build_engine.py --precision int8 --dataset /path/to/dataset.yaml
python scripts/validate_model.py --model /path/to/fp16.engine --out benchmarks/results/val_fp16.json
python scripts/validate_model.py --model /path/to/int8.engine --out benchmarks/results/val_int8.json
```

Use a representative calibration split for INT8 and inspect per-class accuracy changes before choosing an engine. Engine files are device-specific and ignored by Git.

3. Benchmark each candidate using the same video, image size, confidence, warmup, and power mode:

```bash
python scripts/benchmark.py --model /path/to/fp16.engine --source data/sample_video/demo.mp4 --out benchmarks/results/perf_fp16.json
python scripts/benchmark.py --model /path/to/int8.engine --source data/sample_video/demo.mp4 --out benchmarks/results/perf_int8.json
python scripts/run_dashboard.py --model /path/to/chosen.engine --source 0 --host 0.0.0.0
```

For power evidence, capture `tegrastats` alongside the benchmark and report the mode and ambient conditions. Access the dashboard over a Tailscale address or SSH tunnel; do not expose it to the public internet without authentication.

## Portfolio evidence to add

- A short camera and video demo, with the twin visible beside detections.
- A benchmark table with PyTorch, ONNX, FP16, and INT8 accuracy; Jetson throughput, p50/p95 pipeline latency, and power.
- Representative successes and failures, including class confusion and low-light or occluded cases.
- The chosen engine and why its measured accuracy/speed tradeoff fits the sorter.

## Learning checkpoints

1. **Validation:** run the PyTorch and ONNX commands, then explain any per-class differences.
2. **Optimization:** build and compare FP16 and INT8 on the Jetson; choose the engine using measured evidence.
3. **System:** configure lanes, run camera and video modes, and inspect track IDs and twin behavior.
4. **Story:** publish a short demo and honest case study with reproducible results.
