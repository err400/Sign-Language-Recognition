# Pose-Guided Temporal Sign-Language Recognition

This project classifies isolated dynamic sign-language gestures from short clips. It extracts hand and upper-body landmarks with MediaPipe, normalizes them into variable-length temporal sequences, and trains a PyTorch bidirectional LSTM classifier with padding-aware sequence handling.

It is not a continuous sign-language translation system. It predicts one class for one isolated gesture clip.

## Pipeline

```mermaid
flowchart LR
  A[Video] --> B[MediaPipe landmarks]
  B --> C[Normalization and masking]
  C --> D[BiLSTM]
  D --> E[Confidence-ranked sign predictions]
```

## Why temporal modelling is required

Many signs are defined by motion, not by a single hand pose. The model therefore receives a sequence of landmark frames and learns how positions change over time. A bidirectional LSTM is used because the full isolated clip is available at classification time, so the model can use past and future context within the clip.

## Landmark extraction and normalization

`src/features/extraction.py` samples frames from each video and uses MediaPipe Holistic to extract left hand, right hand and upper-body pose landmarks. Each landmark contributes `x, y, z, visibility`, producing a 220-dimensional feature vector per frame.

`src/features/normalization.py` translates coordinates by the midpoint of the two shoulders and scales them by shoulder width. If shoulders are missing or the scale is too small, the scale falls back to `1.0`. Missing landmarks are zero-filled and tracked with an explicit Boolean presence mask.

## Padding masks and packed sequences

Videos have different lengths after sampling and extraction. `collate_landmark_sequences` pads a batch to the longest sequence and returns:

- `lengths`: the true number of frames for each sample.
- `padding_mask`: `True` for padded timesteps and `False` for valid timesteps.

The BiLSTM uses `pack_padded_sequence`, so padded timesteps are not processed as real data. Temporal pooling is also mask-aware, so padded rows do not affect logits.

## Installation

Use Python 3.10, 3.11 or 3.12. PyTorch and MediaPipe may not support newer Python releases immediately.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Smoke test

The smoke test creates a tiny synthetic landmark dataset, trains for a few epochs, evaluates, and runs feature-file inference.

```bash
python -m pytest
python scripts/smoke_test.py --config configs/smoke.yaml
```

## WLASL subset workflow

Download WLASL metadata and videos into `data/raw/`. Large files are intentionally ignored by Git.

```bash
python scripts/prepare_subset.py wlasl \
  --json data/raw/WLASL_v0.3.json \
  --videos data/raw/videos \
  --out data/interim/wlasl_subset_manifest.json \
  --num-classes 20 \
  --max-samples-per-class 40

python scripts/preprocess_videos.py \
  --config configs/default.yaml \
  --manifest data/interim/wlasl_subset_manifest.json

python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt --split test
```

If WLASL is unavailable, use a generic class-folder video layout:

```text
data/raw/generic/
  hello/
    sample1.mp4
  thanks/
    sample1.mp4
```

Then run:

```bash
python scripts/prepare_subset.py generic --videos data/raw/generic --out data/interim/generic_manifest.json
python scripts/preprocess_videos.py --config configs/default.yaml --manifest data/interim/generic_manifest.json
```

## Inference

For an uploaded or local video:

```bash
python scripts/infer_video.py --checkpoint outputs/checkpoints/best.pt --video path/to/video.mp4 --top-k 3
```

For a pre-extracted `.npz` feature file:

```bash
python scripts/infer_video.py --checkpoint outputs/checkpoints/best.pt --features-npz path/to/sample.npz
```

For webcam inference:

```bash
python scripts/webcam_infer.py --checkpoint outputs/checkpoints/best.pt
python scripts/webcam_infer.py --checkpoint outputs/checkpoints/best.pt --headless
```

## Outputs

Training curves, checkpoints, confusion matrices and metrics are saved under `outputs/`. The repository does not include fabricated metrics; run the commands above to generate results on your machine and dataset.

## Verified Run In This Workspace

Executed with Python 3.12 in `.venv` on August 16, 2026:

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/smoke_test.py --config configs/smoke.yaml
.venv/bin/python scripts/infer_video.py --checkpoint outputs/smoke/checkpoints/best.pt --features-npz data/processed/smoke/test/synthetic_0/synthetic_0_000.npz --top-k 3 --device cpu
```

Results:

- Unit tests: 7 passed.
- Synthetic smoke test: test accuracy `1.0`, macro F1 `1.0`.
- Held-out synthetic feature inference: top prediction `synthetic_0` with confidence `0.9954`.

These are synthetic verification results only. A real WLASL subset run and uploaded-video inference on a real trained checkpoint require local WLASL metadata/videos and an actual video sample.

## Limitations

The model is sensitive to signer variation, camera angle, occlusion, motion blur and landmark extraction failures. Mirroring is optional because left-right changes can alter meaning for some signs. The current system handles isolated signs only; future work could add temporal transformers, signer adaptation, stronger video augmentation, and continuous-sign segmentation/recognition.
