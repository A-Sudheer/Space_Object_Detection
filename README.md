# Space Object Detection

A YOLO-based object detection system for identifying satellites, space debris, and named spacecraft in orbital imagery, built and trained on Kaggle.

## Overview

This project trains and compares multiple YOLO architectures (YOLOv8, YOLOv10, YOLO11) to detect and classify space objects across 13 classes, ranging from generic categories (debris, satellite, asteroid) to specific named ESA missions (CHEOPS, SOHO, XMM-Newton, PROBA-2/3, and others). The project was built incrementally, adding dataset sources in stages to measure their individual impact on model performance, followed by hyperparameter tuning on the top-performing architectures.

## Classes (13)

```
cheops, debris, double_start, earth_observation_sat_1, lisa_pathfinder,
proba_2, proba_3_csc, proba_3_ocs, smart_1, soho, xmm_newton,
spacecraft, asteroid
```

## Datasets

| Source | Classes | Images | Notes |
|---|---|---|---|
| [Space Debris (Roboflow)](https://universe.roboflow.com/woah-noah/space-debris-mugw2) | 11 (named spacecraft + debris) | ~2,100 | Balanced (~170–215 instances/class) |
| [Spacecraft Detection (Roboflow)](https://universe.roboflow.com/rohit-kumar-jha/spacecraft-detection-vncj3) | 1 (spacecraft) | ~2,850 | Single continuous video sequence |
| Asteroid Detection (Roboflow) | 1 (asteroid) | ~800 | Sequential frame source |

All datasets were merged into a unified YOLO-format structure with a single `data.yaml` and consistent class indexing.

## Methodology

Training was staged into milestones to isolate the effect of each added dataset on model performance, rather than merging everything blind:

1. **Milestone 1** — Baseline training on the 11-class debris/spacecraft dataset alone, across YOLOv8n, YOLOv10n, and YOLO11n.
2. **Milestone 2** — Added the generic `spacecraft` class and retrained, comparing per-class metrics against the baseline to check for regressions.
3. **Milestone 3** — Added the `asteroid` class and retrained again.
4. **Hyperparameter tuning** — Applied Ultralytics' evolutionary tuner (`model.tune()`) to the top two performing architectures to search learning rate, augmentation, and optimizer settings before a final full-length training run.

### A note on data leakage

An early version of the merged dataset showed a suspiciously perfect score (mAP50-95 ≈ 0.995) on the `spacecraft` class. Investigation traced this to sequential video-frame filenames being randomly split across train/validation, causing near-duplicate frames to leak across the split. This was corrected using a **frame-aware block split** (grouping and splitting by contiguous frame ranges with a buffer gap at each boundary) instead of a random shuffle. This is a known methodological consideration documented here for transparency — the `asteroid` class source data shows the same sequential-frame pattern and should be treated with the same caveat unless re-split accordingly.

## Results

### Milestone 1 — Baseline (11 classes)

| Model | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| YOLOv8n | 0.891 | 0.481 | 0.881 | 0.858 |
| YOLOv10n | 0.848 | 0.468 | 0.832 | 0.776 |
| YOLO11n | 0.890 | 0.480 | 0.886 | 0.809 |

### Milestone 2 → 3 — YOLOv8n (progressive dataset merging)

| Stage | mAP50 | mAP50-95 |
|---|---|---|
| Milestone 1 (11 classes) | 0.891 | 0.481 |
| Milestone 2 (+spacecraft, leakage-corrected) | 0.897 | 0.527 |
| Milestone 3 (+asteroid) | 0.903 | 0.571 |

Full per-class breakdowns are available in `/results/`.

## Tech Stack

- **Framework**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (v8, v10, v11)
- **Training platform**: Kaggle Notebooks (GPU T4 x2)
- **Language**: Python 3.12, PyTorch
- **Dataset sourcing**: Roboflow Universe, Kaggle Datasets

## Project Structure

```
├── notebooks/
│   ├── 01_milestone1_baseline.ipynb
│   ├── 02_milestone2_spacecraft.ipynb
│   ├── 03_milestone3_asteroid.ipynb
│   └── 04_hyperparameter_tuning.ipynb
├── scripts/
│   ├── merge_datasets.py
│   ├── block_split_by_frame.py
│   └── class_remap.py
├── results/
│   ├── milestone1_comparison.csv
│   ├── milestone2_per_class.csv
│   └── milestone3_per_class.csv
└── data.yaml
```

## Known Limitations

- The `spacecraft` and `asteroid` classes originate from single continuous video sequences rather than diverse independent images, limiting true visual variety in validation/test sets even after frame-aware splitting.
- Class imbalance exists between generic categories (hundreds of images) and named spacecraft classes (~20 instances each).
- Some named spacecraft classes (`smart_1`, `xmm_newton`) show consistently lower recall, likely due to visual similarity with other classes at the current resolution/model capacity.

## Acknowledgments

Datasets sourced from Roboflow Universe contributors, licensed under CC BY 4.0.
