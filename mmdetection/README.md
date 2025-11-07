# Minecraft Detection Project

## Overview
This repository hosts the coursework project for fine-tuning FCOS and YOLO detectors on the Minecraft mobs dataset. It contains notebook-driven experimentation backed by reusable Python modules and MMDetection/YoloV8 configurations.

## Repository Layout
- `datasets/` – expected location for the Minecraft dataset (`images/`, `labels/`, `annotations/`, and `video.mp4`).
- `configs/` – custom training configs (e.g. `fcos_minecraft.py`).
- `src/` – auxiliary Python modules for data loading, training loops, evaluation, and visualization.
- `artifacts/` – logs, metrics, figures, checkpoints, and inference outputs produced during experiments.
- `checkpoints/` – pretrained weights to bootstrap fine-tuning runs.
- `notebook.ipynb` – main notebook with EDA, training, evaluation, and reporting steps.

## Workflow Plan
1. Validate dataset structure, class coverage, and sample annotations via EDA.
2. Adapt FCOS MMDetection config, run pretrained inference, then fine-tune and log metrics.
3. Prepare YOLO dataset YAML, run pretrained inference, fine-tune YOLOv8s, and export logs.
4. Benchmark both models on images and video, capturing FPS, accuracy metrics, visuals, and videos.
5. Aggregate metrics into comparison tables/plots and compile the final PDF report.

## Setup Checklist
- Install Python 3.10+, CUDA toolkit, PyTorch, MMDetection, and Ultralytics YOLO dependencies (`pip install -r requirements.txt`).
- Populate `datasets/minecraft/` with the raw Roboflow export (`train/`, `val/`, `test/`, XML annotations, and video).
- Convert annotations and structure the dataset via:
  - `python -m src.data.voc_to_coco datasets/minecraft`
  - `python -m src.data.coco_to_yolo datasets/minecraft`
- Ensure images reside in `datasets/minecraft/{split}/images/` and YOLO labels under `datasets/minecraft/labels/{split}/`.
- Keep `datasets/minecraft/data_coco.yaml` up to date; Ultralytics reads this file during training/inference.
- Download pretrained weights (e.g. `fcos_r50_caffe_fpn_gn-head_1x_coco.pth`, `yolov8s.pt`) into `checkpoints/`.
- Open `notebook.ipynb` to execute EDA, training, evaluation, and reporting steps in sequence.

## Deliverables
- Updated configs, training utilities, and executed notebook covering all mandatory stages.
- Fine-tuned FCOS and YOLO weights, logs, metrics visualisations (`artifacts/fcos/`, `artifacts/yolo/`, `artifacts/metrics/`).
- Image galleries and videos showcasing inference quality (`artifacts/inference/`, `artifacts/videos/`).
- Comparative metrics table saved to `artifacts/metrics/metrics_comparison.csv`.
- Final PDF report in `artifacts/report.pdf`, with a short textual summary appended to this README after experiments.
