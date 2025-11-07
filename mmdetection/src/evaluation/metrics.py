"""Metric computation helpers for FCOS and YOLO outputs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import json
import pandas as pd


@dataclass
class MetricsSummary:
    map50: float
    map: float
    precision: float
    recall: float
    f1: float
    fps: float


def load_fcos_metrics(log_path: Path) -> List[Dict]:
    """Load metrics from MMDetection `log.json` for plotting."""
    with log_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_yolo_metrics(csv_path: Path) -> pd.DataFrame:
    """Read Ultralytics results CSV."""
    return pd.read_csv(csv_path)


def export_comparison(metrics: List[Dict[str, float]], output_path: Path) -> None:
    """Save combined metrics to CSV for the final report table."""
    df = pd.DataFrame(metrics)
    df.to_csv(output_path, index=False)


def compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def summarise_yolo_results(df: pd.DataFrame, speed_column: str = "speed/img") -> MetricsSummary:
    if df.empty:
        raise ValueError("YOLO results CSV is empty")

    last = df.iloc[-1]
    precision = float(last.get("metrics/precision(B)", 0.0))
    recall = float(last.get("metrics/recall(B)", 0.0))
    map50 = float(last.get("metrics/mAP50(B)", last.get("metrics/mAP50", 0.0)))
    map5095 = float(last.get("metrics/mAP50-95(B)", last.get("metrics/mAP50-95", 0.0)))
    inference_ms = float(last.get(speed_column, last.get("speed", 0.0)))
    fps = 0.0 if inference_ms == 0 else 1000.0 / inference_ms

    return MetricsSummary(
        map50=map50,
        map=map5095,
        precision=precision,
        recall=recall,
        f1=compute_f1(precision, recall),
        fps=fps,
    )


def summarise_fcos_results(entries: List[Dict], precision_key: str = "precision", recall_key: str = "recall") -> MetricsSummary:
    if not entries:
        raise ValueError("FCOS log is empty")

    val_entries = [entry for entry in entries if entry.get("mode") == "val"]
    if not val_entries:
        raise ValueError("No validation entries found in FCOS log")

    last = val_entries[-1]
    precision = float(last.get(precision_key, 0.0))
    recall = float(last.get(recall_key, 0.0))
    map50 = float(last.get("coco/bbox_mAP_50", 0.0))
    map_mean = float(last.get("coco/bbox_mAP", 0.0))
    fps = float(last.get("fps", 0.0))

    return MetricsSummary(
        map50=map50,
        map=map_mean,
        precision=precision,
        recall=recall,
        f1=compute_f1(precision, recall),
        fps=fps,
    )
