"""Plotting helpers for dataset statistics and inference outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_class_distribution(counts: Dict[str, int], output_path: Path | None = None) -> None:
    """Produce and optionally save a bar chart for class counts."""
    classes = list(counts.keys())
    values = [counts[name] for name in classes]

    plt.figure(figsize=(12, 5))
    plt.bar(classes, values, color="#4c72b0")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Instances")
    plt.title("Minecraft dataset class distribution")
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    plt.close()


def visualize_bboxes(
    image_path: Path,
    annotations: Iterable[Mapping],
    category_id_to_name: Mapping[int, str],
    output_path: Path | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """Render bounding boxes for a single image using matplotlib."""

    image = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)

    for annotation in annotations:
        bbox = annotation.get("bbox")
        if bbox is None:
            continue
        x, y, w, h = bbox
        label = category_id_to_name.get(annotation.get("category_id", -1), "unknown")

        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(x, y, label, color="white", fontsize=9, bbox=dict(facecolor="green", alpha=0.6))

    ax.axis("off")
    ax.set_title(image_path.name)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
