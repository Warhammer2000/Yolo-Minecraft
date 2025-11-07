"""Utilities for validating the Minecraft COCO dataset structure."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import json

CLASSES = (
    "bee", "chicken", "cow", "creeper", "enderman", "fox", "frog", "ghast",
    "goat", "llama", "pig", "sheep", "skeleton", "spider", "turtle", "wolf", "zombie"
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_coco_annotations(annotation_path: Path) -> Dict:
    """Load a COCO annotation JSON file.

    Args:
        annotation_path: Path to the `_annotations.coco.json` file.

    Returns:
        Parsed JSON dictionary.
    """
    with annotation_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_image_annotation_counts(annotations: Dict) -> Tuple[int, int]:
    """Return the number of images and annotations present in a COCO dictionary."""
    return len(annotations.get("images", [])), len(annotations.get("annotations", []))


def iter_image_files(split_dir: Path) -> Iterable[Path]:
    """Yield all image files under a dataset split directory."""

    for extension in IMAGE_EXTENSIONS:
        yield from split_dir.glob(f"*{extension}")


def ensure_annotation_files_exist(dataset_root: Path) -> Dict[str, Path]:
    """Ensure the expected COCO annotation files exist for each split."""

    annotation_dir = dataset_root / "annotations"
    required = {
        "train": annotation_dir / "train_annotations.coco.json",
        "val": annotation_dir / "val_annotations.coco.json",
        "test": annotation_dir / "test_annotations.coco.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing annotation files: {', '.join(missing)}")
    return required


def ensure_class_coverage(annotations: Dict, expected_classes: Iterable[str] = CLASSES) -> None:
    """Validate that all expected classes exist in the COCO metadata.

    Raises:
        ValueError: When a class is missing.
    """
    categories = {category["name"] for category in annotations.get("categories", [])}
    missing = set(expected_classes) - categories
    if missing:
        raise ValueError(f"Missing classes in annotations: {sorted(missing)}")


def summarize_class_distribution(annotations: Dict) -> Dict[str, int]:
    """Compute per-class instance counts from annotation records."""
    categories = {category["id"]: category["name"] for category in annotations.get("categories", [])}
    counts: Dict[str, int] = {name: 0 for name in categories.values()}
    for ann in annotations.get("annotations", []):
        name = categories.get(ann["category_id"])
        if name is None:
            continue
        counts[name] = counts.get(name, 0) + 1
    return counts
