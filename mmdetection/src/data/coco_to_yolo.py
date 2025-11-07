"""Utilities to convert COCO annotations into YOLO txt label files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence


@dataclass(frozen=True)
class ConversionSpec:
    dataset_root: Path
    annotations_dir: Path
    output_dir: Path
    classes: Sequence[str]


def _load_coco(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalise_bbox(bbox: Sequence[float], width: int, height: int) -> Sequence[float]:
    x, y, w, h = bbox
    x_centre = (x + w / 2.0) / width
    y_centre = (y + h / 2.0) / height
    w_norm = w / width
    h_norm = h / height
    return (
        max(0.0, min(1.0, x_centre)),
        max(0.0, min(1.0, y_centre)),
        max(0.0, min(1.0, w_norm)),
        max(0.0, min(1.0, h_norm)),
    )


def convert_split(spec: ConversionSpec, split: str, annotation_file: Path) -> None:
    coco = _load_coco(annotation_file)
    category_id_to_index = {}
    for category in coco.get("categories", []):
        try:
            class_index = spec.classes.index(category["name"])
        except ValueError:
            continue
        category_id_to_index[category["id"]] = class_index

    images = {image["id"]: image for image in coco.get("images", [])}

    labels_root = spec.output_dir / split
    labels_root.mkdir(parents=True, exist_ok=True)

    image_to_annotations: Dict[int, list] = {img_id: [] for img_id in images}
    for annotation in coco.get("annotations", []):
        image_to_annotations.setdefault(annotation["image_id"], []).append(annotation)

    for image_id, image_info in images.items():
        file_name = Path(image_info["file_name"]).stem
        width = int(image_info["width"])
        height = int(image_info["height"])

        annotations = image_to_annotations.get(image_id, [])
        label_lines = []
        for annotation in annotations:
            category_index = category_id_to_index.get(annotation["category_id"])
            if category_index is None:
                continue
            bbox = _normalise_bbox(annotation["bbox"], width, height)
            label_lines.append(
                f"{category_index} " + " ".join(f"{value:.6f}" for value in bbox)
            )

        label_path = labels_root / f"{file_name}.txt"
        with label_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(label_lines))


def convert(spec: ConversionSpec, splits: Iterable[str]) -> None:
    annotation_paths = {
        split: spec.annotations_dir / f"{split}_annotations.coco.json" for split in splits
    }
    for split, annotation_path in annotation_paths.items():
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        convert_split(spec, split, annotation_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=None,
        help="Directory containing COCO annotation json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to place generated YOLO label files. Defaults to <dataset_root>/labels",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=("train", "val", "test"),
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=(
            "bee",
            "chicken",
            "cow",
            "creeper",
            "enderman",
            "fox",
            "frog",
            "ghast",
            "goat",
            "llama",
            "pig",
            "sheep",
            "skeleton",
            "spider",
            "turtle",
            "wolf",
            "zombie",
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    annotations_dir = args.annotations_dir or (args.dataset_root / "annotations")
    output_dir = args.output_dir or (args.dataset_root / "labels")
    spec = ConversionSpec(
        dataset_root=args.dataset_root,
        annotations_dir=annotations_dir,
        output_dir=output_dir,
        classes=tuple(args.classes),
    )
    convert(spec, args.splits)


if __name__ == "__main__":
    main()


