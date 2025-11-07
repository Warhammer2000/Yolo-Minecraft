"""Conversion utilities from Pascal VOC XML annotations to COCO JSON format.

This module prepares the Minecraft mobs dataset for usage with MMDetection and
Ultralytics YOLO pipelines by turning the supplied Pascal VOC annotations into
COCO-compliant JSON files.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import xml.etree.ElementTree as ET


LOGGER = logging.getLogger(__name__)


CLASSES: Sequence[str] = (
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
)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class CocoPaths:
    """Container with dataset root and annotation output directory."""

    dataset_root: Path
    annotations_dir: Path


def _category_mapping(classes: Sequence[str]) -> Dict[str, int]:
    """Create a mapping from class name to COCO category id."""

    return {name: idx + 1 for idx, name in enumerate(classes)}


def _find_image_file(xml_path: Path) -> Path:
    """Find the corresponding image file for a given VOC XML annotation."""

    stem = xml_path.stem
    for extension in IMAGE_EXTENSIONS:
        candidate = xml_path.with_suffix(extension)
        if candidate.exists():
            return candidate
        nested = xml_path.parent / "images" / f"{stem}{extension}"
        if nested.exists():
            return nested
    raise FileNotFoundError(f"No image file found for annotation '{xml_path.name}'")


def _parse_int(node: ET.Element | None, default: int = 0) -> int:
    if node is None or node.text is None:
        return default
    return int(float(node.text))


def _load_annotation(xml_path: Path) -> ET.Element:
    return ET.parse(str(xml_path)).getroot()


def _create_image_record(image_id: int, image_path: Path, size_node: ET.Element) -> Dict:
    width = _parse_int(size_node.find("width"))
    height = _parse_int(size_node.find("height"))
    return {
        "id": image_id,
        "file_name": image_path.name,
        "width": width,
        "height": height,
    }


def _create_annotation_records(
    annotation_id_start: int,
    image_id: int,
    object_nodes: Iterable[ET.Element],
    class_to_id: Dict[str, int],
) -> List[Dict]:
    annotations: List[Dict] = []
    annotation_id = annotation_id_start
    for obj in object_nodes:
        name = obj.findtext("name")
        if name not in class_to_id:
            LOGGER.warning("Skipping unknown class '%s'", name)
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            LOGGER.warning("Missing bounding box for object '%s'", name)
            continue

        xmin = _parse_int(bndbox.find("xmin"))
        ymin = _parse_int(bndbox.find("ymin"))
        xmax = _parse_int(bndbox.find("xmax"))
        ymax = _parse_int(bndbox.find("ymax"))

        x = max(0, xmin - 1)
        y = max(0, ymin - 1)
        width = max(0, xmax - xmin + 1)
        height = max(0, ymax - ymin + 1)
        area = float(width * height)

        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_to_id[name],
                "bbox": [float(x), float(y), float(width), float(height)],
                "area": area,
                "segmentation": [],
                "iscrowd": 0,
            }
        )
        annotation_id += 1

    return annotations


def convert_split(split_dir: Path, classes: Sequence[str]) -> Dict:
    """Convert a Pascal VOC split directory to a COCO dictionary."""

    xml_files = sorted(split_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML annotations found in '{split_dir}'")

    class_to_id = _category_mapping(classes)
    categories = [
        {"id": cid, "name": name, "supercategory": "mob"}
        for name, cid in class_to_id.items()
    ]

    images: List[Dict] = []
    annotations: List[Dict] = []

    image_id = 1
    annotation_id = 1

    for xml_path in xml_files:
        root = _load_annotation(xml_path)
        try:
            image_path = _find_image_file(xml_path)
        except FileNotFoundError as exc:
            LOGGER.warning(str(exc))
            continue

        size_node = root.find("size")
        if size_node is None:
            LOGGER.warning("Missing <size> tag in '%s'", xml_path.name)
            continue

        images.append(_create_image_record(image_id, image_path, size_node))

        object_nodes = root.findall("object")
        if not object_nodes:
            LOGGER.warning("No objects annotated in '%s'", xml_path.name)

        new_annotations = _create_annotation_records(
            annotation_id_start=annotation_id,
            image_id=image_id,
            object_nodes=object_nodes,
            class_to_id=class_to_id,
        )
        annotations.extend(new_annotations)
        annotation_id += len(new_annotations)
        image_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def save_coco_dict(coco_dict: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(coco_dict, handle, ensure_ascii=False, indent=2)


def convert_dataset(paths: CocoPaths, splits: Sequence[str]) -> Dict[str, Path]:
    """Convert multiple dataset splits and write COCO JSON files."""

    results: Dict[str, Path] = {}
    for split in splits:
        split_dir = paths.dataset_root / split
        if not split_dir.exists():
            LOGGER.warning("Split directory '%s' does not exist, skipping", split_dir)
            continue

        LOGGER.info("Converting split '%s'", split)
        coco_dict = convert_split(split_dir, CLASSES)
        output_path = paths.annotations_dir / f"{split}.json"
        save_coco_dict(coco_dict, output_path)
        results[split] = output_path

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the dataset root containing train/val/test folders",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=None,
        help="Directory to place generated COCO json files. Defaults to <dataset_root>/annotations",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=("train", "val", "test"),
        help="Dataset splits to convert",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG)

    annotations_dir = args.annotations_dir or (args.dataset_root / "annotations")
    paths = CocoPaths(dataset_root=args.dataset_root, annotations_dir=annotations_dir)

    convert_dataset(paths, args.splits)
    LOGGER.info("COCO annotations created in '%s'", annotations_dir)


if __name__ == "__main__":
    main()