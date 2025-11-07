"""YOLOv8 fine-tuning utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO


@dataclass
class YOLOConfig:
    weights: str
    data_yaml: Path
    project_dir: Path
    device: Optional[str] = None
    epochs: int = 50
    imgsz: int = 512
    batch: Optional[int] = None
    name: str = "minecraft"


class YOLOFineTuner:
    """Minimal wrapper around the Ultralytics YOLO API."""

    def __init__(self, cfg: YOLOConfig) -> None:
        self.cfg = cfg
        self.model: Optional[YOLO] = None
        self.last_run_dir: Optional[Path] = None

    def setup(self) -> None:
        """Load the pretrained YOLO checkpoint."""

        if self.model is None:
            self.model = YOLO(self.cfg.weights)

    def train(self) -> None:
        """Run fine-tuning with the provided options."""

        if self.model is None:
            self.setup()
        assert self.model is not None

        self.cfg.project_dir.mkdir(parents=True, exist_ok=True)
        results = self.model.train(
            data=str(self.cfg.data_yaml),
            device=self.cfg.device,
            epochs=self.cfg.epochs,
            imgsz=self.cfg.imgsz,
            batch=self.cfg.batch,
            project=str(self.cfg.project_dir),
            name=self.cfg.name,
            exist_ok=True,
        )
        self.last_run_dir = Path(results.save_dir)

    def validation_metrics(self) -> Path:
        """Return the CSV file generated after training for plotting purposes."""

        if self.last_run_dir is None:
            raise RuntimeError("Training has not been executed yet")
        return self.last_run_dir / "results.csv"

    def inference_on_image(self, image_path: Path, output_dir: Path, conf: float = 0.25) -> Path:
        """Run inference on a single image and save the visualised output."""

        if self.model is None:
            self.setup()
        assert self.model is not None

        output_dir.mkdir(parents=True, exist_ok=True)
        results = self.model.predict(
            source=str(image_path),
            device=self.cfg.device,
            imgsz=self.cfg.imgsz,
            conf=conf,
            save=False,
        )

        if not results:
            raise RuntimeError("YOLO prediction returned no results")

        plotted = results[0].plot()
        output_path = output_dir / f"{image_path.stem}.jpg"
        cv2.imwrite(str(output_path), plotted)
        return output_path

    def inference_on_video(self, video_path: Path, output_path: Path, conf: float = 0.25) -> Path:
        """Run YOLO inference on a video and save the annotated output."""

        if self.model is None:
            self.setup()
        assert self.model is not None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions = self.model.predict(
            source=str(video_path),
            device=self.cfg.device,
            imgsz=self.cfg.imgsz,
            conf=conf,
            save=True,
            project=str(output_path.parent),
            name=output_path.stem,
            exist_ok=True,
        )

        generated_dir = Path(predictions[0].save_dir) if predictions else output_path.parent
        candidate = generated_dir / Path(video_path).name
        if candidate.exists():
            candidate.replace(output_path)
        return output_path
