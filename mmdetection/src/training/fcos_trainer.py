"""High-level helpers for fine-tuning FCOS via MMDetection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import cv2
import torch
from mmcv import imread
from mmengine.config import Config
from mmengine.fileio import mkdir_or_exist
from mmengine.registry import VISUALIZERS
from mmengine.runner import Runner

from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules


@dataclass
class FCOSConfig:
    """Container for FCOS training configuration paths."""

    config_path: Path
    work_dir: Path
    checkpoint_path: Optional[Path] = None
    device: Optional[str] = None
    cfg_options: Optional[dict] = None


class FCOSFineTuner:
    """Wraps MMDetection training/inference routines for FCOS."""

    def __init__(self, cfg: FCOSConfig) -> None:
        self.cfg = cfg
        self.runner: Optional[Runner] = None
        self.mm_config: Optional[Config] = None
        self._inference_model = None
        self._inference_checkpoint: Optional[Path] = None
        register_all_modules()

    def setup(self) -> None:
        """Load the config, initialise datasets, and build the MMEngine runner."""

        mm_cfg = Config.fromfile(str(self.cfg.config_path))
        if self.cfg.cfg_options:
            mm_cfg.merge_from_dict(self.cfg.cfg_options)

        mm_cfg.work_dir = str(self.cfg.work_dir)
        if self.cfg.checkpoint_path:
            mm_cfg.load_from = str(self.cfg.checkpoint_path)

        mkdir_or_exist(mm_cfg.work_dir)

        self.mm_config = mm_cfg
        self.runner = Runner.from_cfg(mm_cfg)

    def _ensure_model(self, checkpoint_path: Optional[Path]) -> Any:
        if self.mm_config is None:
            self.setup()
        assert self.mm_config is not None

        checkpoint = checkpoint_path or self.cfg.checkpoint_path
        checkpoint_str = Path(checkpoint) if checkpoint else None
        if (
            self._inference_model is None
            or (checkpoint_str and self._inference_checkpoint != checkpoint_str)
        ):
            device = self.cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            self._inference_model = init_detector(self.mm_config, checkpoint=checkpoint, device=device)
            self._inference_checkpoint = checkpoint_str
        return self._inference_model

    def train(self) -> None:
        """Launch fine-tuning using the configured runner."""

        if self.runner is None:
            self.setup()
        assert self.runner is not None
        self.runner.train()

    def inference_on_image(
        self,
        image_path: Path,
        output_path: Path,
        checkpoint_path: Optional[Path] = None,
        score_thr: float = 0.25,
    ) -> None:
        """Run inference on a single image and save the visualised output."""

        model = self._ensure_model(checkpoint_path)

        data_sample = inference_detector(model, str(image_path))
        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.dataset_meta = model.dataset_meta

        image = imread(str(image_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        visualizer.add_datasample(
            name=image_path.stem,
            image=image,
            data_sample=data_sample,
            draw_gt=False,
            show=False,
            out_file=str(output_path),
            pred_score_thr=score_thr,
        )

    def inference_on_video(
        self,
        video_path: Path,
        output_path: Path,
        checkpoint_path: Optional[Path] = None,
        score_thr: float = 0.25,
    ) -> None:
        """Run inference on a video stream and persist the annotated recording."""

        model = self._ensure_model(checkpoint_path)
        dataset_meta = model.dataset_meta or {}
        classes = dataset_meta.get("classes", [])

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                data_sample = inference_detector(model, frame)
                instances = data_sample.pred_instances
                if instances is not None:
                    boxes = instances.bboxes.cpu().numpy()
                    scores = instances.scores.cpu().numpy()
                    labels = instances.labels.cpu().numpy()

                    for box, score, label in zip(boxes, scores, labels):
                        if score < score_thr:
                            continue
                        x1, y1, x2, y2 = box.astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        class_name = classes[label] if label < len(classes) else str(label)
                        cv2.putText(
                            frame,
                            f"{class_name}:{score:.2f}",
                            (x1, max(y1 - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )

                writer.write(frame)
        finally:
            cap.release()
            writer.release()

    def export_metrics(self) -> Path:
        """Return the path to the metrics log for downstream plotting."""

        return Path(self.mm_config.work_dir if self.mm_config else self.cfg.work_dir) / "log.json"
