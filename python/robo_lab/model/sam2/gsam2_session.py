"""
Grounded-SAM2 Session: Text-prompted image segmentation.

Usage:
    from robo_lab.model.sam2 import GSamSession

    session = GSamSession(device="cuda")
    result = session.segment(image, prompt="car. person.")

    # result.masks      - binary masks (N, H, W)
    # result.boxes      - bounding boxes (N, 4) in xyxy format
    # result.labels     - detected class names
    # result.scores     - confidence scores
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

# Add Grounded-SAM-2 to path.
# Repo layout from this file is: python/robo_lab/model/sam2/gsam2_session.py
# The vendored dependency lives in: python/third_party/Grounded-SAM-2
_PYTHON_ROOT = Path(__file__).resolve().parents[3]
_GSAM2_ROOT = _PYTHON_ROOT / "third_party" / "Grounded-SAM-2"
if str(_GSAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(_GSAM2_ROOT))

# Also ensure the grounding_dino subdir is accessible for "groundingdino" imports
_GDINO_ROOT = _GSAM2_ROOT / "grounding_dino"
if str(_GDINO_ROOT) not in sys.path:
    sys.path.insert(0, str(_GDINO_ROOT))


@dataclass
class SegmentResult:
    """
    Result from Grounded-SAM2 segmentation.

    Attributes:
        masks: Binary segmentation masks. Shape: (N, H, W) where N is number of detections.
               Each mask is a boolean array where True = object pixel.

        boxes: Bounding boxes in xyxy format (x_min, y_min, x_max, y_max).
               Shape: (N, 4), pixel coordinates.

        labels: Class names for each detection. List of N strings.
                These come from parsing the text prompt (e.g., "car", "person").

        scores: Confidence scores for each detection. Shape: (N,), range [0, 1].
                Higher = more confident. Combines GroundingDINO box confidence.

        image_size: Original image size as (height, width).

    Example:
        result = session.segment(image, "car. tire.")
        for i in range(len(result)):
            print(f"Object {i}: {result.labels[i]} ({result.scores[i]:.2f})")
            mask = result.masks[i]  # H x W boolean mask
            box = result.boxes[i]   # [x1, y1, x2, y2]
    """
    masks: np.ndarray
    boxes: np.ndarray
    labels: list[str]
    scores: np.ndarray
    image_size: tuple[int, int]

    def __len__(self) -> int:
        return len(self.labels)

    def __repr__(self) -> str:
        return (
            f"SegmentResult(n_detections={len(self)}, "
            f"labels={self.labels}, image_size={self.image_size})"
        )

    def to_supervision(self):
        """Convert to supervision.Detections for visualization."""
        import supervision as sv
        return sv.Detections(
            xyxy=self.boxes,
            mask=self.masks.astype(bool),
            class_id=np.arange(len(self)),
            confidence=self.scores,
        )


class GSamSession:
    """
    Session wrapper for Grounded-SAM2 inference.

    Combines GroundingDINO (text-to-box) + SAM2 (box-to-mask) for
    text-prompted instance segmentation.

    Args:
        sam2_checkpoint: Path to SAM2 checkpoint (.pt file).
        sam2_config: SAM2 model config name (e.g., "sam2.1_hiera_l.yaml").
        gdino_config: GroundingDINO config path.
        gdino_checkpoint: GroundingDINO checkpoint path.
        device: "cuda" or "cpu".
        box_threshold: Confidence threshold for GroundingDINO boxes.
        text_threshold: Text similarity threshold for GroundingDINO.

    Example:
        session = GSamSession()
        result = session.segment(image, "cup. bottle. keyboard.")
        print(f"Found {len(result)} objects")
    """

    def __init__(
        self,
        sam2_checkpoint: Optional[str] = None,
        sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        gdino_config: Optional[str] = None,
        gdino_checkpoint: Optional[str] = None,
        device: str = "cuda",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Default paths relative to Grounded-SAM-2
        if sam2_checkpoint is None:
            sam2_checkpoint = str(_GSAM2_ROOT / "checkpoints" / "sam2.1_hiera_large.pt")
        if gdino_config is None:
            gdino_config = str(
                _GSAM2_ROOT / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
            )
        if gdino_checkpoint is None:
            gdino_checkpoint = str(_GSAM2_ROOT / "gdino_checkpoints" / "groundingdino_swint_ogc.pth")

        self._sam2_checkpoint = sam2_checkpoint
        self._sam2_config = sam2_config
        self._gdino_config = gdino_config
        self._gdino_checkpoint = gdino_checkpoint

        self._sam2_predictor = None
        self._gdino_model = None
        self._loaded = False

    def _lazy_load(self):
        """Load models on first use (avoids slow import at session creation)."""
        if self._loaded:
            return

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from groundingdino.util.inference import load_model

        # Build SAM2
        sam2_model = build_sam2(self._sam2_config, self._sam2_checkpoint, device=self.device)
        self._sam2_predictor = SAM2ImagePredictor(sam2_model)

        # Build GroundingDINO
        self._gdino_model = load_model(
            model_config_path=self._gdino_config,
            model_checkpoint_path=self._gdino_checkpoint,
            device=self.device,
        )

        self._loaded = True

    def segment(
        self,
        image: Union[np.ndarray, str, Path],
        prompt: str,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        multimask_output: bool = False,
    ) -> SegmentResult:
        """
        Segment objects in image based on text prompt.

        Args:
            image: Input image as numpy array (H, W, 3) BGR/RGB, or path to image file.
            prompt: Text prompt describing objects to find.
                    Format: "object1. object2. object3." (dot-separated, lowercase).
                    Example: "car. person. traffic light."
            box_threshold: Override default box confidence threshold.
            text_threshold: Override default text similarity threshold.
            multimask_output: If True, SAM2 outputs multiple mask candidates per box.

        Returns:
            SegmentResult with masks, boxes, labels, and scores.
        """
        self._lazy_load()

        from torchvision.ops import box_convert
        from groundingdino.util.inference import load_image, predict

        box_threshold = box_threshold or self.box_threshold
        text_threshold = text_threshold or self.text_threshold

        # Load image
        if isinstance(image, (str, Path)):
            image_source, image_tensor = load_image(str(image))
        else:
            # numpy array input
            image_source = image
            image_tensor = self._preprocess_image(image)

        h, w = image_source.shape[:2]

        # GroundingDINO: text -> boxes
        boxes, confidences, labels = predict(
            model=self._gdino_model,
            image=image_tensor,
            caption=prompt.lower(),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        if len(boxes) == 0:
            return SegmentResult(
                masks=np.zeros((0, h, w), dtype=bool),
                boxes=np.zeros((0, 4), dtype=np.float32),
                labels=[],
                scores=np.array([], dtype=np.float32),
                image_size=(h, w),
            )

        # Convert boxes to pixel coordinates (xyxy)
        boxes_pixel = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes_pixel, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # SAM2: boxes -> masks
        self._sam2_predictor.set_image(image_source)

        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            masks, scores, _ = self._sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=multimask_output,
            )

        # Select best mask per box if multimask
        if multimask_output and masks.ndim == 4:
            best_idx = np.argmax(scores, axis=1)
            masks = masks[np.arange(masks.shape[0]), best_idx]
            scores = scores[np.arange(scores.shape[0]), best_idx]

        # Ensure shape is (N, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        return SegmentResult(
            masks=masks.astype(bool),
            boxes=input_boxes.astype(np.float32),
            labels=list(labels),
            scores=confidences.numpy().astype(np.float32),
            image_size=(h, w),
        )

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to tensor for GroundingDINO."""
        import torchvision.transforms as T

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Assume RGB input; if BGR, user should convert
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        return transform(image)

    def visualize(
        self,
        image: np.ndarray,
        result: SegmentResult,
        output_path: Optional[str] = None,
        show_boxes: bool = True,
        show_masks: bool = True,
        show_labels: bool = True,
    ) -> np.ndarray:
        """
        Visualize segmentation results on image.

        Args:
            image: Original image (H, W, 3) BGR format for cv2.
            result: SegmentResult from segment().
            output_path: If provided, save annotated image to this path.
            show_boxes: Draw bounding boxes.
            show_masks: Overlay segmentation masks.
            show_labels: Draw class labels.

        Returns:
            Annotated image as numpy array.
        """
        import cv2
        import supervision as sv

        detections = result.to_supervision()
        annotated = image.copy()

        if show_boxes:
            box_annotator = sv.BoxAnnotator()
            annotated = box_annotator.annotate(scene=annotated, detections=detections)

        if show_labels:
            labels_with_conf = [
                f"{label} {score:.2f}"
                for label, score in zip(result.labels, result.scores)
            ]
            label_annotator = sv.LabelAnnotator()
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels_with_conf
            )

        if show_masks:
            mask_annotator = sv.MaskAnnotator()
            annotated = mask_annotator.annotate(scene=annotated, detections=detections)

        if output_path:
            cv2.imwrite(output_path, annotated)

        return annotated


# Quick test
if __name__ == "__main__":
    import cv2

    session = GSamSession(device="cuda")

    # Test with the demo image
    test_image = str(_GSAM2_ROOT / "notebooks" / "images" / "truck.jpg")
    result = session.segment(test_image, prompt="car. tire.")

    print(f"Found {len(result)} objects:")
    for i in range(len(result)):
        print(f"  [{i}] {result.labels[i]}: score={result.scores[i]:.3f}, "
              f"box={result.boxes[i].astype(int).tolist()}, "
              f"mask_area={result.masks[i].sum()} px")

    # Visualize
    img = cv2.imread(test_image)
    annotated = session.visualize(img, result, output_path="gsam2_test_output.jpg")
    print(f"\nSaved visualization to gsam2_test_output.jpg")
