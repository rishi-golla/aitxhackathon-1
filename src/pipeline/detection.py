"""
YOLO-World Zero-Shot Detection Module for PPE and Hazard Detection.

Uses YOLO-World for text-prompted object detection without retraining.
"""

import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import structlog

log = structlog.get_logger()

# Try importing ultralytics
try:
    from ultralytics import YOLOWorld, YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    log.warning("ultralytics_not_installed")

# Try importing torch for device detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class Detection:
    """Single detection result."""
    class_name: str  # "hardhat", "safety_vest", "bare_hands"
    confidence: float  # 0.0-1.0
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    track_id: Optional[int] = None  # From tracker


@dataclass
class DetectionResult:
    """Result from running detection on a frame."""
    frame_id: int
    timestamp: float
    detections: list[Detection]
    inference_time_ms: float
    model_name: str


class PPEDetector:
    """
    PPE and hazard detector using YOLO-World zero-shot detection.

    Features:
    - Text-prompted detection without retraining
    - Dynamic class updates
    - Batch inference support
    - TensorRT export for speedup
    """

    # Default classes for factory safety detection
    DEFAULT_CLASSES = [
        # Workers
        "person", "worker", "human",
        # Head protection
        "hardhat", "safety helmet", "hard hat", "helmet",
        # Eye protection
        "safety glasses", "goggles", "face shield", "welding mask",
        # Body protection
        "safety vest", "high visibility vest", "reflective vest",
        # Hand protection
        "gloves", "safety gloves", "work gloves", "bare hands", "exposed hands",
        # Foot protection
        "safety boots", "steel toe boots",
        # Equipment/Hazards
        "forklift", "machinery", "industrial machine", "conveyor belt",
        "ladder", "scaffolding",
        # Hazardous materials
        "chemical container", "gas cylinder", "warning sign",
        "fire extinguisher",
        # PPE violations
        "no hardhat", "no vest", "no gloves"
    ]

    def __init__(
        self,
        model_size: str = "yolov8x-worldv2",
        confidence_threshold: float = 0.3,
        custom_classes: Optional[list[str]] = None,
        device: str = "auto",
        enable_tracking: bool = False
    ):
        """
        Initialize the PPE detector.

        Args:
            model_size: YOLO-World model variant
                       ("yolov8s-worldv2", "yolov8m-worldv2", "yolov8x-worldv2")
            confidence_threshold: Minimum confidence for detections
            custom_classes: Optional custom class list (uses DEFAULT_CLASSES if None)
            device: Device to run on ("cuda", "cpu", or "auto")
            enable_tracking: Enable ByteTrack for object tracking
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")

        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.classes = custom_classes or self.DEFAULT_CLASSES.copy()
        self.enable_tracking = enable_tracking
        self._model: Optional[YOLOWorld] = None
        self._warmed_up = False

        # Determine device
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        log.info(
            "ppe_detector_init",
            model_size=model_size,
            device=self.device,
            num_classes=len(self.classes)
        )

        self._load_model()

    def _load_model(self) -> None:
        """Load the YOLO-World model."""
        try:
            # Check for local model first
            local_path = f"models/{self.model_size}.pt"
            try:
                self._model = YOLOWorld(local_path)
                log.info("model_loaded_local", path=local_path)
            except Exception:
                # Download from hub
                self._model = YOLOWorld(self.model_size)
                log.info("model_loaded_hub", model=self.model_size)

            # Move to device
            if self.device == "cuda":
                self._model.to("cuda")

            # Set initial classes
            self._model.set_classes(self.classes)

            log.info("ppe_detector_ready", device=self.device, classes=len(self.classes))

        except Exception as e:
            log.error("model_load_failed", error=str(e))
            raise

    def _warmup(self) -> None:
        """Warmup the model with a dummy inference."""
        if self._warmed_up:
            return

        try:
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model.predict(dummy_frame, verbose=False)
            self._warmed_up = True
            log.info("model_warmup_complete")
        except Exception as e:
            log.warning("model_warmup_failed", error=str(e))

    def detect(
        self,
        frame: np.ndarray,
        confidence_override: Optional[float] = None
    ) -> DetectionResult:
        """
        Run detection on a single frame.

        Args:
            frame: BGR image as numpy array
            confidence_override: Override default confidence threshold

        Returns:
            DetectionResult with all detections
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        # Warmup on first call
        if not self._warmed_up:
            self._warmup()

        conf = confidence_override or self.confidence_threshold
        start_time = time.time()

        # Run inference
        if self.enable_tracking:
            results = self._model.track(
                frame,
                conf=conf,
                verbose=False,
                persist=True
            )
        else:
            results = self._model.predict(
                frame,
                conf=conf,
                verbose=False
            )

        inference_time = (time.time() - start_time) * 1000

        # Parse results
        detections = []
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Get track ID if tracking enabled
                track_id = None
                if self.enable_tracking and box.id is not None:
                    track_id = int(box.id[0])

                detection = Detection(
                    class_name=result.names[cls_id],
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    track_id=track_id
                )
                detections.append(detection)

        return DetectionResult(
            frame_id=0,  # Set by caller
            timestamp=time.time(),
            detections=detections,
            inference_time_ms=inference_time,
            model_name=self.model_size
        )

    def detect_batch(
        self,
        frames: list[np.ndarray],
        confidence_override: Optional[float] = None
    ) -> list[DetectionResult]:
        """
        Run detection on a batch of frames.

        Args:
            frames: List of BGR images
            confidence_override: Override default confidence threshold

        Returns:
            List of DetectionResult, one per frame
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        if not self._warmed_up:
            self._warmup()

        conf = confidence_override or self.confidence_threshold
        start_time = time.time()

        # Batch inference
        results = self._model.predict(
            frames,
            conf=conf,
            verbose=False
        )

        total_time = (time.time() - start_time) * 1000
        per_frame_time = total_time / len(frames)

        # Parse all results
        detection_results = []

        for idx, result in enumerate(results):
            detections = []

            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    detection = Detection(
                        class_name=result.names[cls_id],
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        track_id=None
                    )
                    detections.append(detection)

            detection_results.append(DetectionResult(
                frame_id=idx,
                timestamp=time.time(),
                detections=detections,
                inference_time_ms=per_frame_time,
                model_name=self.model_size
            ))

        return detection_results

    def update_classes(self, classes: list[str]) -> None:
        """
        Update detection classes without reloading model.

        Args:
            classes: New list of class names for detection
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        self.classes = classes
        self._model.set_classes(classes)
        log.info("classes_updated", num_classes=len(classes))

    def add_classes(self, new_classes: list[str]) -> None:
        """Add new classes to existing set."""
        updated = list(set(self.classes + new_classes))
        self.update_classes(updated)

    def remove_classes(self, classes_to_remove: list[str]) -> None:
        """Remove classes from detection set."""
        updated = [c for c in self.classes if c not in classes_to_remove]
        self.update_classes(updated)

    def set_confidence(self, threshold: float) -> None:
        """Update confidence threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        self.confidence_threshold = threshold
        log.info("confidence_updated", threshold=threshold)

    def export_tensorrt(
        self,
        output_path: str,
        imgsz: int = 640,
        half: bool = True
    ) -> str:
        """
        Export model to TensorRT for faster inference.

        Args:
            output_path: Path for exported model
            imgsz: Input image size
            half: Use FP16 (recommended for speed)

        Returns:
            Path to exported model
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        log.info("tensorrt_export_starting", output_path=output_path)

        exported_path = self._model.export(
            format="engine",
            imgsz=imgsz,
            half=half,
            device=0 if self.device == "cuda" else "cpu"
        )

        log.info("tensorrt_export_complete", path=exported_path)
        return exported_path

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "num_classes": len(self.classes),
            "classes": self.classes,
            "tracking_enabled": self.enable_tracking,
            "warmed_up": self._warmed_up
        }


def draw_detections(
    frame: np.ndarray,
    result: DetectionResult,
    violation_classes: Optional[set[str]] = None,
    show_confidence: bool = True
) -> np.ndarray:
    """
    Draw detection bounding boxes on frame.

    Args:
        frame: BGR image
        result: Detection result
        violation_classes: Class names to highlight in red
        show_confidence: Show confidence scores

    Returns:
        Annotated frame
    """
    import cv2

    violation_classes = violation_classes or {"bare hands", "no hardhat", "no vest"}
    annotated = frame.copy()

    for det in result.detections:
        x1, y1, x2, y2 = det.bbox

        # Color based on violation status
        if det.class_name.lower() in violation_classes:
            color = (0, 0, 255)  # Red for violations
        else:
            color = (0, 255, 0)  # Green for compliant

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label
        label = det.class_name
        if show_confidence:
            label += f" {det.confidence:.2f}"
        if det.track_id is not None:
            label += f" #{det.track_id}"

        # Label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(
            annotated,
            (x1, y1 - label_h - 10),
            (x1 + label_w, y1),
            color,
            -1
        )
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

    return annotated
