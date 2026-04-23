"""YOLOv8 Part Detector Module.

This module provides YOLOv8-based object detection for auto parts.
"""
from ultralytics import YOLO
from typing import List, Optional
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of part detection."""
    part_type: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    class_name: str = ""


class YOLOPartDetector:
    """YOLOv8-based auto part detector."""
    
    # Auto parts categories mapping
    # Maps YOLO class names to auto part categories
    AUTO_PARTS_CATEGORIES = {
        'brake': ['brake_pad', 'brake_disc', 'brake_caliper', 'brake', 'rotor'],
        'filter': ['oil_filter', 'air_filter', 'fuel_filter', 'filter', 'cabin_filter'],
        'battery': ['car_battery', 'battery', 'accumulator'],
        'spark_plug': ['spark_plug', 'ignition_coil', 'plug'],
        'suspension': ['shock_absorber', 'strut', 'spring', 'suspension', 'damper'],
        'engine': ['engine_part', 'piston', 'gasket', 'timing_belt', 'serpentine_belt', 'belt'],
        'lighting': ['headlight', 'taillight', 'bulb', 'lamp', 'light'],
        'tire': ['tire', 'wheel', 'rim'],
        'exhaust': ['exhaust', 'muffler', 'catalytic', 'pipe'],
        'clutch': ['clutch', 'flywheel'],
        'radiator': ['radiator', 'cooling', 'fan'],
        'alternator': ['alternator', 'generator', 'starter'],
        'wiper': ['wiper', 'wiper_blade'],
    }
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        use_gpu: bool = False
    ):
        """Initialize the YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
            use_gpu: Whether to use GPU for inference
        """
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self._model = None
        
        logger.info(f"Initializing YOLO detector with model: {model_path}")
    
    @property
    def model(self) -> YOLO:
        """Lazy-load the model."""
        if self._model is None:
            logger.info(f"Loading YOLO model from {self.model_path}...")
            self._model = YOLO(self.model_path)
        return self._model
    
    def detect(self, image: np.ndarray) -> Optional[DetectionResult]:
        """Detect auto part in image.
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            DetectionResult if part detected, None otherwise
        """
        try:
            results = self.model(image, verbose=False)
            
            if not results or len(results[0].boxes) == 0:
                logger.debug("No objects detected in image")
                return None
            
            # Get highest confidence detection
            boxes = results[0].boxes
            best_idx = boxes.conf.argmax().item()
            
            confidence = boxes.conf[best_idx].item()
            if confidence < self.confidence_threshold:
                logger.debug(f"Detection confidence {confidence:.2f} below threshold {self.confidence_threshold}")
                return None
            
            cls_id = int(boxes.cls[best_idx].item())
            class_name = self.model.names[cls_id]
            
            # Map to auto part category
            part_type = self._map_to_part_type(class_name)
            
            bbox = boxes.xyxy[best_idx].cpu().numpy().astype(int).tolist()
            
            logger.info(f"Detected: {class_name} -> {part_type} (confidence: {confidence:.2f})")
            
            return DetectionResult(
                part_type=part_type,
                confidence=confidence,
                bbox=bbox,
                class_name=class_name
            )
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return None
    
    def detect_all(self, image: np.ndarray, top_k: int = 5) -> List[DetectionResult]:
        """Detect all auto parts in image.
        
        Args:
            image: Input image as numpy array (RGB)
            top_k: Maximum number of detections to return
            
        Returns:
            List of DetectionResult objects
        """
        try:
            results = self.model(image, verbose=False)
            
            if not results or len(results[0].boxes) == 0:
                return []
            
            boxes = results[0].boxes
            
            # Filter by confidence and sort
            detections = []
            for i in range(len(boxes)):
                confidence = boxes.conf[i].item()
                if confidence >= self.confidence_threshold:
                    cls_id = int(boxes.cls[i].item())
                    class_name = self.model.names[cls_id]
                    part_type = self._map_to_part_type(class_name)
                    bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                    
                    detections.append(DetectionResult(
                        part_type=part_type,
                        confidence=confidence,
                        bbox=bbox,
                        class_name=class_name
                    ))
            
            # Sort by confidence and return top_k
            detections.sort(key=lambda x: x.confidence, reverse=True)
            return detections[:top_k]
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []
    
    def _map_to_part_type(self, class_name: str) -> str:
        """Map YOLO class name to auto part category.
        
        Args:
            class_name: YOLO detected class name
            
        Returns:
            Auto part category string
        """
        class_lower = class_name.lower()
        
        for category, keywords in self.AUTO_PARTS_CATEGORIES.items():
            if any(kw in class_lower for kw in keywords):
                return category
        
        # If no mapping found, check for car-related objects
        car_related = ['car', 'vehicle', 'truck', 'motorcycle', 'wheel', 'tire']
        if any(kw in class_lower for kw in car_related):
            return "automotive"
        
        return "unknown"
    
    def get_supported_categories(self) -> List[str]:
        """Get list of supported auto part categories.
        
        Returns:
            List of category names
        """
        return list(self.AUTO_PARTS_CATEGORIES.keys())
