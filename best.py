#!/usr/bin/env python3
"""
Features:
- Kalman Filter tracking for smooth object motion
- Temporal smoothing for masks, boxes, and threat levels
- Async processing pipeline for better FPS
- Detection persistence to prevent flickering
- Exponential moving average for smooth transitions
- Memory-efficient operations
- Configurable quality presets
"""

import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import warnings
import torch
from threading import Thread, Lock
from queue import Queue
import copy

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class QualityPreset(Enum):
    PERFORMANCE = "performance"  # Max FPS, lower quality
    BALANCED = "balanced"        # Good balance
    QUALITY = "quality"          # Best visuals, lower FPS


class Config:
    # ─── Source Settings ───
    USE_TEST_VIDEO: bool = True
    TEST_VIDEO_PATH: str = "test3.mp4"
    LIVE_CAMERA_INDEX: int = 0
    
    # ─── Output Settings ───
    SAVE_OUTPUT: bool = True
    OUTPUT_PATH: str = "output_optimized.mp4"
    
    # ─── Quality Preset ───
    QUALITY_PRESET: QualityPreset = QualityPreset.BALANCED
    
    # ─── Model Settings ───
    YOLO_MODEL: str = "yolov8m-seg.pt"
    CONFIDENCE_THRESHOLD: float = 0.35  # Slightly lower for better recall
    IOU_THRESHOLD: float = 0.5  # NMS threshold
    
    # ─── Indian Road Classes ───
    ALL_CLASSES: List[int] = [0, 1, 2, 3, 5, 7, 16, 17, 18, 19]
    
    # ─── Danger Thresholds ───
    CRITICAL_AREA: float = 0.18
    HIGH_AREA: float = 0.08
    MEDIUM_AREA: float = 0.03
    CENTER_ZONE: float = 0.30
    
    # ─── Tracking Settings ───
    TRACK_MAX_AGE: int = 15          # Frames to keep lost tracks
    TRACK_MIN_HITS: int = 2          # Min detections before confirmed
    TRACK_IOU_THRESHOLD: float = 0.3  # IOU for track matching
    
    # ─── Smoothing Settings ───
    BOX_SMOOTHING: float = 0.4       # EMA alpha for boxes (lower = smoother)
    MASK_SMOOTHING: float = 0.5      # EMA alpha for masks
    THREAT_SMOOTHING: int = 8        # Frames to smooth threat level
    
    # ─── Processing Settings ───
    SKIP_FRAMES: int = 1             # Process every N frames (1 = all frames)
    ASYNC_PROCESSING: bool = True    # Use separate thread for detection
    
    # ─── Overlay Settings ───
    MASK_ALPHA: float = 0.45
    EDGE_GLOW: bool = True
    ANTI_ALIAS: bool = True
    
    @classmethod
    def apply_preset(cls):
        """Apply quality preset settings."""
        if cls.QUALITY_PRESET == QualityPreset.PERFORMANCE:
            cls.YOLO_MODEL = "yolov8s-seg.pt"
            cls.SKIP_FRAMES = 2
            cls.BOX_SMOOTHING = 0.5
            cls.MASK_ALPHA = 0.4
            cls.EDGE_GLOW = False
            cls.ANTI_ALIAS = False
        elif cls.QUALITY_PRESET == QualityPreset.QUALITY:
            cls.YOLO_MODEL = "yolov8m-seg.pt"
            cls.SKIP_FRAMES = 1
            cls.BOX_SMOOTHING = 0.3
            cls.MASK_ALPHA = 0.5
            cls.EDGE_GLOW = True
            cls.ANTI_ALIAS = True


# Apply preset on import
Config.apply_preset()


# ═══════════════════════════════════════════════════════════════════════════════
#                              DEVICE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_device():
    """Detect the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
#                              COLORS
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    # UI Colors
    CYAN = (255, 220, 100)
    WHITE = (255, 255, 255)
    
    # Segmentation Overlay Colors (BGR)
    SEG_SAFE = (180, 130, 70)
    SEG_LOW = (150, 180, 80)
    SEG_MEDIUM = (80, 180, 200)
    SEG_HIGH = (80, 140, 230)
    SEG_CRITICAL = (80, 80, 230)
    SEG_PEDESTRIAN = (200, 100, 180)
    SEG_ANIMAL = (100, 180, 200)
    
    # Alert Colors
    GREEN = (100, 230, 100)
    YELLOW = (80, 220, 255)
    ORANGE = (80, 180, 255)
    RED = (80, 80, 255)
    
    # Neutrals
    LIGHT_GRAY = (200, 200, 200)
    GRAY = (130, 130, 130)
    DARK_GRAY = (60, 60, 60)
    NEAR_BLACK = (20, 20, 20)
    BLACK = (0, 0, 0)
    
    # Road
    ROAD_GRAY = (50, 50, 55)
    LANE_MARK = (180, 180, 180)
    ROAD_EDGE = (80, 80, 90)


# ═══════════════════════════════════════════════════════════════════════════════
#                              DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class ThreatLevel(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    mask: Optional[np.ndarray]
    class_id: int
    label: str
    confidence: float
    center: Tuple[int, int]
    relative_area: float
    position: str
    distance: float
    threat: ThreatLevel
    track_id: int = -1
    age: int = 0  # How many frames this detection has existed
    smoothed_bbox: Optional[Tuple[float, float, float, float]] = None
    smoothed_mask: Optional[np.ndarray] = None


@dataclass
class LaneData:
    left_points: List[Tuple[int, int]] = field(default_factory=list)
    right_points: List[Tuple[int, int]] = field(default_factory=list)
    center_offset: float = 0.0
    detected: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
#                         KALMAN FILTER TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class KalmanBoxTracker:
    """
    Kalman Filter for tracking bounding boxes.
    State: [x_center, y_center, area, aspect_ratio, vx, vy, va]
    """
    count = 0
    
    def __init__(self, bbox, class_id, mask=None):
        # Initialize Kalman Filter
        self.kf = cv2.KalmanFilter(7, 4)
        
        # State transition matrix
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        
        # Process noise
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.03
        self.kf.processNoiseCov[4:, 4:] *= 0.01  # Lower noise for velocity
        
        # Measurement noise
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # Initialize state
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        area = w * h
        aspect = w / max(h, 1)
        
        self.kf.statePost = np.array([
            [cx], [cy], [area], [aspect], [0], [0], [0]
        ], dtype=np.float32)
        
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.class_id = class_id
        
        # Mask smoothing
        self.mask = mask
        self.smoothed_mask = mask.copy() if mask is not None else None
        
        # History for visualization
        self.history = deque(maxlen=30)
        
        # Threat level smoothing
        self.threat_history = deque(maxlen=Config.THREAT_SMOOTHING)
    
    def predict(self):
        """Predict next state."""
        # Prevent area from going negative
        if self.kf.statePost[2, 0] + self.kf.statePost[6, 0] <= 0:
            self.kf.statePost[6, 0] = 0
        
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        
        return self.get_state()
    
    def update(self, bbox, mask=None, confidence=1.0):
        """Update with new measurement."""
        self.time_since_update = 0
        self.hits += 1
        
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        area = w * h
        aspect = w / max(h, 1)
        
        measurement = np.array([
            [cx], [cy], [area], [aspect]
        ], dtype=np.float32)
        
        self.kf.correct(measurement)
        
        # Smooth mask update
        if mask is not None:
            if self.smoothed_mask is not None:
                alpha = Config.MASK_SMOOTHING
                self.smoothed_mask = (
                    alpha * mask.astype(np.float32) + 
                    (1 - alpha) * self.smoothed_mask.astype(np.float32)
                )
                self.smoothed_mask = (self.smoothed_mask > 0.5).astype(np.uint8)
            else:
                self.smoothed_mask = mask.copy()
            self.mask = mask
        
        # Store center for trail
        self.history.append((int(cx), int(cy)))
    
    def update_threat(self, threat: ThreatLevel):
        """Smooth threat level transitions."""
        self.threat_history.append(threat.value)
    
    def get_smoothed_threat(self) -> ThreatLevel:
        """Get smoothed threat level (mode of recent values)."""
        if not self.threat_history:
            return ThreatLevel.NONE
        
        # Use maximum threat from recent history for safety
        max_threat = max(self.threat_history)
        return ThreatLevel(max_threat)
    
    def get_state(self):
        """Get current bounding box from state."""
        state = self.kf.statePost
        cx = state[0, 0]
        cy = state[1, 0]
        area = max(state[2, 0], 1)
        aspect = max(state[3, 0], 0.1)
        
        w = np.sqrt(area * aspect)
        h = area / max(w, 1)
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return (x1, y1, x2, y2)
    
    def get_smooth_bbox(self):
        """Get integer bounding box."""
        x1, y1, x2, y2 = self.get_state()
        return (int(x1), int(y1), int(x2), int(y2))


# ═══════════════════════════════════════════════════════════════════════════════
#                         ADVANCED TRACKER (SORT-like)
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedTracker:
    """
    Multi-object tracker using Kalman filters and Hungarian algorithm.
    Based on SORT (Simple Online and Realtime Tracking).
    """
    
    def __init__(self, width, height):
        self.trackers: Dict[int, KalmanBoxTracker] = {}
        self.width = width
        self.height = height
        self.frame_count = 0
    
    @staticmethod
    def iou(bb1, bb2):
        """Calculate IoU between two boxes."""
        x1 = max(bb1[0], bb2[0])
        y1 = max(bb1[1], bb2[1])
        x2 = min(bb1[2], bb2[2])
        y2 = min(bb1[3], bb2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
        
        union_area = bb1_area + bb2_area - inter_area
        
        return inter_area / max(union_area, 1e-6)
    
    def update(self, detections: List[Detection]) -> List[Detection]:
        """Update tracker with new detections."""
        self.frame_count += 1
        
        # Predict new positions for existing trackers
        for tracker in self.trackers.values():
            tracker.predict()
        
        # Build cost matrix using IoU
        if detections and self.trackers:
            tracker_ids = list(self.trackers.keys())
            cost_matrix = np.zeros((len(detections), len(tracker_ids)))
            
            for d_idx, det in enumerate(detections):
                for t_idx, tid in enumerate(tracker_ids):
                    tracker = self.trackers[tid]
                    # Only match same class
                    if det.class_id == tracker.class_id:
                        iou_score = self.iou(det.bbox, tracker.get_smooth_bbox())
                        cost_matrix[d_idx, t_idx] = iou_score
                    else:
                        cost_matrix[d_idx, t_idx] = 0
            
            # Hungarian algorithm (greedy approximation for speed)
            matched_det = set()
            matched_tracker = set()
            matches = []
            
            # Sort by IoU score (descending)
            indices = np.unravel_index(
                np.argsort(cost_matrix.ravel())[::-1], 
                cost_matrix.shape
            )
            
            for d_idx, t_idx in zip(indices[0], indices[1]):
                if d_idx in matched_det or t_idx in matched_tracker:
                    continue
                if cost_matrix[d_idx, t_idx] >= Config.TRACK_IOU_THRESHOLD:
                    matches.append((d_idx, t_idx))
                    matched_det.add(d_idx)
                    matched_tracker.add(t_idx)
            
            # Update matched trackers
            for d_idx, t_idx in matches:
                det = detections[d_idx]
                tid = tracker_ids[t_idx]
                tracker = self.trackers[tid]
                tracker.update(det.bbox, det.mask, det.confidence)
                tracker.update_threat(det.threat)
                det.track_id = tid
                det.age = tracker.hits
                det.smoothed_bbox = tracker.get_state()
                det.smoothed_mask = tracker.smoothed_mask
            
            # Create new trackers for unmatched detections
            for d_idx, det in enumerate(detections):
                if d_idx not in matched_det:
                    tracker = KalmanBoxTracker(det.bbox, det.class_id, det.mask)
                    tracker.update_threat(det.threat)
                    self.trackers[tracker.id] = tracker
                    det.track_id = tracker.id
                    det.age = 1
        
        elif detections:
            # No existing trackers, create new ones
            for det in detections:
                tracker = KalmanBoxTracker(det.bbox, det.class_id, det.mask)
                tracker.update_threat(det.threat)
                self.trackers[tracker.id] = tracker
                det.track_id = tracker.id
                det.age = 1
        
        # Remove old trackers
        to_remove = []
        for tid, tracker in self.trackers.items():
            if tracker.time_since_update > Config.TRACK_MAX_AGE:
                to_remove.append(tid)
        
        for tid in to_remove:
            del self.trackers[tid]
        
        # Generate detections from active trackers (including those not matched this frame)
        output_detections = []
        
        for det in detections:
            if det.track_id in self.trackers:
                tracker = self.trackers[det.track_id]
                if tracker.hits >= Config.TRACK_MIN_HITS:
                    # Use smoothed values
                    if det.smoothed_bbox:
                        x1, y1, x2, y2 = det.smoothed_bbox
                        det.bbox = (int(x1), int(y1), int(x2), int(y2))
                        det.center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    if det.smoothed_mask is not None:
                        # Ensure mask dimensions match
                        mask = det.smoothed_mask
                        if mask.shape[0] != self.height or mask.shape[1] != self.width:
                            mask = cv2.resize(
                                mask.astype(np.float32), 
                                (self.width, self.height)
                            )
                            mask = (mask > 0.5).astype(np.uint8)
                        det.mask = mask
                    det.threat = tracker.get_smoothed_threat()
                    output_detections.append(det)
        
        # Add ghost detections for recently lost tracks (prevents flickering)
        for tid, tracker in self.trackers.items():
            if 0 < tracker.time_since_update <= 3 and tracker.hits >= Config.TRACK_MIN_HITS:
                # Create ghost detection from predicted position
                x1, y1, x2, y2 = tracker.get_smooth_bbox()
                
                # Bounds check
                x1 = max(0, min(self.width - 1, x1))
                x2 = max(0, min(self.width, x2))
                y1 = max(0, min(self.height - 1, y1))
                y2 = max(0, min(self.height, y2))
                
                if x2 > x1 and y2 > y1:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    w, h = x2 - x1, y2 - y1
                    
                    # Ensure mask dimensions match expected size
                    ghost_mask = tracker.smoothed_mask
                    if ghost_mask is not None:
                        if ghost_mask.shape[0] != self.height or ghost_mask.shape[1] != self.width:
                            ghost_mask = cv2.resize(
                                ghost_mask.astype(np.float32), 
                                (self.width, self.height)
                            )
                            ghost_mask = (ghost_mask > 0.5).astype(np.uint8)
                    
                    ghost = Detection(
                        bbox=(x1, y1, x2, y2),
                        mask=ghost_mask,
                        class_id=tracker.class_id,
                        label=self._get_label(tracker.class_id),
                        confidence=0.5,  # Lower confidence for ghost
                        center=(cx, cy),
                        relative_area=(w * h) / (self.width * self.height),
                        position=self._get_position(cx),
                        distance=self._estimate_distance(w, tracker.class_id),
                        threat=tracker.get_smoothed_threat(),
                        track_id=tid,
                        age=tracker.hits
                    )
                    output_detections.append(ghost)
        
        return output_detections
    
    def _get_label(self, class_id):
        labels = {
            0: "PEDESTRIAN", 1: "BICYCLE", 2: "CAR", 3: "MOTORCYCLE",
            5: "BUS", 7: "TRUCK", 16: "DOG", 17: "HORSE", 18: "SHEEP", 19: "COW"
        }
        return labels.get(class_id, "OBJECT")
    
    def _get_position(self, cx):
        center_left = self.width * (0.5 - Config.CENTER_ZONE / 2)
        center_right = self.width * (0.5 + Config.CENTER_ZONE / 2)
        if cx < center_left:
            return "LEFT"
        elif cx > center_right:
            return "RIGHT"
        return "CENTER"
    
    def _estimate_distance(self, width, class_id):
        widths = {
            0: 0.5, 1: 0.6, 2: 1.8, 3: 0.8, 5: 2.5, 7: 2.5,
            16: 0.3, 17: 0.6, 18: 0.4, 19: 0.8
        }
        real_w = widths.get(class_id, 1.0)
        return (real_w * 800) / max(width, 1)
    
    def get_track_history(self, track_id):
        """Get motion history for a track."""
        if track_id in self.trackers:
            return list(self.trackers[track_id].history)
        return []


# ═══════════════════════════════════════════════════════════════════════════════
#                         ASYNC DETECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class AsyncDetectionEngine:
    """
    Asynchronous detection engine that runs YOLO in a separate thread.
    Provides non-blocking detection with frame interpolation.
    """
    
    def __init__(self, width, height):
        print("  → Loading YOLO Segmentation model...")
        
        self.device = get_device()
        device_names = {
            "mps": "Apple Silicon GPU (MPS)",
            "cuda": "NVIDIA GPU (CUDA)",
            "cpu": "CPU"
        }
        print(f"  → Using {device_names.get(self.device, self.device)}")
        
        self.model = YOLO(Config.YOLO_MODEL)
        
        # Warm up model
        print("  → Warming up model...")
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        self.model(dummy, verbose=False, device=self.device)
        
        self.width = width
        self.height = height
        self.area = width * height
        
        # Advanced tracker
        self.tracker = AdvancedTracker(width, height)
        
        # Async processing
        self.input_queue = Queue(maxsize=2)
        self.output_queue = Queue(maxsize=2)
        self.running = True
        self.lock = Lock()
        
        self.latest_detections = []
        self.frame_count = 0
        
        self.labels = {
            0: "PEDESTRIAN", 1: "BICYCLE", 2: "CAR", 3: "MOTORCYCLE",
            5: "BUS", 7: "TRUCK", 16: "DOG", 17: "HORSE", 18: "SHEEP", 19: "COW"
        }
        
        self.widths = {
            0: 0.5, 1: 0.6, 2: 1.8, 3: 0.8, 5: 2.5, 7: 2.5,
            16: 0.3, 17: 0.6, 18: 0.4, 19: 0.8
        }
        
        # Start worker thread
        if Config.ASYNC_PROCESSING:
            self.worker = Thread(target=self._detection_worker, daemon=True)
            self.worker.start()
            print("  → Async processing enabled")
    
    def _detection_worker(self):
        """Background worker for detection."""
        while self.running:
            try:
                frame, frame_id = self.input_queue.get(timeout=0.1)
                detections = self._run_detection(frame)
                self.output_queue.put((detections, frame_id))
            except:
                continue
    
    def _run_detection(self, frame):
        """Run YOLO detection on frame."""
        results = self.model(
            frame,
            verbose=False,
            conf=Config.CONFIDENCE_THRESHOLD,
            iou=Config.IOU_THRESHOLD,
            classes=Config.ALL_CLASSES,
            device=self.device,
            retina_masks=True
        )
        
        detections = []
        
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            
            masks = None
            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()
            
            for idx, box in enumerate(r.boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Bounds check
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(self.width, x2)
                y2 = min(self.height, y2)
                
                mask = None
                if masks is not None and idx < len(masks):
                    mask = masks[idx]
                    mask = cv2.resize(mask, (self.width, self.height))
                    mask = (mask > 0.5).astype(np.uint8)
                
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                w, h = x2 - x1, y2 - y1
                rel_area = (w * h) / self.area
                
                # Position
                center_left = self.width * (0.5 - Config.CENTER_ZONE / 2)
                center_right = self.width * (0.5 + Config.CENTER_ZONE / 2)
                
                if cx < center_left:
                    pos = "LEFT"
                elif cx > center_right:
                    pos = "RIGHT"
                else:
                    pos = "CENTER"
                
                # Distance
                real_w = self.widths.get(cls, 1.0)
                dist = (real_w * 800) / w if w > 0 else 99
                
                # Threat level
                threat = ThreatLevel.NONE
                if pos == "CENTER":
                    if rel_area > Config.CRITICAL_AREA:
                        threat = ThreatLevel.CRITICAL
                    elif rel_area > Config.HIGH_AREA:
                        threat = ThreatLevel.HIGH
                    elif rel_area > Config.MEDIUM_AREA:
                        threat = ThreatLevel.MEDIUM
                    elif rel_area > 0.01:
                        threat = ThreatLevel.LOW
                elif rel_area > Config.HIGH_AREA:
                    threat = ThreatLevel.MEDIUM
                
                # Pedestrian/animal boost
                if cls in [0, 16, 17, 18, 19] and pos == "CENTER" and rel_area > 0.015:
                    threat = ThreatLevel(min(threat.value + 1, 4))
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    mask=mask,
                    class_id=cls,
                    label=self.labels.get(cls, "OBJECT"),
                    confidence=conf,
                    center=(cx, cy),
                    relative_area=rel_area,
                    position=pos,
                    distance=round(dist, 1),
                    threat=threat
                ))
        
        return detections
    
    def detect(self, frame):
        """Get detections for frame (async or sync)."""
        self.frame_count += 1
        
        # Skip frames for performance
        should_detect = (self.frame_count % Config.SKIP_FRAMES == 0)
        
        if Config.ASYNC_PROCESSING:
            # Submit frame for processing
            if should_detect and not self.input_queue.full():
                self.input_queue.put((frame.copy(), self.frame_count))
            
            # Get latest results
            while not self.output_queue.empty():
                detections, _ = self.output_queue.get()
                with self.lock:
                    self.latest_detections = detections
            
            with self.lock:
                detections = self.latest_detections
        else:
            # Synchronous processing
            if should_detect:
                detections = self._run_detection(frame)
                self.latest_detections = detections
            else:
                detections = self.latest_detections
        
        # Always run tracker for smooth updates
        return self.tracker.update(detections)
    
    def get_track_history(self, track_id):
        """Get motion trail for track."""
        return self.tracker.get_track_history(track_id)
    
    def stop(self):
        """Stop async worker."""
        self.running = False


# ═══════════════════════════════════════════════════════════════════════════════
#                              LANE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class LaneDetector:
    """Smoothed lane detection with temporal filtering."""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.left_history = deque(maxlen=12)  # Increased for smoother lines
        self.right_history = deque(maxlen=12)
        self.smooth_left = None
        self.smooth_right = None
    
    def detect(self, frame):
        h, w = frame.shape[:2]
        
        roi_top = int(h * 0.55)
        roi = frame[roi_top:h, :]
        
        # Preprocessing
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding for better edge detection
        edges = cv2.Canny(blur, 50, 150)
        
        # ROI mask
        mask = np.zeros_like(edges)
        pts = np.array([
            [0, edges.shape[0]],
            [w * 0.1, 0],
            [w * 0.9, 0],
            [w, edges.shape[0]]
        ], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        masked = cv2.bitwise_and(edges, mask)
        
        # Hough lines
        lines = cv2.HoughLinesP(masked, 1, np.pi/180, 40, 
                               minLineLength=40, maxLineGap=100)
        
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.3:
                    continue
                
                y1 += roi_top
                y2 += roi_top
                
                if slope < 0:
                    left_lines.append((x1, y1, x2, y2))
                else:
                    right_lines.append((x1, y1, x2, y2))
        
        lane_data = LaneData()
        
        # Smooth left lane
        if left_lines:
            avg = np.mean(left_lines, axis=0)
            self.left_history.append(avg)
        
        if self.left_history:
            avg = np.mean(self.left_history, axis=0)
            if self.smooth_left is None:
                self.smooth_left = avg
            else:
                alpha = 0.3
                self.smooth_left = alpha * avg + (1 - alpha) * self.smooth_left
            
            pts = self.smooth_left.astype(int)
            lane_data.left_points = [(pts[0], pts[1]), (pts[2], pts[3])]
        
        # Smooth right lane
        if right_lines:
            avg = np.mean(right_lines, axis=0)
            self.right_history.append(avg)
        
        if self.right_history:
            avg = np.mean(self.right_history, axis=0)
            if self.smooth_right is None:
                self.smooth_right = avg
            else:
                alpha = 0.3
                self.smooth_right = alpha * avg + (1 - alpha) * self.smooth_right
            
            pts = self.smooth_right.astype(int)
            lane_data.right_points = [(pts[0], pts[1]), (pts[2], pts[3])]
        
        lane_data.detected = bool(lane_data.left_points or lane_data.right_points)
        
        # Center offset
        if lane_data.left_points and lane_data.right_points:
            left_x = (lane_data.left_points[0][0] + lane_data.left_points[1][0]) / 2
            right_x = (lane_data.right_points[0][0] + lane_data.right_points[1][0]) / 2
            center = (left_x + right_x) / 2
            lane_data.center_offset = (w / 2 - center) / (w / 2)
        
        return lane_data


# ═══════════════════════════════════════════════════════════════════════════════
#                           OPTIMIZED HUD RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

class OptimizedHUD:
    """High-performance HUD with anti-aliasing and smooth animations."""
    
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.frame_num = 0
        
        # Instruction state with smoothing
        self.main_text = "READY"
        self.sub_text = "System Online"
        self.alert_color = Colors.CYAN
        self.target_main = "READY"
        self.target_sub = "System Online"
        self.target_color = Colors.CYAN
        self.transition_progress = 1.0
        
        # Precomputed overlays for performance
        self._create_static_overlays()
        
        # Animation state
        self.pulse_phase = 0
    
    def _create_static_overlays(self):
        """Pre-render static overlay elements."""
        # Road perspective overlay
        self.road_overlay = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self._draw_static_road(self.road_overlay)
        
        # Top bar dimensions
        self.top_bar_height = 50
        self.top_bar = np.zeros((self.top_bar_height, self.w, 3), dtype=np.uint8)
        for y in range(self.top_bar_height):
            alpha = 1 - (y / self.top_bar_height) ** 2
            self.top_bar[y, :] = [int(c * alpha * 0.85) for c in Colors.NEAR_BLACK]
        
        # Bottom panel dimensions
        self.bottom_panel_height = 85
        self.bottom_panel = np.zeros((self.bottom_panel_height, self.w, 3), dtype=np.uint8)
        for y in range(self.bottom_panel_height):
            progress = y / self.bottom_panel_height
            alpha = progress ** 1.5
            self.bottom_panel[y, :] = [int(c * alpha * 0.85) for c in Colors.NEAR_BLACK]
    
    def _draw_static_road(self, overlay):
        """Draw static road elements."""
        h, w = overlay.shape[:2]
        road_h = int(h * 0.25)
        road_top = h - road_h
        
        vp_x = w // 2
        vp_y = road_top
        left_bottom = int(w * 0.08)
        right_bottom = int(w * 0.92)
        
        # Road surface
        road_pts = np.array([
            [vp_x, vp_y],
            [right_bottom, h],
            [left_bottom, h]
        ], np.int32)
        cv2.fillPoly(overlay, [road_pts], Colors.ROAD_GRAY)
    
    def _get_segment_color(self, detection):
        """Get overlay color based on object type and threat."""
        if detection.class_id == 0:
            return Colors.SEG_PEDESTRIAN
        elif detection.class_id in [16, 17, 18, 19]:
            return Colors.SEG_ANIMAL
        else:
            threat_colors = {
                ThreatLevel.NONE: Colors.SEG_SAFE,
                ThreatLevel.LOW: Colors.SEG_LOW,
                ThreatLevel.MEDIUM: Colors.SEG_MEDIUM,
                ThreatLevel.HIGH: Colors.SEG_HIGH,
                ThreatLevel.CRITICAL: Colors.SEG_CRITICAL
            }
            return threat_colors.get(detection.threat, Colors.SEG_SAFE)
    
    def _threat_color(self, threat):
        """Get alert color for threat level."""
        return {
            ThreatLevel.NONE: Colors.CYAN,
            ThreatLevel.LOW: Colors.GREEN,
            ThreatLevel.MEDIUM: Colors.YELLOW,
            ThreatLevel.HIGH: Colors.ORANGE,
            ThreatLevel.CRITICAL: Colors.RED
        }.get(threat, Colors.CYAN)
    
    def _draw_segmentation_overlays(self, frame, detections):
        """Draw transparent segmentation masks with smooth edges."""
        if not detections:
            return frame
        
        overlay = frame.copy()
        edge_overlay = np.zeros_like(frame)
        h, w = frame.shape[:2]
        
        for det in detections:
            if det.mask is None:
                continue
            
            # Ensure mask matches frame dimensions
            mask = det.mask
            if mask.shape[0] != h or mask.shape[1] != w:
                mask = cv2.resize(mask.astype(np.float32), (w, h))
                mask = (mask > 0.5).astype(np.uint8)
            
            color = self._get_segment_color(det)
            
            # Apply Gaussian blur to mask for smoother edges
            if Config.ANTI_ALIAS:
                smooth_mask = cv2.GaussianBlur(
                    mask.astype(np.float32), (5, 5), 0
                )
            else:
                smooth_mask = mask.astype(np.float32)
            
            # Create colored overlay
            for c in range(3):
                overlay[:, :, c] = (
                    overlay[:, :, c] * (1 - smooth_mask * Config.MASK_ALPHA) +
                    color[c] * smooth_mask * Config.MASK_ALPHA
                ).astype(np.uint8)
            
            # Edge glow
            if Config.EDGE_GLOW:
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    cv2.drawContours(edge_overlay, contours, -1, color, 4)
                    cv2.drawContours(edge_overlay, contours, -1, Colors.WHITE, 1)
        
        # Blend edge glow
        if Config.EDGE_GLOW:
            # Blur edges for glow effect
            edge_overlay = cv2.GaussianBlur(edge_overlay, (3, 3), 0)
            mask = np.any(edge_overlay > 0, axis=2)
            overlay[mask] = cv2.addWeighted(
                edge_overlay, 0.5, overlay, 0.5, 0
            )[mask]
        
        return overlay
    
    def _draw_object_labels(self, frame, detections, detector=None):
        """Draw minimal floating labels with motion trails."""
        for det in detections:
            if det.relative_area < 0.008 and det.threat == ThreatLevel.NONE:
                continue
            
            x1, y1, x2, y2 = det.bbox
            cx, cy = det.center
            color = self._get_segment_color(det)
            
            # Motion trail (if tracker available)
            if detector and det.track_id >= 0:
                history = detector.get_track_history(det.track_id)
                if len(history) > 1:
                    pts = np.array(history, dtype=np.int32)
                    # Fading trail
                    for i in range(1, len(pts)):
                        alpha = i / len(pts)
                        pt_color = tuple(int(c * alpha) for c in color)
                        cv2.line(frame, tuple(pts[i-1]), tuple(pts[i]), 
                                pt_color, max(1, int(alpha * 2)))
            
            # Label
            label = f"{det.label}"
            dist_label = f"{det.distance:.0f}m"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            
            label_size = cv2.getTextSize(label, font, scale, 1)[0]
            dist_size = cv2.getTextSize(dist_label, font, scale, 1)[0]
            total_width = label_size[0] + dist_size[0] + 15
            
            text_x = cx - total_width // 2
            text_y = y1 - 12
            if text_y < 60:
                text_y = y2 + 25
            
            # Background pill
            padding = 6
            pill_x1 = text_x - padding
            pill_y1 = text_y - label_size[1] - padding
            pill_x2 = text_x + total_width + padding
            pill_y2 = text_y + padding
            
            # Rounded rectangle
            cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2),
                         Colors.NEAR_BLACK, -1)
            cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2),
                         color, 1)
            
            # Text
            cv2.putText(frame, label, (text_x, text_y), font, scale,
                       color, 1, cv2.LINE_AA)
            cv2.putText(frame, dist_label, (text_x + label_size[0] + 10, text_y),
                       font, scale, Colors.LIGHT_GRAY, 1, cv2.LINE_AA)
            
            # Connecting line
            line_start_y = pill_y2 if text_y < cy else pill_y1
            cv2.line(frame, (cx, line_start_y), (cx, cy), color, 1)
            cv2.circle(frame, (cx, cy), 3, color, -1)
            
            # Track ID indicator (small)
            if det.track_id >= 0 and det.age > 3:
                id_text = f"#{det.track_id}"
                cv2.putText(frame, id_text, (pill_x2 + 3, text_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, Colors.GRAY, 
                           1, cv2.LINE_AA)
    
    def _draw_road_overlay(self, frame):
        """Draw road perspective with animated center lines."""
        h, w = frame.shape[:2]
        road_h = int(h * 0.25)
        road_top = h - road_h
        vp_x = w // 2
        vp_y = road_top
        
        # Blend road overlay (ensure dimensions match)
        if self.road_overlay.shape[:2] == frame.shape[:2]:
            road_mask = self.road_overlay > 0
            frame_blend = cv2.addWeighted(
                self.road_overlay, 0.25, frame, 0.75, 0
            )
            frame[road_mask] = frame_blend[road_mask]
        else:
            # Recreate road overlay if dimensions don't match
            self.road_overlay = np.zeros((h, w, 3), dtype=np.uint8)
            self._draw_static_road(self.road_overlay)
            road_mask = self.road_overlay > 0
            frame_blend = cv2.addWeighted(
                self.road_overlay, 0.25, frame, 0.75, 0
            )
            frame[road_mask] = frame_blend[road_mask]
        
        # Road edges
        left_bottom = int(w * 0.08)
        right_bottom = int(w * 0.92)
        cv2.line(frame, (vp_x, vp_y), (left_bottom, h), Colors.ROAD_EDGE, 2)
        cv2.line(frame, (vp_x, vp_y), (right_bottom, h), Colors.ROAD_EDGE, 2)
        
        # Animated center dashes
        num_dashes = 6
        offset = (self.frame_num * 0.02) % 1  # Animation offset
        
        for i in range(num_dashes + 1):
            p1 = (i - offset) / num_dashes
            p2 = (i + 0.4 - offset) / num_dashes
            
            if p1 < 0 or p2 > 1:
                continue
            
            y1 = int(vp_y + (h - vp_y) * p1)
            y2 = int(vp_y + (h - vp_y) * p2)
            
            thickness = max(1, int(2 + p1 * 2))
            alpha = 0.2 + p1 * 0.4
            
            # Draw with alpha blending
            overlay = frame.copy()
            cv2.line(overlay, (vp_x, y1), (vp_x, y2), Colors.LANE_MARK, thickness)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def _draw_distance_arcs(self, frame, detections):
        """Draw distance arcs on road."""
        h, w = frame.shape[:2]
        road_top = h - int(h * 0.25)
        
        for det in detections:
            if det.position != "CENTER" or det.threat == ThreatLevel.NONE:
                continue
            
            _, cy = det.center
            progress = (cy - road_top) / (h - road_top)
            progress = max(0.1, min(1, progress))
            
            arc_y = int(road_top + (h - road_top) * progress * 0.7)
            arc_width = int(40 + 120 * progress)
            
            color = self._threat_color(det.threat)
            
            cv2.ellipse(frame, (w // 2, arc_y), (arc_width, 12), 
                       0, 0, 180, color, 2)
            
            dist_text = f"{det.distance:.0f}m"
            text_size = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.45, 1)[0]
            text_x = w // 2 - text_size[0] // 2
            
            cv2.rectangle(frame, (text_x - 4, arc_y - 18),
                         (text_x + text_size[0] + 4, arc_y - 4), 
                         Colors.NEAR_BLACK, -1)
            cv2.putText(frame, dist_text, (text_x, arc_y - 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    
    def _draw_lane_overlay(self, frame, lane_data):
        """Draw detected lanes."""
        if not lane_data.detected:
            return frame
        
        overlay = frame.copy()
        
        if lane_data.left_points:
            cv2.line(overlay, lane_data.left_points[0], 
                    lane_data.left_points[1], Colors.GREEN, 4)
        
        if lane_data.right_points:
            cv2.line(overlay, lane_data.right_points[0],
                    lane_data.right_points[1], Colors.GREEN, 4)
        
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        return frame
    
    def _draw_top_bar(self, frame, fps, num_objects):
        """Draw top status bar."""
        bar_h = min(self.top_bar_height, frame.shape[0])
        
        # Blend pre-rendered gradient (with size safety)
        if bar_h > 0:
            overlay_region = self.top_bar[:bar_h, :frame.shape[1]]
            frame_region = frame[:bar_h, :]
            if overlay_region.shape == frame_region.shape:
                frame[:bar_h, :] = cv2.addWeighted(
                    overlay_region, 1.0, frame_region, 0.15, 0
                )
        
        # Bottom line
        cv2.line(frame, (0, 45), (self.w, 45), Colors.CYAN, 1)
        
        # Time
        time_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, time_str, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, Colors.WHITE, 1, cv2.LINE_AA)
        
        # Title
        title = "AI DRIVING ASSISTANT"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        title_x = (self.w - title_size[0]) // 2
        cv2.putText(frame, title, (title_x, 28), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, Colors.CYAN, 1, cv2.LINE_AA)
        
        # Objects
        cv2.putText(frame, f"TRACKING: {num_objects}", (self.w - 180, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.GRAY, 1, cv2.LINE_AA)
        
        # FPS
        fps_color = Colors.GREEN if fps > 20 else Colors.YELLOW if fps > 10 else Colors.RED
        cv2.putText(frame, f"{int(fps)} FPS", (self.w - 70, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1, cv2.LINE_AA)
    
    def _draw_bottom_panel(self, frame):
        """Draw instruction panel with smooth transitions."""
        panel_h = self.bottom_panel_height
        panel_top = self.h - panel_h
        
        # Safety check for panel dimensions
        if panel_top < 0:
            panel_top = 0
            panel_h = self.h
        
        # Get actual slice dimensions
        frame_slice = frame[panel_top:, :]
        actual_h = frame_slice.shape[0]
        actual_w = frame_slice.shape[1]
        
        # Blend pre-rendered gradient (with size safety)
        if actual_h > 0 and actual_w > 0:
            # Resize bottom_panel if needed to match frame slice
            if self.bottom_panel.shape[0] != actual_h or self.bottom_panel.shape[1] != actual_w:
                overlay_region = cv2.resize(self.bottom_panel, (actual_w, actual_h))
            else:
                overlay_region = self.bottom_panel
            
            frame[panel_top:, :] = cv2.addWeighted(
                overlay_region, 1.0, frame_slice, 0.15, 0
            )
        
        # Top line with accents
        cv2.line(frame, (0, panel_top), (self.w, panel_top), Colors.DARK_GRAY, 1)
        cv2.line(frame, (0, panel_top), (80, panel_top), self.alert_color, 2)
        cv2.line(frame, (self.w - 80, panel_top), (self.w, panel_top), 
                self.alert_color, 2)
        
        # Main instruction
        font = cv2.FONT_HERSHEY_DUPLEX
        main_size = cv2.getTextSize(self.main_text, font, 1.2, 2)[0]
        main_x = (self.w - main_size[0]) // 2
        main_y = panel_top + 42
        
        # Ensure text is within frame bounds
        if main_y < self.h:
            # Shadow + text
            cv2.putText(frame, self.main_text, (main_x + 2, main_y + 2), font,
                       1.2, Colors.BLACK, 2, cv2.LINE_AA)
            cv2.putText(frame, self.main_text, (main_x, main_y), font,
                       1.2, self.alert_color, 2, cv2.LINE_AA)
        
        # Sub text
        sub_y = panel_top + 68
        if sub_y < self.h:
            sub_size = cv2.getTextSize(self.sub_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.5, 1)[0]
            sub_x = (self.w - sub_size[0]) // 2
            cv2.putText(frame, self.sub_text, (sub_x, sub_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.LIGHT_GRAY, 
                       1, cv2.LINE_AA)
    
    def _draw_side_indicators(self, frame, detections):
        """Draw side traffic indicators."""
        left_count = sum(1 for d in detections if d.position == "LEFT")
        right_count = sum(1 for d in detections if d.position == "RIGHT")
        
        y = self.h // 2
        
        if left_count > 0:
            pts = np.array([[15, y], [35, y - 18], [35, y + 18]], np.int32)
            cv2.fillPoly(frame, [pts], Colors.YELLOW)
            cv2.putText(frame, str(left_count), (40, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.YELLOW, 
                       1, cv2.LINE_AA)
        
        if right_count > 0:
            pts = np.array([[self.w - 15, y], [self.w - 35, y - 18],
                           [self.w - 35, y + 18]], np.int32)
            cv2.fillPoly(frame, [pts], Colors.YELLOW)
            cv2.putText(frame, str(right_count), (self.w - 55, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.YELLOW,
                       1, cv2.LINE_AA)
    
    def _draw_mini_map(self, frame, detections):
        """Draw mini radar map."""
        map_w, map_h = 100, 80
        map_x = self.w - map_w - 15
        map_y = 55
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (map_x, map_y), (map_x + map_w, map_y + map_h),
                     Colors.NEAR_BLACK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (map_x, map_y), (map_x + map_w, map_y + map_h),
                     Colors.DARK_GRAY, 1)
        
        # Road shape
        road_pts = np.array([
            [map_x + map_w // 2, map_y + 8],
            [map_x + map_w - 15, map_y + map_h - 8],
            [map_x + 15, map_y + map_h - 8]
        ], np.int32)
        cv2.polylines(frame, [road_pts], True, Colors.ROAD_EDGE, 1)
        
        # Self indicator
        car_x = map_x + map_w // 2
        car_y = map_y + map_h - 15
        car_pts = np.array([[car_x, car_y - 6], [car_x - 4, car_y + 3],
                           [car_x + 4, car_y + 3]], np.int32)
        cv2.fillPoly(frame, [car_pts], Colors.CYAN)
        
        # Detection dots
        for det in detections:
            rel_x = (det.center[0] / self.w - 0.5) * 0.7
            rel_y = 1 - (det.center[1] / self.h)
            
            dot_x = int(map_x + map_w // 2 + rel_x * map_w)
            dot_y = int(map_y + map_h - 12 - rel_y * (map_h - 25))
            
            dot_x = max(map_x + 4, min(map_x + map_w - 4, dot_x))
            dot_y = max(map_y + 4, min(map_y + map_h - 4, dot_y))
            
            color = self._threat_color(det.threat)
            cv2.circle(frame, (dot_x, dot_y), 3, color, -1)
    
    def _draw_center_reticle(self, frame):
        """Draw center crosshair."""
        cx, cy = self.w // 2, self.h // 2 - 30
        size = 18
        gap = 5
        
        cv2.line(frame, (cx - size, cy), (cx - gap, cy), Colors.CYAN, 1)
        cv2.line(frame, (cx + gap, cy), (cx + size, cy), Colors.CYAN, 1)
        cv2.line(frame, (cx, cy - size), (cx, cy - gap), Colors.CYAN, 1)
        cv2.line(frame, (cx, cy + gap), (cx, cy + size), Colors.CYAN, 1)
        cv2.circle(frame, (cx, cy), 2, Colors.CYAN, -1)
    
    def _draw_threat_indicator(self, frame, max_threat):
        """Draw overall threat level with pulse effect."""
        if max_threat == ThreatLevel.NONE:
            return
        
        ind_x = 20
        ind_y = 60
        
        color = self._threat_color(max_threat)
        label = {
            ThreatLevel.LOW: "LOW",
            ThreatLevel.MEDIUM: "MEDIUM",
            ThreatLevel.HIGH: "HIGH",
            ThreatLevel.CRITICAL: "CRITICAL"
        }.get(max_threat, "")
        
        # Pulse effect for critical
        if max_threat == ThreatLevel.CRITICAL:
            pulse = abs(math.sin(self.pulse_phase))
            if pulse > 0.5:
                cv2.rectangle(frame, (ind_x - 5, ind_y - 18),
                             (ind_x + 85, ind_y + 8), Colors.RED, 2)
        
        # Background
        cv2.rectangle(frame, (ind_x, ind_y - 15), (ind_x + 80, ind_y + 5),
                     Colors.NEAR_BLACK, -1)
        cv2.rectangle(frame, (ind_x, ind_y - 15), (ind_x + 80, ind_y + 5),
                     color, 1)
        
        # Indicator circle
        cv2.circle(frame, (ind_x + 12, ind_y - 5), 5, color, -1)
        
        # Label
        cv2.putText(frame, label, (ind_x + 22, ind_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    
    def update_instruction(self, main, sub, color, threat):
        """Update instruction with smooth transition."""
        # Immediate update for critical threats
        if threat == ThreatLevel.CRITICAL:
            self.main_text = main
            self.sub_text = sub
            self.alert_color = color
            return
        
        # Smooth transition for other states
        if main != self.target_main:
            self.target_main = main
            self.target_sub = sub
            self.target_color = color
            self.transition_progress = 0
        
        # Progress transition
        if self.transition_progress < 1:
            self.transition_progress += 0.15
            if self.transition_progress >= 1:
                self.main_text = self.target_main
                self.sub_text = self.target_sub
                self.alert_color = self.target_color
    
    def render(self, frame, detections, lane_data, fps, detector=None):
        """Render complete HUD."""
        self.frame_num += 1
        self.pulse_phase += 0.15
        
        # 1. Lane overlay
        frame = self._draw_lane_overlay(frame, lane_data)
        
        # 2. Segmentation masks
        frame = self._draw_segmentation_overlays(frame, detections)
        
        # 3. Road perspective
        frame = self._draw_road_overlay(frame)
        
        # 4. Distance arcs
        self._draw_distance_arcs(frame, detections)
        
        # 5. Object labels with trails
        self._draw_object_labels(frame, detections, detector)
        
        # 6. Center reticle
        self._draw_center_reticle(frame)
        
        # 7. Side indicators
        self._draw_side_indicators(frame, detections)
        
        # 8. Mini map
        self._draw_mini_map(frame, detections)
        
        # 9. Threat indicator
        max_threat = max((d.threat for d in detections), 
                        default=ThreatLevel.NONE, key=lambda t: t.value)
        self._draw_threat_indicator(frame, max_threat)
        
        # 10. Top bar
        self._draw_top_bar(frame, fps, len(detections))
        
        # 11. Bottom panel
        self._draw_bottom_panel(frame)
        
        return frame


# ═══════════════════════════════════════════════════════════════════════════════
#                              DRIVING LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

class DrivingLogic:
    """Driving instruction generator with temporal smoothing."""
    
    def __init__(self):
        self.instruction_history = deque(maxlen=10)
        self.last_instruction = ("CLEAR", "Road ahead is clear", Colors.CYAN, 
                                ThreatLevel.NONE)
    
    def analyze(self, detections, lane_data):
        """Analyze scene and generate driving instruction."""
        critical = [d for d in detections if d.threat == ThreatLevel.CRITICAL]
        high = [d for d in detections if d.threat == ThreatLevel.HIGH]
        center = [d for d in detections 
                 if d.position == "CENTER" and d.threat.value >= 2]
        
        max_threat = max((d.threat for d in detections), 
                        default=ThreatLevel.NONE, key=lambda t: t.value)
        
        instruction = None
        
        if critical:
            d = critical[0]
            if d.class_id in [0, 16, 17, 18, 19]:
                instruction = ("⚠ STOP", f"{d.label} on road!", 
                             Colors.RED, max_threat)
            else:
                instruction = ("⚠ BRAKE", 
                             f"Collision warning - {d.distance:.0f}m",
                             Colors.RED, max_threat)
        
        elif high:
            d = high[0]
            instruction = ("SLOW DOWN", f"{d.label} ahead - {d.distance:.0f}m",
                          Colors.ORANGE, max_threat)
        
        elif center:
            d = center[0]
            left_clear = not any(det.position == "LEFT" for det in detections)
            right_clear = not any(det.position == "RIGHT" for det in detections)
            
            if right_clear:
                instruction = ("PASS RIGHT", f"Clear to overtake {d.label}",
                             Colors.GREEN, max_threat)
            elif left_clear:
                instruction = ("PASS LEFT", f"Clear to overtake {d.label}",
                             Colors.GREEN, max_threat)
            else:
                instruction = ("HOLD", "Wait for opening", 
                             Colors.YELLOW, max_threat)
        
        elif lane_data.detected and abs(lane_data.center_offset) > 0.3:
            if lane_data.center_offset > 0:
                instruction = ("DRIFT LEFT", "Steer right to correct",
                             Colors.YELLOW, max_threat)
            else:
                instruction = ("DRIFT RIGHT", "Steer left to correct",
                             Colors.YELLOW, max_threat)
        
        else:
            instruction = ("CLEAR", "Road ahead is clear", 
                          Colors.CYAN, max_threat)
        
        # Temporal smoothing for non-critical instructions
        if instruction[3] != ThreatLevel.CRITICAL:
            self.instruction_history.append(instruction)
            
            # Only change instruction if it's been consistent
            if len(self.instruction_history) >= 3:
                recent = list(self.instruction_history)[-3:]
                if all(r[0] == recent[0][0] for r in recent):
                    self.last_instruction = instruction
                # else keep previous instruction
            
            return self.last_instruction
        
        return instruction


# ═══════════════════════════════════════════════════════════════════════════════
#                              MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class DrivingAssistant:
    """Main application with all optimizations."""
    
    def __init__(self):
        print("\n" + "═" * 60)
        print("   OPTIMIZED AI DRIVING ASSISTANT v6.0")
        print("   Kalman Tracking • Temporal Smoothing • Async Processing")
        print("═" * 60 + "\n")
        
        print(f"   Quality Preset: {Config.QUALITY_PRESET.value.upper()}")
        print(f"   Model: {Config.YOLO_MODEL}")
        print(f"   Skip Frames: {Config.SKIP_FRAMES}")
        print(f"   Async: {Config.ASYNC_PROCESSING}")
        print()
        
        # Video source
        print("[1/4] Opening video source...")
        if Config.USE_TEST_VIDEO:
            source = Config.TEST_VIDEO_PATH
            print(f"      Mode: Test Video")
            print(f"      File: {source}")
        else:
            source = Config.LIVE_CAMERA_INDEX
            print(f"      Mode: Live Camera ({source})")
        
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open: {source}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        
        print(f"      Resolution: {self.width}x{self.height} @ {self.video_fps:.0f}fps")
        
        # Components
        print("\n[2/4] Loading detection engine...")
        self.detector = AsyncDetectionEngine(self.width, self.height)
        
        print("\n[3/4] Initializing lane detection...")
        self.lane_detector = LaneDetector(self.width, self.height)
        print("      Lane detection ready")
        
        print("\n[4/4] Setting up HUD...")
        self.hud = OptimizedHUD(self.width, self.height)
        self.logic = DrivingLogic()
        print("      Optimized HUD ready")
        
        # Output
        self.writer = None
        if Config.SAVE_OUTPUT:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                Config.OUTPUT_PATH, fourcc, self.video_fps,
                (self.width, self.height)
            )
            print(f"\n      Recording to: {Config.OUTPUT_PATH}")
        
        # Performance tracking
        self.fps_tracker = deque(maxlen=60)
        self.frame_times = deque(maxlen=100)
        
        print("\n" + "═" * 60)
        print("   Ready! Controls:")
        print("   Q - Quit")
        print("   R - Restart video")
        print("   1/2/3 - Change quality preset")
        print("═" * 60 + "\n")
    
    def run(self):
        """Main processing loop."""
        try:
            while True:
                start = time.time()
                
                ret, frame = self.cap.read()
                
                if not ret:
                    if Config.USE_TEST_VIDEO:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                
                # Detection with tracking
                detections = self.detector.detect(frame)
                
                # Lane detection
                lane_data = self.lane_detector.detect(frame)
                
                # Driving logic
                main, sub, color, threat = self.logic.analyze(detections, lane_data)
                self.hud.update_instruction(main, sub, color, threat)
                
                # FPS calculation
                elapsed = time.time() - start
                self.frame_times.append(elapsed)
                current_fps = 1 / elapsed if elapsed > 0 else 0
                self.fps_tracker.append(current_fps)
                avg_fps = sum(self.fps_tracker) / len(self.fps_tracker)
                
                # Render HUD
                output = self.hud.render(frame, detections, lane_data, 
                                        avg_fps, self.detector)
                
                # Save
                if self.writer:
                    self.writer.write(output)
                
                # Display
                cv2.imshow("AI Driving Assistant - Optimized", output)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('r') or key == ord('R'):
                    if Config.USE_TEST_VIDEO:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        print("Video restarted")
                elif key == ord('1'):
                    Config.QUALITY_PRESET = QualityPreset.PERFORMANCE
                    Config.apply_preset()
                    print("Switched to PERFORMANCE preset")
                elif key == ord('2'):
                    Config.QUALITY_PRESET = QualityPreset.BALANCED
                    Config.apply_preset()
                    print("Switched to BALANCED preset")
                elif key == ord('3'):
                    Config.QUALITY_PRESET = QualityPreset.QUALITY
                    Config.apply_preset()
                    print("Switched to QUALITY preset")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        
        # Stop async detector
        if hasattr(self, 'detector'):
            self.detector.stop()
        
        if self.writer:
            self.writer.release()
            print(f"  ✓ Saved: {Config.OUTPUT_PATH}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print performance stats
        if self.frame_times:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            print(f"  ✓ Average frame time: {avg_time*1000:.1f}ms")
            print(f"  ✓ Average FPS: {1/avg_time:.1f}")
        
        print("  ✓ Done!\n")


# ═══════════════════════════════════════════════════════════════════════════════
#                              ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        app = DrivingAssistant()
        app.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("   Ensure video file exists in the current directory.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
