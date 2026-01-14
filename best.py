#!/usr/bin/env python3


import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
from datetime import datetime
from collections import deque, defaultdict
import threading
from queue import Queue, Empty
import colorsys
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# # Install with pip
# pip install ultralytics opencv-python numpy

# # For CUDA acceleration (recommended for best performance)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# ═══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    """Central configuration for the entire system."""
    
    # ─── Source Configuration ───
    USE_TEST_VIDEO: bool = True
    TEST_VIDEO_PATH: str = "test.mp4"
    LIVE_CAMERA_INDEX: int = 0
    
    # ─── Output Configuration ───
    SAVE_OUTPUT: bool = True
    OUTPUT_PATH: str = "output_hud.mp4"
    
    # ─── Model Configuration ───
    YOLO_MODEL: str = "yolov8x.pt"  # Using largest model for best accuracy
    CONFIDENCE_THRESHOLD: float = 0.35
    IOU_THRESHOLD: float = 0.45
    
    # ─── Processing Configuration ───
    PROCESSING_WIDTH: int = 1280
    PROCESSING_HEIGHT: int = 720
    USE_MULTITHREADING: bool = True
    DETECTION_INTERVAL: int = 1  # Detect every N frames (1 = every frame)
    
    # ─── Indian Road Specific Classes (COCO) ───
    # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
    # 16: dog, 17: horse, 18: sheep, 19: cow
    VEHICLE_CLASSES: List[int] = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    PEDESTRIAN_CLASSES: List[int] = [0]  # person
    BICYCLE_CLASSES: List[int] = [1]  # bicycle
    ANIMAL_CLASSES: List[int] = [16, 17, 18, 19]  # dog, horse, sheep, cow
    ALL_DANGER_CLASSES: List[int] = [0, 1, 2, 3, 5, 7, 16, 17, 18, 19]
    
    # ─── Danger Zone Configuration ───
    DANGER_ZONE_CRITICAL: float = 0.20  # Object occupies 20%+ of frame = CRITICAL
    DANGER_ZONE_HIGH: float = 0.10      # 10%+ = HIGH
    DANGER_ZONE_MEDIUM: float = 0.04    # 4%+ = MEDIUM
    CENTER_LANE_WIDTH: float = 0.35     # Center 35% of frame width
    
    # ─── Tracking Configuration ───
    MAX_TRACK_AGE: int = 30
    MIN_HITS_TO_TRACK: int = 3
    TRAJECTORY_LENGTH: int = 30
    
    # ─── Smoothing Configuration ───
    INSTRUCTION_SMOOTHING_FRAMES: int = 8
    ALERT_IMMEDIATE_OVERRIDE: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
#                              COLOR PALETTE
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    """Futuristic color palette (BGR format for OpenCV)."""
    
    # ─── Primary Colors ───
    NEON_CYAN = (255, 255, 0)
    NEON_BLUE = (255, 191, 0)
    ELECTRIC_BLUE = (255, 144, 30)
    NEON_GREEN = (57, 255, 20)
    LIME_GREEN = (0, 255, 128)
    
    # ─── Alert Colors ───
    WARNING_ORANGE = (0, 165, 255)
    ALERT_YELLOW = (0, 255, 255)
    DANGER_RED = (60, 76, 231)
    CRITICAL_RED = (0, 0, 255)
    
    # ─── Neutral Colors ───
    PURE_WHITE = (255, 255, 255)
    SOFT_WHITE = (230, 230, 230)
    STEEL_GRAY = (120, 120, 120)
    DARK_GRAY = (40, 40, 40)
    NEAR_BLACK = (15, 15, 15)
    PURE_BLACK = (0, 0, 0)
    
    # ─── Accent Colors ───
    HOLOGRAM_BLUE = (255, 200, 100)
    PLASMA_PURPLE = (255, 100, 180)
    MATRIX_GREEN = (0, 200, 0)
    
    # ─── Gradient Stops ───
    GRADIENT_START = (255, 150, 50)
    GRADIENT_END = (255, 50, 150)
    
    @staticmethod
    def with_alpha(color: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int, int]:
        """Add alpha channel to color."""
        return (*color, int(alpha * 255))
    
    @staticmethod
    def interpolate(color1: Tuple, color2: Tuple, t: float) -> Tuple[int, int, int]:
        """Interpolate between two colors."""
        t = max(0, min(1, t))
        return tuple(int(c1 + (c2 - c1) * t) for c1, c2 in zip(color1, color2))
    
    @staticmethod
    def pulse(base_color: Tuple, intensity: float = 0.3) -> Tuple[int, int, int]:
        """Create pulsing color effect."""
        pulse_factor = 1 + intensity * math.sin(time.time() * 5)
        return tuple(int(min(255, c * pulse_factor)) for c in base_color)


# ═══════════════════════════════════════════════════════════════════════════════
#                              DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class ThreatLevel(Enum):
    """Threat classification levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Detection:
    """Represents a single detected object."""
    bbox: Tuple[int, int, int, int]
    class_id: int
    class_name: str
    confidence: float
    center: Tuple[int, int]
    area: float
    relative_area: float
    position: str  # "LEFT", "CENTER", "RIGHT"
    distance_estimate: float
    threat_level: ThreatLevel
    velocity: Tuple[float, float] = (0.0, 0.0)
    time_to_collision: float = float('inf')
    track_id: int = -1
    trajectory: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class LaneInfo:
    """Lane detection information."""
    left_lane: Optional[np.ndarray] = None
    right_lane: Optional[np.ndarray] = None
    center_offset: float = 0.0
    lane_curvature: float = 0.0
    steering_suggestion: str = "CENTERED"
    lane_departure: bool = False
    confidence: float = 0.0


@dataclass
class FrameAnalysis:
    """Complete analysis of a single frame."""
    detections: List[Detection]
    lane_info: LaneInfo
    overall_threat: ThreatLevel
    frame_time: float
    fps: float


# ═══════════════════════════════════════════════════════════════════════════════
#                              KALMAN TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class KalmanTracker:
    """Kalman filter-based object tracker for smooth trajectory estimation."""
    
    def __init__(self, initial_bbox: Tuple[int, int, int, int], track_id: int):
        self.track_id = track_id
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.trajectory = deque(maxlen=Config.TRAJECTORY_LENGTH)
        
        # Initialize Kalman filter (state: [x, y, w, h, vx, vy])
        self.kf = cv2.KalmanFilter(6, 4)
        
        # Transition matrix
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
        
        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # Initialize state
        x1, y1, x2, y2 = initial_bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        self.kf.statePost = np.array([[cx], [cy], [w], [h], [0], [0]], dtype=np.float32)
        
        self.trajectory.append((int(cx), int(cy)))
    
    def predict(self) -> Tuple[int, int, int, int]:
        """Predict next state."""
        prediction = self.kf.predict()
        cx, cy, w, h = prediction[:4].flatten()
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        self.age += 1
        self.time_since_update += 1
        return (x1, y1, x2, y2)
    
    def update(self, bbox: Tuple[int, int, int, int]):
        """Update with new measurement."""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        
        measurement = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        self.kf.correct(measurement)
        
        self.hits += 1
        self.time_since_update = 0
        self.trajectory.append((int(cx), int(cy)))
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate."""
        state = self.kf.statePost.flatten()
        return (state[4], state[5])
    
    def get_bbox(self) -> Tuple[int, int, int, int]:
        """Get current bounding box estimate."""
        state = self.kf.statePost.flatten()
        cx, cy, w, h = state[:4]
        return (int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2))


class MultiObjectTracker:
    """Multi-object tracker using Kalman filters and Hungarian algorithm."""
    
    def __init__(self):
        self.trackers: Dict[int, KalmanTracker] = {}
        self.next_id = 0
        self.iou_threshold = 0.3
    
    def _iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate Intersection over Union."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union_area = bbox1_area + bbox2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0
    
    def update(self, detections: List[Tuple[int, int, int, int]]) -> Dict[int, KalmanTracker]:
        """Update trackers with new detections."""
        # Predict all existing trackers
        predictions = {}
        for track_id, tracker in self.trackers.items():
            predictions[track_id] = tracker.predict()
        
        # Match detections to predictions using IoU
        matched = set()
        matched_trackers = set()
        
        if predictions and detections:
            # Build cost matrix
            cost_matrix = np.zeros((len(detections), len(predictions)))
            track_ids = list(predictions.keys())
            
            for i, det in enumerate(detections):
                for j, track_id in enumerate(track_ids):
                    cost_matrix[i, j] = 1 - self._iou(det, predictions[track_id])
            
            # Hungarian algorithm (greedy approximation for simplicity)
            for i, det in enumerate(detections):
                best_j = -1
                best_iou = self.iou_threshold
                for j, track_id in enumerate(track_ids):
                    if track_id in matched_trackers:
                        continue
                    iou = 1 - cost_matrix[i, j]
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                
                if best_j >= 0:
                    track_id = track_ids[best_j]
                    self.trackers[track_id].update(det)
                    matched.add(i)
                    matched_trackers.add(track_id)
        
        # Create new trackers for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched:
                self.trackers[self.next_id] = KalmanTracker(det, self.next_id)
                self.next_id += 1
        
        # Remove stale trackers
        stale_ids = []
        for track_id, tracker in self.trackers.items():
            if tracker.time_since_update > Config.MAX_TRACK_AGE:
                stale_ids.append(track_id)
        
        for track_id in stale_ids:
            del self.trackers[track_id]
        
        return self.trackers


# ═══════════════════════════════════════════════════════════════════════════════
#                           ADVANCED LANE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedLaneDetector:
    """Advanced lane detection using perspective transform and polynomial fitting."""
    
    def __init__(self, frame_width: int, frame_height: int):
        self.width = frame_width
        self.height = frame_height
        
        # Perspective transform points (tuned for Indian roads)
        self.src_points = np.float32([
            [frame_width * 0.1, frame_height],
            [frame_width * 0.4, frame_height * 0.6],
            [frame_width * 0.6, frame_height * 0.6],
            [frame_width * 0.9, frame_height]
        ])
        
        self.dst_points = np.float32([
            [frame_width * 0.2, frame_height],
            [frame_width * 0.2, 0],
            [frame_width * 0.8, 0],
            [frame_width * 0.8, frame_height]
        ])
        
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
        
        # Lane history for smoothing
        self.left_fit_history = deque(maxlen=10)
        self.right_fit_history = deque(maxlen=10)
        
        # Sliding window parameters
        self.n_windows = 12
        self.window_margin = 80
        self.min_pixels = 40
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Advanced preprocessing for lane detection."""
        # Convert to different color spaces
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # S channel from HLS (good for yellow lines)
        s_channel = hls[:, :, 2]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel > 100) & (s_channel < 255)] = 1
        
        # L channel from LAB (good for white lines)
        l_channel = lab[:, :, 0]
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel > 200)] = 1
        
        # Sobel edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx)) if np.max(abs_sobelx) > 0 else np.zeros_like(abs_sobelx, dtype=np.uint8)
        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[(scaled_sobel > 30) & (scaled_sobel < 150)] = 1
        
        # Combine
        combined = np.zeros_like(s_binary)
        combined[(s_binary == 1) | (l_binary == 1) | (sobel_binary == 1)] = 255
        
        return combined.astype(np.uint8)
    
    def _sliding_window(self, binary_warped: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """Find lane pixels using sliding window technique."""
        # Histogram of bottom half
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        
        # Starting points
        midpoint = len(histogram) // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Window parameters
        window_height = binary_warped.shape[0] // self.n_windows
        
        # Identify nonzero pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # Lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        # Visualization
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        
        for window in range(self.n_windows):
            # Window boundaries
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            
            win_xleft_low = leftx_current - self.window_margin
            win_xleft_high = leftx_current + self.window_margin
            win_xright_low = rightx_current - self.window_margin
            win_xright_high = rightx_current + self.window_margin
            
            # Identify pixels in window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # Recenter
            if len(good_left_inds) > self.min_pixels:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.min_pixels:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate indices
        left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([])
        right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([])
        
        # Extract pixel positions
        leftx = nonzerox[left_lane_inds] if len(left_lane_inds) > 0 else None
        lefty = nonzeroy[left_lane_inds] if len(left_lane_inds) > 0 else None
        rightx = nonzerox[right_lane_inds] if len(right_lane_inds) > 0 else None
        righty = nonzeroy[right_lane_inds] if len(right_lane_inds) > 0 else None
        
        # Fit polynomial
        left_fit = None
        right_fit = None
        
        if leftx is not None and len(leftx) > 100:
            try:
                left_fit = np.polyfit(lefty, leftx, 2)
                self.left_fit_history.append(left_fit)
            except:
                pass
        
        if rightx is not None and len(rightx) > 100:
            try:
                right_fit = np.polyfit(righty, rightx, 2)
                self.right_fit_history.append(right_fit)
            except:
                pass
        
        # Use averaged fits for smoothing
        if left_fit is None and self.left_fit_history:
            left_fit = np.mean(self.left_fit_history, axis=0)
        if right_fit is None and self.right_fit_history:
            right_fit = np.mean(self.right_fit_history, axis=0)
        
        return left_fit, right_fit, out_img
    
    def detect(self, frame: np.ndarray) -> LaneInfo:
        """Main lane detection method."""
        lane_info = LaneInfo()
        
        # Preprocess
        binary = self._preprocess(frame)
        
        # Perspective transform
        binary_warped = cv2.warpPerspective(binary, self.M, (self.width, self.height))
        
        # Find lanes
        left_fit, right_fit, _ = self._sliding_window(binary_warped)
        
        if left_fit is not None:
            lane_info.left_lane = left_fit
            lane_info.confidence += 0.5
        
        if right_fit is not None:
            lane_info.right_lane = right_fit
            lane_info.confidence += 0.5
        
        # Calculate center offset
        if left_fit is not None and right_fit is not None:
            y_eval = self.height - 1
            left_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
            right_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
            lane_center = (left_x + right_x) / 2
            car_center = self.width / 2
            
            lane_info.center_offset = (car_center - lane_center) / (self.width / 2)  # Normalized
            
            # Calculate curvature
            y_m_per_pix = 30 / self.height
            x_m_per_pix = 3.7 / (right_x - left_x) if right_x > left_x else 3.7 / self.width
            
            left_fit_cr = np.polyfit(np.array([y_eval]) * y_m_per_pix, 
                                    np.array([left_x]) * x_m_per_pix, 2)
            lane_info.lane_curvature = ((1 + (2*left_fit_cr[0]*y_eval*y_m_per_pix + left_fit_cr[1])**2)**1.5) / abs(2*left_fit_cr[0]) if left_fit_cr[0] != 0 else float('inf')
            
            # Steering suggestion
            if abs(lane_info.center_offset) < 0.1:
                lane_info.steering_suggestion = "CENTERED"
            elif lane_info.center_offset > 0.1:
                lane_info.steering_suggestion = "DRIFT RIGHT → STEER LEFT"
                lane_info.lane_departure = lane_info.center_offset > 0.3
            else:
                lane_info.steering_suggestion = "DRIFT LEFT → STEER RIGHT"
                lane_info.lane_departure = lane_info.center_offset < -0.3
        
        return lane_info
    
    def draw_lanes(self, frame: np.ndarray, lane_info: LaneInfo) -> np.ndarray:
        """Draw detected lanes on frame."""
        if lane_info.left_lane is None and lane_info.right_lane is None:
            return frame
        
        # Create blank for drawing
        lane_overlay = np.zeros_like(frame)
        
        # Generate points
        ploty = np.linspace(0, self.height - 1, self.height)
        
        pts_left = None
        pts_right = None
        
        if lane_info.left_lane is not None:
            left_fitx = lane_info.left_lane[0]*ploty**2 + lane_info.left_lane[1]*ploty + lane_info.left_lane[2]
            left_fitx = np.clip(left_fitx, 0, self.width - 1)
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        
        if lane_info.right_lane is not None:
            right_fitx = lane_info.right_lane[0]*ploty**2 + lane_info.right_lane[1]*ploty + lane_info.right_lane[2]
            right_fitx = np.clip(right_fitx, 0, self.width - 1)
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        
        # Draw lane area
        if pts_left is not None and pts_right is not None:
            pts = np.hstack((pts_left, pts_right))
            
            # Choose color based on lane departure
            if lane_info.lane_departure:
                fill_color = Colors.DANGER_RED
            else:
                fill_color = Colors.NEON_GREEN
            
            # Fill lane area
            cv2.fillPoly(lane_overlay, np.int_([pts]), fill_color)
        
        # Draw lane lines
        if pts_left is not None:
            cv2.polylines(lane_overlay, np.int_([pts_left]), False, Colors.NEON_CYAN, 8)
        if pts_right is not None:
            cv2.polylines(lane_overlay, np.int_([pts_right]), False, Colors.NEON_CYAN, 8)
        
        # Inverse perspective transform
        lane_overlay_warped = cv2.warpPerspective(lane_overlay, self.M_inv, (self.width, self.height))
        
        # Blend
        result = cv2.addWeighted(frame, 1, lane_overlay_warped, 0.3, 0)
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
#                           OBJECT DETECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ObjectDetectionEngine:
    """Advanced object detection with YOLO and multi-object tracking."""
    
    def __init__(self, frame_width: int, frame_height: int):
        print(f"    → Loading YOLO model: {Config.YOLO_MODEL}")
        self.model = YOLO(Config.YOLO_MODEL)
        self.width = frame_width
        self.height = frame_height
        self.frame_area = frame_width * frame_height
        
        # Tracker
        self.tracker = MultiObjectTracker()
        
        # Class name mapping for Indian roads
        self.indian_class_names = {
            0: "PEDESTRIAN",
            1: "BICYCLE",
            2: "CAR",
            3: "MOTORCYCLE",  # Includes auto-rickshaws visually
            5: "BUS",
            7: "TRUCK",
            16: "DOG",
            17: "HORSE",
            18: "SHEEP", 
            19: "COW"
        }
        
        # Typical real-world widths (meters) for distance estimation
        self.typical_widths = {
            0: 0.5,   # person
            1: 0.6,   # bicycle
            2: 1.8,   # car
            3: 0.8,   # motorcycle
            5: 2.5,   # bus
            7: 2.5,   # truck
            16: 0.3,  # dog
            17: 0.5,  # horse
            18: 0.4,  # sheep
            19: 0.8   # cow
        }
        
        # Focal length estimate (pixels) - calibrate for your camera
        self.focal_length = 800
    
    def _estimate_distance(self, bbox_width: int, class_id: int) -> float:
        """Estimate distance using pinhole camera model."""
        real_width = self.typical_widths.get(class_id, 1.0)
        if bbox_width > 0:
            distance = (real_width * self.focal_length) / bbox_width
            return round(distance, 1)
        return float('inf')
    
    def _calculate_threat_level(self, detection: Detection) -> ThreatLevel:
        """Calculate threat level based on multiple factors."""
        threat_score = 0
        
        # Factor 1: Relative area (proximity)
        if detection.relative_area > Config.DANGER_ZONE_CRITICAL:
            threat_score += 4
        elif detection.relative_area > Config.DANGER_ZONE_HIGH:
            threat_score += 3
        elif detection.relative_area > Config.DANGER_ZONE_MEDIUM:
            threat_score += 2
        
        # Factor 2: Position (center is more dangerous)
        if detection.position == "CENTER":
            threat_score += 2
        
        # Factor 3: Object type (pedestrians and animals are high priority)
        if detection.class_id in Config.PEDESTRIAN_CLASSES:
            threat_score += 2
        elif detection.class_id in Config.ANIMAL_CLASSES:
            threat_score += 2
        elif detection.class_id in Config.BICYCLE_CLASSES:
            threat_score += 1
        
        # Factor 4: Approaching velocity (negative vy means approaching in image coords)
        if detection.velocity[1] > 10:  # Moving towards bottom of frame = approaching
            threat_score += 1
        
        # Factor 5: Time to collision
        if detection.time_to_collision < 2.0:
            threat_score += 3
        elif detection.time_to_collision < 5.0:
            threat_score += 1
        
        # Map score to threat level
        if threat_score >= 6:
            return ThreatLevel.CRITICAL
        elif threat_score >= 4:
            return ThreatLevel.HIGH
        elif threat_score >= 2:
            return ThreatLevel.MEDIUM
        elif threat_score >= 1:
            return ThreatLevel.LOW
        return ThreatLevel.NONE
    
    def _get_position(self, center_x: int) -> str:
        """Determine position relative to center of frame."""
        center_zone_left = self.width * (0.5 - Config.CENTER_LANE_WIDTH / 2)
        center_zone_right = self.width * (0.5 + Config.CENTER_LANE_WIDTH / 2)
        
        if center_x < center_zone_left:
            return "LEFT"
        elif center_x > center_zone_right:
            return "RIGHT"
        return "CENTER"
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection and tracking on frame."""
        # Run YOLO
        results = self.model(
            frame, 
            stream=True, 
            verbose=False,
            conf=Config.CONFIDENCE_THRESHOLD,
            iou=Config.IOU_THRESHOLD,
            classes=Config.ALL_DANGER_CLASSES
        )
        
        raw_detections = []
        detection_bboxes = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Bounds checking
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(self.width, x2), min(self.height, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                area = (x2 - x1) * (y2 - y1)
                relative_area = area / self.frame_area
                
                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=cls,
                    class_name=self.indian_class_names.get(cls, self.model.names[cls].upper()),
                    confidence=conf,
                    center=center,
                    area=area,
                    relative_area=relative_area,
                    position=self._get_position(center[0]),
                    distance_estimate=self._estimate_distance(x2 - x1, cls),
                    threat_level=ThreatLevel.NONE
                )
                
                raw_detections.append(detection)
                detection_bboxes.append((x1, y1, x2, y2))
        
        # Update tracker
        tracked = self.tracker.update(detection_bboxes)
        
        # Associate tracks with detections and calculate velocities
        final_detections = []
        for detection in raw_detections:
            best_track = None
            best_iou = 0.3
            
            for track_id, tracker in tracked.items():
                iou = self.tracker._iou(detection.bbox, tracker.get_bbox())
                if iou > best_iou:
                    best_iou = iou
                    best_track = tracker
            
            if best_track is not None:
                detection.track_id = best_track.track_id
                detection.velocity = best_track.get_velocity()
                detection.trajectory = list(best_track.trajectory)
                
                # Calculate TTC
                vy = detection.velocity[1]
                if vy > 5:  # Approaching
                    remaining_distance = self.height - detection.center[1]
                    if remaining_distance > 0:
                        detection.time_to_collision = remaining_distance / vy / 30  # Approximate seconds
            
            # Calculate final threat level
            detection.threat_level = self._calculate_threat_level(detection)
            final_detections.append(detection)
        
        return final_detections


# ═══════════════════════════════════════════════════════════════════════════════
#                              FUTURISTIC HUD RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

class FuturisticHUD:
    """Ultra-modern, futuristic heads-up display renderer."""
    
    def __init__(self, frame_width: int, frame_height: int):
        self.width = frame_width
        self.height = frame_height
        self.frame_count = 0
        
        # Animation state
        self.crosshair_rotation = 0
        self.scan_line_y = 0
        self.radar_rotation = 0
        self.pulse_phase = 0
        
        # Fonts
        self.font_primary = cv2.FONT_HERSHEY_DUPLEX
        self.font_secondary = cv2.FONT_HERSHEY_SIMPLEX
        
        # Instruction smoothing
        self.display_main = "SYSTEM READY"
        self.display_sub = "Initializing..."
        self.display_color = Colors.NEON_CYAN
        self.last_raw_main = ""
        self.consistency_counter = 0
        
        # Pre-computed elements
        self._generate_static_elements()
    
    def _generate_static_elements(self):
        """Pre-generate static HUD elements for performance."""
        # Vignette mask
        self.vignette = np.zeros((self.height, self.width), dtype=np.float32)
        cy, cx = self.height // 2, self.width // 2
        max_dist = math.sqrt(cx**2 + cy**2)
        
        for y in range(self.height):
            for x in range(self.width):
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                self.vignette[y, x] = 1 - (dist / max_dist) ** 1.5
        
        self.vignette = np.clip(self.vignette * 1.3, 0.3, 1.0)
        self.vignette = cv2.merge([self.vignette, self.vignette, self.vignette])
    
    def _draw_text_with_glow(self, img: np.ndarray, text: str, pos: Tuple[int, int], 
                            font_scale: float, color: Tuple, thickness: int = 1,
                            glow_color: Optional[Tuple] = None, glow_strength: int = 2):
        """Draw text with optional glow effect."""
        x, y = pos
        
        # Glow effect (multiple passes)
        if glow_color:
            for i in range(glow_strength, 0, -1):
                alpha = 0.3 / i
                glow_size = thickness + i * 2
                temp_overlay = img.copy()
                cv2.putText(temp_overlay, text, (x, y), self.font_primary, 
                           font_scale, glow_color, glow_size)
                cv2.addWeighted(temp_overlay, alpha, img, 1 - alpha, 0, img)
        
        # Outline
        cv2.putText(img, text, (x, y), self.font_primary, font_scale, 
                   Colors.PURE_BLACK, thickness + 3)
        # Main text
        cv2.putText(img, text, (x, y), self.font_primary, font_scale, color, thickness)
    
    def _draw_animated_brackets(self, img: np.ndarray, bbox: Tuple[int, int, int, int], 
                                color: Tuple, threat_level: ThreatLevel):
        """Draw animated corner brackets with threat indication."""
        x1, y1, x2, y2 = bbox
        
        # Dynamic sizing based on threat
        base_length = 20
        thickness = 2
        
        if threat_level == ThreatLevel.CRITICAL:
            # Pulsing effect for critical threats
            pulse = abs(math.sin(self.pulse_phase * 2)) * 0.5 + 0.5
            base_length = int(30 + 10 * pulse)
            thickness = 3
            color = Colors.pulse(Colors.CRITICAL_RED, 0.5)
        elif threat_level == ThreatLevel.HIGH:
            base_length = 25
            thickness = 2
        
        length = min(base_length, min(x2-x1, y2-y1) // 3)
        
        # Corner brackets with slight animation
        offset = int(3 * math.sin(self.pulse_phase))
        
        # Top Left
        cv2.line(img, (x1 - offset, y1), (x1 + length, y1), color, thickness)
        cv2.line(img, (x1, y1 - offset), (x1, y1 + length), color, thickness)
        
        # Top Right
        cv2.line(img, (x2 + offset, y1), (x2 - length, y1), color, thickness)
        cv2.line(img, (x2, y1 - offset), (x2, y1 + length), color, thickness)
        
        # Bottom Left
        cv2.line(img, (x1 - offset, y2), (x1 + length, y2), color, thickness)
        cv2.line(img, (x1, y2 + offset), (x1, y2 - length), color, thickness)
        
        # Bottom Right
        cv2.line(img, (x2 + offset, y2), (x2 - length, y2), color, thickness)
        cv2.line(img, (x2, y2 + offset), (x2, y2 - length), color, thickness)
        
        # Inner glow for high threats
        if threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
            overlay = img.copy()
            glow_alpha = 0.15 + 0.1 * abs(math.sin(self.pulse_phase))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, glow_alpha, img, 1 - glow_alpha, 0, img)
    
    def _draw_object_info(self, img: np.ndarray, detection: Detection):
        """Draw detailed object information."""
        x1, y1, x2, y2 = detection.bbox
        
        # Choose color based on threat
        color = {
            ThreatLevel.NONE: Colors.NEON_CYAN,
            ThreatLevel.LOW: Colors.LIME_GREEN,
            ThreatLevel.MEDIUM: Colors.ALERT_YELLOW,
            ThreatLevel.HIGH: Colors.WARNING_ORANGE,
            ThreatLevel.CRITICAL: Colors.CRITICAL_RED
        }.get(detection.threat_level, Colors.NEON_CYAN)
        
        # Draw brackets
        self._draw_animated_brackets(img, detection.bbox, color, detection.threat_level)
        
        # Label background
        label = f"{detection.class_name}"
        dist_label = f"{detection.distance_estimate}m"
        
        label_size = cv2.getTextSize(label, self.font_secondary, 0.5, 1)[0]
        
        # Draw label with background
        label_y = max(y1 - 25, 20)
        cv2.rectangle(img, (x1, label_y - 15), (x1 + label_size[0] + 60, label_y + 5), 
                     Colors.NEAR_BLACK, -1)
        cv2.rectangle(img, (x1, label_y - 15), (x1 + label_size[0] + 60, label_y + 5), 
                     color, 1)
        
        # Label text
        cv2.putText(img, label, (x1 + 5, label_y), self.font_secondary, 0.5, color, 1)
        cv2.putText(img, dist_label, (x1 + label_size[0] + 10, label_y), 
                   self.font_secondary, 0.5, Colors.SOFT_WHITE, 1)
        
        # Confidence bar
        bar_width = x2 - x1
        conf_width = int(bar_width * detection.confidence)
        cv2.rectangle(img, (x1, y2 + 5), (x2, y2 + 8), Colors.DARK_GRAY, -1)
        cv2.rectangle(img, (x1, y2 + 5), (x1 + conf_width, y2 + 8), color, -1)
        
        # Draw trajectory
        if len(detection.trajectory) > 5:
            pts = np.array(detection.trajectory, dtype=np.int32)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                pt_color = Colors.interpolate(Colors.DARK_GRAY, color, alpha)
                cv2.line(img, tuple(pts[i-1]), tuple(pts[i]), pt_color, 
                        max(1, int(3 * alpha)))
    
    def _draw_animated_crosshair(self, img: np.ndarray):
        """Draw animated center crosshair."""
        cx, cy = self.width // 2, self.height // 2
        
        # Outer rotating elements
        radius = 40
        for i in range(4):
            angle = self.crosshair_rotation + i * (math.pi / 2)
            x_start = int(cx + radius * math.cos(angle))
            y_start = int(cy + radius * math.sin(angle))
            x_end = int(cx + (radius + 15) * math.cos(angle))
            y_end = int(cy + (radius + 15) * math.sin(angle))
            cv2.line(img, (x_start, y_start), (x_end, y_end), Colors.NEON_CYAN, 1)
        
        # Inner crosshair with gap
        gap = 8
        line_len = 25
        
        # Horizontal lines
        cv2.line(img, (cx - line_len - gap, cy), (cx - gap, cy), Colors.NEON_CYAN, 1)
        cv2.line(img, (cx + gap, cy), (cx + line_len + gap, cy), Colors.NEON_CYAN, 1)
        
        # Vertical lines
        cv2.line(img, (cx, cy - line_len - gap), (cx, cy - gap), Colors.NEON_CYAN, 1)
        cv2.line(img, (cx, cy + gap), (cx, cy + line_len + gap), Colors.NEON_CYAN, 1)
        
        # Center dot (pulsing)
        dot_size = int(2 + abs(math.sin(self.pulse_phase)) * 2)
        cv2.circle(img, (cx, cy), dot_size, Colors.NEON_CYAN, -1)
        
        # Outer ring
        cv2.circle(img, (cx, cy), 50, Colors.NEON_CYAN, 1)
    
    def _draw_radar(self, img: np.ndarray, detections: List[Detection]):
        """Draw mini radar/minimap."""
        # Radar position and size
        radar_x, radar_y = self.width - 150, 150
        radar_radius = 70
        
        # Background
        overlay = img.copy()
        cv2.circle(overlay, (radar_x, radar_y), radar_radius + 5, Colors.NEAR_BLACK, -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        
        # Radar rings
        for r in range(20, radar_radius, 20):
            cv2.circle(img, (radar_x, radar_y), r, Colors.DARK_GRAY, 1)
        
        # Radar sweep
        sweep_angle = self.radar_rotation
        sweep_x = int(radar_x + radar_radius * math.cos(sweep_angle))
        sweep_y = int(radar_y + radar_radius * math.sin(sweep_angle))
        
        # Sweep arc
        for i in range(30):
            a = sweep_angle - i * 0.05
            alpha = (30 - i) / 30
            x = int(radar_x + radar_radius * math.cos(a))
            y = int(radar_y + radar_radius * math.sin(a))
            color = Colors.interpolate(Colors.PURE_BLACK, Colors.NEON_GREEN, alpha)
            cv2.line(img, (radar_x, radar_y), (x, y), color, 1)
        
        # Main sweep line
        cv2.line(img, (radar_x, radar_y), (sweep_x, sweep_y), Colors.NEON_GREEN, 2)
        
        # Plot detections on radar
        for det in detections:
            # Map detection position to radar
            rel_x = (det.center[0] - self.width / 2) / (self.width / 2)
            rel_y = (self.height - det.center[1]) / self.height
            
            radar_det_x = int(radar_x + rel_x * radar_radius * 0.8)
            radar_det_y = int(radar_y - rel_y * radar_radius * 0.8)
            
            # Color by threat
            det_color = {
                ThreatLevel.CRITICAL: Colors.CRITICAL_RED,
                ThreatLevel.HIGH: Colors.WARNING_ORANGE,
                ThreatLevel.MEDIUM: Colors.ALERT_YELLOW,
                ThreatLevel.LOW: Colors.LIME_GREEN,
                ThreatLevel.NONE: Colors.NEON_CYAN
            }.get(det.threat_level, Colors.NEON_CYAN)
            
            # Pulsing for high threats
            size = 4
            if det.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
                size = int(4 + 2 * abs(math.sin(self.pulse_phase * 2)))
            
            cv2.circle(img, (radar_det_x, radar_det_y), size, det_color, -1)
        
        # Center dot (vehicle)
        cv2.circle(img, (radar_x, radar_y + radar_radius // 2), 5, Colors.NEON_CYAN, -1)
        
        # Cardinal directions
        cv2.putText(img, "N", (radar_x - 5, radar_y - radar_radius - 5), 
                   self.font_secondary, 0.4, Colors.SOFT_WHITE, 1)
        
        # Outer border
        cv2.circle(img, (radar_x, radar_y), radar_radius, Colors.NEON_CYAN, 2)
        
        # Label
        cv2.putText(img, "RADAR", (radar_x - 25, radar_y + radar_radius + 20), 
                   self.font_secondary, 0.5, Colors.STEEL_GRAY, 1)
    
    def _draw_threat_gauge(self, img: np.ndarray, threat_level: ThreatLevel):
        """Draw threat level gauge."""
        gauge_x, gauge_y = 30, self.height - 200
        gauge_width = 15
        gauge_height = 100
        
        # Background
        cv2.rectangle(img, (gauge_x - 2, gauge_y - 2), 
                     (gauge_x + gauge_width + 2, gauge_y + gauge_height + 2),
                     Colors.NEAR_BLACK, -1)
        cv2.rectangle(img, (gauge_x - 2, gauge_y - 2), 
                     (gauge_x + gauge_width + 2, gauge_y + gauge_height + 2),
                     Colors.STEEL_GRAY, 1)
        
        # Threat level to height
        level_heights = {
            ThreatLevel.NONE: 0.0,
            ThreatLevel.LOW: 0.25,
            ThreatLevel.MEDIUM: 0.5,
            ThreatLevel.HIGH: 0.75,
            ThreatLevel.CRITICAL: 1.0
        }
        
        fill_ratio = level_heights.get(threat_level, 0)
        fill_height = int(gauge_height * fill_ratio)
        
        # Draw gradient fill
        for i in range(fill_height):
            ratio = i / gauge_height
            color = Colors.interpolate(Colors.NEON_GREEN, Colors.CRITICAL_RED, ratio)
            y = gauge_y + gauge_height - i
            cv2.line(img, (gauge_x, y), (gauge_x + gauge_width, y), color, 1)
        
        # Tick marks
        for i in range(5):
            y = gauge_y + int(i * gauge_height / 4)
            cv2.line(img, (gauge_x - 5, y), (gauge_x, y), Colors.SOFT_WHITE, 1)
        
        # Labels
        cv2.putText(img, "THREAT", (gauge_x - 5, gauge_y - 10), 
                   self.font_secondary, 0.4, Colors.STEEL_GRAY, 1)
        
        # Level indicator
        level_names = {
            ThreatLevel.NONE: "CLEAR",
            ThreatLevel.LOW: "LOW",
            ThreatLevel.MEDIUM: "MED",
            ThreatLevel.HIGH: "HIGH",
            ThreatLevel.CRITICAL: "CRIT"
        }
        level_text = level_names.get(threat_level, "---")
        
        text_color = Colors.interpolate(Colors.NEON_GREEN, Colors.CRITICAL_RED, fill_ratio)
        cv2.putText(img, level_text, (gauge_x - 5, gauge_y + gauge_height + 20), 
                   self.font_secondary, 0.4, text_color, 1)
    
    def _draw_telemetry_panel(self, img: np.ndarray, fps: float, 
                              num_detections: int, lane_info: LaneInfo):
        """Draw telemetry data panel."""
        panel_x, panel_y = 20, 80
        panel_width = 200
        panel_height = 120
        
        # Panel background
        overlay = img.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     Colors.NEAR_BLACK, -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Border
        cv2.rectangle(img, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     Colors.NEON_CYAN, 1)
        
        # Corner accents
        accent_len = 15
        cv2.line(img, (panel_x, panel_y), (panel_x + accent_len, panel_y), Colors.NEON_CYAN, 2)
        cv2.line(img, (panel_x, panel_y), (panel_x, panel_y + accent_len), Colors.NEON_CYAN, 2)
        cv2.line(img, (panel_x + panel_width, panel_y), 
                (panel_x + panel_width - accent_len, panel_y), Colors.NEON_CYAN, 2)
        cv2.line(img, (panel_x + panel_width, panel_y), 
                (panel_x + panel_width, panel_y + accent_len), Colors.NEON_CYAN, 2)
        
        # Title
        cv2.putText(img, "TELEMETRY", (panel_x + 10, panel_y + 20), 
                   self.font_secondary, 0.5, Colors.NEON_CYAN, 1)
        cv2.line(img, (panel_x + 10, panel_y + 25), 
                (panel_x + panel_width - 10, panel_y + 25), Colors.STEEL_GRAY, 1)
        
        # Data rows
        data_start_y = panel_y + 45
        row_height = 22
        
        # FPS
        fps_color = Colors.NEON_GREEN if fps > 20 else Colors.ALERT_YELLOW if fps > 10 else Colors.CRITICAL_RED
        cv2.putText(img, f"FPS:", (panel_x + 15, data_start_y), 
                   self.font_secondary, 0.45, Colors.STEEL_GRAY, 1)
        cv2.putText(img, f"{int(fps)}", (panel_x + 80, data_start_y), 
                   self.font_secondary, 0.45, fps_color, 1)
        
        # Objects
        cv2.putText(img, f"OBJECTS:", (panel_x + 15, data_start_y + row_height), 
                   self.font_secondary, 0.45, Colors.STEEL_GRAY, 1)
        cv2.putText(img, f"{num_detections}", (panel_x + 80, data_start_y + row_height), 
                   self.font_secondary, 0.45, Colors.NEON_CYAN, 1)
        
        # Lane status
        lane_status = "OK" if lane_info.confidence > 0.5 else "LOW"
        lane_color = Colors.NEON_GREEN if lane_info.confidence > 0.5 else Colors.ALERT_YELLOW
        cv2.putText(img, f"LANE:", (panel_x + 15, data_start_y + row_height * 2), 
                   self.font_secondary, 0.45, Colors.STEEL_GRAY, 1)
        cv2.putText(img, lane_status, (panel_x + 80, data_start_y + row_height * 2), 
                   self.font_secondary, 0.45, lane_color, 1)
        
        # Lane offset indicator
        offset = lane_info.center_offset
        bar_x = panel_x + 100
        bar_y = data_start_y + row_height * 2 - 10
        bar_width = 80
        bar_height = 8
        
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     Colors.DARK_GRAY, -1)
        
        indicator_x = bar_x + bar_width // 2 + int(offset * bar_width / 2)
        indicator_x = max(bar_x + 3, min(bar_x + bar_width - 3, indicator_x))
        cv2.circle(img, (indicator_x, bar_y + bar_height // 2), 4, Colors.NEON_CYAN, -1)
    
    def _draw_top_bar(self, img: np.ndarray):
        """Draw top status bar."""
        bar_height = 50
        
        # Background gradient effect
        overlay = img.copy()
        for i in range(bar_height):
            alpha = 1 - (i / bar_height)
            color = Colors.interpolate(Colors.NEAR_BLACK, Colors.PURE_BLACK, alpha)
            cv2.line(overlay, (0, i), (self.width, i), color, 1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        
        # Border line
        cv2.line(img, (0, bar_height), (self.width, bar_height), Colors.NEON_CYAN, 1)
        
        # Corner accents
        accent_len = 100
        cv2.line(img, (0, bar_height), (accent_len, bar_height), Colors.NEON_CYAN, 2)
        cv2.line(img, (self.width - accent_len, bar_height), (self.width, bar_height), Colors.NEON_CYAN, 2)
        
        # Title
        title = "◆ AI DRIVING ASSISTANT ◆"
        title_size = cv2.getTextSize(title, self.font_primary, 0.6, 1)[0]
        title_x = (self.width - title_size[0]) // 2
        self._draw_text_with_glow(img, title, (title_x, 35), 0.6, Colors.NEON_CYAN, 1, 
                                 Colors.ELECTRIC_BLUE, 2)
        
        # Time
        time_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(img, time_str, (20, 35), self.font_secondary, 0.6, Colors.SOFT_WHITE, 1)
        
        # Mode indicator
        mode = "◉ TEST MODE" if Config.USE_TEST_VIDEO else "◉ LIVE FEED"
        mode_color = Colors.ALERT_YELLOW if Config.USE_TEST_VIDEO else Colors.NEON_GREEN
        cv2.putText(img, mode, (self.width - 150, 35), self.font_secondary, 0.5, mode_color, 1)
    
    def _draw_bottom_dashboard(self, img: np.ndarray):
        """Draw bottom instruction dashboard."""
        dash_height = 110
        dash_y = self.height - dash_height
        
        # Background with gradient
        overlay = img.copy()
        for i in range(dash_height):
            alpha = i / dash_height
            color = Colors.interpolate(Colors.PURE_BLACK, Colors.NEAR_BLACK, alpha)
            cv2.line(overlay, (0, dash_y + i), (self.width, dash_y + i), color, 1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        
        # Top border
        cv2.line(img, (0, dash_y), (self.width, dash_y), Colors.NEON_CYAN, 1)
        
        # Corner accents
        accent_len = 150
        cv2.line(img, (0, dash_y), (accent_len, dash_y), Colors.NEON_CYAN, 2)
        cv2.line(img, (self.width - accent_len, dash_y), (self.width, dash_y), Colors.NEON_CYAN, 2)
        
        # Main instruction (large, centered)
        main_size = cv2.getTextSize(self.display_main, self.font_primary, 1.8, 2)[0]
        main_x = (self.width - main_size[0]) // 2
        main_y = dash_y + 55
        
        self._draw_text_with_glow(img, self.display_main, (main_x, main_y), 1.8, 
                                 self.display_color, 2, self.display_color, 3)
        
        # Sub instruction
        sub_size = cv2.getTextSize(self.display_sub, self.font_secondary, 0.7, 1)[0]
        sub_x = (self.width - sub_size[0]) // 2
        sub_y = dash_y + 85
        
        cv2.putText(img, self.display_sub, (sub_x, sub_y), self.font_secondary, 0.7, 
                   Colors.SOFT_WHITE, 1)
        
        # Side decorations
        cv2.line(img, (20, dash_y + 30), (20, dash_y + 80), Colors.NEON_CYAN, 2)
        cv2.line(img, (self.width - 20, dash_y + 30), (self.width - 20, dash_y + 80), Colors.NEON_CYAN, 2)
    
    def _draw_scan_lines(self, img: np.ndarray):
        """Draw subtle scan line effect."""
        # Moving scan line
        scan_y = self.scan_line_y % self.height
        
        # Main bright line
        overlay = img.copy()
        cv2.line(overlay, (0, scan_y), (self.width, scan_y), Colors.NEON_CYAN, 1)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        
        # Trail
        for i in range(1, 20):
            y = (scan_y - i * 3) % self.height
            alpha = 0.1 * (1 - i / 20)
            overlay = img.copy()
            cv2.line(overlay, (0, y), (self.width, y), Colors.NEON_CYAN, 1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Static subtle scan lines (every 4 pixels)
        for y in range(0, self.height, 4):
            cv2.line(img, (0, y), (self.width, y), Colors.PURE_BLACK, 1)
            cv2.addWeighted(img, 0.97, img, 0.03, 0, img)
    
    def _apply_vignette(self, img: np.ndarray) -> np.ndarray:
        """Apply vignette effect."""
        return (img * self.vignette).astype(np.uint8)
    
    def update_instruction(self, main: str, sub: str, color: Tuple, 
                          threat_level: ThreatLevel):
        """Update displayed instruction with smoothing."""
        # Immediate update for critical threats
        if threat_level == ThreatLevel.CRITICAL:
            self.display_main = main
            self.display_sub = sub
            self.display_color = color
            self.consistency_counter = 0
            self.last_raw_main = main
            return
        
        # Smoothing for other instructions
        if main == self.last_raw_main:
            self.consistency_counter += 1
        else:
            self.consistency_counter = 0
            self.last_raw_main = main
        
        if self.consistency_counter >= Config.INSTRUCTION_SMOOTHING_FRAMES:
            self.display_main = main
            self.display_sub = sub
            self.display_color = color
    
    def render(self, frame: np.ndarray, analysis: FrameAnalysis) -> np.ndarray:
        """Render complete HUD on frame."""
        self.frame_count += 1
        
        # Update animations
        self.crosshair_rotation += 0.02
        self.scan_line_y += 3
        self.radar_rotation += 0.05
        self.pulse_phase += 0.1
        
        # Create working copy
        hud_frame = frame.copy()
        
        # Draw lane overlay first (behind everything)
        # (This is done in the lane detector's draw_lanes method)
        
        # Draw detections
        for detection in analysis.detections:
            self._draw_object_info(hud_frame, detection)
        
        # Draw crosshair
        self._draw_animated_crosshair(hud_frame)
        
        # Draw radar
        self._draw_radar(hud_frame, analysis.detections)
        
        # Draw threat gauge
        self._draw_threat_gauge(hud_frame, analysis.overall_threat)
        
        # Draw telemetry panel
        self._draw_telemetry_panel(hud_frame, analysis.fps, 
                                   len(analysis.detections), analysis.lane_info)
        
        # Draw bars (on top)
        self._draw_top_bar(hud_frame)
        self._draw_bottom_dashboard(hud_frame)
        
        # Optional scan lines (subtle)
        # self._draw_scan_lines(hud_frame)
        
        # Apply vignette
        hud_frame = self._apply_vignette(hud_frame)
        
        return hud_frame


# ═══════════════════════════════════════════════════════════════════════════════
#                           DRIVING INTELLIGENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class DrivingIntelligence:
    """AI decision engine for driving instructions."""
    
    def __init__(self):
        self.last_instruction_time = time.time()
        self.instruction_history = deque(maxlen=30)
    
    def analyze(self, detections: List[Detection], lane_info: LaneInfo) -> Tuple[str, str, Tuple, ThreatLevel]:
        """Generate driving instruction based on analysis."""
        
        # Determine overall threat
        threat_levels = [d.threat_level for d in detections]
        if ThreatLevel.CRITICAL in threat_levels:
            overall_threat = ThreatLevel.CRITICAL
        elif ThreatLevel.HIGH in threat_levels:
            overall_threat = ThreatLevel.HIGH
        elif ThreatLevel.MEDIUM in threat_levels:
            overall_threat = ThreatLevel.MEDIUM
        elif ThreatLevel.LOW in threat_levels:
            overall_threat = ThreatLevel.LOW
        else:
            overall_threat = ThreatLevel.NONE
        
        # Find most critical detection
        critical_detections = [d for d in detections if d.threat_level == ThreatLevel.CRITICAL]
        high_detections = [d for d in detections if d.threat_level == ThreatLevel.HIGH]
        center_detections = [d for d in detections if d.position == "CENTER"]
        
        # Generate instruction based on situation
        main_text = "CRUISING"
        sub_text = "Path Clear"
        color = Colors.NEON_CYAN
        
        # CRITICAL: Immediate danger
        if critical_detections:
            det = critical_detections[0]
            if det.class_id in Config.PEDESTRIAN_CLASSES:
                main_text = "⚠ PEDESTRIAN ⚠"
                sub_text = f"BRAKE NOW - {det.distance_estimate}m ahead"
            elif det.class_id in Config.ANIMAL_CLASSES:
                main_text = "⚠ ANIMAL ⚠"
                sub_text = f"BRAKE NOW - {det.class_name} {det.distance_estimate}m"
            else:
                main_text = "⚠ BRAKE NOW ⚠"
                sub_text = f"Collision Risk - {det.distance_estimate}m"
            color = Colors.CRITICAL_RED
            return main_text, sub_text, color, overall_threat
        
        # HIGH: Need action soon
        if high_detections:
            det = high_detections[0]
            main_text = "SLOW DOWN"
            sub_text = f"{det.class_name} Ahead - {det.distance_estimate}m"
            color = Colors.WARNING_ORANGE
            return main_text, sub_text, color, overall_threat
        
        # Check for overtaking opportunities
        if center_detections:
            det = center_detections[0]
            left_clear = not any(d.position == "LEFT" for d in detections)
            right_clear = not any(d.position == "RIGHT" for d in detections)
            
            if det.threat_level == ThreatLevel.MEDIUM:
                if right_clear:
                    main_text = "OVERTAKE RIGHT"
                    sub_text = f"{det.class_name} ahead - Right lane clear"
                    color = Colors.NEON_GREEN
                elif left_clear:
                    main_text = "OVERTAKE LEFT"
                    sub_text = f"{det.class_name} ahead - Left lane clear"
                    color = Colors.NEON_GREEN
                else:
                    main_text = "MAINTAIN DISTANCE"
                    sub_text = "Both lanes occupied"
                    color = Colors.ALERT_YELLOW
                return main_text, sub_text, color, overall_threat
            
            elif det.threat_level == ThreatLevel.LOW:
                main_text = "TRAFFIC AHEAD"
                sub_text = f"{det.class_name} - {det.distance_estimate}m"
                color = Colors.LIME_GREEN
                return main_text, sub_text, color, overall_threat
        
        # Lane departure warning
        if lane_info.lane_departure:
            main_text = "↺ LANE DEPARTURE ↻"
            sub_text = lane_info.steering_suggestion
            color = Colors.ALERT_YELLOW
            return main_text, sub_text, color, overall_threat
        
        # Gentle lane correction
        if "STEER" in lane_info.steering_suggestion:
            main_text = "LANE CORRECTION"
            sub_text = lane_info.steering_suggestion
            color = Colors.LIME_GREEN
            return main_text, sub_text, color, overall_threat
        
        # All clear
        return main_text, sub_text, color, overall_threat


# ═══════════════════════════════════════════════════════════════════════════════
#                              MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class UltimateDrivingAssistant:
    """Main application orchestrator."""
    
    def __init__(self):
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║         ULTIMATE AI DRIVING ASSISTANT v3.0                   ║")
        print("║         Optimized for Indian Roads                           ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print()
        
        # Initialize video source
        print("[1/5] Initializing video source...")
        if Config.USE_TEST_VIDEO:
            print(f"      → Test Mode: {Config.TEST_VIDEO_PATH}")
            self.source = Config.TEST_VIDEO_PATH
        else:
            print(f"      → Live Mode: Camera {Config.LIVE_CAMERA_INDEX}")
            self.source = Config.LIVE_CAMERA_INDEX
        
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {self.source}")
        
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0 or math.isnan(self.fps):
            self.fps = 30.0
        
        print(f"      → Resolution: {self.frame_width}x{self.frame_height} @ {self.fps:.1f}fps")
        
        # Initialize components
        print("[2/5] Loading object detection engine...")
        self.detector = ObjectDetectionEngine(self.frame_width, self.frame_height)
        
        print("[3/5] Initializing lane detection system...")
        self.lane_detector = AdvancedLaneDetector(self.frame_width, self.frame_height)
        
        print("[4/5] Setting up HUD renderer...")
        self.hud = FuturisticHUD(self.frame_width, self.frame_height)
        
        print("[5/5] Initializing driving intelligence...")
        self.intelligence = DrivingIntelligence()
        
        # Output writer
        self.writer = None
        if Config.SAVE_OUTPUT:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                Config.OUTPUT_PATH, fourcc, self.fps, 
                (self.frame_width, self.frame_height)
            )
            print(f"      → Recording to: {Config.OUTPUT_PATH}")
        
        # Performance tracking
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        
        print()
        print("═" * 60)
        print("  System Ready! Press 'Q' to quit.")
        print("═" * 60)
        print()
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the entire pipeline."""
        start_time = time.time()
        
        # Object detection
        detections = self.detector.detect(frame)
        
        # Lane detection
        lane_info = self.lane_detector.detect(frame)
        
        # Draw lanes on frame
        frame_with_lanes = self.lane_detector.draw_lanes(frame, lane_info)
        
        # Generate driving instruction
        main_text, sub_text, color, overall_threat = self.intelligence.analyze(
            detections, lane_info
        )
        
        # Update HUD instruction
        self.hud.update_instruction(main_text, sub_text, color, overall_threat)
        
        # Calculate FPS
        process_time = time.time() - start_time
        current_fps = 1.0 / process_time if process_time > 0 else 0
        self.fps_history.append(current_fps)
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        # Create analysis object
        analysis = FrameAnalysis(
            detections=detections,
            lane_info=lane_info,
            overall_threat=overall_threat,
            frame_time=process_time,
            fps=avg_fps
        )
        
        # Render HUD
        final_frame = self.hud.render(frame_with_lanes, analysis)
        
        return final_frame
    
    def run(self):
        """Main processing loop."""
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    if Config.USE_TEST_VIDEO:
                        # Loop test video
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("Video stream ended.")
                        break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Write output
                if self.writer:
                    self.writer.write(processed_frame)
                
                # Display
                cv2.imshow('AI Driving Assistant | Press Q to Exit', processed_frame)
                
                self.frame_count += 1
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\nExiting...")
                    break
                elif key == ord('r') or key == ord('R'):
                    # Reset to beginning for test video
                    if Config.USE_TEST_VIDEO:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        print("Video reset to beginning.")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        
        if self.writer:
            self.writer.release()
            print(f"  → Output saved to: {Config.OUTPUT_PATH}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"  → Processed {self.frame_count} frames")
        print("\nGoodbye!")


# ═══════════════════════════════════════════════════════════════════════════════
#                              ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        assistant = UltimateDrivingAssistant()
        assistant.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()