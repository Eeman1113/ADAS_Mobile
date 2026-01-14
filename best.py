#!/usr/bin/env python3


import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum
import warnings
import torch

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    # ─── Video Source ───
    USE_TEST_VIDEO: bool = True
    TEST_VIDEO_PATH: str = "test3.mp4"
    LIVE_CAMERA_INDEX: int = 0
    
    # ─── Output ───
    SAVE_OUTPUT: bool = True
    OUTPUT_PATH: str = "output_final.mp4"
    
    # ─── Model ───
    YOLO_MODEL: str = "yolov8s-seg.pt"
    CONFIDENCE_THRESHOLD: float = 0.40
    
    # ─── Detection Zones ───
    LEFT_ZONE_END: float = 0.38
    RIGHT_ZONE_START: float = 0.62
    
    # ─── Object Classes ───
    ALL_CLASSES: List[int] = [0, 1, 2, 3, 5, 7, 16, 17, 18, 19]
    PERSON_CLASSES: List[int] = [0]
    ANIMAL_CLASSES: List[int] = [16, 17, 18, 19]
    
    # ─── Threat Thresholds ───
    CRITICAL_AREA: float = 0.14
    HIGH_AREA: float = 0.06
    MEDIUM_AREA: float = 0.02
    
    # ─── Passing Logic ───
    ENABLE_PASS_SUGGESTIONS: bool = True
    MIN_PASS_DISTANCE: float = 15.0
    REQUIRE_LANE_DETECTION: bool = True
    
    # ─── Filtering ───
    SKY_CUTOFF: float = 0.25
    MIN_AREA: float = 0.001
    MAX_AREA: float = 0.80
    
    # ─── Visual ───
    MASK_ALPHA: float = 0.40


# ═══════════════════════════════════════════════════════════════════════════════
#                              UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_device():
    """Get best available compute device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Colors:
    """Color definitions (BGR format)."""
    # UI Colors
    CYAN = (255, 220, 100)
    WHITE = (255, 255, 255)
    GREEN = (100, 230, 100)
    YELLOW = (80, 220, 255)
    ORANGE = (80, 180, 255)
    RED = (80, 80, 255)
    
    # Segmentation Colors
    SEG_VEHICLE = (180, 140, 80)
    SEG_PERSON = (200, 100, 180)
    SEG_ANIMAL = (100, 180, 200)
    
    # Neutral Colors
    LIGHT_GRAY = (200, 200, 200)
    GRAY = (140, 140, 140)
    DARK_GRAY = (60, 60, 60)
    NEAR_BLACK = (20, 20, 20)
    BLACK = (0, 0, 0)
    
    # Road Colors
    ROAD_SURFACE = (45, 45, 50)
    ROAD_EDGE = (70, 70, 80)


class ThreatLevel(Enum):
    """Threat level classification."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Detection:
    """Detection data structure."""
    bbox: Tuple[int, int, int, int]
    mask: Optional[np.ndarray]
    class_id: int
    label: str
    confidence: float
    center: Tuple[int, int]
    area_ratio: float
    zone: str
    distance: float
    threat: ThreatLevel
    is_confirmed: bool = True
    track_id: int = -1
    frames_seen: int = 1


@dataclass
class LaneInfo:
    """Lane detection data."""
    left_line: Optional[List[Tuple[int, int]]] = None
    right_line: Optional[List[Tuple[int, int]]] = None
    left_lane_found: bool = False
    right_lane_found: bool = False
    center_offset: float = 0.0
    valid: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
#                              OBJECT TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class ObjectTracker:
    """Simple centroid-based object tracker."""
    
    def __init__(self):
        self.objects: Dict[int, dict] = {}
        self.next_id = 0
    
    def update(self, detections: List[Detection]) -> List[Detection]:
        """Update tracks with new detections."""
        current = [(d.center, d.class_id, i) for i, d in enumerate(detections)]
        
        if not self.objects:
            for center, cls_id, idx in current:
                self.objects[self.next_id] = {
                    'center': center, 'class': cls_id, 'frames': 1, 'missing': 0
                }
                detections[idx].track_id = self.next_id
                detections[idx].frames_seen = 1
                detections[idx].is_confirmed = False
                self.next_id += 1
            return detections
        
        used_det, used_track = set(), set()
        matches = []
        
        for tid, track in self.objects.items():
            for center, cls_id, idx in current:
                dist = math.hypot(center[0] - track['center'][0], 
                                 center[1] - track['center'][1])
                if dist < 150:
                    matches.append((dist, tid, idx, cls_id))
        
        matches.sort(key=lambda x: x[0])
        
        for dist, tid, idx, cls_id in matches:
            if tid in used_track or idx in used_det:
                continue
            
            self.objects[tid].update({
                'center': detections[idx].center,
                'frames': self.objects[tid]['frames'] + 1,
                'missing': 0,
                'class': cls_id
            })
            detections[idx].track_id = tid
            detections[idx].frames_seen = self.objects[tid]['frames']
            detections[idx].is_confirmed = self.objects[tid]['frames'] >= 2
            used_track.add(tid)
            used_det.add(idx)
        
        for center, cls_id, idx in current:
            if idx not in used_det:
                self.objects[self.next_id] = {
                    'center': center, 'class': cls_id, 'frames': 1, 'missing': 0
                }
                detections[idx].track_id = self.next_id
                detections[idx].frames_seen = 1
                detections[idx].is_confirmed = False
                self.next_id += 1
        
        expired = []
        for tid in self.objects:
            if tid not in used_track:
                self.objects[tid]['missing'] += 1
                if self.objects[tid]['missing'] > 8:
                    expired.append(tid)
        
        for tid in expired:
            del self.objects[tid]
        
        return detections


# ═══════════════════════════════════════════════════════════════════════════════
#                              DETECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class DetectionEngine:
    """YOLO-based segmentation detection engine."""
    
    def __init__(self, width: int, height: int):
        print("  Loading YOLO segmentation model...")
        self.device = get_device()
        print(f"  Device: {self.device.upper()}")
        
        self.model = YOLO(Config.YOLO_MODEL)
        self.w = width
        self.h = height
        self.frame_area = width * height
        self.tracker = ObjectTracker()
        
        self.left_boundary = int(width * Config.LEFT_ZONE_END)
        self.right_boundary = int(width * Config.RIGHT_ZONE_START)
        self.sky_line = int(height * Config.SKY_CUTOFF)
        
        self.labels = {
            0: "PERSON", 1: "BICYCLE", 2: "CAR", 3: "BIKE",
            5: "BUS", 7: "TRUCK", 16: "DOG", 17: "HORSE",
            18: "SHEEP", 19: "COW"
        }
        
        self.real_widths = {
            0: 0.5, 1: 0.65, 2: 1.8, 3: 0.8, 5: 2.5, 7: 2.4,
            16: 0.35, 17: 0.6, 18: 0.45, 19: 0.85
        }
    
    def _get_zone(self, cx: int) -> str:
        """Determine zone based on x position."""
        if cx < self.left_boundary:
            return "LEFT"
        elif cx > self.right_boundary:
            return "RIGHT"
        return "CENTER"
    
    def _estimate_distance(self, width_px: int, class_id: int) -> float:
        """Estimate distance using pinhole camera model."""
        real_width = self.real_widths.get(class_id, 1.5)
        if width_px > 0:
            return max(1.0, min(100.0, (real_width * 750) / width_px))
        return 99.0
    
    def _compute_threat(self, zone: str, area_ratio: float, class_id: int) -> ThreatLevel:
        """Compute threat level based on zone and size."""
        if zone == "CENTER":
            if area_ratio > Config.CRITICAL_AREA:
                return ThreatLevel.CRITICAL
            elif area_ratio > Config.HIGH_AREA:
                return ThreatLevel.HIGH
            elif area_ratio > Config.MEDIUM_AREA:
                return ThreatLevel.MEDIUM
            elif area_ratio > 0.008:
                return ThreatLevel.LOW
        else:
            if area_ratio > Config.CRITICAL_AREA:
                return ThreatLevel.HIGH
            elif area_ratio > Config.HIGH_AREA:
                return ThreatLevel.MEDIUM
            elif area_ratio > Config.MEDIUM_AREA:
                return ThreatLevel.LOW
        return ThreatLevel.NONE
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on frame."""
        results = self.model(
            frame, verbose=False, conf=Config.CONFIDENCE_THRESHOLD,
            classes=Config.ALL_CLASSES, device=self.device, retina_masks=True
        )
        
        detections = []
        
        for result in results:
            if result.boxes is None:
                continue
            
            masks = result.masks.data.cpu().numpy() if result.masks is not None else None
            
            for i, box in enumerate(result.boxes):
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                w_box, h_box = x2 - x1, y2 - y1
                area_ratio = (w_box * h_box) / self.frame_area
                
                # Filter sky detections
                if cy < self.sky_line and y2 < self.sky_line:
                    continue
                
                # Filter by size
                if area_ratio < Config.MIN_AREA or area_ratio > Config.MAX_AREA:
                    continue
                
                # Get mask
                mask = None
                if masks is not None and i < len(masks):
                    mask = cv2.resize(masks[i], (self.w, self.h))
                    mask = (mask > 0.5).astype(np.uint8)
                
                zone = self._get_zone(cx)
                distance = self._estimate_distance(w_box, cls)
                threat = self._compute_threat(zone, area_ratio, cls)
                
                # Boost threat for pedestrians/animals in center
                if cls in Config.PERSON_CLASSES + Config.ANIMAL_CLASSES:
                    if zone == "CENTER" and cy > self.h * 0.4:
                        threat = ThreatLevel(min(threat.value + 1, 4))
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    mask=mask,
                    class_id=cls,
                    label=self.labels.get(cls, "OBJECT"),
                    confidence=conf,
                    center=(cx, cy),
                    area_ratio=area_ratio,
                    zone=zone,
                    distance=round(distance, 1),
                    threat=threat
                ))
        
        return self.tracker.update(detections)


# ═══════════════════════════════════════════════════════════════════════════════
#                              LANE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class LaneDetector:
    """Simple lane detection using Hough transform."""
    
    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self.history_left = deque(maxlen=10)
        self.history_right = deque(maxlen=10)
        self.left_confidence = deque(maxlen=15)
        self.right_confidence = deque(maxlen=15)
    
    def detect(self, frame: np.ndarray) -> LaneInfo:
        """Detect lane lines in frame."""
        info = LaneInfo()
        h, w = frame.shape[:2]
        roi_y = int(h * 0.55)
        roi = frame[roi_y:h, :]
        
        # Preprocessing
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # ROI mask
        mask = np.zeros_like(edges)
        poly = np.array([[(0, edges.shape[0]), (int(w * 0.15), 0),
                         (int(w * 0.85), 0), (w, edges.shape[0])]], np.int32)
        cv2.fillPoly(mask, poly, 255)
        edges = cv2.bitwise_and(edges, mask)
        
        # Hough lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=80)
        
        left_lines, right_lines = [], []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.3 or abs(slope) > 3:
                    continue
                
                y1 += roi_y
                y2 += roi_y
                
                if slope < 0:
                    left_lines.append([x1, y1, x2, y2])
                else:
                    right_lines.append([x1, y1, x2, y2])
        
        # Track lane confidence
        self.left_confidence.append(1 if len(left_lines) >= 2 else 0)
        self.right_confidence.append(1 if len(right_lines) >= 2 else 0)
        
        info.left_lane_found = sum(self.left_confidence) / max(1, len(self.left_confidence)) > 0.5
        info.right_lane_found = sum(self.right_confidence) / max(1, len(self.right_confidence)) > 0.5
        
        # Average and smooth left lane
        if left_lines:
            avg = np.mean(left_lines, axis=0).astype(int)
            self.history_left.append(avg)
        
        if self.history_left:
            avg = np.mean(list(self.history_left), axis=0).astype(int)
            info.left_line = [(avg[0], avg[1]), (avg[2], avg[3])]
        
        # Average and smooth right lane
        if right_lines:
            avg = np.mean(right_lines, axis=0).astype(int)
            self.history_right.append(avg)
        
        if self.history_right:
            avg = np.mean(list(self.history_right), axis=0).astype(int)
            info.right_line = [(avg[0], avg[1]), (avg[2], avg[3])]
        
        info.valid = info.left_line is not None or info.right_line is not None
        
        # Calculate center offset
        if info.left_line and info.right_line:
            lx = (info.left_line[0][0] + info.left_line[1][0]) / 2
            rx = (info.right_line[0][0] + info.right_line[1][0]) / 2
            info.center_offset = (w / 2 - (lx + rx) / 2) / (w / 2)
        
        return info


# ═══════════════════════════════════════════════════════════════════════════════
#                              DRIVING ADVISOR
# ═══════════════════════════════════════════════════════════════════════════════

class DrivingAdvisor:
    """Context-aware driving advice generator."""
    
    def __init__(self, frame_width: int):
        self.w = frame_width
    
    def _find_primary_obstacle(self, detections: List[Detection]) -> Optional[Detection]:
        """Find the most important obstacle."""
        confirmed = [d for d in detections if d.is_confirmed]
        if not confirmed:
            return None
        
        confirmed.sort(key=lambda d: (-d.threat.value, -d.area_ratio, d.distance))
        return confirmed[0]
    
    def _is_side_clear(self, side: str, detections: List[Detection]) -> bool:
        """Check if a side is clear."""
        confirmed = [d for d in detections if d.is_confirmed]
        side_objects = [d for d in confirmed if d.zone == side]
        return len(side_objects) == 0
    
    def _can_suggest_pass(self, obstacle: Detection, target_side: str,
                         detections: List[Detection], lanes: LaneInfo) -> Tuple[bool, str]:
        """Determine if passing is safe."""
        
        if not Config.ENABLE_PASS_SUGGESTIONS:
            return False, "disabled"
        
        # Never pass toward the obstacle
        if obstacle.zone == target_side:
            return False, "obstacle_on_side"
        
        # Check distance
        if obstacle.distance < Config.MIN_PASS_DISTANCE:
            return False, "too_close"
        
        # Check if side is clear
        if not self._is_side_clear(target_side, detections):
            return False, "side_occupied"
        
        # Check lane detection requirement
        if Config.REQUIRE_LANE_DETECTION:
            if target_side == "LEFT" and not lanes.left_lane_found:
                return False, "no_left_lane"
            if target_side == "RIGHT" and not lanes.right_lane_found:
                return False, "no_right_lane"
        
        return True, "clear"
    
    def analyze(self, detections: List[Detection], lanes: LaneInfo):
        """Generate driving advice."""
        
        confirmed = [d for d in detections if d.is_confirmed]
        
        if not confirmed:
            return "CLEAR", "Road ahead is clear", Colors.CYAN, ThreatLevel.NONE
        
        obstacle = self._find_primary_obstacle(confirmed)
        if not obstacle:
            return "CLEAR", "Road ahead is clear", Colors.CYAN, ThreatLevel.NONE
        
        max_threat = max((d.threat for d in confirmed), default=ThreatLevel.NONE,
                        key=lambda x: x.value)
        
        # Critical threat
        if obstacle.threat == ThreatLevel.CRITICAL:
            if obstacle.class_id in Config.PERSON_CLASSES + Config.ANIMAL_CLASSES:
                return "STOP", f"{obstacle.label} ON ROAD!", Colors.RED, max_threat
            return "BRAKE", f"Obstacle {obstacle.distance:.0f}m!", Colors.RED, max_threat
        
        # High threat
        if obstacle.threat == ThreatLevel.HIGH:
            return "SLOW DOWN", f"{obstacle.label} ahead - {obstacle.distance:.0f}m", Colors.ORANGE, max_threat
        
        # Side traffic
        if obstacle.zone == "LEFT":
            return "TRAFFIC LEFT", f"{obstacle.label} on left - {obstacle.distance:.0f}m", Colors.YELLOW, max_threat
        
        if obstacle.zone == "RIGHT":
            return "TRAFFIC RIGHT", f"{obstacle.label} on right - {obstacle.distance:.0f}m", Colors.YELLOW, max_threat
        
        # Center obstacle - check passing
        if obstacle.zone == "CENTER":
            can_left, _ = self._can_suggest_pass(obstacle, "LEFT", confirmed, lanes)
            can_right, _ = self._can_suggest_pass(obstacle, "RIGHT", confirmed, lanes)
            
            if can_left:
                return "PASS LEFT", f"{obstacle.label} ahead - left clear", Colors.GREEN, max_threat
            elif can_right:
                return "PASS RIGHT", f"{obstacle.label} ahead - right clear", Colors.GREEN, max_threat
            else:
                if obstacle.distance < Config.MIN_PASS_DISTANCE:
                    return "MAINTAIN GAP", f"Following {obstacle.label} - {obstacle.distance:.0f}m", Colors.YELLOW, max_threat
                else:
                    return "FOLLOW", f"Behind {obstacle.label} - no clear lane", Colors.YELLOW, max_threat
        
        return "CAUTION", f"{obstacle.label} detected", Colors.YELLOW, max_threat


# ═══════════════════════════════════════════════════════════════════════════════
#                              HUD RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

class HUDRenderer:
    """Clean HUD renderer with segmentation overlays."""
    
    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self.frame_count = 0
        self.main_text = "STARTING"
        self.sub_text = "Initializing..."
        self.main_color = Colors.CYAN
        self.stable_frames = 0
        self.last_main = ""
    
    def _get_color(self, det: Detection) -> Tuple[int, int, int]:
        """Get color for detection based on type and threat."""
        if det.class_id in Config.PERSON_CLASSES:
            return Colors.SEG_PERSON
        elif det.class_id in Config.ANIMAL_CLASSES:
            return Colors.SEG_ANIMAL
        
        threat_colors = {
            ThreatLevel.CRITICAL: Colors.RED,
            ThreatLevel.HIGH: Colors.ORANGE,
            ThreatLevel.MEDIUM: Colors.YELLOW,
            ThreatLevel.LOW: Colors.GREEN,
            ThreatLevel.NONE: Colors.SEG_VEHICLE
        }
        return threat_colors.get(det.threat, Colors.SEG_VEHICLE)
    
    def _draw_segmentation(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw segmentation masks."""
        if not detections:
            return frame
        
        overlay = frame.copy()
        edge_layer = np.zeros_like(frame)
        
        for det in detections:
            color = self._get_color(det)
            
            if det.mask is not None:
                mask_bool = det.mask > 0
                overlay[mask_bool] = color
                
                contours, _ = cv2.findContours(det.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(edge_layer, contours, -1, color, 4)
                    cv2.drawContours(edge_layer, contours, -1, Colors.WHITE, 1)
        
        result = cv2.addWeighted(overlay, Config.MASK_ALPHA, frame, 1 - Config.MASK_ALPHA, 0)
        result = cv2.addWeighted(edge_layer, 0.5, result, 1, 0)
        return result
    
    def _draw_labels(self, frame: np.ndarray, detections: List[Detection]):
        """Draw floating labels for detections."""
        for det in detections:
            if det.area_ratio < 0.005 and det.threat == ThreatLevel.NONE:
                continue
            
            x1, y1, x2, y2 = det.bbox
            cx, cy = det.center
            color = self._get_color(det)
            
            label = det.label
            dist_text = f"{det.distance:.0f}m"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.48
            
            lbl_sz = cv2.getTextSize(label, font, scale, 1)[0]
            dist_sz = cv2.getTextSize(dist_text, font, scale, 1)[0]
            total_w = lbl_sz[0] + dist_sz[0] + 10
            
            tx = cx - total_w // 2
            ty = y1 - 10 if y1 > 60 else y2 + 20
            
            # Background
            cv2.rectangle(frame, (tx - 4, ty - lbl_sz[1] - 4),
                         (tx + total_w + 4, ty + 6), Colors.NEAR_BLACK, -1)
            cv2.rectangle(frame, (tx - 4, ty - lbl_sz[1] - 4),
                         (tx + total_w + 4, ty + 6), color, 1)
            
            # Text
            cv2.putText(frame, label, (tx, ty), font, scale, color, 1, cv2.LINE_AA)
            cv2.putText(frame, dist_text, (tx + lbl_sz[0] + 8, ty),
                       font, scale, Colors.LIGHT_GRAY, 1, cv2.LINE_AA)
            
            # Connection line
            line_y = ty + 6 if ty > cy else ty - lbl_sz[1] - 4
            cv2.line(frame, (cx, line_y), (cx, cy), color, 1)
            cv2.circle(frame, (cx, cy), 3, color, -1)
    
    def _draw_road_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw road perspective overlay."""
        h, w = self.h, self.w
        overlay = frame.copy()
        
        road_h = int(h * 0.22)
        road_top = h - road_h
        vp_x, vp_y = w // 2, road_top
        
        pts = np.array([[vp_x, vp_y], [int(w * 0.92), h], [int(w * 0.08), h]], np.int32)
        road_layer = frame.copy()
        cv2.fillPoly(road_layer, [pts], Colors.ROAD_SURFACE)
        overlay = cv2.addWeighted(road_layer, 0.2, overlay, 0.8, 0)
        
        cv2.line(overlay, (vp_x, vp_y), (int(w * 0.08), h), Colors.ROAD_EDGE, 2)
        cv2.line(overlay, (vp_x, vp_y), (int(w * 0.92), h), Colors.ROAD_EDGE, 2)
        
        # Center dashes
        for i in range(5):
            t1, t2 = i / 5, (i + 0.3) / 5
            y1 = int(vp_y + (h - vp_y) * t1)
            y2 = int(vp_y + (h - vp_y) * t2)
            thick = max(1, int(1 + t1 * 2))
            alpha = 0.15 + t1 * 0.4
            
            tmp = overlay.copy()
            cv2.line(tmp, (vp_x, y1), (vp_x, y2), Colors.LIGHT_GRAY, thick)
            overlay = cv2.addWeighted(tmp, alpha, overlay, 1 - alpha, 0)
        
        return overlay
    
    def _draw_lanes(self, frame: np.ndarray, lanes: LaneInfo) -> np.ndarray:
        """Draw detected lane lines."""
        if not lanes.valid:
            return frame
        
        overlay = frame.copy()
        
        if lanes.left_line:
            cv2.line(overlay, lanes.left_line[0], lanes.left_line[1], Colors.GREEN, 3)
        if lanes.right_line:
            cv2.line(overlay, lanes.right_line[0], lanes.right_line[1], Colors.GREEN, 3)
        
        return cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    
    def _draw_top_bar(self, frame: np.ndarray, fps: float, obj_count: int, lanes: LaneInfo):
        """Draw top status bar."""
        bar_h = 40
        
        # Background gradient
        for y in range(bar_h):
            alpha = 1 - (y / bar_h) ** 2
            cv2.line(frame, (0, y), (self.w, y),
                    tuple(int(c * alpha) for c in Colors.NEAR_BLACK), 1)
        
        cv2.line(frame, (0, bar_h), (self.w, bar_h), Colors.CYAN, 1)
        
        # Time
        cv2.putText(frame, datetime.now().strftime("%H:%M"), (15, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, Colors.WHITE, 1, cv2.LINE_AA)
        
        # Title
        title = "AI DRIVING ASSISTANT"
        tsz = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(frame, title, ((self.w - tsz[0]) // 2, 27),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.CYAN, 1, cv2.LINE_AA)
        
        # Lane status
        left_ind = "L" if lanes.left_lane_found else "-"
        right_ind = "R" if lanes.right_lane_found else "-"
        cv2.putText(frame, f"LANE:{left_ind}|{right_ind}", (self.w - 200, 27),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, Colors.GRAY, 1, cv2.LINE_AA)
        
        # Object count
        obj_color = Colors.WHITE if obj_count > 0 else Colors.GRAY
        cv2.putText(frame, f"OBJ:{obj_count}", (self.w - 120, 27),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, obj_color, 1, cv2.LINE_AA)
        
        # FPS
        fps_color = Colors.GREEN if fps > 20 else Colors.YELLOW if fps > 12 else Colors.RED
        cv2.putText(frame, f"{int(fps)}FPS", (self.w - 55, 27),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, fps_color, 1, cv2.LINE_AA)
    
    def _draw_bottom_panel(self, frame: np.ndarray):
        """Draw bottom instruction panel."""
        panel_h = 75
        panel_y = self.h - panel_h
        
        # Background gradient
        for y in range(panel_h):
            alpha = (y / panel_h) ** 1.5
            cv2.line(frame, (0, panel_y + y), (self.w, panel_y + y),
                    tuple(int(c * alpha) for c in Colors.NEAR_BLACK), 1)
        
        # Top line
        cv2.line(frame, (0, panel_y), (self.w, panel_y), Colors.DARK_GRAY, 1)
        cv2.line(frame, (0, panel_y), (60, panel_y), self.main_color, 2)
        cv2.line(frame, (self.w - 60, panel_y), (self.w, panel_y), self.main_color, 2)
        
        # Main text
        font = cv2.FONT_HERSHEY_DUPLEX
        msz = cv2.getTextSize(self.main_text, font, 1.1, 2)[0]
        mx = (self.w - msz[0]) // 2
        my = panel_y + 38
        
        cv2.putText(frame, self.main_text, (mx + 2, my + 2), font, 1.1, Colors.BLACK, 2, cv2.LINE_AA)
        cv2.putText(frame, self.main_text, (mx, my), font, 1.1, self.main_color, 2, cv2.LINE_AA)
        
        # Sub text
        ssz = cv2.getTextSize(self.sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.putText(frame, self.sub_text, ((self.w - ssz[0]) // 2, panel_y + 58),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, Colors.LIGHT_GRAY, 1, cv2.LINE_AA)
    
    def _draw_side_indicators(self, frame: np.ndarray, detections: List[Detection]):
        """Draw side traffic indicators."""
        left_n = sum(1 for d in detections if d.zone == "LEFT" and d.is_confirmed)
        right_n = sum(1 for d in detections if d.zone == "RIGHT" and d.is_confirmed)
        
        mid_y = self.h // 2
        
        if left_n > 0:
            pts = np.array([[12, mid_y], [28, mid_y - 14], [28, mid_y + 14]], np.int32)
            cv2.fillPoly(frame, [pts], Colors.YELLOW)
            cv2.putText(frame, str(left_n), (32, mid_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, Colors.YELLOW, 1, cv2.LINE_AA)
        
        if right_n > 0:
            pts = np.array([[self.w - 12, mid_y], [self.w - 28, mid_y - 14],
                           [self.w - 28, mid_y + 14]], np.int32)
            cv2.fillPoly(frame, [pts], Colors.YELLOW)
            cv2.putText(frame, str(right_n), (self.w - 45, mid_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, Colors.YELLOW, 1, cv2.LINE_AA)
    
    def _draw_minimap(self, frame: np.ndarray, detections: List[Detection]):
        """Draw radar minimap."""
        mw, mh = 85, 68
        mx, my = self.w - mw - 10, 48
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (mx, my), (mx + mw, my + mh), Colors.NEAR_BLACK, -1)
        frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), Colors.DARK_GRAY, 1)
        
        # Road shape
        pts = np.array([[mx + mw // 2, my + 5], [mx + mw - 10, my + mh - 5],
                       [mx + 10, my + mh - 5]], np.int32)
        cv2.polylines(frame, [pts], True, Colors.ROAD_EDGE, 1)
        
        # Self indicator
        cx, cy = mx + mw // 2, my + mh - 10
        cv2.fillPoly(frame, [np.array([[cx, cy - 5], [cx - 4, cy + 3], [cx + 4, cy + 3]])], Colors.CYAN)
        
        # Other objects
        for det in detections:
            if not det.is_confirmed:
                continue
            
            rx = det.center[0] / self.w
            ry = 1 - (det.center[1] / self.h)
            
            dx = int(mx + rx * mw)
            dy = int(my + mh - 8 - ry * (mh - 18))
            
            dx = max(mx + 4, min(mx + mw - 4, dx))
            dy = max(my + 4, min(my + mh - 4, dy))
            
            cv2.circle(frame, (dx, dy), 3, self._get_color(det), -1)
    
    def _draw_threat_badge(self, frame: np.ndarray, threat: ThreatLevel):
        """Draw threat level badge."""
        if threat == ThreatLevel.NONE:
            return
        
        labels = {
            ThreatLevel.LOW: ("LOW", Colors.GREEN),
            ThreatLevel.MEDIUM: ("MEDIUM", Colors.YELLOW),
            ThreatLevel.HIGH: ("HIGH", Colors.ORANGE),
            ThreatLevel.CRITICAL: ("CRITICAL", Colors.RED)
        }
        
        text, color = labels.get(threat, ("", Colors.WHITE))
        x, y = 15, 52
        
        # Pulse for critical
        if threat == ThreatLevel.CRITICAL and int(self.frame_count * 0.15) % 2:
            cv2.rectangle(frame, (x - 4, y - 14), (x + 72, y + 6), Colors.RED, 2)
        
        cv2.rectangle(frame, (x, y - 12), (x + 68, y + 4), Colors.NEAR_BLACK, -1)
        cv2.rectangle(frame, (x, y - 12), (x + 68, y + 4), color, 1)
        cv2.circle(frame, (x + 10, y - 3), 4, color, -1)
        cv2.putText(frame, text, (x + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1, cv2.LINE_AA)
    
    def _draw_crosshair(self, frame: np.ndarray):
        """Draw center crosshair."""
        cx, cy = self.w // 2, self.h // 2 - 20
        size, gap = 15, 5
        
        cv2.line(frame, (cx - size, cy), (cx - gap, cy), Colors.CYAN, 1)
        cv2.line(frame, (cx + gap, cy), (cx + size, cy), Colors.CYAN, 1)
        cv2.line(frame, (cx, cy - size), (cx, cy - gap), Colors.CYAN, 1)
        cv2.line(frame, (cx, cy + gap), (cx, cy + size), Colors.CYAN, 1)
        cv2.circle(frame, (cx, cy), 2, Colors.CYAN, -1)
    
    def update_text(self, main: str, sub: str, color: Tuple, threat: ThreatLevel):
        """Update display text with smoothing."""
        if threat == ThreatLevel.CRITICAL:
            self.main_text = main
            self.sub_text = sub
            self.main_color = color
            self.stable_frames = 0
            self.last_main = main
            return
        
        if main == self.last_main:
            self.stable_frames += 1
        else:
            self.stable_frames = 0
            self.last_main = main
        
        if self.stable_frames >= 4:
            self.main_text = main
            self.sub_text = sub
            self.main_color = color
    
    def render(self, frame: np.ndarray, detections: List[Detection],
               lanes: LaneInfo, fps: float) -> np.ndarray:
        """Render complete HUD."""
        self.frame_count += 1
        
        # Draw layers
        frame = self._draw_lanes(frame, lanes)
        frame = self._draw_segmentation(frame, detections)
        frame = self._draw_road_overlay(frame)
        self._draw_labels(frame, detections)
        self._draw_crosshair(frame)
        self._draw_side_indicators(frame, detections)
        self._draw_minimap(frame, detections)
        
        # Threat badge
        confirmed = [d for d in detections if d.is_confirmed]
        max_threat = max((d.threat for d in confirmed), default=ThreatLevel.NONE,
                        key=lambda x: x.value)
        self._draw_threat_badge(frame, max_threat)
        
        # UI panels
        self._draw_top_bar(frame, fps, len(confirmed), lanes)
        self._draw_bottom_panel(frame)
        
        return frame


# ═══════════════════════════════════════════════════════════════════════════════
#                              MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class DrivingAssistant:
    """Main application class."""
    
    def __init__(self):
        print("\n" + "=" * 60)
        print("   AI DRIVING ASSISTANT v8.1 - FINAL")
        print("   Smart Detection - Context-Aware Logic")
        print("=" * 60 + "\n")
        
        # Open video source
        source = Config.TEST_VIDEO_PATH if Config.USE_TEST_VIDEO else Config.LIVE_CAMERA_INDEX
        print(f"[1/4] Opening: {source}")
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"      Resolution: {self.w}x{self.h} @ {self.fps:.0f}fps")
        
        # Initialize components
        print("\n[2/4] Loading detection engine...")
        self.detector = DetectionEngine(self.w, self.h)
        
        print("\n[3/4] Initializing lane detection...")
        self.lane_detector = LaneDetector(self.w, self.h)
        
        print("\n[4/4] Setting up HUD...")
        self.hud = HUDRenderer(self.w, self.h)
        self.advisor = DrivingAdvisor(self.w)
        
        # Output writer
        self.writer = None
        if Config.SAVE_OUTPUT:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(Config.OUTPUT_PATH, fourcc, self.fps, (self.w, self.h))
            print(f"\n      Recording to: {Config.OUTPUT_PATH}")
        
        self.fps_buffer = deque(maxlen=30)
        
        print("\n" + "=" * 60)
        print("   Ready! Controls:")
        print("   [Q] Quit    [R] Restart    [Space] Pause")
        print("=" * 60 + "\n")
    
    def run(self):
        """Main processing loop."""
        paused = False
        
        try:
            while True:
                if not paused:
                    t0 = time.time()
                    
                    ret, frame = self.cap.read()
                    if not ret:
                        if Config.USE_TEST_VIDEO:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        break
                    
                    # Process frame
                    detections = self.detector.detect(frame)
                    lanes = self.lane_detector.detect(frame)
                    
                    # Get advice
                    main, sub, color, threat = self.advisor.analyze(detections, lanes)
                    self.hud.update_text(main, sub, color, threat)
                    
                    # Calculate FPS
                    dt = time.time() - t0
                    self.fps_buffer.append(1 / dt if dt > 0 else 0)
                    avg_fps = sum(self.fps_buffer) / len(self.fps_buffer)
                    
                    # Render
                    output = self.hud.render(frame, detections, lanes, avg_fps)
                    
                    # Save
                    if self.writer:
                        self.writer.write(output)
                    
                    # Display
                    cv2.imshow("AI Driving Assistant", output)
                
                # Handle keyboard
                key = cv2.waitKey(1 if not paused else 100) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('r') or key == ord('R'):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    print("Video restarted")
                elif key == ord(' '):
                    paused = not paused
                    print("Paused" if paused else "Playing")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        print("\nCleaning up...")
        if self.writer:
            self.writer.release()
            print(f"  Saved: {Config.OUTPUT_PATH}")
        self.cap.release()
        cv2.destroyAllWindows()
        print("  Done!\n")


# ═══════════════════════════════════════════════════════════════════════════════
#                              ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        app = DrivingAssistant()
        app.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
