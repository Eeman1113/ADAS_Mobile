#!/usr/bin/env python3


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

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    # Video Source
    USE_TEST_VIDEO: bool = True
    TEST_VIDEO_PATH: str = "test.mp4"
    LIVE_CAMERA_INDEX: int = 0
    
    # Output
    SAVE_OUTPUT: bool = True
    OUTPUT_PATH: str = "output_v9.mp4"
    
    # Model
    YOLO_MODEL: str = "yolov8s-seg.pt"
    CONFIDENCE: float = 0.35
    
    # Object Classes
    ALL_CLASSES: List[int] = [0, 1, 2, 3, 5, 7, 16, 17, 18, 19]
    VEHICLE_CLASSES: List[int] = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
    PERSON_CLASSES: List[int] = [0]
    ANIMAL_CLASSES: List[int] = [16, 17, 18, 19]
    
    # Detection
    SKY_CUTOFF: float = 0.28
    MIN_AREA: float = 0.0008
    MAX_AREA: float = 0.75
    
    # Our Lane - WIDER definition (25% to 75% of frame)
    OUR_LANE_LEFT: float = 0.22
    OUR_LANE_RIGHT: float = 0.78
    
    # Threat Distances (meters)
    CRITICAL_DISTANCE: float = 5.0
    CLOSE_DISTANCE: float = 12.0
    MEDIUM_DISTANCE: float = 25.0
    FAR_DISTANCE: float = 50.0
    
    # Threat Areas (relative to frame)
    CRITICAL_AREA: float = 0.15
    HIGH_AREA: float = 0.06
    MEDIUM_AREA: float = 0.025
    
    # Passing
    MIN_PASS_DISTANCE: float = 18.0
    
    # Visual
    MASK_ALPHA: float = 0.38


# ═══════════════════════════════════════════════════════════════════════════════
#                              UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Colors:
    # Primary
    CYAN = (255, 220, 100)
    WHITE = (255, 255, 255)
    GREEN = (100, 230, 100)
    YELLOW = (80, 220, 255)
    ORANGE = (80, 165, 255)
    RED = (70, 70, 240)
    
    # Segments
    SEG_VEHICLE = (170, 140, 90)
    SEG_LEAD = (100, 200, 120)      # Green tint for lead vehicle
    SEG_PERSON = (200, 100, 180)
    SEG_ANIMAL = (100, 180, 200)
    
    # UI
    LIGHT_GRAY = (200, 200, 200)
    GRAY = (140, 140, 140)
    DIM_GRAY = (100, 100, 100)
    DARK_GRAY = (55, 55, 55)
    NEAR_BLACK = (22, 22, 22)
    BLACK = (0, 0, 0)
    
    # Road
    ROAD = (42, 42, 48)
    ROAD_EDGE = (65, 65, 75)
    LANE_LINE = (190, 190, 190)


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
    area_ratio: float
    distance: float
    threat: ThreatLevel
    in_our_lane: bool = False
    is_lead: bool = False
    is_confirmed: bool = True
    track_id: int = -1
    frames_seen: int = 1


@dataclass
class LaneInfo:
    left_line: Optional[List[Tuple[int, int]]] = None
    right_line: Optional[List[Tuple[int, int]]] = None
    left_detected: bool = False
    right_detected: bool = False
    center_offset: float = 0.0


@dataclass
class DrivingState:
    """Current driving situation analysis."""
    lead_vehicle: Optional[Detection] = None
    left_traffic: List[Detection] = field(default_factory=list)
    right_traffic: List[Detection] = field(default_factory=list)
    pedestrians: List[Detection] = field(default_factory=list)
    max_threat: ThreatLevel = ThreatLevel.NONE
    road_clear: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
#                              TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class ObjectTracker:
    def __init__(self):
        self.tracks: Dict[int, dict] = {}
        self.next_id = 0
        self.distance_history: Dict[int, deque] = {}
    
    def update(self, detections: List[Detection]) -> List[Detection]:
        current = [(d.center, d.class_id, d.distance, i) for i, d in enumerate(detections)]
        
        if not self.tracks:
            for center, cls, dist, idx in current:
                self._create_track(detections[idx], center, cls, dist)
            return detections
        
        used_det, used_track = set(), set()
        matches = []
        
        for tid, track in self.tracks.items():
            for center, cls, dist, idx in current:
                d = math.hypot(center[0] - track['center'][0], center[1] - track['center'][1])
                if d < 150:
                    matches.append((d, tid, idx, center, cls, dist))
        
        matches.sort(key=lambda x: x[0])
        
        for d, tid, idx, center, cls, dist in matches:
            if tid in used_track or idx in used_det:
                continue
            
            self.tracks[tid].update({
                'center': center, 'class': cls, 
                'frames': self.tracks[tid]['frames'] + 1, 'missing': 0
            })
            
            # Track distance history for approach detection
            if tid not in self.distance_history:
                self.distance_history[tid] = deque(maxlen=15)
            self.distance_history[tid].append(dist)
            
            detections[idx].track_id = tid
            detections[idx].frames_seen = self.tracks[tid]['frames']
            detections[idx].is_confirmed = self.tracks[tid]['frames'] >= 2
            
            used_track.add(tid)
            used_det.add(idx)
        
        for center, cls, dist, idx in current:
            if idx not in used_det:
                self._create_track(detections[idx], center, cls, dist)
        
        # Age out old tracks
        expired = []
        for tid in self.tracks:
            if tid not in used_track:
                self.tracks[tid]['missing'] += 1
                if self.tracks[tid]['missing'] > 10:
                    expired.append(tid)
        for tid in expired:
            del self.tracks[tid]
            if tid in self.distance_history:
                del self.distance_history[tid]
        
        return detections
    
    def _create_track(self, det, center, cls, dist):
        self.tracks[self.next_id] = {
            'center': center, 'class': cls, 'frames': 1, 'missing': 0
        }
        self.distance_history[self.next_id] = deque(maxlen=15)
        self.distance_history[self.next_id].append(dist)
        det.track_id = self.next_id
        det.frames_seen = 1
        det.is_confirmed = False
        self.next_id += 1
    
    def is_approaching(self, track_id: int) -> bool:
        """Check if object is getting closer."""
        if track_id not in self.distance_history:
            return False
        hist = self.distance_history[track_id]
        if len(hist) < 5:
            return False
        
        # Compare recent average to older average
        recent = sum(list(hist)[-3:]) / 3
        older = sum(list(hist)[:3]) / 3
        return recent < older - 1.5  # Getting closer by >1.5m


# ═══════════════════════════════════════════════════════════════════════════════
#                              DETECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class DetectionEngine:
    def __init__(self, width: int, height: int):
        print("  Loading YOLO model...")
        self.device = get_device()
        print(f"  Device: {self.device.upper()}")
        
        self.model = YOLO(Config.YOLO_MODEL)
        self.w = width
        self.h = height
        self.area = width * height
        self.tracker = ObjectTracker()
        
        # Lane boundaries
        self.lane_left = int(width * Config.OUR_LANE_LEFT)
        self.lane_right = int(width * Config.OUR_LANE_RIGHT)
        self.sky_y = int(height * Config.SKY_CUTOFF)
        self.road_y = int(height * 0.40)
        
        self.labels = {
            0: "PERSON", 1: "CYCLIST", 2: "CAR", 3: "BIKE",
            5: "BUS", 7: "TRUCK", 16: "DOG", 17: "HORSE", 18: "SHEEP", 19: "COW"
        }
        
        self.widths = {
            0: 0.5, 1: 0.65, 2: 1.8, 3: 0.75, 5: 2.5, 7: 2.4,
            16: 0.35, 17: 0.55, 18: 0.4, 19: 0.8
        }
    
    def _estimate_distance(self, w_px: int, cls: int) -> float:
        real = self.widths.get(cls, 1.5)
        if w_px > 10:
            return max(2.0, min(99.0, (real * 720) / w_px))
        return 99.0
    
    def _compute_threat(self, area: float, dist: float, in_lane: bool, cls: int) -> ThreatLevel:
        # Pedestrians/animals are always higher threat
        is_vulnerable = cls in Config.PERSON_CLASSES + Config.ANIMAL_CLASSES
        
        if in_lane:
            if area > Config.CRITICAL_AREA or dist < Config.CRITICAL_DISTANCE:
                return ThreatLevel.CRITICAL
            elif area > Config.HIGH_AREA or dist < Config.CLOSE_DISTANCE:
                return ThreatLevel.HIGH if is_vulnerable else ThreatLevel.HIGH
            elif area > Config.MEDIUM_AREA or dist < Config.MEDIUM_DISTANCE:
                return ThreatLevel.MEDIUM
            elif dist < Config.FAR_DISTANCE:
                return ThreatLevel.LOW
        else:
            # Side traffic - lower threat
            if area > Config.CRITICAL_AREA:
                return ThreatLevel.MEDIUM
            elif area > Config.HIGH_AREA:
                return ThreatLevel.LOW
        
        return ThreatLevel.NONE
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(
            frame, verbose=False, conf=Config.CONFIDENCE,
            classes=Config.ALL_CLASSES, device=self.device, retina_masks=True
        )
        
        detections = []
        
        for r in results:
            if r.boxes is None:
                continue
            
            masks = r.masks.data.cpu().numpy() if r.masks is not None else None
            
            for i, box in enumerate(r.boxes):
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                w_box, h_box = x2 - x1, y2 - y1
                area = (w_box * h_box) / self.area
                
                # Filter
                if cy < self.sky_y:
                    continue
                if area < Config.MIN_AREA or area > Config.MAX_AREA:
                    continue
                
                # Mask
                mask = None
                if masks is not None and i < len(masks):
                    mask = cv2.resize(masks[i], (self.w, self.h))
                    mask = (mask > 0.5).astype(np.uint8)
                
                # In our lane?
                in_lane = self.lane_left < cx < self.lane_right and cy > self.road_y
                
                dist = self._estimate_distance(w_box, cls)
                threat = self._compute_threat(area, dist, in_lane, cls)
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    mask=mask,
                    class_id=cls,
                    label=self.labels.get(cls, "OBJECT"),
                    confidence=conf,
                    center=(cx, cy),
                    area_ratio=area,
                    distance=round(dist, 1),
                    threat=threat,
                    in_our_lane=in_lane
                ))
        
        return self.tracker.update(detections)
    
    def is_approaching(self, det: Detection) -> bool:
        return self.tracker.is_approaching(det.track_id)


# ═══════════════════════════════════════════════════════════════════════════════
#                              LANE DETECTOR  
# ═══════════════════════════════════════════════════════════════════════════════

class LaneDetector:
    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self.left_hist = deque(maxlen=12)
        self.right_hist = deque(maxlen=12)
        self.left_conf = deque(maxlen=20)
        self.right_conf = deque(maxlen=20)
    
    def detect(self, frame: np.ndarray) -> LaneInfo:
        info = LaneInfo()
        h, w = frame.shape[:2]
        
        roi_y = int(h * 0.52)
        roi = frame[roi_y:h, :]
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        mask = np.zeros_like(edges)
        poly = np.array([[(int(w*0.05), edges.shape[0]), (int(w*0.40), 0),
                         (int(w*0.60), 0), (int(w*0.95), edges.shape[0])]], np.int32)
        cv2.fillPoly(mask, poly, 255)
        edges = cv2.bitwise_and(edges, mask)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 35, minLineLength=35, maxLineGap=100)
        
        left_lines, right_lines = [], []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.25 or abs(slope) > 4:
                    continue
                
                y1 += roi_y
                y2 += roi_y
                
                if slope < 0 and x1 < w * 0.55:
                    left_lines.append([x1, y1, x2, y2])
                elif slope > 0 and x1 > w * 0.45:
                    right_lines.append([x1, y1, x2, y2])
        
        self.left_conf.append(1 if len(left_lines) >= 2 else 0)
        self.right_conf.append(1 if len(right_lines) >= 2 else 0)
        
        info.left_detected = sum(self.left_conf) / max(1, len(self.left_conf)) > 0.4
        info.right_detected = sum(self.right_conf) / max(1, len(self.right_conf)) > 0.4
        
        if left_lines:
            self.left_hist.append(np.mean(left_lines, axis=0).astype(int))
        if self.left_hist:
            avg = np.mean(list(self.left_hist), axis=0).astype(int)
            info.left_line = [(avg[0], avg[1]), (avg[2], avg[3])]
        
        if right_lines:
            self.right_hist.append(np.mean(right_lines, axis=0).astype(int))
        if self.right_hist:
            avg = np.mean(list(self.right_hist), axis=0).astype(int)
            info.right_line = [(avg[0], avg[1]), (avg[2], avg[3])]
        
        if info.left_line and info.right_line:
            lx = (info.left_line[0][0] + info.left_line[1][0]) / 2
            rx = (info.right_line[0][0] + info.right_line[1][0]) / 2
            info.center_offset = (w/2 - (lx + rx)/2) / (w/2)
        
        return info


# ═══════════════════════════════════════════════════════════════════════════════
#                          INTELLIGENT DRIVING ADVISOR
# ═══════════════════════════════════════════════════════════════════════════════

class IntelligentAdvisor:
    """
    Smart driving advisor that understands context:
    - Identifies lead vehicle (car you're following)
    - Gives natural advice: FOLLOW, CRUISING, SLOW DOWN, etc.
    - Side traffic is secondary information
    """
    
    def __init__(self, width: int, height: int, detector: DetectionEngine):
        self.w = width
        self.h = height
        self.detector = detector
        
        self.lane_left = int(width * Config.OUR_LANE_LEFT)
        self.lane_right = int(width * Config.OUR_LANE_RIGHT)
    
    def _analyze_scene(self, detections: List[Detection]) -> DrivingState:
        """Analyze the driving scene and categorize objects."""
        state = DrivingState()
        
        confirmed = [d for d in detections if d.is_confirmed]
        
        if not confirmed:
            return state
        
        state.road_clear = False
        state.max_threat = max((d.threat for d in confirmed), default=ThreatLevel.NONE, key=lambda x: x.value)
        
        # Categorize detections
        lane_vehicles = []
        
        for d in confirmed:
            cx = d.center[0]
            
            # Pedestrians/animals - special category
            if d.class_id in Config.PERSON_CLASSES + Config.ANIMAL_CLASSES:
                if d.in_our_lane:
                    state.pedestrians.append(d)
                continue
            
            # Vehicles
            if d.in_our_lane:
                lane_vehicles.append(d)
            elif cx < self.lane_left:
                state.left_traffic.append(d)
            elif cx > self.lane_right:
                state.right_traffic.append(d)
        
        # Find lead vehicle - the most important vehicle ahead
        if lane_vehicles:
            # Sort by: threat (highest first), then by vertical position (lower = closer)
            lane_vehicles.sort(key=lambda d: (-d.threat.value, -d.center[1]))
            state.lead_vehicle = lane_vehicles[0]
            state.lead_vehicle.is_lead = True
        
        # If no vehicle in lane but pedestrian exists, that's our "lead"
        if not state.lead_vehicle and state.pedestrians:
            state.pedestrians.sort(key=lambda d: (-d.threat.value, -d.center[1]))
            state.lead_vehicle = state.pedestrians[0]
            state.lead_vehicle.is_lead = True
        
        return state
    
    def _can_pass(self, side: str, state: DrivingState, lanes: LaneInfo) -> bool:
        """Check if safe to pass on given side."""
        if not state.lead_vehicle:
            return False
        
        if state.lead_vehicle.distance < Config.MIN_PASS_DISTANCE:
            return False
        
        if side == "LEFT":
            return len(state.left_traffic) == 0 and lanes.left_detected
        else:
            return len(state.right_traffic) == 0 and lanes.right_detected
    
    def analyze(self, detections: List[Detection], lanes: LaneInfo) -> Tuple[str, str, Tuple, ThreatLevel, DrivingState]:
        """Generate intelligent driving advice."""
        
        state = self._analyze_scene(detections)
        
        # Road is completely clear
        if state.road_clear or (not state.lead_vehicle and not state.left_traffic and not state.right_traffic):
            return "CLEAR ROAD", "No obstacles detected", Colors.CYAN, ThreatLevel.NONE, state
        
        lead = state.lead_vehicle
        
        # No lead vehicle but side traffic exists
        if not lead:
            left_n = len(state.left_traffic)
            right_n = len(state.right_traffic)
            
            if left_n and right_n:
                return "CAUTION", f"Traffic on both sides", Colors.YELLOW, state.max_threat, state
            elif left_n:
                return "STAY RIGHT", f"{left_n} vehicle(s) on left", Colors.CYAN, state.max_threat, state
            elif right_n:
                return "STAY LEFT", f"{right_n} vehicle(s) on right", Colors.CYAN, state.max_threat, state
            
            return "CLEAR ROAD", "Lane is clear", Colors.CYAN, ThreatLevel.NONE, state
        
        # We have a lead vehicle - provide intelligent advice
        is_pedestrian = lead.class_id in Config.PERSON_CLASSES + Config.ANIMAL_CLASSES
        approaching = self.detector.is_approaching(lead)
        
        # ═══ CRITICAL - EMERGENCY ═══
        if lead.threat == ThreatLevel.CRITICAL:
            if is_pedestrian:
                return "STOP!", f"{lead.label} on road!", Colors.RED, ThreatLevel.CRITICAL, state
            if approaching:
                return "BRAKE!", f"Collision risk - {lead.distance:.0f}m", Colors.RED, ThreatLevel.CRITICAL, state
            return "BRAKE!", f"Too close - {lead.distance:.0f}m", Colors.RED, ThreatLevel.CRITICAL, state
        
        # ═══ HIGH - SLOW DOWN ═══
        if lead.threat == ThreatLevel.HIGH:
            if approaching:
                return "SLOW DOWN", f"Approaching {lead.label} - {lead.distance:.0f}m", Colors.ORANGE, ThreatLevel.HIGH, state
            return "CAUTION", f"{lead.label} close - {lead.distance:.0f}m", Colors.ORANGE, ThreatLevel.HIGH, state
        
        # ═══ MEDIUM - FOLLOWING ═══
        if lead.threat == ThreatLevel.MEDIUM:
            # Check passing options
            can_left = self._can_pass("LEFT", state, lanes)
            can_right = self._can_pass("RIGHT", state, lanes)
            
            if can_left and lead.distance > Config.MIN_PASS_DISTANCE:
                return "PASS LEFT", f"Clear to overtake {lead.label}", Colors.GREEN, ThreatLevel.MEDIUM, state
            elif can_right and lead.distance > Config.MIN_PASS_DISTANCE:
                return "PASS RIGHT", f"Clear to overtake {lead.label}", Colors.GREEN, ThreatLevel.MEDIUM, state
            
            if approaching:
                return "EASE OFF", f"Closing on {lead.label} - {lead.distance:.0f}m", Colors.YELLOW, ThreatLevel.MEDIUM, state
            
            return "FOLLOW", f"Behind {lead.label} - {lead.distance:.0f}m", Colors.YELLOW, ThreatLevel.MEDIUM, state
        
        # ═══ LOW - CRUISING ═══
        if lead.threat == ThreatLevel.LOW:
            return "CRUISING", f"{lead.label} ahead - {lead.distance:.0f}m", Colors.CYAN, ThreatLevel.LOW, state
        
        # ═══ CLEAR ═══
        return "CLEAR ROAD", "Lane is clear", Colors.CYAN, ThreatLevel.NONE, state


# ═══════════════════════════════════════════════════════════════════════════════
#                              HUD RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

class HUDRenderer:
    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self.frame_n = 0
        
        self.main_text = "STARTING"
        self.sub_text = "Initializing..."
        self.main_color = Colors.CYAN
        self.stable = 0
        self.last_main = ""
        
        self.lane_left = int(width * Config.OUR_LANE_LEFT)
        self.lane_right = int(width * Config.OUR_LANE_RIGHT)
    
    def _get_color(self, det: Detection) -> Tuple[int, int, int]:
        if det.is_lead:
            # Lead vehicle gets special color based on threat
            if det.threat == ThreatLevel.CRITICAL:
                return Colors.RED
            elif det.threat == ThreatLevel.HIGH:
                return Colors.ORANGE
            elif det.threat == ThreatLevel.MEDIUM:
                return Colors.YELLOW
            return Colors.SEG_LEAD
        
        if det.class_id in Config.PERSON_CLASSES:
            return Colors.SEG_PERSON
        elif det.class_id in Config.ANIMAL_CLASSES:
            return Colors.SEG_ANIMAL
        
        return Colors.SEG_VEHICLE
    
    def _draw_segments(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        if not detections:
            return frame
        
        overlay = frame.copy()
        edges = np.zeros_like(frame)
        
        for det in detections:
            color = self._get_color(det)
            
            if det.mask is not None:
                m = det.mask > 0
                overlay[m] = color
                
                contours, _ = cv2.findContours(det.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    thickness = 5 if det.is_lead else 3
                    cv2.drawContours(edges, contours, -1, color, thickness)
                    cv2.drawContours(edges, contours, -1, Colors.WHITE, 1)
        
        result = cv2.addWeighted(overlay, Config.MASK_ALPHA, frame, 1 - Config.MASK_ALPHA, 0)
        result = cv2.addWeighted(edges, 0.5, result, 1, 0)
        return result
    
    def _draw_labels(self, frame: np.ndarray, detections: List[Detection], state: DrivingState):
        for det in detections:
            if det.area_ratio < 0.004 and det.threat == ThreatLevel.NONE:
                continue
            
            x1, y1, x2, y2 = det.bbox
            cx, cy = det.center
            color = self._get_color(det)
            
            # Label
            prefix = ">> " if det.is_lead else ""
            label = f"{prefix}{det.label}"
            dist = f"{det.distance:.0f}m"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.52 if det.is_lead else 0.46
            thick = 1
            
            lsz = cv2.getTextSize(label, font, scale, thick)[0]
            dsz = cv2.getTextSize(dist, font, scale, thick)[0]
            total = lsz[0] + dsz[0] + 12
            
            tx = cx - total // 2
            ty = y1 - 12 if y1 > 65 else y2 + 22
            
            # Background
            pad = 5
            cv2.rectangle(frame, (tx - pad, ty - lsz[1] - pad), 
                         (tx + total + pad, ty + pad + 2), Colors.NEAR_BLACK, -1)
            
            # Border - thicker for lead
            border_thick = 2 if det.is_lead else 1
            cv2.rectangle(frame, (tx - pad, ty - lsz[1] - pad),
                         (tx + total + pad, ty + pad + 2), color, border_thick)
            
            # Text
            cv2.putText(frame, label, (tx, ty), font, scale, color, thick, cv2.LINE_AA)
            cv2.putText(frame, dist, (tx + lsz[0] + 10, ty), font, scale, Colors.LIGHT_GRAY, thick, cv2.LINE_AA)
            
            # Connection line
            ly = ty + pad + 2 if ty < cy else ty - lsz[1] - pad
            cv2.line(frame, (cx, ly), (cx, cy), color, 1)
            cv2.circle(frame, (cx, cy), 4 if det.is_lead else 3, color, -1)
    
    def _draw_road(self, frame: np.ndarray) -> np.ndarray:
        h, w = self.h, self.w
        overlay = frame.copy()
        
        rh = int(h * 0.23)
        rt = h - rh
        vx, vy = w // 2, rt
        lb, rb = int(w * 0.06), int(w * 0.94)
        
        # Road surface
        pts = np.array([[vx, vy], [rb, h], [lb, h]], np.int32)
        road = frame.copy()
        cv2.fillPoly(road, [pts], Colors.ROAD)
        overlay = cv2.addWeighted(road, 0.22, overlay, 0.78, 0)
        
        # Edges
        cv2.line(overlay, (vx, vy), (lb, h), Colors.ROAD_EDGE, 2)
        cv2.line(overlay, (vx, vy), (rb, h), Colors.ROAD_EDGE, 2)
        
        # Center dashes
        for i in range(6):
            t1, t2 = i / 6, (i + 0.35) / 6
            y1 = int(vy + (h - vy) * t1)
            y2 = int(vy + (h - vy) * t2)
            th = max(1, int(1 + t1 * 2.5))
            
            tmp = overlay.copy()
            cv2.line(tmp, (vx, y1), (vx, y2), Colors.LANE_LINE, th)
            overlay = cv2.addWeighted(tmp, 0.15 + t1 * 0.45, overlay, 0.85 - t1 * 0.45, 0)
        
        return overlay
    
    def _draw_lanes(self, frame: np.ndarray, lanes: LaneInfo) -> np.ndarray:
        overlay = frame.copy()
        
        if lanes.left_line:
            cv2.line(overlay, lanes.left_line[0], lanes.left_line[1], Colors.GREEN, 3)
        if lanes.right_line:
            cv2.line(overlay, lanes.right_line[0], lanes.right_line[1], Colors.GREEN, 3)
        
        return cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    
    def _draw_top_bar(self, frame: np.ndarray, fps: float, state: DrivingState, lanes: LaneInfo):
        h = 42
        
        for y in range(h):
            a = 1 - (y / h) ** 2
            cv2.line(frame, (0, y), (self.w, y), tuple(int(c * a) for c in Colors.NEAR_BLACK), 1)
        
        cv2.line(frame, (0, h), (self.w, h), Colors.CYAN, 1)
        
        # Time
        cv2.putText(frame, datetime.now().strftime("%H:%M"), (15, 29),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, Colors.WHITE, 1, cv2.LINE_AA)
        
        # Title
        title = "AI DRIVING ASSISTANT"
        tsz = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(frame, title, ((self.w - tsz[0]) // 2, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.CYAN, 1, cv2.LINE_AA)
        
        # Lane indicators
        li = "L" if lanes.left_detected else "-"
        ri = "R" if lanes.right_detected else "-"
        cv2.putText(frame, f"LANES:{li}|{ri}", (self.w - 210, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, Colors.GRAY, 1, cv2.LINE_AA)
        
        # Traffic counts
        ln = len(state.left_traffic)
        rn = len(state.right_traffic)
        if ln or rn:
            cv2.putText(frame, f"L:{ln} R:{rn}", (self.w - 130, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, Colors.YELLOW, 1, cv2.LINE_AA)
        
        # FPS
        fc = Colors.GREEN if fps > 22 else Colors.YELLOW if fps > 14 else Colors.RED
        cv2.putText(frame, f"{int(fps)}fps", (self.w - 50, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, fc, 1, cv2.LINE_AA)
    
    def _draw_bottom(self, frame: np.ndarray, state: DrivingState):
        ph = 78
        py = self.h - ph
        
        for y in range(ph):
            a = (y / ph) ** 1.6
            cv2.line(frame, (0, py + y), (self.w, py + y),
                    tuple(int(c * a) for c in Colors.NEAR_BLACK), 1)
        
        cv2.line(frame, (0, py), (self.w, py), Colors.DARK_GRAY, 1)
        cv2.line(frame, (0, py), (70, py), self.main_color, 2)
        cv2.line(frame, (self.w - 70, py), (self.w, py), self.main_color, 2)
        
        # Main text
        font = cv2.FONT_HERSHEY_DUPLEX
        msz = cv2.getTextSize(self.main_text, font, 1.15, 2)[0]
        mx = (self.w - msz[0]) // 2
        my = py + 40
        
        cv2.putText(frame, self.main_text, (mx + 2, my + 2), font, 1.15, Colors.BLACK, 2, cv2.LINE_AA)
        cv2.putText(frame, self.main_text, (mx, my), font, 1.15, self.main_color, 2, cv2.LINE_AA)
        
        # Sub text
        ssz = cv2.getTextSize(self.sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0]
        cv2.putText(frame, self.sub_text, ((self.w - ssz[0]) // 2, py + 62),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.46, Colors.LIGHT_GRAY, 1, cv2.LINE_AA)
    
    def _draw_side_arrows(self, frame: np.ndarray, state: DrivingState):
        my = self.h // 2
        
        if state.left_traffic:
            n = len(state.left_traffic)
            pts = np.array([[14, my], [32, my - 16], [32, my + 16]], np.int32)
            cv2.fillPoly(frame, [pts], Colors.YELLOW)
            cv2.putText(frame, str(n), (36, my + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, Colors.YELLOW, 1, cv2.LINE_AA)
        
        if state.right_traffic:
            n = len(state.right_traffic)
            pts = np.array([[self.w - 14, my], [self.w - 32, my - 16], [self.w - 32, my + 16]], np.int32)
            cv2.fillPoly(frame, [pts], Colors.YELLOW)
            cv2.putText(frame, str(n), (self.w - 50, my + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, Colors.YELLOW, 1, cv2.LINE_AA)
    
    def _draw_minimap(self, frame: np.ndarray, detections: List[Detection], state: DrivingState):
        mw, mh = 90, 72
        mx, my = self.w - mw - 12, 50
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (mx, my), (mx + mw, my + mh), Colors.NEAR_BLACK, -1)
        frame[:] = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
        cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), Colors.DARK_GRAY, 1)
        
        # Lane boundaries on map
        llx = mx + int(mw * Config.OUR_LANE_LEFT)
        rlx = mx + int(mw * Config.OUR_LANE_RIGHT)
        cv2.line(frame, (llx, my + 5), (llx, my + mh - 5), Colors.DARK_GRAY, 1)
        cv2.line(frame, (rlx, my + 5), (rlx, my + mh - 5), Colors.DARK_GRAY, 1)
        
        # Road
        pts = np.array([[mx + mw//2, my + 6], [mx + mw - 12, my + mh - 6], [mx + 12, my + mh - 6]], np.int32)
        cv2.polylines(frame, [pts], True, Colors.ROAD_EDGE, 1)
        
        # Self
        sx, sy = mx + mw // 2, my + mh - 12
        cv2.fillPoly(frame, [np.array([[sx, sy - 6], [sx - 5, sy + 4], [sx + 5, sy + 4]])], Colors.CYAN)
        
        # Others
        for det in detections:
            if not det.is_confirmed:
                continue
            
            rx = det.center[0] / self.w
            ry = 1 - (det.center[1] / self.h)
            
            dx = int(mx + rx * mw)
            dy = int(my + mh - 10 - ry * (mh - 20))
            dx = max(mx + 5, min(mx + mw - 5, dx))
            dy = max(my + 5, min(my + mh - 5, dy))
            
            color = self._get_color(det)
            size = 5 if det.is_lead else 3
            cv2.circle(frame, (dx, dy), size, color, -1)
    
    def _draw_threat(self, frame: np.ndarray, threat: ThreatLevel):
        if threat == ThreatLevel.NONE:
            return
        
        labels = {
            ThreatLevel.LOW: ("LOW", Colors.GREEN),
            ThreatLevel.MEDIUM: ("MEDIUM", Colors.YELLOW),
            ThreatLevel.HIGH: ("HIGH", Colors.ORANGE),
            ThreatLevel.CRITICAL: ("CRITICAL", Colors.RED)
        }
        
        txt, col = labels.get(threat, ("", Colors.WHITE))
        x, y = 15, 54
        
        if threat == ThreatLevel.CRITICAL and int(self.frame_n * 0.12) % 2:
            cv2.rectangle(frame, (x - 5, y - 16), (x + 78, y + 8), Colors.RED, 2)
        
        cv2.rectangle(frame, (x, y - 14), (x + 72, y + 5), Colors.NEAR_BLACK, -1)
        cv2.rectangle(frame, (x, y - 14), (x + 72, y + 5), col, 1)
        cv2.circle(frame, (x + 11, y - 4), 5, col, -1)
        cv2.putText(frame, txt, (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
    
    def _draw_crosshair(self, frame: np.ndarray):
        cx, cy = self.w // 2, self.h // 2 - 22
        s, g = 16, 6
        
        cv2.line(frame, (cx - s, cy), (cx - g, cy), Colors.CYAN, 1)
        cv2.line(frame, (cx + g, cy), (cx + s, cy), Colors.CYAN, 1)
        cv2.line(frame, (cx, cy - s), (cx, cy - g), Colors.CYAN, 1)
        cv2.line(frame, (cx, cy + g), (cx, cy + s), Colors.CYAN, 1)
        cv2.circle(frame, (cx, cy), 2, Colors.CYAN, -1)
    
    def _draw_lead_indicator(self, frame: np.ndarray, state: DrivingState):
        """Draw special indicator for lead vehicle."""
        if not state.lead_vehicle:
            return
        
        lead = state.lead_vehicle
        _, y1, _, y2 = lead.bbox
        cx = lead.center[0]
        
        # Distance arc
        arc_y = y2 + 15
        if arc_y < self.h - 100:
            color = self._get_color(lead)
            cv2.ellipse(frame, (cx, arc_y), (40, 8), 0, 0, 180, color, 2)
            
            txt = f"{lead.distance:.0f}m"
            tsz = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            tx = cx - tsz[0] // 2
            
            cv2.rectangle(frame, (tx - 4, arc_y + 2), (tx + tsz[0] + 4, arc_y + 18), Colors.NEAR_BLACK, -1)
            cv2.putText(frame, txt, (tx, arc_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    
    def update_text(self, main: str, sub: str, color: Tuple, threat: ThreatLevel):
        if threat == ThreatLevel.CRITICAL:
            self.main_text = main
            self.sub_text = sub
            self.main_color = color
            self.stable = 0
            self.last_main = main
            return
        
        if main == self.last_main:
            self.stable += 1
        else:
            self.stable = 0
            self.last_main = main
        
        if self.stable >= 3:
            self.main_text = main
            self.sub_text = sub
            self.main_color = color
    
    def render(self, frame: np.ndarray, detections: List[Detection], 
               lanes: LaneInfo, fps: float, state: DrivingState) -> np.ndarray:
        self.frame_n += 1
        
        frame = self._draw_lanes(frame, lanes)
        frame = self._draw_segments(frame, detections)
        frame = self._draw_road(frame)
        
        self._draw_lead_indicator(frame, state)
        self._draw_labels(frame, detections, state)
        self._draw_crosshair(frame)
        self._draw_side_arrows(frame, state)
        self._draw_minimap(frame, detections, state)
        self._draw_threat(frame, state.max_threat)
        self._draw_top_bar(frame, fps, state, lanes)
        self._draw_bottom(frame, state)
        
        return frame


# ═══════════════════════════════════════════════════════════════════════════════
#                              MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class DrivingAssistant:
    def __init__(self):
        print("\n" + "=" * 60)
        print("   INTELLIGENT DRIVING ASSISTANT v9.0")
        print("   Lead Vehicle Tracking - Context-Aware Advice")
        print("=" * 60 + "\n")
        
        source = Config.TEST_VIDEO_PATH if Config.USE_TEST_VIDEO else Config.LIVE_CAMERA_INDEX
        print(f"[1/4] Opening: {source}")
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open: {source}")
        
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"      {self.w}x{self.h} @ {self.fps:.0f}fps")
        
        print("\n[2/4] Loading detection engine...")
        self.detector = DetectionEngine(self.w, self.h)
        
        print("\n[3/4] Initializing lane detection...")
        self.lanes = LaneDetector(self.w, self.h)
        
        print("\n[4/4] Setting up advisor & HUD...")
        self.advisor = IntelligentAdvisor(self.w, self.h, self.detector)
        self.hud = HUDRenderer(self.w, self.h)
        
        self.writer = None
        if Config.SAVE_OUTPUT:
            self.writer = cv2.VideoWriter(
                Config.OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'),
                self.fps, (self.w, self.h)
            )
            print(f"\n      Recording: {Config.OUTPUT_PATH}")
        
        self.fps_buf = deque(maxlen=30)
        
        print("\n" + "=" * 60)
        print("   [Q] Quit  [R] Restart  [Space] Pause  [S] Screenshot")
        print("=" * 60 + "\n")
    
    def run(self):
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
                    
                    # Detect
                    detections = self.detector.detect(frame)
                    lane_info = self.lanes.detect(frame)
                    
                    # Analyze
                    main, sub, color, threat, state = self.advisor.analyze(detections, lane_info)
                    self.hud.update_text(main, sub, color, threat)
                    
                    # FPS
                    dt = time.time() - t0
                    self.fps_buf.append(1 / dt if dt > 0 else 0)
                    avg_fps = sum(self.fps_buf) / len(self.fps_buf)
                    
                    # Render
                    output = self.hud.render(frame, detections, lane_info, avg_fps, state)
                    
                    if self.writer:
                        self.writer.write(output)
                    
                    cv2.imshow("AI Driving Assistant v9", output)
                
                key = cv2.waitKey(1 if not paused else 100) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('r') or key == ord('R'):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    print("Restarted")
                elif key == ord(' '):
                    paused = not paused
                    print("Paused" if paused else "Playing")
                elif key == ord('s') or key == ord('S'):
                    fname = f"screenshot_{int(time.time())}.png"
                    cv2.imwrite(fname, output)
                    print(f"Saved: {fname}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        print("\nCleaning up...")
        if self.writer:
            self.writer.release()
            print(f"  Saved: {Config.OUTPUT_PATH}")
        self.cap.release()
        cv2.destroyAllWindows()
        print("  Done!\n")


if __name__ == "__main__":
    try:
        app = DrivingAssistant()
        app.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
