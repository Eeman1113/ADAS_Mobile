#!/usr/bin/env python3


import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
import warnings
import torch

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    # ─── Source Settings ───
    USE_TEST_VIDEO: bool = True
    TEST_VIDEO_PATH: str = "test1.mp4"
    LIVE_CAMERA_INDEX: int = 0
    
    # ─── Output Settings ───
    SAVE_OUTPUT: bool = True
    OUTPUT_PATH: str = "output_segmented.mp4"
    
    # ─── Model Settings ───
    # Segmentation models: yolov8n-seg.pt, yolov8s-seg.pt, yolov8m-seg.pt, yolov8x-seg.pt
    YOLO_MODEL: str = "yolov8s-seg.pt"  # Segmentation model
    CONFIDENCE_THRESHOLD: float = 0.4
    
    # ─── Indian Road Classes ───
    ALL_CLASSES: List[int] = [0, 1, 2, 3, 5, 7, 16, 17, 18, 19]
    
    # ─── Danger Thresholds ───
    CRITICAL_AREA: float = 0.18
    HIGH_AREA: float = 0.08
    MEDIUM_AREA: float = 0.03
    CENTER_ZONE: float = 0.30
    
    # ─── Overlay Settings ───
    MASK_ALPHA: float = 0.45  # Transparency of segmentation masks
    EDGE_GLOW: bool = True    # Add glow effect to mask edges


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
    
    # Segmentation Overlay Colors (BGR) - Softer, more pleasing
    SEG_SAFE = (180, 130, 70)       # Soft blue - safe objects
    SEG_LOW = (150, 180, 80)        # Teal - low threat
    SEG_MEDIUM = (80, 180, 200)     # Yellow/gold - medium threat
    SEG_HIGH = (80, 140, 230)       # Orange - high threat
    SEG_CRITICAL = (80, 80, 230)    # Red - critical threat
    
    # Pedestrian/Animal special color
    SEG_PEDESTRIAN = (200, 100, 180)  # Purple tint for pedestrians
    SEG_ANIMAL = (100, 180, 200)       # Amber for animals
    
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
    mask: Optional[np.ndarray]  # Segmentation mask
    class_id: int
    label: str
    confidence: float
    center: Tuple[int, int]
    relative_area: float
    position: str
    distance: float
    threat: ThreatLevel
    track_id: int = -1


@dataclass
class LaneData:
    left_points: List[Tuple[int, int]] = field(default_factory=list)
    right_points: List[Tuple[int, int]] = field(default_factory=list)
    center_offset: float = 0.0
    detected: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
#                              SIMPLE TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.max_age = 10
    
    def update(self, detections):
        new_tracks = {}
        used_dets = set()
        
        for tid, (cx, cy, age) in self.tracks.items():
            best_dist = float('inf')
            best_idx = -1
            
            for i, det in enumerate(detections):
                if i in used_dets:
                    continue
                dx = det.center[0] - cx
                dy = det.center[1] - cy
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < best_dist and dist < 150:
                    best_dist = dist
                    best_idx = i
            
            if best_idx >= 0:
                det = detections[best_idx]
                new_tracks[tid] = (det.center[0], det.center[1], 0)
                det.track_id = tid
                used_dets.add(best_idx)
            elif age < self.max_age:
                new_tracks[tid] = (cx, cy, age + 1)
        
        for i, det in enumerate(detections):
            if i not in used_dets:
                new_tracks[self.next_id] = (det.center[0], det.center[1], 0)
                det.track_id = self.next_id
                self.next_id += 1
        
        self.tracks = new_tracks
        return detections


# ═══════════════════════════════════════════════════════════════════════════════
#                         SEGMENTATION DETECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SegmentationEngine:
    """YOLO Segmentation-based detection engine."""
    
    def __init__(self, width, height):
        print("  → Loading YOLO Segmentation model...")
        
        self.device = get_device()
        device_names = {
            "mps": "Apple Silicon GPU (MPS)",
            "cuda": "NVIDIA GPU (CUDA)",
            "cpu": "CPU"
        }
        print(f"  → Using {device_names.get(self.device, self.device)}")
        
        # Load segmentation model
        self.model = YOLO(Config.YOLO_MODEL)
        
        self.width = width
        self.height = height
        self.area = width * height
        self.tracker = SimpleTracker()
        
        self.labels = {
            0: "PEDESTRIAN", 1: "BICYCLE", 2: "CAR", 3: "MOTORCYCLE",
            5: "BUS", 7: "TRUCK", 16: "DOG", 17: "HORSE", 18: "SHEEP", 19: "COW"
        }
        
        self.widths = {
            0: 0.5, 1: 0.6, 2: 1.8, 3: 0.8, 5: 2.5, 7: 2.5,
            16: 0.3, 17: 0.6, 18: 0.4, 19: 0.8
        }
    
    def detect(self, frame):
        """Run segmentation on frame."""
        results = self.model(
            frame,
            verbose=False,
            conf=Config.CONFIDENCE_THRESHOLD,
            classes=Config.ALL_CLASSES,
            device=self.device,
            retina_masks=True  # Higher quality masks
        )
        
        detections = []
        
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            
            # Get masks if available
            masks = None
            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()
            
            for idx, box in enumerate(r.boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get mask for this detection
                mask = None
                if masks is not None and idx < len(masks):
                    mask = masks[idx]
                    # Resize mask to frame size
                    mask = cv2.resize(mask, (self.width, self.height))
                    mask = (mask > 0.5).astype(np.uint8)
                
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                w, h = x2 - x1, y2 - y1
                rel_area = (w * h) / self.area
                
                # Position determination
                center_left = self.width * (0.5 - Config.CENTER_ZONE / 2)
                center_right = self.width * (0.5 + Config.CENTER_ZONE / 2)
                
                if cx < center_left:
                    pos = "LEFT"
                elif cx > center_right:
                    pos = "RIGHT"
                else:
                    pos = "CENTER"
                
                # Distance estimate
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
                
                # Extra threat for pedestrians/animals
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
        
        return self.tracker.update(detections)


# ═══════════════════════════════════════════════════════════════════════════════
#                              LANE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class LaneDetector:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.left_history = deque(maxlen=8)
        self.right_history = deque(maxlen=8)
    
    def detect(self, frame):
        h, w = frame.shape[:2]
        
        roi_top = int(h * 0.55)
        roi = frame[roi_top:h, :]
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        mask = np.zeros_like(edges)
        pts = np.array([
            [0, edges.shape[0]],
            [w * 0.1, 0],
            [w * 0.9, 0],
            [w, edges.shape[0]]
        ], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        masked = cv2.bitwise_and(edges, mask)
        
        lines = cv2.HoughLinesP(masked, 1, np.pi/180, 40, minLineLength=40, maxLineGap=100)
        
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
        
        if left_lines:
            avg = np.mean(left_lines, axis=0).astype(int)
            self.left_history.append(avg)
        
        if self.left_history:
            avg = np.mean(self.left_history, axis=0).astype(int)
            lane_data.left_points = [(avg[0], avg[1]), (avg[2], avg[3])]
        
        if right_lines:
            avg = np.mean(right_lines, axis=0).astype(int)
            self.right_history.append(avg)
        
        if self.right_history:
            avg = np.mean(self.right_history, axis=0).astype(int)
            lane_data.right_points = [(avg[0], avg[1]), (avg[2], avg[3])]
        
        lane_data.detected = bool(lane_data.left_points or lane_data.right_points)
        
        if lane_data.left_points and lane_data.right_points:
            left_x = (lane_data.left_points[0][0] + lane_data.left_points[1][0]) / 2
            right_x = (lane_data.right_points[0][0] + lane_data.right_points[1][0]) / 2
            center = (left_x + right_x) / 2
            lane_data.center_offset = (w / 2 - center) / (w / 2)
        
        return lane_data


# ═══════════════════════════════════════════════════════════════════════════════
#                           CLEAN SEGMENTATION HUD
# ═══════════════════════════════════════════════════════════════════════════════

class SegmentationHUD:
    """Clean HUD with segmentation mask overlays."""
    
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.frame_num = 0
        
        # Instruction state
        self.main_text = "READY"
        self.sub_text = "System Online"
        self.alert_color = Colors.CYAN
        self.last_main = ""
        self.stable_count = 0
    
    def _get_segment_color(self, detection):
        """Get overlay color based on object type and threat."""
        # Special colors for pedestrians and animals
        if detection.class_id == 0:  # Pedestrian
            base_color = Colors.SEG_PEDESTRIAN
        elif detection.class_id in [16, 17, 18, 19]:  # Animals
            base_color = Colors.SEG_ANIMAL
        else:
            # Color by threat level
            threat_colors = {
                ThreatLevel.NONE: Colors.SEG_SAFE,
                ThreatLevel.LOW: Colors.SEG_LOW,
                ThreatLevel.MEDIUM: Colors.SEG_MEDIUM,
                ThreatLevel.HIGH: Colors.SEG_HIGH,
                ThreatLevel.CRITICAL: Colors.SEG_CRITICAL
            }
            base_color = threat_colors.get(detection.threat, Colors.SEG_SAFE)
        
        return base_color
    
    def _draw_segmentation_overlays(self, frame, detections):
        """Draw transparent segmentation masks over detected objects."""
        if not detections:
            return frame
        
        overlay = frame.copy()
        edge_overlay = frame.copy()
        
        for det in detections:
            if det.mask is None:
                continue
            
            color = self._get_segment_color(det)
            
            # Create colored mask
            mask_3ch = np.zeros_like(frame)
            mask_3ch[det.mask == 1] = color
            
            # Apply mask to overlay
            mask_bool = det.mask == 1
            overlay[mask_bool] = mask_3ch[mask_bool]
            
            # Add edge glow effect
            if Config.EDGE_GLOW:
                # Find contours for edge effect
                contours, _ = cv2.findContours(
                    det.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                if contours:
                    # Draw glowing edge
                    # Outer glow
                    cv2.drawContours(edge_overlay, contours, -1, color, 6)
                    cv2.drawContours(edge_overlay, contours, -1, Colors.WHITE, 2)
        
        # Blend segmentation overlay
        alpha = Config.MASK_ALPHA
        result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Blend edge glow
        if Config.EDGE_GLOW:
            result = cv2.addWeighted(edge_overlay, 0.3, result, 0.7, 0)
        
        return result
    
    def _draw_object_labels(self, frame, detections):
        """Draw minimal floating labels for detected objects."""
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx, cy = det.center
            
            # Only label significant objects
            if det.relative_area < 0.01 and det.threat == ThreatLevel.NONE:
                continue
            
            color = self._get_segment_color(det)
            
            # Label position - above object
            label = f"{det.label}"
            dist_label = f"{det.distance:.0f}m"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            
            # Get text sizes
            label_size = cv2.getTextSize(label, font, scale, 1)[0]
            dist_size = cv2.getTextSize(dist_label, font, scale, 1)[0]
            
            total_width = label_size[0] + dist_size[0] + 15
            
            # Position label above center of object
            text_x = cx - total_width // 2
            text_y = y1 - 12
            
            if text_y < 60:
                text_y = y2 + 25
            
            # Background pill shape
            padding = 6
            pill_x1 = text_x - padding
            pill_y1 = text_y - label_size[1] - padding
            pill_x2 = text_x + total_width + padding
            pill_y2 = text_y + padding
            
            # Draw rounded rectangle background
            cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), 
                         Colors.NEAR_BLACK, -1)
            cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), 
                         color, 1)
            
            # Draw label text
            cv2.putText(frame, label, (text_x, text_y), font, scale, 
                       color, 1, cv2.LINE_AA)
            
            # Draw distance
            cv2.putText(frame, dist_label, (text_x + label_size[0] + 10, text_y), 
                       font, scale, Colors.LIGHT_GRAY, 1, cv2.LINE_AA)
            
            # Draw connecting line to object center
            line_start_y = pill_y2 if text_y < cy else pill_y1
            cv2.line(frame, (cx, line_start_y), (cx, cy), color, 1)
            cv2.circle(frame, (cx, cy), 3, color, -1)
    
    def _threat_color(self, threat):
        """Get alert color for threat level."""
        return {
            ThreatLevel.NONE: Colors.CYAN,
            ThreatLevel.LOW: Colors.GREEN,
            ThreatLevel.MEDIUM: Colors.YELLOW,
            ThreatLevel.HIGH: Colors.ORANGE,
            ThreatLevel.CRITICAL: Colors.RED
        }.get(threat, Colors.CYAN)
    
    def _draw_road_perspective(self, frame):
        """Draw clean road perspective."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
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
        
        road_surface = frame.copy()
        cv2.fillPoly(road_surface, [road_pts], Colors.ROAD_GRAY)
        cv2.addWeighted(road_surface, 0.25, overlay, 0.75, 0, overlay)
        
        # Road edges
        cv2.line(overlay, (vp_x, vp_y), (left_bottom, h), Colors.ROAD_EDGE, 2)
        cv2.line(overlay, (vp_x, vp_y), (right_bottom, h), Colors.ROAD_EDGE, 2)
        
        # Center line dashes
        num_dashes = 6
        for i in range(num_dashes):
            p1 = i / num_dashes
            p2 = (i + 0.4) / num_dashes
            
            y1 = int(vp_y + (h - vp_y) * p1)
            y2 = int(vp_y + (h - vp_y) * p2)
            
            thickness = max(1, int(2 + p1 * 2))
            alpha = 0.2 + p1 * 0.4
            
            dash_overlay = overlay.copy()
            cv2.line(dash_overlay, (vp_x, y1), (vp_x, y2), Colors.LANE_MARK, thickness)
            cv2.addWeighted(dash_overlay, alpha, overlay, 1 - alpha, 0, overlay)
        
        return overlay
    
    def _draw_distance_arcs(self, frame, detections):
        """Draw distance arcs on road for center objects."""
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
            
            # Draw arc
            cv2.ellipse(frame, (w // 2, arc_y), (arc_width, 12), 0, 0, 180, color, 2)
            
            # Distance label
            dist_text = f"{det.distance:.0f}m"
            text_size = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            text_x = w // 2 - text_size[0] // 2
            
            cv2.rectangle(frame, (text_x - 4, arc_y - 18), 
                         (text_x + text_size[0] + 4, arc_y - 4), Colors.NEAR_BLACK, -1)
            cv2.putText(frame, dist_text, (text_x, arc_y - 7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    
    def _draw_lane_overlay(self, frame, lane_data):
        """Draw detected lanes."""
        if not lane_data.detected:
            return frame
        
        overlay = frame.copy()
        
        if lane_data.left_points:
            cv2.line(overlay, lane_data.left_points[0], lane_data.left_points[1], 
                    Colors.GREEN, 4)
        
        if lane_data.right_points:
            cv2.line(overlay, lane_data.right_points[0], lane_data.right_points[1], 
                    Colors.GREEN, 4)
        
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        return frame
    
    def _draw_top_bar(self, frame, fps, num_objects):
        """Draw minimal top bar."""
        bar_h = 45
        
        # Gradient background
        overlay = frame.copy()
        for y in range(bar_h):
            alpha = 1 - (y / bar_h) ** 2
            color = tuple(int(c * alpha) for c in Colors.NEAR_BLACK)
            cv2.line(overlay, (0, y), (self.w, y), color, 1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Bottom line
        cv2.line(frame, (0, bar_h), (self.w, bar_h), Colors.CYAN, 1)
        
        # Time
        time_str = datetime.now().strftime("%H:%M")
        cv2.putText(frame, time_str, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, Colors.WHITE, 1, cv2.LINE_AA)
        
        # Title
        title = "AI DRIVING ASSISTANT"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        title_x = (self.w - title_size[0]) // 2
        cv2.putText(frame, title, (title_x, 28), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, Colors.CYAN, 1, cv2.LINE_AA)
        
        # Objects count
        cv2.putText(frame, f"OBJ: {num_objects}", (self.w - 160, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.GRAY, 1, cv2.LINE_AA)
        
        # FPS
        fps_color = Colors.GREEN if fps > 20 else Colors.YELLOW if fps > 10 else Colors.RED
        cv2.putText(frame, f"{int(fps)} FPS", (self.w - 70, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1, cv2.LINE_AA)
    
    def _draw_bottom_panel(self, frame):
        """Draw instruction panel."""
        panel_h = 85
        panel_top = self.h - panel_h
        
        # Gradient background
        overlay = frame.copy()
        for y in range(panel_h):
            progress = y / panel_h
            alpha = progress ** 1.5
            color = tuple(int(c * alpha) for c in Colors.NEAR_BLACK)
            cv2.line(overlay, (0, panel_top + y), (self.w, panel_top + y), color, 1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Top line with accents
        cv2.line(frame, (0, panel_top), (self.w, panel_top), Colors.DARK_GRAY, 1)
        cv2.line(frame, (0, panel_top), (80, panel_top), self.alert_color, 2)
        cv2.line(frame, (self.w - 80, panel_top), (self.w, panel_top), self.alert_color, 2)
        
        # Main instruction
        font = cv2.FONT_HERSHEY_DUPLEX
        main_size = cv2.getTextSize(self.main_text, font, 1.2, 2)[0]
        main_x = (self.w - main_size[0]) // 2
        main_y = panel_top + 42
        
        # Shadow
        cv2.putText(frame, self.main_text, (main_x + 2, main_y + 2), font,
                   1.2, Colors.BLACK, 2, cv2.LINE_AA)
        cv2.putText(frame, self.main_text, (main_x, main_y), font,
                   1.2, self.alert_color, 2, cv2.LINE_AA)
        
        # Sub text
        sub_size = cv2.getTextSize(self.sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        sub_x = (self.w - sub_size[0]) // 2
        cv2.putText(frame, self.sub_text, (sub_x, panel_top + 68), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.LIGHT_GRAY, 1, cv2.LINE_AA)
    
    def _draw_side_indicators(self, frame, detections):
        """Draw side traffic indicators."""
        left_count = sum(1 for d in detections if d.position == "LEFT")
        right_count = sum(1 for d in detections if d.position == "RIGHT")
        
        y = self.h // 2
        
        if left_count > 0:
            pts = np.array([[15, y], [35, y - 18], [35, y + 18]], np.int32)
            cv2.fillPoly(frame, [pts], Colors.YELLOW)
            cv2.putText(frame, str(left_count), (40, y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.YELLOW, 1, cv2.LINE_AA)
        
        if right_count > 0:
            pts = np.array([[self.w - 15, y], [self.w - 35, y - 18], 
                           [self.w - 35, y + 18]], np.int32)
            cv2.fillPoly(frame, [pts], Colors.YELLOW)
            cv2.putText(frame, str(right_count), (self.w - 55, y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.YELLOW, 1, cv2.LINE_AA)
    
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
        """Draw minimal center crosshair."""
        cx, cy = self.w // 2, self.h // 2 - 30
        
        size = 18
        gap = 5
        
        cv2.line(frame, (cx - size, cy), (cx - gap, cy), Colors.CYAN, 1)
        cv2.line(frame, (cx + gap, cy), (cx + size, cy), Colors.CYAN, 1)
        cv2.line(frame, (cx, cy - size), (cx, cy - gap), Colors.CYAN, 1)
        cv2.line(frame, (cx, cy + gap), (cx, cy + size), Colors.CYAN, 1)
        cv2.circle(frame, (cx, cy), 2, Colors.CYAN, -1)
    
    def _draw_threat_indicator(self, frame, max_threat):
        """Draw overall threat level indicator."""
        if max_threat == ThreatLevel.NONE:
            return
        
        # Position
        ind_x = 20
        ind_y = 60
        
        color = self._threat_color(max_threat)
        label = {
            ThreatLevel.LOW: "LOW",
            ThreatLevel.MEDIUM: "MEDIUM",
            ThreatLevel.HIGH: "HIGH",
            ThreatLevel.CRITICAL: "CRITICAL"
        }.get(max_threat, "")
        
        # Pulsing for critical
        if max_threat == ThreatLevel.CRITICAL:
            pulse = abs(math.sin(self.frame_num * 0.15))
            if pulse > 0.5:
                cv2.rectangle(frame, (ind_x - 5, ind_y - 18), 
                             (ind_x + 85, ind_y + 8), Colors.RED, 2)
        
        # Background
        cv2.rectangle(frame, (ind_x, ind_y - 15), (ind_x + 80, ind_y + 5), 
                     Colors.NEAR_BLACK, -1)
        cv2.rectangle(frame, (ind_x, ind_y - 15), (ind_x + 80, ind_y + 5), color, 1)
        
        # Circle indicator
        cv2.circle(frame, (ind_x + 12, ind_y - 5), 5, color, -1)
        
        # Label
        cv2.putText(frame, label, (ind_x + 22, ind_y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.4, color, 1, cv2.LINE_AA)
    
    def update_instruction(self, main, sub, color, threat):
        """Update instruction with smoothing."""
        if threat == ThreatLevel.CRITICAL:
            self.main_text = main
            self.sub_text = sub
            self.alert_color = color
            self.stable_count = 0
            self.last_main = main
            return
        
        if main == self.last_main:
            self.stable_count += 1
        else:
            self.stable_count = 0
            self.last_main = main
        
        if self.stable_count >= 5:
            self.main_text = main
            self.sub_text = sub
            self.alert_color = color
    
    def render(self, frame, detections, lane_data, fps):
        """Render complete segmentation HUD."""
        self.frame_num += 1
        
        # 1. Lane overlay (behind everything)
        frame = self._draw_lane_overlay(frame, lane_data)
        
        # 2. Segmentation masks (main feature!)
        frame = self._draw_segmentation_overlays(frame, detections)
        
        # 3. Road perspective
        frame = self._draw_road_perspective(frame)
        
        # 4. Distance arcs
        self._draw_distance_arcs(frame, detections)
        
        # 5. Object labels (minimal floating labels)
        self._draw_object_labels(frame, detections)
        
        # 6. Center reticle
        self._draw_center_reticle(frame)
        
        # 7. Side indicators
        self._draw_side_indicators(frame, detections)
        
        # 8. Mini map
        self._draw_mini_map(frame, detections)
        
        # 9. Threat indicator
        max_threat = max((d.threat for d in detections), default=ThreatLevel.NONE, 
                        key=lambda t: t.value)
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
    def analyze(self, detections, lane_data):
        critical = [d for d in detections if d.threat == ThreatLevel.CRITICAL]
        high = [d for d in detections if d.threat == ThreatLevel.HIGH]
        center = [d for d in detections if d.position == "CENTER" and d.threat.value >= 2]
        
        max_threat = max((d.threat for d in detections), default=ThreatLevel.NONE,
                        key=lambda t: t.value)
        
        if critical:
            d = critical[0]
            if d.class_id in [0, 16, 17, 18, 19]:
                return "⚠ STOP", f"{d.label} on road!", Colors.RED, max_threat
            return "⚠ BRAKE", f"Collision warning - {d.distance:.0f}m", Colors.RED, max_threat
        
        if high:
            d = high[0]
            return "SLOW DOWN", f"{d.label} ahead - {d.distance:.0f}m", Colors.ORANGE, max_threat
        
        if center:
            d = center[0]
            left_clear = not any(det.position == "LEFT" for det in detections)
            right_clear = not any(det.position == "RIGHT" for det in detections)
            
            if right_clear:
                return "PASS RIGHT", f"Clear to overtake {d.label}", Colors.GREEN, max_threat
            elif left_clear:
                return "PASS LEFT", f"Clear to overtake {d.label}", Colors.GREEN, max_threat
            else:
                return "HOLD", "Wait for opening", Colors.YELLOW, max_threat
        
        if lane_data.detected and abs(lane_data.center_offset) > 0.3:
            if lane_data.center_offset > 0:
                return "DRIFT LEFT", "Steer right to correct", Colors.YELLOW, max_threat
            else:
                return "DRIFT RIGHT", "Steer left to correct", Colors.YELLOW, max_threat
        
        return "CLEAR", "Road ahead is clear", Colors.CYAN, max_threat


# ═══════════════════════════════════════════════════════════════════════════════
#                              MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class DrivingAssistant:
    def __init__(self):
        print("\n" + "═" * 55)
        print("   SEGMENTATION AI DRIVING ASSISTANT v5.0")
        print("   Instance Segmentation • Clean Overlays")
        print("   macOS Optimized • Indian Roads")
        print("═" * 55 + "\n")
        
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
        print("\n[2/4] Loading segmentation engine...")
        self.detector = SegmentationEngine(self.width, self.height)
        
        print("\n[3/4] Initializing lane detection...")
        self.lane_detector = LaneDetector(self.width, self.height)
        print("      Lane detection ready")
        
        print("\n[4/4] Setting up HUD...")
        self.hud = SegmentationHUD(self.width, self.height)
        self.logic = DrivingLogic()
        print("      Segmentation HUD ready")
        
        # Output
        self.writer = None
        if Config.SAVE_OUTPUT:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                Config.OUTPUT_PATH, fourcc, self.video_fps,
                (self.width, self.height)
            )
            print(f"\n      Recording to: {Config.OUTPUT_PATH}")
        
        self.fps_tracker = deque(maxlen=30)
        
        print("\n" + "═" * 55)
        print("   Ready! Press 'Q' to quit, 'R' to restart video")
        print("═" * 55 + "\n")
    
    def run(self):
        try:
            while True:
                start = time.time()
                
                ret, frame = self.cap.read()
                
                if not ret:
                    if Config.USE_TEST_VIDEO:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                
                # Detection with segmentation
                detections = self.detector.detect(frame)
                
                # Lane detection
                lane_data = self.lane_detector.detect(frame)
                
                # Driving logic
                main, sub, color, threat = self.logic.analyze(detections, lane_data)
                self.hud.update_instruction(main, sub, color, threat)
                
                # FPS calculation
                elapsed = time.time() - start
                current_fps = 1 / elapsed if elapsed > 0 else 0
                self.fps_tracker.append(current_fps)
                avg_fps = sum(self.fps_tracker) / len(self.fps_tracker)
                
                # Render HUD
                output = self.hud.render(frame, detections, lane_data, avg_fps)
                
                # Save
                if self.writer:
                    self.writer.write(output)
                
                # Display
                cv2.imshow("AI Driving Assistant - Segmentation", output)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('r') or key == ord('R'):
                    if Config.USE_TEST_VIDEO:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        print("Video restarted")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        print("\nCleaning up...")
        if self.writer:
            self.writer.release()
            print(f"  ✓ Saved: {Config.OUTPUT_PATH}")
        self.cap.release()
        cv2.destroyAllWindows()
        print("  ✓ Done!\n")


# ═══════════════════════════════════════════════════════════════════════════════
#                              ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        app = DrivingAssistant()
        app.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("   Ensure 'test.mp4' exists in the current directory.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
