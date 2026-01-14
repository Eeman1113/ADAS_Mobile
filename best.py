#!/usr/bin/env python3


import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    # Source
    USE_TEST_VIDEO: bool = True
    TEST_VIDEO_PATH: str = "test.mp4"
    LIVE_CAMERA_INDEX: int = 0
    
    # Output
    SAVE_OUTPUT: bool = True
    OUTPUT_PATH: str = "output_clean.mp4"
    
    # Model - Using best model
    YOLO_MODEL: str = "yolov8x.pt"
    CONFIDENCE_THRESHOLD: float = 0.4
    
    # Indian Road Classes
    ALL_CLASSES: List[int] = [0, 1, 2, 3, 5, 7, 16, 17, 18, 19]
    
    # Danger Thresholds
    CRITICAL_AREA: float = 0.18
    HIGH_AREA: float = 0.08
    MEDIUM_AREA: float = 0.03
    CENTER_ZONE: float = 0.30


# ═══════════════════════════════════════════════════════════════════════════════
#                              COLORS - CLEAN PALETTE
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    # Primary
    CYAN = (255, 220, 100)
    TEAL = (200, 180, 80)
    WHITE = (255, 255, 255)
    
    # Alerts
    GREEN = (100, 230, 100)
    YELLOW = (80, 220, 255)
    ORANGE = (80, 180, 255)
    RED = (80, 80, 255)
    
    # Neutrals
    LIGHT_GRAY = (200, 200, 200)
    GRAY = (130, 130, 130)
    DARK_GRAY = (60, 60, 60)
    DARKER = (35, 35, 35)
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
    class_id: int
    label: str
    confidence: float
    center: Tuple[int, int]
    relative_area: float
    position: str
    distance: float
    threat: ThreatLevel
    track_id: int = -1
    trajectory: List[Tuple[int, int]] = field(default_factory=list)


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
        # Simple centroid tracking
        new_tracks = {}
        used_dets = set()
        
        for tid, (cx, cy, age) in self.tracks.items():
            best_dist = 100
            best_idx = -1
            
            for i, det in enumerate(detections):
                if i in used_dets:
                    continue
                dx = det.center[0] - cx
                dy = det.center[1] - cy
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            
            if best_idx >= 0 and best_dist < 150:
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
#                              DETECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class DetectionEngine:
    def __init__(self, width, height):
        print("  → Loading YOLO model...")
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
        results = self.model(frame, verbose=False, conf=Config.CONFIDENCE_THRESHOLD, 
                            classes=Config.ALL_CLASSES)
        
        detections = []
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
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
                elif rel_area > Config.HIGH_AREA:
                    threat = ThreatLevel.MEDIUM
                
                # Extra threat for pedestrians/animals
                if cls in [0, 16, 17, 18, 19] and pos == "CENTER" and rel_area > 0.02:
                    threat = ThreatLevel(min(threat.value + 1, 4))
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
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
        
        # ROI - bottom portion
        roi_top = int(h * 0.55)
        roi = frame[roi_top:h, :]
        
        # Process
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # Mask for road area (trapezoid)
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
                
                # Adjust y coordinates back to full frame
                y1 += roi_top
                y2 += roi_top
                
                if slope < 0:
                    left_lines.append((x1, y1, x2, y2))
                else:
                    right_lines.append((x1, y1, x2, y2))
        
        lane_data = LaneData()
        
        # Average left lane
        if left_lines:
            avg = np.mean(left_lines, axis=0).astype(int)
            self.left_history.append(avg)
        
        if self.left_history:
            avg = np.mean(self.left_history, axis=0).astype(int)
            lane_data.left_points = [(avg[0], avg[1]), (avg[2], avg[3])]
        
        # Average right lane
        if right_lines:
            avg = np.mean(right_lines, axis=0).astype(int)
            self.right_history.append(avg)
        
        if self.right_history:
            avg = np.mean(self.right_history, axis=0).astype(int)
            lane_data.right_points = [(avg[0], avg[1]), (avg[2], avg[3])]
        
        lane_data.detected = bool(lane_data.left_points or lane_data.right_points)
        
        # Calculate offset
        if lane_data.left_points and lane_data.right_points:
            left_x = (lane_data.left_points[0][0] + lane_data.left_points[1][0]) / 2
            right_x = (lane_data.right_points[0][0] + lane_data.right_points[1][0]) / 2
            center = (left_x + right_x) / 2
            lane_data.center_offset = (w / 2 - center) / (w / 2)
        
        return lane_data


# ═══════════════════════════════════════════════════════════════════════════════
#                              CLEAN HUD RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

class CleanHUD:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        
        # Animation
        self.frame_num = 0
        
        # Instruction state
        self.main_text = "READY"
        self.sub_text = "System Online"
        self.alert_color = Colors.CYAN
        self.last_main = ""
        self.stable_count = 0
        
        # Pre-render road graphic
        self._create_road_overlay()
    
    def _create_road_overlay(self):
        """Create the stylized road perspective overlay."""
        self.road_overlay = np.zeros((self.h, self.w, 4), dtype=np.uint8)
        
        road_h = int(self.h * 0.35)
        road_top = self.h - road_h
        
        # Road trapezoid points
        top_left = int(self.w * 0.35)
        top_right = int(self.w * 0.65)
        bottom_left = 0
        bottom_right = self.w
        
        # Draw road surface gradient
        for y in range(road_h):
            progress = y / road_h
            alpha = int(60 + progress * 80)
            
            # Calculate x positions for this y
            left_x = int(top_left + (bottom_left - top_left) * progress)
            right_x = int(top_right + (bottom_right - top_right) * progress)
            
            # Road surface
            cv2.line(self.road_overlay, (left_x, road_top + y), (right_x, road_top + y),
                    (*Colors.ROAD_GRAY, alpha), 1)
    
    def _draw_road_perspective(self, frame):
        """Draw clean road perspective with lane markings."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        road_h = int(h * 0.30)
        road_top = h - road_h
        
        # Vanishing point
        vp_x = w // 2
        vp_y = road_top - 20
        
        # Road edges
        left_bottom = int(w * 0.05)
        right_bottom = int(w * 0.95)
        
        # Draw road surface (semi-transparent)
        road_pts = np.array([
            [vp_x, vp_y],
            [right_bottom, h],
            [left_bottom, h]
        ], np.int32)
        
        road_surface = frame.copy()
        cv2.fillPoly(road_surface, [road_pts], Colors.ROAD_GRAY)
        cv2.addWeighted(road_surface, 0.3, overlay, 0.7, 0, overlay)
        
        # Road edge lines
        cv2.line(overlay, (vp_x, vp_y), (left_bottom, h), Colors.ROAD_EDGE, 2)
        cv2.line(overlay, (vp_x, vp_y), (right_bottom, h), Colors.ROAD_EDGE, 2)
        
        # Center dashed line
        num_dashes = 8
        for i in range(num_dashes):
            progress1 = i / num_dashes
            progress2 = (i + 0.5) / num_dashes
            
            y1 = int(vp_y + (h - vp_y) * progress1)
            y2 = int(vp_y + (h - vp_y) * progress2)
            
            # Perspective scaling
            scale1 = 0.1 + progress1 * 0.9
            scale2 = 0.1 + progress2 * 0.9
            
            thickness = max(1, int(3 * scale1))
            
            # Fade alpha
            alpha = 0.3 + progress1 * 0.5
            
            dash_overlay = overlay.copy()
            cv2.line(dash_overlay, (vp_x, y1), (vp_x, y2), Colors.LANE_MARK, thickness)
            cv2.addWeighted(dash_overlay, alpha, overlay, 1 - alpha, 0, overlay)
        
        # Side lane markers (dotted)
        for side in [-1, 1]:
            offset = int(w * 0.15)
            for i in range(num_dashes):
                progress = i / num_dashes
                y = int(vp_y + (h - vp_y) * progress)
                
                # X position with perspective
                center_to_edge = (side * (right_bottom - w//2)) if side > 0 else (side * (w//2 - left_bottom))
                x = vp_x + int(center_to_edge * progress * 0.6)
                
                # Draw small marker
                size = max(2, int(4 * progress))
                alpha = 0.2 + progress * 0.4
                
                marker_overlay = overlay.copy()
                cv2.circle(marker_overlay, (x, y), size, Colors.LANE_MARK, -1)
                cv2.addWeighted(marker_overlay, alpha, overlay, 1 - alpha, 0, overlay)
        
        return overlay
    
    def _draw_distance_markers(self, frame, detections):
        """Draw distance markers on the road for detected objects."""
        h, w = frame.shape[:2]
        vp_y = h - int(h * 0.30) - 20
        
        for det in detections:
            if det.position != "CENTER":
                continue
            
            cx, cy = det.center
            
            # Map to road position
            road_progress = (cy - vp_y) / (h - vp_y)
            road_progress = max(0, min(1, road_progress))
            
            if road_progress < 0.1:
                continue
            
            # Draw distance arc on road
            arc_y = int(vp_y + (h - vp_y) * road_progress * 0.8)
            arc_width = int(50 + 150 * road_progress)
            
            # Distance arc
            color = self._threat_color(det.threat)
            cv2.ellipse(frame, (w // 2, arc_y), (arc_width, 15), 0, 0, 180, color, 2)
            
            # Distance text
            dist_text = f"{det.distance:.0f}m"
            text_size = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = w // 2 - text_size[0] // 2
            
            # Background
            cv2.rectangle(frame, (text_x - 5, arc_y - 20), 
                         (text_x + text_size[0] + 5, arc_y - 5), Colors.NEAR_BLACK, -1)
            cv2.putText(frame, dist_text, (text_x, arc_y - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    def _threat_color(self, threat):
        """Get color for threat level."""
        return {
            ThreatLevel.NONE: Colors.CYAN,
            ThreatLevel.LOW: Colors.GREEN,
            ThreatLevel.MEDIUM: Colors.YELLOW,
            ThreatLevel.HIGH: Colors.ORANGE,
            ThreatLevel.CRITICAL: Colors.RED
        }.get(threat, Colors.CYAN)
    
    def _draw_detection_box(self, frame, det):
        """Draw clean, minimal detection box."""
        x1, y1, x2, y2 = det.bbox
        color = self._threat_color(det.threat)
        
        # Corner length
        length = min(25, min(x2 - x1, y2 - y1) // 4)
        thick = 2
        
        # Just corners - very clean
        # Top left
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, thick)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, thick)
        # Top right
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, thick)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, thick)
        # Bottom left
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, thick)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, thick)
        # Bottom right
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, thick)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, thick)
        
        # Simple label above
        label = f"{det.label}  {det.distance:.0f}m"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        
        text_size = cv2.getTextSize(label, font, scale, 1)[0]
        text_x = x1
        text_y = y1 - 8
        
        if text_y < 20:
            text_y = y2 + 18
        
        # Label background
        pad = 4
        cv2.rectangle(frame, (text_x - pad, text_y - text_size[1] - pad),
                     (text_x + text_size[0] + pad, text_y + pad), Colors.NEAR_BLACK, -1)
        cv2.rectangle(frame, (text_x - pad, text_y - text_size[1] - pad),
                     (text_x + text_size[0] + pad, text_y + pad), color, 1)
        
        cv2.putText(frame, label, (text_x, text_y), font, scale, color, 1, cv2.LINE_AA)
        
        # Pulsing effect for critical threats
        if det.threat == ThreatLevel.CRITICAL:
            pulse = abs(math.sin(self.frame_num * 0.15))
            if pulse > 0.5:
                cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), Colors.RED, 2)
    
    def _draw_lane_overlay(self, frame, lane_data):
        """Draw detected lane lines."""
        if not lane_data.detected:
            return frame
        
        overlay = frame.copy()
        
        # Left lane
        if lane_data.left_points:
            pts = lane_data.left_points
            cv2.line(overlay, pts[0], pts[1], Colors.GREEN, 4)
        
        # Right lane
        if lane_data.right_points:
            pts = lane_data.right_points
            cv2.line(overlay, pts[0], pts[1], Colors.GREEN, 4)
        
        # Blend
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        return frame
    
    def _draw_top_bar(self, frame, fps):
        """Draw minimal top status bar."""
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
        
        # Time - left
        time_str = datetime.now().strftime("%H:%M")
        cv2.putText(frame, time_str, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, Colors.WHITE, 1, cv2.LINE_AA)
        
        # Title - center
        title = "AI DRIVING ASSISTANT"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        title_x = (self.w - title_size[0]) // 2
        cv2.putText(frame, title, (title_x, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.55, Colors.CYAN, 1, cv2.LINE_AA)
        
        # FPS - right
        fps_color = Colors.GREEN if fps > 20 else Colors.YELLOW if fps > 10 else Colors.RED
        cv2.putText(frame, f"{int(fps)} FPS", (self.w - 80, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, fps_color, 1, cv2.LINE_AA)
    
    def _draw_bottom_panel(self, frame):
        """Draw clean bottom instruction panel."""
        panel_h = 90
        panel_top = self.h - panel_h
        
        # Gradient background
        overlay = frame.copy()
        for y in range(panel_h):
            progress = y / panel_h
            alpha = progress ** 1.5
            color = tuple(int(c * alpha) for c in Colors.NEAR_BLACK)
            cv2.line(overlay, (0, panel_top + y), (self.w, panel_top + y), color, 1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Top line
        cv2.line(frame, (0, panel_top), (self.w, panel_top), Colors.DARK_GRAY, 1)
        
        # Side accents
        accent_w = 100
        cv2.line(frame, (0, panel_top), (accent_w, panel_top), self.alert_color, 2)
        cv2.line(frame, (self.w - accent_w, panel_top), (self.w, panel_top), self.alert_color, 2)
        
        # Main instruction
        font = cv2.FONT_HERSHEY_DUPLEX
        main_size = cv2.getTextSize(self.main_text, font, 1.3, 2)[0]
        main_x = (self.w - main_size[0]) // 2
        main_y = panel_top + 45
        
        # Shadow
        cv2.putText(frame, self.main_text, (main_x + 2, main_y + 2), font,
                   1.3, Colors.BLACK, 2, cv2.LINE_AA)
        # Text
        cv2.putText(frame, self.main_text, (main_x, main_y), font,
                   1.3, self.alert_color, 2, cv2.LINE_AA)
        
        # Sub instruction
        sub_size = cv2.getTextSize(self.sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        sub_x = (self.w - sub_size[0]) // 2
        sub_y = panel_top + 72
        
        cv2.putText(frame, self.sub_text, (sub_x, sub_y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.55, Colors.LIGHT_GRAY, 1, cv2.LINE_AA)
    
    def _draw_side_indicators(self, frame, detections):
        """Draw side traffic indicators."""
        left_count = sum(1 for d in detections if d.position == "LEFT")
        right_count = sum(1 for d in detections if d.position == "RIGHT")
        
        indicator_y = self.h // 2
        
        # Left indicator
        if left_count > 0:
            pts = np.array([
                [15, indicator_y],
                [35, indicator_y - 20],
                [35, indicator_y + 20]
            ], np.int32)
            cv2.fillPoly(frame, [pts], Colors.YELLOW)
            cv2.putText(frame, str(left_count), (40, indicator_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.YELLOW, 1, cv2.LINE_AA)
        
        # Right indicator
        if right_count > 0:
            pts = np.array([
                [self.w - 15, indicator_y],
                [self.w - 35, indicator_y - 20],
                [self.w - 35, indicator_y + 20]
            ], np.int32)
            cv2.fillPoly(frame, [pts], Colors.YELLOW)
            cv2.putText(frame, str(right_count), (self.w - 55, indicator_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.YELLOW, 1, cv2.LINE_AA)
    
    def _draw_center_reticle(self, frame):
        """Draw minimal center reticle."""
        cx, cy = self.w // 2, self.h // 2 - 30
        
        size = 20
        gap = 6
        
        color = (*Colors.CYAN[:3],)
        
        # Simple cross with gap
        cv2.line(frame, (cx - size, cy), (cx - gap, cy), color, 1)
        cv2.line(frame, (cx + gap, cy), (cx + size, cy), color, 1)
        cv2.line(frame, (cx, cy - size), (cx, cy - gap), color, 1)
        cv2.line(frame, (cx, cy + gap), (cx, cy + size), color, 1)
        
        # Tiny center dot
        cv2.circle(frame, (cx, cy), 2, color, -1)
    
    def _draw_mini_map(self, frame, detections):
        """Draw a small, clean mini-map."""
        map_w, map_h = 120, 100
        map_x = self.w - map_w - 15
        map_y = 60
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (map_x, map_y), (map_x + map_w, map_y + map_h), 
                     Colors.NEAR_BLACK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, (map_x, map_y), (map_x + map_w, map_y + map_h), 
                     Colors.DARK_GRAY, 1)
        
        # Road shape
        road_pts = np.array([
            [map_x + map_w // 2, map_y + 10],
            [map_x + map_w - 20, map_y + map_h - 10],
            [map_x + 20, map_y + map_h - 10]
        ], np.int32)
        cv2.polylines(frame, [road_pts], True, Colors.ROAD_EDGE, 1)
        
        # Vehicle (self) - triangle at bottom
        car_x = map_x + map_w // 2
        car_y = map_y + map_h - 20
        car_pts = np.array([
            [car_x, car_y - 8],
            [car_x - 5, car_y + 4],
            [car_x + 5, car_y + 4]
        ], np.int32)
        cv2.fillPoly(frame, [car_pts], Colors.CYAN)
        
        # Plot detections
        for det in detections:
            # Map position
            rel_x = (det.center[0] / self.w - 0.5) * 0.8
            rel_y = 1 - (det.center[1] / self.h)
            
            dot_x = int(map_x + map_w // 2 + rel_x * map_w)
            dot_y = int(map_y + map_h - 15 - rel_y * (map_h - 30))
            
            dot_x = max(map_x + 5, min(map_x + map_w - 5, dot_x))
            dot_y = max(map_y + 5, min(map_y + map_h - 5, dot_y))
            
            color = self._threat_color(det.threat)
            cv2.circle(frame, (dot_x, dot_y), 4, color, -1)
    
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
        
        if self.stable_count >= 6:
            self.main_text = main
            self.sub_text = sub
            self.alert_color = color
    
    def render(self, frame, detections, lane_data, fps):
        """Render the complete clean HUD."""
        self.frame_num += 1
        
        # 1. Lane overlay
        frame = self._draw_lane_overlay(frame, lane_data)
        
        # 2. Road perspective
        frame = self._draw_road_perspective(frame)
        
        # 3. Distance markers on road
        self._draw_distance_markers(frame, detections)
        
        # 4. Detection boxes
        for det in detections:
            self._draw_detection_box(frame, det)
        
        # 5. Center reticle
        self._draw_center_reticle(frame)
        
        # 6. Side indicators
        self._draw_side_indicators(frame, detections)
        
        # 7. Mini map
        self._draw_mini_map(frame, detections)
        
        # 8. Top bar
        self._draw_top_bar(frame, fps)
        
        # 9. Bottom panel
        self._draw_bottom_panel(frame)
        
        return frame


# ═══════════════════════════════════════════════════════════════════════════════
#                              DRIVING LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

class DrivingLogic:
    def analyze(self, detections, lane_data):
        """Generate driving instructions."""
        
        # Find threats
        critical = [d for d in detections if d.threat == ThreatLevel.CRITICAL]
        high = [d for d in detections if d.threat == ThreatLevel.HIGH]
        center = [d for d in detections if d.position == "CENTER" and d.threat.value >= 2]
        
        max_threat = ThreatLevel.NONE
        for d in detections:
            if d.threat.value > max_threat.value:
                max_threat = d.threat
        
        # Critical - brake immediately
        if critical:
            d = critical[0]
            if d.class_id in [0, 16, 17, 18, 19]:  # Pedestrian/animal
                return "⚠ STOP", f"{d.label} on road!", Colors.RED, max_threat
            return "⚠ BRAKE", f"Collision warning - {d.distance:.0f}m", Colors.RED, max_threat
        
        # High threat
        if high:
            d = high[0]
            return "SLOW DOWN", f"{d.label} ahead - {d.distance:.0f}m", Colors.ORANGE, max_threat
        
        # Center obstacle - check overtake options
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
        
        # Lane departure
        if lane_data.detected and abs(lane_data.center_offset) > 0.3:
            if lane_data.center_offset > 0:
                return "DRIFT LEFT", "Steer right to correct", Colors.YELLOW, max_threat
            else:
                return "DRIFT RIGHT", "Steer left to correct", Colors.YELLOW, max_threat
        
        # All clear
        return "CLEAR", "Road ahead is clear", Colors.CYAN, max_threat


# ═══════════════════════════════════════════════════════════════════════════════
#                              MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

class DrivingAssistant:
    def __init__(self):
        print("\n" + "═" * 50)
        print("  CLEAN AI DRIVING ASSISTANT v4.0")
        print("  Optimized for Indian Roads")
        print("═" * 50 + "\n")
        
        # Source
        print("[1/4] Opening video source...")
        source = Config.TEST_VIDEO_PATH if Config.USE_TEST_VIDEO else Config.LIVE_CAMERA_INDEX
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open: {source}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        
        print(f"      Source: {'Test Video' if Config.USE_TEST_VIDEO else 'Live Camera'}")
        print(f"      Resolution: {self.width}x{self.height}")
        
        # Components
        print("[2/4] Loading detection engine...")
        self.detector = DetectionEngine(self.width, self.height)
        
        print("[3/4] Initializing lane detection...")
        self.lane_detector = LaneDetector(self.width, self.height)
        
        print("[4/4] Setting up HUD...")
        self.hud = CleanHUD(self.width, self.height)
        self.logic = DrivingLogic()
        
        # Output
        self.writer = None
        if Config.SAVE_OUTPUT:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(Config.OUTPUT_PATH, fourcc, self.fps,
                                          (self.width, self.height))
        
        self.fps_tracker = deque(maxlen=30)
        
        print("\n" + "═" * 50)
        print("  Ready! Press Q to quit")
        print("═" * 50 + "\n")
    
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
                
                # Detect
                detections = self.detector.detect(frame)
                lane_data = self.lane_detector.detect(frame)
                
                # Logic
                main, sub, color, threat = self.logic.analyze(detections, lane_data)
                self.hud.update_instruction(main, sub, color, threat)
                
                # FPS
                elapsed = time.time() - start
                current_fps = 1 / elapsed if elapsed > 0 else 0
                self.fps_tracker.append(current_fps)
                avg_fps = sum(self.fps_tracker) / len(self.fps_tracker)
                
                # Render
                output = self.hud.render(frame, detections, lane_data, avg_fps)
                
                # Save
                if self.writer:
                    self.writer.write(output)
                
                # Display
                cv2.imshow("AI Driving Assistant", output)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        print("\nShutting down...")
        if self.writer:
            self.writer.release()
            print(f"  Saved: {Config.OUTPUT_PATH}")
        self.cap.release()
        cv2.destroyAllWindows()
        print("  Done!")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        app = DrivingAssistant()
        app.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
