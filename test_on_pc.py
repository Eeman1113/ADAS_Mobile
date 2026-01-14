import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
USE_TEST_VIDEO = True
TEST_VIDEO_PATH = "test.mp4"
SAVE_OUTPUT = True
VIDEO_SOURCE = 0 

# Aesthetic Colors (BGR Format)
NEON_CYAN = (255, 255, 0)
NEON_GREEN = (50, 255, 50)
WARNING_RED = (0, 0, 255)
ALERT_YELLOW = (0, 255, 255)
PURE_WHITE = (255, 255, 255)
DARK_OVERLAY = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_DUPLEX

class DrivingAssistant:
    def __init__(self):
        print("Initializing AI Co-Pilot HUD...")
        
        # Determine Source
        if USE_TEST_VIDEO:
            print(f"Testing Mode: Loading {TEST_VIDEO_PATH}...")
            self.source = TEST_VIDEO_PATH
        else:
            print(f"Live Mode: connecting to camera {VIDEO_SOURCE}...")
            self.source = VIDEO_SOURCE

        self.model = YOLO('yolov8n.pt') 
        self.cap = cv2.VideoCapture(self.source)
        
        if not USE_TEST_VIDEO:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source {self.source}.")

        # Recording Setup
        self.out = None
        if SAVE_OUTPUT:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or math.isnan(fps): fps = 30.0
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

        # Smoothing Variables
        self.last_raw_main = "SYSTEM READY"
        self.consistency_counter = 0
        self.CONSISTENCY_THRESHOLD = 6
        self.display_main = "SYSTEM READY"
        self.display_sub = "Initialize Drive"
        self.display_color = PURE_WHITE

    def draw_text_with_outline(self, img, text, pos, font_scale, color, thickness=1):
        """Draws text with a black outline for better visibility."""
        x, y = pos
        cv2.putText(img, text, (x, y), FONT, font_scale, (0, 0, 0), thickness + 3) # Outline
        cv2.putText(img, text, (x, y), FONT, font_scale, color, thickness)       # Text

    def draw_corner_brackets(self, img, bbox, color, thickness=2, length=15):
        """Draws futuristic corner brackets instead of a full box."""
        x1, y1, x2, y2 = bbox
        # Top Left
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
        # Top Right
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
        # Bottom Left
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
        # Bottom Right
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

    def draw_crosshair(self, img, center, color=NEON_CYAN, size=20):
        """Draws a tactical center crosshair."""
        cx, cy = center
        # Center gap
        gap = 5
        cv2.line(img, (cx - size, cy), (cx - gap, cy), color, 1)
        cv2.line(img, (cx + gap, cy), (cx + size, cy), color, 1)
        cv2.line(img, (cx, cy - size), (cx, cy - gap), color, 1)
        cv2.line(img, (cx, cy + gap), (cx, cy + size), color, 1)
        # Central Dot
        cv2.circle(img, (cx, cy), 2, color, -1)

    def detect_objects(self, frame):
        results = self.model(frame, stream=True, verbose=False)
        detections = []
        height, width, _ = frame.shape
        center_x = width // 2
        
        target_classes = [2, 3, 5, 7] # Car, Motorbike, Bus, Truck
        danger_level = 0 
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls in target_classes and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    obj_center_x = (x1 + x2) // 2
                    obj_w = x2 - x1
                    obj_h = y2 - y1
                    relative_area = (obj_w * obj_h) / (width * height)
                    
                    position = "Center"
                    if obj_center_x < center_x - (width * 0.15): position = "Left"
                    elif obj_center_x > center_x + (width * 0.15): position = "Right"

                    is_danger = False
                    if position == "Center":
                        if relative_area > 0.15: 
                            danger_level = max(danger_level, 2)
                            is_danger = True
                        elif relative_area > 0.05:
                            danger_level = max(danger_level, 1)

                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "label": self.model.names[cls].upper(),
                        "position": position,
                        "area": relative_area,
                        "is_danger": is_danger
                    })
        return detections, danger_level

    def detect_lanes(self, frame):
        height, width, _ = frame.shape
        # ROI focused on road
        polygon = np.array([[(0, height), (width, height), (width, int(height * 0.6)), (0, int(height * 0.6))]], np.int32)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        mask_poly = np.zeros_like(edges)
        cv2.fillPoly(mask_poly, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask_poly)
        
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=150)
        
        steering_advice = "Lane Hold"
        detected_lines = []
        
        if lines is not None:
            left_lines = []
            right_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1: continue 
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) > 0.3:
                    detected_lines.append(line[0])
                    if slope < 0: left_lines.append(line)
                    else: right_lines.append(line)
            
            if len(right_lines) > 0 and len(left_lines) == 0: steering_advice = "Drift Left -> Steer R"
            elif len(left_lines) > 0 and len(right_lines) == 0: steering_advice = "Drift Right -> Steer L"
        
        return steering_advice, detected_lines

    def generate_instruction(self, detections, danger_level, lane_advice):
        main = "CRUISING"
        sub = "Path Clear"
        color = NEON_CYAN

        if danger_level == 2:
            main = "BRAKE NOW"
            sub = "Collision Imminent"
            color = WARNING_RED
            return main, sub, color

        car_ahead = any(d['position'] == 'Center' for d in detections)
        car_left = any(d['position'] == 'Left' for d in detections)
        car_right = any(d['position'] == 'Right' for d in detections)

        if danger_level == 1:
            main = "SLOW DOWN"
            sub = "Traffic Ahead"
            color = ALERT_YELLOW
        elif car_ahead:
            if not car_right:
                main = "OVERTAKE RIGHT"
                sub = "Right Lane Open"
                color = NEON_GREEN
            elif not car_left:
                main = "OVERTAKE LEFT"
                sub = "Left Lane Open"
                color = NEON_GREEN
            else:
                main = "MAINTAIN DIST"
                sub = "Lanes Blocked"
                color = ALERT_YELLOW
        elif "Drift" in lane_advice:
            main = "CORRECT LANE"
            sub = lane_advice
            color = ALERT_YELLOW
        
        return main, sub, color

    def update_smooth_instruction(self, raw_main, raw_sub, raw_color, danger_level):
        if danger_level == 2:
            self.display_main, self.display_sub, self.display_color = raw_main, raw_sub, raw_color
            self.consistency_counter = 0
            self.last_raw_main = raw_main
            return

        if raw_main == self.last_raw_main:
            self.consistency_counter += 1
        else:
            self.consistency_counter = 0
            self.last_raw_main = raw_main

        if self.consistency_counter > self.CONSISTENCY_THRESHOLD:
            self.display_main, self.display_sub, self.display_color = raw_main, raw_sub, raw_color

    def draw_hud(self, frame, main_text, sub_text, color, detections, lane_lines):
        h, w, _ = frame.shape
        overlay = frame.copy()
        
        # 1. Lane Projection (Transparent)
        lane_overlay = frame.copy()
        for line in lane_lines:
            x1, y1, x2, y2 = line
            cv2.line(lane_overlay, (x1, y1), (x2, y2), NEON_GREEN, 10) # Thick lines
        cv2.addWeighted(lane_overlay, 0.3, frame, 0.7, 0, frame) # Blend

        # 2. Object Brackets
        for d in detections:
            box_color = WARNING_RED if d['is_danger'] else NEON_CYAN
            self.draw_corner_brackets(frame, d['bbox'], box_color)
            
            # Tiny label
            x1, y1, _, _ = d['bbox']
            label = f"{d['label']} {int(d['area']*100)}%"
            self.draw_text_with_outline(frame, label, (x1, y1-10), 0.4, box_color, 1)

        # 3. Center Crosshair
        self.draw_crosshair(frame, (w//2, h//2))

        # 4. Top Status Bar (Glass Effect)
        cv2.rectangle(overlay, (0, 0), (w, 60), DARK_OVERLAY, -1)
        
        # 5. Bottom Dashboard (Glass Effect)
        cv2.rectangle(overlay, (0, h-100), (w, h), DARK_OVERLAY, -1)
        
        # Apply Glass Transparency
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 6. Top Bar Data
        time_str = datetime.now().strftime("%H:%M:%S")
        self.draw_text_with_outline(frame, f"T: {time_str}", (20, 40), 0.6, NEON_CYAN)
        self.draw_text_with_outline(frame, f"OBJ: {len(detections)}", (w-150, 40), 0.6, NEON_CYAN)
        mode = "TEST FILE" if USE_TEST_VIDEO else "LIVE CAM"
        self.draw_text_with_outline(frame, f"MODE: {mode}", (w//2 - 80, 40), 0.6, PURE_WHITE)

        # 7. Main Dashboard Instructions
        # Main instruction (Big, Centered)
        t_size = cv2.getTextSize(main_text, FONT, 1.5, 2)[0]
        t_x = (w - t_size[0]) // 2
        self.draw_text_with_outline(frame, main_text, (t_x, h-50), 1.5, color, 2)
        
        # Sub instruction (Small, below)
        s_size = cv2.getTextSize(sub_text, FONT, 0.6, 1)[0]
        s_x = (w - s_size[0]) // 2
        self.draw_text_with_outline(frame, sub_text, (s_x, h-25), 0.6, PURE_WHITE, 1)

        return frame

    def run(self):
        prev_time = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                if USE_TEST_VIDEO:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else: break

            # Detection
            detections, danger_level = self.detect_objects(frame)
            steering_advice, lane_lines = self.detect_lanes(frame)
            
            # Logic
            raw_main, raw_sub, raw_color = self.generate_instruction(detections, danger_level, steering_advice)
            self.update_smooth_instruction(raw_main, raw_sub, raw_color, danger_level)
            
            # Draw
            frame = self.draw_hud(frame, self.display_main, self.display_sub, self.display_color, detections, lane_lines)
            
            if self.out: self.out.write(frame)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
            prev_time = curr_time
            self.draw_text_with_outline(frame, f"FPS: {int(fps)}", (120, 40), 0.6, NEON_GREEN)

            cv2.imshow('AI Co-Pilot HUD', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        if self.out: self.out.release()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        app = DrivingAssistant()
        app.run()
    except Exception as e:
        print(f"Error: {e}")