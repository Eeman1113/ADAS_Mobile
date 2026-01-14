import cv2
import numpy as np
from ultralytics import YOLO
import math
import time

# Configuration
VIDEO_SOURCE = 0 

# HUD Settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
HUD_BG_COLOR = (0, 0, 0)

class DrivingAssistant:
    def __init__(self, source=0):
        print("Initializing AI Co-Pilot...")
        
        # Load the YOLOv8 Nano model (Fastest and lightweight for laptops)
        # It will auto-download 'yolov8n.pt' on the first run.
        self.model = YOLO('yolov8n.pt') 
        
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source {source}. Try changing VIDEO_SOURCE index.")

        # State variables for smoothing instructions
        self.last_instruction = "Ready"
        self.instruction_buffer = []
        self.lane_center_history = []

    def detect_objects(self, frame):
        """
        Runs YOLOv8 inference to find vehicles and pedestrians.
        Returns a list of significant detections with logic data.
        """
        results = self.model(frame, stream=True, verbose=False)
        
        detections = []
        height, width, _ = frame.shape
        center_x = width // 2
        
        # Classes we care about (COCO dataset indices)
        # 2: car, 3: motorcycle, 5: bus, 7: truck
        target_classes = [2, 3, 5, 7] 

        danger_level = 0 # 0: Safe, 1: Caution, 2: DANGER
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls in target_classes and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Calculate relative position
                    obj_center_x = (x1 + x2) // 2
                    obj_width = x2 - x1
                    obj_height = y2 - y1
                    
                    # Estimate distance based on object size in frame (heuristic)
                    relative_area = (obj_width * obj_height) / (width * height)
                    
                    # Determine position relative to our car
                    position = "Center"
                    if obj_center_x < center_x - (width * 0.15):
                        position = "Left"
                    elif obj_center_x > center_x + (width * 0.15):
                        position = "Right"

                    # Danger Logic
                    is_danger = False
                    if position == "Center":
                        if relative_area > 0.15: # Very close
                            danger_level = max(danger_level, 2)
                            is_danger = True
                        elif relative_area > 0.05: # Medium distance
                            danger_level = max(danger_level, 1)

                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "label": self.model.names[cls],
                        "position": position,
                        "area": relative_area,
                        "is_danger": is_danger
                    })

        return detections, danger_level

    def detect_lanes(self, frame):
        """
        Simple lane detection using edge detection and Hough transform.
        Returns a suggested steering adjustment and lines to draw.
        """
        height, width, _ = frame.shape
        
        # Region of Interest (ROI) - Focus on the bottom half of the road
        polygon = np.array([[
            (0, height),
            (width, height),
            (width, int(height * 0.6)),
            (0, int(height * 0.6)),
        ]], np.int32)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # Mask everything except the road area
        mask_poly = np.zeros_like(edges)
        cv2.fillPoly(mask_poly, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask_poly)
        
        # Detect lines
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=150)
        
        steering_advice = "Hold Lane"
        lane_center_offset = 0
        detected_lines = []
        
        if lines is not None:
            left_lines = []
            right_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1: continue # vertical line
                slope = (y2 - y1) / (x2 - x1)
                
                # Filter vertical-ish lines (lanes usually have steep slopes)
                if abs(slope) > 0.3:
                    detected_lines.append(line[0]) # Store for drawing
                    if slope < 0: # Left lane
                        left_lines.append(line)
                    else: # Right lane
                        right_lines.append(line)
            
            # Simple logic
            if len(right_lines) > 0 and len(left_lines) == 0:
                steering_advice = "Drifting Left -> Correct Right"
                lane_center_offset = -1
            elif len(left_lines) > 0 and len(right_lines) == 0:
                steering_advice = "Drifting Right -> Correct Left"
                lane_center_offset = 1
        
        return steering_advice, lane_center_offset, detected_lines

    def generate_instruction(self, detections, danger_level, lane_advice):
        """
        The 'Brain' that combines visual data into textual commands.
        """
        main_text = "CRUISING"
        sub_text = "Maintain Speed"
        color = GREEN

        # Priority 1: Immediate Collision Danger
        if danger_level == 2:
            main_text = "BRAKE NOW"
            sub_text = "Obstacle Ahead!"
            color = RED
            return main_text, sub_text, color

        # Priority 2: Traffic flow
        car_ahead = any(d['position'] == 'Center' for d in detections)
        car_left = any(d['position'] == 'Left' for d in detections)
        car_right = any(d['position'] == 'Right' for d in detections)

        if danger_level == 1:
            main_text = "SLOW DOWN"
            sub_text = "Traffic Ahead"
            color = YELLOW
        elif car_ahead:
            # Logic for overtaking
            if not car_right:
                main_text = "OVERTAKE RIGHT"
                sub_text = "Right Lane Clear"
                color = CYAN
            elif not car_left:
                main_text = "OVERTAKE LEFT"
                sub_text = "Left Lane Clear"
                color = CYAN
            else:
                main_text = "FOLLOW TRAFFIC"
                sub_text = "Lanes Blocked"
                color = YELLOW
        elif "Drifting" in lane_advice:
            main_text = "ADJUST STEERING"
            sub_text = lane_advice
            color = YELLOW
        else:
            main_text = "CLEAR ROAD"
            sub_text = "Gear Up / Cruise"
            color = GREEN
            
        return main_text, sub_text, color

    def draw_hud(self, frame, main_text, sub_text, color, detections, lane_lines):
        """
        Draws the sci-fi style interface with semi-transparent overlays.
        """
        h, w, _ = frame.shape
        overlay = frame.copy()
        
        # 1. Draw Lanes
        for line in lane_lines:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), GREEN, 3)
            
        # 2. Draw Bounding Boxes with Labels
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            box_color = RED if d['is_danger'] else CYAN
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Label background
            label = f"{d['label']} {int(d['area']*100)}%"
            t_size = cv2.getTextSize(label, FONT, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1-20), (x1+t_size[0]+10, y1), box_color, -1)
            cv2.putText(frame, label, (x1+5, y1-5), FONT, 0.5, (0,0,0), 1)

        # 3. Semi-transparent Dashboard
        cv2.rectangle(overlay, (0, h-120), (w, h), HUD_BG_COLOR, -1)
        
        # Apply transparency
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # 4. Main Instruction Center
        text_size = cv2.getTextSize(main_text, FONT, 2, 3)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, main_text, (text_x, h-50), FONT, 2, color, 3)
        
        # 5. Sub Instruction
        sub_size = cv2.getTextSize(sub_text, FONT, 0.8, 2)[0]
        sub_x = (w - sub_size[0]) // 2
        cv2.putText(frame, sub_text, (sub_x, h-20), FONT, 0.8, WHITE, 2)

        # 6. Side Stats
        cv2.putText(frame, "SYS: ONLINE", (20, h-80), FONT, 0.6, GREEN, 2)
        cv2.putText(frame, "CAM: IPHONE", (20, h-50), FONT, 0.6, GREEN, 2)
        cv2.putText(frame, f"OBJECTS: {len(detections)}", (w-200, h-80), FONT, 0.6, CYAN, 2)
        
        return frame

    def run(self):
        prev_time = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame. Check camera connection.")
                break

            # 1. Analysis
            detections, danger_level = self.detect_objects(frame)
            steering_advice, lane_offset, lane_lines = self.detect_lanes(frame)
            
            # 2. Decision Making
            main_text, sub_text, color = self.generate_instruction(detections, danger_level, steering_advice)
            
            # 3. Visualization
            frame = self.draw_hud(frame, main_text, sub_text, color, detections, lane_lines)
            
            # FPS Calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), FONT, 1, GREEN, 2)

            # Display
            cv2.imshow('AI Co-Pilot Assistant', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        app = DrivingAssistant(source=VIDEO_SOURCE)
        app.run()
    except Exception as e:
        print(f"Error: {e}")