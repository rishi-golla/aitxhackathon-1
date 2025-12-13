import json
import cv2
import uvicorn
import os
from fastapi import FastAPI
from ultralytics import YOLOWorld
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any

# --- 1. Load Rules (Existing Logic) ---
try:
    with open('osha_rules.json', 'r') as f:
        OSHA_RULES = json.load(f)
except FileNotFoundError:
    print("Warning: osha_rules.json not found. Rules will be empty.")
    OSHA_RULES = []

def check_violation(detected_objects: List[str]) -> List[Dict[str, Any]]:
    """
    detected_objects: List of strings from YOLO-World 
    e.g. ['bare_hand', 'industrial_machine', 'helmet']
    """
    violations_found = []

    for rule in OSHA_RULES:
        # Check if ANY trigger is present
        trigger_hit = any(t in detected_objects for t in rule['triggers'])
        
        # Check if ANY required context is present (or if none is required)
        if not rule['required_context']:
            context_hit = True
        else:
            context_hit = any(c in detected_objects for c in rule['required_context'])

        # If we have a Trigger + Context, it's a violation
        if trigger_hit and context_hit:
            violations_found.append({
                "code": rule['code'],
                "title": rule['title'],
                "text": rule['legal_text'],
                "penalty": rule['penalty_max']
            })

    return violations_found

# --- 2. FastAPI App Setup ---
app = FastAPI(title="OSHA-Vision Backend")

# Global state
current_violations: List[Dict[str, Any]] = []
# Initialize YOLO-World Model
# Using yolov8s-world.pt for balance of speed/accuracy on DGX Spark
model = YOLOWorld('yolov8s-world.pt') 
model.set_classes(["bare_hand", "gloved_hand", "safety_glasses", "face", "industrial_machine"])
# model.set_classes(["bare_hand", "gloved_hand", "safety_glasses", "face", "industrial_machine"])
def generate_frames():
    global current_violations
    
    # Determine video source: Env var 'VIDEO_SOURCE' or default to 0 (webcam)
    video_source = os.getenv("VIDEO_SOURCE", "0")
    
    # If it's a digit, convert to int (for webcam index)
    if video_source.isdigit():
        video_source = int(video_source)
        
    print(f"Starting video stream from source: {video_source}")
    cap = cv2.VideoCapture(video_source) 
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        return

    while True:
        success, frame = cap.read()
        if not success:
            # If reading from a file, loop it
            if isinstance(video_source, str) and os.path.exists(video_source):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        
        # --- AI INFERENCE ---
        # --- AI INFERENCE ---
        results = model.predict(frame, conf=0.15, verbose=False)
        result = results[0]
        
        # Extract detected class names
        detected_objects = []
        if result.boxes:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = result.names[cls_id]
                detected_objects.append(class_name)
                
                # Draw Bounding Box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw all detections in Green
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- RULE CHECKING ---
        current_violations = check_violation(detected_objects)

        # --- VISUALIZATION OF VIOLATIONS ---
        if current_violations:
            # Draw a big red border or warning
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
            cv2.putText(frame, f"VIOLATION DETECTED: {len(current_violations)}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/status")
async def get_status():
    return {"violations": current_violations}

if __name__ == "__main__":
    # Run on 0.0.0.0 to be accessible if needed, port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
