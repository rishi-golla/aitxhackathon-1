"""
OSHA Vision Backend - Multi-Camera Safety Monitoring System

Serves multiple video streams from Egocentric-10K dataset with YOLO-World
detection overlays and real-time OSHA violation checking.
"""

import json
import cv2
import uvicorn
import os
import torch
import sys
import threading
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLOWorld
from typing import List, Dict, Any, Optional
import base64
import time

# Add root directory to sys.path to allow importing archivist
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from archivist.vector_store import VectorStore
except ImportError:
    print("Warning: Could not import archivist.vector_store. Search will be disabled.")
    VectorStore = None

# --- Configuration ---
VIDEO_DIR = Path(root_dir) / "data"
CONFIG_DIR = Path(root_dir) / "config"
MODELS_DIR = Path(root_dir) / "models"

# --- 1. Load OSHA Rules ---
try:
    with open(CONFIG_DIR / 'osha_rules.json', 'r') as f:
        OSHA_RULES = json.load(f)
except FileNotFoundError:
    print("Warning: config/osha_rules.json not found. Rules will be empty.")
    OSHA_RULES = []


def check_violation(detected_objects: List[str]) -> List[Dict[str, Any]]:
    """
    Check detected objects against OSHA rules.

    Args:
        detected_objects: List of class names from YOLO-World

    Returns:
        List of violation dictionaries
    """
    violations_found = []

    for rule in OSHA_RULES:
        trigger_hit = any(t in detected_objects for t in rule['triggers'])

        if not rule['required_context']:
            context_hit = True
        else:
            context_hit = any(c in detected_objects for c in rule['required_context'])

        if trigger_hit and context_hit:
            violations_found.append({
                "code": rule['code'],
                "title": rule['title'],
                "text": rule['legal_text'],
                "penalty": rule['penalty_max']
            })

    return violations_found


# --- 2. FastAPI App Setup ---
app = FastAPI(
    title="OSHA-Vision Backend",
    description="Multi-camera safety monitoring with YOLO-World detection",
    version="1.0.0"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Global State ---
cameras: Dict[str, Dict[str, Any]] = {}
camera_violations: Dict[str, List[Dict[str, Any]]] = {}
camera_summaries: Dict[str, Dict[str, Any]] = {}  # Cache for video summaries
model_lock = threading.Lock()
vlm_lock = threading.Lock()

# Initialize Vector Store
vector_store = None
if VectorStore:
    try:
        vector_store = VectorStore()
    except Exception as e:
        print(f"Warning: Could not initialize VectorStore: {e}")

# --- 4. Initialize YOLO-World Model ---
print("Initializing YOLO-World model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLOWorld(str(MODELS_DIR / 'yolov8s-world.pt'))
if device == 'cuda':
    model.to('cuda')
    print("Model moved to CUDA.")

model.set_classes([
    # Hand detection - multiple variants for better recall
    "bare_hand", "hand", "human hand", "fingers", "palm",
    # Safety equipment
    "gloved_hand", "glove", "work glove", "safety glove",
    "safety_glasses", "goggles", "eye_protection", "face_shield",
    # People
    "face", "person", "worker",
    # Industrial context
    "industrial_machine", "machine", "tool", "power tool", "grinder", "saw"
])
print("YOLO-World model ready.")

# --- 4b. Initialize VLM for Video Summarization (DGX Spark Superpower) ---
vlm_model = None
vlm_processor = None

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    print("Loading VLM for video summarization (DGX Spark)...")

    # Use Microsoft's Florence-2 - excellent for image captioning and runs well on GPU
    vlm_model_name = "microsoft/Florence-2-base"
    vlm_processor = AutoProcessor.from_pretrained(vlm_model_name, trust_remote_code=True)
    vlm_model = AutoModelForCausalLM.from_pretrained(
        vlm_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
    )
    if device == 'cuda':
        vlm_model = vlm_model.to('cuda')
    print(f"VLM loaded successfully on {device}")
except Exception as e:
    print(f"Warning: Could not load VLM model: {e}")
    print("Video summarization will use YOLO detections as fallback.")


def generate_video_summary(camera_id: str) -> str:
    """
    Generate an AI summary of the video content using VLM.
    Extracts a frame and describes what's happening.
    """
    global camera_summaries

    if camera_id not in cameras:
        return "Camera not found."

    cam = cameras[camera_id]
    video_path = cam["video_path"]

    # Check cache (summaries valid for 30 seconds)
    cached = camera_summaries.get(camera_id)
    if cached and (time.time() - cached.get("timestamp", 0)) < 30:
        return cached.get("summary", "Analyzing...")

    try:
        # Extract a frame from the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Unable to access video feed."

        # Get frame from middle of video for better context
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)

        success, frame = cap.read()
        cap.release()

        if not success or frame is None:
            return "Unable to capture frame."

        # Resize for faster processing
        frame = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use VLM if available
        if vlm_model is not None and vlm_processor is not None:
            from PIL import Image
            import numpy as np

            with vlm_lock:
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)

                # Generate description using Florence-2
                prompt = "<MORE_DETAILED_CAPTION>"
                inputs = vlm_processor(text=prompt, images=pil_image, return_tensors="pt")

                if device == 'cuda':
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}

                generated_ids = vlm_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=150,
                    num_beams=3,
                    do_sample=False
                )

                generated_text = vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # Post-process: Extract the actual caption
                if "<MORE_DETAILED_CAPTION>" in generated_text:
                    summary = generated_text.split("<MORE_DETAILED_CAPTION>")[-1].strip()
                else:
                    summary = generated_text.strip()

                # Add OSHA context
                detections = camera_violations.get(camera_id, [])
                if detections:
                    summary += f" [SAFETY ALERT: {len(detections)} OSHA violation(s) detected]"

        else:
            # Fallback: Use YOLO detections to generate summary
            with model_lock:
                results = model.predict(frame, conf=0.1, verbose=False)

            result = results[0]
            detected = []
            if result.boxes and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    class_name = result.names[cls_id]
                    if class_name not in detected:
                        detected.append(class_name)

            if detected:
                summary = f"Detected in frame: {', '.join(detected)}. "

                # Add context based on detections
                if any(d in ["hand", "bare_hand", "fingers", "palm"] for d in detected):
                    summary += "Worker hands visible - monitoring for PPE compliance. "
                if any(d in ["machine", "tool", "grinder", "saw"] for d in detected):
                    summary += "Industrial equipment in use - high-risk zone. "
                if any(d in ["person", "worker", "face"] for d in detected):
                    summary += "Worker present in frame. "

                violations = camera_violations.get(camera_id, [])
                if violations:
                    summary += f"[ALERT: {len(violations)} active violation(s)]"
            else:
                summary = "Monitoring video feed. No significant objects detected in current frame."

        # Cache the result
        camera_summaries[camera_id] = {
            "summary": summary,
            "timestamp": time.time()
        }

        return summary

    except Exception as e:
        print(f"Error generating summary for {camera_id}: {e}")
        return f"AI analysis in progress... (DGX Spark processing)"


# --- 5. Camera Discovery ---
def init_cameras():
    """Scan data/ folder for MP4 files and register as cameras."""
    global cameras, camera_violations

    if not VIDEO_DIR.exists():
        print(f"Warning: Video directory {VIDEO_DIR} does not exist.")
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        return

    mp4_files = sorted(VIDEO_DIR.glob("*.mp4"))

    if not mp4_files:
        print(f"No MP4 files found in {VIDEO_DIR}")
        return

    print(f"Found {len(mp4_files)} video files:")

    for i, video_path in enumerate(mp4_files):
        camera_id = f"cam-{i:02d}"
        label = video_path.stem.replace("_", " ").replace("-", " ").title()

        cameras[camera_id] = {
            "id": camera_id,
            "label": label,
            "video_path": str(video_path),
            "filename": video_path.name,
            "floor": (i % 5) + 1,  # Distribute across floors 1-5
            "status": "active"
        }
        camera_violations[camera_id] = []

        print(f"  [{camera_id}] {label} ({video_path.name})")


@app.on_event("startup")
async def startup():
    """Initialize cameras on server start."""
    init_cameras()
    print(f"\nOSHA Vision Backend ready with {len(cameras)} cameras")


# --- 6. Video Stream Generator ---
# Performance tuning
INFERENCE_INTERVAL = 5  # Run YOLO every N frames (reduced for more responsive detection)
TARGET_FPS = 15  # Target frame rate for smooth playback
FRAME_SKIP = 2  # Skip every N frames from video to reduce load
BOX_PERSISTENCE_FRAMES = 30  # Keep boxes visible for N frames after last detection

def generate_frames(camera_id: str):
    """
    Generate MJPEG frames with YOLO detection overlays.
    Loops video when reaching EOF.
    Optimized: runs inference every INFERENCE_INTERVAL frames.
    Boxes persist for BOX_PERSISTENCE_FRAMES to prevent flickering.
    """
    global camera_violations
    import time

    if camera_id not in cameras:
        return

    cam = cameras[camera_id]
    video_path = cam["video_path"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    print(f"Starting stream for {camera_id}: {video_path}")

    frame_count = 0
    last_boxes = []  # Cache last detection results
    last_detected_objects = []
    frames_since_detection = 0  # Track how long since we had detections
    frame_time = 1.0 / TARGET_FPS

    try:
        while True:
            start_time = time.time()

            # Skip frames to reduce load
            for _ in range(FRAME_SKIP):
                success, frame = cap.read()
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    success, frame = cap.read()
                    if not success:
                        continue

            if frame is None:
                continue

            frame_count += 1
            frames_since_detection += 1
            detected_objects = last_detected_objects

            # --- AI INFERENCE (only every N frames) ---
            if frame_count % INFERENCE_INTERVAL == 0:
                with model_lock:
                    results = model.predict(frame, conf=0.05, verbose=False)  # Lower threshold for better recall

                result = results[0]

                # Extract detected boxes - only update cache if we found something
                new_boxes = []
                new_detected_objects = []
                if result.boxes and len(result.boxes) > 0:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        class_name = result.names[cls_id]
                        new_detected_objects.append(class_name)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        new_boxes.append((class_name, x1, y1, x2, y2, conf))

                    # Update cache with new detections
                    last_boxes = new_boxes
                    last_detected_objects = new_detected_objects
                    detected_objects = new_detected_objects
                    frames_since_detection = 0  # Reset persistence counter

                # --- RULE CHECKING (only when we have detections) ---
                if new_detected_objects:
                    violations = check_violation(new_detected_objects)
                    camera_violations[camera_id] = violations

            # Clear boxes after persistence period expires (only if no new detections)
            if frames_since_detection > BOX_PERSISTENCE_FRAMES:
                last_boxes = []
                last_detected_objects = []
                camera_violations[camera_id] = []

            # --- DRAW CACHED BOXES ---
            for class_name, x1, y1, x2, y2, conf in last_boxes:
                # Color based on detection type
                if class_name in ["bare_hand", "hand", "human hand", "fingers", "palm"]:
                    color = (0, 0, 255)  # Red for potential violations (unprotected hands)
                elif class_name in ["gloved_hand", "glove", "work glove", "safety glove",
                                    "safety_glasses", "goggles", "eye_protection", "face_shield"]:
                    color = (0, 255, 0)  # Green for safety equipment
                elif class_name in ["industrial_machine", "machine", "tool", "power tool", "grinder", "saw"]:
                    color = (255, 165, 0)  # Orange for industrial context
                else:
                    color = (255, 255, 0)  # Yellow for other (face, person, worker)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- VIOLATION OVERLAY ---
            violations = camera_violations.get(camera_id, [])
            if violations:
                # Red border for violations
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 8)

                # Warning text
                cv2.putText(frame, f"VIOLATION: {len(violations)}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                # Show first violation code
                cv2.putText(frame, violations[0]['code'], (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Add camera label overlay
            cv2.putText(frame, cam["label"], (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Frame rate limiting
            elapsed = time.time() - start_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    finally:
        cap.release()
        print(f"Stream ended for {camera_id}")


# --- 7. API Endpoints ---

@app.get("/")
async def root():
    """API root - health check."""
    return {
        "service": "OSHA-Vision Backend",
        "status": "running",
        "cameras": len(cameras),
        "device": device
    }


@app.get("/cameras")
async def list_cameras():
    """List all available cameras."""
    return {
        "cameras": list(cameras.values()),
        "count": len(cameras)
    }


@app.get("/cameras/{camera_id}")
async def get_camera(camera_id: str):
    """Get details for a specific camera."""
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    return cameras[camera_id]


@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: str):
    """Stream MJPEG video with detection overlays for a specific camera."""
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    return StreamingResponse(
        generate_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/video_feed")
async def video_feed_default():
    """Stream first available camera (backwards compatibility)."""
    if not cameras:
        raise HTTPException(status_code=404, detail="No cameras available")

    first_camera = list(cameras.keys())[0]
    return StreamingResponse(
        generate_frames(first_camera),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/status")
async def get_all_status():
    """Get violation status for all cameras."""
    all_violations = []
    for cam_id, violations in camera_violations.items():
        for v in violations:
            all_violations.append({
                "camera_id": cam_id,
                "camera_label": cameras[cam_id]["label"],
                **v
            })

    return {
        "violations": all_violations,
        "total_count": len(all_violations),
        "cameras_with_violations": len([v for v in camera_violations.values() if v])
    }


@app.get("/status/{camera_id}")
async def get_camera_status(camera_id: str):
    """Get violation status for a specific camera."""
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    return {
        "camera_id": camera_id,
        "camera_label": cameras[camera_id]["label"],
        "violations": camera_violations.get(camera_id, [])
    }


@app.get("/search")
async def search_archive(q: str):
    """Search archived footage (if VectorStore is available)."""
    if not vector_store:
        return {"error": "Search engine not available", "results": []}

    results = vector_store.search(q)
    return {"results": results}


@app.get("/summarize/{camera_id}")
async def summarize_video(camera_id: str):
    """
    Generate an AI summary of the video content (DGX Spark Superpower).
    Uses VLM to analyze video frames and describe what's happening.
    """
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    summary = generate_video_summary(camera_id)

    return {
        "camera_id": camera_id,
        "camera_label": cameras[camera_id]["label"],
        "summary": summary,
        "vlm_enabled": vlm_model is not None,
        "device": device,
        "cached": camera_summaries.get(camera_id, {}).get("timestamp", 0) > 0
    }


@app.get("/summarize")
async def summarize_all_videos():
    """
    Generate AI summaries for all cameras (DGX Spark Superpower).
    """
    summaries = {}
    for camera_id in cameras:
        summary = generate_video_summary(camera_id)
        summaries[camera_id] = {
            "label": cameras[camera_id]["label"],
            "summary": summary
        }

    return {
        "summaries": summaries,
        "count": len(summaries),
        "vlm_enabled": vlm_model is not None,
        "device": device
    }


@app.post("/cameras/refresh")
async def refresh_cameras():
    """Rescan video directory for new files."""
    init_cameras()
    return {
        "message": "Cameras refreshed",
        "count": len(cameras)
    }


# --- 8. Main Entry Point ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
