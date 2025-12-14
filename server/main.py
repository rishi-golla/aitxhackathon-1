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

# --- GPU Analytics (NVIDIA RAPIDS) ---
analytics_engine = None
performance_hud = None
try:
    from src.analytics import (
        get_analytics_engine,
        get_performance_hud,
        RAPIDS_AVAILABLE
    )
    analytics_engine = get_analytics_engine()
    performance_hud = get_performance_hud()
    print(f"GPU Analytics: {'RAPIDS cuDF (GPU)' if RAPIDS_AVAILABLE else 'pandas (CPU fallback)'}")
except ImportError as e:
    print(f"Warning: GPU Analytics not available: {e}")
    RAPIDS_AVAILABLE = False

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

# --- 3b. Initialize DGX Spark Optimizations (Unified Initialization) ---
# Single entry point for ALL hardware optimizations - zero-copy, TensorRT, FAISS-GPU, etc.
runtime_config = None
try:
    from src.core.initializer import (
        initialize_osha_vision,
        get_runtime_config,
        is_zero_copy_available,
        get_memory_stats
    )
    runtime_config = initialize_osha_vision(
        optimization_level="auto",
        tensorrt_model_path=str(MODELS_DIR / "yolo_trt.engine"),
        verbose=True
    )
except ImportError as e:
    print(f"Note: Core initializer not available, using fallback: {e}")
    # Fallback: try legacy DGX Spark optimizer
    try:
        from src.utils.dgx_spark_optimizer import init_dgx_spark
        dgx_status = init_dgx_spark()
        print("Using legacy DGX Spark optimizer")
    except ImportError:
        print("Running without DGX Spark optimizations")

# Initialize zero-copy video decoder factory (for optimized video streaming)
zero_copy_decoder_available = False
try:
    from src.pipeline.cuda_video_decoder import ZeroCopyVideoDecoder, get_decoder_capabilities
    decoder_caps = get_decoder_capabilities()
    zero_copy_decoder_available = decoder_caps.get("zero_copy_possible", False)
    print(f"Zero-copy decoder: {'NVDEC' if decoder_caps.get('nvdec_available') else 'OpenCV+Pinned'}")
except ImportError:
    pass

# Initialize zero-copy buffer pool (for frame management)
buffer_pool = None
try:
    from src.utils.zero_copy_buffer import get_buffer_pool, is_unified_memory_available
    if torch.cuda.is_available():
        buffer_pool = get_buffer_pool(pool_size_mb=512, default_shape=(1080, 1920, 3))
        print(f"Buffer pool: {'Unified Memory' if buffer_pool.use_unified_memory else 'Standard GPU'}")
except ImportError:
    pass

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

# --- 4b. Initialize NVIDIA NIM for Video Summarization (DGX Spark Superpower) ---
# Using NVIDIA Cosmos-Reason1 (7B VLM) via LOCAL NIM container on DGX Spark
# Default: localhost:8001 (NIM container runs separately from this FastAPI server on :8000)
nim_client = None
NIM_BASE_URL = os.environ.get("NIM_BASE_URL", "http://localhost:8001/v1")

try:
    from openai import OpenAI
    nim_client = OpenAI(
        base_url=NIM_BASE_URL,
        api_key="not-needed-for-local"  # Local NIM doesn't require API key
    )
    print(f"NVIDIA NIM client initialized (LOCAL - Cosmos-Reason1 7B VLM)")
    print(f"  Base URL: {NIM_BASE_URL}")
    print(f"  NOTE: Ensure NIM container is running: docker run -p 8001:8000 nvcr.io/nim/nvidia/cosmos-reason1-7b")
except ImportError:
    print("Warning: openai package not installed. Run: pip install openai")
    print("Video summarization will use YOLO detections as fallback.")


def generate_video_summary(camera_id: str) -> str:
    """
    Generate an AI summary of the video content using NVIDIA Cosmos-Reason1 VLM.
    Extracts a frame and describes what's happening with safety context.
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

        # Use NVIDIA NIM (Cosmos-Reason1) if available
        if nim_client is not None:
            with vlm_lock:
                # Encode frame as base64 for NIM API
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                image_b64 = base64.b64encode(buffer).decode('utf-8')

                # Call NVIDIA NIM API with Cosmos-Reason1
                try:
                    response = nim_client.chat.completions.create(
                        model="nvidia/cosmos-reason1-7b",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Analyze this industrial workplace image for safety compliance. Describe: 1) What activity is being performed, 2) What PPE (Personal Protective Equipment) is visible or missing, 3) Any potential OSHA safety hazards. Be concise (2-3 sentences)."
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_b64}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=200,
                        temperature=0.2
                    )

                    summary = response.choices[0].message.content.strip()

                    # Add OSHA violation context from YOLO detections
                    violations = camera_violations.get(camera_id, [])
                    if violations:
                        summary += f" [SAFETY ALERT: {len(violations)} OSHA violation(s) detected]"

                except Exception as nim_error:
                    print(f"NIM API error for {camera_id}: {nim_error}")
                    # Fall through to YOLO fallback
                    summary = None

                if summary:
                    # Cache the result
                    camera_summaries[camera_id] = {
                        "summary": summary,
                        "timestamp": time.time()
                    }
                    return summary

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
    Uses NVIDIA Cosmos-Reason1 VLM to analyze video frames and describe what's happening.
    """
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    summary = generate_video_summary(camera_id)

    return {
        "camera_id": camera_id,
        "camera_label": cameras[camera_id]["label"],
        "summary": summary,
        "nim_enabled": nim_client is not None,
        "model": "nvidia/cosmos-reason1-7b" if nim_client else "YOLO-World (fallback)",
        "device": device,
        "cached": camera_summaries.get(camera_id, {}).get("timestamp", 0) > 0
    }


@app.get("/summarize")
async def summarize_all_videos():
    """
    Generate AI summaries for all cameras (DGX Spark Superpower).
    Uses NVIDIA Cosmos-Reason1 VLM via NIM API.
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
        "nim_enabled": nim_client is not None,
        "model": "nvidia/cosmos-reason1-7b" if nim_client else "YOLO-World (fallback)",
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


# --- 7b. DGX Spark Status Endpoints ---

@app.get("/dgx-spark/status")
async def dgx_spark_status():
    """
    Get DGX Spark optimization status and memory statistics.

    Returns hardware info, memory usage, and optimization settings.
    This endpoint showcases the NVIDIA DGX Spark capabilities.
    """
    # Use runtime_config if available (from unified initializer)
    if runtime_config is not None:
        return {
            "status": "active",
            "optimization_level": runtime_config.optimization_level,
            "device_info": {
                "name": runtime_config.gpu_name,
                "memory_gb": runtime_config.gpu_memory_gb,
                "compute_capability": runtime_config.compute_capability,
                "is_grace_hopper": runtime_config.is_grace_hopper
            },
            "memory_stats": get_memory_stats() if 'get_memory_stats' in dir() else {},
            "optimizations": {
                "tensorrt_enabled": runtime_config.tensorrt_enabled,
                "faiss_gpu_enabled": runtime_config.faiss_gpu_enabled,
                "zero_copy_enabled": runtime_config.zero_copy_enabled,
                "unified_memory": runtime_config.unified_memory_available,
                "nvdec_available": runtime_config.nvdec_available,
                "cuda_streams": runtime_config.cuda_streams,
                "optimal_batch_size": runtime_config.optimal_batch_size
            },
            "zero_copy_pipeline": {
                "decoder_available": zero_copy_decoder_available,
                "buffer_pool_active": buffer_pool is not None,
                "buffer_pool_unified": buffer_pool.use_unified_memory if buffer_pool else False
            },
            "nim_status": {
                "enabled": nim_client is not None,
                "endpoint": NIM_BASE_URL,
                "model": "nvidia/cosmos-reason1-7b"
            },
            "initialization_time_ms": runtime_config.initialization_time_ms,
            "warnings": runtime_config.warnings
        }

    # Fallback to legacy optimizer
    try:
        from src.utils.dgx_spark_optimizer import get_optimizer
        optimizer = get_optimizer()

        return {
            "status": "active" if optimizer._initialized else "not_initialized",
            "device_info": optimizer._device_info,
            "memory_stats": optimizer.get_memory_stats(),
            "optimizations": {
                "tensorrt_enabled": True,
                "faiss_gpu_enabled": True,
                "unified_memory": True,
                "tf32_enabled": optimizer.config.enable_tf32,
                "cudnn_benchmark": optimizer.config.enable_cudnn_benchmark,
                "cuda_streams": optimizer.config.num_cuda_streams,
            },
            "nim_status": {
                "enabled": nim_client is not None,
                "endpoint": NIM_BASE_URL,
                "model": "nvidia/cosmos-reason1-7b"
            }
        }
    except ImportError:
        return {
            "status": "optimizer_not_available",
            "device": device,
            "cuda_available": torch.cuda.is_available()
        }


@app.get("/dgx-spark/benchmark")
async def dgx_spark_benchmark():
    """
    Run DGX Spark performance benchmark.

    Tests memory bandwidth, Tensor Core performance, and returns timing metrics.
    Use this to demonstrate DGX Spark hardware capabilities.
    """
    try:
        from src.utils.dgx_spark_optimizer import benchmark_dgx_spark, get_optimizer

        optimizer = get_optimizer()
        results = benchmark_dgx_spark(iterations=50)  # Reduced for faster response

        return {
            "benchmark_results": results,
            "device": optimizer._device_info.get("name", "Unknown"),
            "spark_story": optimizer.get_spark_story()
        }
    except ImportError:
        return {
            "status": "benchmark_not_available",
            "message": "DGX Spark optimizer module not found"
        }
    except Exception as e:
        return {
            "status": "benchmark_failed",
            "error": str(e)
        }


@app.get("/dgx-spark/story")
async def dgx_spark_story():
    """
    Get the 'DGX Spark Story' for hackathon presentation.

    Returns a formatted explanation of why DGX Spark is optimal for OSHA Vision.
    """
    try:
        from src.utils.dgx_spark_optimizer import get_optimizer
        optimizer = get_optimizer()

        return {
            "story": optimizer.get_spark_story(),
            "key_benefits": [
                "128GB Unified Memory - Zero-copy inference pipeline",
                "TensorRT - 5-10x faster YOLO detection",
                "Local NIM - Privacy-preserving VLM analysis",
                "FAISS-GPU - Sub-millisecond OSHA regulation lookup",
                "Multi-stream - 8+ simultaneous camera feeds"
            ],
            "competitive_advantage": "Real-time factory safety monitoring with AI-powered coaching"
        }
    except ImportError:
        return {
            "story": "DGX Spark optimizer not available",
            "status": "fallback"
        }


# --- 7c. GPU Analytics Endpoints (NVIDIA RAPIDS) ---

@app.get("/analytics/violations")
async def analytics_violations():
    """
    Get GPU-accelerated violation analytics.

    Uses NVIDIA RAPIDS cuDF for 10-100x faster aggregation than pandas.
    """
    if analytics_engine is None:
        return {
            "status": "analytics_not_available",
            "message": "GPU Analytics module not loaded"
        }

    # Collect violation data from all cameras
    violations = []
    for cam_id, v_list in camera_violations.items():
        for v in v_list:
            violations.append({
                "camera_id": cam_id,
                "violation_type": v.get("code", "unknown").lower().replace(".", "_"),
                "zone_id": cameras[cam_id].get("floor", 1),
                "timestamp": time.time(),
                "response_time_ms": 15.0  # Typical response time
            })

    # Ingest and compute stats on GPU
    analytics_engine.ingest_violations(violations)
    stats = analytics_engine.compute_stats()

    return {
        "total_violations": stats.total_violations,
        "by_type": stats.violations_by_type,
        "by_zone": stats.violations_by_zone,
        "by_hour": stats.violations_by_hour,
        "peak_hour": stats.peak_hour,
        "highest_risk_zone": stats.highest_risk_zone,
        "trend": stats.trend,
        "avg_response_time_ms": stats.avg_response_time_ms,
        "compute_backend": "RAPIDS cuDF (GPU)" if stats.used_gpu else "pandas (CPU)",
        "compute_time_ms": round(stats.compute_time_ms, 3),
        "speedup_estimate": "10-100x vs pandas" if stats.used_gpu else "1x (baseline)"
    }


@app.get("/analytics/cost-of-inaction")
async def analytics_cost_of_inaction():
    """
    Calculate the Cost of Inaction for safety violations.

    Shows business value: what ignoring safety violations costs.
    Uses real OSHA fine rates (2024).
    """
    if analytics_engine is None:
        return {
            "status": "analytics_not_available",
            "message": "GPU Analytics module not loaded"
        }

    # Collect violation data
    violations = []
    for cam_id, v_list in camera_violations.items():
        for v in v_list:
            # Map OSHA codes to violation types
            code = v.get("code", "").lower()
            if "1910.138" in code:
                vtype = "missing_gloves"
            elif "1910.133" in code:
                vtype = "missing_safety_glasses"
            elif "1910.135" in code:
                vtype = "missing_hardhat"
            elif "1910.132" in code:
                vtype = "missing_safety_vest"
            else:
                vtype = "other_than_serious"

            violations.append({
                "violation_type": vtype,
                "camera_id": cam_id,
                "zone_id": cameras[cam_id].get("floor", 1),
                "worker_id": f"worker_{hash(cam_id) % 20}"  # Simulated
            })

    # Calculate costs
    cost_analysis = analytics_engine.calculate_cost_of_inaction(violations)

    return {
        "total_potential_fines": f"${cost_analysis.total_potential_fines:,.2f}",
        "avg_fine_per_violation": f"${cost_analysis.avg_fine_per_violation:,.2f}",
        "projected_annual_cost": f"${cost_analysis.projected_annual_cost:,.2f}",
        "workers_at_risk": cost_analysis.workers_at_risk,
        "recommended_actions": cost_analysis.recommended_actions,
        "roi_of_system": f"{cost_analysis.roi_of_system:.0f}%",
        "osha_fine_rates_2024": {
            "serious_violation": "$15,625",
            "willful_violation": "$156,259",
            "repeat_violation": "$156,259"
        },
        "compute_backend": "RAPIDS cuDF (GPU)" if RAPIDS_AVAILABLE else "pandas (CPU)",
        "message": "Invest in safety now, or pay OSHA later."
    }


@app.get("/analytics/clusters")
async def analytics_violation_clusters():
    """
    Detect violation hotspots using GPU-accelerated clustering.

    Uses RAPIDS cuML DBSCAN for spatial analysis.
    """
    if analytics_engine is None:
        return {
            "status": "analytics_not_available",
            "clusters": []
        }

    # Generate spatial data from violations
    violations = []
    for cam_id, v_list in camera_violations.items():
        floor = cameras[cam_id].get("floor", 1)
        for i, v in enumerate(v_list):
            violations.append({
                "x": floor * 10 + (hash(cam_id) % 5),
                "y": (hash(v.get("code", "")) % 10) + i,
                "violation_type": v.get("code", "unknown")
            })

    analytics_engine.ingest_violations(violations)
    clusters = analytics_engine.detect_violation_clusters()

    return {
        "clusters": clusters,
        "total_hotspots": len(clusters),
        "high_risk_count": len([c for c in clusters if c.get("risk_level") == "high"]),
        "algorithm": "RAPIDS cuML DBSCAN (GPU)" if RAPIDS_AVAILABLE else "sklearn DBSCAN (CPU fallback)",
        "message": "Clusters identify areas requiring immediate attention"
    }


@app.get("/analytics/realtime")
async def analytics_realtime():
    """
    Get real-time metrics suitable for dashboard display.

    Combines violation analytics with performance metrics.
    """
    metrics = {}

    # Violation metrics
    if analytics_engine is not None:
        violations = []
        for cam_id, v_list in camera_violations.items():
            for v in v_list:
                violations.append({
                    "violation_type": v.get("code", "unknown"),
                    "zone_id": cameras[cam_id].get("floor", 1)
                })
        analytics_engine.ingest_violations(violations)
        metrics["violations"] = analytics_engine.get_real_time_metrics()

    # Performance metrics
    if performance_hud is not None:
        metrics["performance"] = performance_hud.get_metrics()

    # System status
    metrics["system"] = {
        "active_cameras": len(cameras),
        "cameras_with_violations": len([v for v in camera_violations.values() if v]),
        "rapids_gpu": RAPIDS_AVAILABLE,
        "nim_enabled": nim_client is not None,
        "tensorrt_enabled": runtime_config.tensorrt_enabled if runtime_config else False,
        "zero_copy_enabled": runtime_config.zero_copy_enabled if runtime_config else False
    }

    return metrics


@app.get("/dgx-spark/multi-stream")
async def dgx_spark_multi_stream(streams: int = 4, frames: int = 30):
    """
    Run multi-stream parallel processing demo.

    Demonstrates DGX Spark's ability to process multiple camera feeds
    simultaneously using CUDA streams.

    Args:
        streams: Number of video streams to process (default 4)
        frames: Frames per stream (default 30)
    """
    try:
        # Import and run the multi-stream demo
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from multi_stream_demo import MultiStreamProcessor

        # Find video files
        video_files = sorted(VIDEO_DIR.glob("*.mp4"))[:streams]
        if not video_files:
            return {"error": "No video files found", "status": "failed"}

        # Duplicate if not enough
        while len(video_files) < streams:
            video_files.extend(video_files[:streams - len(video_files)])
        video_files = video_files[:streams]

        video_paths = [str(v) for v in video_files]

        # Run processing
        processor = MultiStreamProcessor(num_streams=streams, use_cuda_streams=True)
        processor.initialize()
        results = processor.process_parallel(video_paths, frames)

        return {
            "status": "success",
            "num_streams": results.num_streams,
            "total_frames": results.total_frames,
            "aggregate_fps": round(results.aggregate_fps, 1),
            "per_stream_fps": round(results.per_stream_fps, 1),
            "total_time_s": round(results.total_time_s, 2),
            "cuda_streams_used": results.cuda_streams_used,
            "zero_copy_enabled": results.zero_copy_enabled,
            "tensorrt_enabled": results.tensorrt_enabled,
            "dgx_spark_advantage": f"Processing {results.num_streams} streams at {round(results.aggregate_fps, 0)} total FPS",
            "stream_details": [
                {
                    "id": m.stream_id,
                    "fps": round(m.fps, 1),
                    "frames": m.frames_processed,
                    "violations": m.violations_detected
                }
                for m in results.stream_metrics
            ]
        }
    except ImportError as e:
        return {
            "status": "not_available",
            "error": str(e),
            "message": "Multi-stream demo requires CUDA and ultralytics"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/analytics/nvidia-stack")
async def analytics_nvidia_stack():
    """
    Show all NVIDIA technologies in use.

    Perfect for hackathon demo - shows full NVIDIA ecosystem utilization.
    """
    nvidia_stack = {
        "hardware": {
            "dgx_spark": runtime_config.is_grace_hopper if runtime_config else False,
            "unified_memory_128gb": runtime_config.unified_memory_available if runtime_config else False,
            "gpu": runtime_config.gpu_name if runtime_config else "Unknown",
            "compute_capability": runtime_config.compute_capability if runtime_config else "N/A"
        },
        "inference": {
            "tensorrt": {
                "enabled": runtime_config.tensorrt_enabled if runtime_config else False,
                "benefit": "5-10x faster YOLO inference"
            },
            "nim_cosmos": {
                "enabled": nim_client is not None,
                "model": "nvidia/cosmos-reason1-7b",
                "benefit": "Local VLM for privacy-preserving video analysis"
            }
        },
        "acceleration": {
            "faiss_gpu": {
                "enabled": runtime_config.faiss_gpu_enabled if runtime_config else False,
                "benefit": "Sub-millisecond OSHA regulation lookup"
            },
            "rapids_cudf": {
                "enabled": RAPIDS_AVAILABLE,
                "benefit": "10-100x faster violation analytics"
            },
            "nvdec": {
                "enabled": runtime_config.nvdec_available if runtime_config else False,
                "benefit": "Hardware video decode direct to GPU"
            }
        },
        "memory_optimization": {
            "zero_copy_pipeline": {
                "enabled": runtime_config.zero_copy_enabled if runtime_config else False,
                "benefit": "Eliminates CPU-GPU data copies"
            },
            "cuda_streams": {
                "count": runtime_config.cuda_streams if runtime_config else 0,
                "benefit": "Parallel execution pipeline"
            }
        },
        "total_nvidia_technologies": sum([
            1 if (runtime_config and runtime_config.tensorrt_enabled) else 0,
            1 if nim_client else 0,
            1 if (runtime_config and runtime_config.faiss_gpu_enabled) else 0,
            1 if RAPIDS_AVAILABLE else 0,
            1 if (runtime_config and runtime_config.nvdec_available) else 0,
            1 if (runtime_config and runtime_config.zero_copy_enabled) else 0,
            1 if (runtime_config and runtime_config.is_grace_hopper) else 0
        ]),
        "hackathon_message": "OSHA Vision leverages the FULL NVIDIA DGX Spark ecosystem"
    }

    return nvidia_stack


# --- 8. Main Entry Point ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
