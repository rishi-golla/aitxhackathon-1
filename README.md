# OSHA Vision - Factory Safety Copilot 🏭

**Real-time PPE detection and safety compliance monitoring for industrial environments**

Built for the NVIDIA DGX Spark Frontier Hackathon | Track 2: Factory Safety & Efficiency

## Overview

OSHA Vision is a dual-engine safety monitoring system that acts as a **safety copilot, not a cop** - providing friendly, constructive feedback to help workers stay safe.

### Key Features

- **Real-time PPE Detection** - YOLO-World zero-shot detection for hardhats, vests, glasses, gloves
- **Zone-based Safety Rules** - Configure PPE requirements per zone with OSHA references
- **Three-Agent System** - Perception → Policy → Coach pipeline for intelligent analysis
- **Voice Alerts** - Text-to-speech feedback that helps, not punishes
- **OSHA-Grounded** - RAG system with 29 CFR 1910 standards for compliance

## Architecture

```
Video Source → Python Backend (FastAPI :8000)
                   ├── OpenCV capture
                   ├── YOLOv8-World inference
                   ├── Zone Manager
                   └── Agent Chain (Perception → Policy → Coach)
                            ↓
              Streamlit Dashboard (:8501)
                   └── Live monitoring + alerts
```

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support
- Docker (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/lolout1/osha-vision.git
cd osha-vision

# Install dependencies
pip install -r requirements.txt

# Download YOLO model (if not present)
python -c "from ultralytics import YOLO; YOLO('yolov8s-world.pt')"
```

### Running

**Option 1: Streamlit Dashboard**
```bash
python run_dashboard.py
# Open http://localhost:8501
```

**Option 2: Backend API Only**
```bash
python run_backend.py
# Video feed: http://localhost:8000/video_feed
# Status: http://localhost:8000/status
```

**Option 3: Docker**
```bash
docker-compose up --build
```

### Video Source Configuration

```bash
# Webcam (default)
VIDEO_SOURCE=0 python server/main.py

# Video file
VIDEO_SOURCE="path/to/video.mp4" python server/main.py

# RTSP stream
VIDEO_SOURCE="rtsp://camera/stream" python server/main.py
```

## Project Structure

```
osha-vision/
├── src/
│   ├── pipeline/          # Video capture, detection, orchestration
│   │   ├── video_capture.py
│   │   ├── detection.py
│   │   ├── vlm_analysis.py
│   │   └── orchestrator.py
│   ├── zones/             # Safety zone management
│   │   └── zone_manager.py
│   ├── agents/            # Three-agent system
│   │   ├── perception_agent.py
│   │   ├── policy_agent.py
│   │   ├── coach_agent.py
│   │   └── safety_chain.py
│   ├── rag/               # OSHA standards RAG
│   │   └── osha_rag.py
│   ├── tts/               # Text-to-speech alerts
│   │   └── alert_system.py
│   ├── utils/             # Config, database, logging
│   │   ├── config.py
│   │   ├── database.py
│   │   └── logging_config.py
│   └── dashboard/         # Streamlit UI
│       └── app.py
├── server/                # FastAPI backend
│   └── main.py
├── config/                # Configuration files
│   ├── app.yaml
│   ├── zones.yaml
│   └── osha_rules.json
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Configuration

### Zone Configuration (`config/zones.yaml`)

```yaml
zones:
  - zone_id: "welding_bay_a"
    name: "Welding Bay A"
    zone_type: "ppe_required"
    polygon: [[100, 100], [500, 100], [500, 400], [100, 400]]
    required_ppe:
      - "welding_mask"
      - "safety_gloves"
      - "safety_vest"
    osha_reference: "29 CFR 1910.252"
```

### Detection Classes

Modify `config/app.yaml` to add/remove detection classes:

```yaml
detection:
  classes:
    - person
    - hardhat
    - safety vest
    - safety glasses
    - gloves
    - bare hands
    - forklift
    - machinery
```

## Agent System

### Perception Agent
Analyzes scenes using VLM to understand context beyond raw detections.

### Policy Agent
Evaluates compliance against zone rules and OSHA standards with citations.

### Coach Agent
Generates constructive, friendly feedback - "Hey, grab your hardhat!" not "VIOLATION DETECTED".

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /video_feed` | MJPEG stream with detection overlays |
| `GET /status` | Current violations JSON |

## OSHA Standards Integrated

- 29 CFR 1910.132 - PPE General Requirements
- 29 CFR 1910.133 - Eye and Face Protection
- 29 CFR 1910.135 - Head Protection
- 29 CFR 1910.138 - Hand Protection
- 29 CFR 1910.212 - Machine Guarding
- 29 CFR 1910.252 - Welding Safety

## Tech Stack

- **Vision**: YOLO-World (Ultralytics), OpenCV
- **VLM**: Qwen2.5-VL-7B (optional, requires GPU)
- **Backend**: FastAPI, Uvicorn
- **Dashboard**: Streamlit
- **Database**: SQLite (aiosqlite)
- **RAG**: FAISS, Sentence Transformers
- **TTS**: gTTS / Edge TTS

## DGX Spark Optimization

- 4-bit quantization for VLM (16GB GPU memory)
- TensorRT export support for 2-3x inference speedup
- CUDA-accelerated video decode
- Unified memory utilization

## Demo Script (5 Minutes)

1. **The Problem** (1 min): Factory safety is reactive
2. **Technical Depth** (1 min): YOLO-World, zone config, agent reasoning
3. **NVIDIA Ecosystem** (1 min): GPU utilization, local processing
4. **Live Demo** (1 min): PPE violation → detection → friendly feedback
5. **The Value** (1 min): Copilot, not cop - helping workers stay safe

## License

MIT License

## Acknowledgments

- NVIDIA DGX Spark Hackathon
- Ultralytics YOLO-World
- Anthropic Claude for agent assistance
