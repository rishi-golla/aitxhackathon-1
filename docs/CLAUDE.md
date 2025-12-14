# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OSHA Vision is a dual-engine safety monitoring system for industrial environments, built for the NVIDIA DGX Spark hackathon:

1. **The Sentinel (Real-Time):** Live video monitoring that detects safety violations using YOLO-World zero-shot detection
2. **The Archivist (Batch/Forensic):** Processes historical video datasets (Egocentric-10K) for semantic search and compliance analysis

## Tech Stack

- **Backend:** Python, FastAPI, OpenCV, Ultralytics YOLOv8-World
- **Frontend:** Next.js, React, TypeScript, Tailwind CSS
- **Desktop:** Electron wrapper
- **Target Hardware:** NVIDIA DGX Spark (GB10 Grace Blackwell)

## Commands

### Development
```bash
# Install all dependencies
npm install
pip install -r requirements.txt

# Run frontend + Electron (concurrent)
npm run dev

# Run Python backend separately (port 8000)
python server/main.py

# Run only Next.js frontend (port 3000)
npm run dev:next

# Run only Electron (expects Next.js running)
npm run dev:electron
```

### Building
```bash
npm run build:next    # Build Next.js for production
npm start             # Start Electron with production build
```

### Dataset
```bash
python scripts/download_sample.py    # Download Egocentric-10K sample
```

### Video Source Configuration
```bash
VIDEO_SOURCE=0 python main.py                    # Webcam (default)
VIDEO_SOURCE="path/to/video.mp4" python main.py  # Video file
VIDEO_SOURCE="rtsp://stream/url" python main.py  # RTSP stream
```

## Architecture

```
Video Source → Python Backend (FastAPI :8000)
                   ├── OpenCV capture
                   ├── YOLOv8-World inference
                   ├── OSHA rules engine
                   └── MJPEG stream + status API
                            ↓
              Electron → Next.js Frontend (:3000)
                         └── Displays stream + alerts
```

### API Endpoints
- `GET /video_feed` - MJPEG stream with detection overlays
- `GET /status` - Current violations JSON

### Key Files
| File | Purpose |
|------|---------|
| `main.py` | FastAPI backend with YOLO inference |
| `osha_rules.json` | OSHA violation rules database |
| `backend/main.js` | Electron main process |
| `frontend/app/page.tsx` | Next.js home page |

## OSHA Rules Engine

Rules in `osha_rules.json` have this structure:
- `triggers`: Objects that trigger a violation check
- `required_context`: Additional objects needed for violation
- Detection classes: `bare_hand`, `gloved_hand`, `safety_glasses`, `face`, `industrial_machine`

To modify detection classes, update `model.set_classes([...])` in `main.py`.

## Project Phases

- Phase 1: Camera loop (done)
- Phase 2: YOLO detection + violations (done)
- Phase 3: Dataset ingestion + VLM processing (in progress)
- Phase 4: Unified dashboard with search
