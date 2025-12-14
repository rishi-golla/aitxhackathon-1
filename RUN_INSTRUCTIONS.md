# OSHA-Vision: Running the Full Stack on DGX Spark

## Prerequisites

- NVIDIA DGX Spark (or system with NVIDIA GPU + Docker)
- NGC API Key (already configured in `.env`)
- Docker with NVIDIA runtime
- Node.js 18+
- Python 3.12+

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                        │
│                      http://localhost:3000                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐          ┌─────────────────────────┐   │
│  │   FastAPI Backend   │          │      VSS Engine         │   │
│  │   (YOLO + Streams)  │          │   (Cosmos-Reason1 VLM)  │   │
│  │   localhost:8000    │          │    localhost:8090       │   │
│  └─────────────────────┘          └─────────────────────────┘   │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                     Docker Services                              │
│  Redis (6379) │ VST-Storage (8081) │ Postgres (5432)            │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start (3 Terminals)

### Terminal 1: Start Docker Services + VSS Engine

```bash
cd /home/yvesjr/Developer/aitxhackathon-1

# Start Redis and VST-Storage (if not running)
docker compose up -d redis vst-storage

# Start VSS Engine (downloads Cosmos-Reason1 model on first run ~9.4GB)
source .env && docker compose up vss-engine
```

Wait for VSS to show:
```
***********************************************************
VIA Server loaded
Backend is running at http://0.0.0.0:8090
***********************************************************
```

### Terminal 2: Start FastAPI Backend

```bash
cd /home/yvesjr/Developer/aitxhackathon-1

# Activate virtual environment
source .venv/bin/activate

# Start the backend server
python server/main.py
```

Wait for:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
Found XX video files
```

### Terminal 3: Start Frontend

```bash
cd /home/yvesjr/Developer/aitxhackathon-1/frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

Wait for:
```
▲ Next.js 14.x.x
- Local: http://localhost:3000
```

## Access the Application

| Page | URL | Description |
|------|-----|-------------|
| Dashboard | http://localhost:3000/dashboard | Main overview with metrics |
| Footage Review | http://localhost:3000/dashboard/footage-review | Live camera feeds with YOLO detection |
| OSHA Violations | http://localhost:3000/dashboard/osha-violations | Detected violations with video evidence |
| Demo | http://localhost:3000/dashboard/demo | Upload video for AI analysis |

## How Each Page Works

### Footage Review (`/dashboard/footage-review`)
- Streams live video from `/data/*.mp4` files via FastAPI
- YOLO-World performs real-time object detection
- Click "Generate Summary" to analyze video with Cosmos-Reason1 VLM

### OSHA Violations (`/dashboard/osha-violations`)
- Shows pre-recorded violation videos from `/data/violation_*.mp4`
- Each violation links to OSHA CFR codes and penalties
- Videos stream via FastAPI backend

### Demo Page (`/dashboard/demo`)
- Upload any MP4 video
- Sends to VSS Engine (port 8090) for analysis
- Cosmos-Reason1 VLM generates safety description
- Keywords matched to OSHA regulations

## Port Reference

| Service | Port | Purpose |
|---------|------|---------|
| Frontend | 3000 | Next.js web UI |
| FastAPI Backend | 8000 | YOLO detection, video streaming |
| VSS Engine | 8090 | Cosmos-Reason1 VLM inference |
| Redis | 6379 | VSS caching |
| VST-Storage | 8081 | Video storage service |
| Postgres | 5432 | Database (optional) |

## Stopping Everything

```bash
# Stop VSS Engine
docker stop vss-engine

# Kill FastAPI backend
lsof -ti:8000 | xargs kill -9

# Kill Frontend
lsof -ti:3000 | xargs kill -9

# Stop all Docker services
docker compose down
```

## Troubleshooting

### VSS Engine won't start
```bash
# Check if port 8090 is available
lsof -i :8090

# Check Docker logs
docker logs vss-engine --tail 50
```

### Videos not showing
```bash
# Check backend is running
curl http://localhost:8000/cameras

# Check video files exist
ls -la data/*.mp4
```

### YOLO detection not working
```bash
# Backend needs these packages
source .venv/bin/activate
pip install ultralytics opencv-python
```

### VSS model download stuck
The Cosmos-Reason1 model is ~9.4GB. On first run, it downloads from NGC.
Check progress in Docker logs:
```bash
docker logs vss-engine -f
```

## Video Files

Place MP4 files in the `data/` directory:
- `factory_*.mp4` - Factory footage (appears in Footage Review)
- `violation_*.mp4` - Violation clips (appears in OSHA Violations)

Backend automatically detects and assigns camera IDs based on filename order.

## Environment Variables

Key variables in `.env`:
```
NGC_API_KEY=...      # Required for NIM container
HF_TOKEN=...         # For dataset downloads
```

VSS Engine config in `docker-compose.yml`:
```yaml
BACKEND_PORT=8090
VLM_MODEL_TO_USE=cosmos-reason1
VLLM_GPU_MEMORY_UTILIZATION=0.3
```
