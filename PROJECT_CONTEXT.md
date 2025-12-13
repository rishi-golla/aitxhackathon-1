# PROJECT_CONTEXT.md
# Project: OSHA-Vision (DGX Spark Hackathon)

## 1. Project Overview
**Goal:** Build "OSHA-Vision" – A desktop application that processes industrial video feeds in real-time, detects safety violations (e.g., "Bare Hands") using Zero-Shot AI, and generates audit-grade OSHA citations.

**Core Value Proposition:**
* **Active vs. Passive:** Moves from recording accidents to preventing them.
* **Privacy First:** Runs 100% locally on NVIDIA DGX Spark hardware (no cloud).
* **Zero-Shot:** No training required; detects custom hazards via text prompts immediately.

## 2. Technical Architecture
* **Hardware:** NVIDIA DGX Spark (Powered by GB10 Grace Blackwell Superchip).
* **Frontend:** Electron (React + Vite) - The "Command Center" dashboard.
* **Backend:** Python (FastAPI) - Runs locally on the DGX.
* **AI Engine:** 
    * **Vision:** YOLO-World (Zero-Shot Object Detection) via `ultralytics`.
    * **Optimization:** Leveraging NVIDIA TensorRT for accelerated inference on GB10.
    * **Logic:** Python-based Rule Engine (Hardcoded JSON lookup for hackathon speed).
* **Communication:** 
    * Video Stream: MJPEG Stream (`http://localhost:8000/video_feed`).
    * Data Stream: JSON Polling (`http://localhost:8000/status`).

## 3. NVIDIA Stack Alignment
* **Hardware:** Utilizing the **GB10 Grace Blackwell Superchip** for high-performance local inference.
* **Blueprint Reference:** Architecture inspired by the **NVIDIA VSS (Video Search and Summarization) Blueprint**, adapting the concept of "Video-to-Insight" for industrial safety.
* **Local Inference:** Fully local execution ensures data privacy and low latency, critical for safety applications, aligned with **NVIDIA NIM** deployment patterns.

## 4. Team Roles & Responsibilities

### 👤 Dev A: Frontend (Electron) & Product
**Focus:** The UI, User Experience, and Business Logic.
* **Boilerplate:** Initialize `electron-vite-boilerplate`.
* **Video Component:** Implement a simple `<img src="...">` tag to display the MJPEG stream from the Python backend.
* **Sidebar UI:** Build the "Violation Feed" that polls the backend status.
* **Business Logic:** Implement the "Fine Calculator" (Total Fines = $16,131 * Violation Count).
* **Pitch Assets:** Design the slide deck and "Cost of Inaction" narrative.

### 👤 Dev B: Backend (Python/DGX) & AI
**Focus:** The AI Pipeline, Video Processing, and OSHA Rules.
* **Server:** Create `main.py` using FastAPI to serve the MJPEG stream.
* **AI Integration:** Implement `YOLOWorld` to detect specific prompts: `["bare hand", "gloved hand", "safety glasses", "face", "industrial machine"]`.
* **Visuals:** Use `cv2.rectangle` to draw **RED** bounding boxes on violations directly into the video stream frame buffer.
* **Rule Engine:** Implement the logic to cross-reference detected objects with `osha_rules.json`.
* **API:** expose `/status` endpoint returning the current violation state.

## 5. Development Phases (36-Hour Sprint)

**Phase 1: The Handshake (Hours 0-4)**
* [ ] **Dev A:** Electron app launches and displays a blank window.
* [ ] **Dev B:** Python script streams a dummy MP4 file to `localhost:8000/video_feed`.
* [ ] **Joint:** Electron displays the stream.

**Phase 2: The Red Box (Hours 4-12)**
* [ ] **Dev B:** YOLO-World correctly detects "bare hand" in the Voxel51 dataset.
* [ ] **Dev B:** Bounding boxes are drawn on the stream.
* [ ] **Dev A:** Sidebar UI shell is built.

**Phase 3: The Lawyer (Hours 12-24)**
* [ ] **Dev B:** Integrate `osha_rules.json` logic (e.g., `If Bare Hand + Machine -> Trigger 1910.138`).
* [ ] **Dev A:** Sidebar updates with real OSHA codes when violations trigger.
* [ ] **Dev A:** "Fine Calculator" updates in real-time.

**Phase 4: Polish (Hours 24-36)**
* [ ] **Joint:** Latency optimization (frame skipping if needed).
* [ ] **Joint:** Dark Mode UI.

## 6. Key Data Structures

### `osha_rules.json` (The Source of Truth)
```json
[
  {
    "id": "hand_protection_general",
    "triggers": ["bare_hand", "exposed_skin"],
    "required_context": ["industrial_machine", "sharp_tool", "chemical_container", "spinning_blade"],
    "code": "29 CFR 1910.138(a)",
    "title": "General Requirements - Hand Protection",
    "legal_text": "Employers shall select and require employees to use appropriate hand protection when employees' hands are exposed to hazards.",
    "penalty_max": 16131
  },
  {
    "id": "machine_guarding",
    "triggers": ["unguarded_belt", "exposed_gears", "rotating_parts"],
    "required_context": [],
    "code": "29 CFR 1910.212(a)(1)",
    "title": "Machine Guarding - General",
    "legal_text": "One or more methods of machine guarding shall be provided to protect the operator and other employees in the machine area.",
    "penalty_max": 16131
  }
]