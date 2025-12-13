# Project Context & Roadmap

## Project Goal
To build a **Dual-Engine Safety System** for industrial environments that utilizes the NVIDIA DGX Spark to its full potential.
1.  **The Sentinel (Real-Time):** An active "teammate" that watches live video and intervenes immediately when safety violations occur (e.g., "No Gloves").
2.  **The Archivist (Batch/Forensic):** A semantic search engine that processes massive datasets (Egocentric-10K) to allow safety managers to query historical footage using natural language (e.g., "Show me all instances of workers operating machinery without safety glasses").

## NVIDIA Stack Alignment
*   **Compute:** NVIDIA DGX Spark (GB10 Grace Blackwell Superchip).
*   **Inference (Real-Time):** TensorRT optimized YOLO-World for zero-shot detection.
*   **Compute (Batch):** NVIDIA NIMs (VILA or LLaVA) for video captioning/embedding.
*   **Search:** RAPIDS (RAFT/CuVS) for GPU-accelerated vector search over the dataset.

## Current Status
*   **Date:** December 12, 2025
*   **Phase:** Entering Phase 3 (Dataset Ingestion)

---

## Roadmap

### Phase 1: The Handshake (Completed)
*   **Goal:** Establish the basic loop: Camera -> Code -> Screen.
*   **Tech:** Python, OpenCV.
*   **Status:** ✅ Done.

### Phase 2: The Red Box (Completed)
*   **Goal:** Detect specific objects and overlay bounding boxes.
*   **Tech:** YOLO-World (Zero-Shot), FastAPI.
*   **Status:** ✅ Done.
    *   Backend serves MJPEG stream.
    *   YOLO-World detects hands, gloves, and safety glasses.
    *   Logic triggers "Violation" alerts.

### Phase 3: The Archivist (Dataset Ingestion)
*   **Goal:** Utilize the DGX to process the `Egocentric-10K` dataset.
*   **Tasks:**
    1.  **Ingest Pipeline:** Create scripts to download and chunk the dataset.
    2.  **VLM Processing:** Use a Vision-Language Model (e.g., VILA via NIM) to generate dense descriptions/embeddings for video clips.
    3.  **Vector Store:** Index these embeddings using a local vector store (FAISS or RAPIDS RAFT).
    4.  **Search API:** Create an endpoint to query the dataset (Text-to-Video search).

### Phase 4: The Interface (Frontend)
*   **Goal:** A unified dashboard for the live feed and the historical search.
*   **Tech:** Electron/React.
*   **Tasks:**
    *   Display the `/video_feed` stream.
    *   Show real-time alerts.
    *   Add a search bar for the "Archivist" engine.

---

## Technical Architecture

### Engine 1: The Sentinel (Live)
*   **Input:** Webcam or RTSP Stream.
*   **Model:** `yolov8s-world.pt` (Object Detection).
*   **Latency:** < 100ms.
*   **Output:** Bounding boxes, immediate alerts.

### Engine 2: The Archivist (Batch)
*   **Input:** `Egocentric-10K` Dataset.
*   **Model:** Large VLM (VILA/LLaVA) + Embedding Model.
*   **Compute:** High-load batch processing on DGX.
*   **Output:** Searchable Video Index.
