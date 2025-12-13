# How It Works: OSHA-Vision

## The Philosophy: "Active Teammate"
OSHA-Vision is not just a passive recorder. It is designed as an active teammate that operates on two time horizons:
1.  **The Now (Milliseconds):** Preventing accidents before they happen.
2.  **The Past (History):** Learning from massive amounts of data to improve future safety.

## Architecture: The Dual-Engine System

### Engine 1: The Sentinel (Real-Time Edge)
*   **Goal:** Immediate Intervention.
*   **Hardware:** Runs locally on the DGX Spark (or Edge Device).
*   **Model:** `YOLO-World` (Zero-Shot Object Detection).
*   **Workflow:**
    1.  **Input:** Ingests 1080p/30fps video from the worker's POV (Egocentric-10K style).
    2.  **Inference:** Uses TensorRT-optimized YOLO-World to detect safety-critical objects (`bare_hand`, `industrial_machine`, `safety_glasses`) in <50ms.
    3.  **Logic:** A Python-based Rule Engine checks for dangerous combinations (e.g., `bare_hand` + `active_machine`).
    4.  **Output:** Triggers an immediate visual alert (Red Box) and logs the violation.

### Engine 2: The Archivist (Semantic Search Core)
*   **Goal:** Knowledge Retrieval & Pattern Discovery.
*   **Hardware:** **Exclusive to NVIDIA DGX Spark** (Requires high VRAM & Compute).
*   **Models:** NVIDIA VSS Blueprint (Cosmos-Reason1 / VILA) running locally.
*   **Workflow:**
    1.  **Ingest:** The system downloads raw footage from the `Egocentric-10K` dataset.
    2.  **Understand (The "Spark Story"):**
        *   We use the **NVIDIA VSS Event Reviewer** blueprint running locally on the DGX.
        *   The VLM (Cosmos-Reason1) watches every second of video.
        *   The VLM generates dense, descriptive captions: *"A worker is using a grinder. Sparks are flying. He is not wearing a face shield."*
        *   *Why DGX?* Processing 10,000 hours of video requires massive parallel compute that only a workstation like the DGX can handle efficiently.
    3.  **Embed:** These captions are converted into high-dimensional vectors.
    4.  **Index:** We use **RAPIDS RAFT** (or local vector store) to build a GPU-accelerated vector index.
    5.  **Search:** A safety manager asks: *"Show me near-misses involving forklifts."* The system retrieves the exact video clips instantly.

## Design Choices & Hackathon Alignment

| Feature | Hackathon Goal | Why We Did It |
| :--- | :--- | :--- |
| **Zero-Shot Detection** | **Factory Safety** | Traditional AI requires training on thousands of images. Zero-shot allows us to detect *new* hazards (e.g., "spilled chemical") simply by typing a text prompt, making the system adaptable to any factory. |
| **VLM Indexing** | **NVIDIA Ecosystem** | We leverage **NVIDIA VSS Blueprint** to turn "dumb" pixels into searchable text. This showcases the power of Generative AI for understanding complex industrial context. |
| **Local Inference** | **Privacy / Spark** | Factories are air-gapped. Sending video to the cloud is a security risk. The **DGX Spark** allows us to run both the Sentinel and the Archivist entirely on-premise. |
| **Egocentric-10K** | **The Dataset** | We specifically designed the system to handle the challenges of POV video (motion blur, occlusion) by validating against this massive real-world dataset. |

## The User Journey
1.  **The Worker:** Puts on the camera. The "Sentinel" watches their back, warning them if they forget gloves.
2.  **The Manager:** Uses the "Archivist" dashboard to search for trends. *"Why did we have 5 hand injuries last month?"* -> *Search: "Hand near blade"* -> *Result: "Workers are removing guards to clear jams."* -> *Action: Update training.*

This closes the loop between **Real-time Safety** and **Long-term Operational Improvement**.
