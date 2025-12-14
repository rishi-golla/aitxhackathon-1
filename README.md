# AI Safety Sentinel 🛡️

A hybrid Edge/Local AI system for industrial safety monitoring, running entirely on NVIDIA DGX Spark.

## 🏗️ Architecture

This project uses a dual-engine architecture to maximize performance and capability:

1.  **The Sentinel (Real-Time):**
    *   **Model:** YOLO-World (Real-time Object Detection)
    *   **Hardware:** Runs locally on the DGX/Edge device.
    *   **Function:** Monitors live video feeds for immediate safety violations (e.g., missing PPE).

2.  **The Archivist (Forensic Search):**
    *   **Model:** NVIDIA VSS Blueprint (Cosmos-Reason1 VLM).
    *   **Hardware:** Runs locally on the DGX Spark via Docker.
    *   **Function:** Processes historical footage to generate semantic descriptions and enable natural language search.

## 📂 Project Structure

*   `server/` - The main FastAPI backend (The Sentinel).
*   `archivist/` - VLM integration and vector storage logic.
*   `backend/` - Electron-based desktop application window.
*   `frontend/` - HTML/JS User Interface.
*   `models/` - Local AI model weights.
*   `data/` - Video storage for ingestion.
*   `config/` - Safety rules and configuration.

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   Node.js & npm
*   Docker & NVIDIA Container Runtime
*   NVIDIA NGC API Key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repo-url>
    cd aitxhackathon-1
    ```

2.  **Install Python Dependencies:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Install Node Dependencies:**
    ```bash
    npm install
    ```

4.  **Configure Environment:**
    Create a `.env` file in the root directory:
    ```env
    NGC_API_KEY=nvapi-your-key-here
    ```

### Running the Application

1.  **Start the Local AI Engine (VSS):**
    ```bash
    docker compose up -d
    # Wait for the model to download (check logs: docker logs -f vss-engine)
    ```

2.  **Start the Backend (Sentinel & Archivist API):**
    ```bash
    .venv/bin/python server/main.py
    ```

3.  **Ingest Video Data:**
    ```bash
    .venv/bin/python -m archivist.ingest
    ```

4.  **Start the Frontend (Desktop App):**
    ```bash
    npm start
    ```