# Hackathon Requirements: Factory Safety & Efficiency

## 2. Factory Safety & Efficiency
**Prizes:** $1,000 Cash

### The Dataset
**Egocentric-10K** (POV Video of industrial tasks - Pre-download required) https://huggingface.co/datasets/builddotai/Egocentric-10K

### The Problem
In high-risk environments, security cameras are usually passive "black boxes"—they only record accidents for liability after the fact. Workers often get hurt because they miss a small detail, skip a safety step, or lack immediate guidance. We need a system that acts as an active teammate, watching the same video stream but understanding it fast enough to intervene or capture knowledge.

### The Goal
Build a system that processes egocentric (POV) video in real-time to generate safety interventions or operational knowledge.

### Suggested Directions (Inspiration Only)
*   **Real-Time Intervention:** Trigger a TTS warning within 2 seconds of seeing a safety violation (e.g., "No Gloves").
*   **Automated Documentation:** Watch a task once and auto-generate a step-by-step training manual.
*   **Ergonomic Analysis:** Track hand movements to predict fatigue or repetitive strain injury.

### Examples
*   **The "Angel on Your Shoulder":** A low-latency agent that detects safety violations (e.g., "No Gloves") and triggers a verbal warning before the accident happens.
*   **The SOP Architect:** An agent that watches an expert perform a task once and automatically generates a step-by-step manual with screenshots and text, saving hours of documentation time.

## 3. Judging Criteria (100 Points Total)

**Philosophy:** We are judging **Systems Engineering**. A winning project isn't just a slide deck or a simple API wrapper; it is a functioning system that ingests raw data, processes it locally using the DGX Spark, and produces a valuable result.

### 1. Technical Execution & Completeness (30 Points)
*   **Completeness (15 pts):** Did they actually build a working, complex system? Does the system successfully complete the full data workflow without crashing?
*   **Technical Depth (15 pts):** Is there significant engineering "under the hood"? Did they build a complex pipeline (e.g., Simulation, RAG, Fine-Tuning, or Custom Logic) rather than just a simple static dashboard or basic API wrapper?

### 2. NVIDIA Ecosystem & Spark Utility (30 Points)
*   **The Stack (15 pts):** Did they use at least one major NVIDIA library/tool? (e.g., NIMs, RAPIDS, cuOpt, Modulus, NeMo Models). *Note: Merely calling GPT-4 via API gets 0 points here.*
*   **The "Spark Story" (15 pts):** Can they articulate **why** this runs better on a DGX Spark?
    *   *Example:* "We used the 128GB Unified Memory to hold the video buffer and the LLM context simultaneously" or "We ran inference locally to ensure privacy/latency."

### 3. Value & Impact (20 Points)
*   **Insight Quality (10 pts):** Is the insight non-obvious and valuable? (e.g., "Traffic jams happen at 5 PM" is obvious. "Rain causes specific stalls on this specific ramp" is valuable).
*   **Usability (10 pts):** Could a real Fire Chief, City Planner, or Factory Foreman actually use this tool to make a decision tomorrow?

### 4. The "Frontier" Factor (20 Points)
*   **Creativity (10 pts):** Did they combine data or models in a novel way? (e.g., Using vision models to "read" traffic maps).
*   **Performance (10 pts):** Did they optimize the system for speed or scale? (e.g., "We optimized the simulation to run at 50x real-time speed").

## 4. Important Dates
*   **Code Freeze:** Dec 14th, 11:00 AM
*   **Judging:** Dec 14th, 11:30 AM - 12:30 PM
*   **Hack Fair & Public Voting:** Dec 14th, 12:00 PM - 4:00 PM

