**Project Title: OSHA-Vision on DGX**

### ***Turning "Silicon" into a Digital Safety Officer***

Challenge Track: Factory Safety & Efficiency  
Github: \- https://github.com/lolout1

Dataset: Egocentric-10K (POV Factory Video)

### **💡 The Pitch**

**OSHA-Vision** is an active AI teammate that processes worker POV video in real-time to detect safety violations (like missing PPE) and instantly cross-references them with federal regulations to generate audit-grade citations. It moves beyond simple "detection" to legal "compliance."

### **🚨 The Problem**

* **Reactive vs. Proactive:** Current safety cameras are passive "black boxes" that only record accidents *after* they happen.  
* **The Knowledge Gap:** Workers often don't know *why* a specific action is unsafe.  
* **The Data Bottleneck:** Factory video data is massive, unlabeled, and sensitive. It cannot be sent to the cloud for processing, and training custom models for every new hazard takes months.

### **🛠️ The Solution (The Workflow)**

1. **See (The Eyes):** A Zero-Shot Vision Model (**YOLO-World**) scans the worker’s POV feed in real-time. It detects custom objects via text prompts (e.g., "bare hands," "spinning blade," "chemical container") without needing pre-trained labels.  
2. **Think (The Brain):** When a hazard is detected (e.g., "Hand" inside "Blade Zone"), the system uses **RAG (Retrieval-Augmented Generation)** to search a local vector database of **OSHA 1910 Standards**.  
3. **Act (The Output):** An **NVIDIA NIM (Llama-3)** synthesizes the visual data with the legal text to output a structured citation: *“Violation Detected: 29 CFR 1910.212 \- Point of Operation Guarding. Potential Penalty: $16,131.”*

### **🏗️ The Tech Stack (DGX Optimized)**

* **Infrastructure:** **NVIDIA DGX** (Local Inference). We prioritize data privacy by keeping all video processing on-premise, avoiding public API calls.  
* **Inference Engine:** **NVIDIA NIMs** (Llama-3-70B & Vision NIMs) for optimized, low-latency performance.  
* **Vision Model:** **YOLO-World** (Zero-Shot Object Detection). Chosen for its ability to detect *unseen* hazards instantly via text prompts, solving the "unlabeled dataset" problem.  
* **Orchestration:** **LlamaIndex** \+ **Milvus** (Vector DB) for grounding AI reasoning in official OSHA PDF manuals.  
* **Frontend:** **Streamlit**. A split-screen dashboard showing the live video with "Red Bounding Boxes" alongside a scrolling feed of regulatory citations.

### **🏆 Why It Wins (Alignment)**

* **AITX Theme ("Silicon to Systems to Software"):** We aren't just running a model; we are building a complete *software system* that turns raw DGX compute ("Silicon") into a usable safety utility.  
* **Real-World Utility:** By using "Zero-Shot" detection, our system works on *any* factory floor immediately—no training required. This addresses the challenge of the unlabeled Egocentric-10K dataset.  
* **Sponsor Goal (NVIDIA):** Demonstrates high-performance local computing using **NIMs**, proving we can run enterprise-grade AI securely on the edge.  
* **The "Wow" Factor:** The combination of **Visual Proof** (Bounding Box) \+ **Regulatory Authority** (OSHA Citation) creates an undeniable value proposition for judges.

Githubs:  
[github.com/armaanamatya](http://github.com/armaanamatya)

Here is the prioritized list of deliverables for your 2-person team. You can copy-paste this directly into your project management doc.

### **📋 OSHA-Vision Deliverables Checklist**

#### **👤 Dev A: Frontend (Electron) & Product**

*Focus: The "Face" of the demo. Making the backend's work visible and selling the business value.*

* **1\. The "Empty" Shell (Hour 0-2):**  
  * \[ \] Initialize electron-vite-boilerplate.  
  * \[ \] Verify it builds and launches on the target machine.  
* **2\. The Video Receiver (Hour 2-4):**  
  * \[ \] Implement the video player component.  
  * \[ \] **Crucial:** Ensure it simply points to the backend stream URL (\<img src="http://localhost:8000/video\_feed" /\>) rather than trying to decode frames manually.  
* **3\. The Alert Sidebar (Hour 4-12):**  
  * \[ \] Create a polling mechanism (React hook) that pings /status every 1 second.  
  * \[ \] Design the **"Violation Card"** component: Needs a red border, a placeholder for the OSHA code (e.g., "1910.138"), and a "Potential Fine" amount.  
* **4\. The "Fine Calculator" (Hour 12-24):**  
  * \[ \] Implement a simple counter state that increments a total dollar amount (e.g., \+$16,131) every time a new violation is detected. Judges love seeing a "Cost of Inaction" metric.  
* **5\. The Pitch Deck (Hour 24-36):**  
  * \[ \] **Slide 1:** The Hook (Factory injuries cost billions).  
  * \[ \] **Slide 2:** The Solution (OSHA-Vision: Active AI vs. Passive Video).  
  * \[ \] **Slide 3:** The "Secret Sauce" (Zero-Shot Detection on DGX Spark).  
  * \[ \] **Slide 4:** The Tech Stack (Electron \+ Python \+ NVIDIA NIMs).

#### **👤 Dev B: Backend (Python/DGX) & AI**

*Focus: The "Brain" of the demo. Making the detection work and serving the video.*

* **1\. The Streaming Server (Hour 0-2):**  
  * \[ \] Get main.py running with **FastAPI**.  
  * \[ \] Successfully stream a dummy video file (MJPEG format) to localhost:8000/video\_feed.  
* **2\. The "Eyes" (YOLO-World) (Hour 2-10):**  
  * \[ \] Integrate ultralytics YOLO-World.  
  * \[ \] Define the prompt classes: \['bare hand', 'gloved hand', 'safety glasses', 'face', 'machine'\].  
  * \[ \] **Visual Proof:** Ensure bounding boxes are drawn *directly* onto the video frames before streaming (uses cv2.rectangle).  
* **3\. The "Lawyer" (Logic Layer) (Hour 10-20):**  
  * \[ \] Create the **Violation Logic**: IF "bare hand" detected AND "machine" detected \-\> Violation \= True.  
  * \[ \] Build the **OSHA Lookup Dict**: Map the specific violation to a text string (e.g., "Citation: 29 CFR 1910.138(a) \- Hand Protection"). *Do not over-engineer a full PDF RAG if a lookup table works for the demo.*  
* **4\. The Status API (Hour 20-24):**  
  * \[ \] Build the /status endpoint that returns the *latest* violation text and boolean flag to Dev A.  
* **5\. Integration & Optimization (Hour 24-36):**  
  * \[ \] **Latency Check:** If the stream lags, implement "Frame Skipping" (process only every 3rd or 5th frame).  
  * \[ \] **Data Privacy Check:** Ensure no images are saved to disk, only processed in RAM (key selling point for DGX).

### **🤝 Joint Milestones**

* **Checkpoint 1 (Hour 4):** "The Handshake" — Electron app displays the video stream from Python (even if it's just a black screen).  
* **Checkpoint 2 (Hour 18):** "The Red Box" — System correctly detects a bare hand in the sample video and draws a box.  
* **Checkpoint 3 (Hour 36):** "The Full Loop" — A bare hand detection triggers a card to appear in the Electron sidebar.

---

You have correctly identified the "secret sauce" for this hackathon: **The lookup table.**

For a 48-hour demo, **do not** try to parse the entire OSHA database. It’s 50,000 pages of bureaucracy. Instead, you only need the **"Subpart I" (PPE)** and **"Subpart O" (Machinery)** standards. These cover 90% of what you will see in the Egocentric-10K video dataset.

Here is the **"OSHA Cheat Sheet"** for your team.

### **📜 The "Starter Pack" Violations (Hardcode These)**

Give this JSON-like structure to **Dev B**. These are the only 4 citations you need to make the demo work perfectly with the "Bare Hands" and "Spinning Machine" detection.

| Logic (If YOLO sees...) | The Citation (Standard Number) | The "Legal Speak" (To display in UI) | Max Penalty (2024) |
| :---- | :---- | :---- | :---- |
| Bare Hand \+ Machine | **29 CFR 1910.138(a)** | *"Employer failed to select and require appropriate hand protection when employees' hands are exposed to hazards."* | **$16,131** |
| No Safety Glasses | **29 CFR 1910.133(a)(1)** | *"Employer shall ensure each affected employee uses appropriate eye or face protection when exposed to eye or face hazards."* | **$16,131** |
| Spinning Part(Unguarded) | **29 CFR 1910.212(a)(1)** | *"One or more methods of machine guarding shall be provided to protect the operator and other employees in the machine area."* | **$16,131** |
| Point of Operation | **29 CFR 1910.212(a)(3)(ii)** | *"The point of operation of machines whose operation exposes an employee to injury, shall be guarded."* | **$16,131** |

---

### **📂 The Source Text (For RAG)**

If you want to build the RAG system (Phase 3 of your plan), do **not** download the whole OSHA website. Download only these specific PDF sections.

**1\. Subpart I \- Personal Protective Equipment (PPE)**

* **What it covers:** Gloves, Glasses, Helmets.  
* **Direct Link:** [OSHA 1910 Subpart I (eCFR)](https://www.ecfr.gov/current/title-29/subtitle-B/chapter-XVII/part-1910/subpart-I)  
* **Key Section:** 1910.138 (Hand Protection) – This is your money maker.

**2\. Subpart O \- Machinery and Machine Guarding**

* **What it covers:** Spinning blades, flywheels, nip points.  
* **Direct Link:** [OSHA 1910 Subpart O](https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.212)  
* **Key Section:** 1910.212 (General Requirements) – This is the catch-all for "that machine looks dangerous."

### **💡 Pro-Tip for the Demo**

When you present, don't say "We reference OSHA standards."

Say: "We cross-reference 29 CFR 1910 Subpart I and O, specifically targeting the top 10 most cited violations of 2024, including Machine Guarding ($1,541 violations last year)."1

This makes you sound like an industry expert, not just a hacker.

**Next Step:** Would you like the **JSON file** of these violations formatted so **Dev B** can drop it directly into the Python code?

