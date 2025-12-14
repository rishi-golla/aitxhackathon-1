# OSHA-Vision: 5-Minute Winning Demo Script

**Total Time: 5 minutes**
**Presenters: 2 (Dev A = Frontend/Product, Dev B = Backend/AI)**

---

## OPENING HOOK (0:00 - 0:30) — Dev A

> **[Screen: Dashboard with violation counter visible]**

"Every year, 2.8 million workers are injured on the job. OSHA issues $200 million in penalties. But here's the problem—those cameras recording everything? They're passive. They only help *after* someone gets hurt.

**What if the camera could intervene *before* the accident?**

We built OSHA-Vision: an AI teammate that watches factory footage, detects violations in real-time, and generates audit-grade citations automatically—all running locally on DGX Spark."

---

## PART 1: THE ARCHITECTURE — "How It Works" Page (0:30 - 1:30) — Dev B

> **[Navigate to: /dashboard/demo — "How it Works" page]**
> **[Scroll slowly through each section as you explain]**

"Let me show you how the system works. Click on **How it Works** in the sidebar.

### The Dual-Engine Architecture

We built two AI engines that work together:

**Engine 1: The Sentinel** — Real-time edge inference
- YOLO-World runs at 15 FPS on every camera feed
- Zero-shot detection means we didn't train on a single image—we just describe what to look for: *'bare hands, spinning blades, missing safety glasses'*
- Detects violations in under 50 milliseconds

**Engine 2: The Archivist** — Deep video understanding
- Uses NVIDIA's Cosmos-Reason1 VLM to *watch* video and describe what's happening
- Generates dense captions: *'A worker is grinding metal. Sparks are flying. No face shield visible.'*
- These descriptions are matched against OSHA 1910 regulations using RAG

> **[Point to the 'Spark Story' section]**

### The 'Spark Story' — Why DGX Spark?

This only works on hardware like the DGX Spark. Here's why:

1. **128GB Unified Memory** — We hold the video buffer, VLM weights, and vector embeddings *simultaneously* in GPU memory. No swapping. No bottlenecks.

2. **Zero Cloud Calls** — Factory video contains faces, worker behavior, proprietary processes. We can't send this to OpenAI. Everything runs 100% local.

3. **Sub-2-second inference** — Real-time intervention requires instant response. DGX delivers."

---

## PART 2: LIVE DETECTION DEMO — "Footage Review" Page (1:30 - 3:00) — Dev A

> **[Navigate to: /dashboard/live — "Footage Review" page]**

"Now let's see it in action. This is our **Footage Review** dashboard.

> **[Point to the grid of camera feeds]**

We're processing footage from the Egocentric-10K dataset—10,000 hours of real factory worker POV video. Each tile is a separate camera feed being analyzed by YOLO-World in real-time.

> **[Point to a feed showing detection boxes]**

See these bounding boxes?
- **Green** = Safety equipment detected (gloves, glasses)
- **Red** = Violation detected (bare hands near machinery)

> **[Point to the 'AI Backend Connected' indicator]**

That green dot means our Python backend is streaming frames with AI inference overlays at 15 FPS.

> **[Wait for a VIOLATION to appear, or point to one]**

**There!** See that red border? The system just detected a bare hand near industrial equipment. Watch the violation counter...

> **[Point to violation sidebar or counter]**

That's **29 CFR 1910.138(a)** — Hand Protection. Penalty: $16,131.

This didn't require months of training. We just told YOLO-World to look for 'bare_hand' and 'industrial_machine' and wrote a simple rule: if both present, trigger violation."

---

## PART 3: THE VLM DEEP ANALYSIS (3:00 - 4:00) — Dev B

> **[Navigate back to: /dashboard/demo]**
> **[Upload a sample video or show results]**

"For deeper analysis, we use NVIDIA's VSS Engine with Cosmos-Reason1.

> **[If video already uploaded, point to results. Otherwise, click Analyze]**

When a safety manager uploads yesterday's DVR export, the VLM *watches* the entire video and describes what it sees:

> **[Read the VLM caption aloud]**

*'A worker is operating a grinding machine. Sparks are visible. The worker is not wearing safety glasses or a face shield.'*

That natural language description gets matched against our OSHA regulation database:

> **[Point to the generated citations]**

- **Eye Protection** — 29 CFR 1910.133(a)(1) — $16,131
- **Machine Guarding** — 29 CFR 1910.212(a)(1) — $16,131

We're not just detecting objects. We're generating **audit-grade legal citations** that a compliance officer can use tomorrow."

---

## PART 4: THE DIGITAL TWIN (4:00 - 4:30) — Dev A

> **[Navigate to: /dashboard — Main Dashboard]**

"Finally, our factory digital twin.

> **[Click on a camera node in the 3D view]**

Each glowing dot represents a camera in the facility. Click one, and you get the live feed with AI overlays.

> **[Show the camera popup with streaming video]**

This is the full loop: 3D spatial awareness of your facility, real-time AI monitoring, and instant drill-down to any camera."

---

## CLOSING — THE VALUE PROPOSITION (4:30 - 5:00) — Dev A

> **[Show the Cost Counter on the demo page]**

"See this number? That's the **Cost of Inaction** — estimated penalties accumulating from unaddressed violations.

Every 2 seconds, that number climbs. With OSHA-Vision, you catch violations *before* the inspector arrives—or before someone gets hurt.

**To summarize:**

| Judging Criteria | How We Score |
|------------------|--------------|
| **Technical Execution (30 pts)** | Complete pipeline: YOLO + VLM + RAG + OSHA rules |
| **NVIDIA Ecosystem (30 pts)** | VSS Engine, Cosmos-Reason1 NIM, 100% local DGX inference |
| **Value & Impact (20 pts)** | A safety manager can use this *today* |
| **Frontier Factor (20 pts)** | First system combining Vision AI + Legal RAG for automated compliance |

We're not building a demo. We're building the future of workplace safety.

**Thank you.**"

---

## BACKUP: Q&A Talking Points

**Q: How is this different from existing safety cameras?**
> "Traditional cameras are passive recorders. They help *after* an accident for investigation. OSHA-Vision is an active teammate that prevents accidents by detecting violations in real-time and alerting workers *before* they get hurt."

**Q: Why YOLO-World instead of a trained model?**
> "The Egocentric-10K dataset has no labels. Training a custom model would take months. Zero-shot detection lets us describe hazards in plain English and start detecting immediately. We can add new hazard types with a text prompt, not a training run."

**Q: Why does this need DGX Spark?**
> "Three reasons: (1) 128GB unified memory to hold video + VLM + embeddings simultaneously, (2) data privacy—factory video can't leave the premises, (3) real-time latency—cloud inference adds 500ms+ round-trip, too slow for intervention."

**Q: What NVIDIA tools did you use?**
> "VSS Engine for video decode and summarization, Cosmos-Reason1 NIM for scene understanding, YOLO-World optimized with TensorRT, and we're building toward RAPIDS RAFT for vector search."

**Q: Is the OSHA matching real?**
> "Yes. We've embedded the actual text of 29 CFR 1910 Subpart I (PPE) and Subpart O (Machine Guarding)—the top 10 most-cited violations. The citations you see are real OSHA standards with 2024 penalty rates."

---

## Screen Checklist Before Demo

- [ ] Backend running: `python server/main.py` (port 8000)
- [ ] Frontend running: `npm run dev` (port 3000)
- [ ] At least 2-3 MP4 files in `data/` folder
- [ ] Browser open to `/dashboard/demo`
- [ ] Violation counter visible and incrementing
- [ ] Sample video ready to upload (or pre-uploaded)

---

## Timing Guide

| Section | Time | Duration |
|---------|------|----------|
| Opening Hook | 0:00 | 30s |
| How It Works walkthrough | 0:30 | 60s |
| Live Footage Review demo | 1:30 | 90s |
| VLM Deep Analysis | 3:00 | 60s |
| Digital Twin showcase | 4:00 | 30s |
| Closing & Value Prop | 4:30 | 30s |
| **Total** | | **5:00** |
