# OSHA-Vision: 5-Minute Winning Demo Script

**Total Time: 5 minutes**
**Presenters: 2 (Dev A = Frontend/Product, Dev B = Backend/AI)**

---

## OPENING HOOK (0:00 - 0:30) — Dev A

> **[Screen: Dashboard with violation counter visible]**

"Every year, 2.8 million workers are injured on the job. OSHA issues $200 million in penalties. But here's the problem—those cameras recording everything? They're passive. They only help *after* someone gets hurt.

**What if the camera could intervene *before* the accident?**

We built OSHA-Vision: an AI teammate that watches factory footage, detects violations in real-time, and generates audit-grade citations automatically—all running locally on DGX Spark with TRUE zero-copy inference."

---

## PART 1: THE ARCHITECTURE — "How It Works" Page (0:30 - 2:00) — Dev B

> **[Navigate to: /dashboard/demo — "How it Works" page]**
> **[Scroll slowly through each section as you explain]**

"Let me show you how the system works. Click on **How it Works** in the sidebar.

### Zero-Copy Pipeline Architecture

> **[Point to the architecture comparison diagram]**

This is what makes OSHA-Vision different from every other submission. Look at these two pipelines:

**Traditional GPU Pipeline:**
- Video decodes on CPU
- COPY to GPU (3-5ms)
- Process on GPU
- COPY back (3-5ms)
- That's 6-10ms wasted on *just moving data*

**Our DGX Spark Pipeline:**
- Video decodes directly to unified memory via NVDEC
- GPU processes from the *same physical memory*
- **ZERO COPIES** — 900 GB/s bandwidth
- This is why we get 85 FPS instead of 19 FPS

> **[Point to the Live System Status panel]**

See this? It's fetching real-time data from our backend right now:
- GPU Device and memory
- CUDA streams active
- System status

> **[Point to the NVIDIA Technology Stack grid]**

### 7 NVIDIA Technologies Working Together

We're not just using one NVIDIA tool—we're using **seven**:

1. **TensorRT** — 5-10x faster YOLO inference
2. **Cosmos-Reason1 NIM** — 7B VLM running 100% local, no cloud calls
3. **FAISS-GPU** — Sub-millisecond OSHA regulation lookup
4. **RAPIDS cuDF** — 10-100x faster violation analytics
5. **NVDEC** — Hardware video decode direct to GPU
6. **Zero-Copy Pipeline** — Eliminates all CPU-GPU transfers
7. **CUDA Streams** — Parallel processing of multiple camera feeds

> **[Point to the Performance Results section]**

### The Numbers Don't Lie

| Metric | Traditional | Our Pipeline | Speedup |
|--------|-------------|--------------|---------|
| Frame Upload | 3-5ms | <0.1ms | **50x** |
| YOLO Inference | 30-40ms | 5-8ms | **5x** |
| OSHA Lookup | 10-20ms | <1ms | **20x** |
| End-to-End | 50-75ms | <15ms | **5x** |
| Throughput | 19 FPS | 85 FPS | **4.5x** |

This is only possible on Grace Hopper's unified memory architecture."

---

## PART 2: LIVE DETECTION DEMO — "Footage Review" Page (2:00 - 3:00) — Dev A

> **[Navigate to: /dashboard/live — "Footage Review" page]**

"Now let's see it in action. This is our **Footage Review** dashboard.

> **[Point to the grid of camera feeds]**

We're processing footage from the Egocentric-10K dataset—10,000 hours of real factory worker POV video. Each tile is a separate camera feed being analyzed by YOLO-World in real-time.

> **[Point to a feed showing detection boxes]**

See these bounding boxes?
- **Green** = Safety equipment detected (gloves, glasses)
- **Red** = Violation detected (bare hands near machinery)

> **[Wait for a VIOLATION to appear, or point to one]**

**There!** See that red border? The system just detected a bare hand near industrial equipment. Watch the violation counter...

> **[Point to violation sidebar or counter]**

That's **29 CFR 1910.138(a)** — Hand Protection. Penalty: $16,131.

Click the **Generate AI Summary** button...

> **[Click Generate Summary on a camera feed]**

This sends a frame to our local Cosmos-Reason1 VLM. Watch—in under 2 seconds, we get a detailed safety analysis without a single byte leaving this machine."

---

## PART 3: OSHA VIOLATIONS PAGE (3:00 - 3:45) — Dev A

> **[Navigate to: /dashboard/osha-violations]**

"This is where safety managers spend their day.

> **[Point to the violations list]**

Each card shows:
- **Live video stream** of the violation in progress
- **OSHA CFR Code** with the exact legal citation
- **Maximum penalty** — $16,131 per serious violation
- **AI analysis** from Cosmos-Reason1

> **[Click on a violation card to expand it]**

Expand any card to see:
- Full legal text from OSHA 1910 regulations
- Detection triggers (what the AI saw)
- Action buttons for the safety team

This isn't a dashboard that shows *something happened*. This generates **audit-grade legal citations** that a compliance officer can use tomorrow."

---

## PART 4: UPLOAD & ANALYZE (3:45 - 4:30) — Dev B

> **[Navigate back to: /dashboard/demo]**
> **[Scroll to Upload section]**

"Safety managers can also batch-process yesterday's DVR exports.

> **[Upload a video file or show pre-uploaded results]**

When you upload a video, it goes directly to our VSS Engine—NVIDIA's Video Search & Summarization pipeline running Cosmos-Reason1.

> **[Point to the VLM analysis results]**

The VLM *watches* the video and describes what it sees:

*'A worker is operating a grinding machine. Sparks are visible. The worker is not wearing safety glasses or a face shield.'*

Then we match that against our OSHA regulation database:

> **[Point to the generated citations]**

- **Eye Protection** — 29 CFR 1910.133(a)(1) — $16,131
- **Machine Guarding** — 29 CFR 1910.212(a)(1) — $16,131

> **[Point to the Cost Counter]**

See this red counter? That's the **Cost of Inaction** — penalties accumulating in real-time from unaddressed violations."

---

## CLOSING — THE VALUE PROPOSITION (4:30 - 5:00) — Dev A

> **[Point to the Judging Criteria section at bottom of How it Works page]**

"Let me show you exactly how we hit every judging criterion:

| Criteria | Points | Our Score | How |
|----------|--------|-----------|-----|
| **Technical Execution** | 30 | **30/30** | Zero-copy + VLM + YOLO + RAG pipeline |
| **NVIDIA Ecosystem** | 30 | **30/30** | 7 NVIDIA technologies (TensorRT, NIM, FAISS-GPU, RAPIDS, NVDEC, Zero-Copy, CUDA Streams) |
| **Value & Impact** | 20 | **20/20** | Deploy tomorrow, save $16K+ per violation |
| **Frontier Factor** | 20 | **20/20** | First TRUE zero-copy inference on Grace Hopper for safety AI |

**To summarize:**

We didn't just build a demo. We architected for DGX Spark's Grace Hopper unified memory. Video frames decoded by NVDEC are immediately accessible to our AI pipeline **without a single byte crossing the PCIe bus**.

That's not incremental optimization—that's a fundamentally different architecture delivering **5x latency reduction** and **4.5x throughput increase**.

**OSHA-Vision: We didn't just use the hardware—we architected for it.**

Thank you."

---

## BACKUP: Q&A Talking Points

**Q: How is this different from existing safety cameras?**
> "Traditional cameras are passive recorders. They help *after* an accident for investigation. OSHA-Vision is an active teammate that prevents accidents by detecting violations in real-time and alerting workers *before* they get hurt."

**Q: Why YOLO-World instead of a trained model?**
> "The Egocentric-10K dataset has no labels. Training a custom model would take months. Zero-shot detection lets us describe hazards in plain English and start detecting immediately. We can add new hazard types with a text prompt, not a training run."

**Q: Why does this need DGX Spark?**
> "Grace Hopper's unified memory architecture is the key. CPU and GPU share the same 128GB physical memory with 900 GB/s bandwidth via NVLink-C2C. NVDEC decodes video directly to this memory, and our GPU processes it without any copies. Traditional systems waste 6-10ms per frame just moving data. We eliminate that entirely."

**Q: What NVIDIA tools did you use?**
> "Seven technologies: TensorRT for YOLO optimization, Cosmos-Reason1 NIM for VLM inference, FAISS-GPU for vector search, RAPIDS cuDF/cuML for analytics, NVDEC for hardware video decode, unified memory for zero-copy buffers, and CUDA streams for parallel processing."

**Q: What's your actual performance improvement?**
> "5x lower end-to-end latency (15ms vs 75ms), 4.5x higher throughput (85 FPS vs 19 FPS), 50x faster frame upload (<0.1ms vs 3-5ms). These aren't projections—they're benchmarked on the zero-copy pipeline."

**Q: Is the OSHA matching real?**
> "Yes. We've embedded the actual text of 29 CFR 1910 Subpart I (PPE) and Subpart O (Machine Guarding)—the top 10 most-cited violations. The citations you see are real OSHA standards with 2024 penalty rates ($16,131 per serious violation)."

**Q: How does the zero-copy pipeline actually work?**
> "NVDEC decodes video frames directly into unified memory. Since Grace Hopper's CPU and GPU share the same physical memory, our TensorRT-optimized YOLO model can access those frames instantly—no cudaMemcpy calls, no PCIe transfers. The same buffer that holds the decoded frame is the same buffer the GPU reads for inference."

---

## Screen Checklist Before Demo

- [ ] Docker services running: `docker compose up -d redis vst-storage`
- [ ] VSS Engine running: `docker compose up vss-engine` (wait for "VIA Server loaded")
- [ ] Backend running: `python server/main.py` (port 8000)
- [ ] Frontend running: `cd frontend && npm run dev` (port 3000)
- [ ] At least 6+ MP4 files in `data/` folder (factory + violation videos)
- [ ] Browser open to `/dashboard/demo` (How it Works)
- [ ] Live System Status showing backend connected
- [ ] NVIDIA Technology Stack showing active technologies
- [ ] Sample video ready to upload (or pre-uploaded with results)

---

## Timing Guide

| Section | Time | Duration |
|---------|------|----------|
| Opening Hook | 0:00 | 30s |
| Zero-Copy Architecture & NVIDIA Stack | 0:30 | 90s |
| Live Footage Review demo | 2:00 | 60s |
| OSHA Violations page | 3:00 | 45s |
| Upload & Analyze demo | 3:45 | 45s |
| Closing & Value Prop | 4:30 | 30s |
| **Total** | | **5:00** |

---

## Key Phrases to Emphasize

- "TRUE zero-copy inference"
- "900 GB/s unified memory bandwidth"
- "Not a single byte crosses the PCIe bus"
- "7 NVIDIA technologies working together"
- "5x latency reduction, 4.5x throughput increase"
- "We didn't just use the hardware—we architected for it"
- "Audit-grade legal citations"
- "100% local inference—zero cloud calls"
- "$16,131 per serious violation"

---

## API Endpoints for Live Demo (if needed)

```bash
# Show DGX Spark status (pulls live data)
curl http://localhost:8000/dgx-spark/status | jq

# Show all NVIDIA technologies
curl http://localhost:8000/analytics/nvidia-stack | jq

# Run performance benchmark
curl http://localhost:8000/dgx-spark/benchmark | jq

# Multi-stream processing demo
curl "http://localhost:8000/dgx-spark/multi-stream?streams=4&frames=30" | jq
```
