# Next Steps for OSHA-Vision

## Current Status (Updated Dec 13 - Evening)
- **VSS Engine**: Docker image rebuilt with GStreamer H.264/H.265 codecs
- **Configuration**: `DISABLE_CA_RAG=false` enables Chat/VLM endpoints
- **Archivist VLM Client**: Updated to use correct VSS API signature (`/summarize` with `enable_chat`)
- **Known Issue**: GStreamer decodebin caps negotiation failure inside VSS Engine's Python code

## Outstanding Issue: GStreamer Decodebin Caps Negotiation

The VSS Engine's video processing pipeline fails with `not-negotiated` error when trying to decode H.264 videos. Direct GStreamer command-line tests work fine, but the Python code's internal decodebin fails.

**Root Cause Analysis:**
1. The `avdec_h264` decoder works when tested with `gst-launch-1.0`
2. The VSS Engine's `video_file_frame_getter.py` creates pipelines dynamically
3. Inside `uridecodebin` -> `decodebin` -> `qtdemux`, the demuxer can't link to decoder
4. The issue appears to be caps negotiation between internal elements
5. Environment variable `GST_PLUGIN_FEATURE_RANK=avdec_h264:MAX` does not fix it

**Potential Solutions to Try:**
1. **Use nvv4l2decoder instead** - The NVIDIA hardware decoder may work where libav fails
2. **Use decodebin3** - Newer decodebin version with better negotiation
3. **Patch video_file_frame_getter.py** - Use explicit decoder elements instead of auto-plugging
4. **Use FFmpeg for frame extraction** - Bypass GStreamer entirely

## Key Discovery: GStreamer Codec Fix
The base VSS Engine image has a broken dpkg state where library packages show as installed but the actual .so files are missing. The fix in `Dockerfile.vss` requires:
1. Installing the packages normally
2. Force reinstalling critical libs: `apt-get reinstall -y libzvbi0t64 libslang2 libmp3lame0 ...`
3. Creating a symlink for libvpx: `ln -sf /usr/local/lib/python3.12/dist-packages/opencv_python.libs/libvpx-43afa4ba.so.9.0.0 /usr/lib/aarch64-linux-gnu/libvpx.so.9`

## VSS Engine API Reference

### Upload Video
```bash
curl -X POST "http://localhost:8000/files" \
  -F "file=@video.mp4" \
  -F "purpose=vision" \
  -F "media_type=video"
# Returns: {"id": "<asset_id>", ...}
```

### Analyze Video (Correct Method)
```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosmos-reason1",
    "id": "<asset_id>",
    "prompt": "Describe safety hazards in this video",
    "caption_summarization_prompt": "Generate safety-focused captions",
    "summary_aggregation_prompt": "Aggregate into safety report",
    "enable_chat": true
  }'
```

**Note:** `/chat/completions` uses `id` (not `video_uuid`) and returns a message saying "Chat functionality disabled" unless you use `/summarize` with `enable_chat: true`.

## Immediate Actions (Next Session)

1. **Debug GStreamer caps negotiation** - The core issue blocking video analysis:
   ```bash
   # Test if nvv4l2decoder works
   docker exec vss-engine gst-inspect-1.0 nvv4l2decoder

   # Try with explicit decoder
   docker exec vss-engine gst-launch-1.0 \
     filesrc location=/data/test_footage.mp4 ! qtdemux ! h264parse ! avdec_h264 ! \
     videoconvert ! video/x-raw,format=GBR ! fakesink -v
   ```

2. **Alternative: Use FFmpeg for frame extraction** in `video_file_frame_getter.py`:
   - Replace GStreamer pipeline with subprocess call to ffmpeg
   - Extract frames to numpy arrays

3. **Frontend Development** (Phase 4):
   - Replace placeholder HTML with Next.js
   - Implement video stream display from `/video_feed`
   - Add search interface for Archivist queries
   - Real-time violation alerts from `/status`

## Files Modified This Session
- `Dockerfile.vss` - Fixed GStreamer codec issues with force reinstall
- `archivist/vlm.py` - Updated to correct VSS API signature
- `docker-compose.yml` - Added `GST_PLUGIN_FEATURE_RANK=avdec_h264:MAX,avdec_h265:MAX`
- `video_file_frame_getter.py` - Changed to use uridecodebin for files, fixed capsfilter for software videoconvert

## Changes Made to video_file_frame_getter.py
1. Added `VSS_USE_URIDECODEBIN=true` env check to use uridecodebin for local files (not just RTSP)
2. Fixed capsfilter to use `video/x-raw` instead of `video/x-raw(memory:NVMM)` when nvvideoconvert unavailable

## Notes
- Model warmup takes ~2-3 minutes after container start
- First summarization request may be slow (CUDA graph compilation)
- Neo4j and Milvus connection errors in logs are expected (not configured)
- The container needs to be rebuilt if library fixes were applied manually inside the container
- `nvvideoconvert` and other DeepStream elements require tegra libraries not available on DGX
