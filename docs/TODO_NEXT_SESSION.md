# Next Steps for OSHA-Vision

## Current Status (Updated Dec 13)
- **VSS Engine**: Docker image rebuilt with working GStreamer H.264/H.265 codecs
- **Configuration**: `DISABLE_CA_RAG=false` enables Chat/VLM endpoints
- **Archivist VLM Client**: Updated to use correct VSS API signature (`/summarize` with `enable_chat`)
- **Testing**: API structure verified, awaiting full pipeline test

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

1. **Rebuild Docker Image** (recommended for clean state):
   ```bash
   docker compose down vss-engine
   docker compose build --no-cache vss-engine
   docker compose up -d vss-engine
   # Wait ~3-5 min for model to load
   ```

2. **Test Full Pipeline**:
   ```bash
   # Check if API is ready
   curl http://localhost:8000/models

   # Upload and analyze test video
   python -m archivist.ingest
   ```

3. **Frontend Development** (Phase 4):
   - Replace placeholder HTML with Next.js
   - Implement video stream display from `/video_feed`
   - Add search interface for Archivist queries
   - Real-time violation alerts from `/status`

## Files Modified This Session
- `Dockerfile.vss` - Fixed GStreamer codec issues with force reinstall
- `archivist/vlm.py` - Updated to correct VSS API signature
- `docker-compose.yml` - `DISABLE_CA_RAG=false` (already set)

## Notes
- Model warmup takes ~2-3 minutes after container start
- First summarization request may be slow (CUDA graph compilation)
- Neo4j and Milvus connection errors in logs are expected (not configured)
- The container needs to be rebuilt if library fixes were applied manually inside the container
