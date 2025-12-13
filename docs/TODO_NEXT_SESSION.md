# Next Steps for OSHA-Vision

## Current Status
- **VSS Engine**: Successfully built custom image with GStreamer dependencies. Container starts up and loads the model (`cosmos-reason1`).
- **Configuration**: Updated `docker-compose.yml` to set `DISABLE_CA_RAG=false` to enable the Chat/VLM endpoints.
- **Testing**: Initial API tests failed because Chat was disabled.

## Immediate Actions (Next Session)
1.  **Restart VSS Engine**:
    Run `docker compose up -d --build vss-engine` (or just `up -d` since we edited compose) to apply the `DISABLE_CA_RAG=false` change.
    *Note: This will trigger a model reload/warmup which takes ~2-3 minutes.*

2.  **Verify API**:
    Once the server is ready (check logs for `Uvicorn running`), test the chat endpoint again.
    You can use a simple curl command or re-create the test script:
    ```bash
    curl -X POST "http://localhost:8000/chat/completions" \
         -H "Content-Type: application/json" \
         -d '{
               "model": "cosmos-reason1",
               "messages": [{"role": "user", "content": "Describe this video"}],
               "video_uuid": ["<YOUR_ASSET_ID>"]
             }'
    ```

3.  **Ingestion Pipeline**:
    - Update `archivist/vlm.py` and `archivist/ingest.py` to match the verified API signature.
    - Run `python3 -m archivist.ingest` to process the `data/` folder.

4.  **Search & Frontend**:
    - Complete Phase 3 (Archivist) and move to Phase 4 (Frontend).
