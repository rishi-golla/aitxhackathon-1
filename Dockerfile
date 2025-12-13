# OSHA Vision - Factory Safety Copilot
# Optimized for NVIDIA DGX Spark (GB10 Grace Blackwell)

FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download models (cache in image)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s-world.pt')" || true
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')" || true

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/osha_standards logs .alert_cache

# Expose ports
# 8501 - Streamlit Dashboard
# 8000 - FastAPI Backend
EXPOSE 8501 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command - run Streamlit dashboard
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
