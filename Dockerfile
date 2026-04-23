# ML Search Service Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and ML libraries
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgflags-dev \
    libsnappy-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for models and data
RUN mkdir -p /app/models /app/data/faiss_index

# Pre-download models during build (optional - reduces startup time)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || echo "YOLO model download skipped"
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='en', show_log=False)" || echo "OCR model download skipped"

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8001/health')" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
