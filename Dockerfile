FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install TTS (coqui) system-level dep if needed; skip for now (commented in flake)
# RUN apt-get install -y ... 

WORKDIR /app

# Python dependencies
# Install torch/torchaudio separately first (large, benefits from layer caching)
RUN pip install --no-cache-dir \
    torch \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python packages
RUN pip install --no-cache-dir \
    transformers \
    librosa \
    dtw-python \
    phonemizer \
    numpy \
    fastapi \
    python-multipart \
    pydub \
    gTTS \
    fastdtw \
    scipy \
    levenshtein \
    scikit-learn \
    streamlit \
    plotly \
    requests \
    uvicorn \
    soundfile \
    huggingface-hub

# Environment variables (mirrors shellHook)
ENV HF_HOME=/app/.cache/huggingface
ENV TF_CPP_MIN_LOG_LEVEL=3

# Copy application source
COPY . .

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Default command — override as needed
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
