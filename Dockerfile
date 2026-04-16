FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages (torch/torchaudio already in base image)
RUN pip install --no-cache-dir \
    torchaudio \
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

# Environment variables
ENV HF_HOME=/app/.cache/huggingface
ENV TF_CPP_MIN_LOG_LEVEL=3

# Copy application source
COPY . .

EXPOSE 8000

RUN python -c "import speech; import audio"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
