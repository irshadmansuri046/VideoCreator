# Production Dockerfile for Image-to-Cartoon-Video Generator
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies (FFmpeg, OpenCV shared libraries, Python3)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install Python requirements
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy application codebase and assets
COPY . /app

# Expose default Gradio port
EXPOSE 7860

# Run Gradio application
CMD ["python3", "app.py", "--server_name", "0.0.0.0", "--server_port", "7860"]
