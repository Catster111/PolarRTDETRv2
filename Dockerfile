# Use PyTorch with CUDA support as base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    unzip \
    cmake \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libopencv-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install dlib dependencies and dlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir dlib

# Install additional face detection and landmark specific packages
RUN pip install --no-cache-dir \
    face-alignment \
    facenet-pytorch \
    insightface

# Create directories for data, models, and outputs
RUN mkdir -p /app/data /app/models /app/outputs /app/configs

# Set up PYTHONPATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Set default command to bash
CMD ["/bin/bash"]

# Add entrypoint script for training
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
