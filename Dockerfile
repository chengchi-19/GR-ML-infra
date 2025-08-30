# Multi-stage Dockerfile for GR Inference Optimization
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python
RUN ln -s /usr/bin/python3.9 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Triton
RUN pip install triton

# Install TensorRT (if available)
RUN pip install tensorrt || echo "TensorRT installation skipped"

# Set working directory
WORKDIR /workspace

# Copy source code
COPY . .

# Build TensorRT plugin
RUN cd kernels/trt_plugin_skeleton && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# Create model repository directory
RUN mkdir -p /models

# Copy model repository
COPY triton_model_repo /models/

# Set environment variables for Triton
ENV TRITON_MODEL_REPO=/models
ENV TRITON_SERVER_ARGS="--model-repository=/models --strict-model-config=false"

# Expose Triton ports
EXPOSE 8000 8001 8002

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting GR Inference Optimization Server..."\n\
echo "Model repository: $TRITON_MODEL_REPO"\n\
echo "Server arguments: $TRITON_SERVER_ARGS"\n\
\n\
# Start Triton server\n\
tritonserver $TRITON_SERVER_ARGS\n\
' > /workspace/start_server.sh && chmod +x /workspace/start_server.sh

# Create health check script
RUN echo '#!/bin/bash\n\
# Health check for Triton server\n\
curl -f http://localhost:8000/v2/health/ready || exit 1\n\
' > /workspace/health_check.sh && chmod +x /workspace/health_check.sh

# Set default command
CMD ["/workspace/start_server.sh"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install \
    pytest \
    pytest-cov \
    black \
    flake8 \
    isort \
    mypy \
    jupyter \
    ipython

# Create development workspace
WORKDIR /workspace/dev

# Copy development scripts
COPY scripts/ /workspace/scripts/
COPY tests/ /workspace/tests/

# Set development environment
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Development command
CMD ["/bin/bash"]

# Production stage
FROM base as production

# Optimize for production
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Remove development files
RUN rm -rf /workspace/tests /workspace/scripts

# Create non-root user
RUN useradd -m -u 1000 gruser && \
    chown -R gruser:gruser /workspace

USER gruser

# Production command
CMD ["/workspace/start_server.sh"]
