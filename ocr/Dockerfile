# Dockerfile for building the OCR image.

# The base image, an example deep learning VM.
# For a full list, see https://us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# For info, see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
FROM gcr.io/deeplearning-platform-release/pytorch-cu118.2-0

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Detectron2 dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pre-download models during build
COPY setup.py .
RUN python setup.py
RUN rm ./setup.py

# Copy your source files
COPY src/ .

# Starts your model server.
CMD ["uvicorn", "ocr_server:app", "--port", "5003", "--host", "0.0.0.0"]
