# Dockerfile for base image (pytorch 2.7, cuda 12.8)

# The base image, a deep learning VM with PyTorch.
# For a full list, see https://us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# For info, see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
FROM gcr.io/deeplearning-platform-release/pytorch-cu124

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

RUN pip install -U pip
RUN pip install torch torchvision torchaudio -U --index-url https://download.pytorch.org/whl/cu128
RUN pip install huggingface-hub
