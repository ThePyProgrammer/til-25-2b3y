# Dockerfile for building the CV image.
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# replace the ubuntu mirror, retry more
RUN sed 's/archive.ubuntu.com/asia-southeast1.gce.archive.ubuntu.com/' -i /etc/apt/sources.list
RUN sed 's/security.ubuntu.com/asia-southeast1.gce.archive.ubuntu.com/' -i /etc/apt/sources.list
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries

# install opencv dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git

# install mm components
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install mmpretrain

# install mmcv
WORKDIR /deps
RUN git clone https://github.com/open-mmlab/mmcv.git --branch=v2.2.0
WORKDIR /deps/mmcv
RUN pip install opencv-python-headless
RUN pip install -r requirements/optional.txt
ARG FORCE_CUDA=1
ARG TORCH_CUDA_ARCH_LIST='Turing Ada Blackwell'
RUN pip install . -v

# install mmdetection
WORKDIR /workspace
COPY mmdetection .
RUN pip install -e .

# fix pycocotools
RUN pip install pycocotools==2.0.8

# install albumentations
RUN pip install albumentations