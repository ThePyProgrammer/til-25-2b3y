# Dockerfile for building the ASR image.


# The base image, a deep learning VM with PyTorch.
# For a full list, see https://us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# For info, see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
# FROM gcr.io/deeplearning-platform-release/pytorch-cu124
FROM nvcr.io/nvidia/pytorch:25.05-py3

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

RUN apt-get update && apt-get install -y libsndfile1 ffmpeg

# Installs your dependencies.
RUN pip install -U pip

RUN pip install huggingface-hub==0.21.3

COPY download_model.py .
RUN mkdir ./models && python download_model.py --model_name nvidia/parakeet-tdt-0.6b-v2 --path ./models/
RUN rm ./download_model.py

RUN pip install Cython packaging
RUN pip install nemo_toolkit["asr"]

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies your source files and setup script.
# COPY setup.py .
# RUN python setup.py
# RUN rm ./setup.py

COPY src/asr_* .
COPY src/word* .

# Starts your model server.
CMD ["uvicorn", "asr_server:app", "--port", "5001", "--host", "0.0.0.0"]