# Dockerfile for building the CV image.


# The base image, an example deep learning VM.
# For a full list, see https://us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# For info, see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
FROM 2b3y-mmdetection:latest

# install api server dependencies
RUN pip install fastapi==0.115.12
RUN pip install uvicorn[standard]==0.34.2
RUN pip install more-itertools
RUN pip install sahi

# Copy model weights and config
COPY convnext_extremesmall_2_ep11.pth model.pth
COPY mmdetection/configs/til-ai/convnext_extremesmall_12ep_siou.py mmdetection/configs/til-ai/curr_conf.py

WORKDIR /workspace

# Copy source files.
COPY src .
COPY mmdetection mmdetection

# force no weights only load
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Starts your model server.
CMD uvicorn cv_server:app --port 5002 --host 0.0.0.0
