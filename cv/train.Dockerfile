# Dockerfile for building the CV image.


# The base image, an example deep learning VM.
# For a full list, see https://us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# For info, see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
FROM 2b3y-mmdetection:latest

WORKDIR /workspace/mmdetection

# Start training
CMD python tools/train.py configs/til-ai/convnext_extremesmall_12ep_siou.py --amp --auto-scale-lr
