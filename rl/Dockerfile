# Dockerfile for building the RL image.


# The base image. Using a regular Python base because RL agents don't typically
# require much GPU. This will speed up your evaluation.
FROM python:3.12-slim

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies your source files.
COPY src/rl_* .
COPY src/grid ./grid
COPY src/networks ./networks
COPY src/models ./models
COPY src/utils ./utils
COPY src/agent ./agent


# Starts your model server.
CMD ["uvicorn", "rl_server:app", "--port", "5004", "--host", "0.0.0.0"]
