FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies your source files.
COPY src .

# Starts your model server.
CMD ["uvicorn", "surprise_server:app", "--port", "5005", "--host", "0.0.0.0"]