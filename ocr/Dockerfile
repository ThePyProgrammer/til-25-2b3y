# Dockerfile for building the OCR image.

FROM base-torch2.7-cu128

RUN pip install python-doctr layoutparser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models during build
# COPY setup.py .
# RUN python setup.py
# RUN rm ./setup.py

# Copy your source files
COPY src .

# Starts your model server.
CMD ["uvicorn", "ocr_server:app", "--port", "5003", "--host", "0.0.0.0"]
