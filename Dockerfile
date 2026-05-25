FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# llama-cpp-python prebuilt wheel (CPU, HF Spaces compatible)
RUN pip install --no-cache-dir \
    https://huggingface.co/Luigi/llama-cpp-python-wheels-hf-spaces-free-cpu/resolve/main/llama_cpp_python-0.3.22-cp310-cp310-linux_x86_64.whl

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# V4 engine modules
COPY server.py .
COPY features.py .
COPY sde.py .
COPY calibration.py .
COPY economic.py .
COPY edge_laplace.py .

# Extraction + forecasting
COPY medgemma_amr_extractor.py .
COPY forecasting_engine.py .

# Unified entrypoint
COPY main.py .

RUN mkdir -p /data

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
