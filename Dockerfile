# RunPod serverless worker for RobustVideoMatting (video background removal).
# Build from repo root: docker build -f runpod-rvm-worker/Dockerfile -t runpod-rvm-worker:latest runpod-rvm-worker
# Or from this folder: docker build -t runpod-rvm-worker:latest .

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime@sha256:82e0d379a5dedd6303c89eda57bcc434c40be11f249ddfadfd5673b84351e806

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV APT_OPTS="-o Acquire::Retries=3 -o Acquire::http::Timeout=60"

RUN apt-get update ${APT_OPTS} \
    && apt-get install -y --no-install-recommends git wget ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN git clone --depth 1 https://github.com/PeterL1n/RobustVideoMatting.git /app/RobustVideoMatting \
    && mkdir -p /app/RobustVideoMatting/checkpoints \
    && wget -q https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth \
        -O /app/RobustVideoMatting/checkpoints/rvm_mobilenetv3.pth

COPY patches/inference_utils.py /app/RobustVideoMatting/inference_utils.py

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/handler.py /app/handler.py

ENV PYTHONPATH=/app/RobustVideoMatting
ENV RUNPOD_INIT_TIMEOUT=600

CMD ["python", "-u", "/app/handler.py"]
