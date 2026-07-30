# ==========================================
# 1. AŞAMA: Derleme ve Bağımlılık Kurulumu
# ==========================================
FROM cgr.dev/chainguard/python:latest-dev AS builder
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ==========================================
# 2. AŞAMA: Çalışma Zamanı (Runtime Image)
# ==========================================
FROM cgr.dev/chainguard/python:latest AS runner
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Sadece zorunlu sistem paketleri ve temizlik
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-tur \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --from=builder /install /usr/local
COPY . /app/

RUN mkdir -p /app/data/raw /app/data/thumbnails

# PyTorch bellek parçalanmasını önlemek için
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -sSf http://localhost:8000/docs || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]