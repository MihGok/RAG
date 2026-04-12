# ─────────────────────────────────────────────
#  Base: CUDA 12.9 devel
# ─────────────────────────────────────────────
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# ── Системные пакеты ────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates cmake pkg-config \
    python3 python3-dev python3-venv python3-distutils python3-pip \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ── Виртуальное окружение ───────────────────
ENV VENV_PATH=/opt/venv
RUN python3 -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

# ── pip / build tools ───────────────────────
RUN python -m pip install --upgrade pip setuptools wheel cmake

# ── PyTorch (CUDA 12.1 wheel, совместим с CUDA 12.9) ─
RUN python -m pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121

# ── llama-cpp-python с CUDA (GGML_CUDA=ON) ──
RUN python -m pip uninstall -y llama-cpp-python || true && \
    python -m pip cache purge || true && \
    CMAKE_ARGS="-DGGML_CUDA=ON" \
    python -m pip install --no-cache-dir --force-reinstall \
      llama-cpp-python \
      --config-setting="cmake.args=-DGGML_CUDA=ON"

# ── faster-whisper ───────────────────────────
RUN python -m pip install --no-cache-dir \
    faster-whisper \
    ctranslate2

# ── yt-dlp ──────────────────────────────────
RUN python -m pip install --no-cache-dir yt-dlp

# ── FastAPI + сервер ─────────────────────────
RUN python -m pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    httpx \
    aiofiles \
    python-multipart

# ── Метрики качества ─────────────────────────
# sacrebleu — BLEU / chrF
# rouge-score — ROUGE-1/2/L
# bert-score — BERTScore (F1 на эмбеддингах)
# nltk — вспомогательные токенизаторы
RUN python -m pip install --no-cache-dir \
    sacrebleu \
    rouge-score \
    bert-score \
    nltk

# Предзагрузка NLTK токенизаторов
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# ── Google Generative AI (Gemini) ────────────
RUN python -m pip install --no-cache-dir \
    google-generativeai

# ── Рабочая директория (монтируется снаружи) ─
WORKDIR /srv/endpoint

EXPOSE 8000

CMD ["/opt/venv/bin/uvicorn", "main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--reload"]
