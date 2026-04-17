"""
stage2/embedder.py
──────────────────
Загружает JSON-файлы уроков и векторизует их названия через /task endpoint.
"""

from __future__ import annotations

import os
import json
import logging
from typing import List, Tuple, Dict, Any

import numpy as np
import requests

from config import AppConfig

logger = logging.getLogger(__name__)

EMBED_URL   = f"{AppConfig.ML_SERVER_URL.rstrip('/')}/task"
DEFAULT_MODEL = "Qwen3-Embedding-0.6B-f16.gguf"


# ─── I/O ─────────────────────────────────────────────────────────────────────

def load_lesson_files(raw_data_dir: str) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Читает все .json файлы из raw_data_dir.

    Returns:
        List of (filename, lesson_name, data_dict)
    """
    lessons: List[Tuple[str, str, Dict]] = []
    if not os.path.isdir(raw_data_dir):
        logger.warning("raw_data_dir не найден: %s", raw_data_dir)
        return lessons

    for fname in sorted(os.listdir(raw_data_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(raw_data_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            lesson_name = data.get("lesson_name") or fname.replace(".json", "")
            lessons.append((fname, lesson_name, data))
        except Exception as e:
            logger.warning("Не удалось прочитать %s: %s", fname, e)

    logger.info("Загружено уроков: %d", len(lessons))
    return lessons


# ─── EMBEDDING ───────────────────────────────────────────────────────────────

def _embed_batch(texts: List[str], model_name: str) -> List[List[float]]:
    """Отправляет батч текстов на /task (embed) и возвращает эмбеддинги."""
    payload = {
        "task_type":  "embed",
        "model_name": model_name,
        "texts":      texts,
        "n_ctx":      512,
    }
    try:
        resp = requests.post(EMBED_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("embeddings", [])
    except requests.exceptions.ConnectionError:
        logger.error("Не могу подключиться к ML backend: %s", EMBED_URL)
        return [[] for _ in texts]
    except Exception as e:
        logger.error("Embed error: %s", e)
        return [[] for _ in texts]


def embed_lesson_names(
    raw_data_dir: str,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    log_fn=None,
) -> Tuple[List[Tuple[str, str, Dict]], np.ndarray]:
    """
    Загружает уроки, векторизует их названия батчами.

    Returns:
        valid_lessons: list of (filename, lesson_name, data) — только те, для которых
                       удалось получить эмбеддинг
        embeddings: np.ndarray shape (n, dim)
    """
    def _log(msg: str):
        logger.info(msg)
        if log_fn:
            log_fn(msg)

    lessons = load_lesson_files(raw_data_dir)
    if not lessons:
        return [], np.array([])

    names = [l[1] for l in lessons]
    all_embeddings: List[List[float]] = []

    for i in range(0, len(names), batch_size):
        batch = names[i : i + batch_size]
        embs  = _embed_batch(batch, model_name)
        all_embeddings.extend(embs)
        _log(f"[Embed] {min(i + batch_size, len(names))}/{len(names)} уроков")

    # Фильтруем пустые эмбеддинги
    valid_lessons: List[Tuple[str, str, Dict]] = []
    valid_embeddings: List[List[float]] = []

    for lesson, emb in zip(lessons, all_embeddings):
        if emb and len(emb) > 0:
            valid_lessons.append(lesson)
            valid_embeddings.append(emb)
        else:
            _log(f"[Embed] Пропущен урок (нет эмбеддинга): {lesson[1]}")

    if not valid_embeddings:
        return valid_lessons, np.array([])

    embeddings_matrix = np.array(valid_embeddings, dtype=np.float32)
    _log(f"[Embed] Готово: {len(valid_lessons)} уроков, dim={embeddings_matrix.shape[1]}")
    return valid_lessons, embeddings_matrix


# ─── SINGLE TEXT ─────────────────────────────────────────────────────────────

def embed_text(text: str, model_name: str = DEFAULT_MODEL) -> List[float]:
    """Векторизует одну строку. Используется при индексации в Qdrant."""
    payload = {
        "task_type":  "embed",
        "model_name": model_name,
        "text":       text,
        "n_ctx":      512,
    }
    try:
        resp = requests.post(EMBED_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("embedding", [])
    except Exception as e:
        logger.error("embed_text error: %s", e)
        return []