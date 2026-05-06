from __future__ import annotations

import json
import logging
import os
from typing import Callable, Dict, Any, List, Optional

from .embedder       import embed_lesson_names
from .clusterer      import cluster_embeddings, labels_to_groups
from .merger         import decide_merges_in_cluster, generate_chunk
from db.qdrant_indexer import index_all_chunks
from .course_generator import generate_course_structure, save_course
from .doc_generator  import generate_course_docx

logger = logging.getLogger(__name__)

EMBED_MODEL  = "Qwen3-Embedding-0.6B-BF16.gguf"
MERGE_MODEL  = "Qwen2.5-3B-Instruct-Q4_K_L.gguf"
COURSE_MODEL = "t-lite-it-1.0-q4_k_m.gguf"


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _load_tag_map(session_dir: str) -> Dict[str, List[str]]:
    path = os.path.join(session_dir, "tag_map.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_session_info(session_dir: str) -> Dict[str, Any]:
    path = os.path.join(session_dir, "session_info.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_chunks(chunks: List[Dict], session_dir: str):
    final_dir = os.path.join(session_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    with open(os.path.join(final_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def _save_course_structure(structure: Dict, session_dir: str):
    final_dir = os.path.join(session_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    with open(os.path.join(final_dir, "course_structure.json"), "w", encoding="utf-8") as f:
        json.dump(structure, f, ensure_ascii=False, indent=2)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def run_stage2(
    session_dir: str,
    user_id: int = 0,
    chat_id: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Полный Stage 2.

    Returns dict с ключами:
        lessons_count, clusters_count, chunks_count, indexed_count,
        collection_name, modules_count, docx_path, mongo_id
    """

    def _log(msg: str):
        logger.info(msg)
        if log_fn:
            log_fn(msg)

    raw_data_dir = os.path.join(session_dir, "raw_data")
    session_info = _load_session_info(session_dir)
    session_id   = session_info.get("session_id", os.path.basename(session_dir))
    topic        = session_info.get("topic", "Курс")
    tag_map      = _load_tag_map(session_dir)

    collection_name = f"rag_{session_id[:20].replace('-', '_')}"

    _log(f"\n{'='*55}")
    _log(f"  STAGE 2  |  Тема: {topic}")
    _log(f"{'='*55}")

    # ── 1. Загрузка и векторизация ────────────────────────────────────────
    _log("\n[1/6] Векторизация названий уроков...")
    lessons, embeddings = embed_lesson_names(raw_data_dir, EMBED_MODEL, log_fn=_log)

    if not lessons:
        _log("❌ Уроки не найдены в raw_data/")
        return {"error": "no lessons"}

    _log(f"      ✅ {len(lessons)} уроков")

    # ── 2. Кластеризация ─────────────────────────────────────────────────
    _log("\n[2/6] Кластеризация...")
    import numpy as np
    if len(embeddings) > 0:
        labels         = cluster_embeddings(embeddings, log_fn=_log)
        cluster_groups = labels_to_groups(len(lessons), labels)
    else:
        cluster_groups = [[i] for i in range(len(lessons))]

    _log(f"      ✅ {len(cluster_groups)} кластеров")

    # ── 3–4. Решения о слиянии + генерация чанков ────────────────────────
    _log("\n[3/6] Генерация чанков знаний (Qwen)...")
    all_chunks: List[Dict[str, Any]] = []

    for ci, group_indices in enumerate(cluster_groups):
        cluster_lessons = [lessons[i] for i in group_indices]
        merge_groups    = decide_merges_in_cluster(cluster_lessons)

        for mg in merge_groups:
            to_merge = [cluster_lessons[j] for j in mg]
            titles   = [l[1] for l in to_merge]
            _log(f"  Чанк [{ci+1}]: {titles}")
            chunk = generate_chunk(to_merge, tag_map, session_dir)
            all_chunks.append(chunk)

    _log(f"      ✅ Создано чанков: {len(all_chunks)}")
    _save_chunks(all_chunks, session_dir)

    # ── 5. Индексация в Qdrant ────────────────────────────────────────────
    _log(f"\n[4/6] Индексация в Qdrant ({collection_name})...")
    indexed_count = index_all_chunks(
        chunks=all_chunks,
        collection_name=collection_name,
        embed_model=EMBED_MODEL,
        log_fn=_log,
    )
    _log(f"      ✅ Проиндексировано: {indexed_count}/{len(all_chunks)}")

    # ── 6. Структура курса (T-lite) ───────────────────────────────────────
    _log("\n[5/6] Генерация структуры курса (T-lite)...")
    course_structure = generate_course_structure(
        topic=topic,
        chunks=all_chunks,
        tag_map=tag_map,
        model_name=COURSE_MODEL,
        log_fn=_log,
    )
    _save_course_structure(course_structure, session_dir)
    modules_count = len(course_structure.get("modules", []))
    _log(f"      ✅ {modules_count} модулей")

    # ── 7. DOCX документ ─────────────────────────────────────────────────
    _log("\n[6/6] Генерация DOCX-документа...")
    docx_path = ""
    try:
        docx_out = os.path.join(session_dir, "final", "course.docx")
        docx_path = generate_course_docx(
            course_structure=course_structure,
            chunks=all_chunks,
            output_path=docx_out,
            topic=topic,
        )
        _log(f"      ✅ Документ: {docx_path}")
    except Exception as e:
        _log(f"      ⚠️ DOCX ошибка: {e}")

    # ── 8. Сохранение в БД ───────────────────────────────────────────────
    _log("\n[DB] Сохранение в MongoDB + PostgreSQL...")
    mongo_id = ""
    try:
        mongo_id = save_course(
            course_structure=course_structure,
            session_id=session_id,
            topic=topic,
            chunks_count=len(all_chunks),
            user_id=user_id,
            chat_id=chat_id,
            log_fn=_log,
        )
    except Exception as e:
        _log(f"      ⚠️ DB ошибка: {e}")

    result = {
        "session_id":      session_id,
        "topic":           topic,
        "lessons_count":   len(lessons),
        "clusters_count":  len(cluster_groups),
        "chunks_count":    len(all_chunks),
        "indexed_count":   indexed_count,
        "collection_name": collection_name,
        "modules_count":   modules_count,
        "docx_path":       docx_path,
        "mongo_id":        mongo_id,
    }

    _log(f"\n{'='*55}")
    _log(f"  ✅ STAGE 2 ЗАВЕРШЁН")
    _log(f"  Уроков: {len(lessons)} → Чанков: {len(all_chunks)}")
    _log(f"  Коллекция: {collection_name}")
    _log(f"  Документ: {docx_path}")
    _log(f"{'='*55}\n")

    return result