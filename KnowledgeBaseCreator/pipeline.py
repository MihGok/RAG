"""
KnowledgeBaseCreator/pipeline.py
──────────────────────────────────
Stage 2: кластеризация → генерация чанков → индексация → DOCX.

Ключевое изменение — секвенциальная delta-генерация:

  Модули обрабатываются строго по порядку (0, 1, 2, ...).
  Кластеры внутри каждого модуля сортируются по средней позиции уроков.
  Каждый следующий чанк получает:
    previously_covered_in_module   — learned_concepts всех предыдущих чанков модуля
    cross_module_context           — накопленный список {title, concepts} по всем
                                     предыдущим модулям (последние 3 подробно,
                                     остальные — только названия)
  Это гарантирует связность и отсутствие дублирования материала.

  decide_merges_in_cluster() также получает module_info — LLM видит цели
  и темы модуля при группировке уроков.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from .embedder  import embed_lesson_names
from .clusterer import (
    cluster_embeddings,
    cluster_by_module_and_embeddings,
    labels_to_groups,
)
from .merger         import decide_merges_in_cluster, generate_chunk
from db.qdrant_indexer import index_all_chunks
from .course_generator import generate_course_structure, save_course
from .doc_generator    import generate_course_docx
from db.mongo_service  import save_syllabus

logger = logging.getLogger(__name__)

EMBED_MODEL  = "Qwen3-Embedding-0.6B-BF16.gguf"
COURSE_MODEL = "Qwen3.5-9B-Q5_K_M.gguf"


# Максимум концептов из одного модуля в cross-module контексте
_MAX_CONCEPTS_PER_MODULE = 15


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _load_tags(session_dir: str) -> List[str]:
    for fname in ("tags.json", "tag_map.json"):
        path = os.path.join(session_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        tags: List[str] = []
        for v in (data.values() if isinstance(data, dict) else []):
            if isinstance(v, list):
                tags.extend(v)
        return tags
    return []


def _load_session_info(session_dir: str) -> Dict[str, Any]:
    path = os.path.join(session_dir, "session_info.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_syllabus(session_dir: str) -> Dict[str, Any]:
    """Силлабус Stage 1 (9B thinking): course_structure.json в корне сессии."""
    path = os.path.join(session_dir, "course_structure.json")
    if not os.path.exists(path):
        logger.warning("Силлабус Stage 1 не найден: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_chunks(chunks: List[Dict], session_dir: str) -> None:
    final_dir = os.path.join(session_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    with open(os.path.join(final_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def _save_stage2_course_structure(structure: Dict, session_dir: str) -> None:
    """Stage 2 структура — в final/, не путать с Stage 1 в корне."""
    final_dir = os.path.join(session_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    with open(os.path.join(final_dir, "course_structure.json"), "w", encoding="utf-8") as f:
        json.dump(structure, f, ensure_ascii=False, indent=2)


def _module_ids_from_lessons(lessons: List) -> List[int]:
    return [int(data.get("module_id", -1)) for _, _, data in lessons]


def _avg_position(group_indices: List[int], lessons: List) -> float:
    """Средняя позиция уроков группы — для сортировки кластеров внутри модуля."""
    positions = [lessons[i][2].get("position", 0) for i in group_indices]
    return sum(positions) / len(positions) if positions else 0.0


def _group_majority_module(group_indices: List[int], module_ids: List[int]) -> int:
    """module_id большинства уроков в группе."""
    mids = [module_ids[i] for i in group_indices]
    return Counter(mids).most_common(1)[0][0]


def _build_cross_module_context(
    completed_modules: List[Dict],
) -> List[Dict]:
    """
    Строит cross_module_context для передачи в generate_chunk.
    completed_modules: [{"title": "...", "concepts": [...all learned_concepts...]}]
    Ограничивает количество концептов на модуль.
    """
    result = []
    for m in completed_modules:
        concepts = m.get("concepts", [])[:_MAX_CONCEPTS_PER_MODULE]
        result.append({"title": m["title"], "concepts": concepts})
    return result


# ─── SEQUENTIAL GENERATION ───────────────────────────────────────────────────

def _generate_chunks_sequentially(
    cluster_groups: List[List[int]],
    lessons: List,
    module_ids: List[int],
    syllabus: Dict[str, Any],
    tags: List[str],
    session_dir: str,
    log_fn=None,
) -> List[Dict[str, Any]]:
    """
    Секвенциальная delta-генерация чанков.

    Алгоритм:
    1. Группируем кластеры по module_id
    2. Сортируем кластеры внутри модуля по средней позиции уроков
    3. Обрабатываем модули в порядке 0, 1, 2, ...
       (нераспределённые module_id=-1 идут последними)
    4. Каждый чанк получает накапливаемый контекст

    Returns: все чанки в порядке генерации
    """
    def _log(msg: str):
        logger.info(msg)
        if log_fn:
            log_fn(msg)

    course_title = syllabus.get("course_title", "")
    # Индекс модулей силлабуса: module_id → module_dict
    syllabus_modules: Dict[int, Dict] = {
        m["id"]: m for m in syllabus.get("modules", [])
    }

    # ── 1. Группируем кластеры по module_id ──────────────────────────
    module_to_groups: Dict[int, List[List[int]]] = defaultdict(list)
    for group in cluster_groups:
        mid = _group_majority_module(group, module_ids)
        module_to_groups[mid].append(group)

    # ── 2. Сортируем кластеры внутри модуля по позиции уроков ────────
    for mid in module_to_groups:
        module_to_groups[mid].sort(key=lambda g: _avg_position(g, lessons))

    # ── 3. Порядок обработки модулей: 0, 1, 2, ..., потом -1 ─────────
    assigned_mids = sorted(k for k in module_to_groups if k >= 0)
    unassigned    = [-1] if -1 in module_to_groups else []
    module_order  = assigned_mids + unassigned

    all_chunks: List[Dict[str, Any]] = []
    # cross_module_context накапливается по мере завершения модулей
    completed_modules: List[Dict] = []

    # ── 4. Основной цикл ──────────────────────────────────────────────
    for mid in module_order:
        module_info     = syllabus_modules.get(mid)
        module_title    = module_info.get("title", f"Модуль {mid}") if module_info else (
            "Нераспределённые уроки" if mid == -1 else f"Модуль {mid}"
        )
        groups_in_module = module_to_groups[mid]

        _log(
            f"\n  {'─'*48}\n"
            f"  Модуль {mid}: «{module_title}» "
            f"({len(groups_in_module)} кластеров)"
        )

        # Контекст предыдущих модулей
        cross_ctx = _build_cross_module_context(completed_modules) if completed_modules else None

        # Концепции, объяснённые в текущем модуле (накапливается)
        module_covered: List[str] = []

        for gi, group_indices in enumerate(groups_in_module):
            cluster_lessons = [lessons[i] for i in group_indices]

            # Решение о слиянии — LLM видит цели модуля
            merge_groups = decide_merges_in_cluster(
                cluster_lessons,
                module_info=module_info,
                course_title=course_title,
            )

            for mg in merge_groups:
                to_merge = [cluster_lessons[j] for j in mg]
                titles   = [l[1] for l in to_merge]

                _log(
                    f"    Чанк {len(all_chunks)+1} "
                    f"[mod={mid}, pos≈{_avg_position(group_indices, lessons):.0f}]: "
                    f"{titles[:2]}{'...' if len(titles) > 2 else ''}"
                )
                if module_covered:
                    _log(f"      ↳ delta: передаю {len(module_covered)} освоенных концептов")
                if cross_ctx:
                    _log(f"      ↳ cross: {len(cross_ctx)} предыдущих модулей")

                chunk = generate_chunk(
                    lessons_to_merge              = to_merge,
                    known_tags                    = tags,
                    session_dir                   = session_dir,
                    module_info                   = module_info,
                    course_title                  = course_title,
                    previously_covered_in_module  = module_covered.copy() or None,
                    cross_module_context          = cross_ctx,
                )
                all_chunks.append(chunk)

                # Накапливаем learned_concepts текущего модуля
                new_concepts = chunk.get("learned_concepts", [])
                module_covered.extend(new_concepts)

        # Завершили модуль — сохраняем его концепции для следующих
        if mid >= 0:
            completed_modules.append({
                "title":    module_title,
                "concepts": module_covered,
            })
            _log(
                f"  ✅ Модуль {mid} завершён. "
                f"Освоено концептов: {len(module_covered)}"
            )

    return all_chunks


# ─── MAIN ────────────────────────────────────────────────────────────────────

def run_stage2(
    session_dir: str,
    user_id: int = 0,
    chat_id: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Полный Stage 2.

    Returns dict:
        lessons_count, clusters_count, chunks_count, indexed_count,
        collection_name, modules_count, docx_path, mongo_id,
        module_chunk_distribution
    """
    def _log(msg: str):
        logger.info(msg)
        if log_fn:
            log_fn(msg)

    raw_data_dir = os.path.join(session_dir, "raw_data")
    session_info = _load_session_info(session_dir)
    session_id   = session_info.get("session_id", os.path.basename(session_dir))
    topic        = session_info.get("topic", "Курс")

    tags     = _load_tags(session_dir)
    syllabus = _load_syllabus(session_dir)

    collection_name = f"rag_{session_id[:20].replace('-', '_')}"

    _log(f"\n{'='*55}")
    _log(f"  STAGE 2  |  Тема: {topic}")
    _log(f"  Силлабус: {len(syllabus.get('modules', []))} модулей")
    _log(f"  Тегов:    {len(tags)}")
    _log(f"{'='*55}")

    # ── 0. Сохраняем силлабус Stage 1 в MongoDB ──────────────────────
    if syllabus:
        _log("\n[0/6] Сохраняю силлабус Stage 1 в MongoDB...")
        try:
            save_syllabus(session_id, syllabus, topic)
            _log(f"      ✅ syllabi/{session_id}")
        except Exception as e:
            _log(f"      ⚠ MongoDB syllabi: {e}")

    # ── 1. Загрузка уроков + векторизация ────────────────────────────
    _log("\n[1/6] Векторизация названий уроков...")
    lessons, embeddings = embed_lesson_names(raw_data_dir, EMBED_MODEL, log_fn=_log)

    if not lessons:
        _log("❌ Уроки не найдены в raw_data/")
        return {"error": "no lessons"}

    _log(f"      ✅ {len(lessons)} уроков")

    # ── 2. Кластеризация ──────────────────────────────────────────────
    _log("\n[2/6] Кластеризация...")

    import numpy as np
    module_ids     = _module_ids_from_lessons(lessons)
    has_module_ids = any(mid >= 0 for mid in module_ids)

    if has_module_ids and syllabus and len(embeddings) > 0:
        _log("      Режим: двухуровневый (module_id + embeddings)")
        cluster_groups = cluster_by_module_and_embeddings(
            lessons, embeddings, module_ids, log_fn=_log
        )
    elif len(embeddings) > 0:
        _log("      Режим: глобальный ансамбль (нет module_id)")
        labels         = cluster_embeddings(embeddings, log_fn=_log)
        cluster_groups = labels_to_groups(len(lessons), labels)
    else:
        _log("      Режим: каждый урок в своём кластере")
        cluster_groups = [[i] for i in range(len(lessons))]

    _log(f"      ✅ {len(cluster_groups)} кластеров")

    # ── 3. Секвенциальная delta-генерация чанков ─────────────────────
    _log("\n[3/6] Генерация чанков (секвенциальная delta-генерация)...")
    all_chunks = _generate_chunks_sequentially(
        cluster_groups=cluster_groups,
        lessons=lessons,
        module_ids=module_ids,
        syllabus=syllabus,
        tags=tags,
        session_dir=session_dir,
        log_fn=_log,
    )

    _log(f"\n      ✅ Создано чанков: {len(all_chunks)}")

    # Статистика по модулям
    module_chunk_counts: Dict[int, int] = {}
    for chunk in all_chunks:
        mid = chunk.get("module_id", -1)
        module_chunk_counts[mid] = module_chunk_counts.get(mid, 0) + 1
    for mid, cnt in sorted(module_chunk_counts.items()):
        label = f"Модуль {mid}" if mid >= 0 else "Нераспределённые"
        _log(f"      {label}: {cnt} чанков")

    _save_chunks(all_chunks, session_dir)

    # ── 4. Индексация в Qdrant ────────────────────────────────────────
    _log(f"\n[4/6] Индексация в Qdrant ({collection_name})...")
    indexed_count = index_all_chunks(
        chunks=all_chunks,
        collection_name=collection_name,
        embed_model=EMBED_MODEL,
        log_fn=_log,
    )
    _log(f"      ✅ Проиндексировано: {indexed_count}/{len(all_chunks)}")

    # ── 5. Структура курса (T-lite) ───────────────────────────────────
    _log("\n[5/6] Генерация структуры курса (T-lite)...")
    course_structure = generate_course_structure(
        topic=topic,
        model_name=COURSE_MODEL,
        chunks=all_chunks,
        tag_map=tags,
        log_fn=_log,
    )
    _save_stage2_course_structure(course_structure, session_dir)
    modules_count = len(course_structure.get("modules", []))
    _log(f"      ✅ {modules_count} модулей")

    # ── 6. DOCX ───────────────────────────────────────────────────────
    _log("\n[6/6] Генерация DOCX...")
    docx_path = ""
    try:
        docx_out  = os.path.join(session_dir, "final", "course.docx")
        docx_path = generate_course_docx(
            course_structure=course_structure,
            chunks=all_chunks,
            output_path=docx_out,
            topic=topic,
        )
        _log(f"      ✅ {docx_path}")
    except Exception as e:
        _log(f"      ⚠ DOCX: {e}")

    # ── 7. БД ─────────────────────────────────────────────────────────
    _log("\n[DB] MongoDB + PostgreSQL...")
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
        _log(f"      ⚠ DB: {e}")

    result = {
        "session_id":               session_id,
        "topic":                    topic,
        "lessons_count":            len(lessons),
        "clusters_count":           len(cluster_groups),
        "chunks_count":             len(all_chunks),
        "indexed_count":            indexed_count,
        "collection_name":          collection_name,
        "modules_count":            modules_count,
        "docx_path":                docx_path,
        "mongo_id":                 mongo_id,
        "module_chunk_distribution": module_chunk_counts,
    }

    _log(f"\n{'='*55}")
    _log(f"  ✅ STAGE 2 ЗАВЕРШЁН")
    _log(f"  Уроков: {len(lessons)} → Чанков: {len(all_chunks)}")
    _log(f"  Индексировано в Qdrant: {indexed_count}")
    _log(f"  Коллекция: {collection_name}")
    _log(f"  Документ:  {docx_path}")
    _log(f"{'='*55}\n")

    return result