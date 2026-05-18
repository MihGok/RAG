"""
KnowledgeBaseCreator/merger.py
───────────────────────────────
Объединение уроков в чанки.

Изменения:
  decide_merges_in_cluster() — теперь получает module_info и course_title,
    передаёт их в промпт для осознанного решения о слиянии.

  generate_chunk() — принимает delta-контекст:
    module_info                    — цели и темы текущего модуля
    previously_covered_in_module   — из learned_concepts предыдущих чанков
    cross_module_context           — сводка по пройденным модулям
  Добавляет module_id и learned_concepts в результат.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import AppConfig
from LLMprompts import PromptBank

logger    = logging.getLogger(__name__)
TASK_URL  = f"{AppConfig.ML_SERVER_URL.rstrip('/')}/task"
MAIN_MODEL = "Qwen2.5-3B-Instruct-Q4_K_L.gguf"
MAX_TEXT_CHARS = 6000


# ─── LLM HELPER ──────────────────────────────────────────────────────────────

def _call_qwen(
    prompt: str,
    schema_name: str,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    n_ctx: int = 4096,
) -> dict:
    payload = {
        "task_type":       "llm",
        "model_name":      MAIN_MODEL,
        "text":            prompt,
        "schema_name":     schema_name,
        "max_tokens":      max_tokens,
        "temperature":     temperature,
        "top_p":           0.9,
        "n_ctx":           n_ctx,
        "enable_thinking": False,
    }
    try:
        resp = requests.post(TASK_URL, json=payload, timeout=300)
        resp.raise_for_status()
        result = resp.json().get("result", {})
        return _parse(result)
    except Exception as e:
        logger.error("Qwen error: %s", e)
        return {}


def _parse(result: Any) -> dict:
    if isinstance(result, dict) and "response" in result and len(result) == 1:
        m = re.search(r"\{.*\}", result["response"], re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return result if isinstance(result, dict) else {}


# ─── MERGE DECISION ──────────────────────────────────────────────────────────

def decide_merges_in_cluster(
    cluster_lessons: List[Tuple[str, str, Dict]],
    module_info: Optional[Dict] = None,
    course_title: str = "",
) -> List[List[int]]:
    """
    Определяет, какие уроки внутри кластера объединить в один чанк.

    Args:
        cluster_lessons: список (filename, lesson_name, data_dict)
        module_info:     описание текущего модуля из Stage 1 силлабуса
                         (title, description, goals, key_topics)
        course_title:    название курса для контекста
    """
    n = len(cluster_lessons)
    if n == 1:
        return [[0]]

    titles = [l[1] for l in cluster_lessons]
    prompt = PromptBank.cluster_merge_decision(
        lesson_titles=titles,
        module_info=module_info,
        course_title=course_title,
    )
    result = _call_qwen(prompt, "cluster_merge_decision", max_tokens=768, n_ctx=3072)

    raw_groups = result.get("groups", [])
    if not raw_groups:
        return [list(range(n))]

    groups: List[List[int]] = []
    seen: set = set()
    for g in raw_groups:
        indices = [int(i) for i in g.get("indices", []) if 0 <= int(i) < n]
        valid   = [i for i in indices if i not in seen]
        if valid:
            groups.append(valid)
            seen.update(valid)

    for i in range(n):
        if i not in seen:
            groups.append([i])

    return groups


# ─── CHUNK GENERATION ────────────────────────────────────────────────────────

def _build_combined_text(lessons: List[Tuple[str, str, Dict]]) -> str:
    parts = []
    for _, name, data in lessons:
        lesson_parts = []
        if data.get("text"):
            lesson_parts.append(data["text"])
        if data.get("transcript"):
            lesson_parts.append(f"[Транскрипция]\n{data['transcript']}")
        if lesson_parts:
            parts.append(f"=== {name} ===\n" + "\n\n".join(lesson_parts))
    combined = "\n\n".join(parts)
    if len(combined) > MAX_TEXT_CHARS:
        combined = combined[:MAX_TEXT_CHARS] + "\n...[сокращено]"
    return combined


def _save_tags(tags: List[str], session_dir: str) -> None:
    path = os.path.join(session_dir, "tags.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tags, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("Не удалось сохранить tags.json: %s", e)


def _majority_module_id(lessons: List[Tuple[str, str, Dict]]) -> int:
    ids = [int(d.get("module_id", -1)) for _, _, d in lessons]
    assigned = [mid for mid in ids if mid >= 0]
    if assigned:
        return Counter(assigned).most_common(1)[0][0]
    return -1


def generate_chunk(
    lessons_to_merge: List[Tuple[str, str, Dict]],
    known_tags: List[str],
    session_dir: str,
    # ── Delta-контекст ───────────────────────────────────────────────────
    module_info: Optional[Dict] = None,
    course_title: str = "",
    previously_covered_in_module: Optional[List[str]] = None,
    cross_module_context: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Генерирует чанк знаний.

    Args:
        lessons_to_merge:             уроки для объединения
        known_tags:                   текущий список тегов проекта
        session_dir:                  папка сессии (для сохранения обновлённых тегов)
        module_info:                  описание текущего модуля (title, goals, key_topics)
        course_title:                 название курса
        previously_covered_in_module: learned_concepts предыдущих чанков модуля
        cross_module_context:         [{title, concepts}] для предыдущих модулей

    Returns:
        dict с полями по схеме chunk_generation + module_id
    """
    titles   = [l[1] for l in lessons_to_merge]
    combined = _build_combined_text(lessons_to_merge)

    # Для чанков с контекстом нужно больше токенов и контекстного окна
    has_context = bool(previously_covered_in_module or cross_module_context)
    n_ctx       = 6144 if has_context else 4096
    max_tokens  = 2560 if has_context else 2048

    prompt = PromptBank.chunk_generation(
        titles=titles,
        known_tags=known_tags,
        combined_text=combined,
        module_info=module_info,
        course_title=course_title,
        previously_covered_in_module=previously_covered_in_module,
        cross_module_context=cross_module_context,
    )
    result = _call_qwen(
        prompt,
        "chunk_generation",
        max_tokens=max_tokens,
        temperature=0.3,
        n_ctx=n_ctx,
    )

    if not result.get("final_title"):
        logger.warning("Qwen не вернула чанк для %s — fallback", titles)
        result = {
            "final_title":      titles[0] if titles else "Урок",
            "summary":          f"Учебный материал: {', '.join(titles)}",
            "tags":             known_tags[:15],
            "merged_text":      combined[:7000],
            "learned_concepts": [],
            "assumed_knowledge": [],
        }

    # Обеспечиваем наличие delta-полей
    result.setdefault("learned_concepts", [])
    result.setdefault("assumed_knowledge", [])

    # Расширяем список тегов проекта
    known_lower = {t.lower().strip() for t in known_tags}
    new_tags: List[str] = []
    for tag in result.get("tags", []):
        tag_l = tag.lower().strip()
        if tag_l and tag_l not in known_lower and len(tag.split()) <= 4:
            new_tags.append(tag)
            known_lower.add(tag_l)
    if new_tags:
        known_tags.extend(new_tags)
        _save_tags(known_tags, session_dir)
        logger.info("Теги расширены: %s", new_tags)

    # Метаданные источников
    result["source_lesson_ids"] = [d.get("lesson_id") for _, _, d in lessons_to_merge]
    result["source_course_ids"] = list({d.get("course_id") for _, _, d in lessons_to_merge})
    result["source_filenames"]  = [l[0] for l in lessons_to_merge]
    result["module_id"]         = _majority_module_id(lessons_to_merge)

    return result