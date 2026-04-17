"""
stage2/merger.py
────────────────
1. LLM решает, какие уроки внутри кластера объединять.
2. Для каждой группы уроков генерируется чанк через локальную Qwen:
   final_title, summary, tags, merged_text.
3. Автоматическое расширение карты тегов новыми тегами.
"""

from __future__ import annotations

import json
import os
import re
import logging
from typing import List, Tuple, Dict, Any

import requests

from config import AppConfig

logger = logging.getLogger(__name__)

TASK_URL       = f"{AppConfig.ML_SERVER_URL.rstrip('/')}/task"
MAIN_MODEL     = "Qwen2.5-3B-Instruct-Q4_K_L.gguf"
MAX_TEXT_CHARS = 6000


# ─── LLM HELPER ──────────────────────────────────────────────────────────────

def _call_qwen(
    prompt: str,
    schema_name: str,
    max_tokens: int = 2048,
    temperature: float = 0.2,
) -> dict:
    """Вызывает локальную Qwen через /task endpoint."""
    payload = {
        "task_type":   "llm",
        "model_name":  MAIN_MODEL,
        "text":        prompt,
        "schema_name": schema_name,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "top_p":       0.9,
        "n_ctx":       4096,
    }
    try:
        resp = requests.post(TASK_URL, json=payload, timeout=240)
        resp.raise_for_status()
        result = resp.json().get("result", {})
        return _parse_result(result)
    except Exception as e:
        logger.error("Qwen error: %s", e)
        return {}


def _parse_result(result: Any) -> dict:
    if isinstance(result, dict) and "response" in result and len(result) == 1:
        match = re.search(r"\{.*\}", result["response"], re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return result if isinstance(result, dict) else {}


# ─── MERGE DECISION ──────────────────────────────────────────────────────────

def decide_merges_in_cluster(
    cluster_lessons: List[Tuple[str, str, Dict]],
) -> List[List[int]]:
    """
    LLM решает, какие уроки внутри кластера объединять.
    Возвращает группы индексов.
    """
    n = len(cluster_lessons)
    if n == 1:
        return [[0]]

    titles   = [l[1] for l in cluster_lessons]
    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(titles))

    prompt = (
        f"Список уроков одного тематического кластера:\n{numbered}\n\n"
        "Определи, какие уроки стоит объединить в один блок знаний:\n"
        "- Уроки с практически одинаковой темой → объединяй\n"
        "- Уроки с разными аспектами → оставляй отдельно\n"
        "Каждый индекс (0..N-1) должен быть ровно в одной группе.\n"
        "Верни JSON по схеме cluster_merge_decision."
    )

    result      = _call_qwen(prompt, "cluster_merge_decision", max_tokens=512)
    raw_groups  = result.get("groups", [])

    if not raw_groups:
        return [list(range(n))]

    groups: List[List[int]] = []
    seen: set[int] = set()

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


def generate_chunk(
    lessons_to_merge: List[Tuple[str, str, Dict]],
    tag_map: Dict[str, List[str]],
    session_dir: str,
) -> Dict[str, Any]:
    """
    Генерирует чанк через локальную Qwen:
    final_title, summary, tags, merged_text.
    Расширяет tag_map новыми тегами.
    """
    titles = [l[1] for l in lessons_to_merge]

    known_tags: set[str] = set()
    for tag_list in tag_map.values():
        known_tags.update(t.lower().strip() for t in tag_list)
    known_sample = sorted(known_tags)[:60]

    combined = _build_combined_text(lessons_to_merge)

    prompt = (
        f"Создай структурированный блок знаний из следующих учебных уроков:\n"
        f"Названия: {', '.join(titles)}\n\n"
        f"Предпочтительные теги (используй из этого списка):\n"
        f"{', '.join(known_sample)}\n\n"
        f"Текст уроков:\n{combined}\n\n"
        f"Требования:\n"
        f"1. final_title — краткое итоговое название блока\n"
        f"2. summary — 3-5 предложений: ключевые понятия и взаимосвязи "
        f"   (пример: 'Рассматривается взаимосвязь accuracy и F1-score...')\n"
        f"3. tags — 5-15 конкретных тегов (приоритет тегам из списка выше)\n"
        f"4. merged_text — глубокий структурированный конспект, "
        f"   максимально информативный для последующего поиска"
    )

    result = _call_qwen(prompt, "chunk_generation", max_tokens=2048, temperature=0.3)

    if not result.get("final_title"):
        logger.warning("Qwen не вернула чанк для %s — fallback", titles)
        result = {
            "final_title": titles[0] if titles else "Урок",
            "summary":     f"Учебный материал: {', '.join(titles)}",
            "tags":        list(known_sample[:5]),
            "merged_text": combined[:3000],
        }

    # Расширяем tag_map новыми тегами
    new_tags = []
    for tag in result.get("tags", []):
        if tag.lower().strip() not in known_tags:
            new_tags.append(tag)
            known_tags.add(tag.lower().strip())

    if new_tags:
        tag_map.setdefault("auto_extended", []).extend(new_tags)
        _save_tag_map(tag_map, session_dir)
        logger.info("TagMap расширена: %s", new_tags)

    result["source_lesson_ids"] = [d.get("lesson_id") for _, _, d in lessons_to_merge]
    result["source_course_ids"] = list({d.get("course_id") for _, _, d in lessons_to_merge})
    result["source_filenames"]  = [l[0] for l in lessons_to_merge]

    return result


def _save_tag_map(tag_map: dict, session_dir: str):
    path = os.path.join(session_dir, "tag_map.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tag_map, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("Не удалось сохранить tag_map: %s", e)