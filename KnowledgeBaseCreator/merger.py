"""
stage2/merger.py
────────────────
Объединение уроков в чанки через локальную Qwen.
Теги — плоский список строк (не словарь).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Tuple

import requests

from config import AppConfig
from LLMprompts import PromptBank

logger    = logging.getLogger(__name__)
TASK_URL  = f"{AppConfig.ML_SERVER_URL.rstrip('/')}/task"
MAIN_MODEL = "Qwen2.5-3B-Instruct-Q4_K_L.gguf"
MAX_TEXT_CHARS = 6000


# ─── LLM HELPER ──────────────────────────────────────────────────────────────

def _call_qwen(prompt: str, schema_name: str,
               max_tokens: int = 2048, temperature: float = 0.2) -> dict:
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
) -> List[List[int]]:
    n = len(cluster_lessons)
    if n == 1:
        return [[0]]

    titles = [l[1] for l in cluster_lessons]
    prompt = PromptBank.cluster_merge_decision(titles)
    result = _call_qwen(prompt, "cluster_merge_decision", max_tokens=512)

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


def _save_tags(tags: List[str], session_dir: str):
    path = os.path.join(session_dir, "tags.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tags, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("Не удалось сохранить tags.json: %s", e)


def generate_chunk(
    lessons_to_merge: List[Tuple[str, str, Dict]],
    known_tags: List[str],           # плоский список
    session_dir: str,
) -> Dict[str, Any]:
    """
    Генерирует чанк знаний через Qwen.
    Расширяет known_tags (плоский список) новыми тегами и сохраняет.
    """
    titles   = [l[1] for l in lessons_to_merge]
    combined = _build_combined_text(lessons_to_merge)

    prompt = PromptBank.chunk_generation(
        titles=titles,
        known_tags=known_tags,
        combined_text=combined,
    )
    result = _call_qwen(prompt, "chunk_generation", max_tokens=2048, temperature=0.3)

    if not result.get("final_title"):
        logger.warning("Qwen не вернула чанк для %s — fallback", titles)
        result = {
            "final_title": titles[0] if titles else "Урок",
            "summary":     f"Учебный материал: {', '.join(titles)}",
            "tags":        known_tags[:5],
            "merged_text": combined[:3000],
        }

    # Расширяем список тегов новыми
    known_lower = {t.lower().strip() for t in known_tags}
    new_tags    = []
    for tag in result.get("tags", []):
        tag_l = tag.lower().strip()
        if tag_l and tag_l not in known_lower:
            # Принимаем только конкретные термины (≤ 4 слова)
            if len(tag.split()) <= 4:
                new_tags.append(tag)
                known_lower.add(tag_l)

    if new_tags:
        known_tags.extend(new_tags)
        _save_tags(known_tags, session_dir)
        logger.info("Теги расширены: %s", new_tags)

    result["source_lesson_ids"] = [d.get("lesson_id") for _, _, d in lessons_to_merge]
    result["source_course_ids"] = list({d.get("course_id") for _, _, d in lessons_to_merge})
    result["source_filenames"]  = [l[0] for l in lessons_to_merge]

    return result
