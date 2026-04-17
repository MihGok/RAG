"""
stage2/course_generator.py
──────────────────────────
Генерирует структуру курса через T-lite-it-1.0-Q4_K_M.gguf
и сохраняет результат в MongoDB + PostgreSQL.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Dict, Any, Optional

import requests

from config import AppConfig
from db.mongo_service import save_course_structure
from db.postgres_service import save_course_with_sections

logger = logging.getLogger(__name__)

TASK_URL = f"{AppConfig.ML_SERVER_URL.rstrip('/')}/task"
COURSE_STRUCTURE_MODEL = "T-lite-it-1.0-Q4_K_M.gguf"


# ─── LLM CALL ────────────────────────────────────────────────────────────────

def _call_llm(prompt: str, schema_name: str, model_name: str, n_ctx: int = 8192) -> dict:
    payload = {
        "task_type":   "llm",
        "model_name":  model_name,
        "text":        prompt,
        "schema_name": schema_name,
        "max_tokens":  4096,
        "temperature": 0.4,
        "top_p":       0.9,
        "n_ctx":       n_ctx,
    }
    try:
        resp = requests.post(TASK_URL, json=payload, timeout=360)
        resp.raise_for_status()
        result = resp.json().get("result", {})

        # Распарсить JSON из текстового ответа если нужно
        if isinstance(result, dict) and "response" in result and len(result) == 1:
            match = re.search(r"\{.*\}", result["response"], re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return result if isinstance(result, dict) else {}
    except Exception as e:
        logger.error("CourseGen LLM error: %s", e)
        return {}


# ─── GENERATE ────────────────────────────────────────────────────────────────

def generate_course_structure(
    topic: str,
    chunks: List[Dict[str, Any]],
    tag_map: Dict[str, List[str]],
    model_name: str = COURSE_STRUCTURE_MODEL,
    log_fn=None,
) -> Dict[str, Any]:
    """
    Генерирует структуру курса на основе чанков и карты тегов.

    Структура:
    {
      "course_title": "...",
      "modules": [
        {
          "title": "...",
          "steps": [
            {
              "title": "...",
              "query_texts": ["...", "...", "..."],
              "tags": ["...", ...]
            }, ...
          ]
        }, ...
      ]
    }
    """
    def _log(msg: str):
        logger.info(msg)
        if log_fn:
            log_fn(msg)

    # Краткая сводка чанков для подачи в LLM
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        chunk_summaries.append({
            "id":    i,
            "title": chunk.get("final_title", ""),
            "tags":  chunk.get("tags", [])[:8],
            "brief": chunk.get("summary", "")[:150],
        })

    # Карта тегов: первые 60 тегов
    all_tags: list[str] = []
    for tlist in tag_map.values():
        all_tags.extend(tlist)
    tag_sample = sorted(set(all_tags))[:60]

    prompt = (
        f"Создай структуру образовательного курса по теме: \"{topic}\"\n\n"
        f"Доступные учебные материалы (чанки знаний):\n"
        f"{json.dumps(chunk_summaries, ensure_ascii=False, indent=2)}\n\n"
        f"Карта тегов (обязательно используй при тегировании шагов):\n"
        f"{', '.join(tag_sample)}\n\n"
        f"Требования к структуре:\n"
        f"- 3–6 модулей, логически выстроенных от простого к сложному\n"
        f"- В каждом модуле 3–7 шагов\n"
        f"- Для каждого шага — ровно 3 поисковых запроса к базе знаний\n"
        f"  (query_texts: конкретные формулировки для семантического поиска)\n"
        f"- Для каждого шага — список тегов из карты тегов\n"
        f"- Верни только JSON по схеме course_structure"
    )

    _log("[CourseGen] Генерирую структуру курса через LLM...")
    result = _call_llm(prompt, "course_structure", model_name)

    if not result.get("course_title") or not result.get("modules"):
        _log("[CourseGen] LLM не вернула структуру — создаю базовую")
        result = _build_fallback_structure(topic, chunks)

    _log(
        f"[CourseGen] Структура: {len(result.get('modules', []))} модулей, "
        f"{sum(len(m.get('steps', [])) for m in result.get('modules', []))} шагов"
    )
    return result


def _build_fallback_structure(topic: str, chunks: List[Dict]) -> Dict:
    """Базовая структура курса если LLM недоступна."""
    steps = [
        {
            "title":       chunk.get("final_title", f"Шаг {i+1}"),
            "query_texts": [
                chunk.get("final_title", ""),
                chunk.get("summary", "")[:80],
                " ".join(chunk.get("tags", [])[:4]),
            ],
            "tags":        chunk.get("tags", [])[:5],
        }
        for i, chunk in enumerate(chunks)
    ]

    # Разбить на модули по ~5 шагов
    modules = []
    for i in range(0, len(steps), 5):
        modules.append({
            "title": f"Модуль {len(modules)+1}",
            "steps": steps[i:i+5],
        })

    return {"course_title": f"Курс: {topic}", "modules": modules}


# ─── SAVE ────────────────────────────────────────────────────────────────────

def save_course(
    course_structure: Dict[str, Any],
    session_id: str,
    topic: str,
    chunks_count: int,
    user_id: int,
    chat_id: Optional[int] = None,
    log_fn=None,
) -> str:
    """
    Сохраняет структуру курса в MongoDB и PostgreSQL.

    Returns:
        mongo_doc_id — ID документа в MongoDB
    """
    def _log(msg: str):
        logger.info(msg)
        if log_fn:
            log_fn(msg)

    mongo_id = ""
    try:
        mongo_id = save_course_structure(
            course_structure=course_structure,
            session_id=session_id,
            topic=topic,
            chunks_count=chunks_count,
        )
        _log(f"[MongoDB] Курс сохранён: {mongo_id}")
    except Exception as e:
        _log(f"[MongoDB] Ошибка сохранения: {e}")

    try:
        pg_row = save_course_with_sections(
            user_id=user_id,
            course_structure=course_structure,
            mongo_doc_id=mongo_id,
            chat_id=chat_id,
        )
        _log(f"[PostgreSQL] Курс сохранён: course_id={pg_row.get('course_id')}")
    except Exception as e:
        _log(f"[PostgreSQL] Ошибка сохранения: {e}")

    return mongo_id
