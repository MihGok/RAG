"""
loading_workflow.py
────────────────────
Stage 1: поиск, фильтрация и скачивание курсов со Stepik.
Все промпты → LLMprompts.py. Теги — плоский список строк.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from config import AppConfig
from CourseProcessor.CourseLoader import StepikCourseLoader
from LLMprompts import PromptBank

SESSIONS_DIR           = "sessions"
TASK_URL               = f"{AppConfig.ML_SERVER_URL.rstrip('/')}/task"
MAIN_MODEL             = "Qwen2.5-3B-Instruct-Q4_K_L.gguf"
DEFAULT_MAX_COURSES    = 5
DEFAULT_LIMIT_PER_QUERY = 30


# ════════════════════════════════════════════════════════════════════════════
#  LLM HELPER
# ════════════════════════════════════════════════════════════════════════════

def _call_llm(
    prompt: str,
    schema_name: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    model: str = MAIN_MODEL,
) -> dict:
    payload = {
        "task_type":   "llm",
        "model_name":  model,
        "text":        prompt,
        "schema_name": schema_name,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "top_p":       0.9,
        "n_ctx":       4096,
    }
    try:
        resp = requests.post(TASK_URL, json=payload, timeout=180)
        resp.raise_for_status()
        result = resp.json().get("result", {})
        if isinstance(result, dict) and "response" in result and len(result) == 1:
            import re
            m = re.search(r"\{.*\}", result["response"], re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        return result if isinstance(result, dict) else {}
    except requests.exceptions.ConnectionError:
        print(f"[LLM] Нет соединения с ML backend: {TASK_URL}")
        return {}
    except Exception as e:
        print(f"[LLM] {type(e).__name__}: {e}")
        return {}


# ════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — LLM задачи
# ════════════════════════════════════════════════════════════════════════════

def generate_clarifying_questions(topic: str) -> List[str]:
    result = _call_llm(PromptBank.clarifying_questions(topic), "clarifying_questions")
    questions = result.get("questions", [])
    if not questions:
        questions = [
            "Какой уровень подготовки у целевой аудитории?",
            "Какие конкретные инструменты или фреймворки важно охватить?",
            "Какова главная практическая цель обучения?",
            "Какие смежные темы стоит включить?",
        ]
    return questions


def generate_pipeline_setup(topic: str, user_answers: str) -> Dict[str, Any]:
    """
    Возвращает:
        search_queries: List[str]  — 3 запроса для Stepik
        tags: List[str]            — плоский список тегов
    """
    prompt = PromptBank.pipeline_setup(topic, user_answers)
    result = _call_llm(prompt, "pipeline_setup", temperature=0.4, max_tokens=2048)

    if not result.get("search_queries"):
        result["search_queries"] = [topic, f"{topic} курс", f"{topic} практика"]
    if not result.get("tags"):
        result["tags"] = [topic]
    return result


def rank_courses_with_llm(
    topic: str,
    user_context: str,
    courses: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not courses:
        return []

    def _summary(c: Dict) -> Dict:
        return {
            "id":             c.get("id"),
            "title":          c.get("title", ""),
            "is_popular":     c.get("is_popular", False),
            "sections_count": len(c.get("sections", [])),
            "learners_count": c.get("learners_count", 0),
        }

    rankings_map: Dict[int, Dict] = {}
    BATCH = 20

    for i in range(0, len(courses), BATCH):
        chunk     = courses[i : i + BATCH]
        summaries = [_summary(c) for c in chunk]
        prompt    = PromptBank.course_ranking(topic, user_context, summaries)
        result    = _call_llm(prompt, "course_ranking", temperature=0.3)
        for r in result.get("rankings", []):
            cid = r.get("id")
            if cid:
                rankings_map[cid] = r
        if i + BATCH < len(courses):
            time.sleep(0.5)

    for course in courses:
        info = rankings_map.get(course.get("id"), {})
        course["_score"]       = info.get("score", 0)
        course["_rank_reason"] = info.get("reason", "")

    return sorted(courses, key=lambda x: x.get("_score", 0), reverse=True)


# ════════════════════════════════════════════════════════════════════════════
#  SESSION (flat tags)
# ════════════════════════════════════════════════════════════════════════════

def create_session(
    topic: str,
    tags: List[str],
    user_context: str = "",
) -> Tuple[str, str]:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid  = str(uuid.uuid4())[:8]
    session_id  = f"{ts}_{short_uuid}"
    session_dir = os.path.join(SESSIONS_DIR, session_id)

    os.makedirs(os.path.join(session_dir, "raw_data"), exist_ok=True)
    os.makedirs(os.path.join(session_dir, "final"), exist_ok=True)

    with open(os.path.join(session_dir, "session_info.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"session_id": session_id, "topic": topic,
             "user_context": user_context, "created_at": datetime.now().isoformat()},
            f, ensure_ascii=False, indent=2,
        )
    # Сохраняем плоский список тегов
    with open(os.path.join(session_dir, "tags.json"), "w", encoding="utf-8") as f:
        json.dump(tags, f, ensure_ascii=False, indent=2)

    return session_id, session_dir


def load_tags(session_dir: str) -> List[str]:
    """Загружает плоский список тегов сессии."""
    path = os.path.join(session_dir, "tags.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Обратная совместимость: старый tag_map
    old = os.path.join(session_dir, "tag_map.json")
    if os.path.exists(old):
        with open(old, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        # Из словаря — разворачиваем в плоский список
        tags: List[str] = []
        for v in data.values():
            if isinstance(v, list):
                tags.extend(v)
        return tags
    return []


# ════════════════════════════════════════════════════════════════════════════
#  STEPIK SEARCH
# ════════════════════════════════════════════════════════════════════════════

def search_all_queries(
    loader: StepikCourseLoader,
    queries: List[str],
    limit_per_query: int = DEFAULT_LIMIT_PER_QUERY,
) -> List[Dict[str, Any]]:
    all_ids: List[int] = []
    seen: set = set()
    for query in queries:
        ids = loader.get_course_ids_by_query(query, limit=limit_per_query)
        for cid in ids:
            if cid not in seen:
                all_ids.append(cid)
                seen.add(cid)
    if not all_ids:
        return []
    courses = loader.fetch_objects("courses", all_ids)
    return [c for c in courses if c.get("is_public") and not c.get("is_paid")]


# ════════════════════════════════════════════════════════════════════════════
#  DOWNLOAD
# ════════════════════════════════════════════════════════════════════════════

def download_courses(
    loader: StepikCourseLoader,
    courses: List[Dict[str, Any]],
    session_dir: str,
    max_courses: int = DEFAULT_MAX_COURSES,
    transcribe: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
):
    def _log(msg: str):
        print(msg)
        if log_fn:
            log_fn(msg)

    raw_data_dir = os.path.join(session_dir, "raw_data")
    for i, course in enumerate(courses[:max_courses], 1):
        cid   = course.get("id")
        title = course.get("title", f"course_{cid}")
        score = course.get("_score", "?")
        _log(f"[{i}/{min(max_courses, len(courses))}] {title}  (score:{score})")
        try:
            loader.process_course_to_session(course, raw_data_dir, transcribe=transcribe)
        except Exception as e:
            _log(f"  ❌ {e}")


# ════════════════════════════════════════════════════════════════════════════
#  FULL STAGE 1 (вызывается из front.py)
# ════════════════════════════════════════════════════════════════════════════

def run_stage1(
    topic: str,
    user_answers: str,
    max_courses: int = DEFAULT_MAX_COURSES,
    limit_per_query: int = DEFAULT_LIMIT_PER_QUERY,
    transcribe: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    def _log(msg: str):
        print(msg)
        if log_fn:
            log_fn(msg)

    _log("🔧 Строю план поиска и карту тегов...")
    setup   = generate_pipeline_setup(topic, user_answers)
    queries = setup.get("search_queries", [topic])
    tags    = setup.get("tags", [topic])

    _log(f"📋 Запросы: {' | '.join(queries)}")
    _log(f"🏷️  Тегов: {len(tags)}")

    user_context = f"Тема: {topic}\nОтветы: {user_answers}"
    session_id, session_dir = create_session(topic, tags, user_context)
    _log(f"📁 Сессия: {session_id}")

    _log("🔍 Авторизация Stepik...")
    try:
        loader = StepikCourseLoader()
    except Exception as e:
        _log(f"❌ Stepik ошибка: {e}")
        return None

    _log("🔍 Поиск курсов...")
    courses = search_all_queries(loader, queries, limit_per_query=limit_per_query)
    _log(f"📚 Найдено курсов: {len(courses)}")

    if not courses:
        _log("⚠️ Курсы не найдены")
        return session_dir

    _log("🤖 Ранжирование...")
    ranked = rank_courses_with_llm(topic, user_context, courses)
    top3   = " | ".join(c.get("title", "")[:40] for c in ranked[:3])
    _log(f"🏆 Топ-3: {top3}")

    _log(f"⬇️  Скачиваю топ-{max_courses} курсов...")
    download_courses(
        loader, ranked, session_dir,
        max_courses=max_courses, transcribe=transcribe, log_fn=_log,
    )

    saved = len([f for f in os.listdir(os.path.join(session_dir, "raw_data"))
                 if f.endswith(".json")])
    _log(f"✅ Загружено уроков: {saved}")
    return session_dir


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    max_courses: int = DEFAULT_MAX_COURSES,
    transcribe: bool = True,
    limit_per_query: int = DEFAULT_LIMIT_PER_QUERY,
) -> Optional[str]:
    print("\n" + "=" * 60)
    topic = input("Введите тему: ").strip()
    if not topic:
        return None

    questions = generate_clarifying_questions(topic)
    print("\n[AI] Вопросы:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")

    print("\nОтветьте (пустая строка = завершить):")
    lines = []
    while True:
        line = input("> ")
        if line == "" and lines:
            break
        if line:
            lines.append(line)

    return run_stage1(
        topic=topic, user_answers="\n".join(lines),
        max_courses=max_courses, limit_per_query=limit_per_query,
        transcribe=transcribe,
    )
