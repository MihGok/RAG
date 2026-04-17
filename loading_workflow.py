"""
loading_workflow.py
────────────────────
Пайплайн Stage 1: поиск, фильтрация и скачивание курсов со Stepik.
Все LLM-вызовы → локальная модель (Qwen) через /task endpoint.
"""

from __future__ import annotations
from LLMprompts import PromptBank
import os
import json
import time
import uuid
import requests
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Callable

from CourseProcessor.CourseLoader import StepikCourseLoader
from config import AppConfig

# ─── Константы ──────────────────────────────────────────────────────────────

SESSIONS_DIR         = "sessions"
TASK_URL             = f"{AppConfig.ML_SERVER_URL.rstrip('/')}/task"
MAIN_MODEL           = "Qwen2.5-3B-Instruct-Q4_K_L.gguf"

DEFAULT_MAX_COURSES      = 5
DEFAULT_LIMIT_PER_QUERY  = 30


# ════════════════════════════════════════════════════════════════════════════
#  LLM HELPER — единственная точка вызова локальной LLM
# ════════════════════════════════════════════════════════════════════════════

def _call_llm(
    prompt: str,
    schema_name: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    model: str = MAIN_MODEL,
) -> dict:
    """
    Вызывает локальный /task (task_type=llm).
    Возвращает распарсенный dict или {} при ошибке.
    """
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

        # Если модель вернула текст с JSON внутри — извлекаем
        if isinstance(result, dict) and "response" in result and len(result) == 1:
            import re
            match = re.search(r"\{.*\}", result["response"], re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return result if isinstance(result, dict) else {}

    except requests.exceptions.ConnectionError:
        print(f"[LLM] Не могу подключиться к ML backend: {TASK_URL}")
        return {}
    except Exception as e:
        print(f"[LLM] Ошибка: {type(e).__name__}: {e}")
        return {}


# ════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — LLM задачи
# ════════════════════════════════════════════════════════════════════════════

def generate_clarifying_questions(topic: str) -> List[str]:
    """Генерирует 3–5 уточняющих вопросов по теме."""
    prompt = PromptBank.clarifying_questions(topic)
    result = _call_llm(prompt, "clarifying_questions")
    questions = result.get("questions", [])
    if not questions:
        questions = [
            "Какой уровень подготовки у целевой аудитории?",
            "Какие конкретные темы и инструменты важно охватить?",
            "Какова главная цель обучения?",
            "Какой формат предпочтителен — теория, практика или смешанный?",
        ]
    return questions


def generate_pipeline_setup(topic: str, user_answers: str) -> Dict[str, Any]:
    """Создаёт 3 поисковых запроса для Stepik + карту тегов."""
    prompt = (
        f'Тема курса: "{topic}"\n\n'
        f"Уточнения от пользователя:\n{user_answers}\n\n"
        "Создай:\n"
        "1. РОВНО 3 поисковых запроса для Stepik (русский язык): "
        "широкий, точный, практико-ориентированный.\n"
        "2. Карту тегов (40–60 слов) для классификации материала. "
        "Категории: основные_концепции, инструменты_и_технологии, "
        "практические_навыки, смежные_темы, уровень_сложности."
    )
    result = _call_llm(prompt, "pipeline_setup", temperature=0.4)
    if not result.get("search_queries"):
        result["search_queries"] = [topic, f"{topic} курс", f"{topic} обучение"]
    if not result.get("tag_map"):
        result["tag_map"] = {"общие": [topic]}
    return result


def rank_courses_with_llm(
    topic: str,
    user_context: str,
    courses: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Ранжирует курсы по релевантности теме."""
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

    BATCH = 20
    rankings_map: Dict[int, Dict] = {}

    for i in range(0, len(courses), BATCH):
        chunk     = courses[i : i + BATCH]
        summaries = [_summary(c) for c in chunk]

        prompt = (
            f'Тема: "{topic}"\nКонтекст: {user_context}\n\n'
            f"Оцени релевантность каждого курса по шкале 1–10. "
            f"Главный критерий — соответствие теме и уровню пользователя. "
            f"Вторичные — популярность и объём.\n\n"
            f"Курсы:\n{json.dumps(summaries, ensure_ascii=False)}"
        )
        result = _call_llm(prompt, "course_ranking", temperature=0.1)
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
#  SESSION
# ════════════════════════════════════════════════════════════════════════════

def create_session(
    topic: str,
    tag_map: dict,
    user_context: str = "",
) -> Tuple[str, str]:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    session_id = f"{ts}_{short_uuid}"
    session_dir = os.path.join(SESSIONS_DIR, session_id)

    os.makedirs(os.path.join(session_dir, "raw_data"), exist_ok=True)
    os.makedirs(os.path.join(session_dir, "final"), exist_ok=True)

    with open(os.path.join(session_dir, "session_info.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"session_id": session_id, "topic": topic,
             "user_context": user_context, "created_at": datetime.now().isoformat()},
            f, ensure_ascii=False, indent=2,
        )
    with open(os.path.join(session_dir, "tag_map.json"), "w", encoding="utf-8") as f:
        json.dump(tag_map, f, ensure_ascii=False, indent=2)

    return session_id, session_dir


def load_tag_map(session_dir: str) -> Dict[str, Any]:
    path = os.path.join(session_dir, "tag_map.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    top = courses[:max_courses]

    for i, course in enumerate(top, 1):
        cid   = course.get("id")
        title = course.get("title", f"course_{cid}")
        score = course.get("_score", "?")
        _log(f"[{i}/{len(top)}] {title}  (score:{score})")
        try:
            loader.process_course_to_session(course, raw_data_dir, transcribe=transcribe)
        except Exception as e:
            _log(f"  ❌ {e}")


# ════════════════════════════════════════════════════════════════════════════
#  FULL STAGE 1 PIPELINE (используется из front.py)
# ════════════════════════════════════════════════════════════════════════════

def run_stage1(
    topic: str,
    user_answers: str,
    max_courses: int = DEFAULT_MAX_COURSES,
    limit_per_query: int = DEFAULT_LIMIT_PER_QUERY,
    transcribe: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """
    Запускает полный Stage 1.
    Возвращает session_dir или None при ошибке.
    """
    def _log(msg: str):
        print(msg)
        if log_fn:
            log_fn(msg)

    # 1. План поиска
    _log("🔧 Строю план поиска...")
    setup   = generate_pipeline_setup(topic, user_answers)
    queries = setup.get("search_queries", [topic])
    tag_map = setup.get("tag_map", {"общие": [topic]})

    q_str = " | ".join(queries)
    _log(f"📋 Запросы: {q_str}")

    # 2. Создаём сессию
    user_context = f"Тема: {topic}\nОтветы: {user_answers}"
    session_id, session_dir = create_session(topic, tag_map, user_context)
    _log(f"📁 Сессия: {session_id}")

    # 3. Stepik
    _log("🔍 Авторизация Stepik...")
    try:
        loader = StepikCourseLoader()
    except Exception as e:
        _log(f"❌ Stepik ошибка: {e}")
        return None

    _log(f"🔍 Поиск курсов...")
    courses = search_all_queries(loader, queries, limit_per_query=limit_per_query)
    _log(f"📚 Найдено курсов: {len(courses)}")

    if not courses:
        _log("⚠️ Курсы не найдены")
        return session_dir

    # 4. Ранжирование
    _log("🤖 Ранжирование по релевантности...")
    ranked = rank_courses_with_llm(topic, user_context, courses)
    top3   = " | ".join(c.get("title", "")[:40] for c in ranked[:3])
    _log(f"🏆 Топ-3: {top3}")

    # 5. Скачивание
    _log(f"⬇️  Скачиваю топ-{max_courses} курсов...")
    download_courses(
        loader, ranked, session_dir,
        max_courses=max_courses,
        transcribe=transcribe,
        log_fn=_log,
    )

    saved = len([f for f in os.listdir(os.path.join(session_dir, "raw_data"))
                 if f.endswith(".json")])
    _log(f"✅ Загружено уроков: {saved}")
    return session_dir


# ════════════════════════════════════════════════════════════════════════════
#  CLI (python loading_workflow.py для отладки)
# ════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    max_courses: int = DEFAULT_MAX_COURSES,
    transcribe: bool = True,
    limit_per_query: int = DEFAULT_LIMIT_PER_QUERY,
) -> Optional[str]:
    print("\n" + "=" * 60)
    print("  ПОИСК И СКАЧИВАНИЕ КУРСОВ СО STEPIK")
    print("=" * 60)

    topic = input("\nВведите тему: ").strip()
    if not topic:
        return None

    questions = generate_clarifying_questions(topic)
    print("\n[AI] Вопросы:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")

    print("\nОтветьте на вопросы (пустая строка = завершить):")
    lines = []
    while True:
        line = input("> ")
        if line == "" and lines:
            break
        if line:
            lines.append(line)

    return run_stage1(
        topic=topic,
        user_answers="\n".join(lines),
        max_courses=max_courses,
        limit_per_query=limit_per_query,
        transcribe=transcribe,
    )