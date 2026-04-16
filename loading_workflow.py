"""
loading_workflow.py
────────────────────
Интерактивный пайплайн поиска и скачивания курсов со Stepik.

Поток:
  1. Пользователь вводит тему
  2. LLM генерирует уточняющие вопросы
  3. Пользователь отвечает на вопросы
  4. LLM создаёт: 3 поисковых запроса + карту тегов
  5. Создаётся уникальная сессия (папка sessions/{id}/)
  6. Stepik ищется по 3 запросам, результаты дедуплицируются
  7. LLM ранжирует курсы по релевантности + популярности + объёму
  8. Скачиваются все уроки топ-N курсов в sessions/{id}/raw_data/
  9. Видео-уроки транскрибируются автоматически
"""

import os
import sys
import json
import time
import uuid
import requests
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

from CourseProcessor.CourseLoader import StepikCourseLoader
from config import AppConfig

# ─── Константы ──────────────────────────────────────────────────────────────

SESSIONS_DIR = "sessions"
GEMINI_URL = f"{AppConfig.ML_SERVER_URL.rstrip('/')}/gemini"
SESSIONS_DIR = "sessions"
LOCAL_TASK_URL = "http://localhost:8000/task"
DEFAULT_LLM_MODEL = "Qwen2.5-3B-Instruct-Q4_K_L.gguf"

DEFAULT_MAX_COURSES = 5 
DEFAULT_LIMIT_PER_QUERY = 30


# ════════════════════════════════════════════════════════════════════════════
#  LLM HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _call_llm(
    prompt: str,
    schema_name: str,
    system_prompt: str = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> dict:
    """
    Вызывает локальный /task endpoint с указанной схемой.
    Возвращает структурированный dict или {} при ошибке.
    """
    payload = {
        "task_type": "llm",
        "model_name": DEFAULT_LLM_MODEL,
        "text": prompt,
        "system_prompt": system_prompt,
        "schema_name": schema_name,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "n_ctx": 4096
    }
    try:
        resp = requests.post(LOCAL_TASK_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        
        # Получаем поле result из ответа FastAPI
        result = data.get("result", {})
        
        # Если Llama-cpp вернул {"response": "строка с json"}, парсим JSON из текста
        if isinstance(result, dict) and "response" in result and len(result) == 1:
            import re
            text = result["response"]
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return result
        return result
    except requests.exceptions.ConnectionError:
        print(f"[LLM Error] Не могу подключиться к ML backend: {LOCAL_TASK_URL}")
        return {}
    except Exception as e:
        print(f"[LLM Error] {type(e).__name__}: {e}")
        return {}



def generate_clarifying_questions(topic: str) -> List[str]:
    """LLM генерирует 3–5 уточняющих вопросов по теме."""
    print("\n[AI] Анализирую тему и генерирую уточняющие вопросы...")

    prompt = (
        f'Пользователь хочет найти образовательные курсы по теме: "{topic}"\n\n'
        "Сгенерируй 3–5 уточняющих вопроса на русском языке, чтобы понять:\n"
        "- Уровень подготовки (новичок / средний / продвинутый)\n"
        "- Конкретные аспекты темы, которые интересуют\n"
        "- Цель обучения (карьера / хобби / учёба / проект)\n"
        "- Предпочтения по стилю (теория / практика / проекты)\n\n"
        "Вопросы должны быть конкретными и помогать лучше подобрать курсы."
    )

    result = _call_llm(prompt, schema_name="clarifying_questions")
    questions = result.get("questions", [])

    if not questions:
        # Fallback если LLM недоступен
        questions = [
            "Какой у вас текущий уровень подготовки по этой теме?",
            "Что конкретно вы хотите научиться делать или понять?",
            "Для какой цели изучаете тему — работа, проект, саморазвитие?",
            "Какой формат обучения предпочитаете — теория, практические задачи, видео?",
        ]
        print("[AI] Использую стандартные вопросы (LLM недоступен)")

    return questions


def generate_pipeline_setup(topic: str, user_answers: str) -> Dict[str, Any]:
    """
    LLM создаёт план поиска:
    - 3 поисковых запроса для Stepik
    - Карту тегов для последующего тегирования материала
    """
    print("\n[AI] Создаю план поиска и карту тегов...")

    prompt = (
        f'Тема обучения: "{topic}"\n\n'
        f"Ответы пользователя на уточняющие вопросы:\n{user_answers}\n\n"
        "На основе этой информации создай:\n\n"
        "1. РОВНО 3 поисковых запроса для платформы Stepik (на русском языке).\n"
        "   Каждый запрос — разная формулировка одной темы: широкая, узкая, практическая.\n\n"
        "2. Подробную карту тегов (40–60 ключевых слов) для последующей классификации\n"
        "   учебного материала. Разбей теги по категориям:\n"
        "   - основные_концепции\n"
        "   - инструменты_и_технологии\n"
        "   - практические_навыки\n"
        "   - смежные_темы\n"
        "   - уровень_сложности\n"
        "   (можно добавить свои категории если они уместны)"
    )

    result = _call_llm(prompt, schema_name="pipeline_setup", temperature=0.4)

    # Fallback
    if not result.get("search_queries"):
        result["search_queries"] = [topic, f"{topic} курс", f"{topic} обучение"]
        print("[AI] Использую базовые запросы (LLM недоступен)")

    if not result.get("tag_map"):
        result["tag_map"] = {"общие": [topic]}

    return result


def rank_courses_with_llm(
    topic: str,
    user_context: str,
    courses: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    LLM ранжирует курсы по релевантности теме + метаданным.
    Обрабатывает батчами по 20 курсов.
    """
    if not courses:
        return []

    print(f"\n[AI] Ранжирую {len(courses)} курсов по релевантности...")

    # Готовим краткие сводки курсов для LLM
    def _make_summary(c: Dict) -> Dict:
        return {
            "id": c.get("id"),
            "title": c.get("title", ""),
            "is_popular": c.get("is_popular", False),
            "sections_count": len(c.get("sections", [])),
            "learners_count": c.get("learners_count", 0),
            "rating": c.get("rating", 0),
        }

    # Батчинг по 20 курсов
    BATCH = 20
    rankings_map: Dict[int, Dict] = {}

    for i in range(0, len(courses), BATCH):
        chunk = courses[i:i + BATCH]
        summaries = [_make_summary(c) for c in chunk]

        prompt = (
            f'Тема поиска: "{topic}"\n'
            f"Контекст пользователя: {user_context}\n\n"
            "Оцени релевантность каждого курса по шкале 1–10.\n"
            "Учитывай:\n"
            "  - соответствие теме и уровню пользователя (главный критерий)\n"
            "  - популярность курса (is_popular, learners_count)\n"
            "  - объём материала (sections_count)\n\n"
            f"Список курсов:\n{json.dumps(summaries, ensure_ascii=False)}\n\n"
            "ВАЖНО: верни оценку для КАЖДОГО курса из списка."
        )

        result = _call_llm(prompt, schema_name="course_ranking", temperature=0.1)
        for r in result.get("rankings", []):
            cid = r.get("id")
            if cid:
                rankings_map[cid] = r

        if i + BATCH < len(courses):
            time.sleep(1) 

    # Применяем оценки
    for course in courses:
        cid = course.get("id")
        rank_info = rankings_map.get(cid, {})
        course["_score"] = rank_info.get("score", 0)
        course["_rank_reason"] = rank_info.get("reason", "")

    ranked = sorted(courses, key=lambda x: x.get("_score", 0), reverse=True)

    print(f"[AI] Ранжирование завершено. Топ-3:")
    for c in ranked[:3]:
        print(f"  [{c.get('_score', '?')}] {c.get('title')} — {c.get('_rank_reason', '')}")

    return ranked


# ════════════════════════════════════════════════════════════════════════════
#  SESSION MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════

def create_session(topic: str, tag_map: dict, user_context: str = "") -> Tuple[str, str]:
    """
    Создаёт уникальную папку сессии:
        sessions/{YYYYMMDD_HHMMSS}_{uuid8}/
            raw_data/      ← скачанные уроки
            final/         ← обработанные материалы
            session_info.json
            tag_map.json
    """
    os.makedirs(SESSIONS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    session_id = f"{timestamp}_{short_uuid}"
    session_dir = os.path.join(SESSIONS_DIR, session_id)

    os.makedirs(os.path.join(session_dir, "raw_data"), exist_ok=True)
    os.makedirs(os.path.join(session_dir, "final"), exist_ok=True)

    session_info = {
        "session_id": session_id,
        "topic": topic,
        "user_context": user_context,
        "created_at": datetime.now().isoformat(),
    }

    with open(os.path.join(session_dir, "session_info.json"), "w", encoding="utf-8") as f:
        json.dump(session_info, f, ensure_ascii=False, indent=2)

    with open(os.path.join(session_dir, "tag_map.json"), "w", encoding="utf-8") as f:
        json.dump(tag_map, f, ensure_ascii=False, indent=2)

    print(f"\n[Session] ID: {session_id}")
    print(f"[Session] Папка: {session_dir}")
    return session_id, session_dir


def load_tag_map(session_dir: str) -> Dict[str, Any]:
    """Загружает карту тегов из папки сессии."""
    path = os.path.join(session_dir, "tag_map.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════════════════
#  STEPIK SEARCH
# ════════════════════════════════════════════════════════════════════════════

def fetch_stepik_courses(topic: str, limit: int = 100) -> Tuple[StepikCourseLoader, List[Dict]]:
    """Deprecated — используй search_all_queries. Оставлен для совместимости."""
    print(f"[Stepik] Поиск курсов по теме: {topic}...")
    loader = StepikCourseLoader()
    course_ids = loader.get_course_ids_by_query(query=topic, limit=limit)
    if not course_ids:
        return loader, []
    raw_courses = loader.fetch_objects("courses", course_ids)
    print(f"[Stepik] Загружено: {len(raw_courses)} курсов")
    return loader, raw_courses


def search_all_queries(
    loader: StepikCourseLoader,
    queries: List[str],
    limit_per_query: int = DEFAULT_LIMIT_PER_QUERY,
) -> List[Dict[str, Any]]:
    """
    Ищет курсы по нескольким запросам и дедуплицирует результаты.
    Возвращает полные объекты курсов.
    """
    all_ids: List[int] = []
    seen_ids: set = set()

    for query in queries:
        ids = loader.get_course_ids_by_query(query, limit=limit_per_query)
        new_count = 0
        for cid in ids:
            if cid not in seen_ids:
                all_ids.append(cid)
                seen_ids.add(cid)
                new_count += 1
        print(f"  Запрос '{query}': +{new_count} новых (итого {len(all_ids)})")

    if not all_ids:
        return []

    print(f"[Search] Всего уникальных курсов: {len(all_ids)}. Загружаю метаданные...")
    courses = loader.fetch_objects("courses", all_ids)

    # Оставляем только публичные и бесплатные
    courses = [c for c in courses if c.get("is_public") and not c.get("is_paid")]
    print(f"[Search] Публичных и бесплатных: {len(courses)}")

    return courses


# ════════════════════════════════════════════════════════════════════════════
#  DOWNLOAD
# ════════════════════════════════════════════════════════════════════════════

def download_courses(
    loader: StepikCourseLoader,
    courses: List[Dict[str, Any]],
    session_dir: str,
    max_courses: int = DEFAULT_MAX_COURSES,
):
    """
    Скачивает топ-N курсов в {session_dir}/raw_data/.
    Каждый урок → отдельный {lesson_name}.json с текстом и транскрипцией.
    """
    raw_data_dir = os.path.join(session_dir, "raw_data")
    top_courses = courses[:max_courses]

    print(f"\n{'='*60}")
    print(f"СКАЧИВАНИЕ {len(top_courses)} КУРСОВ")
    print(f"{'='*60}")

    for i, course in enumerate(top_courses, 1):
        course_id = course.get("id")
        course_title = course.get("title", f"course_{course_id}")
        score = course.get("_score", "?")
        print(f"\n[{i}/{len(top_courses)}] {course_title}  (score: {score})")
        print(f"  Причина: {course.get('_rank_reason', '—')}")

        try:
            loader.process_course_to_session(course, raw_data_dir)
        except Exception as e:
            print(f"  [ERROR] Ошибка при скачивании курса {course_id}: {e}")

    # Итог
    saved_files = [f for f in os.listdir(raw_data_dir) if f.endswith(".json")]
    print(f"\n[DONE] Сохранено уроков: {len(saved_files)}")
    print(f"[DONE] Папка: {raw_data_dir}")


# ════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ════════════════════════════════════════════════════════════════════════════

def print_top_results(ranked_courses: List[Dict], top_n: int = 15):
    """Выводит топ курсов в консоль."""
    print(f"\n{'='*60}")
    print(f"ТОП-{min(top_n, len(ranked_courses))} КУРСОВ ПО РЕЛЕВАНТНОСТИ")
    print(f"{'='*60}")
    for course in ranked_courses[:top_n]:
        popular = "⭐" if course.get("is_popular") else "  "
        sections = len(course.get("sections", []))
        print(
            f"[{course.get('_score', '?'):>2}] {popular} {course.get('title')}\n"
            f"       ID: {course.get('id')} | Разделов: {sections} | "
            f"Слушателей: {course.get('learners_count', '?')}\n"
            f"       {course.get('_rank_reason', '')}\n"
        )


# ════════════════════════════════════════════════════════════════════════════
#  MAIN INTERACTIVE PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def run_pipeline(max_courses: int = DEFAULT_MAX_COURSES) -> Optional[str]:
    """
    Запускает полный интерактивный пайплайн.

    Returns:
        session_dir — путь к папке сессии, или None при ошибке.
    """
    print("\n" + "=" * 60)
    print("   ПОИСК И СКАЧИВАНИЕ КУРСОВ СО STEPIK")
    print("=" * 60)

    # ── 1. Тема ──────────────────────────────────────────────────────────
    topic = input("\nВведите тему для обучения: ").strip()
    if not topic:
        print("[ERROR] Тема не введена.")
        return None

    # ── 2. Уточняющие вопросы ─────────────────────────────────────────────
    questions = generate_clarifying_questions(topic)

    print("\n[AI] Уточняющие вопросы:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")

    print("\nОтветьте на вопросы (одним сообщением или несколькими строками).")
    print("Когда закончите, введите пустую строку:")

    answer_lines = []
    while True:
        line = input("> ")
        if line == "" and answer_lines:
            break
        if line:
            answer_lines.append(line)

    user_answers = "\n".join(answer_lines)
    if not user_answers:
        print("[WARN] Ответы не введены, продолжаем с базовыми настройками.")
        user_answers = "нет дополнительной информации"

    # ── 3. План поиска + карта тегов ─────────────────────────────────────
    setup = generate_pipeline_setup(topic, user_answers)

    search_queries: List[str] = setup.get("search_queries", [topic])
    tag_map: Dict = setup.get("tag_map", {"общие": [topic]})

    print(f"\n[AI] Поисковые запросы:")
    for i, q in enumerate(search_queries, 1):
        print(f"  {i}. {q}")

    total_tags = sum(len(v) for v in tag_map.values())
    print(f"[AI] Карта тегов: {len(tag_map)} категорий, {total_tags} тегов")

    # ── 4. Создаём сессию ─────────────────────────────────────────────────
    user_context = f"Тема: {topic}\nОтветы: {user_answers}"
    session_id, session_dir = create_session(topic, tag_map, user_context)

    # ── 5. Поиск на Stepik ────────────────────────────────────────────────
    print("\n[Stepik] Авторизация и поиск...")
    try:
        loader = StepikCourseLoader()
    except Exception as e:
        print(f"[ERROR] Не удалось создать загрузчик: {e}")
        return None

    courses = search_all_queries(loader, search_queries)

    if not courses:
        print("[WARN] Курсы не найдены. Попробуйте другую формулировку темы.")
        return session_dir

    # ── 6. Ранжирование ───────────────────────────────────────────────────
    ranked_courses = rank_courses_with_llm(topic, user_context, courses)
    print_top_results(ranked_courses)

    # ── 7. Подтверждение ─────────────────────────────────────────────────
    print(f"\nБудет скачано топ-{max_courses} курсов.")
    confirm = input("Начать скачивание? [Y/n]: ").strip().lower()
    if confirm not in ("", "y", "yes", "д", "да"):
        print("[SKIP] Скачивание отменено.")
        return session_dir

    # ── 8. Скачивание ─────────────────────────────────────────────────────
    download_courses(loader, ranked_courses, session_dir, max_courses=max_courses)

    print(f"\n✅ Готово! Данные сохранены в: {session_dir}/raw_data/")
    print(f"   Карта тегов: {session_dir}/tag_map.json")

    return session_dir