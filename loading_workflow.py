"""
loading_workflow.py
────────────────────
Stage 1 v2: поиск, фильтрация и скачивание курсов со Stepik.

Новый пайплайн (7 шагов + 2 раунда оценки):

  Шаг 1  | generate_clarifying_questions(topic)
             └→ вызывается ДО run_stage1, из front.py / CLI

  Шаг 2  | 9B + thinking → детальная структура курса (модули, цели, темы)
             └→ выгрузка 9B из VRAM

  Шаг 3  | 3B → поисковые запросы (5 шт.) + карта тегов (40-70) для Stage 2
             └→ выгрузка 3B из VRAM

  Шаг 4  | Stepik search → для каждого найденного курса:
             4a. get_course_outline → список уроков (title, lesson_id)
             4b. 4B → распределение уроков по модулям (1 запрос / курс)
             └→ выгрузка 4B из VRAM

  Шаг 5  | Скачивание + транскрибация назначенных уроков
             (lesson JSON дополняется полем module_id)

  Шаг 6  | 9B + thinking → оценка покрытия раунд 1
             └→ выгрузка 9B + доп. поиск + раздача + скачивание

  Шаг 7  | 9B + thinking → оценка покрытия раунд 2
             └→ выгрузка 9B + доп. поиск + раздача + скачивание

Модели:
  THINKING_MODEL     = Qwen3.5-9B-Q5_K_M.gguf          (шаги 2, 6, 7)
  MAIN_MODEL         = Qwen2.5-3B-Instruct-Q4_K_L.gguf  (шаги 1, 3)
  DISTRIBUTION_MODEL = RuadaptQwen3-4B-Hybrid-Q8_0.gguf (шаги 4b, 6 доп., 7 доп.)

front.py совместимость: сигнатура run_stage1 расширена параметром
  clarifying_questions: list[str] | None = None
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from config import AppConfig
from CourseProcessor.CourseLoader import StepikCourseLoader
from LLMprompts import PromptBank

# ════════════════════════════════════════════════════════════════════════════
#  Константы моделей
# ════════════════════════════════════════════════════════════════════════════

# 9B с thinking: генерация структуры + оценка покрытия
THINKING_MODEL     = "Qwen3.5-9B-Q5_K_M.gguf"

# 3B без thinking: уточняющие вопросы + поисковые запросы + теги
MAIN_MODEL         = "RuadaptQwen3-4B-Hybrid-Q8_0.gguf"

# 4B hybrid: распределение уроков по модулям (per-course, no thinking)
# Имя файла квантования RuadaptQwen3-4B-Hybrid-GGUF
DISTRIBUTION_MODEL = "RuadaptQwen3-4B-Hybrid-Q8_0.gguf"

# Параметры запросов к ML-бэкенду
SESSIONS_DIR            = "sessions"
_BASE_URL               = AppConfig.ML_SERVER_URL.rstrip("/")
TASK_URL                = f"{_BASE_URL}/task"
UNLOAD_URL              = f"{_BASE_URL}/models"

DEFAULT_MAX_COURSES     = 5
DEFAULT_LIMIT_PER_QUERY = 30

# Максимум уроков одного курса, подаваемых на distribution (контекст 4B)
MAX_LESSONS_PER_DIST    = 80

# ════════════════════════════════════════════════════════════════════════════
#  Низкоуровневые HTTP-хелперы
# ════════════════════════════════════════════════════════════════════════════

def _call_llm(
    prompt: str,
    schema_name: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    model: str = MAIN_MODEL,
    enable_thinking: bool = False,
    n_ctx: int = 4096,
    timeout: int = 900,
) -> dict:
    """
    POST /task к ML-бэкенду. Возвращает распарсенный dict или {}.

    timeout:  секунды ожидания ответа.
              Малые модели (3-4B): 300–900 с.
              9B с thinking:      1800 с (30 минут) — генерация + thinking.
    """
    payload = {
        "task_type":       "llm",
        "model_name":      model,
        "text":            prompt,
        "schema_name":     schema_name,
        "max_tokens":      max_tokens,
        "temperature":     temperature,
        "top_p":           0.9,
        "n_ctx":           n_ctx,
        "enable_thinking": enable_thinking,
    }
    try:
        resp = requests.post(TASK_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        result = resp.json().get("result", {})
        # Стандартный fallback: JSON-объект внутри текстового ответа
        if isinstance(result, dict) and "response" in result and len(result) == 1:
            m = re.search(r"\{.*\}", result["response"], re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        return result if isinstance(result, dict) else {}
    except requests.exceptions.ConnectionError:
        print(f"[LLM] Нет соединения с ML-бэкендом: {TASK_URL}")
        return {}
    except Exception as e:
        print(f"[LLM] {type(e).__name__}: {e}")
        return {}


def _unload_model(model_name: str, log_fn=None) -> bool:
    """
    DELETE /models/{name} — выгружает модель из VRAM на ML-бэкенде.
    Вызывается между этапами пайплайна.
    """
    def _log(msg):
        print(msg)
        if log_fn:
            log_fn(msg)

    try:
        resp = requests.delete(
            f"{UNLOAD_URL}/{model_name}",
            timeout=30,
        )
        data = resp.json()
        count = data.get("count", 0)
        if count > 0:
            _log(f"[VRAM] Выгружена: {model_name} ({count} вариантов)")
        else:
            _log(f"[VRAM] {model_name} не была в кэше")
        return True
    except Exception as e:
        _log(f"[VRAM] Ошибка выгрузки {model_name}: {e}")
        return False


# ════════════════════════════════════════════════════════════════════════════
#  Шаг 1 — уточняющие вопросы (3B, без thinking)
# ════════════════════════════════════════════════════════════════════════════

def generate_clarifying_questions(topic: str) -> List[str]:
    """Вызывается из front.py ДО run_stage1."""
    result = _call_llm(
        PromptBank.clarifying_questions(topic),
        "clarifying_questions",
        temperature=0.4,
        max_tokens=512,
        model=MAIN_MODEL,
        enable_thinking=False,
        n_ctx=2048,
    )
    questions = result.get("questions", [])
    if not questions:
        questions = [
            "Какой уровень подготовки у целевой аудитории?",
            "Какие конкретные инструменты или фреймворки важно охватить?",
            "Какова главная практическая цель обучения?",
            "Какие смежные темы стоит включить?",
        ]
    return questions


# ════════════════════════════════════════════════════════════════════════════
#  Шаг 2 — детальная структура курса (9B + thinking)
# ════════════════════════════════════════════════════════════════════════════

def generate_course_structure(
    topic: str,
    clarifying_questions: List[str],
    user_answers: str,
    log_fn=None,
) -> Dict[str, Any]:
    """
    9B с thinking → детальная структура курса.
    После вызова: выгрузить THINKING_MODEL.
    """
    def _log(msg):
        print(msg)
        if log_fn:
            log_fn(msg)

    _log(f"[Структура] Генерирую с thinking ({THINKING_MODEL})...")
    prompt = PromptBank.course_structure_thinking(topic, clarifying_questions, user_answers)
    result = _call_llm(
        prompt,
        "course_structure_detailed",
        temperature=0.5,
        max_tokens=4096,  
        model=THINKING_MODEL,
        enable_thinking=True,
        n_ctx=4096,
        timeout=1800,
    )

    if not result.get("modules"):
        _log("[Структура] LLM не вернула модули — создаю базовую структуру")
        result = _build_fallback_structure(topic)

    n_modules = len(result.get("modules", []))
    _log(f"[Структура] Готово: {n_modules} модулей")
    return result


def _build_fallback_structure(topic: str) -> Dict[str, Any]:
    """Минимальная структура если 9B упала или вернула пустой ответ."""
    return {
        "course_title":       f"Курс: {topic}",
        "course_description": f"Учебный курс по теме «{topic}».",
        "course_goals":       [f"Освоить ключевые аспекты темы «{topic}»"],
        "modules": [
            {
                "id": 0, "title": "Введение и основы",
                "description": "Базовые понятия и концепции.",
                "goals": ["Понять основные принципы"],
                "key_topics": [topic],
            },
            {
                "id": 1, "title": "Практическое применение",
                "description": "Практические навыки и инструменты.",
                "goals": ["Применять знания на практике"],
                "key_topics": ["практика", "инструменты"],
            },
            {
                "id": 2, "title": "Продвинутые темы",
                "description": "Углублённое изучение и специализация.",
                "goals": ["Освоить продвинутые концепции"],
                "key_topics": ["продвинутый уровень"],
            },
        ],
    }


# ════════════════════════════════════════════════════════════════════════════
#  Шаг 3 — поисковые запросы + теги (3B, без thinking)
# ════════════════════════════════════════════════════════════════════════════

def generate_search_setup(
    topic: str,
    course_structure: Dict[str, Any],
    user_answers: str,
    log_fn=None,
) -> Dict[str, Any]:
    """
    3B без thinking → 5 запросов для Stepik + плоский список тегов.
    После вызова: выгрузить MAIN_MODEL.
    """
    def _log(msg):
        print(msg)
        if log_fn:
            log_fn(msg)

    _log(f"[Поиск] Генерирую запросы и теги ({MAIN_MODEL})...")
    prompt = PromptBank.search_setup_from_structure(topic, course_structure, user_answers)
    result = _call_llm(
        prompt,
        "pipeline_setup",
        temperature=0.3,
        max_tokens=2048,
        model=MAIN_MODEL,
        enable_thinking=False,
        n_ctx=4096,
    )

    if not result.get("search_queries"):
        result["search_queries"] = [
            topic,
            f"{topic} курс",
            f"{topic} практика",
            f"обучение {topic}",
            f"{topic} для начинающих",
        ]
    if not result.get("tags"):
        result["tags"] = [topic]

    _log(f"[Поиск] Запросов: {len(result['search_queries'])}, тегов: {len(result['tags'])}")
    return result


# ════════════════════════════════════════════════════════════════════════════
#  Шаги 4a-4b — Stepik поиск + распределение уроков (4B, без thinking)
# ════════════════════════════════════════════════════════════════════════════

def search_stepik(
    loader: StepikCourseLoader,
    queries: List[str],
    limit_per_query: int = DEFAULT_LIMIT_PER_QUERY,
    log_fn=None,
) -> List[Dict[str, Any]]:
    """Поиск курсов по всем запросам, дедупликация."""
    def _log(msg):
        print(msg)
        if log_fn:
            log_fn(msg)

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
    public_free = [c for c in courses if c.get("is_public") and not c.get("is_paid")]
    _log(f"[Stepik] Найдено курсов: {len(public_free)} (из {len(courses)} всего)")
    return public_free


def distribute_lessons_for_course(
    loader: StepikCourseLoader,
    course: Dict[str, Any],
    modules: List[Dict[str, Any]],
    log_fn=None,
) -> Dict[int, int]:
    """
    Для одного курса:
    1. Получает список уроков через get_course_outline
    2. Одним запросом к 4B модели распределяет уроки по модулям

    Returns:
        {lesson_id: module_id}  — только для assigned уроков
    """
    def _log(msg):
        print(msg)
        if log_fn:
            log_fn(msg)

    course_title = course.get("title", f"course_{course.get('id')}")

    # Получаем оглавление курса (только метаданные, без скачивания)
    outline = loader.get_course_outline(course)
    if not outline:
        _log(f"  [Dist] {course_title}: нет уроков в оглавлении")
        return {}

    # Ограничиваем размер для контекста 4B
    lessons_for_prompt = outline[:MAX_LESSONS_PER_DIST]
    if len(outline) > MAX_LESSONS_PER_DIST:
        _log(
            f"  [Dist] {course_title}: {len(outline)} уроков → "
            f"подаём первые {MAX_LESSONS_PER_DIST}"
        )

    prompt = PromptBank.lesson_distribution(modules, lessons_for_prompt, course_title)
    result = _call_llm(
        prompt,
        "lesson_distribution",
        temperature=0.1,       # детерминированность важна для корректной разметки
        max_tokens=2048,
        model=DISTRIBUTION_MODEL,
        enable_thinking=False,
        n_ctx=6144,
    )

    # Парсим ответ
    assignment_map: Dict[int, int] = {}
    for item in result.get("assignments", []):
        lid = item.get("lesson_id")
        mid = item.get("module_id")
        if lid is not None and mid is not None:
            assignment_map[int(lid)] = int(mid)

    n_assigned   = len(assignment_map)
    n_unassigned = len(result.get("unassigned", []))
    _log(
        f"  [Dist] {course_title}: {n_assigned} назначено, "
        f"{n_unassigned} не подошло"
    )
    return assignment_map


def distribute_lessons_for_courses(
    loader: StepikCourseLoader,
    courses: List[Dict[str, Any]],
    course_structure: Dict[str, Any],
    log_fn=None,
) -> Dict[int, int]:
    """
    Распределяет уроки всех курсов.
    Returns: глобальная карта {lesson_id: module_id}
    """
    def _log(msg):
        print(msg)
        if log_fn:
            log_fn(msg)

    modules = course_structure.get("modules", [])
    global_map: Dict[int, int] = {}

    _log(f"[Dist] Распределение уроков для {len(courses)} курсов ({DISTRIBUTION_MODEL})...")
    for i, course in enumerate(courses, 1):
        cid = course.get("id")
        title = course.get("title", str(cid))
        _log(f"  [{i}/{len(courses)}] {title}")
        partial = distribute_lessons_for_course(loader, course, modules, log_fn)
        global_map.update(partial)
        time.sleep(0.3)  # throttle

    _log(f"[Dist] Итого назначено уроков: {len(global_map)}")
    return global_map


# ════════════════════════════════════════════════════════════════════════════
#  Шаг 5 — скачивание + простановка module_id
# ════════════════════════════════════════════════════════════════════════════

def download_assigned_courses(
    loader: StepikCourseLoader,
    courses: List[Dict[str, Any]],
    session_dir: str,
    lesson_module_map: Dict[int, int],
    transcribe: bool = True,
    max_courses: int = DEFAULT_MAX_COURSES,
    log_fn=None,
) -> int:
    """
    Скачивает уроки из назначенных курсов, проставляет module_id в JSON.
    Скачиваются только курсы из списка (по max_courses).
    Returns: количество скачанных файлов (новых).
    """
    def _log(msg):
        print(msg)
        if log_fn:
            log_fn(msg)

    raw_data_dir = os.path.join(session_dir, "raw_data")
    os.makedirs(raw_data_dir, exist_ok=True)

    files_before = set(_json_files(raw_data_dir))

    for i, course in enumerate(courses[:max_courses], 1):
        cid   = course.get("id")
        title = course.get("title", str(cid))
        _log(f"  [{i}/{min(max_courses, len(courses))}] Скачиваю: {title}")

        # Проверяем/записываемся на курс
        if not course.get("is_enrolled"):
            if not loader.check_enrollment(cid):
                loader.enroll_in_course(cid)
                time.sleep(1)

        try:
            loader.process_course_to_session(course, raw_data_dir, transcribe=transcribe)
        except Exception as e:
            _log(f"    ❌ Ошибка скачивания: {e}")

    # Простановка module_id во все новые файлы
    files_after = set(_json_files(raw_data_dir))
    new_files   = files_after - files_before
    _tag_files_with_modules(raw_data_dir, new_files, lesson_module_map, log_fn)

    _log(f"[Download] Новых файлов: {len(new_files)}")
    return len(new_files)


def _json_files(directory: str) -> List[str]:
    """Список имён .json файлов в директории."""
    if not os.path.isdir(directory):
        return []
    return [f for f in os.listdir(directory) if f.endswith(".json")]


def _tag_files_with_modules(
    raw_data_dir: str,
    filenames: set,
    lesson_module_map: Dict[int, int],
    log_fn=None,
) -> None:
    """
    Читает каждый новый JSON-файл, добавляет поле module_id по lesson_id.
    Уроки без назначения получают module_id = -1 (неназначенные).
    """
    for fname in filenames:
        path = os.path.join(raw_data_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            lesson_id = data.get("lesson_id")
            if lesson_id is not None:
                module_id = lesson_module_map.get(int(lesson_id), -1)
                data["module_id"] = module_id

                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if log_fn:
                log_fn(f"    [Tag] Ошибка {fname}: {e}")


# ════════════════════════════════════════════════════════════════════════════
#  Шаги 6-7 — оценка покрытия (9B + thinking)
# ════════════════════════════════════════════════════════════════════════════

def build_lessons_by_module(raw_data_dir: str) -> Dict[int, List[str]]:
    """
    Читает все JSON из raw_data_dir и группирует названия уроков по module_id.
    module_id = -1 → неназначенные.
    """
    by_module: Dict[int, List[str]] = {}
    if not os.path.isdir(raw_data_dir):
        return by_module

    for fname in sorted(os.listdir(raw_data_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(raw_data_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            mid   = int(data.get("module_id", -1))
            title = data.get("lesson_name", fname.replace(".json", ""))
            by_module.setdefault(mid, []).append(title)
        except Exception:
            pass

    return by_module


def evaluate_coverage(
    course_structure: Dict[str, Any],
    session_dir: str,
    round_num: int,
    previous_assessment: str = "",
    log_fn=None,
) -> Dict[str, Any]:
    """
    9B + thinking → оценка покрытия + дополнительные запросы.
    После вызова: выгрузить THINKING_MODEL.

    Returns:
        dict по схеме coverage_evaluation
    """
    def _log(msg):
        print(msg)
        if log_fn:
            log_fn(msg)

    raw_data_dir    = os.path.join(session_dir, "raw_data")
    lessons_by_mod  = build_lessons_by_module(raw_data_dir)
    total           = sum(len(v) for v in lessons_by_mod.values())
    _log(f"[Coverage] Раунд {round_num}: {total} уроков, {THINKING_MODEL}...")

    prompt = PromptBank.coverage_evaluation(
        course_structure    = course_structure,
        lessons_by_module   = lessons_by_mod,
        round_num           = round_num,
        previous_assessment = previous_assessment,
    )
    result = _call_llm(
        prompt,
        "coverage_evaluation",
        temperature=0.3,
        max_tokens=3500,  # thinking + JSON оценки
        model=THINKING_MODEL,
        enable_thinking=True,
        n_ctx=12288,
        timeout=1800,  # 30 мин
    )

    score       = result.get("coverage_score", 0.0)
    add_queries = result.get("additional_queries", [])
    assessment  = result.get("assessment", "")
    _log(
        f"[Coverage] Раунд {round_num}: score={score:.2f}, "
        f"доп. запросов={len(add_queries)}"
    )
    _log(f"[Coverage] Оценка: {assessment}")
    return result


# ════════════════════════════════════════════════════════════════════════════
#  Сессия
# ════════════════════════════════════════════════════════════════════════════

def create_session(
    topic: str,
    tags: List[str],
    course_structure: Dict[str, Any],
    user_answers: str = "",
    clarifying_questions: List[str] = None,
) -> Tuple[str, str]:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{ts}_{str(uuid.uuid4())[:8]}"
    session_dir = os.path.join(SESSIONS_DIR, session_id)

    os.makedirs(os.path.join(session_dir, "raw_data"), exist_ok=True)
    os.makedirs(os.path.join(session_dir, "final"), exist_ok=True)

    # session_info.json
    with open(os.path.join(session_dir, "session_info.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "session_id":           session_id,
                "topic":                topic,
                "user_answers":         user_answers,
                "clarifying_questions": clarifying_questions or [],
                "created_at":           datetime.now().isoformat(),
            },
            f, ensure_ascii=False, indent=2,
        )

    # tags.json — плоский список для Stage 2
    with open(os.path.join(session_dir, "tags.json"), "w", encoding="utf-8") as f:
        json.dump(tags, f, ensure_ascii=False, indent=2)

    # course_structure.json — детальная структура от 9B
    with open(os.path.join(session_dir, "course_structure.json"), "w", encoding="utf-8") as f:
        json.dump(course_structure, f, ensure_ascii=False, indent=2)

    return session_id, session_dir


def _save_coverage_report(session_dir: str, round_num: int, report: Dict[str, Any]) -> None:
    path = os.path.join(session_dir, f"coverage_round{round_num}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def _save_lesson_module_map(session_dir: str, mapping: Dict[int, int]) -> None:
    path = os.path.join(session_dir, "lesson_module_map.json")
    # Если файл уже есть — мержим (новые данные не перетирают старые)
    existing: Dict[str, int] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update({str(k): v for k, v in mapping.items()})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


# ════════════════════════════════════════════════════════════════════════════
#  Вспомогательные (совместимость со Stage 2)
# ════════════════════════════════════════════════════════════════════════════

def load_tags(session_dir: str) -> List[str]:
    path = os.path.join(session_dir, "tags.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # backward compat: tag_map.json (словарь или список)
    old = os.path.join(session_dir, "tag_map.json")
    if os.path.exists(old):
        with open(old, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        tags: List[str] = []
        for v in data.values():
            if isinstance(v, list):
                tags.extend(v)
        return tags
    return []


# ════════════════════════════════════════════════════════════════════════════
#  ГЛАВНЫЙ ПАЙПЛАЙН
# ════════════════════════════════════════════════════════════════════════════

def run_stage1(
    topic: str,
    user_answers: str,
    clarifying_questions: Optional[List[str]] = None,
    max_courses: int = DEFAULT_MAX_COURSES,
    limit_per_query: int = DEFAULT_LIMIT_PER_QUERY,
    transcribe: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """
    Полный Stage 1: от ответов пользователя до скачанных уроков с module_id.

    Args:
        topic:                Тема курса.
        user_answers:         Ответы пользователя на уточняющие вопросы.
        clarifying_questions: Сами вопросы (для контекста 9B-промпта).
                              Если None — prompt будет без них.
        max_courses:          Сколько курсов скачивать за один раунд.
        limit_per_query:      Лимит ID курсов на один поисковый запрос.
        transcribe:           Транскрибировать видео-шаги.
        log_fn:               Callback для стриминга логов в front.py.

    Returns:
        Путь к папке сессии или None при критической ошибке.
    """
    def _log(msg: str):
        print(msg)
        if log_fn:
            log_fn(msg)

    qs = clarifying_questions or []

    _log(f"\n{'='*60}")
    _log(f"  STAGE 1  |  Тема: «{topic}»")
    _log(f"{'='*60}")

    # ── Шаг 2: структура курса (9B + thinking) ────────────────────────
    _log("\n[1/7] Генерирую структуру курса (9B + thinking)...")
    course_structure = generate_course_structure(topic, qs, user_answers, log_fn)
    _unload_model(THINKING_MODEL, log_fn)

    # ── Шаг 3: запросы + теги (3B) ────────────────────────────────────
    _log("\n[2/7] Генерирую поисковые запросы и теги (3B)...")
    setup   = generate_search_setup(topic, course_structure, user_answers, log_fn)
    queries = setup.get("search_queries", [topic])
    tags    = setup.get("tags", [topic])
    _unload_model(MAIN_MODEL, log_fn)

    _log(f"  Запросов: {len(queries)} | Тегов: {len(tags)}")

    # ── Создаём сессию ────────────────────────────────────────────────
    session_id, session_dir = create_session(
        topic, tags, course_structure, user_answers, qs
    )
    _log(f"\n📁 Сессия: {session_id}")

    # ── Авторизация Stepik ───────────────────────────────────────────
    _log("\n[3/7] Авторизация Stepik...")
    try:
        loader = StepikCourseLoader()
    except Exception as e:
        _log(f"❌ Stepik ошибка авторизации: {e}")
        return None

    # ── Шаг 4a: поиск курсов ─────────────────────────────────────────
    _log(f"\n[4/7] Поиск курсов ({len(queries)} запросов)...")
    courses = search_stepik(loader, queries, limit_per_query, log_fn)
    _log(f"  Найдено курсов: {len(courses)}")
    if not courses:
        _log("⚠ Курсы не найдены. Продолжаю с пустой базой.")

    # ── Шаг 4b: распределение уроков (4B) ────────────────────────────
    _log("\n[5/7] Распределение уроков по модулям (4B)...")
    lesson_module_map: Dict[int, int] = {}
    if courses:
        lesson_module_map = distribute_lessons_for_courses(
            loader, courses, course_structure, log_fn
        )
        _save_lesson_module_map(session_dir, lesson_module_map)
    _unload_model(DISTRIBUTION_MODEL, log_fn)

    # ── Шаг 5: скачивание ────────────────────────────────────────────
    _log(f"\n[6/7] Скачиваю уроки (топ-{max_courses} курсов)...")
    if courses:
        download_assigned_courses(
            loader, courses, session_dir,
            lesson_module_map, transcribe, max_courses, log_fn,
        )

    saved = len(_json_files(os.path.join(session_dir, "raw_data")))
    _log(f"  Уроков в базе: {saved}")

    # ── Шаги 6-7: два раунда оценки покрытия ─────────────────────────
    previous_assessment = ""

    for round_num in (1, 2):
        _log(f"\n[7/7] Оценка покрытия — раунд {round_num}/2 (9B + thinking)...")
        eval_result = evaluate_coverage(
            course_structure, session_dir, round_num, previous_assessment, log_fn
        )
        _save_coverage_report(session_dir, round_num, eval_result)
        _unload_model(THINKING_MODEL, log_fn)

        previous_assessment = eval_result.get("assessment", "")
        add_queries         = eval_result.get("additional_queries", [])
        score               = eval_result.get("coverage_score", 1.0)

        if not add_queries:
            _log(f"  Дополнительных запросов нет (score={score:.2f}) — раунд {round_num} завершён")
            continue

        _log(
            f"  score={score:.2f} | Доп. запросы ({len(add_queries)}): "
            f"{' | '.join(add_queries[:3])}{'...' if len(add_queries) > 3 else ''}"
        )

        # Доп. поиск
        _log(f"  Ищу дополнительные курсы...")
        extra_courses = search_stepik(loader, add_queries, limit_per_query, log_fn)
        _log(f"  Найдено доп. курсов: {len(extra_courses)}")

        if extra_courses:
            # Доп. распределение (4B)
            _log(f"  Распределяю уроки доп. курсов (4B)...")
            extra_map = distribute_lessons_for_courses(
                loader, extra_courses, course_structure, log_fn
            )
            lesson_module_map.update(extra_map)
            _save_lesson_module_map(session_dir, lesson_module_map)
            _unload_model(DISTRIBUTION_MODEL, log_fn)

            # Доп. скачивание
            _log(f"  Скачиваю доп. уроки...")
            download_assigned_courses(
                loader, extra_courses, session_dir,
                lesson_module_map, transcribe, max_courses, log_fn,
            )

        saved = len(_json_files(os.path.join(session_dir, "raw_data")))
        _log(f"  Уроков в базе после раунда {round_num}: {saved}")

    # ── Финальный отчёт ───────────────────────────────────────────────
    final_count = len(_json_files(os.path.join(session_dir, "raw_data")))
    _log(f"\n{'='*60}")
    _log(f"  ✅ STAGE 1 ЗАВЕРШЁН")
    _log(f"  Уроков скачано: {final_count}")
    _log(f"  Модулей курса:  {len(course_structure.get('modules', []))}")
    _log(f"  Сессия:         {session_dir}")
    _log(f"{'='*60}\n")

    return session_dir


# ════════════════════════════════════════════════════════════════════════════
#  CLI (совместимость с main.py)
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
    print("\n[AI] Уточняющие вопросы:")
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
        topic                = topic,
        user_answers         = "\n".join(lines),
        clarifying_questions = questions,
        max_courses          = max_courses,
        limit_per_query      = limit_per_query,
        transcribe           = transcribe,
    )