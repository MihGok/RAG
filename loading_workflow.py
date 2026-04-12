import sys
import os
import requests
from tqdm import tqdm
from typing import List, Dict, Any


from CourseProcessor.CourseLoader import StepikCourseLoader



def _chunk_list(lst, n):
    """Разбивает список на части по n элементов."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def _analyze_batch(session: requests.Session, courses_chunk: List[Dict], topic: str, llm_endpoint: str) -> List[Dict]:
    """Внутренняя функция для отправки одного батча в LLM."""
    prompt = build_course_analysis_prompt(topic, courses_chunk)
    
    payload = {
        "prompt": prompt,
        "response_schema": COURSE_ANALYSIS_SCHEMA,
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.9,
        "n_ctx": 2048,        
        "n_gpu_layers": -1 
    }

    try:
        response = session.post(llm_endpoint, json=payload, timeout=240)
        response.raise_for_status()
        res_data = response.json()
        
        if res_data.get("success") and "json" in res_data:
            parsed = res_data["json"]
            # Защита от разных форматов ответа
            results = parsed if isinstance(parsed, list) else parsed.get("results", [])
            
            # Синхронизация ID (защита от галлюцинаций модели по ID)
            for i, res in enumerate(results):
                if i < len(courses_chunk):
                    res['course_id'] = courses_chunk[i].get('id')
                    # Сохраняем оригинальное название для надежности
                    res['course_title'] = courses_chunk[i].get('title') 
            return results
            
    except Exception as e:
        print(f"\n[LLM Error] Ошибка батча: {e}")
    
    return []


# def _analyze_lesson_batch(session: requests.Session, topic: str, course_title: str, lessons_chunk: List[Dict], llm_endpoint: str) -> List[Dict]:
#     """Отправляет батч уроков в LLM."""
#     prompt = build_lesson_analysis_prompt(topic, course_title, lessons_chunk)
    
#     payload = {
#         "prompt": prompt,
#         "response_schema": LESSON_ANALYSIS_SCHEMA,
#         "max_tokens": 1024, 
#         "temperature": 0.1,
#         "top_p": 0.9,
#         "n_ctx": 2048,
#         "n_gpu_layers": -1 
#     }

#     try:
#         response = session.post(llm_endpoint, json=payload, timeout=300)
#         response.raise_for_status()
#         res_data = response.json()
        
#         if res_data.get("success") and "json" in res_data:
#             parsed = res_data["json"]
#             # Схема возвращает объект {"lessons": [...]}, извлекаем список
#             if isinstance(parsed, dict):
#                 return parsed.get("lessons", [])
#             elif isinstance(parsed, list):
#                 return parsed
            
#     except Exception as e:
#         print(f"   [Lesson LLM Error] {e}")
    
#     return []


# def filter_course_content(
#     loader: StepikCourseLoader, 
#     course_obj: Dict, 
#     topic: str, 
#     llm_endpoint: str
# ) -> List[int]:
#     """
#     1. Получает структуру уроков.
#     2. Прогоняет через LLM.
#     3. Возвращает список ID уроков, которые нужно скачать.
#     """
#     course_id = course_obj['id']
#     course_title = course_obj['title']
    
    
#     lessons_metadata = loader.get_course_outline(course_obj)
#     if not lessons_metadata:
#         print(f"   [WARN] В курсе {course_id} не найдено уроков.")
#         return []

#     print(f"   [AI] Анализ {len(lessons_metadata)} уроков на полезность...")
    
#     # 2. Батчинг и отправка в LLM
#     session = ProxyConfig.get_session_with_proxy(use_proxy=False)
#     approved_ids = []
    
#     # Размер батча для уроков можно сделать больше, т.к. там только заголовки
#     lesson_batch_size = 5 
#     chunks = list(_chunk_list(lessons_metadata, lesson_batch_size))
    
#     for chunk in tqdm(chunks, desc="   Фильтрация уроков", leave=False):
#         results = _analyze_lesson_batch(session, topic, course_title, chunk, llm_endpoint)
        
#         for res in results:
#             score = res.get('lesson_score', 0)
#             lid = res.get('lesson_id')
            
#             if score >= 5:
#                 approved_ids.append(lid)
#             else:
#                 # Можно раскомментировать для отладки, чтобы видеть, что отсеялось
#                 print(f"      [-] Отсеян урок {lid}: {res.get('lesson_title')} (Score: {score})")
#                 pass

#     print(f"   [RESULT] Одобрено {len(approved_ids)} из {len(lessons_metadata)} уроков.")
#     return approved_ids


def fetch_stepik_courses(topic: str, limit: int = 100) -> tuple[StepikCourseLoader, List[Dict]]:
    """
    Ищет курсы на Stepik и загружает их метаданные.
    Возвращает экземпляр лоадера (чтобы не создавать заново) и список курсов.
    """
    print(f"[Stepik] Поиск курсов по теме: {topic}...")
    loader = StepikCourseLoader()
    
    # Получаем ID
    course_ids = loader.get_course_ids_by_query(query=topic, limit=limit)
    if not course_ids:
        print("Курсы не найдены.")
        return loader, []

    # Получаем полные объекты курсов
    raw_courses = loader.fetch_objects('courses', course_ids)
    print(f"[Stepik] Загружено метаданных: {len(raw_courses)}")
    
    return loader, raw_courses


def analyze_courses_relevance(
    raw_courses: List[Dict], 
    topic: str, 
    llm_endpoint: str, 
    batch_size: int = 10
) -> List[Dict]:
    """
    Прогоняет список курсов через LLM для оценки релевантности.
    Возвращает список проанализированных объектов (отсортированный по убыванию score).
    """
    if not raw_courses:
        return []

    # Сессия без прокси для локального Docker
    session = requests.Session()
    
    all_analyzed = []
    chunks = list(_chunk_list(raw_courses, batch_size))
    
    print(f"[AI] Анализ релевантности (всего {len(raw_courses)} курсов)...")
    
    for chunk in tqdm(chunks, desc="Обработка батчей LLM"):
        results = _analyze_batch(session, chunk, topic, llm_endpoint)
        all_analyzed.extend(results)

    # Сортировка
    all_analyzed.sort(key=lambda x: x.get('course_score', 0), reverse=True)
    return all_analyzed


def print_top_results(analyzed_courses: List[Dict], top_n: int = 20):
    """Выводит красивые результаты в консоль."""
    print("\n" + "="*60)
    print(f"ТОП-{top_n} РЕЛЕВАНТНЫХ КУРСОВ")
    print("="*60)
    
    for item in analyzed_courses[:top_n]:
        print(f"[{item.get('course_score', 0)}] {item.get('course_title')} (ID: {item.get('course_id')})")
        print(f"   Обоснование: {item.get('reasoning')}\n")


def download_top_courses(
    loader: StepikCourseLoader,
    analyzed_courses: List[Dict], 
    raw_courses: List[Dict], 
    min_score: int,
    topic: str,          
    llm_endpoint: str
):
    print("\n" + "="*60)
    print(f"УМНАЯ ЗАГРУЗКА КУРСОВ (Score > {min_score})")
    print("="*60)

    raw_courses_map = {c['id']: c for c in raw_courses}
    
    for item in analyzed_courses:
        score = item.get('course_score', 0)
        course_id = item.get('course_id')

        if score > min_score:
            print(f"\n[>>>] Обработка курса ID: {course_id} (Score: {score})")
            
            full_course_obj = raw_courses_map.get(course_id)
            if not full_course_obj:
                full_course_obj = loader.fetch_object_single('courses', course_id)

            if full_course_obj:
                try:
                    # ЭТАП 1: Анализ уроков
                    relevant_lesson_ids = filter_course_content(
                        loader, full_course_obj, topic, llm_endpoint
                    )
                    
                    if not relevant_lesson_ids:
                        print("   [SKIP] В курсе нет релевантных уроков после фильтрации.")
                        continue

                    # ЭТАП 2: Скачивание (передаем список разрешенных ID)
                    loader.process_course(full_course_obj, allowed_lesson_ids=relevant_lesson_ids)
                    
                except Exception as e:
                    print(f"[ERROR] Ошибка обработки {course_id}: {e}")