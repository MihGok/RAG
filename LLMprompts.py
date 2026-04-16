from __future__ import annotations

import json
from textwrap import dedent
from typing import Any


class PromptBank:
    @staticmethod
    def clarifying_questions(topic: str, user_goal: str = "", level_hint: str = "") -> str:
        goal_block = f"\nЦель пользователя: {user_goal}" if user_goal else ""
        level_block = f"\nПодсказка по уровню: {level_hint}" if level_hint else ""
        return dedent(f"""
        Пользователь хочет найти образовательные курсы по теме: "{topic}".
        {goal_block}{level_block}

        Сгенерируй 3–5 уточняющих вопросов на русском языке, чтобы понять:
        - уровень подготовки;
        - конкретные подтемы и инструменты;
        - цель обучения;
        - желаемый формат обучения.

        Требования:
        - вопросы должны быть конкретными;
        - не повторяй одну и ту же мысль;
        - избегай слишком общих формулировок;
        - вопросы должны помогать точнее подобрать курсы;
        - верни только JSON по схеме.
        """).strip()

    @staticmethod
    def pipeline_setup(topic: str, user_answers: str) -> str:
        return dedent(f"""
        Тема обучения: "{topic}"

        Ответы пользователя:
        {user_answers}

        На основе этой информации создай:

        1. РОВНО 3 поисковых запроса для Stepik (на русском языке).
           Каждый запрос должен быть отдельной формулировкой одной и той же темы:
           - широкий
           - точный
           - практико-ориентированный

        2. Подробную карту тегов для последующей классификации учебного материала.
           Разбей теги по категориям:
           - основные_концепции
           - инструменты_и_технологии
           - практические_навыки
           - смежные_темы
           - уровень_сложности

        Жёсткие правила:
        - запросы должны оставаться в рамках темы пользователя;
        - не подменяй тему слишком общими словами;
        - не добавляй нерелевантные области;
        - не дублируй одни и те же запросы;
        - верни только JSON по схеме.
        """).strip()

    @staticmethod
    def course_ranking(topic: str, user_context: str, courses: list[dict[str, Any]]) -> str:
        return dedent(f"""
        Тема поиска: "{topic}"

        Контекст пользователя:
        {user_context}

        Оцени релевантность каждого курса по шкале 1–10.

        Учитывай:
          - соответствие теме и уровню пользователя — главный критерий;
          - популярность курса и число слушателей — только вторичные сигналы;
          - объём материала;
          - не повышай нерелевантный курс только из-за популярности.

        Список курсов:
        {json.dumps(courses, ensure_ascii=False, indent=2)}

        ВАЖНО:
        - верни оценку для каждого курса;
        - не пропускай элементы;
        - если курс не по теме, ставь низкий балл;
        - пиши краткое обоснование одной фразой;
        - верни только JSON по схеме.
        """).strip()

    @staticmethod
    def lesson_merge(cluster_id: str, lesson_titles: list[str], short_snippets: list[str] | None = None) -> str:
        snippets_block = ""
        if short_snippets:
            snippets_block = "\nФрагменты уроков:\n" + "\n".join(
                f"- {t}: {s}" for t, s in zip(lesson_titles, short_snippets)
            )

        return dedent(f"""
        Перед тобой кластер уроков с похожими названиями.

        Задача:
        - объединить уроки в один канонический урок;
        - убрать повторения;
        - сохранить все уникальные и важные аспекты;
        - не добавлять лишние темы;
        - не выходить за пределы кластера.

        cluster_id = {cluster_id}
        titles = {json.dumps(lesson_titles, ensure_ascii=False)}{snippets_block}

        Верни:
        - merged_title
        - merged_summary
        - merged_key_points
        - merged_tags
        - topics_covered
        - source_lessons_count
        - duplicate_content_found

        Пиши кратко и без лишних объяснений. Верни только JSON по схеме.
        """).strip()

    @staticmethod
    def lesson_text_merge(title: str, lesson_texts: list[str], lesson_ids: list[str] | None = None) -> str:
        ids_block = f"\nlesson_ids = {json.dumps(lesson_ids, ensure_ascii=False)}" if lesson_ids else ""
        texts_block = "\n".join(f"### Урок {i+1}\n{text}" for i, text in enumerate(lesson_texts))

        return dedent(f"""
        Объедини тексты нескольких уроков в один компактный и структурированный конспект.

        Тема/заголовок кластера: "{title}"{ids_block}

        Требования:
        - убрать повторения;
        - сохранить определения и ключевые этапы процесса;
        - не добавлять лишние утверждения;
        - не терять смысл исходных уроков;
        - результат должен быть пригоден для RAG-чанка.

        Тексты:
        {texts_block}

        Верни JSON по схеме summary:
        - summary
        - key_points
        - topics
        - difficulty_level
        - language

        Если уровень неочевиден, выбирай наиболее вероятный.
        """).strip()


TASK_TO_SCHEMA = {
    "clarifying_questions": "clarifying_questions",
    "pipeline_setup": "pipeline_setup",
    "course_ranking": "course_ranking",
    "lesson_merge": "lesson_merge",
    "lesson_text_merge": "summary",
}
