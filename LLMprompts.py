"""
LLMprompts.py
─────────────
Все системные промпты для LLM в одном месте.

Ключевые изменения в Stage 2:

cluster_merge_decision():
  Теперь принимает module_info и course_title — LLM принимает решение
  о слиянии уроков с пониманием целей модуля и его места в курсе.

chunk_generation():
  Добавлен delta-контекст:
    previously_covered_in_module  — что уже объяснено в текущем модуле
    previously_covered_in_course  — что изучено в предыдущих модулях
    module_info                   — цели и темы текущего модуля
  Модель генерирует learned_concepts — список понятий, которые освоит
  читатель, для передачи следующему чанку.
"""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any


class PromptBank:

    # ══════════════════════════════════════════════════════════════════════
    #  STAGE 1 — промпты без изменений
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def clarifying_questions(topic: str) -> str:
        return dedent(f"""
        Ты — IT-методист, проектирующий персональный образовательный трек.
        Пользователь хочет создать учебный курс по теме: "{topic}".
        Сгенерируй 3–5 уточняющих вопроса на русском языке.
        ЖЁСТКИЕ ПРАВИЛА:
        1. Спрашивай только о конкретных технологиях, фреймворках, уровне знаний
           и узкой предметной области — НЕ о формате курса.
        2. Каждый вопрос должен провоцировать пользователя назвать конкретный
           термин (название библиотеки, алгоритма, задачи), который войдёт в теги.
        Верни ТОЛЬКО валидный JSON по схеме clarifying_questions.
        """).strip()

    @staticmethod
    def pipeline_setup(topic: str, user_answers: str) -> str:
        return dedent(f"""
        Ты — AI-ассистент, формирующий метаданные для RAG-системы поиска по курсам Stepik.
        Тема курса: "{topic}"
        Ответы пользователя: {user_answers}
        Создай 3 поисковых запроса для Stepik и плоский список тегов (40–70 штук).
        Теги — конкретные термины: «TensorFlow», «градиентный спуск».
        ЗАПРЕЩЕНО: «обучение», «курс», «урок».
        Верни ТОЛЬКО валидный JSON по схеме pipeline_setup.
        """).strip()

    @staticmethod
    def course_ranking(topic: str, user_context: str, courses: list[dict]) -> str:
        return dedent(f"""
        Ты — эксперт по оценке образовательного контента на платформе Stepik.
        Тема: "{topic}". Контекст пользователя: {user_context}
        Оцени каждый курс по шкале 1–10 (главный критерий — соответствие теме).
        Нерелевантный курс ставь ≤ 3. Оцени КАЖДЫЙ курс.
        Курсы: {json.dumps(courses, ensure_ascii=False, indent=2)}
        Верни ТОЛЬКО валидный JSON по схеме course_ranking.
        """).strip()

    @staticmethod
    def course_structure_thinking(
        topic: str,
        clarifying_questions: list[str],
        user_answers: str,
    ) -> str:
        q_block = "\n".join(
            f"  {i+1}. {q}" for i, q in enumerate(clarifying_questions)
        ) if clarifying_questions else "  (не задавались)"

        return dedent(f"""
        Ты — опытный методист. Спроектируй детальную структуру учебного курса.

        ТЕМА: «{topic}»
        ВОПРОСЫ ПОЛЬЗОВАТЕЛЮ:
        {q_block}
        ОТВЕТЫ: {user_answers}

        Создай 5–8 логических модулей (от основ к продвинутым темам).
        Для каждого модуля: id (0-based), title, description (3–5 предл.),
        goals (2–4 цели), key_topics (8–12 конкретных термина).
        Для курса: course_title, course_description, course_goals.

        Рассуждай структурированно, но кратко — не более 1200 токенов на анализ.
        Как только план готов — сразу переходи к JSON без повторений.
        Верни ТОЛЬКО валидный JSON по схеме course_structure_detailed.
        """).strip()

    @staticmethod
    def search_setup_from_structure(
        topic: str,
        course_structure: dict,
        user_answers: str,
    ) -> str:
        modules_summary = "\n".join(
            f"  Модуль {m['id']}: «{m['title']}» — {', '.join(m.get('key_topics', [])[:6])}"
            for m in course_structure.get("modules", [])
        ) or "  (нет данных)"

        return dedent(f"""
        ТЕМА: «{topic}»
        СТРУКТУРА КУРСА:
        {modules_summary}
        ОТВЕТЫ ПОЛЬЗОВАТЕЛЯ: {user_answers}

        Сгенерируй ровно 5 поисковых запросов для Stepik (поиск по курсам, не по урокам)
        и плоский список тегов (40–70) для кластеризации уроков в Stage 2.
        Верни ТОЛЬКО валидный JSON по схеме pipeline_setup.
        """).strip()

    @staticmethod
    def lesson_distribution(
        modules: list[dict],
        lessons: list[dict],
        course_title: str = "",
    ) -> str:
        mod_lines = "\n\n".join(
            f"МОДУЛЬ {m['id']}: «{m['title']}»\n"
            f"  Описание: {m.get('description', '')[:180]}\n"
            f"  Темы: {', '.join(m.get('key_topics', [])[:8])}"
            for m in modules
        )
        lesson_lines = "\n".join(
            f"  [{l.get('lesson_id', 0)}] [{l.get('section_title', '')}] {l.get('title', '')}"
            for l in lessons
        )
        return dedent(f"""
        Распредели уроки курса «{course_title}» по модулям программы.

        МОДУЛИ:
        {mod_lines}

        УРОКИ ([lesson_id] [Раздел] Название):
        {lesson_lines}

        Каждый урок → один module_id или в unassigned.
        Верни ТОЛЬКО валидный JSON по схеме lesson_distribution.
        Обработай ВСЕ {len(lessons)} уроков.
        """).strip()

    @staticmethod
    def coverage_evaluation(
        course_structure: dict,
        lessons_by_module: dict,
        round_num: int = 1,
        previous_assessment: str = "",
    ) -> str:
        course_goals_text = "\n".join(
            f"  - {g}" for g in course_structure.get("course_goals", [])
        ) or "  (не указаны)"

        module_blocks = []
        for m in course_structure.get("modules", []):
            mid = m["id"]
            lesson_titles = lessons_by_module.get(mid, [])
            n = len(lesson_titles)
            shown = "\n".join(f"      • {t}" for t in lesson_titles[:25])
            if n > 25:
                shown += f"\n      ... и ещё {n - 25}"
            if not shown:
                shown = "      ⚠ уроки не найдены"
            module_blocks.append(
                f"МОДУЛЬ {mid}: «{m.get('title', '')}» ({n} уроков)\n"
                f"  Темы: {', '.join(m.get('key_topics', []))}\n"
                f"  Уроки:\n{shown}"
            )

        prev_block = (
            f"ПРЕДЫДУЩАЯ ОЦЕНКА (раунд {round_num-1}):\n{previous_assessment}\n\n"
            if previous_assessment and round_num > 1 else ""
        )
        total = sum(len(v) for v in lessons_by_module.values())

        return dedent(f"""
        Оцени полноту покрытия курса (раунд {round_num}/2).
        Курс: «{course_structure.get("course_title", "")}»
        Цели: {course_goals_text}
        Всего уроков: {total}

        {prev_block}
        {chr(10).join(module_blocks)}

        Определи coverage_score (0–1), status каждого модуля (good/partial/poor),
        overall_missing_topics и additional_queries (3–7 запросов для Stepik по курсам,
        на русском, фокус на poor/partial модулях).
        Рассуждай кратко и по существу — не более 800 токенов на анализ.
        Как только картина ясна — сразу переходи к JSON.
        Верни ТОЛЬКО валидный JSON по схеме coverage_evaluation.
        """).strip()

    # ══════════════════════════════════════════════════════════════════════
    #  STAGE 2 — ОБНОВЛЁННЫЕ ПРОМПТЫ
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def cluster_merge_decision(
        lesson_titles: list[str],
        module_info: dict | None = None,
        course_title: str = "",
    ) -> str:
        """
        Решение об объединении уроков внутри кластера.

        Теперь LLM видит контекст модуля: его цели и ключевые темы.
        Это позволяет принимать решения о слиянии осознанно —
        понимая, что модуль должен в итоге охватить.
        """
        numbered = "\n".join(f"  {i}. {t}" for i, t in enumerate(lesson_titles))

        if module_info:
            goals_str  = "; ".join(module_info.get("goals", [])[:3])
            topics_str = ", ".join(module_info.get("key_topics", [])[:10])
            context_block = dedent(f"""
            КОНТЕКСТ:
            Курс: «{course_title}»
            Модуль: «{module_info.get("title", "")}»
            Описание: {module_info.get("description", "")[:220]}
            Цели модуля: {goals_str}
            Ключевые темы: {topics_str}
            """).strip()
        else:
            context_block = ""

        return dedent(f"""
        {context_block}

        УРОКИ ДЛЯ ГРУППИРОВКИ (все из одного подкластера модуля):
        {numbered}

        ЗАДАЧА:
        Определи, какие уроки объединить в один конспект.

        КРИТЕРИИ ОБЪЕДИНЕНИЯ (в порядке приоритета):
        1. Урок-теория + его практика/упражнения → ОБЪЕДИНЯЙ
        2. Уроки, раскрывающие одну концепцию с разных сторон → ОБЪЕДИНЯЙ
        3. Последовательные уроки одной мини-темы → ОБЪЕДИНЯЙ
        4. Уроки разных аспектов модуля (разные концепции) → РАЗДЕЛЯЙ

        ВАЖНО: учитывай цели модуля при группировке — каждая группа должна
        формировать законченный смысловой блок в рамках модуля.

        Каждый индекс (0..{len(lesson_titles)-1}) должен быть ровно в одной группе.
        Верни ТОЛЬКО валидный JSON по схеме cluster_merge_decision.
        """).strip()

    @staticmethod
    def chunk_generation(
        titles: list[str],
        known_tags: list[str],
        combined_text: str,
        # ── Delta-контекст ──────────────────────────────────────────────
        module_info: dict | None = None,
        course_title: str = "",
        previously_covered_in_module: list[str] | None = None,
        cross_module_context: list[dict] | None = None,
    ) -> str:
        """
        Генерация чанка знаний с delta-контекстом.

        Args:
            module_info:                    Описание и цели текущего модуля.
            course_title:                   Название курса.
            previously_covered_in_module:   Концепции, уже объяснённые в этом модуле
                                            (из learned_concepts предыдущих чанков).
            cross_module_context:           Контекст предыдущих модулей.
                                            Формат: [{"title": "...", "concepts": [...]}]
        """
        tags_hint = ", ".join(known_tags[:60]) or "(пусто)"

        # ── Блок контекста модуля ────────────────────────────────────
        if module_info:
            goals_str = "\n".join(f"    • {g}" for g in module_info.get("goals", []))
            module_block = dedent(f"""
            ═══ КОНТЕКСТ КУРСА ═══
            Курс: «{course_title}»
            Текущий модуль: «{module_info.get("title", "")}»
            Описание модуля: {module_info.get("description", "")[:300]}
            Цели модуля:
            {goals_str}
            ═════════════════════
            """).strip()
        else:
            module_block = ""

        # ── Блок межмодульного контекста ─────────────────────────────
        if cross_module_context:
            prev_mod_lines = []
            # Показываем последние 3 модуля подробно, остальные — только названия
            recent = cross_module_context[-3:]
            older  = cross_module_context[:-3]
            if older:
                older_titles = ", ".join(f"«{m['title']}»" for m in older)
                prev_mod_lines.append(f"  [Ранее пройдено: {older_titles}]")
            for mod in recent:
                concepts = mod.get("concepts", [])[:12]
                c_str = "; ".join(concepts) if concepts else "—"
                prev_mod_lines.append(
                    f"  [{mod['title']}]\n    Освоено: {c_str}"
                )
            cross_block = (
                "УЖЕ ИЗУЧЕНО В ПРЕДЫДУЩИХ МОДУЛЯХ (не объяснять заново, "
                "можно ссылаться):\n" + "\n".join(prev_mod_lines)
            )
        else:
            cross_block = ""

        # ── Блок внутримодульного контекста ──────────────────────────
        if previously_covered_in_module:
            covered_str = "\n".join(
                f"  • {c}" for c in previously_covered_in_module[:20]
            )
            module_covered_block = (
                "УЖЕ ОБЪЯСНЕНО В ЭТОМ МОДУЛЕ (НЕ ПОВТОРЯТЬ — "
                "читатель это знает):\n" + covered_str
            )
        else:
            module_covered_block = ""

        # ── Инструкция по merged_text ─────────────────────────────────
        if previously_covered_in_module or cross_module_context:
            text_instruction = dedent("""
            ТРЕБОВАНИЯ К merged_text:
            • НЕ повторяй концепции из «уже объяснено» — переходи сразу к новому
            • Если нужно ссылаться на ранее изученное: «как мы уже рассмотрели...»
              или «опираясь на знание X...» — одной фразой, без развёртывания
            • Фокус ТОЛЬКО на том, что читатель узнаёт впервые
            • Сохраняй определения, формулы, примеры кода, конкретные свойства
            • Структурируй для лёгкого извлечения фактов LLM
            • Убери воду, приветствия, повторяющийся материал между уроками
            """).strip()
        else:
            text_instruction = dedent("""
            ТРЕБОВАНИЯ К merged_text:
            • Сохраняй определения, формулы, примеры кода, конкретные свойства
            • Структурируй для лёгкого извлечения фактов LLM
            • Убери воду, приветствия, повторяющийся материал между уроками
            """).strip()

        return dedent(f"""
        Создай структурированный блок знаний из учебных уроков.

        {module_block}

        {cross_block}

        {module_covered_block}

        УРОКИ ДЛЯ КОНСПЕКТИРОВАНИЯ: {", ".join(titles)}

        Известные теги проекта (использовать в первую очередь):
        {tags_hint}

        Текст уроков:
        {combined_text}

        СТРУКТУРА ОТВЕТА:

        1. final_title — краткое название блока (≤ 10 слов)

        2. summary — 3–5 предложений:
           • О чём этот блок и почему он важен в контексте модуля
           • Какую проблему решает или какой навык формирует

        3. tags — 5–15 тегов. Приоритет — из известных тегов проекта.
           Новые теги только если это конкретный термин, отсутствующий в списке.

        4. merged_text — конспект (см. требования выше)

        5. learned_concepts — ОБЯЗАТЕЛЬНО — 5–10 конкретных понятий/навыков,
           которые читатель освоит ВПЕРВЫЕ из этого блока.
           Эти данные будут переданы следующему блоку как «уже изучено».
           Формат: конкретные термины («метод Adam», «learning rate scheduling»),
           НЕ абстрактные категории («оптимизация нейросетей»).

        6. assumed_knowledge — что читатель должен знать ДО этого блока
           (может быть пустым для первого блока модуля)

        Верни ТОЛЬКО валидный JSON по схеме chunk_generation.
        """).strip()

    @staticmethod
    def course_structure(
        topic: str,
        chunk_summaries: list[dict],
        tag_sample: list[str],
    ) -> str:
        return dedent(f"""
        Создай структуру образовательного курса по теме: "{topic}"

        Доступные блоки знаний:
        {json.dumps(chunk_summaries, ensure_ascii=False, indent=2)}

        Теги проекта: {", ".join(tag_sample)}

        ТРЕБОВАНИЯ: 3–6 модулей, в каждом 3–7 шагов.
        Каждый шаг: title, query_texts (РОВНО 3), tags (3-8).
        Верни ТОЛЬКО валидный JSON по схеме course_structure.
        """).strip()

    @staticmethod
    def rag_answer(question: str, context: str) -> str:
        return dedent(f"""
        Ты — образовательный ассистент. Ответь на вопрос, используя ТОЛЬКО контекст.

        Контекст:
        {context}

        Вопрос: {question}

        Если ответ есть в контексте — отвечай точно. Не придумывай.
        """).strip()


TASK_TO_SCHEMA = {
    "clarifying_questions":        "clarifying_questions",
    "pipeline_setup":              "pipeline_setup",
    "course_ranking":              "course_ranking",
    "course_structure_thinking":   "course_structure_detailed",
    "search_setup_from_structure": "pipeline_setup",
    "lesson_distribution":         "lesson_distribution",
    "coverage_evaluation":         "coverage_evaluation",
    "cluster_merge_decision":      "cluster_merge_decision",
    "chunk_generation":            "chunk_generation",
    "course_structure":            "course_structure",
}