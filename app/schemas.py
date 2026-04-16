"""
schemas.py
──────────
JSON-схемы для структурированного вывода LLM и Gemini.
"""

from typing import Any

Schema = dict[str, Any]


# ════════════════════════════════════════════════════════════════════════════
#  RAG-задачи
# ════════════════════════════════════════════════════════════════════════════

TAGGING: Schema = {
    "type": "object",
    "properties": {
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Список тегов/ключевых слов для данного текста.",
        },
        "primary_tag": {
            "type": "string",
            "description": "Главный тег (наиболее релевантный).",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Уверенность в наборе тегов.",
        },
    },
    "required": ["tags", "primary_tag"],
}

SUMMARY: Schema = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "Краткое резюме (2–5 предложений).",
        },
        "key_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Ключевые тезисы (3–7 пунктов).",
        },
        "topics": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Основные темы текста.",
        },
        "difficulty_level": {
            "type": "string",
            "enum": ["beginner", "intermediate", "advanced"],
            "description": "Уровень сложности материала.",
        },
        "language": {
            "type": "string",
            "description": "Язык текста (ISO 639-1).",
        },
    },
    "required": ["summary", "key_points", "topics"],
}

LESSON_MERGE: Schema = {
    "type": "object",
    "properties": {
        "merged_title": {"type": "string"},
        "merged_summary": {"type": "string"},
        "merged_key_points": {
            "type": "array",
            "items": {"type": "string"},
        },
        "merged_tags": {
            "type": "array",
            "items": {"type": "string"},
        },
        "topics_covered": {
            "type": "array",
            "items": {"type": "string"},
        },
        "source_lessons_count": {"type": "integer"},
        "duplicate_content_found": {"type": "boolean"},
    },
    "required": ["merged_title", "merged_summary", "merged_key_points", "merged_tags"],
}


# ════════════════════════════════════════════════════════════════════════════
#  PIPELINE-схемы (для интерактивного пайплайна поиска курсов)
# ════════════════════════════════════════════════════════════════════════════

CLARIFYING_QUESTIONS: Schema = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3–5 уточняющих вопросов пользователю на русском языке.",
        }
    },
    "required": ["questions"],
}

PIPELINE_SETUP: Schema = {
    "type": "object",
    "properties": {
        "search_queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Ровно 3 поисковых запроса для платформы Stepik на русском языке.",
        },
        "tag_map": {
            "type": "object",
            "description": "Карта тегов по категориям. Ключи — категории, значения — списки тегов.",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    },
    "required": ["search_queries", "tag_map"],
}

COURSE_RANKING: Schema = {
    "type": "object",
    "properties": {
        "rankings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "ID курса из входного списка.",
                    },
                    "score": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Оценка релевантности от 1 до 10.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Краткое обоснование оценки.",
                    },
                },
                "required": ["id", "score", "reason"],
            },
        }
    },
    "required": ["rankings"],
}


# ════════════════════════════════════════════════════════════════════════════
#  Общие схемы
# ════════════════════════════════════════════════════════════════════════════

TEXT_RESPONSE: Schema = {
    "type": "object",
    "properties": {
        "response": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["response"],
}

CLASSIFICATION: Schema = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "labels": {"type": "array", "items": {"type": "string"}},
        "explanation": {"type": "string"},
    },
    "required": ["label", "labels"],
}

NER: Schema = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["PERSON", "ORG", "LOC", "DATE", "MONEY", "PRODUCT", "OTHER"],
                    },
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                },
                "required": ["text", "type"],
            },
        }
    },
    "required": ["entities"],
}

SENTIMENT: Schema = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral", "mixed"],
        },
        "score": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "aspects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "aspect": {"type": "string"},
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"],
                    },
                },
                "required": ["aspect", "sentiment"],
            },
        },
    },
    "required": ["sentiment", "score"],
}

QA: Schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "source_excerpt": {"type": "string"},
        "is_answerable": {"type": "boolean"},
    },
    "required": ["answer", "is_answerable"],
}

TRANSCRIPTION: Schema = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "language": {"type": "string"},
        "duration_seconds": {"type": "number"},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "text": {"type": "string"},
                },
                "required": ["start", "end", "text"],
            },
        },
    },
    "required": ["text", "language"],
}


# ════════════════════════════════════════════════════════════════════════════
#  Реестр
# ════════════════════════════════════════════════════════════════════════════

SCHEMAS: dict[str, Schema] = {
    # RAG-задачи
    "tagging":               TAGGING,
    "summary":               SUMMARY,
    "lesson_merge":          LESSON_MERGE,
    # Pipeline
    "clarifying_questions":  CLARIFYING_QUESTIONS,
    "pipeline_setup":        PIPELINE_SETUP,
    "course_ranking":        COURSE_RANKING,
    # Общие
    "text":                  TEXT_RESPONSE,
    "classification":        CLASSIFICATION,
    "ner":                   NER,
    "sentiment":             SENTIMENT,
    "qa":                    QA,
    "transcription":         TRANSCRIPTION,
}

SCHEMA_DESCRIPTIONS: dict[str, str] = {
    "tagging":               "Тегирование текста: список тегов + главный тег",
    "summary":               "Суммаризация: резюме + ключевые тезисы + темы + уровень сложности",
    "lesson_merge":          "Объединение уроков: единое резюме, теги, ключевые тезисы",
    "clarifying_questions":  "Pipeline: уточняющие вопросы к пользователю",
    "pipeline_setup":        "Pipeline: 3 поисковых запроса + карта тегов",
    "course_ranking":        "Pipeline: ранжирование курсов по релевантности",
    "text":                  "Простой текстовый ответ",
    "classification":        "Классификация текста",
    "ner":                   "Извлечение именованных сущностей",
    "sentiment":             "Анализ тональности",
    "qa":                    "Вопрос-ответ с источником",
    "transcription":         "Транскрипция с сегментами",
}


def get_schema(name: str) -> Schema:
    if name not in SCHEMAS:
        raise ValueError(
            f"Схема '{name}' не найдена. Доступные: {list(SCHEMAS.keys())}"
        )
    return SCHEMAS[name]


def list_schemas() -> list[dict]:
    return [
        {"name": name, "description": SCHEMA_DESCRIPTIONS.get(name, "")}
        for name in SCHEMAS
    ]