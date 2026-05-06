"""
schemas.py
──────────
JSON-схемы для структурированного вывода LLM.
Карта тегов — плоская (список строк, не словарь категорий).
"""

from typing import Any
Schema = dict[str, Any]


# ════════════════════════════════════════════════════════════════════════════
#  RAG-задачи
# ════════════════════════════════════════════════════════════════════════════

TAGGING: Schema = {
    "type": "object",
    "properties": {
        "tags": {"type": "array", "items": {"type": "string"}},
        "primary_tag": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["tags", "primary_tag"],
}

SUMMARY: Schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "topics": {"type": "array", "items": {"type": "string"}},
        "difficulty_level": {
            "type": "string",
            "enum": ["beginner", "intermediate", "advanced"],
        },
        "language": {"type": "string"},
    },
    "required": ["summary", "key_points", "topics"],
}

LESSON_MERGE: Schema = {
    "type": "object",
    "properties": {
        "merged_title": {"type": "string"},
        "merged_summary": {"type": "string"},
        "merged_key_points": {"type": "array", "items": {"type": "string"}},
        "merged_tags": {"type": "array", "items": {"type": "string"}},
        "topics_covered": {"type": "array", "items": {"type": "string"}},
        "source_lessons_count": {"type": "integer"},
        "duplicate_content_found": {"type": "boolean"},
    },
    "required": ["merged_title", "merged_summary", "merged_key_points", "merged_tags"],
}


# ════════════════════════════════════════════════════════════════════════════
#  PIPELINE Stage 1
# ════════════════════════════════════════════════════════════════════════════

CLARIFYING_QUESTIONS: Schema = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3–5 уточняющих вопросов на русском языке.",
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
            "description": "3 поисковых запроса для Stepik.",
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Плоский список тегов (40–70 штук): конкретные термины, "
                "названия библиотек, алгоритмов, концепций. "
                "Например: ['TensorFlow', 'PyTorch', 'градиентный спуск', ...]"
            ),
        },
    },
    "required": ["search_queries", "tags"],
}

COURSE_RANKING: Schema = {
    "type": "object",
    "properties": {
        "rankings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {"type": "integer", "minimum": 1, "maximum": 10},
                    "reason": {"type": "string"},
                },
                "required": ["id", "score", "reason"],
            },
        }
    },
    "required": ["rankings"],
}


# ════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — кластеризация, чанки, структура курса
# ════════════════════════════════════════════════════════════════════════════

CLUSTER_MERGE_DECISION: Schema = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Индексы уроков (0-based) для объединения.",
                    },
                    "reason": {"type": "string"},
                },
                "required": ["indices"],
            },
        }
    },
    "required": ["groups"],
}

CHUNK_GENERATION: Schema = {
    "type": "object",
    "properties": {
        "final_title": {
            "type": "string",
            "description": "Краткое итоговое название блока знаний.",
        },
        "summary": {
            "type": "string",
            "description": "3–5 предложений: ключевые понятия и взаимосвязи.",
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "5–15 конкретных тегов из проектного списка.",
        },
        "merged_text": {
            "type": "string",
            "description": "Глубокий структурированный конспект материала.",
        },
    },
    "required": ["final_title", "summary", "tags", "merged_text"],
}

COURSE_STRUCTURE: Schema = {
    "type": "object",
    "properties": {
        "course_title": {"type": "string"},
        "modules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "query_texts": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Ровно 3 поисковых запроса.",
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["title", "query_texts", "tags"],
                        },
                    },
                },
                "required": ["title", "steps"],
            },
        },
    },
    "required": ["course_title", "modules"],
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
    "tagging":                TAGGING,
    "summary":                SUMMARY,
    "lesson_merge":           LESSON_MERGE,
    "clarifying_questions":   CLARIFYING_QUESTIONS,
    "pipeline_setup":         PIPELINE_SETUP,
    "course_ranking":         COURSE_RANKING,
    "cluster_merge_decision": CLUSTER_MERGE_DECISION,
    "chunk_generation":       CHUNK_GENERATION,
    "course_structure":       COURSE_STRUCTURE,
    "text":                   TEXT_RESPONSE,
    "classification":         CLASSIFICATION,
    "ner":                    NER,
    "sentiment":              SENTIMENT,
    "qa":                     QA,
    "transcription":          TRANSCRIPTION,
}

SCHEMA_DESCRIPTIONS: dict[str, str] = {
    "tagging":                "Тегирование: список тегов + главный тег",
    "summary":                "Суммаризация: резюме + тезисы + темы",
    "lesson_merge":           "Объединение уроков",
    "clarifying_questions":   "Stage1: уточняющие вопросы",
    "pipeline_setup":         "Stage1: запросы + плоский список тегов",
    "course_ranking":         "Stage1: ранжирование курсов",
    "cluster_merge_decision": "Stage2: решение об объединении уроков в кластере",
    "chunk_generation":       "Stage2: генерация чанка (title, summary, tags, text)",
    "course_structure":       "Stage2: структура курса (модули, шаги, запросы, теги)",
    "text":                   "Простой текстовый ответ",
    "classification":         "Классификация",
    "ner":                    "Именованные сущности",
    "sentiment":              "Тональность",
    "qa":                     "Вопрос-ответ",
    "transcription":          "Транскрипция",
}


def get_schema(name: str) -> Schema:
    if name not in SCHEMAS:
        raise ValueError(f"Схема '{name}' не найдена. Доступные: {list(SCHEMAS.keys())}")
    return SCHEMAS[name]


def list_schemas() -> list[dict]:
    return [
        {"name": n, "description": SCHEMA_DESCRIPTIONS.get(n, "")}
        for n in SCHEMAS
    ]