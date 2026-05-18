"""
schemas.py
──────────
JSON-схемы для структурированного вывода LLM.

Изменения:
  CHUNK_GENERATION — добавлены поля для delta-связности:
    learned_concepts  — что читатель освоит из этого блока
                        (передаётся в следующий блок как «уже изучено»)
    assumed_knowledge — что читатель должен знать до этого блока
                        (валидация порядка подачи материала)
"""

from typing import Any
Schema = dict[str, Any]


# ════════════════════════════════════════════════════════════════════════════
#  RAG-задачи
# ════════════════════════════════════════════════════════════════════════════

TAGGING: Schema = {
    "type": "object",
    "properties": {
        "tags":        {"type": "array", "items": {"type": "string"}},
        "primary_tag": {"type": "string"},
        "confidence":  {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["tags", "primary_tag"],
}

SUMMARY: Schema = {
    "type": "object",
    "properties": {
        "summary":          {"type": "string"},
        "key_points":       {"type": "array", "items": {"type": "string"}},
        "topics":           {"type": "array", "items": {"type": "string"}},
        "difficulty_level": {"type": "string", "enum": ["beginner", "intermediate", "advanced"]},
        "language":         {"type": "string"},
    },
    "required": ["summary", "key_points", "topics"],
}

LESSON_MERGE: Schema = {
    "type": "object",
    "properties": {
        "merged_title":            {"type": "string"},
        "merged_summary":          {"type": "string"},
        "merged_key_points":       {"type": "array", "items": {"type": "string"}},
        "merged_tags":             {"type": "array", "items": {"type": "string"}},
        "topics_covered":          {"type": "array", "items": {"type": "string"}},
        "source_lessons_count":    {"type": "integer"},
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
        "questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["questions"],
}

PIPELINE_SETUP: Schema = {
    "type": "object",
    "properties": {
        "search_queries": {"type": "array", "items": {"type": "string"}},
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Плоский список тегов (40–70 штук).",
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
                    "id":     {"type": "integer"},
                    "score":  {"type": "integer", "minimum": 1, "maximum": 10},
                    "reason": {"type": "string"},
                },
                "required": ["id", "score", "reason"],
            },
        }
    },
    "required": ["rankings"],
}

COURSE_STRUCTURE_DETAILED: Schema = {
    "type": "object",
    "properties": {
        "course_title":       {"type": "string"},
        "course_description": {"type": "string"},
        "course_goals":       {"type": "array", "items": {"type": "string"}},
        "modules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id":          {"type": "integer"},
                    "title":       {"type": "string"},
                    "description": {"type": "string"},
                    "goals":       {"type": "array", "items": {"type": "string"}},
                    "key_topics":  {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "title", "description", "goals", "key_topics"],
            },
        },
    },
    "required": ["course_title", "course_description", "course_goals", "modules"],
}

LESSON_DISTRIBUTION: Schema = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "lesson_id": {"type": "integer"},
                    "module_id": {"type": "integer"},
                    "relevance": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["lesson_id", "module_id", "relevance"],
            },
        },
        "unassigned": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["assignments", "unassigned"],
}

COVERAGE_EVALUATION: Schema = {
    "type": "object",
    "properties": {
        "coverage_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "module_status": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "module_id":      {"type": "integer"},
                    "lesson_count":   {"type": "integer"},
                    "status":         {"type": "string", "enum": ["good", "partial", "poor"]},
                    "missing_topics": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["module_id", "lesson_count", "status"],
            },
        },
        "overall_missing_topics": {"type": "array", "items": {"type": "string"}},
        "additional_queries":     {"type": "array", "items": {"type": "string"}},
        "assessment":             {"type": "string"},
    },
    "required": ["coverage_score", "module_status", "additional_queries", "assessment"],
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
                    "indices": {"type": "array", "items": {"type": "integer"}},
                    "reason":  {"type": "string"},
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
            "description": "Краткое итоговое название блока (≤ 10 слов).",
        },
        "summary": {
            "type": "string",
            "description": "3–5 предложений: о чём этот блок и почему важен в курсе.",
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "5–15 конкретных тегов.",
        },
        "merged_text": {
            "type": "string",
            "description": "Глубокий структурированный конспект без повторения уже изученного.",
        },
        # ── Delta-поля ────────────────────────────────────────────────────
        "learned_concepts": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "5–10 конкретных понятий/навыков, которые читатель ВПЕРВЫЕ освоит "
                "из этого блока. Используются следующим блоком как 'уже изучено'. "
                "Формат: конкретные термины ('градиентный спуск', 'Adam-оптимизатор'), "
                "не абстрактные категории ('оптимизация')."
            ),
        },
        "assumed_knowledge": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Понятия, которые читатель должен знать ДО этого блока. "
                "Позволяет проверить корректность порядка подачи материала."
            ),
        },
    },
    "required": ["final_title", "summary", "tags", "merged_text", "learned_concepts"],
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
                                "title":       {"type": "string"},
                                "query_texts": {"type": "array", "items": {"type": "string"}},
                                "tags":        {"type": "array", "items": {"type": "string"}},
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

TEXT_RESPONSE:  Schema = {"type": "object", "properties": {"response": {"type": "string"}, "confidence": {"type": "number"}}, "required": ["response"]}
CLASSIFICATION: Schema = {"type": "object", "properties": {"label": {"type": "string"}, "labels": {"type": "array", "items": {"type": "string"}}, "explanation": {"type": "string"}}, "required": ["label", "labels"]}
NER:            Schema = {"type": "object", "properties": {"entities": {"type": "array", "items": {"type": "object", "properties": {"text": {"type": "string"}, "type": {"type": "string", "enum": ["PERSON", "ORG", "LOC", "DATE", "MONEY", "PRODUCT", "OTHER"]}, "start": {"type": "integer"}, "end": {"type": "integer"}}, "required": ["text", "type"]}}}, "required": ["entities"]}
SENTIMENT:      Schema = {"type": "object", "properties": {"sentiment": {"type": "string", "enum": ["positive", "negative", "neutral", "mixed"]}, "score": {"type": "number"}}, "required": ["sentiment", "score"]}
QA:             Schema = {"type": "object", "properties": {"answer": {"type": "string"}, "source_excerpt": {"type": "string"}, "is_answerable": {"type": "boolean"}}, "required": ["answer", "is_answerable"]}
TRANSCRIPTION:  Schema = {"type": "object", "properties": {"text": {"type": "string"}, "language": {"type": "string"}, "duration_seconds": {"type": "number"}, "segments": {"type": "array", "items": {"type": "object", "properties": {"start": {"type": "number"}, "end": {"type": "number"}, "text": {"type": "string"}}, "required": ["start", "end", "text"]}}}, "required": ["text", "language"]}


# ════════════════════════════════════════════════════════════════════════════
#  Реестр
# ════════════════════════════════════════════════════════════════════════════

SCHEMAS: dict[str, Schema] = {
    # Stage 1 — новые
    "course_structure_detailed": COURSE_STRUCTURE_DETAILED,
    "lesson_distribution":       LESSON_DISTRIBUTION,
    "coverage_evaluation":       COVERAGE_EVALUATION,
    # Stage 1 — старые
    "clarifying_questions":      CLARIFYING_QUESTIONS,
    "pipeline_setup":            PIPELINE_SETUP,
    "course_ranking":            COURSE_RANKING,
    # Stage 2
    "cluster_merge_decision":    CLUSTER_MERGE_DECISION,
    "chunk_generation":          CHUNK_GENERATION,
    "course_structure":          COURSE_STRUCTURE,
    # RAG
    "tagging":                   TAGGING,
    "summary":                   SUMMARY,
    "lesson_merge":              LESSON_MERGE,
    # Общие
    "text":                      TEXT_RESPONSE,
    "classification":            CLASSIFICATION,
    "ner":                       NER,
    "sentiment":                 SENTIMENT,
    "qa":                        QA,
    "transcription":             TRANSCRIPTION,
}

SCHEMA_DESCRIPTIONS: dict[str, str] = {
    "course_structure_detailed": "Stage1: детальная структура курса (9B thinking)",
    "lesson_distribution":       "Stage1: распределение уроков по модулям (4B)",
    "coverage_evaluation":       "Stage1: оценка полноты + доп. запросы (9B)",
    "clarifying_questions":      "Stage1: уточняющие вопросы",
    "pipeline_setup":            "Stage1: запросы + теги",
    "course_ranking":            "Stage1: ранжирование курсов",
    "cluster_merge_decision":    "Stage2: решение об объединении уроков в кластере",
    "chunk_generation":          "Stage2: чанк (title, summary, tags, text, learned_concepts)",
    "course_structure":          "Stage2: структура курса для RAG",
    "tagging":                   "Тегирование",
    "summary":                   "Суммаризация",
    "lesson_merge":              "Объединение уроков",
    "text":                      "Простой текстовый ответ",
    "classification":            "Классификация",
    "ner":                       "Именованные сущности",
    "sentiment":                 "Тональность",
    "qa":                        "Вопрос-ответ",
    "transcription":             "Транскрипция",
}


def get_schema(name: str) -> Schema:
    if name not in SCHEMAS:
        raise ValueError(f"Схема '{name}' не найдена. Доступные: {list(SCHEMAS.keys())}")
    return SCHEMAS[name]


def list_schemas() -> list[dict]:
    return [{"name": n, "description": SCHEMA_DESCRIPTIONS.get(n, "")} for n in SCHEMAS]