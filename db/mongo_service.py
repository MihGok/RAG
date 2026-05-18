"""
db/mongo_service.py
───────────────────
MongoDB для хранения структур курсов.

Коллекции:
  generated_courses — Stage 2 структура (модули → шаги с query_texts/tags)
  syllabi           — Stage 1 силлабус (модули → цели, key_topics)
                      Хранит детальный план от 9B-модели с thinking.
  sessions          — мета-информация о сессиях
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection

logger = logging.getLogger(__name__)

_MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
_DB_NAME   = os.getenv("MONGO_DB", "rag_db")

_client: Optional[MongoClient] = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(_MONGO_URI, serverSelectionTimeoutMS=5000)
        logger.info("MongoDB: подключение установлено")
    return _client


def _col(name: str) -> Collection:
    return get_client()[_DB_NAME][name]


# ─── SYLLABUS (Stage 1) ───────────────────────────────────────────────────────

def save_syllabus(
    session_id: str,
    syllabus: Dict[str, Any],
    topic: str,
) -> str:
    """
    Сохраняет детальную структуру курса из Stage 1 (9B + thinking).

    Коллекция: syllabi
    Upsert по session_id — перезапись при повторном запуске Stage 1.

    Returns:
        session_id (используется как идентификатор в downstream)
    """
    doc = {
        "session_id":        session_id,
        "topic":             topic,
        "course_title":      syllabus.get("course_title", ""),
        "course_description": syllabus.get("course_description", ""),
        "course_goals":      syllabus.get("course_goals", []),
        "modules": [
            {
                "id":          m.get("id"),
                "title":       m.get("title", ""),
                "description": m.get("description", ""),
                "goals":       m.get("goals", []),
                "key_topics":  m.get("key_topics", []),
            }
            for m in syllabus.get("modules", [])
        ],
        "n_modules": len(syllabus.get("modules", [])),
    }

    _col("syllabi").replace_one(
        {"session_id": session_id},
        doc,
        upsert=True,
    )
    logger.info("MongoDB[syllabi]: силлабус сохранён, session_id=%s", session_id)
    return session_id


def get_syllabus(session_id: str) -> Optional[Dict[str, Any]]:
    """Загрузить силлабус Stage 1 по session_id."""
    doc = _col("syllabi").find_one({"session_id": session_id})
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc


# ─── COURSE STRUCTURE (Stage 2) ───────────────────────────────────────────────

def save_course_structure(
    course_structure: Dict[str, Any],
    session_id: str,
    topic: str,
    chunks_count: int = 0,
    syllabus_session_id: Optional[str] = None,
) -> str:
    """
    Сохраняет Stage 2 структуру курса (T-lite: модули → шаги).

    Args:
        syllabus_session_id: session_id силлабуса Stage 1 для связи.
                             Обычно совпадает с session_id.

    Returns:
        Строковый MongoDB _id документа.
    """
    doc = {
        "session_id":          session_id,
        "topic":               topic,
        "course_title":        course_structure.get("course_title", ""),
        "modules":             course_structure.get("modules", []),
        "chunks_count":        chunks_count,
        # Ссылка на Stage 1 силлабус для трассировки
        "syllabus_session_id": syllabus_session_id or session_id,
    }

    result  = _col("generated_courses").insert_one(doc)
    mongo_id = str(result.inserted_id)
    logger.info("MongoDB[generated_courses]: курс сохранён, id=%s", mongo_id)
    return mongo_id


def get_course_structure(mongo_id: str) -> Optional[Dict[str, Any]]:
    doc = _col("generated_courses").find_one({"_id": ObjectId(mongo_id)})
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc


def list_courses_by_session(session_id: str) -> List[Dict[str, Any]]:
    docs = list(_col("generated_courses").find({"session_id": session_id}))
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


# ─── SESSION META ─────────────────────────────────────────────────────────────

def save_session_meta(session_id: str, meta: Dict[str, Any]) -> str:
    doc = {"session_id": session_id, **meta}
    _col("sessions").replace_one({"session_id": session_id}, doc, upsert=True)
    return session_id


def get_session_meta(session_id: str) -> Optional[Dict[str, Any]]:
    doc = _col("sessions").find_one({"session_id": session_id})
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc