"""
db/mongo_service.py
───────────────────
MongoDB для хранения структуры сгенерированных курсов.
Коллекция: generated_courses
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any

from pymongo import MongoClient
from pymongo.collection import Collection
from bson import ObjectId

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


def get_courses_collection() -> Collection:
    return get_client()[_DB_NAME]["generated_courses"]


def get_sessions_collection() -> Collection:
    return get_client()[_DB_NAME]["sessions"]


# ─── COURSE STRUCTURE ───────────────────────────────────────────────────────

def save_course_structure(
    course_structure: dict,
    session_id: str,
    topic: str,
    chunks_count: int = 0,
) -> str:
    """
    Сохраняет полную структуру курса в MongoDB.
    Возвращает строковый ID документа.
    """
    doc = {
        "session_id":    session_id,
        "topic":         topic,
        "course_title":  course_structure.get("course_title", ""),
        "modules":       course_structure.get("modules", []),
        "chunks_count":  chunks_count,
    }
    col = get_courses_collection()
    result = col.insert_one(doc)
    mongo_id = str(result.inserted_id)
    logger.info("MongoDB: курс сохранён, id=%s", mongo_id)
    return mongo_id


def get_course_structure(mongo_id: str) -> Optional[dict]:
    col = get_courses_collection()
    doc = col.find_one({"_id": ObjectId(mongo_id)})
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc


def list_courses_by_session(session_id: str) -> list[dict]:
    col = get_courses_collection()
    docs = list(col.find({"session_id": session_id}))
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


# ─── SESSION META ────────────────────────────────────────────────────────────

def save_session_meta(session_id: str, meta: dict) -> str:
    col = get_sessions_collection()
    doc = {"session_id": session_id, **meta}
    result = col.replace_one(
        {"session_id": session_id},
        doc,
        upsert=True,
    )
    return session_id


def get_session_meta(session_id: str) -> Optional[dict]:
    col = get_sessions_collection()
    doc = col.find_one({"session_id": session_id})
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc