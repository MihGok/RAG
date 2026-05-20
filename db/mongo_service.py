"""
db/mongo_service.py
───────────────────
MongoDB для хранения структур курсов.

BUG FIX: добавлен ?authSource=admin в URI по умолчанию для Docker-окружения
         с аутентификацией через MONGO_INITDB_ROOT_USERNAME/PASSWORD.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection

logger = logging.getLogger(__name__)

# BUG FIX: при наличии логина/пароля в URI MongoDB требует authSource=admin
# (корневой пользователь создаётся в admin, но по умолчанию аутентификация
# происходит против базы из URI-пути, которой не существует)
_MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
_DB_NAME   = os.getenv("MONGO_DB", "rag_db")

_client: Optional[MongoClient] = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        # authSource=admin нужен когда пользователь создан через
        # MONGO_INITDB_ROOT_USERNAME (он живёт в базе admin)
        connect_uri = _MONGO_URI
        if "@" in _MONGO_URI and "authSource" not in _MONGO_URI:
            sep = "?" if "?" not in _MONGO_URI else "&"
            connect_uri = f"{_MONGO_URI}{sep}authSource=admin"

        _client = MongoClient(connect_uri, serverSelectionTimeoutMS=5000)
        logger.info("MongoDB: подключение установлено (%s)", connect_uri)
    return _client


def _col(name: str) -> Collection:
    return get_client()[_DB_NAME][name]


# ─── SYLLABUS (Stage 1) ───────────────────────────────────────────────────────

def save_syllabus(
    session_id: str,
    syllabus: Dict[str, Any],
    topic: str,
) -> str:
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
    doc = {
        "session_id":          session_id,
        "topic":               topic,
        "course_title":        course_structure.get("course_title", ""),
        "modules":             course_structure.get("modules", []),
        "chunks_count":        chunks_count,
        "syllabus_session_id": syllabus_session_id or session_id,
    }

    result   = _col("generated_courses").insert_one(doc)
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