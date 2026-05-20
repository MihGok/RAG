from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import hashlib
import os
import logging
import secrets
from contextlib import contextmanager
from typing import Optional
 
import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor
 
logger = logging.getLogger(__name__)
 
_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://postgres:postgres@localhost:5432/rag_db",
)
_pool: Optional[psycopg2.pool.SimpleConnectionPool] = None
 
 
def get_pool() -> psycopg2.pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.SimpleConnectionPool(1, 10, dsn=_DSN)
    return _pool
 
 
@contextmanager
def get_conn():
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)
 
 
# ─── DDL ────────────────────────────────────────────────────────────────────
 
DDL = """
CREATE TABLE IF NOT EXISTS users (
    user_id       BIGSERIAL PRIMARY KEY,
    email         VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(512)        NOT NULL,
    full_name     VARCHAR(255),
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);
 
CREATE TABLE IF NOT EXISTS chats (
    chat_id    BIGSERIAL PRIMARY KEY,
    user_id    BIGINT REFERENCES users(user_id) ON DELETE CASCADE,
    title      VARCHAR(512),
    meta       JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
 
CREATE TABLE IF NOT EXISTS messages (
    message_id  BIGSERIAL PRIMARY KEY,
    chat_id     BIGINT REFERENCES chats(chat_id) ON DELETE CASCADE,
    sender_role BOOLEAN     NOT NULL,
    content     TEXT        NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
 
CREATE TABLE IF NOT EXISTS generated_courses (
    course_id  BIGSERIAL PRIMARY KEY,
    user_id    BIGINT REFERENCES users(user_id) ON DELETE SET NULL,
    chat_id    BIGINT REFERENCES chats(chat_id) ON DELETE SET NULL,
    title      VARCHAR(512),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
 
CREATE TABLE IF NOT EXISTS generated_course_sections (
    section_id       BIGSERIAL PRIMARY KEY,
    course_id        BIGINT REFERENCES generated_courses(course_id) ON DELETE CASCADE,
    section_order    INT          NOT NULL DEFAULT 0,
    title            VARCHAR(512),
    mongo_section_id VARCHAR(128),
    created_at       TIMESTAMPTZ DEFAULT NOW()
);
"""
 
 
def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)
    logger.info("PostgreSQL: таблицы готовы")
 
 
# ─── PASSWORD ────────────────────────────────────────────────────────────────
 
def hash_password(password: str) -> str:
    """PBKDF2-HMAC-SHA256 с солью. Без внешних зависимостей."""
    salt = secrets.token_bytes(32)
    key  = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
    return salt.hex() + ":" + key.hex()
 
 
def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, key_hex = stored_hash.split(":")
        salt = bytes.fromhex(salt_hex)
        key  = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
        return secrets.compare_digest(key.hex(), key_hex)
    except Exception:
        return False
 
 
# ─── USERS ──────────────────────────────────────────────────────────────────
 
def create_user(email: str, password: str, full_name: str = "") -> dict:
    """Создаёт пользователя с хешированным паролем."""
    pwd_hash = hash_password(password)
    sql = """
        INSERT INTO users (email, password_hash, full_name)
        VALUES (%s, %s, %s)
        RETURNING user_id, email, full_name, created_at
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (email.lower().strip(), pwd_hash, full_name))
            return dict(cur.fetchone())
 
 
def authenticate_user(email: str, password: str) -> Optional[dict]:
    """
    Проверяет пароль. Возвращает user dict без password_hash или None.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM users WHERE email = %s",
                (email.lower().strip(),),
            )
            row = cur.fetchone()
 
    if not row:
        return None
    if not verify_password(password, row["password_hash"]):
        return None
 
    user = dict(row)
    user.pop("password_hash", None)
    return user
 
 
def get_user_by_email(email: str) -> Optional[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT user_id, email, full_name, created_at FROM users WHERE email = %s",
                (email.lower().strip(),),
            )
            row = cur.fetchone()
            return dict(row) if row else None
 
 
def get_or_create_default_user() -> dict:
    user = get_user_by_email("default@rag.local")
    if user:
        return user
    return create_user("default@rag.local", "default_pwd_2025", "Default User")
 
 
# ─── CHATS ──────────────────────────────────────────────────────────────────
 
def create_chat(user_id: int, title: str = "Новый чат", meta: dict = None) -> dict:
    sql = """
        INSERT INTO chats (user_id, title, meta)
        VALUES (%s, %s, %s::jsonb)
        RETURNING *
    """
    import json as _json
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (user_id, title, _json.dumps(meta or {})))
            return dict(cur.fetchone())
 
 
def update_chat_meta(chat_id: int, meta: dict):
    import json as _json
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chats SET meta = %s::jsonb, updated_at = NOW() WHERE chat_id = %s",
                (_json.dumps(meta), chat_id),
            )
 
 
def update_chat_title(chat_id: int, title: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chats SET title = %s, updated_at = NOW() WHERE chat_id = %s",
                (title, chat_id),
            )
 
 
def list_chats(user_id: int) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM chats WHERE user_id = %s ORDER BY updated_at DESC",
                (user_id,),
            )
            return [dict(r) for r in cur.fetchall()]
 
 
def delete_chat(chat_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chats WHERE chat_id = %s", (chat_id,))
 
 
# ─── MESSAGES ───────────────────────────────────────────────────────────────

def add_message(chat_id: int, sender_role: bool, content: str) -> dict:
    """
    BUG FIX: fetchone() вызывался ПОСЛЕ второго execute (UPDATE),
    тогда курсор уже не содержал результатов INSERT RETURNING.
    dict(None) → TypeError, молча подавляемый в _save_pair.
    Теперь fetchone() вызывается сразу после INSERT, до UPDATE.
    """
    insert_sql = """
        INSERT INTO messages (chat_id, sender_role, content)
        VALUES (%s, %s, %s)
        RETURNING *
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(insert_sql, (chat_id, sender_role, content))
            # ↓ MUST fetch before next execute() replaces the result set
            row = dict(cur.fetchone())
            cur.execute(
                "UPDATE chats SET updated_at = NOW() WHERE chat_id = %s",
                (chat_id,),
            )
            return row
 
 
def get_messages(chat_id: int, limit: int = 200) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM messages WHERE chat_id = %s "
                "ORDER BY created_at ASC LIMIT %s",
                (chat_id, limit),
            )
            return [dict(r) for r in cur.fetchall()]
 
 
# ─── GENERATED_COURSES ──────────────────────────────────────────────────────
 
def create_generated_course(
    user_id: int, title: str, chat_id: Optional[int] = None
) -> dict:
    sql = """
        INSERT INTO generated_courses (user_id, chat_id, title)
        VALUES (%s, %s, %s)
        RETURNING *
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (user_id, chat_id, title))
            return dict(cur.fetchone())
 
 
def list_generated_courses(user_id: int) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM generated_courses WHERE user_id = %s "
                "ORDER BY created_at DESC",
                (user_id,),
            )
            return [dict(r) for r in cur.fetchall()]
 
 
def add_course_section(
    course_id: int, section_order: int, title: str, mongo_section_id: str = ""
) -> dict:
    sql = """
        INSERT INTO generated_course_sections
            (course_id, section_order, title, mongo_section_id)
        VALUES (%s, %s, %s, %s)
        RETURNING *
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (course_id, section_order, title, mongo_section_id))
            return dict(cur.fetchone())
 
 
def save_course_with_sections(
    user_id: int,
    course_structure: dict,
    mongo_doc_id: str = "",
    chat_id: Optional[int] = None,
) -> dict:
    course_title = course_structure.get("course_title", "Курс")
    course_row   = create_generated_course(user_id, course_title, chat_id)
    course_id    = course_row["course_id"]
 
    for order, module in enumerate(course_structure.get("modules", [])):
        add_course_section(
            course_id=course_id,
            section_order=order,
            title=module.get("title", f"Модуль {order+1}"),
            mongo_section_id=mongo_doc_id,
        )
    return course_row
