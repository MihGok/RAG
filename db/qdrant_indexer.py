"""
db/qdrant_indexer.py
─────────────────────
Индексация чанков в Qdrant.

Изменения:
  • module_id добавлен в payload каждой точки (из Stage 1 распределения)
  • search_similar_in_module() — семантический поиск с фильтром по module_id
  • search_by_tags() сохранён без изменений
  • ensure_collection() — создаёт поле module_id как indexed для фильтрации

Векторы:
  Dense  (DENSE_NAME)  — embed(summary), семантический поиск
  Sparse (SPARSE_NAME) — BM25 по тегам, лексический поиск
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from KnowledgeBaseCreator.embedder import embed_text
from config import AppConfig

logger = logging.getLogger(__name__)

DENSE_NAME  = "summary_vec"
SPARSE_NAME = "tags_bm25"
SPARSE_DIM  = 65536


# ─── CLIENT ──────────────────────────────────────────────────────────────────

def get_client() -> QdrantClient:
    return QdrantClient(
        host=AppConfig.QDRANT_HOST,
        port=AppConfig.QDRANT_PORT,
    )


# ─── COLLECTION ──────────────────────────────────────────────────────────────

def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    embedding_dim: int,
) -> None:
    """
    Создаёт коллекцию если её нет.
    После создания индексирует поле module_id для фильтрации.
    """
    try:
        client.get_collection(collection_name)
        logger.info("Qdrant: коллекция '%s' уже существует", collection_name)
        return
    except Exception:
        pass

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_NAME: VectorParams(size=embedding_dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )

    # Создаём payload-индекс для поля module_id — ускоряет фильтрованный поиск
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="module_id",
            field_schema=PayloadSchemaType.INTEGER,
        )
    except Exception as e:
        logger.warning("Не удалось создать индекс module_id: %s", e)

    logger.info(
        "Qdrant: создана коллекция '%s' (dim=%d, module_id indexed)",
        collection_name, embedding_dim,
    )


# ─── SPARSE VECTOR ────────────────────────────────────────────────────────────

def _tags_to_sparse(tags: List[str]) -> SparseVector:
    if not tags:
        return SparseVector(indices=[], values=[])

    tf: Dict[int, float] = {}
    for tag in tags:
        idx = int(hashlib.md5(tag.lower().strip().encode()).hexdigest()[:8], 16) % SPARSE_DIM
        tf[idx] = tf.get(idx, 0.0) + 1.0

    total   = sum(tf.values())
    indices = list(tf.keys())
    values  = [v / total for v in tf.values()]
    return SparseVector(indices=indices, values=values)


# ─── INDEX ────────────────────────────────────────────────────────────────────

def index_chunk(
    client: QdrantClient,
    collection_name: str,
    chunk: Dict[str, Any],
    point_id: int,
    embed_model: str,
) -> bool:
    """
    Индексирует один чанк.

    Payload включает module_id (из Stage 1 распределения, -1 = нераспределён).
    Это позволяет фильтровать поиск по конкретному модулю курса.
    """
    summary = chunk.get("summary", "")
    tags    = chunk.get("tags", [])

    dense_vec = embed_text(summary, embed_model)
    if not dense_vec:
        logger.warning("Нет эмбеддинга для chunk id=%d", point_id)
        return False

    sparse_vec = _tags_to_sparse(tags)

    payload = {
        "final_title":       chunk.get("final_title", ""),
        "summary":           summary,
        "tags":              tags,
        "text":              chunk.get("merged_text", ""),
        # module_id из Stage 1 — ключевое поле для фильтрации
        "module_id":         int(chunk.get("module_id", -1)),
        "source_lesson_ids": chunk.get("source_lesson_ids", []),
        "source_course_ids": chunk.get("source_course_ids", []),
        "source_filenames":  chunk.get("source_filenames", []),
    }

    point = PointStruct(
        id=point_id,
        vector={
            DENSE_NAME:  dense_vec,
            SPARSE_NAME: sparse_vec,
        },
        payload=payload,
    )

    client.upsert(collection_name=collection_name, points=[point])
    logger.debug(
        "Indexed chunk %d (module_id=%d): %s",
        point_id, payload["module_id"], chunk.get("final_title"),
    )
    return True


def index_all_chunks(
    chunks: List[Dict[str, Any]],
    collection_name: str,
    embed_model: str,
    log_fn=None,
) -> int:
    """
    Индексирует список чанков в Qdrant.
    Returns: количество успешно проиндексированных.
    """
    def _log(msg: str):
        logger.info(msg)
        if log_fn:
            log_fn(msg)

    if not chunks:
        return 0

    client = get_client()

    first_summary = chunks[0].get("summary", chunks[0].get("final_title", ""))
    first_emb = embed_text(first_summary, embed_model)
    if not first_emb:
        _log("[Qdrant] Не удалось получить тестовый эмбеддинг")
        return 0

    embedding_dim = len(first_emb)
    ensure_collection(client, collection_name, embedding_dim)

    ok_count = 0
    for i, chunk in enumerate(chunks):
        success = index_chunk(client, collection_name, chunk, point_id=i, embed_model=embed_model)
        if success:
            ok_count += 1
        _log(
            f"[Qdrant] {i+1}/{len(chunks)}: "
            f"module={chunk.get('module_id', -1)} | "
            f"{chunk.get('final_title', '?')} "
            f"{'✅' if success else '❌'}"
        )

    _log(f"[Qdrant] Проиндексировано {ok_count}/{len(chunks)} чанков в '{collection_name}'")
    return ok_count


# ─── SEARCH ──────────────────────────────────────────────────────────────────

def search_similar(
    query: str,
    collection_name: str,
    embed_model: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Семантический поиск по summary без фильтров (глобальный)."""
    client    = get_client()
    query_vec = embed_text(query, embed_model)
    if not query_vec:
        return []

    results = client.search(
        collection_name=collection_name,
        query_vector=(DENSE_NAME, query_vec),
        limit=limit,
        with_payload=True,
    )
    return [{"score": r.score, **r.payload} for r in results]


def search_similar_in_module(
    query: str,
    collection_name: str,
    embed_model: str,
    module_id: int,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Семантический поиск, ограниченный конкретным модулем.

    Используется в RAG-чате когда пользователь работает
    в контексте определённого модуля курса.

    Args:
        module_id: 0-based индекс модуля из Stage 1. -1 = поиск по всем.
    """
    client    = get_client()
    query_vec = embed_text(query, embed_model)
    if not query_vec:
        return []

    query_filter: Optional[Filter] = None
    if module_id >= 0:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="module_id",
                    match=MatchValue(value=module_id),
                )
            ]
        )

    results = client.search(
        collection_name=collection_name,
        query_vector=(DENSE_NAME, query_vec),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    return [{"score": r.score, **r.payload} for r in results]


def search_by_tags(
    tags: List[str],
    collection_name: str,
    limit: int = 5,
    module_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    BM25-поиск по тегам.
    Опциональная фильтрация по module_id для структуры курса.
    """
    client     = get_client()
    sparse_vec = _tags_to_sparse(tags)
    if not sparse_vec.indices:
        return []

    query_filter: Optional[Filter] = None
    if module_id is not None and module_id >= 0:
        query_filter = Filter(
            must=[FieldCondition(key="module_id", match=MatchValue(value=module_id))]
        )

    results = client.search(
        collection_name=collection_name,
        query_vector=(SPARSE_NAME, sparse_vec),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    return [{"score": r.score, **r.payload} for r in results]


def get_chunks_by_module(
    collection_name: str,
    module_id: int,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Возвращает все чанки конкретного модуля без векторного поиска.
    Полезно для генерации DOCX и структуры курса.
    """
    client = get_client()
    results, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[FieldCondition(key="module_id", match=MatchValue(value=module_id))]
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return [r.payload for r in results if r.payload]