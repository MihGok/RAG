"""
stage2/qdrant_indexer.py
────────────────────────
Индексация чанков в Qdrant:
  - Dense vector: эмбеддинг summary (семантический поиск)
  - Sparse vector: BM25 по тегам (лексический поиск)
"""

from __future__ import annotations

import hashlib
import logging
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
)

from KnowledgeBaseCreator.embedder import embed_text
from config import AppConfig

logger = logging.getLogger(__name__)

DENSE_NAME  = "summary_vec"
SPARSE_NAME = "tags_bm25"
SPARSE_DIM  = 65536  # пространство хеш-индексов для тегов


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
):
    """Создаёт коллекцию если её нет."""
    try:
        info = client.get_collection(collection_name)
        logger.info("Qdrant: коллекция '%s' уже существует", collection_name)
        return info
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
    logger.info(
        "Qdrant: создана коллекция '%s' (dim=%d)", collection_name, embedding_dim
    )


# ─── SPARSE VECTOR ───────────────────────────────────────────────────────────

def _tags_to_sparse(tags: List[str]) -> SparseVector:
    """
    Конвертирует список тегов в разреженный вектор для BM25-поиска.
    Индекс = hash(tag) % SPARSE_DIM, значение = TF-нормализованный вес.
    """
    if not tags:
        return SparseVector(indices=[], values=[])

    tf: Dict[int, float] = {}
    for tag in tags:
        idx = int(hashlib.md5(tag.lower().strip().encode()).hexdigest()[:8], 16) % SPARSE_DIM
        tf[idx] = tf.get(idx, 0.0) + 1.0

    total = sum(tf.values())
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
    Индексирует один чанк:
    - Dense:  embed(summary)
    - Sparse: BM25 на тегах
    - Payload: всё содержимое чанка

    Returns:
        True если успешно, False при ошибке.
    """
    summary = chunk.get("summary", "")
    tags    = chunk.get("tags", [])

    # Dense vector
    dense_vec = embed_text(summary, embed_model)
    if not dense_vec:
        logger.warning("Нет эмбеддинга для chunk id=%d, пропускаю", point_id)
        return False

    # Sparse vector
    sparse_vec = _tags_to_sparse(tags)

    payload = {
        "final_title":       chunk.get("final_title", ""),
        "summary":           summary,
        "tags":              tags,
        "text":              chunk.get("merged_text", ""),
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
    logger.debug("Indexed chunk %d: %s", point_id, chunk.get("final_title"))
    return True


def index_all_chunks(
    chunks: List[Dict[str, Any]],
    collection_name: str,
    embed_model: str,
    log_fn=None,
) -> int:
    """
    Индексирует список чанков в Qdrant.

    Returns:
        Количество успешно проиндексированных чанков.
    """
    def _log(msg: str):
        logger.info(msg)
        if log_fn:
            log_fn(msg)

    if not chunks:
        return 0

    client = get_client()

    # Определяем размерность из первого эмбеддинга
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
        _log(f"[Qdrant] {i+1}/{len(chunks)}: {chunk.get('final_title', '?')} {'✅' if success else '❌'}")

    _log(f"[Qdrant] Проиндексировано {ok_count}/{len(chunks)} чанков в '{collection_name}'")
    return ok_count


# ─── SEARCH ──────────────────────────────────────────────────────────────────

def search_similar(
    query: str,
    collection_name: str,
    embed_model: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Семантический поиск по summary. Используется в RAG-чате.
    """
    client = get_client()
    query_vec = embed_text(query, embed_model)
    if not query_vec:
        return []

    results = client.search(
        collection_name=collection_name,
        query_vector=(DENSE_NAME, query_vec),
        limit=limit,
        with_payload=True,
    )
    return [
        {"score": r.score, **r.payload}
        for r in results
    ]


def search_by_tags(
    tags: List[str],
    collection_name: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    BM25-поиск по тегам. Используется для структуры курса.
    """
    client = get_client()
    sparse_vec = _tags_to_sparse(tags)
    if not sparse_vec.indices:
        return []

    results = client.search(
        collection_name=collection_name,
        query_vector=(SPARSE_NAME, sparse_vec),
        limit=limit,
        with_payload=True,
    )
    return [
        {"score": r.score, **r.payload}
        for r in results
    ]
