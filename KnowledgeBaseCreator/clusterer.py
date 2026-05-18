"""
KnowledgeBaseCreator/clusterer.py
──────────────────────────────────
Кластеризация уроков.

Новое: cluster_by_module_and_embeddings()
  Двухуровневая стратегия:
    1. Первичная группировка по module_id (результат Stage 1)
    2. Внутри каждого модуля — ансамблевая кластеризация по эмбеддингам
       (для поиска близких уроков, которые стоит объединить)
    3. Уроки с module_id=-1 (нераспределённые) кластеризуются глобально

Старое: cluster_embeddings() сохранено для fallback (нет module_id).

Настройки смещены в сторону КРУПНЫХ / РАЗМЫТЫХ кластеров —
так LLM в merger.py получает больший контекст и сама решает, что объединять.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, OPTICS, AffinityPropagation
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import fcluster, linkage

logger = logging.getLogger(__name__)


# ─── ВСПОМОГАТЕЛЬНЫЕ ─────────────────────────────────────────────────────────

def _reassign_noise(labels: np.ndarray, dist_matrix: np.ndarray) -> np.ndarray:
    """Шумовые точки (-1) → ближайший кластер."""
    labels = labels.copy()
    valid = set(labels[labels != -1])
    if not valid:
        return np.zeros(len(labels), dtype=int)
    for i, lbl in enumerate(labels):
        if lbl == -1:
            best, bd = -1, float("inf")
            for j, lbl2 in enumerate(labels):
                if lbl2 != -1 and dist_matrix[i, j] < bd:
                    bd, best = dist_matrix[i, j], lbl2
            labels[i] = best if best != -1 else 0
    return labels


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    mapping: dict = {}
    out = np.empty_like(labels)
    counter = 0
    for i, l in enumerate(labels):
        if l not in mapping:
            mapping[l] = counter
            counter += 1
        out[i] = mapping[l]
    return out


# ─── ОДИНОЧНЫЕ МЕТОДЫ ────────────────────────────────────────────────────────

def _run_dbscan(dist: np.ndarray, eps: float, min_samples: int = 2) -> np.ndarray:
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = model.fit_predict(dist)
    labels = _reassign_noise(labels, dist)
    logger.info("DBSCAN(eps=%.2f): %d кластеров", eps, len(set(labels)))
    return _normalize_labels(labels)


def _run_optics(dist: np.ndarray, min_samples: int = 2, xi: float = 0.03) -> np.ndarray:
    try:
        model = OPTICS(min_samples=min_samples, xi=xi, metric="precomputed", cluster_method="xi")
        labels = model.fit_predict(dist)
        labels = _reassign_noise(labels, dist)
        logger.info("OPTICS(xi=%.3f): %d кластеров", xi, len(set(labels)))
        return _normalize_labels(labels)
    except Exception as e:
        logger.warning("OPTICS пропущен: %s", e)
        return np.zeros(dist.shape[0], dtype=int)


def _run_affinity_prop(normed: np.ndarray, damping: float = 0.92) -> np.ndarray:
    sim = normed @ normed.T
    pref = np.percentile(sim, 10)
    try:
        model = AffinityPropagation(damping=damping, preference=pref, max_iter=400, random_state=42)
        labels = model.fit_predict(sim)
        logger.info("AffinityProp(d=%.2f): %d кластеров", damping, len(set(labels)))
        return _normalize_labels(labels)
    except Exception as e:
        logger.warning("AffinityProp пропущен: %s", e)
        return np.zeros(normed.shape[0], dtype=int)


# ─── АНСАМБЛЬ ────────────────────────────────────────────────────────────────

def _co_association(label_sets: List[np.ndarray], n: int) -> np.ndarray:
    co = np.zeros((n, n), dtype=float)
    for labels in label_sets:
        for i in range(n):
            for j in range(i, n):
                if labels[i] == labels[j]:
                    co[i, j] += 1
                    co[j, i] += 1
    co /= len(label_sets)
    return co


def _ensemble_cluster(label_sets: List[np.ndarray], n: int) -> List[int]:
    if len(label_sets) == 1:
        return label_sets[0].tolist()
    co = _co_association(label_sets, n)
    dist = np.clip(1.0 - co, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    try:
        Z = linkage(dist[np.triu_indices(n, 1)], method="average")
        labels = fcluster(Z, t=0.40, criterion="distance")
        labels = labels - 1
        logger.info("Ensemble: %d кластеров (threshold=0.40)", len(set(labels)))
        return labels.tolist()
    except Exception as e:
        logger.warning("Ensemble linkage failed: %s", e)
        return label_sets[0].tolist()


# ─── ПУБЛИЧНЫЙ API (глобальная кластеризация) ────────────────────────────────

def cluster_embeddings(
    embeddings: np.ndarray,
    log_fn=None,
) -> List[int]:
    """
    Глобальная ансамблевая кластеризация без указания числа кластеров.
    Используется как fallback (нет module_id) и для внутримодульной кластеризации.
    """
    def _log(msg: str):
        logger.info(msg)
        if log_fn:
            log_fn(msg)

    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [0]
    if n == 2:
        return [0, 0]

    normed = normalize(embeddings, norm="l2")
    dist = np.clip(cdist(normed, normed, metric="cosine"), 0.0, 2.0)

    all_labels: List[np.ndarray] = []

    try:
        all_labels.append(_run_dbscan(dist, eps=0.55))
    except Exception as e:
        _log(f"[Cluster] DBSCAN-loose: {e}")

    try:
        all_labels.append(_run_dbscan(dist, eps=0.40))
    except Exception as e:
        _log(f"[Cluster] DBSCAN-tight: {e}")

    try:
        all_labels.append(_run_optics(dist, min_samples=2, xi=0.03))
    except Exception as e:
        _log(f"[Cluster] OPTICS: {e}")

    try:
        all_labels.append(_run_affinity_prop(normed, damping=0.92))
    except Exception as e:
        _log(f"[Cluster] AffinityProp: {e}")

    if not all_labels:
        return list(range(n))

    final = _ensemble_cluster(all_labels, n)
    _log(f"[Cluster] {n} уроков → {len(set(final))} кластеров")
    return final


def labels_to_groups(n_items: int, labels: List[int]) -> List[List[int]]:
    groups: dict = defaultdict(list)
    for i, lbl in enumerate(labels):
        groups[lbl].append(i)
    return list(groups.values())


# ─── НОВОЕ: ДВУХУРОВНЕВАЯ КЛАСТЕРИЗАЦИЯ С MODULE_ID ─────────────────────────

def cluster_by_module_and_embeddings(
    lessons: list,
    embeddings: np.ndarray,
    module_ids: List[int],
    log_fn=None,
) -> List[List[int]]:
    """
    Двухуровневая кластеризация, учитывающая распределение Stage 1.

    Уровень 1 — группировка по module_id:
      Уроки одного модуля не смешиваются с уроками другого.
      Это жёсткое ограничение: семантика модуля важнее
      близости эмбеддингов между разными модулями.

    Уровень 2 — кластеризация внутри модуля:
      ≤ 3  уроков → один кластер (нет смысла делить)
      4–8  уроков → один кластер (merger.py сам решит, что объединять)
      > 8  уроков → ансамблевая кластеризация по эмбеддингам,
                    чтобы найти подгруппы для слияния

    module_id = -1 → «нераспределённые» — глобальная кластеризация.

    Args:
        lessons:    список (filename, lesson_name, data_dict)
        embeddings: np.ndarray shape (n, dim)
        module_ids: list[int], len == len(lessons); -1 = unassigned
        log_fn:     callback для логов

    Returns:
        List[List[int]] — каждый элемент: список индексов одного кластера
    """
    def _log(msg: str):
        logger.info(msg)
        if log_fn:
            log_fn(msg)

    n = len(lessons)
    if n == 0:
        return []

    # ── Группировка по module_id ──────────────────────────────────────
    module_groups: dict = defaultdict(list)
    for i, mid in enumerate(module_ids):
        module_groups[mid].append(i)

    final_groups: List[List[int]] = []

    # ── Назначенные модули (mid ≥ 0) ──────────────────────────────────
    assigned_mids = sorted(k for k in module_groups if k >= 0)
    for mid in assigned_mids:
        indices = module_groups[mid]
        mod_n = len(indices)

        if mod_n <= 8:
            # Небольшая группа — весь модуль в один кластер
            final_groups.append(indices)
            _log(f"  Модуль {mid}: {mod_n} уроков → 1 кластер")
        else:
            # Большая группа — подкластеризация по эмбеддингам
            sub_embeds = embeddings[indices] if len(embeddings) > 0 else np.array([])
            if len(sub_embeds) == 0:
                final_groups.append(indices)
                continue
            sub_labels = cluster_embeddings(sub_embeds, log_fn=None)
            sub_groups: dict = defaultdict(list)
            for local_idx, sub_lbl in enumerate(sub_labels):
                sub_groups[sub_lbl].append(indices[local_idx])
            sub_group_list = list(sub_groups.values())
            final_groups.extend(sub_group_list)
            _log(
                f"  Модуль {mid}: {mod_n} уроков → "
                f"{len(sub_group_list)} подкластеров"
            )

    # ── Нераспределённые (mid = -1) — глобальная кластеризация ───────
    unassigned = module_groups.get(-1, [])
    if unassigned:
        _log(f"  Нераспределённые: {len(unassigned)} уроков")
        if len(unassigned) <= 2:
            # По одному в кластер
            for idx in unassigned:
                final_groups.append([idx])
        else:
            sub_embeds = embeddings[unassigned] if len(embeddings) > 0 else np.array([])
            if len(sub_embeds) > 0:
                sub_labels = cluster_embeddings(sub_embeds, log_fn=None)
                sub_groups: dict = defaultdict(list)
                for local_idx, sub_lbl in enumerate(sub_labels):
                    sub_groups[sub_lbl].append(unassigned[local_idx])
                final_groups.extend(sub_groups.values())
            else:
                final_groups.append(unassigned)

    total_clusters = len(final_groups)
    _log(
        f"[Cluster] Итого: {n} уроков, "
        f"{len(assigned_mids)} модулей → {total_clusters} кластеров"
    )
    return final_groups