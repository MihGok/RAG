"""
Ансамблевая кластеризация без заданного числа кластеров.

Методы (все автоматически определяют k):
  1. DBSCAN        — плотностная, eps=0.55, размытые кластеры
  2. OPTICS        — адаптивная плотностная, xi=0.03
  3. AffinityProp  — propagation с высоким damping=0.92 (мало кластеров)
  4. DBSCAN-tight  — DBSCAN с eps=0.40 (для разнообразия ансамбля)

Ансамбль: матрица совместной принадлежности (co-association).
Финальная кластеризация: hierarchical с порогом 0.4 (при высоком сходстве
объединяем агрессивно — предпочитаем меньше крупных кластеров).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List

import numpy as np
from sklearn.cluster import DBSCAN, OPTICS, AffinityPropagation
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import fcluster, linkage

logger = logging.getLogger(__name__)


# ─── ПЕРЕНАЗНАЧЕНИЕ ШУМА ─────────────────────────────────────────────────────

def _reassign_noise(labels: np.ndarray, dist_matrix: np.ndarray) -> np.ndarray:
    """Шумовые точки (-1) → ближайший кластер по расстоянию."""
    labels = labels.copy()
    valid  = set(labels[labels != -1])
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
    """Переименовываем метки в 0..K-1."""
    mapping = {}
    out     = np.empty_like(labels)
    counter = 0
    for i, l in enumerate(labels):
        if l not in mapping:
            mapping[l] = counter
            counter += 1
        out[i] = mapping[l]
    return out


# ─── ОТДЕЛЬНЫЕ КЛАСТЕРИЗАТОРЫ ────────────────────────────────────────────────

def _run_dbscan(dist: np.ndarray, eps: float, min_samples: int = 2) -> np.ndarray:
    model  = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = model.fit_predict(dist)
    labels = _reassign_noise(labels, dist)
    n_cl   = len(set(labels))
    logger.info("DBSCAN(eps=%.2f): %d кластеров", eps, n_cl)
    return _normalize_labels(labels)


def _run_optics(dist: np.ndarray, min_samples: int = 2, xi: float = 0.03) -> np.ndarray:
    try:
        model  = OPTICS(min_samples=min_samples, xi=xi, metric="precomputed",
                        cluster_method="xi")
        labels = model.fit_predict(dist)
        labels = _reassign_noise(labels, dist)
        n_cl   = len(set(labels))
        logger.info("OPTICS(xi=%.3f): %d кластеров", xi, n_cl)
        return _normalize_labels(labels)
    except Exception as e:
        logger.warning("OPTICS пропущен: %s", e)
        return np.zeros(dist.shape[0], dtype=int)


def _run_affinity_prop(normed: np.ndarray, damping: float = 0.92) -> np.ndarray:
    # Similarity = cosine similarity ([-1, 1]), нам нужна матрица S
    sim    = normed @ normed.T
    # Низкое preference = мало кластеров (желаем объединять)
    pref   = np.percentile(sim, 10)
    try:
        model  = AffinityPropagation(damping=damping, preference=pref,
                                     max_iter=400, random_state=42)
        labels = model.fit_predict(sim)
        n_cl   = len(set(labels))
        logger.info("AffinityProp(d=%.2f): %d кластеров", damping, n_cl)
        return _normalize_labels(labels)
    except Exception as e:
        logger.warning("AffinityProp пропущен: %s", e)
        return np.zeros(normed.shape[0], dtype=int)


# ─── CO-ASSOCIATION ENSEMBLE ─────────────────────────────────────────────────

def _co_association(label_sets: List[np.ndarray], n: int) -> np.ndarray:
    """co[i,j] = доля методов, объединивших i и j в один кластер."""
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

    co   = _co_association(label_sets, n)
    dist = np.clip(1.0 - co, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)

    # Иерархическая кластеризация с порогом 0.4:
    # два объекта объединяются если их «co-association» ≥ 0.6
    # → предпочитаем КРУПНЫЕ / РАЗМЫТЫЕ кластеры
    try:
        Z      = linkage(dist[np.triu_indices(n, 1)], method="average")
        labels = fcluster(Z, t=0.40, criterion="distance")
        labels = labels - 1  # 1-based → 0-based
        n_cl   = len(set(labels))
        logger.info("Ensemble final: %d кластеров (threshold=0.40)", n_cl)
        return labels.tolist()
    except Exception as e:
        logger.warning("Ensemble linkage failed: %s — fallback", e)
        return label_sets[0].tolist()


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def cluster_embeddings(
    embeddings: np.ndarray,
    log_fn=None,
) -> List[int]:
    """
    Ансамблевая кластеризация без указания числа кластеров.
    Настройки смещены в сторону КРУПНЫХ / РАЗМЫТЫХ кластеров
    (лучше для LLM, которая сама решит об объединении).
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
        return [0, 0]  # маленькая коллекция → объединяем

    normed = normalize(embeddings, norm="l2")
    dist   = cdist(normed, normed, metric="cosine")
    dist   = np.clip(dist, 0.0, 2.0)

    all_labels: List[np.ndarray] = []

    # 1. DBSCAN с большим eps (размытые кластеры)
    try:
        all_labels.append(_run_dbscan(dist, eps=0.55))
    except Exception as e:
        _log(f"[Cluster] DBSCAN-loose пропущен: {e}")

    # 2. DBSCAN с меньшим eps (чуть точнее, разнообразие ансамбля)
    try:
        all_labels.append(_run_dbscan(dist, eps=0.40))
    except Exception as e:
        _log(f"[Cluster] DBSCAN-tight пропущен: {e}")

    # 3. OPTICS — адаптивная плотностная
    try:
        all_labels.append(_run_optics(dist, min_samples=2, xi=0.03))
    except Exception as e:
        _log(f"[Cluster] OPTICS пропущен: {e}")

    # 4. AffinityPropagation с высоким damping (мало кластеров)
    try:
        all_labels.append(_run_affinity_prop(normed, damping=0.92))
    except Exception as e:
        _log(f"[Cluster] AffinityProp пропущен: {e}")

    if not all_labels:
        _log("[Cluster] Все методы не сработали — каждый урок в своём кластере")
        return list(range(n))

    final  = _ensemble_cluster(all_labels, n)
    n_cl   = len(set(final))
    _log(f"[Cluster] Ансамбль: {n} уроков → {n_cl} кластеров")
    return final


def labels_to_groups(n_items: int, labels: List[int]) -> List[List[int]]:
    groups: dict = defaultdict(list)
    for i, lbl in enumerate(labels):
        groups[lbl].append(i)
    return list(groups.values())
