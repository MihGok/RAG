"""
stage2/clusterer.py
───────────────────
Ансамблевая кластеризация эмбеддингов уроков.
Использует K-Means, DBSCAN и AgglomerativeClustering, затем объединяет через
матрицу совместной принадлежности (co-association matrix).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List, Tuple, Any

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _estimate_k(n: int) -> int:
    """Эвристическая оценка числа кластеров."""
    if n <= 3:
        return 1
    if n <= 8:
        return max(2, n // 3)
    if n <= 25:
        return max(3, n // 4)
    if n <= 60:
        return max(4, n // 6)
    return max(6, n // 9)


def _reassign_noise(labels: np.ndarray, dist_matrix: np.ndarray) -> np.ndarray:
    """Переназначает шумовые точки (-1) к ближайшему кластеру."""
    labels = labels.copy()
    valid_clusters = set(labels[labels != -1])
    if not valid_clusters:
        return np.zeros(len(labels), dtype=int)
    for i, lbl in enumerate(labels):
        if lbl == -1:
            best_cluster = -1
            best_dist = float("inf")
            for j, lbl2 in enumerate(labels):
                if lbl2 != -1 and dist_matrix[i, j] < best_dist:
                    best_dist = dist_matrix[i, j]
                    best_cluster = lbl2
            labels[i] = best_cluster
    return labels


# ─── INDIVIDUAL CLUSTERERS ────────────────────────────────────────────────────

def _kmeans(normed: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(normed)
    logger.info("K-Means (k=%d): %d кластеров", k, len(set(labels)))
    return labels


def _dbscan(normed: np.ndarray, dist_matrix: np.ndarray) -> np.ndarray:
    db = DBSCAN(eps=0.35, min_samples=2, metric="precomputed")
    labels = db.fit_predict(dist_matrix)
    labels = _reassign_noise(labels, dist_matrix)
    logger.info("DBSCAN: %d кластеров", len(set(labels)))
    return labels


def _agglomerative(normed: np.ndarray, k: int) -> np.ndarray:
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = agg.fit_predict(normed)
    logger.info("Agglomerative (k=%d): %d кластеров", k, len(set(labels)))
    return labels


# ─── ENSEMBLE ────────────────────────────────────────────────────────────────

def _co_association_matrix(label_sets: List[np.ndarray], n: int) -> np.ndarray:
    """
    Строит матрицу совместной принадлежности:
    co[i,j] = доля кластеризаций, в которых i и j попали в один кластер.
    """
    co = np.zeros((n, n), dtype=float)
    for labels in label_sets:
        for i in range(n):
            for j in range(i, n):
                if labels[i] == labels[j]:
                    co[i, j] += 1
                    co[j, i] += 1
    co /= len(label_sets)
    return co


def _ensemble(label_sets: List[np.ndarray], n: int) -> List[int]:
    """Объединяет несколько разбиений через матрицу совместной принадлежности."""
    if len(label_sets) == 1:
        return label_sets[0].tolist()

    co = _co_association_matrix(label_sets, n)

    # Расстояние = 1 - co
    dist = 1.0 - co
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 1.0)

    avg_k = int(round(np.mean([len(set(l)) for l in label_sets])))
    avg_k = max(1, min(avg_k, n))

    try:
        final = AgglomerativeClustering(
            n_clusters=avg_k, metric="precomputed", linkage="complete"
        )
        return final.fit_predict(dist).tolist()
    except Exception as e:
        logger.warning("Ensemble final clustering failed: %s", e)
        return label_sets[0].tolist()


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def cluster_embeddings(
    embeddings: np.ndarray,
    log_fn=None,
) -> List[int]:
    """
    Ансамблевая кластеризация.

    Args:
        embeddings: np.ndarray shape (n, dim)
        log_fn:     опциональная функция логирования

    Returns:
        List[int] — метки кластеров длиной n
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
        return [0, 1]

    normed = normalize(embeddings, norm="l2")
    k = _estimate_k(n)
    k = min(k, n)

    dist_matrix = cdist(normed, normed, metric="cosine")
    dist_matrix = np.clip(dist_matrix, 0.0, 2.0)

    all_labels: List[np.ndarray] = []

    # K-Means
    try:
        all_labels.append(_kmeans(normed, k))
    except Exception as e:
        _log(f"[Cluster] K-Means пропущен: {e}")

    # DBSCAN
    try:
        all_labels.append(_dbscan(normed, dist_matrix))
    except Exception as e:
        _log(f"[Cluster] DBSCAN пропущен: {e}")

    # Agglomerative
    try:
        all_labels.append(_agglomerative(normed, k))
    except Exception as e:
        _log(f"[Cluster] Agglomerative пропущен: {e}")

    if not all_labels:
        # Fallback: every lesson in its own cluster
        return list(range(n))

    final = _ensemble(all_labels, n)
    n_clusters = len(set(final))
    _log(f"[Cluster] Итого: {n_clusters} кластеров из {n} уроков")
    return final


def labels_to_groups(n_items: int, labels: List[int]) -> List[List[int]]:
    """
    Преобразует список меток в группы индексов.

    Returns:
        [[idx1, idx2, ...], [idx3], ...] — по одному списку на кластер
    """
    groups: dict[int, List[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        groups[lbl].append(i)
    return list(groups.values())