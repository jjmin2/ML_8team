"""Shared, leakage-safe helpers for rating and Top-K recommendation evaluation."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def user_candidates(candidate_indices: np.ndarray, seen_indices: np.ndarray) -> np.ndarray:
    """Return globally eligible movies that the user has not already rated."""
    return np.setdiff1d(
        np.asarray(candidate_indices, dtype=np.int64),
        np.asarray(seen_indices, dtype=np.int64),
        assume_unique=False,
    )


def eligible_targets(
    target_indices: np.ndarray,
    target_ratings: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Keep held-out ratings whose movies are in the declared candidate set."""
    target_indices = np.asarray(target_indices, dtype=np.int64)
    target_ratings = np.asarray(target_ratings, dtype=np.float64)
    mask = np.isin(target_indices, candidates, assume_unique=False)
    return target_indices[mask], target_ratings[mask], int((~mask).sum())


def top_k_from_scores(scores: np.ndarray, candidates: np.ndarray, k: int) -> np.ndarray:
    """Rank positive-scoring candidates, breaking ties by movie column index.

    Returning no movies for an all-zero score vector expresses that the model has
    no recommendation signal; callers count this as zero precision/recall.
    """
    if k < 1:
        raise ValueError("k must be positive")
    candidates = np.asarray(candidates, dtype=np.int64)
    if not candidates.size:
        return np.empty(0, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)[candidates]
    positive = values > 0
    if not np.any(positive):
        return np.empty(0, dtype=np.int64)
    candidates, values = candidates[positive], values[positive]
    order = np.lexsort((candidates, -values))[:k]
    return candidates[order]


def ranking_metrics(
    recommended: Iterable[int], relevant: Iterable[int], k: int, candidate_count: int
) -> dict[str, float]:
    """Compute precision, recall, and NDCG for one user with binary relevance."""
    recommended = np.asarray(list(recommended), dtype=np.int64)
    relevant = np.asarray(list(relevant), dtype=np.int64)
    if k < 1 or candidate_count < 0:
        raise ValueError("invalid ranking evaluation configuration")
    relevant_set = set(relevant.tolist())
    hits = np.fromiter((int(movie in relevant_set) for movie in recommended), dtype=np.int8)
    hit_count = int(hits.sum())
    denominator = min(k, candidate_count)
    precision = hit_count / denominator if denominator else 0.0
    recall = hit_count / len(relevant_set) if relevant_set else 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
    dcg = float((hits * discounts).sum())
    ideal_length = min(len(relevant_set), k, candidate_count)
    idcg = float((1.0 / np.log2(np.arange(2, ideal_length + 2))).sum())
    return {"precision": precision, "recall": recall, "ndcg": dcg / idcg if idcg else 0.0}


def chronological_split(matrix, times: np.ndarray, fraction: float = 0.8, min_ratings: int = 5):
    """Split a CSR matrix per user without allowing equal timestamps to straddle sets.

    This is only for MovieLens internal validation. Netflix ``target`` remains
    untouched for final reporting.
    """
    from scipy.sparse import csr_matrix

    if not 0 < fraction < 1 or min_ratings < 2:
        raise ValueError("invalid split configuration")
    matrix = matrix.tocsr()
    times = np.asarray(times, dtype=np.int64)
    if len(times) != matrix.nnz:
        raise ValueError("times must align with CSR data")
    history_data, history_indices, history_indptr = [], [], [0]
    target_data, target_indices, target_indptr = [], [], [0]
    for user in range(matrix.shape[0]):
        start, end = matrix.indptr[user : user + 2]
        values, indices, user_times = matrix.data[start:end], matrix.indices[start:end], times[start:end]
        if len(values) < min_ratings:
            history_mask = np.ones(len(values), dtype=bool)
        else:
            cutoff_position = max(1, int(math.floor(len(values) * fraction))) - 1
            cutoff = user_times[cutoff_position]
            history_mask = user_times <= cutoff
            if history_mask.all():
                history_mask = np.ones(len(values), dtype=bool)
        target_mask = ~history_mask
        history_data.extend(values[history_mask])
        history_indices.extend(indices[history_mask])
        history_indptr.append(len(history_data))
        target_data.extend(values[target_mask])
        target_indices.extend(indices[target_mask])
        target_indptr.append(len(target_data))
    shape = matrix.shape
    return (
        csr_matrix((history_data, history_indices, history_indptr), shape=shape),
        csr_matrix((target_data, target_indices, target_indptr), shape=shape),
    )
