from numbers import Integral

import numpy as np
from scipy.sparse import csr_matrix, issparse


class ContentBasedRecommender:
    def __init__(self, genre_features: np.ndarray, min_rating: float = 4.0):
        features = np.array(genre_features, dtype=np.float64, copy=True)
        if features.ndim != 2 or 0 in features.shape:
            raise ValueError("genre_features must be a nonempty movie-by-genre matrix")
        if not np.isfinite(features).all() or np.any(features < 0):
            raise ValueError("genre_features must be finite and nonnegative")
        if not np.isfinite(min_rating) or not 1 <= min_rating <= 5:
            raise ValueError("min_rating must be between 1 and 5")

        norms = np.linalg.norm(features, axis=1, keepdims=True)
        self.item_features = np.divide(
            features, norms, out=np.zeros_like(features), where=norms > 0
        )
        self.min_rating = float(min_rating)
        self.n_movies = features.shape[0]

    def _user_ratings(self, history: csr_matrix, user_index: int):
        if not issparse(history) or history.format != "csr":
            raise ValueError("history must be a CSR sparse matrix from load_data.py")
        if history.shape[1] != self.n_movies:
            raise ValueError("history and genre_features must share movie columns")
        if isinstance(user_index, bool) or not isinstance(user_index, Integral):
            raise ValueError("user_index must be an integer row index")
        if not 0 <= user_index < history.shape[0]:
            raise IndexError("user_index is outside history")
        start, end = history.indptr[user_index:user_index + 2]
        indices, ratings = history.indices[start:end], history.data[start:end]
        if not np.isfinite(ratings).all() or np.any((ratings < 1) | (ratings > 5)):
            raise ValueError("stored history ratings must be between 1 and 5")
        return indices, ratings

    def user_profile(self, history: csr_matrix, user_index: int) -> np.ndarray:
        indices, ratings = self._user_ratings(history, user_index)
        liked = ratings >= self.min_rating
        if not np.any(liked):
            return np.zeros(self.item_features.shape[1], dtype=np.float64)
        weights = ratings[liked].astype(np.float64)
        return weights @ self.item_features[indices[liked]] / weights.sum()

    def score(self, history: csr_matrix, user_index: int) -> np.ndarray:
        profile = self.user_profile(history, user_index)
        norm = np.linalg.norm(profile)
        if norm == 0:
            return np.zeros(self.n_movies, dtype=np.float64)
        return np.clip(self.item_features @ (profile / norm), 0.0, 1.0)

    def recommend(
        self,
        history: csr_matrix,
        user_index: int,
        *,
        candidate_indices: np.ndarray,
        k: int = 10,
    ) -> list[tuple[int, float]]:
        if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
            raise ValueError("k must be a positive integer")
        candidates = np.asarray(candidate_indices)
        if candidates.ndim != 1:
            raise ValueError("candidate_indices must be a one-dimensional array")
        if candidates.size and (
            not np.issubdtype(candidates.dtype, np.integer)
            or np.any(candidates < 0)
            or np.any(candidates >= self.n_movies)
        ):
            raise ValueError("candidate_indices must contain valid integer movie indices")

        scores = self.score(history, user_index)
        if not np.any(scores):
            return []
        start, end = history.indptr[user_index:user_index + 2]
        seen = history.indices[start:end]
        candidates = np.setdiff1d(candidates.astype(np.intp), seen)
        order = np.lexsort((candidates, -scores[candidates]))[:k]
        return [(int(i), float(scores[i])) for i in candidates[order]]
