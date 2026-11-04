"""Hybrid ranking recommender combining existing content model and item CF."""
import numpy as np
from content_based import ContentBasedRecommender
from item_based_cf import ItemBasedCFRecommender


class HybridRecommender:
    def __init__(self, genre_features, alpha=0.5, cf_neighbors=100, shrinkage=10.0):
        if not 0 <= alpha <= 1:
            raise ValueError('alpha must be in [0, 1]')
        # alpha weights content-based predictions; (1 - alpha) weights CF.
        self.alpha = float(alpha)
        self.content = ContentBasedRecommender(genre_features)
        self.cf = ItemBasedCFRecommender(top_neighbors=cf_neighbors, shrinkage=shrinkage)

    def fit(self, train):
        # Train CF similarities on MovieLens; content features need no fitting.
        self.cf.fit(train)
        return self

    def score(self, history, user_index):
        content = self.content.score(history, user_index)
        cf = self.cf.score(history, user_index)
        # Ranking scores are in [0, 1], not predicted 1-5 star ratings.
        return self.alpha * content + (1.0 - self.alpha) * cf

    def recommend(self, history, user_index, candidate_indices, k=10):
        if k < 1:
            raise ValueError('k must be positive')
        candidates = np.asarray(candidate_indices, dtype=np.int64)
        if candidates.ndim != 1 or np.any(candidates < 0) or np.any(candidates >= history.shape[1]):
            raise ValueError('Invalid candidate_indices')
        scores = self.score(history, user_index)
        # Remove movies the user already rated before selecting the top K.
        seen = history.tocsr().getrow(user_index).indices
        candidates = np.setdiff1d(candidates, seen)
        if not len(candidates) or not np.any(scores[candidates] > 0):
            return []
        order = np.lexsort((candidates, -scores[candidates]))[:k]
        return [(int(j), float(scores[j])) for j in candidates[order]]

    def predict_ratings(self, history, user_index, movie_indices, smoothing=2.0):
        """Predict 1-5 star ratings for specified movie indices.

        This is a separate rating estimator, not a rescaling of ranking scores.
        The content component uses cosine genre similarity; the CF component
        uses learned item similarities. Both predict deviations from the
        user's observed mean, with shrinkage toward that mean.
        """
        indices = np.asarray(movie_indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.content.n_movies):
            raise ValueError('movie_indices must be valid 1-D movie indices')
        if not 0 <= user_index < history.shape[0]:
            raise IndexError('Invalid user_index')
        if history.shape[1] != self.content.n_movies:
            raise ValueError('history and features must share movie columns')
        if smoothing < 0:
            raise ValueError('smoothing must be nonnegative')
        # Estimate CF ratings separately from recommendation ranking scores.
        cf_pred = self.cf.predict_ratings(history, user_index, indices, smoothing)
        if indices.size == 0 or self.alpha == 0:
            return cf_pred
        row = history.tocsr().getrow(user_index)
        # Use the user mean as the baseline, or the training mean if cold-start.
        base = float(row.data.mean()) if row.nnz else self.cf.global_mean_
        if not row.nnz:
            content_pred = np.full(indices.size, base, dtype=np.float64)
        else:
            # Item feature rows are already L2-normalized in content_based.py.
            target_features = self.content.item_features[indices]
            observed_features = self.content.item_features[row.indices]
            # Cosine similarities are dot products of L2-normalized genre vectors.
            sim = target_features @ observed_features.T
            # Adjust the baseline using deviations from the user mean.
            residuals = row.data.astype(np.float64) - base
            numerator = sim @ residuals
            denominator = np.abs(sim).sum(axis=1)
            # Smoothing shrinks weak estimates toward the baseline rating.
            content_pred = np.clip(base + numerator / (denominator + smoothing + 1e-12), 1, 5)
        # Blend two predicted ratings and keep the result on the 1-5 scale.
        return np.clip(self.alpha * content_pred + (1 - self.alpha) * cf_pred, 1, 5)

    def predict_rating(self, history, user_index, movie_index, smoothing=2.0):
        # Convenience wrapper for predicting a single movie.
        return float(self.predict_ratings(history, user_index, [movie_index], smoothing)[0])
