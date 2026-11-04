"""Item-based collaborative filtering for the ML_8team shared data interface."""
import numpy as np
from scipy import sparse


class ItemBasedCFRecommender:
    def __init__(self, top_neighbors=100, shrinkage=10.0, min_rating=4.0):
        self.top_neighbors = int(top_neighbors)
        self.shrinkage = float(shrinkage)
        self.min_rating = float(min_rating)
        if self.top_neighbors < 1 or self.shrinkage < 0 or not 1 <= self.min_rating <= 5:
            raise ValueError('Invalid model parameters')
        self.similarity = None
        # Fallback rating for users without any observed history.
        self.global_mean_ = 3.5

    def fit(self, train):
        """Learn item similarities from MovieLens ratings only (CSR users x movies)."""
        if not sparse.issparse(train):
            raise TypeError('train must be a sparse rating matrix')
        train = train.tocsr().astype(np.float64)
        # Compute the overall MovieLens rating mean for cold-start predictions.
        self.global_mean_ = float(train.data.mean()) if train.nnz else 3.5
        if train.nnz and (np.any(train.data < 1) or np.any(train.data > 5)):
            raise ValueError('Observed ratings must be between 1 and 5')
        # Positive preference signals: 4 -> 1, 5 -> 2; unrated/low-rated -> 0.
        liked = train.copy()
        liked.data = np.maximum(liked.data - (self.min_rating - 1.0), 0)
        liked.eliminate_zeros()
        # L2-normalize each movie column before computing cosine similarity.
        norms = np.sqrt(np.asarray(liked.power(2).sum(axis=0)).ravel())
        inv = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
        normalized = liked @ sparse.diags(inv)
        # Matrix multiplication gives pairwise item cosine similarities.
        sim = (normalized.T @ normalized).tocsr()
        # Discount similarities supported by very few co-rating users.
        binary = liked.copy()
        binary.data[:] = 1.0
        counts = (binary.T @ binary).tocsr()
        # Shrink similarities with few supporting users toward zero.
        counts.data = counts.data / (counts.data + self.shrinkage)
        sim = sim.multiply(counts).tolil()
        # Exclude self-similarity so a movie cannot recommend itself.
        sim.setdiag(0)
        sim = sim.tocsr()
        sim.eliminate_zeros()
        # Keep strongest neighbors per target movie (row).
        rows, cols, values = [], [], []
        for i in range(sim.shape[0]):
            a, b = sim.indptr[i], sim.indptr[i + 1]
            indices, scores = sim.indices[a:b], sim.data[a:b]
            if len(scores) > self.top_neighbors:
                selected = np.argpartition(scores, -self.top_neighbors)[-self.top_neighbors:]
                indices, scores = indices[selected], scores[selected]
            rows.extend([i] * len(indices))
            cols.extend(indices.tolist())
            values.extend(scores.tolist())
        # Store a sparse matrix to avoid a full movie-by-movie dense array.
        self.similarity = sparse.csr_matrix((values, (rows, cols)), shape=sim.shape)
        return self

    def score(self, history, user_index):
        """Return ranking scores in [0, 1] for all movie columns."""
        if self.similarity is None:
            raise RuntimeError('Call fit(train) first')
        if not sparse.issparse(history) or history.shape[1] != self.similarity.shape[0]:
            raise ValueError('history must share movie columns with training data')
        if not 0 <= user_index < history.shape[0]:
            raise IndexError('Invalid user_index')
        row = history.tocsr().getrow(user_index)
        liked = row.copy()
        liked.data = np.maximum(liked.data - (self.min_rating - 1.0), 0)
        liked.eliminate_zeros()
        if liked.nnz == 0:
            return np.zeros(history.shape[1], dtype=np.float64)
        # Weighted mean of item similarities, using only the user's liked movies.
        weights = liked.data / liked.data.sum()
        profile = sparse.csr_matrix((weights, liked.indices, [0, len(weights)]), shape=liked.shape)
        result = (profile @ self.similarity.T).toarray().ravel()
        return np.clip(result, 0.0, 1.0)

    def predict_ratings(self, history, user_index, movie_indices, smoothing=2.0):
        """Predict 1-5 ratings using similarity-weighted deviations from user mean.

        Only observed history ratings are used. Unseen/unsupported items fall back
        toward the user's mean (or MovieLens global mean for a cold-start user).
        """
        if self.similarity is None:
            raise RuntimeError('Call fit(train) first')
        if not sparse.issparse(history) or history.shape[1] != self.similarity.shape[0]:
            raise ValueError('history must share movie columns with training data')
        if not 0 <= user_index < history.shape[0]:
            raise IndexError('Invalid user_index')
        indices = np.asarray(movie_indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= history.shape[1]):
            raise ValueError('movie_indices must be a 1-D array of valid movie indices')
        if smoothing < 0:
            raise ValueError('smoothing must be nonnegative')
        # The user mean is the baseline; missing ratings are not treated as zeros.
        row = history.tocsr().getrow(user_index)
        base = float(np.mean(row.data)) if row.nnz else self.global_mean_
        if indices.size == 0:
            return np.empty(0, dtype=np.float64)
        if not row.nnz:
            return np.full(indices.size, base, dtype=np.float64)
        # Rows are target movies; columns are history movies.
        # Select only similarities between target movies and rated movies.
        similarities = self.similarity[indices][:, row.indices]
        # A residual shows how far each observed rating is from the user mean.
        residuals = row.data.astype(np.float64) - base
        numerator = np.asarray(similarities @ residuals).ravel()
        denominator = np.asarray(abs(similarities).sum(axis=1)).ravel()
        # Smoothing reduces unstable predictions when similarity support is weak.
        # The small constant prevents division by zero.
        predictions = base + numerator / (denominator + smoothing + 1e-12)
        return np.clip(predictions, 1.0, 5.0)

    def predict_rating(self, history, user_index, movie_index, smoothing=2.0):
        # Convenience wrapper for predicting a single movie.
        return float(self.predict_ratings(history, user_index, [movie_index], smoothing)[0])
