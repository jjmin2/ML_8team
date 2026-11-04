"""Evaluate Hybrid 1-5 star predictions against held-out Netflix ratings.

Usage: .venv\\Scripts\\python.exe evaluate_hybrid.py
       .venv\\Scripts\\python.exe evaluate_hybrid.py --max-users 100
"""
import argparse
import numpy as np
from load_data import load_data
from hybrid_model import HybridRecommender


def main():
    # Read optional command-line settings without editing this file.
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-users', type=int, default=100,
                        help='Number of Netflix users to evaluate (0 = all)')
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()
    # Load the shared train/input/target split created by preprocessing.
    d = load_data()
    model = HybridRecommender(d['genre_features'], alpha=args.alpha)
    model.fit(d['train'])  # MovieLens only; no Netflix target leakage.
    # target contains held-out Netflix ratings used only for evaluation.
    # input contains observed Netflix history used to make predictions.
    target = d['target'].tocsr()
    history = d['input'].tocsr()
    # A value of 0 means all users; otherwise cap the evaluation workload.
    n_users = target.shape[0] if args.max_users == 0 else min(args.max_users, target.shape[0])
    # Stream through users: no dense user x movie arrays.
    absolute_error, squared_error, n_ratings = 0.0, 0.0, 0
    for u in range(n_users):
        # CSR indptr marks the range of held-out ratings for this user.
        start, end = target.indptr[u], target.indptr[u + 1]
        movie_indices = target.indices[start:end]
        actual = target.data[start:end].astype(np.float64)
        if not len(actual):
            continue
        # Predict only held-out movies, using observed history (not target).
        predicted = model.predict_ratings(history, u, movie_indices)
        # MAE uses absolute errors; RMSE uses squared errors.
        errors = predicted - actual
        absolute_error += np.abs(errors).sum()
        squared_error += np.square(errors).sum()
        n_ratings += len(actual)
    if not n_ratings:
        raise RuntimeError('No target ratings in selected users')
    print(f'Netflix users checked: {n_users}')
    print(f'Held-out ratings: {n_ratings}')
    print(f'Hybrid alpha (content weight): {args.alpha:.2f}')
    # Divide by the number of ratings, not by the number of users.
    print(f'MAE:  {absolute_error / n_ratings:.4f}')
    print(f'RMSE: {np.sqrt(squared_error / n_ratings):.4f}')
    print('Note: these are evaluation results, not previously validated accuracy claims.')


if __name__ == '__main__':
    main()
