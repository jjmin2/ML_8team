"""Shared model interface. All matrices share the same movie columns."""
from pathlib import Path
import json
import numpy as np
from scipy.sparse import csr_matrix


def load_data(path=None):
    path = Path(path) if path is not None else Path(__file__).resolve().parent / 'data' / 'processed' / 'model_data.npz'
    with np.load(path, allow_pickle=False) as archive:
        result = {k: archive[k] for k in ('movie_ids', 'movie_titles', 'movie_years', 'genre_names', 'genre_features', 'ml_user_ids', 'netflix_user_ids', 'candidate_indices', 'common_movie_indices', 'train_times', 'input_dates', 'target_dates')}
        result['metadata'] = json.loads(str(archive['metadata_json']))
        for name in ('train', 'input', 'target'):
            result[name] = csr_matrix((archive[f'{name}_data'], archive[f'{name}_indices'], archive[f'{name}_indptr']), shape=tuple(archive[f'{name}_shape']))
    return result


if __name__ == '__main__':
    data = load_data()
    for name in ('train', 'input', 'target'):
        print(name, data[name].shape, data[name].nnz)
    print('genre_features', data['genre_features'].shape)
