"""Encode cleaned CSVs into one portable, verified model input archive."""
import argparse
from array import array
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


def pack_ratings(path, items, users=None):
    ids, values, times, pointers = array('i'), array('f'), array('q'), array('i', [0])
    user_ids = []
    previous = None
    with path.open(encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for raw_user, raw_movie, raw_rating, raw_time in reader:
            user, movie, rating = int(raw_user), int(raw_movie), float(raw_rating)
            if user != previous:
                if previous is not None:
                    pointers.append(len(ids))
                    if user <= previous:
                        raise ValueError('CSV must be grouped by ascending user_id')
                user_ids.append(user)
                previous = user
            ids.append(items[movie])
            values.append(rating)
            times.append(int(raw_time.replace('-', '')))
    pointers.append(len(ids))
    actual_users = np.asarray(user_ids, dtype=np.int64)
    if users is not None and not np.array_equal(users, actual_users):
        raise ValueError('Netflix input/target user mappings differ')
    matrix = csr_matrix((np.frombuffer(values, dtype=np.float32), np.frombuffer(ids, dtype=np.int32), np.frombuffer(pointers, dtype=np.int32)), shape=(len(user_ids), len(items)))
    matrix.check_format(full_check=True)
    if not np.all((matrix.data >= 1) & (matrix.data <= 5)):
        raise ValueError('Invalid rating')
    print(f'{path.name}: shape={matrix.shape}, ratings={matrix.nnz:,}', flush=True)
    return matrix, actual_users, np.frombuffer(times, dtype=np.int64)


def prepare(folder, cleanup=True):
    report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    if report.get('status') != 'complete':
        raise ValueError('Full preprocessing must complete before encoding')
    with (folder / 'movies.csv').open(encoding='utf-8', newline='') as f:
        movies = sorted(csv.DictReader(f), key=lambda r: int(r['movie_id']))
    movie_ids = np.array([int(r['movie_id']) for r in movies], dtype=np.int64)
    items = {int(mid): i for i, mid in enumerate(movie_ids)}
    genres = sorted({g for row in movies for g in row['genres'].split('|')})
    genre_index = {g: i for i, g in enumerate(genres)}
    features = np.zeros((len(movies), len(genres)), dtype=np.float32)
    for i, row in enumerate(movies):
        for g in row['genres'].split('|'):
            features[i, genre_index[g]] = 1
    train, ml_users, train_time = pack_ratings(folder / 'movielens_ratings.csv', items)
    history, nf_users, input_time = pack_ratings(folder / 'netflix_input.csv', items)
    target, _, target_time = pack_ratings(folder / 'netflix_target.csv', items, nf_users)
    expected = [report['movielens']['ratings'], report['netflix']['split_rows']['input'], report['netflix']['split_rows']['target']]
    if [train.nnz, history.nnz, target.nnz] != expected:
        raise ValueError('Rating counts disagree with preprocessing report')
    if history.copy().multiply(target.copy()).nnz:
        raise ValueError('Netflix input/target overlap')
    for row in range(len(nf_users)):
        a, b = history.indptr[row:row+2]
        c, d = target.indptr[row:row+2]
        if not a < b or not c < d or input_time[a:b].max() >= target_time[c:d].min():
            raise ValueError('Chronological evaluation leakage')
    with (folder / 'movie_mapping.csv').open(encoding='utf-8', newline='') as f:
        mapping = list(csv.DictReader(f))
    common = np.array(sorted(items[int(r['movielens_id'])] for r in mapping), dtype=np.int32)
    supported = np.asarray(train.getnnz(axis=0)).ravel() > 0
    payload = dict(movie_ids=movie_ids, movie_titles=np.array([r['title'] for r in movies]), movie_years=np.array([int(r['year']) for r in movies], dtype=np.int32), genre_names=np.array(genres), genre_features=features, ml_user_ids=ml_users, netflix_user_ids=nf_users, common_movie_indices=common, candidate_indices=common[supported[common]], metadata_json=np.array(json.dumps(report, ensure_ascii=False)), train_times=train_time, input_dates=input_time, target_dates=target_time)
    for name, matrix in [('train', train), ('input', history), ('target', target)]:
        for key in ('data', 'indices', 'indptr'):
            payload[f'{name}_{key}'] = getattr(matrix, key)
        payload[f'{name}_shape'] = np.array(matrix.shape, dtype=np.int64)
    temporary = folder / 'model_data.tmp.npz'
    destination = folder / 'model_data.npz'
    np.savez_compressed(temporary, **payload)
    # Read every stored array before removing the redundant CSV intermediates.
    with np.load(temporary, allow_pickle=False) as saved:
        for name, expected_array in payload.items():
            if not np.array_equal(saved[name], expected_array):
                raise ValueError(f'Archive verification failed: {name}')
    temporary.replace(destination)
    if cleanup:
        for name in ('movies.csv', 'movielens_ratings.csv', 'netflix_input.csv', 'netflix_target.csv', 'preprocessing.sqlite', 'preprocessing.sqlite-journal', 'unmatched_netflix_movies.csv'):
            path = folder / name
            if path.is_file():
                path.unlink()
    print(f'Verified model input: {destination} ({destination.stat().st_size:,} bytes)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'data' / 'processed')
    args = parser.parse_args()
    prepare(args.output)
