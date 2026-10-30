"""MovieLens training data and Netflix external evaluation preprocessing."""
import argparse
import csv
import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path


def write_csv(path, header, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def title_key(title):
    title = unicodedata.normalize('NFKC', title).casefold().strip()
    title = re.sub(r'^(.*), (the|a|an)$', r'\2 \1', title)
    return ' '.join(re.findall(r'\w+', title))


def fingerprint(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    report = {'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, 'sources': []}
    movies = {}
    keys = defaultdict(list)
    for line in (args.data / 'movies.dat').read_text(encoding='latin-1').splitlines():
        mid, raw, genres = line.split('::')
        m = re.fullmatch(r'(.*?)\s*\((\d{4})\)', raw)
        if not m or int(mid) in movies or not genres:
            raise ValueError(f'Invalid or duplicate movie: {line}')
        title, year = m.group(1), int(m.group(2))
        movies[int(mid)] = (title, year, genres)
        keys[(title_key(title), year)].append(int(mid))
    write_csv(out / 'movies.csv', ['movie_id', 'title', 'year', 'genres'], ((i, *v) for i, v in movies.items()))
    # A temporary on-disk database keeps Netflix processing memory bounded.
    db = sqlite3.connect(out / 'preprocessing.sqlite')
    try:
        db.execute('PRAGMA cache_size=-131072')
        db.execute('PRAGMA synchronous=NORMAL')
        db.executescript('DROP TABLE IF EXISTS ml; DROP TABLE IF EXISTS nf; DROP TABLE IF EXISTS ranked; CREATE TABLE ml(user_id INTEGER, movie_id INTEGER, rating INTEGER, time INTEGER, PRIMARY KEY(user_id,movie_id)); CREATE TABLE nf(user_id INTEGER, movie_id INTEGER, rating INTEGER, time TEXT, PRIMARY KEY(movie_id,user_id)) WITHOUT ROWID;')
        with db:
            batch = []
            with (args.data / 'ratings.dat').open(encoding='ascii') as f:
                for line in f:
                    u, m, r, t = map(int, line.strip().split('::'))
                    if u <= 0 or m not in movies or not 1 <= r <= 5 or t < 0:
                        raise ValueError(f'Invalid MovieLens rating: {line}')
                    batch.append((u, m, r, t))
                    if len(batch) == 50000:
                        db.executemany('INSERT INTO ml VALUES(?,?,?,?)', batch)
                        batch.clear()
                db.executemany('INSERT INTO ml VALUES(?,?,?,?)', batch)
        header = ['user_id', 'movie_id', 'rating', 'timestamp']
        write_csv(out / 'movielens_ratings.csv', header, db.execute('SELECT * FROM ml ORDER BY user_id,time,movie_id'))
        # Netflix is external evaluation; all MovieLens ratings remain training data.
        report['movielens'] = {'movies': len(movies), 'ratings': db.execute('SELECT COUNT(*) FROM ml').fetchone()[0], 'users': db.execute('SELECT COUNT(DISTINCT user_id) FROM ml').fetchone()[0]}
        report['sources'] = [fingerprint(args.data / n) for n in ('movies.dat', 'ratings.dat')]
        print('MovieLens:', report['movielens'], flush=True)
        if args.movielens_only:
            report['status'] = 'movielens_only'
            return report
        nf_dir = args.netflix_dir
        if nf_dir is None:
            import kagglehub
            nf_dir = Path(kagglehub.dataset_download(args.netflix_handle))
        title_files = list(nf_dir.rglob('movie_titles.csv'))
        if len(title_files) != 1:
            raise ValueError('Expected exactly one movie_titles.csv')
        candidates = defaultdict(list)
        netflix_count = 0
        unmatched = []
        with title_files[0].open(encoding='latin-1') as f:
            for line in f:
                nid, year, title = line.rstrip('\r\n').split(',', 2)
                netflix_count += 1
                hits = keys.get((title_key(title), int(year) if year.isdigit() else None), [])
                if len(hits) == 1:
                    candidates[hits[0]].append((int(nid), title, year))
                else:
                    unmatched.append((nid, title, year, 'ambiguous_movielens' if hits else 'no_exact_match'))
        mapping = {}
        matched = []
        for mid, entries in candidates.items():
            if len(entries) == 1:
                nid, title, year = entries[0]
                mapping[nid] = mid
                matched.append((mid, nid, movies[mid][0], title, year))
            else:
                unmatched.extend((nid, title, year, 'ambiguous_netflix') for nid, title, year in entries)
        write_csv(out / 'movie_mapping.csv', ['movielens_id', 'netflix_id', 'movielens_title', 'netflix_title', 'year'], sorted(matched))
        write_csv(out / 'unmatched_netflix_movies.csv', ['netflix_id', 'title', 'year', 'reason'], unmatched)
        parts = [nf_dir / f'combined_data_{i}.txt' for i in range(1, 5)]
        if not all(p.is_file() for p in parts):
            parts = sorted(nf_dir.rglob('combined_data_*.txt'))
        if {p.name for p in parts} != {f'combined_data_{i}.txt' for i in range(1, 5)} or len(parts) != 4:
            raise ValueError('All four combined_data files are required; partial evaluation is rejected')
        total = retained = 0
        for part in parts:
            mid = None
            batch = []
            with db, part.open(encoding='ascii') as f:
                for line in f:
                    line = line.strip()
                    if line.endswith(':'):
                        mid = mapping.get(int(line[:-1]))
                        continue
                    total += 1
                    if mid is None:
                        continue
                    u, r, date = line.split(',')
                    if int(u) <= 0 or not 1 <= int(r) <= 5 or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date):
                        raise ValueError(f'Invalid Netflix rating: {line}')
                    batch.append((int(u), mid, int(r), date))
                    retained += 1
                    if len(batch) == 50000:
                        db.executemany('INSERT INTO nf VALUES(?,?,?,?)', batch)
                        batch.clear()
                db.executemany('INSERT INTO nf VALUES(?,?,?,?)', batch)
            print(f'{part.name}: scanned={total:,}, matched={retained:,}', flush=True)
        if not retained:
            raise ValueError('No matched Netflix ratings')
        # Hold out entire dates: no same-day rating can straddle input and target.
        db.executescript('CREATE TABLE ranked AS SELECT *, ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY time,movie_id) AS rn, COUNT(*) OVER(PARTITION BY user_id) AS n FROM nf; CREATE INDEX ranked_user ON ranked(user_id);')
        db.execute('CREATE TEMP TABLE cutoff AS SELECT user_id, MAX(time) AS date FROM ranked WHERE n>=? AND rn<=CAST(n*? AS INTEGER) GROUP BY user_id', (args.min_ratings, args.input_fraction))
        db.execute('CREATE TEMP TABLE eligible AS SELECT c.* FROM cutoff c WHERE EXISTS(SELECT 1 FROM ranked r WHERE r.user_id=c.user_id AND r.time>c.date)')
        counts = {}
        for name, op in [('input', '<='), ('target', '>')]:
            query = f'SELECT r.user_id,r.movie_id,r.rating,r.time FROM ranked r JOIN eligible e USING(user_id) WHERE r.time {op} e.date ORDER BY r.user_id,r.time,r.movie_id'
            write_csv(out / f'netflix_{name}.csv', ['user_id', 'movie_id', 'rating', 'date'], db.execute(query))
            counts[name] = db.execute(f'SELECT COUNT(*) FROM ranked r JOIN eligible e USING(user_id) WHERE r.time {op} e.date').fetchone()[0]
        users = db.execute('SELECT COUNT(*) FROM eligible').fetchone()[0]
        if not users:
            raise ValueError('No eligible evaluation users after chronological split')
        report['netflix'] = {'movies': netflix_count, 'matched_movies': len(mapping), 'movie_coverage': len(mapping) / netflix_count, 'ratings_scanned': total, 'ratings_matched': retained, 'rating_coverage': retained / total, 'matched_users': db.execute('SELECT COUNT(DISTINCT user_id) FROM nf').fetchone()[0], 'evaluation_users': users, 'split_rows': counts}
        report['sources'] += [fingerprint(p) for p in [title_files[0], *parts]]
        report['status'] = 'complete'
        return report
    finally:
        db.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    p.add_argument('--data', type=Path, default=root / 'data')
    p.add_argument('--output', type=Path, default=root / 'data' / 'processed')
    p.add_argument('--netflix-dir', type=Path)
    p.add_argument('--netflix-handle', default='netflix-inc/netflix-prize-data/versions/2')
    p.add_argument('--movielens-only', action='store_true')
    p.add_argument('--input-fraction', type=float, default=.8)
    p.add_argument('--min-ratings', type=int, default=5)
    a = p.parse_args()
    if not 0 < a.input_fraction < 1 or a.min_ratings < 2:
        p.error('input-fraction must be between 0 and 1; min-ratings must be >= 2')
    # An incomplete run must never leave a stale success marker.
    a.output.mkdir(parents=True, exist_ok=True)
    marker = a.output / 'report.json'
    marker.write_text(json.dumps({'status': 'running'}, indent=2), encoding='utf-8')
    try:
        report = run(a)
        marker.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if report['status'] == 'complete':
            from prepare_model_data import prepare
            prepare(a.output)
    except Exception as exc:
        marker.write_text(json.dumps({'status': 'failed', 'error': str(exc)}, ensure_ascii=False, indent=2), encoding='utf-8')
        raise
    print('Report:', marker)


if __name__ == '__main__':
    main()
