"""Evaluate Content, Item-CF, and Hybrid recommenders on the held-out Netflix split.

This script never tunes on ``target``. Choose ``--alpha`` with ``tune_hybrid.py``
first, then run this file once for final reporting.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from content_based import ContentBasedRecommender
from evaluation_utils import eligible_targets, ranking_metrics, top_k_from_scores, user_candidates
from hybrid_model import HybridRecommender
from item_based_cf import ItemBasedCFRecommender
from load_data import load_data


def parse_ks(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("--ks must be comma-separated positive integers")
    return values


def make_summary(ks):
    return {
        "rating_count": 0,
        "absolute_error": 0.0,
        "squared_error": 0.0,
        "ranking_users": 0,
        "precision": defaultdict(float),
        "recall": defaultdict(float),
        "ndcg": defaultdict(float),
    }


def update_rating(summary, actual, predicted):
    errors = np.asarray(predicted, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    summary["rating_count"] += len(errors)
    summary["absolute_error"] += float(np.abs(errors).sum())
    summary["squared_error"] += float(np.square(errors).sum())


def write_outputs(output_dir, config, summaries, per_user, samples, excluded):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metric_rows = []
    for name, summary in summaries.items():
        ratings = summary["rating_count"]
        users = summary["ranking_users"]
        row = {
            "model": name,
            "rating_count": ratings,
            "mae": summary["absolute_error"] / ratings if ratings else "",
            "rmse": (summary["squared_error"] / ratings) ** 0.5 if ratings else "",
            "ranking_users": users,
        }
        for k in config["ks"]:
            row[f"precision_at_{k}"] = summary["precision"][k] / users if users else ""
            row[f"recall_at_{k}"] = summary["recall"][k] / users if users else ""
            row[f"ndcg_at_{k}"] = summary["ndcg"][k] / users if users else ""
        metric_rows.append(row)
    write_csv(output_dir / "model_metrics.csv", metric_rows)
    write_csv(output_dir / "per_user_metrics.csv", per_user)
    write_csv(output_dir / "recommendation_samples.csv", samples)
    write_csv(output_dir / "excluded_target_summary.csv", [excluded])


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Path to model_data.npz (default: data/processed/model_data.npz)")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--alpha", type=float, required=True, help="Content weight selected by MovieLens tuning")
    parser.add_argument("--cf-neighbors", type=int, default=100)
    parser.add_argument("--shrinkage", type=float, default=10.0)
    parser.add_argument("--relevant-rating", type=float, default=4.0)
    parser.add_argument("--ks", type=parse_ks, default=(5, 10))
    parser.add_argument("--max-users", type=int, default=1000, help="0 evaluates all users")
    parser.add_argument("--sample-users", type=int, default=3)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1 or not 1 <= args.relevant_rating <= 5 or args.max_users < 0:
        parser.error("invalid evaluation parameter")

    d = load_data(args.data)
    history, target = d["input"].tocsr(), d["target"].tocsr()
    candidates = np.asarray(d["candidate_indices"], dtype=np.int64)
    n_users = target.shape[0] if args.max_users == 0 else min(args.max_users, target.shape[0])
    content = ContentBasedRecommender(d["genre_features"])
    cf = ItemBasedCFRecommender(args.cf_neighbors, args.shrinkage).fit(d["train"])
    hybrid = HybridRecommender(d["genre_features"], args.alpha, args.cf_neighbors, args.shrinkage).fit(d["train"])
    rankers = {"content": content, "item_cf": cf, "hybrid": hybrid}
    summaries = {name: make_summary(args.ks) for name in rankers}
    per_user, samples = [], []
    excluded = {"users_checked": n_users, "target_ratings_total": 0, "target_ratings_outside_candidates": 0,
                "positive_targets_total": 0, "positive_targets_outside_candidates": 0}

    for user in range(n_users):
        hs, he = history.indptr[user : user + 2]
        ts, te = target.indptr[user : user + 2]
        available = user_candidates(candidates, history.indices[hs:he])
        target_movies, actual, dropped = eligible_targets(target.indices[ts:te], target.data[ts:te], available)
        raw_actual = target.data[ts:te]
        excluded["target_ratings_total"] += len(raw_actual)
        excluded["target_ratings_outside_candidates"] += dropped
        excluded["positive_targets_total"] += int((raw_actual >= args.relevant_rating).sum())
        excluded["positive_targets_outside_candidates"] += int((raw_actual >= args.relevant_rating).sum()) - int((actual >= args.relevant_rating).sum())
        relevant = target_movies[actual >= args.relevant_rating]

        if len(target_movies):
            update_rating(summaries["item_cf"], actual, cf.predict_ratings(history, user, target_movies))
            update_rating(summaries["hybrid"], actual, hybrid.predict_ratings(history, user, target_movies))
        if not len(available) or not len(relevant):
            continue

        for name, model in rankers.items():
            recommended = top_k_from_scores(model.score(history, user), available, max(args.ks))
            row = {"model": name, "user_index": user, "netflix_user_id": int(d["netflix_user_ids"][user]),
                   "candidate_count": len(available), "relevant_target_count": len(relevant)}
            for k in args.ks:
                values = ranking_metrics(recommended[:k], relevant, k, len(available))
                row.update({f"precision_at_{k}": values["precision"], f"recall_at_{k}": values["recall"], f"ndcg_at_{k}": values["ndcg"]})
                summaries[name]["precision"][k] += values["precision"]
                summaries[name]["recall"][k] += values["recall"]
                summaries[name]["ndcg"][k] += values["ndcg"]
            summaries[name]["ranking_users"] += 1
            per_user.append(row)
            if user < args.sample_users:
                for rank, movie in enumerate(recommended, start=1):
                    samples.append({"model": name, "user_index": user, "netflix_user_id": int(d["netflix_user_ids"][user]),
                                    "rank": rank, "movie_index": int(movie), "movie_id": int(d["movie_ids"][movie]),
                                    "title": str(d["movie_titles"][movie]), "score": float(model.score(history, user)[movie]),
                                    "is_relevant_target": int(movie in set(relevant.tolist()))})

    config = {"split": "Netflix chronological target (final evaluation)", "alpha": args.alpha,
              "cf_neighbors": args.cf_neighbors, "shrinkage": args.shrinkage,
              "relevant_rating": args.relevant_rating, "ks": list(args.ks), "max_users": args.max_users,
              "candidate_rule": "candidate_indices minus each user's input history", "models": list(rankers)}
    write_outputs(args.output, config, summaries, per_user, samples, excluded)
    print(f"Saved evaluation outputs to {args.output.resolve()}")
    for row in sorted(csv.DictReader((args.output / "model_metrics.csv").open(encoding="utf-8")), key=lambda r: r["model"]):
        print(row)


if __name__ == "__main__":
    main()
