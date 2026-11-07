"""
MovieLens 내부 시계열 분할(Chronological Split) 데이터만을 활용하여 
하이브리드 추천 모델의 최적 콘텐츠 가중치(alpha)를 탐색하고 확정하는 튜닝 스크립트입니다.

Data Leakage를 방지하기 위해 넷플릭스(Netflix) target 데이터는 일절 참조하지 않습니다.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from content_based import ContentBasedRecommender
from evaluation_utils import chronological_split, eligible_targets, ranking_metrics, top_k_from_scores, user_candidates
from item_based_cf import ItemBasedCFRecommender
from load_data import load_data


def parse_alphas(value):
    alphas = tuple(sorted({float(part.strip()) for part in value.split(",") if part.strip()}))
    if not alphas or alphas[0] < 0 or alphas[-1] > 1:
        raise argparse.ArgumentTypeError("alphas must be values in [0, 1]")
    return alphas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--alphas", type=parse_alphas, default=tuple(i / 10 for i in range(11)))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=1000, help="0 evaluates all eligible MovieLens users")
    parser.add_argument("--fraction", type=float, default=0.8)
    parser.add_argument("--min-ratings", type=int, default=5)
    parser.add_argument("--relevant-rating", type=float, default=4.0)
    parser.add_argument("--cf-neighbors", type=int, default=100)
    parser.add_argument("--shrinkage", type=float, default=10.0)
    args = parser.parse_args()
    if args.k < 1 or args.max_users < 0:
        parser.error("k must be positive and max-users cannot be negative")

    d = load_data(args.data)
    history, target = chronological_split(d["train"], d["train_times"], args.fraction, args.min_ratings)
    candidates = np.asarray(d["candidate_indices"], dtype=np.int64)
    content = ContentBasedRecommender(d["genre_features"])
    cf = ItemBasedCFRecommender(args.cf_neighbors, args.shrinkage).fit(history)
    totals = {alpha: {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "users": 0} for alpha in args.alphas}
    max_users = history.shape[0] if args.max_users == 0 else min(args.max_users, history.shape[0])

    for user in range(max_users):
        hs, he = history.indptr[user : user + 2]
        ts, te = target.indptr[user : user + 2]
        available = user_candidates(candidates, history.indices[hs:he])
        movies, ratings, _ = eligible_targets(target.indices[ts:te], target.data[ts:te], available)
        relevant = movies[ratings >= args.relevant_rating]
        if not len(available) or not len(relevant):
            continue
        content_scores, cf_scores = content.score(history, user), cf.score(history, user)
        for alpha in args.alphas:
            recommended = top_k_from_scores(alpha * content_scores + (1 - alpha) * cf_scores, available, args.k)
            values = ranking_metrics(recommended, relevant, args.k, len(available))
            for name, value in values.items():
                totals[alpha][name] += value
            totals[alpha]["users"] += 1

    rows = []
    for alpha, result in totals.items():
        users = result["users"]
        rows.append({"alpha_content": alpha, "validation_users": users,
                     "precision_at_k": result["precision"] / users if users else 0.0,
                     "recall_at_k": result["recall"] / users if users else 0.0,
                     "ndcg_at_k": result["ndcg"] / users if users else 0.0})
    rows.sort(key=lambda row: (-row["ndcg_at_k"], -row["precision_at_k"], row["alpha_content"]))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "tuning_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    params = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    chosen = {"selected_alpha": rows[0]["alpha_content"], "selection_metric": f"NDCG@{args.k}",
          "split": "MovieLens chronological internal validation", "parameters": params}
    (args.output / "selected_hybrid.json").write_text(json.dumps(chosen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(chosen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
