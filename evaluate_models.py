"""
held-out 넷플릭스(Netflix) 데이터셋을 대상으로 Content-Based, Item-CF, Hybrid 추천 모델의 
최종 평가(평점 예측 오차 MAE/RMSE 및 Top-K 랭킹 지표 Precision/Recall/NDCG)를 수행합니다.

본 스크립트는 target 데이터셋으로 하이퍼파라미터를 튜닝하지 않습니다 (Data Leakage 방지).
tune_hybrid.py를 통해 최적의 alpha 값을 먼저 선정한 후, 본 스크립트를 단 1회 실행하여 최종 지표를 수출합니다.
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
    """
    CLI 인자 --ks로 전달된 쉼표 구분 문자열(예: "5,10")을 파싱하여
    중복 제거 및 정렬된 양의 정수 튜플로 반환합니다.
    """
    values = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("--ks must be comma-separated positive integers")
    return values


def make_summary(ks):
    """
    각 모델별 성능 지표 집계용 딕셔너리 구조를 생성합니다.
    - rating_count: 평가된 총 평점 개수
    - absolute_error: 절대 오차(Absolute Error) 누적합 (MAE 계산용)
    - squared_error: 제곱 오차(Squared Error) 누적합 (RMSE 계산용)
    - ranking_users: 랭킹 평가가 수행된 총 사용자 수
    - precision, recall, ndcg: K값별 지표 누적 딕셔너리
    """
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
    """
    실제 평점(actual)과 모델의 예측 평점(predicted) 간의 오차를 계산하여
    MAE 및 RMSE 누적값과 평가 건수를 업데이트합니다.
    """
    errors = np.asarray(predicted, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    summary["rating_count"] += len(errors)
    summary["absolute_error"] += float(np.abs(errors).sum())
    summary["squared_error"] += float(np.square(errors).sum())


def write_outputs(output_dir, config, summaries, per_user, samples, excluded):
    """
    최종 평가 결과 및 메타데이터를 지정된 출력 디렉터리(output_dir)에 JSON 및 CSV 파일로 저장합니다.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 평가 설정 메타데이터 저장 (evaluation_config.json)
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    
    # 2. 모델별 요약 성능 지표 집계 (model_metrics.csv)
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
    # 3. 사용자별 개별 성능 지표 저장 (per_user_metrics.csv)
    write_csv(output_dir / "per_user_metrics.csv", per_user)
    # 4. 상위 샘플 사용자 추천 결과 및 타겟 적중 여부 저장 (recommendation_samples.csv)
    write_csv(output_dir / "recommendation_samples.csv", samples)
    # 5. 후보군 밖 타겟 제외 통계 저장 (excluded_target_summary.csv)
    write_csv(output_dir / "excluded_target_summary.csv", [excluded])


def write_csv(path: Path, rows):
    """
    딕셔너리 리스트 형태의 데이터를 지정된 경로에 CSV 파일로 출력하는 유틸리티 함수입니다.
    """
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    # 1. CLI 명령어 인자(Argument) 파싱 설정
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="전처리 데이터 파일 경로 (기본값: data/processed/model_data.npz)")
    parser.add_argument("--output", type=Path, default=Path("results"), help="산출물 저장 디렉터리 경로")
    parser.add_argument("--alpha", type=float, required=True, help="MovieLens 튜닝으로 확정된 Content 가중치 (0.0~1.0)")
    parser.add_argument("--cf-neighbors", type=int, default=100, help="Item-CF 최빈 이웃(k-NN) 개수")
    parser.add_argument("--shrinkage", type=float, default=10.0, help="코사인 유사도 Shrinkage 페널티 계수")
    parser.add_argument("--relevant-rating", type=float, default=4.0, help="선호 타겟으로 간주할 최소 평점 임계값")
    parser.add_argument("--ks", type=parse_ks, default=(5, 10), help="평가할 상위 추천 개수 K 목록 (예: '5,10')")
    parser.add_argument("--max-users", type=int, default=1000, help="평가 대상 사용자 수 제한 (0 입력 시 전체 사용자 평가)")
    parser.add_argument("--sample-users", type=int, default=3, help="상세 추천 결과를 추출할 샘플 사용자 수")
    args = parser.parse_args()
    
    # 입력 인자 범위 유효성 검증
    if not 0 <= args.alpha <= 1 or not 1 <= args.relevant_rating <= 5 or args.max_users < 0:
        parser.error("invalid evaluation parameter")

    # 2. 전처리 데이터 로드 및 CSR 스파스 행렬 구성
    d = load_data(args.data)
    history, target = d["input"].tocsr(), d["target"].tocsr()  # Netflix input(시청 이력) 및 target(정답)
    candidates = np.asarray(d["candidate_indices"], dtype=np.int64)  # 공통 평가 대상 영화 후보 인덱스
    n_users = target.shape[0] if args.max_users == 0 else min(args.max_users, target.shape[0])

    # 3. 모델 객체 생성 및 학습(fit)
    content = ContentBasedRecommender(d["genre_features"])
    # Item-CF 및 Hybrid 모델은 오직 MovieLens train 데이터로만 fit 수행 (Leakage 방지)
    cf = ItemBasedCFRecommender(args.cf_neighbors, args.shrinkage).fit(d["train"])
    hybrid = HybridRecommender(d["genre_features"], args.alpha, args.cf_neighbors, args.shrinkage).fit(d["train"])
    
    rankers = {"content": content, "item_cf": cf, "hybrid": hybrid}
    summaries = {name: make_summary(args.ks) for name in rankers}
    per_user, samples = [], []
    
    # 후보군 밖 타겟 제외 건수를 통계화하기 위한 카운터
    excluded = {
        "users_checked": n_users,
        "target_ratings_total": 0,
        "target_ratings_outside_candidates": 0,
        "positive_targets_total": 0,
        "positive_targets_outside_candidates": 0,
    }

    
    # 4. 사용자별 순회 평가 루프 (User-level Evaluation Loop)
    
    for user in range(n_users):
        # CSR 행렬 인덱스 포인터 구간 슬라이싱
        hs, he = history.indptr[user : user + 2]
        ts, te = target.indptr[user : user + 2]
        
        # 4-1. 해당 사용자가 이미 시청한 영화를 추천 후보군에서 마스킹(제외)
        available = user_candidates(candidates, history.indices[hs:he])
        
        # 4-2. target 평점 중 실제 추천 가능 후보군 내에 있는 항목만 선별
        target_movies, actual, dropped = eligible_targets(target.indices[ts:te], target.data[ts:te], available)
        raw_actual = target.data[ts:te]
        
        # 제외 항목 통계 누적
        excluded["target_ratings_total"] += len(raw_actual)
        excluded["target_ratings_outside_candidates"] += dropped
        excluded["positive_targets_total"] += int((raw_actual >= args.relevant_rating).sum())
        excluded["positive_targets_outside_candidates"] += int((raw_actual >= args.relevant_rating).sum()) - int((actual >= args.relevant_rating).sum())
        
        # 임계값(4.0점) 이상 고평가된 정답 영화 집합 추출
        relevant = target_movies[actual >= args.relevant_rating]

        # 4-3. 평점 예측 오차 평가 (Item-CF 및 Hybrid 모델 대상)
        if len(target_movies):
            update_rating(summaries["item_cf"], actual, cf.predict_ratings(history, user, target_movies))
            update_rating(summaries["hybrid"], actual, hybrid.predict_ratings(history, user, target_movies))
            
        # 추천 가능 후보군이 없거나, 선호 정답(relevant) 영화가 없는 사용자는 랭킹 평가에서 제외
        if not len(available) or not len(relevant):
            continue

        # 4-4. 3개 모델별 Top-K 랭킹 추천 및 정밀도 지표(Precision, Recall, NDCG) 계산
        for name, model in rankers.items():
            # 모델 랭킹 점수 기반 상위 K개 영화 추출
            recommended = top_k_from_scores(model.score(history, user), available, max(args.ks))
            
            row = {
                "model": name,
                "user_index": user,
                "netflix_user_id": int(d["netflix_user_ids"][user]),
                "candidate_count": len(available),
                "relevant_target_count": len(relevant),
            }
            
            # K=5, 10 각 기준별 정밀도 수치 계산 및 누적
            for k in args.ks:
                values = ranking_metrics(recommended[:k], relevant, k, len(available))
                row.update({
                    f"precision_at_{k}": values["precision"],
                    f"recall_at_{k}": values["recall"],
                    f"ndcg_at_{k}": values["ndcg"],
                })
                summaries[name]["precision"][k] += values["precision"]
                summaries[name]["recall"][k] += values["recall"]
                summaries[name]["ndcg"][k] += values["ndcg"]
                
            summaries[name]["ranking_users"] += 1
            per_user.append(row)
            
            # 지정된 수(sample_users) 만큼의 상위 유저에 대해 상세 추천 목록 샘플 추출
            if user < args.sample_users:
                for rank, movie in enumerate(recommended, start=1):
                    samples.append({
                        "model": name,
                        "user_index": user,
                        "netflix_user_id": int(d["netflix_user_ids"][user]),
                        "rank": rank,
                        "movie_index": int(movie),
                        "movie_id": int(d["movie_ids"][movie]),
                        "title": str(d["movie_titles"][movie]),
                        "score": float(model.score(history, user)[movie]),
                        "is_relevant_target": int(movie in set(relevant.tolist())),
                    })

    # 5. 최종 평가 설정 메타데이터 구성 및 출력 저장
    config = {
        "split": "Netflix chronological target (final evaluation)",
        "alpha": args.alpha,
        "cf_neighbors": args.cf_neighbors,
        "shrinkage": args.shrinkage,
        "relevant_rating": args.relevant_rating,
        "ks": list(args.ks),
        "max_users": args.max_users,
        "candidate_rule": "candidate_indices minus each user's input history",
        "models": list(rankers),
    }
    
    write_outputs(args.output, config, summaries, per_user, samples, excluded)
    print(f"Saved evaluation outputs to {args.output.resolve()}")
    
    # 콘솔 터미널에 요약 결과(model_metrics.csv) 출력
    for row in sorted(csv.DictReader((args.output / "model_metrics.csv").open(encoding="utf-8")), key=lambda r: r["model"]):
        print(row)


if __name__ == "__main__":
    main()
