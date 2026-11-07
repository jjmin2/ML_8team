"""
추천 시스템 평가(Rating 오차 및 Top-K 랭킹 지표)와 Data Leakage를 방지하는
시계열 데이터 분할(Chronological Split) 유틸리티 함수 모음입니다.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def user_candidates(candidate_indices: np.ndarray, seen_indices: np.ndarray) -> np.ndarray:
    """
    전체 공통 평가 대상 영화 중, 사용자가 이미 시청/평가한 영화를 제외한
    추천 가능 후보 영화 인덱스 배열을 반환합니다.

    Parameters:
        candidate_indices: 평가 공통 영화 인덱스 배열
        seen_indices: 해당 사용자가 과거 시청한 영화 인덱스 배열 (history/input)

    Returns:
        np.ndarray: 마스킹(제외) 처리된 추천 후보 영화 인덱스 배열
    """
    return np.setdiff1d(
        np.asarray(candidate_indices, dtype=np.int64),
        np.asarray(seen_indices, dtype=np.int64),
        assume_unique=False,
    )


def eligible_targets(
    target_indices: np.ndarray,
    target_ratings: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Held-out 타겟(target) 평점 데이터 중, 선언된 추천 후보군(candidates) 내에 
    존재하는 영화 항목만 필터링하여 남깁니다.

    Parameters:
        target_indices: 사용자의 검증 타겟 영화 인덱스 배열
        target_ratings: 사용자의 검증 타겟 실제 평점 배열
        candidates: 사용자의 유효 추천 후보 영화 인덱스 배열

    Returns:
        tuple: (유효 타겟 영화 인덱스, 유효 타겟 실제 평점, 후보군 밖으로 제외된 타겟 개수)
    """
    target_indices = np.asarray(target_indices, dtype=np.int64)
    target_ratings = np.asarray(target_ratings, dtype=np.float64)
    
    # 후보군 내에 포함되어 있는지 여부 마스킹 생성
    mask = np.isin(target_indices, candidates, assume_unique=False)
    
    # 유효한 항목만 반환하고, 제외된 항목 개수(~mask.sum())를 계산
    return target_indices[mask], target_ratings[mask], int((~mask).sum())


def top_k_from_scores(scores: np.ndarray, candidates: np.ndarray, k: int) -> np.ndarray:
    """
    후보 영화들의 모델 예측 점수(scores)를 바탕으로 상위 K개 영화 인덱스를 정렬하여 반환합니다.
    점수가 동일(Tie)할 경우 영화 열 인덱스(movie_index) 오름차순으로 순위를 결정합니다.

    Parameters:
        scores: 전체 영화에 대한 모델의 예측 점수 배열
        candidates: 추천 대상 후보 영화 인덱스 배열
        k: 추출할 상위 추천 개수

    Returns:
        np.ndarray: 정렬된 상위 K개 추천 영화 인덱스 (양수 점수가 없으면 빈 배열)
    """
    if k < 1:
        raise ValueError("k must be positive")
    candidates = np.asarray(candidates, dtype=np.int64)
    if not candidates.size:
        return np.empty(0, dtype=np.int64)
    
    # 후보 영화들에 해당하는 예측 점수 추출
    values = np.asarray(scores, dtype=np.float64)[candidates]
    positive = values > 0
    
    # 양수 점수를 가진 영화가 하나도 없는 경우 (신호 없음 -> 0점 처리)
    if not np.any(positive):
        return np.empty(0, dtype=np.int64)
        
    candidates, values = candidates[positive], values[positive]
    
    # np.lexsort: 1순위 -values (내림차순 정렬), 2순위 candidates (인덱스 오름차순 동점 처리)
    order = np.lexsort((candidates, -values))[:k]
    return candidates[order]


def ranking_metrics(
    recommended: Iterable[int], relevant: Iterable[int], k: int, candidate_count: int
) -> dict[str, float]:
    """
    단일 사용자에 대한 이진 선호도(Relevant vs Non-relevant) 기반 랭킹 지표 
    (Precision@K, Recall@K, NDCG@K)를 계산합니다.

    Parameters:
        recommended: 상위 K개 추천된 영화 인덱스 목록
        relevant: 사용자가 실제 선호(예: 평점 4.0 이상)한 정답 영화 인덱스 목록
        k: 평가 기준 상위 추천 개수 K
        candidate_count: 추천 가능했던 유효 후보 영화 총 개수

    Returns:
        dict[str, float]: {"precision": float, "recall": float, "ndcg": float}
    """
    recommended = np.asarray(list(recommended), dtype=np.int64)
    relevant = np.asarray(list(relevant), dtype=np.int64)
    
    if k < 1 or candidate_count < 0:
        raise ValueError("invalid ranking evaluation configuration")
        
    relevant_set = set(relevant.tolist())
    
    # 추천 목록 중 실제 선호 영화에 해당하는지 적중 여부(1 또는 0) 확인
    hits = np.fromiter((int(movie in relevant_set) for movie in recommended), dtype=np.int8)
    hit_count = int(hits.sum())
    
    # Precision@K: 적중 수 / min(K, 후보 개수)
    denominator = min(k, candidate_count)
    precision = hit_count / denominator if denominator else 0.0
    
    # Recall@K: 적중 수 / 전체 정답 선호 영화 수
    recall = hit_count / len(relevant_set) if relevant_set else 0.0
    
    # DCG (Discounted Cumulative Gain) 계산: 순위(position)에 따른 감쇄율 적용 (log2(rank + 1))
    discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
    dcg = float((hits * discounts).sum())
    
    # IDCG (Ideal DCG) 계산: 이상적인 정답 순치 배치에서의 DCG 값
    ideal_length = min(len(relevant_set), k, candidate_count)
    idcg = float((1.0 / np.log2(np.arange(2, ideal_length + 2))).sum())
    
    # NDCG@K: DCG / IDCG
    return {"precision": precision, "recall": recall, "ndcg": dcg / idcg if idcg else 0.0}


def chronological_split(matrix, times: np.ndarray, fraction: float = 0.8, min_ratings: int = 5):
    """
    CSR 스파스 행렬을 사용자별 타임스탬프(times) 기준으로 과거(history/train)와 
    미래(target/validation) 세트로 시계열 분할합니다.
    동일한 날짜/시간(timestamp)을 가진 평점이 두 세트에 걸쳐 쪼개지지 않도록 조절합니다.

    이 함수는 MovieLens 내부 하이퍼파라미터 튜닝용 시계열 검증(Validation) 분할 전용입니다.

    Parameters:
        matrix: 평점 CSR 스파스 행렬 (users x movies)
        times: CSR matrix.data와 1:1 대응하는 Unix 타임스탬프 배열
        fraction: 과거 학습용으로 할당할 시계열 비율 (기본값: 0.8 / 80%)
        min_ratings: 시계열 분할을 적용할 최소 평점 개수 (미만일 경우 전량 history로 할당)

    Returns:
        tuple[csr_matrix, csr_matrix]: (history_matrix, target_matrix)
    """
    from scipy.sparse import csr_matrix

    if not 0 < fraction < 1 or min_ratings < 2:
        raise ValueError("invalid split configuration")
        
    matrix = matrix.tocsr()
    times = np.asarray(times, dtype=np.int64)
    
    # CSR 행렬의 non-zero 원소 개수와 타임스탬프 개수 일치 검증
    if len(times) != matrix.nnz:
        raise ValueError("times must align with CSR data")
        
    history_data, history_indices, history_indptr = [], [], [0]
    target_data, target_indices, target_indptr = [], [], [0]
    
    # 사용자별 행 순회
    for user in range(matrix.shape[0]):
        start, end = matrix.indptr[user : user + 2]
        values, indices, user_times = matrix.data[start:end], matrix.indices[start:end], times[start:end]
        
        # 평점 개수가 최소 기준 미만인 경우 전체를 history로 할당
        if len(values) < min_ratings:
            history_mask = np.ones(len(values), dtype=bool)
        else:
            # 80% 분할 시점의 타임스탬프 기준점 계산
            cutoff_position = max(1, int(math.floor(len(values) * fraction))) - 1
            cutoff = user_times[cutoff_position]
            
            # 동일 타임스탬프 훼손을 방지하기 위해 <= cutoff로 분할 마스크 생성
            history_mask = user_times <= cutoff
            if history_mask.all():
                history_mask = np.ones(len(values), dtype=bool)
                
        target_mask = ~history_mask
        
        # history (과거 시청 이력) CSR 구조 누적
        history_data.extend(values[history_mask])
        history_indices.extend(indices[history_mask])
        history_indptr.append(len(history_data))
        
        # target (미래 정답 이력) CSR 구조 누적
        target_data.extend(values[target_mask])
        target_indices.extend(indices[target_mask])
        target_indptr.append(len(target_data))
        
    shape = matrix.shape
    return (
        csr_matrix((history_data, history_indices, history_indptr), shape=shape),
        csr_matrix((target_data, target_indices, target_indptr), shape=shape),
    )
