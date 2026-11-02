# ML_8team 데이터 사용법

## 팀원에게 전달할 것

`data/processed/model_data.npz`, `load_data.py`, `requirements-preprocess.txt`, 이 README만 있으면 모델 구현을 시작할 수 있습니다. `movie_mapping.csv`와 `report.json`은 검토·재현 기록이므로 프로젝트에 보관합니다.

확인 및 전처리 재실행용 MovieLens 원본 data/movies.dat와 data/ratings.dat는 저장소에 포함되어 있습니다. 모델 구현에는 전처리 담당자에게 별도로 받은 model_data.npz를 data/processed/ 폴더에 배치한 뒤 사용합니다. NPZ 파일만 Git에서 제외하며, 영화 매핑표와 전처리 보고서를 포함한 나머지 데이터 파일은 저장소에 포함합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-preprocess.txt
```

```python
from load_data import load_data
d = load_data()
train = d['train']          # MovieLens 사용자 × 영화: 학습용 CSR 희소행렬
features = d['genre_features']  # 영화 × 장르: float32 멀티핫 행렬
history = d['input']        # Netflix 사용자 × 영화: 추천 입력
truth = d['target']         # Netflix 사용자 × 영화: 최종 평가 정답
```

## 반드시 지킬 기준

- 모든 행렬의 영화 열은 동일한 0 기반 연속 인덱스입니다. `movie_ids[j]`가 j열의 원본 MovieLens ID, `movie_titles[j]`가 제목입니다. `genre_names[k]`는 특징 k열의 장르입니다. 제목은 숫자 범주로 학습하지 않습니다.
- `ml_user_ids[i]`는 학습 행 i의 원본 ID입니다. `netflix_user_ids[i]`는 입력과 정답 행 i의 원본 ID입니다. 두 사용자 공간은 다르므로 합치지 않습니다.
- CSR의 저장된 값만 관측 평점입니다. 비어 있는 위치를 실제 0점으로 학습하지 않습니다. 큰 행렬을 `.toarray()`로 변환하지 않습니다.
- Content-based는 genre_features로 영화 유사도를 계산하고, input의 사용자 평점으로 취향을 구성합니다. CF는 train으로 영화 유사도를 학습하고, Netflix input으로 추천 점수를 계산합니다. Hybrid 가중치는 MovieLens 내부 검증으로 정하고 Netflix target으로 조정하지 않습니다.
- train에는 MovieLens 전체 평점이 있습니다. 하이퍼파라미터 탐색에는 담당자가 MovieLens 내부 학습/검증을 먼저 분리해야 합니다.
- 공통 영화 인덱스는 common_movie_indices입니다. candidate_indices는 이 중 MovieLens 학습 평점도 존재하는 영화로 Content/CF/Hybrid의 공통 후보로 사용합니다. 사용자별로 input에서 이미 본 영화는 추천 후보에서 제외합니다. target 중 후보 밖 항목은 평가 분모에서 제외하고 제외 건수도 보고합니다.
- input/target은 매칭 평점 5개 이상 사용자에 대해 날짜순 약 80/20으로 나눴으며 같은 날짜는 나누지 않습니다. 이후 날짜 정답이 없는 사용자는 제외합니다. 이 기본값은 프로포잘 확인 전 임시 기준입니다.
- train_times는 Unix 초, input_dates/target_dates는 YYYYMMDD 정수입니다. 각 시간 배열은 해당 CSR의 data/indices와 위치가 대응합니다. CSR을 정렬하거나 변형하면 이 대응을 유지해야 합니다.
- 추천 지표의 정답 평점 임계값(예: 4점 이상), K, 후보 집합 및 미관측 항목 처리 정책은 평가 담당자가 확정해야 합니다. Content-based 구현 설명은 아래에 있으며, CF/Hybrid와 지표 계산은 아직 구현하지 않았습니다.
- 영화 매칭은 제목 정규화+동일 연도의 일대일 자동 결과이며 수동 검수가 필요합니다. 평가는 전체 Netflix가 아닌 공통 영화 범위에 한정됩니다.

## 전처리 전체 재실행

원본 `data/movies.dat`, `data/ratings.dat`를 두고 아래를 실행합니다. Netflix 버전 2를 자동 다운로드하고 캐시를 재사용합니다.

```powershell
.\.venv\Scripts\python.exe preprocess.py
```

장르 인코딩, 연속 인덱스 변환, 희소행렬 생성 및 검증까지 수행합니다. 검증 성공 후 중간 CSV와 임시 DB를 자동 삭제합니다. 원본 파일과 Kaggle 캐시는 유지합니다. `--input-fraction`, `--min-ratings`, `--netflix-dir`로 설정할 수 있습니다. MovieLens만 검사하는 `--movielens-only` 실행은 완성된 모델 입력을 만들지 않습니다.

이미 생성된 중간 CSV만 숫자 입력으로 변환하려면 `python prepare_model_data.py`를 실행합니다. 최종 입력 파일만 남은 상태에서는 전체 재실행 명령을 사용합니다.

## Content-based Filtering (유병훈)

`content_based.py`는 영화 장르와 사용자 평점 이력으로 Top-K 추천을 생성하는 모듈입니다.
`load_data.py`로 읽은 `genre_features`와 `input`을 사용합니다. 정규화한 영화
벡터 중 사용자가 4점 이상 평가한 영화만 골라 평점으로 가중 평균한 뒤,
취향 벡터와 각 영화의 코사인 유사도를 계산합니다. 선호 기준은 기본 4점이며
모델 생성 시 `min_rating`으로 변경할 수 있습니다.

- `model = ContentBasedRecommender(d['genre_features'])`로 모델을 생성합니다.
- `model.score(d['input'], user_index)`는 전체 영화 열에 대응하는 0~1 점수 벡터를
  반환합니다. Hybrid에서 CF 점수와 결합할 수 있으며, 결합 후 공통 후보 제한과
  이미 본 영화 제외를 적용해야 합니다. 점수는 선호 확률이나 1~5점 예측 평점이
  아니므로 RMSE/MAE에 그대로 사용하지 않습니다.
- `model.recommend(d['input'], user_index, candidate_indices=d['candidate_indices'], k=10)`는
  공통 후보에서 이미 본 영화를 제외한 `(영화 열 인덱스, 점수)` 목록을 반환합니다.
  낮게 평가한 영화도 제외합니다. 동점은 영화 열 인덱스 순이며, 후보가 부족하면
  K보다 적게 반환합니다. 같은 장르 조합은 같은 점수를 받을 수 있습니다.

`d`는 `load_data()`의 반환값입니다. `user_index`는 원본 Netflix ID가 아닌
`input`의 행 번호이며, 반환된 영화 인덱스 `j`는 `d['movie_ids'][j]`로 원본 ID에
대응합니다. 선호 영화가 없거나 취향 벡터가 0이면 점수는 0 벡터, 추천은 빈 목록을
반환합니다. 취향 벡터가 있으면 유사도가 0인 후보도 낮은 순위에 포함될 수 있습니다.
영화 장르 표현은 고정하며, CF 정보와 Netflix `target`은 사용하지 않습니다.
설정을 비교할 경우 MovieLens 내부 검증으로 결정합니다. 입력 CSR의 값이나 순서를
바꾸지 않아 `input_dates`와의 대응도 유지됩니다.
실제 전처리 NPZ로 Netflix 사용자 6·7·10의 Top-5 출력을 확인했습니다.
