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
- 추천 지표의 정답 평점 임계값(예: 4점 이상), K, 후보 집합 및 미관측 항목 처리 정책은 평가 담당자가 확정해야 합니다. 모델 학습과 지표 계산은 아직 구현하지 않았습니다.
- 영화 매칭은 제목 정규화+동일 연도의 일대일 자동 결과이며 수동 검수가 필요합니다. 평가는 전체 Netflix가 아닌 공통 영화 범위에 한정됩니다.

## 전처리 전체 재실행

원본 `data/movies.dat`, `data/ratings.dat`를 두고 아래를 실행합니다. Netflix 버전 2를 자동 다운로드하고 캐시를 재사용합니다.

```powershell
.\.venv\Scripts\python.exe preprocess.py
```

장르 인코딩, 연속 인덱스 변환, 희소행렬 생성 및 검증까지 수행합니다. 검증 성공 후 중간 CSV와 임시 DB를 자동 삭제합니다. 원본 파일과 Kaggle 캐시는 유지합니다. `--input-fraction`, `--min-ratings`, `--netflix-dir`로 설정할 수 있습니다. MovieLens만 검사하는 `--movielens-only` 실행은 완성된 모델 입력을 만들지 않습니다.

이미 생성된 중간 CSV만 숫자 입력으로 변환하려면 `python prepare_model_data.py`를 실행합니다. 최종 입력 파일만 남은 상태에서는 전체 재실행 명령을 사용합니다.
