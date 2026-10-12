# Homework 2: Titanic

제출용 노트북은 프로젝트 루트의 `hw2.ipynb`이다.
`train_and_test_titanic.ipynb`는 초기 작업을 보존한 파일이다.

## 현재 제출용 구현

수정 기준은 `hw2.ipynb`의 코드 셀이다. EarlyStopping도 원본 경로와 변경 이유를 명시해 직접 구현하며,
전처리, 모델, 학습, W&B 실행, 실험 반복, 집계, 예측은 노트북에 직접 정의돼 있다.
외부 Python 파일의 소스를 표시하거나 해당 파일을 실행하는 방식은 사용하지 않는다.
핵심 코드와 실행 보조 코드에는 목적과 결과를 설명하는 주석이 있다.

프로젝트 루트에서 노트북을 실행한다. 원본 CSV와 results의 JSON/CSV/checkpoint는
여전히 데이터 파일로 필요하다. 기본값은 완료한 실험을 불러오는 모드이며,
`RUN_BASELINE`, `RUN_GRID`, `RUN_FINAL`을 켜면 노트북 코드로 새로 훈련한다.
재실행 결과는 새 폴더에 저장되고 새 grid를 수행하면 final도 새로 훈련한다.

노트북은 `submission_reproduced.csv`를 생성해 기존 제출 결과와 비교한다.
이미 제출한 `submission.csv`는 덮어쓰지 않는다. Kaggle 캡처는 attachment로 내장돼 있다.
`build_hw2_notebook.py`는 구버전 보관본이며 덮어쓰기를 막기 위해 CLI 실행을 중단한다.
아래 스크립트 명령은 이전 실험 방식의 이력/참고용이다.

## 재현

프로젝트 루트에서 기존 `.venv` 환경으로 실행한다.

```bash
.venv/bin/python -m unittest discover -s _03_homeworks/homework_2 -p 'test_*.py' -v
bash _03_homeworks/homework_2/run_titanic_grid.sh --wandb \
  --group hw2-repeat --output_dir _03_homeworks/homework_2/results/repeat
.venv/bin/python _03_homeworks/homework_2/summarize_titanic_runs.py \
  _03_homeworks/homework_2/results/repeat --group hw2-repeat
```

반복 실험은 새 그룹 및 새 출력 폴더를 사용한다. 기본 grid는 36개 조합이며
SGD, 은닉층 20/20, seed=42, 최대 10,000 epoch, patience=100, delta=1e-5를 고정한다.
`TITANIC_PYTHON`으로 grid에서 사용할 Python 경로를 지정할 수 있다.

실험별 JSON에는 저장된 모델의 성능, epoch, 체크포인트, 학습 이력 CSV,
W&B 링크가 함께 기록된다. `latest` 체크포인트를 모델 선택에 사용하지 않는다.

```bash
.venv/bin/python _03_homeworks/homework_2/predict_titanic.py PATH_TO_FINAL_RUN_JSON
```

위 명령은 이 폴더의 `submission.csv`를 생성한다. 현재 노트북은 별도 재현 파일을 생성한다.
실제 실험 결과는 `results/baseline`, `results/grid`, `results/final`에 저장된다.
체크포인트 `.pt`는 저장소의 기존 ignore 규칙에 따라 Git에서 제외되므로
다른 컴퓨터에서 재현하려면 체크포인트를 함께 복사하거나 다시 훈련해야 한다.

## 사용자 마무리

1. Kaggle 제출 완료: 제공된 캡처 기준 점수 0.74880, 순위 9,984위.
2. 내장된 점수/순위 캡처의 클라우드 URL을 `hw2.ipynb`에 추가한다.
3. 마지막 `숙제 후기` 안내 문구를 실제 소감으로 교체한다.
4. 노트북 클라우드 업로드 및 노트북/W&B PDF 생성 후 LMS에 제출한다.

PDF 생성, Kaggle 제출, LMS 제출은 자동화하지 않는다.
