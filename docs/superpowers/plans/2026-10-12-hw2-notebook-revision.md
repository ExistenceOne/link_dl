# HW2 Notebook Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** EarlyStopping을 제외한 과제 구현을 hw2.ipynb의 실행 가능한 코드 셀에 직접 작성하고, 베이스 코드의 활용 근거, 핵심 및 실행 보조 코드 주석, 실제 Kaggle 결과를 명확히 제시한다.

**Architecture:** 기존 학습 알고리즘과 실험 결과는 유지하면서 제출 노트북을 구현의 기준으로 전환한다. 데이터 준비, 모델, 학습, W&B, 실험 반복, 결과 비교, 예측을 노트북에서 직접 정의한다. 외부 Python 코드 의존은 EarlyStopping 하나만 허용하며, 원본 CSV/실험 JSON/이력 CSV/체크포인트는 데이터 파일로 읽는다.

**Tech Stack:** 기존 Python/PyTorch/pandas/scikit-learn/W&B/Jupyter 환경. 새 프레임워크 추가 없음.

**Spec:** 2026-10-12 사용자 수정 요청 4개 항목 및 HW_2.pdf의 요구사항 1~4, 핵심 주석과 실행 출력 요구사항. 사용자의 시작 지시에 따라 구현과 검증을 완료했다.

## Global Constraints

- 사용자 정의 외부 코드 import는 `_01_code._99_common_utils.early_stopping.EarlyStopping`만 남긴다. 표준 라이브러리와 설치된 패키지 import는 허용한다.
- 실행 보조 코드도 실제 코드 셀 안에서 `[실행 보조]` 주석으로 역할을 명시한다. 함수/블록별 목적, 필요한 입력 또는 의존 파일, 생성하는 결과나 부수효과를 설명한다. Markdown 설명만으로 대체하지 않는다.
- `_03_homeworks.homework_2`의 모듈 import, 외부 .py 실행, `%run`, subprocess로 학습 스크립트 호출, importlib/exec로 구현 숨기기, inspect.getsource 출력으로 구현 대체를 사용하지 않는다.
- 원본 CSV와 이미 완료한 실험 산출물을 읽는 것은 허용한다. "코드가 노트북에 있다"와 "데이터 파일도 필요 없다"를 혼동하지 않는다.
- 이미 제출한 CSV, 기존 checkpoint, W&B 기록 및 사용자 이미지 원본을 보존한다. 같은 예측을 재현하는지 확인한 후 결과물을 갱신한다.
- 10 -> 20 -> 20 -> 1, SGD, BCEWithLogitsLoss, seed=42, 기존 분할/셔플/배치 처리/최적 epoch 정책을 유지한다. 성능 개선을 위한 재설계는 이번 범위가 아니다.
- 모델 재현 결과가 같으면 36개 전체 실험을 반복하지 않는다. 알고리즘 차이가 발견되면 먼저 차이를 해소하고, 불가피한 변경이 있다면 기존 결과와 구분해 기록한다.
- Kaggle 재제출, 이미지의 외부 업로드, PDF 생성 및 LMS 제출은 수행하지 않는다. 실제 숙제 후기는 사용자가 작성한다.

## Audit Findings

### A. 노트북의 외부 코드 의존

현재 hw2.ipynb의 두 번째 셀(1-based)이 다음 외부 정의를 가져온다.

- titanic_dataset: get_preprocessed_dataset.
- train_titanic_with_argparse_wandb: MyModel, get_data, training_loop, binary_accuracy_f1, build_parser, main.
- predict_titanic: predict_submission.
- summarize_titanic_runs: summarize_runs.
- early_stopping: EarlyStopping. 이 항목만 사용자 허용 예외다.

모델/get_data/training_loop/예측 일부는 inspect.getsource로 출력하지만 노트북에서 정의되는 것은 아니다. 데이터 전처리, optimizer 구성, W&B 초기화와 metric 설정, 실행 진입 함수, 36개 조합 반복, 집계 구현은 제출 노트북에서 전체를 볼 수 없다. 따라서 현재 구성은 사용자 요구에 미달한다.

### B. 베이스 코드 활용 및 변경 규모

기준 파일: `_01_code/_08_learning_and_optimization/c_my_model_training_with_argparse_wandb.py`.
이 파일은 현재 upstream/main과 diff가 없으며 160행이다. 현재 Titanic 학습 스크립트는 306행이다. 행 수 증가만으로 알고리즘 변경 정도를 판단하지 않는다.

| 베이스 위치 | 현재 Titanic 학습 코드 위치 | 재사용/변경 내용 | 판정 |
|---|---|---|---|
| get_data, 19~29행 | get_data, 22~32행 | 두 DataLoader와 batch_size/shuffle 구조 유지. California 데이터 대신 Titanic, 고정 RNG 사용 | 과제 필수 변경 + 재현성 보완 |
| MyModel, 32~46행 | MyModel, 35~55행 | nn.Sequential, Linear 3개, 은닉층 20/20, forward 유지. ReLU만 네 함수 중 선택 | 모델 구조 재사용, 필수 확장 |
| get_model_and_optimizer, 49~53행 | 58~62행 | 입력 8 -> 10, 출력 1 유지. optim.SGD와 설정 기반 learning_rate 유지 | Titanic에 필요한 변경 |
| training_loop, 56~73행 | 78~118행 | epoch/배치 반복, 순전파, zero_grad -> backward -> step 순서 유지. 타깃을 dict에서 꺼내 float (batch,1)로 변환. MSE -> BCEWithLogitsLoss | 회귀를 이진 분류로 바꾸는 필수 변경 |
| 검증, 75~83행 | 119~137행 | no_grad와 검증 loss 계산 유지. eval 및 표본 수 가중 평균 추가 | 정확성 보완 |
| run.log, 85~89행 | 153~163행 | Epoch/Training loss/Validation loss 유지. accuracy/F1과 로컬 history 추가 | 필수 로그 유지 + 보조 지표 |
| main/config, 100~108행 | 180~199행 | config와 함수 호출 흐름 유지. activation/seed/Early Stopping 설정 추가 | 과제와 재현성에 필요한 확장 |
| wandb.init/define_metric, 110~125행 | 201~220행 | 온라인/비활성 모드, project/config, Epoch 기반 두 loss 지표 그대로 활용 | 직접 재사용 |
| 실행 연결, 127~139행 | 222~234행 | get_data -> get_model_and_optimizer -> training_loop 순서 유지 | 직접 재사용 |
| CLI, 143~160행 | build_parser, 264~306행 | 명령행 옵션 확장 | 스크립트에는 유용하지만 노트북에서는 설정 셀로 단순화 가능 |
| 베이스에 없음 | best checkpoint/metrics, 139~151행 | Early Stopping 및 저장 모델과 보고 지표 연결 | 요구사항 3에 필요한 추가 |
| 베이스에 없음 | JSON/CSV/경로 처리, 235~259행 | 반복 실험 산출물 관리 | 보조 기능. 핵심 학습 흐름을 가리지 않게 분리/축소 |

판정: MLP와 학습 알고리즘 자체를 과도하게 바꾼 상태는 아니다. BCE 전환, 활성화 함수 선택, Early Stopping, 표본 수 가중 loss는 유지해야 한다. 다만 제출용으로 외부 스크립트/CLI/빌더/메타데이터 경유 구조를 사용한 것은 과도하며, 베이스 활용 관계와 학생이 설명할 핵심 코드가 잘 보이지 않는다. 이를 노트북 중심으로 바로잡는다.

### C. 핵심 주석

- tokenize로 검사한 현재 노트북 실행 코드 셀 11개의 `#` 주석은 총 0개다.
- 별도 Markdown 설명은 존재한다. 일부 getsource 출력에는 외부 파일 주석이 보이지만, 핵심 구현을 직접 설명하는 코드 주석을 대신하기에는 부족하다.
- 학습 스크립트 주석은 6개이며, 오래된 사용자 경로 및 URL 주석도 포함된다. 실제 동작 설명은 loss 평균, W&B 축, 경로 보존 정도다.
- predict_titanic.py와 summarize_titanic_runs.py는 모듈 docstring만 있고 내부 `#` 주석이 없다.
- 전처리에는 단계 주석이 있으나 titanic_dataset.py:116의 "Age 평균값"은 실제 `.median().round()`와 불일치한다. 노트북으로 옮길 때 "호칭별 나이 중앙값을 반올림"으로 바로잡는다.

### D. Kaggle 이미지

`hw2_kaggle.png`를 직접 확인했다. 보이는 값은 사용자 ExistenceOne, 점수 0.74880, 순위 9984다. 순위는 캡처 시점 값이다. 정확한 캡처 날짜는 이미지에 없으므로 추정하지 않는다. 이미지의 상대 시간 14h를 날짜로 변환하지 않는다.

## Review Focus

- 외부 .py가 없어도 EarlyStopping과 데이터 파일만으로 노트북의 모든 정의/실행이 동작해야 한다: Task 1,5.
- 구현 이전 과정에서 RNG 소비 순서나 모델 초기화가 달라져 제출된 예측이 바뀌지 않아야 한다: Task 2,5.
- 기존 결과를 불러오는 셀과 새로 훈련하는 셀을 명확히 구분하고, 불러온 결과를 이번 노트북에서 새로 훈련했다고 표현하지 않아야 한다: Task 2.
- 코드 주석이 실제 연산, 반환값, 결과의 의미와 일치해야 한다: Task 3.
- 이미지가 노트북 이동 후에도 표시되고, 캡처 점수/순위를 검증 지표나 현재 순위로 오인하지 않아야 한다: Task 4,5.

## Task 1: 노트북 안으로 구현 이전

**Files:** Modify `hw2.ipynb`. 기존 외부 Python 파일은 이력과 회귀 비교용으로 보존한다.

**Interfaces:** 노트북에서 TitanicDataset, TitanicTestDataset, get_preprocessed_dataset 및 전처리 6단계, get_data, MyModel, get_model_and_optimizer, binary_accuracy_f1, training_loop, main, predict_submission을 직접 정의한다. 기존 인자/반환값과 가중치 키 이름을 가능한 한 유지한다.

- [x] 수정 전 노트북/제출 CSV/선택된 checkpoint의 해시와 검증 지표를 기록한다.
- [x] 코드 셀을 import/설정 -> 데이터 클래스 -> 전처리 -> DataLoader -> 모델/optimizer -> 지표 -> 훈련 -> W&B 실행 함수 -> 실험 -> 결과 -> 추론 순으로 나눈다.
- [x] 전처리도 함수 이름만 가져오지 않고 구현 전체를 노트북 셀에 넣는다. `__file__` 기반 데이터 탐색을 노트북 설정의 DATA_DIR로 바꾼다.
- [x] getsource 출력 셀을 실제 정의 셀로 교체한다. EarlyStopping import 외의 로컬 Python import와 sys.path 우회는 없앤다. 실행 위치는 프로젝트 루트로 명시하고 데이터 경로를 설정 셀에 드러낸다.
- [x] argparse/build_parser 대신 설정 셀의 명시적인 값과 SimpleNamespace로 main(args)를 호출한다. 베이스의 함수 이름과 호출 흐름을 유지한다.
- [x] 기존 builder가 다시 import 기반 노트북을 덮어쓰지 않도록, README에서 hw2.ipynb를 수정 기준으로 명시하고 builder는 구버전 생성기임을 표시한다. 새로 수정한 노트북을 그 빌더로 재생성하지 않는다.

## Task 2: 실험과 베이스 코드의 관계를 명확히 표시

**Files:** Modify `hw2.ipynb`, `_03_homeworks/homework_2/README.md`.

**Interfaces:** 36개 조합은 노트북 안의 itertools.product와 main(args)로 실행 가능해야 한다. 비교표는 노트북에 정의된 집계 코드 또는 명시적인 pandas 처리로 구성한다.

- [x] "베이스 코드 활용 및 변경 사항" 표를 위 감사 결과에 따라 노트북에 넣는다. 출처 파일, 원본 행 구간, 현재 셀 제목, 유지/변경 이유를 기록한다.
- [x] 핵심 셀 시작 부분에 출처와 변경 목적을 짧게 주석으로 남긴다. 원본의 MLP, SGD, 역전파 순서, W&B 3개 기본 지표를 직접 읽을 수 있게 한다.
- [x] grid shell 호출 대신 4 x 3 x 3 반복 코드를 노트북에 넣는다. RUN_GRID=False 기본값으로 이미 완료한 36개 결과를 사용하며, True이면 같은 설정으로 새 그룹/폴더에 저장한다.
- [x] JSON/CSV 산출물 저장은 별도 짧은 함수/셀로 분리한다. 경로/CLI 처리로 훈련 루프가 복잡해지지 않게 한다. 기존 기록과 호환되는 최소 필드는 유지한다.
- [x] baseline과 final 재훈련 경로를 노트북 함수로 제공한다. 재훈련 결과를 선택했다면 표/그래프/추론도 그 결과를 사용한다.
- [x] 기존 36개 W&B/메타데이터/체크포인트의 연결 및 loss 기준 선택을 유지한다. 최고 accuracy 설정과 최저 loss 설정이 다르다는 설명을 유지한다.

## Task 3: 핵심 및 실행 보조 코드 주석 보강

**Files:** Modify `hw2.ipynb`의 실제 코드 셀.

- [x] 각 핵심 함수에 목적, 주요 입력, 반환 결과를 짧은 docstring으로 기재한다.
- [x] Dataset: X의 float 변환, 타깃 dtype, 배치 dict 구조와 기대 shape 설명.
- [x] 전처리: 요금 평균/나이 중앙값, 호칭 추출, 가족/alone 생성, 범주 인코딩, 제거 컬럼 및 최종 특성 순서 설명.
- [x] 분할/셔플: generator를 나눈 이유와 seed 고정 결과 설명.
- [x] 모델/loss: 입력 10개, 은닉층 20/20, 출력 logit 1개, float (batch,1) 타깃, sigmoid 중복 적용을 피하는 이유 설명.
- [x] 훈련: zero_grad/backward/step 역할, detach/no_grad/eval 차이, 표본 수 가중 loss 목적 설명.
- [x] W&B: config 기록 목적, Epoch x축, loss 기록 및 best epoch 지표 일치의 의미 설명.
- [x] EarlyStopping 사용부: 최초 저장, delta/patience, 종료 epoch와 복원 epoch 차이 설명. 외부 클래스 본문은 중복 작성하지 않는다.
- [x] 추론/CSV: 명시된 체크포인트 복원, shuffle=False, logit >= 0 기준, PassengerId 순서, index=False 이유 및 결과 418행 설명.
- [x] 주석 개수만 채우지 말고 코드와 내용이 일치하는지 검토한다. Markdown 고찰과 코드 내부 주석을 모두 유지한다.
- [x] 경로/환경 설정에 `[실행 보조]` 주석을 붙인다: 실행 기준 폴더, DATA_DIR/RESULTS 경로, 필요한 외부 데이터 파일, 출력 폴더 생성 목적과 위치를 설명한다.
- [x] 설정/실행 제어에 `[실행 보조]` 주석을 붙인다: SimpleNamespace, RUN_GRID 및 baseline/final 재훈련 스위치, 기본값에서 실제로 실행하거나 재사용하는 범위를 설명한다.
- [x] 실험 반복/이름 생성에 `[실행 보조]` 주석을 붙인다: 36개 조합의 생성, 실행 ID와 W&B group, 결과 충돌을 방지하는 이름, 새 run 생성 및 업로드의 부수효과를 설명한다.
- [x] JSON/CSV 저장과 불러오기에 `[실행 보조]` 주석을 붙인다: 기록 필드의 의미, 체크포인트/이력과의 연결, 상대경로 기준, 어떤 파일을 생성하거나 갱신하는지 설명한다.
- [x] 집계/정렬/검증에 `[실행 보조]` 주석을 붙인다: 실험 누락/중복 검사 목적, 선택 정렬 기준, shape/ID/지표 assert가 보장하는 조건과 실패 의미를 설명한다.
- [x] 그래프/표/이미지 출력에 `[실행 보조]` 주석을 붙인다: 사용하는 결과, 고정하는 변수, 축과 지표, 저장 epoch 표시, 실제 출력되는 표/그림을 설명한다.
- [x] 셀 앞부분에 분류 이름만 넣고 끝내지 않는다. 보조 함수의 docstring과 관련 블록 주석을 함께 사용하되, 단순 대입을 한 줄마다 반복 설명하지 않는다.

예시 주석의 수준:

```python
# [실행 보조: 결과 불러오기]
# 이전에 완료한 36개 실험의 JSON을 읽어 비교표를 만든다.
# 이 경로에서는 재훈련하거나 새 W&B run을 생성하지 않는다.

# [실행 보조: 출력 검증]
# 원본 test.csv와 PassengerId 순서가 같은지 확인한다.
# 다르면 승객과 예측이 잘못 연결될 수 있으므로 CSV 저장 전에 중단한다.
```

## Task 4: 실제 Kaggle 결과 반영

**Files:** Modify `hw2.ipynb`. Read-only source `hw2_kaggle.png`.

- [x] "제출 후 입력" 안내를 실제 결과로 교체한다: 점수 0.74880, 정확도 환산 74.880%, 캡처 시점 순위 9,984, 사용자 ExistenceOne.
- [x] 이미지를 노트북 Markdown attachment로 포함해 노트북 파일을 옮겨도 이미지가 보이게 한다. 원본 PNG도 그대로 둔다.
- [x] 검증 정확도 85.39%와 Kaggle 점수 74.880%를 구분한다. 데이터 분할과 평가 대상이 다르며, 단일 검증 분할에서 설정을 선택한 한계를 설명한다. 성능 차이의 원인을 확인한 것처럼 단정하지 않는다.
- [x] 과제 원문은 이미지 클라우드 URL도 요구하므로 attachment는 표시 안정성 보조라고 명시한다. 현재 전달된 자료에는 URL이 없으며, 사용자가 제공하는 클라우드 URL을 적을 자리를 남긴다. 사용자 승인 없이 공개 업로드하지 않는다.
- [x] 상대 시간 14h를 정확한 제출일/캡처일로 바꾸거나 순위를 현재 순위라고 쓰지 않는다.

## Task 5: 수정 후 검증과 인계

**Files:** `hw2.ipynb` 및 임시 검증 폴더. 기존 실험 산출물/제출 CSV를 비교 기준으로 사용한다.

- [x] 노트북 AST를 검사해 허용된 EarlyStopping 외에 사용자 정의 로컬 모듈 import가 없음을 확인한다. 실제 정의가 코드 셀에 있으며, inspect/getsource/외부 Python 실행으로 대체되지 않았는지 확인한다.
- [x] 원본 CSV, 기존 결과 파일, 필요한 checkpoint, EarlyStopping만 복사한 임시 디렉터리에서 노트북을 실행한다. homework_2의 Python 코드 파일은 복사하지 않는다. 외부 의존이 남은 이전 노트북에서는 실패하고 수정본에서는 성공해야 한다.
- [x] 실제 노트북 정의로 소규모 훈련을 실행해 첫 epoch checkpoint와 표본 수 가중 loss를 검증한다. notebook 정의가 아닌 외부 모듈만 테스트하고 완료 처리하지 않는다.
- [x] 데이터 shape/분할 indices/모델 구조/저장된 가중치 키와 복원 지표가 기존과 일치하는지 확인한다.
- [x] 선택 설정을 W&B 비활성 상태의 임시 경로에서 한 번 재훈련해 best_epoch=626, stopped_epoch=726 및 기존 가중치/예측과의 일치를 확인한다. 불일치 시 RNG와 실행 순서를 조사하고 제출 파일을 덮어쓰지 않는다.
- [x] 임시 CSV와 이미 제출한 CSV의 PassengerId 및 Survived 전체가 동일한지 검사한다. 418행/2열/이진 정수/결측값 없음도 확인한다.
- [x] 전체 셀을 순서대로 실행해 출력과 그래프를 저장한다. Kaggle attachment가 렌더링되고, 핵심 코드와 주석이 노트북에서 바로 보이는지 확인한다.
- [x] 코드 셀별로 핵심/실행 보조 블록을 목록화해 주석 누락을 검토한다. 경로·설정·반복·저장/불러오기·집계·검증·시각화 모두 분류와 목적/결과 설명이 있어야 통과한다. 주석에 적힌 파일 갱신/업로드 여부도 실제 코드와 대조한다.
- [x] 수정된 노트북과 검증 결과를 전달한다. Kaggle 재제출이나 PDF 생성은 하지 않는다. 사용자에게 남는 부분은 실제 숙제 후기와 이미지 클라우드 URL, 최종 PDF/LMS 제출이다.

## Plan Self-Review

- 요구 1(노트북 내 구현, EarlyStopping 예외): Task 1,2,5.
- 요구 2(베이스 코드 활용 위치/과도한 수정 검증): 감사 표 및 Task 2.
- 요구 3 및 추가 지시(핵심과 실행 보조 코드 모두 주석 명시): 감사 C, Global Constraints 및 Task 3,5.
- 요구 4(Kaggle 캡처): 감사 D 및 Task 4,5.
- 실행 완료: 노트북에 구현과 주석을 이전하고 README 및 구버전 builder 안내를 갱신했다. 원본 이미지/제출 CSV/선택 checkpoint는 해시가 동일하다.
- 검증: 격리 실행, 실제 최종 설정 재훈련(가중치 및 418개 예측 동일), 13개 코드 셀 전체 실행, grid 셀 연속 재실행(각 36개 독립 그룹), 렌더링 이미지 확인을 수행했다.
