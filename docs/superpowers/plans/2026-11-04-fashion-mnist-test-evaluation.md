# Fashion MNIST 테스트 코드 및 평가 계획

**목표:** 검증 데이터로 이미 선택한 모델을 복원해 테스트 데이터 10,000개를 평가하고, HW 문제 3·4에 필요한 지표와 사례를 만든다.

**진행 상태:** 사용자 승인에 따라 구현과 테스트 평가 및 보고서 갱신을 완료했다. 재학습이나 테스트 기반 모델 재선택은 수행하지 않았다.

## 평가 대상 고정

- `experiments/final_selection.json`의 winner 중 `seed42_configuration`을 기본 평가 대상으로 삼는다.
- 직접 구현 VGG + feature BN, Dropout 없음, 입력 32x32 resize, 학습 시 augmentation 사용 설정이다.
- 학습 시드 42의 저장 체크포인트를 사용한다. 저장 모델의 검증 정확도는 94.06%이며, 학습 중 최고 정확도 94.36%와 구분한다.
- mean=0.2855656942074541, std=0.3527285864947437을 그대로 사용한다.
- 테스트 성능으로 모델·시드·체크포인트를 다시 고르지 않는다. 5개 모델의 테스트 순위 비교는 이번 범위에 포함하지 않는다.
- 현재 VGG의 3시드 평균 검증 정확도는 93.77%이다. 평균 기준 94%를 안정적으로 넘었다고 주장하지 않는다.

## 파일과 구조

- Modify: `_03_homeworks/homework_3/b_fashion_mnist_train.py`.
  - 현재 main의 모델 생성/계층 조정 부분만 `get_fashion_mnist_model(args)`로 추출한다.
  - 모델 선택 분기, 계층 위치/이름, 입출력 크기와 학습 동작은 유지한다. 학습 main은 이 함수를 호출한다.
  - 별도 모델 파일이나 새 모델 구현을 만들지 않는다. 기존 체크포인트의 state_dict key를 바꾸지 않는다.
- Create: `_03_homeworks/homework_3/c_fashion_mnist_test.py`.
  - 기존 테스트 파일처럼 `main()`에서 데이터 준비 -> 모델 복원 -> 평가 -> 샘플 분석 흐름을 보여준다.
  - 학습 파일의 공통 모델 생성 함수를 import한다. 저장된 result.json의 args로 구조를 재구성한다.
  - 기존 ClassificationTester는 정확도만 출력하고 예측/확률을 반환하지 않으므로, 공통 Tester 파일을 수정하는 대신 이 파일에 배치 추론 함수를 둔다.
  - 주요 인터페이스: `get_parser()`, `predict(model, test_data_loader, transforms, device, batch_size)`, `main(args)`.
  - CLI: `--result`(기본값: final_selection의 대표 실행), `--batch_size`(추론용, 기본 256), `--sample_seed`(기본 42), `--output_dir`, `--wandb/--no-wandb`.
  - 모델과 학습 설정은 result.json에서 읽으며 명령행으로 다시 손입력하지 않는다. 추론 batch size는 학습 batch size와 구분한다.
- Modify/Create: `tests/test_fashion_mnist_ablation.py`, `tests/test_fashion_mnist_evaluation.py`.
- Update: 기존 모델 선택·튜닝 보고서에 테스트 결과를 별도 섹션으로 추가한다. 평가 지표/예측 결과는 `evaluation/`에 저장한다.

## 1. 복원과 전처리 검증

- [x] 기존 학습 코드의 모델 생성 부분을 공통 함수로 추출하고, 기존 20개 모델/BN/Dropout 조합 회귀 검증을 유지한다.
- [x] result.json이 가리키는 체크포인트의 존재를 확인한다. 모델을 재구성한 뒤 `torch.load(..., weights_only=True)`와 `load_state_dict(..., strict=True)`로 복원한다.
- [x] BN의 weight/bias/running_mean/running_var도 복원되었는지 확인한다. VGG의 Lazy 계층을 포함해 누락/추가 key를 허용하지 않는다.
- [x] `model.eval()` 및 `torch.inference_mode()`를 적용한다. 실제 검증 배치의 출력이 `(N, 10)`이고 finite인지 확인했다 (별도 dummy 대신 실제 배치로 확인).
- [x] 테스트 데이터는 반드시 제공된 `get_fashion_mnist_test_data()`로 가져온다.
- [x] 반환된 transform의 placeholder 정규화를 학습 mean/std로 바꾸고, VGG의 32x32 resize를 추가한다. augmentation은 넣지 않는다.
- [x] 학습 검증 때와 같이 입력을 device에 옮긴 뒤 resize/정규화를 적용한다.
- [x] 테스트 데이터에 접근하기 전에 고정된 검증 분할에서 대표 체크포인트의 94.06%가 재현되는지 확인한다. 불일치 시 모델 구조/BN 통계/전처리를 먼저 조사한다.

## 2. 전체 테스트 추론 및 문제 3 평가

- [x] 제공 테스트 로더는 10,000개 전체를 한 배치로 반환한다. GPU에 전부 전달하지 않고 CPU 배치를 추론용 크기로 나눠 순차 처리한다. 제공 Dataset/DataLoader를 재생성하지 않는다.
- [x] 10,000개 각각의 index, 정답, 예측, 10클래스 softmax 확률을 한 번의 전체 평가에서 수집한다. 이후 지표와 사례 분석은 이 결과를 재사용한다.
- [x] 전체 정확도와 정답/오답 개수를 출력한다. 93% 기준 달성 여부를 실제 결과로 판단한다.
- [x] 혼동행렬 원시 개수와 정답 클래스별 비율을 시각화한다. 축을 정답/예측으로 명시한다.
- [x] 대각선을 제외한 단방향 오분류와 양방향 합이 가장 큰 클래스쌍을 각각 제시한다. 알려진 혼동쌍이라고 미리 결과를 단정하지 않는다.
- [x] 클래스별 precision, recall, F1, support와 macro/weighted 평균을 계산한다. 분모가 0인 경우 처리를 명시한다.
- [x] precision/recall이 낮은 클래스의 실제 오분류 이미지를 확인해 원인을 추정하고, 관찰 사실과 추정을 구분한다.

## 3. 문제 4 샘플 분석

- [x] 예측 결과와 무관하게 sample_seed=42로 테스트 index 10개를 중복 없이 추출한다. 정분류/오분류 개수를 맞추기 위해 다시 뽑지 않는다.
- [x] 각 샘플의 원본 이미지, 정답/예측 클래스, 정답 여부, 최대 softmax 확률을 출력한다.
- [x] 확신도 0.9를 해석용 기준으로 사전에 고정해 높은/낮은 확신의 정분류/오분류를 설명한다. 이는 편의적 구분이며 softmax 값의 확률적 보정을 보장하지 않는다.
- [x] 애매함은 상위 두 클래스의 확률 차이 `margin = p_top1 - p_top2`로 정의한다. margin이 작을수록 두 후보를 구분하기 어렵다는 의미로 해석한다.
- [x] 10,000개 전체에서 margin이 최소인 샘플 1개를 제시한다. 동률은 index가 작은 것을 선택한다. 상위 두 클래스/확률, 정답, margin과 선택 이유를 출력한다.
- [x] 최소 margin 샘플이 반드시 오분류일 필요는 없으며, 그 결과를 그대로 보고한다.

## 4. 검증과 산출물

- [x] 인공 logits/labels로 정확도, 혼동행렬 합계, precision/recall, 확신도와 margin 선정이 맞는지 검증한다. 누락 클래스와 마지막 작은 배치를 포함한다.
- [x] 전체 추론 결과의 길이가 10,000인지, index 중복/누락이 없는지, 확률 합이 약 1인지 확인한다.
- [x] 혼동행렬의 총합=10,000, trace=정답 개수인지 확인한다.
- [x] BN 통계가 추론 전후 바뀌지 않는지 확인한다. 동일 입력의 반복 추론과 일괄/분할 추론은 부동소수점 허용오차를 고려해 확인한다.
- [x] 지표 JSON, 클래스별 CSV, 전체 샘플 예측/확률 CSV, 혼동행렬 PNG, 무작위 10개 이미지 PNG, 최소 margin 이미지 PNG를 저장한다.
- [x] 평가 설정, 체크포인트 경로/해시, 사용 데이터 수와 sample_seed를 기록한다.
- [x] `--wandb` 사용 시 별도 평가 run을 `run`으로 명명하고, 체크포인트 출처와 지표/이미지를 기록한다. 기존 학습 run의 결과를 덮어쓰지 않는다.
- [x] PDF 및 출력 포함 노트북에 실제 결과와 해석을 추가하고 렌더링을 검토한다. 최종 ablation 미실시 상태는 계속 명시한다.

## 실행 순서

공통 모델 생성 함수 추출 -> 테스트 코드 구현 -> 작은 데이터/인공 logits 검증 -> 검증 정확도 재현 -> 전체 테스트 10,000개 평가 -> 문제 3·4 분석 -> 보고서 갱신.

테스트 성능이 목표에 미달해도 숫자를 그대로 보고한다. 그 결과를 보고 같은 테스트 세트에 맞춰 추가 튜닝하는 절차는 이 계획에 포함하지 않는다.

## 실제 평가 결과

- 검증 복원 정확도: 94.06%.
- 테스트: 9,329 / 10,000 정답, 93.29%, 93% 기준 충족.
- 가장 큰 혼동쌍: T-shirt/top ↔ Shirt, 양방향 162개.
- 최저 recall: Shirt 75.60%, precision 84.75%.
- 최소 margin: index 9237, 0.00189877 (T-shirt/top을 Pullover로 오분류).
- 검증: 전체 프로젝트 테스트 27개 통과; 예측 CSV와 JSON/혼동행렬/체크포인트 해시 일치.
- 결과 디렉터리: `_03_homeworks/homework_3/evaluation/`.
- 최종 ablation과 입력 정규화 비교, 작성자 숙제 후기는 후속 작업으로 남아 있다.
