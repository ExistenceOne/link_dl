# Fashion MNIST 모델 선택·튜닝 계획

2026-11-04 갱신. 이전의 ablation 우선 실행 계획을 대체한다.

## 현재 목표와 범위

- 우선 모델과 하이퍼파라미터를 검증 데이터로 선정한다.
- 최종 모델·설정이 확정된 뒤 과제의 ablation을 별도로 설계한다.
- 사용자 승인에 따라 2026-11-04 학습 실행을 시작했다. 실제 진행 상태는 `experiments/study_status.json`에 기록한다.
- 구현된 `_03_homeworks/homework_3/b_fashion_mnist_train.py`를 사용한다. 이번 계획 변경으로 코드를 수정하지 않는다.
- 목표는 검증 정확도 94% 이상을 달성할 수 있는 설정 선정이다. 결과 없이 달성을 가정하지 않는다.

## 공통 조건

- 학습 55,000 / 검증 5,000. `--split_seed 42`로 고정한다.
- 문제 1의 mean/std를 학습 subset에서 계산한 뒤 모든 실행에 같은 값을 명시한다. 현재 스크립트는 이를 자동 계산하지 않는다.
- 탐색 시 `--seed 42`를 사용한다. 최종 후보의 반복 검증에서만 43, 44를 추가한다.
- 시작 설정: Adam(`-o 3`), learning rate 0.001, weight decay 0.0001, batch size 128.
- 최대 100 epoch, 매 epoch 검증, Early Stopping patience 10 / delta 0.00001.
- 위 값은 최초 탐색값이며 최적값으로 확정한 것이 아니다.
- W&B 프로젝트는 기존 `cnn_fashion_mnist`. 각 실행의 네 그래프, 설정, 체크포인트와 학습 시간을 보관한다.
- 테스트 데이터는 모델 순위 결정이나 튜닝에 사용하지 않는다.
- 현재 Trainer는 검증 loss로 체크포인트를 저장한다. 실제 저장된 체크포인트를 다시 검증하여 얻은 정확도를 주 비교 지표로 사용하고, 학습 중 최고 검증 정확도는 별도로 기록한다.
- GoogLeNet의 loss는 보조 loss를 포함하므로 다른 모델과 절대 loss 크기로 성능 순위를 정하지 않는다.

## 1단계: 다섯 모델의 최초 비교 — 5회

모든 실행에서 입력 정규화를 사용하고 augmentation은 끈다. 현재 스크립트 기본값에 맡기지 않고 아래 BN/Dropout 상태를 명시한다.

| 모델 | BN | Dropout | 입력/출력 조정 |
|---|---|---|---|
| CNN | 사용 | 사용, p=0.5 | 단일 채널, 첫 Linear 입력 256 |
| VGG | 미사용 | 사용, p=0.5 | 직접 구현 생성 함수, 32x32 resize |
| NiN | 미사용 | 사용, p=0.5 | 기존 GAP와 10채널 출력 유지 |
| GoogLeNet | 미사용 | 기존 보조 분류기에서 사용, p=0.5 | 주 분류기에 Linear(1024,10) 추가 |
| ResNet | 사용 | 미사용 | 기존 BN과 최종 FC 유지 |

이 비교는 각 원본 구성에 가까운 후보의 초기 성능을 확인하는 screening이다. 단일 시드·단일 학습률에서의 순위가 모델의 최적 성능 순위라고 단정하지 않는다. 후보의 정확도, 과적합/미수렴 여부, 속도, 메모리를 확인하고 우선 상위 두 모델을 튜닝한다. 명확히 미수렴한 모델은 낮은 정확도만으로 탈락시키지 않고 학습률을 확인한다.

## 2단계: 유력 후보의 batch size, learning rate, weight decay 탐색

- batch size 후보: 64 / 128 / 256.
- learning rate 후보: 0.0003 / 0.001 / 0.003.
- weight decay 후보: 0 / 0.0001 / 0.001.
- 먼저 weight decay=0.0001에서 각 모델의 batch size x learning rate 3x3 조합을 비교한다. batch size를 바꾸면서 학습률을 하나로 고정하지 않으며, 학습률을 비례식으로 자동 결정하지도 않는다.
- 모델별 상위 두 batch size/learning rate 조합을 남겨 각각 weight decay 세 값을 비교한다. 이미 실행한 설정은 재사용한다. weight decay도 최종적으로 검증 결과로 선택한다.
- 두 모델의 batch size/learning rate 탐색은 총 18개 설정이다. 1단계의 두 실행을 재사용하면 추가 실행은 16회다. 뒤의 weight decay 탐색은 기존 weight decay=0.0001을 제외하고 최대 8회가 추가된다. 1·2단계의 중복 제거 합계는 최대 29회다.
- 이는 전체 3x3x3 탐색을 대신하는 단계적 탐색이며, 낮은 초기 순위의 조합이 다른 weight decay에서 좋아질 가능성을 배제하지 않는다. 결과가 경계값에 몰리거나 순위가 크게 바뀌면 탐색 범위를 재검토한다.
- 최대 epoch와 Early Stopping 조건은 동일하게 유지한다. batch size별 optimizer update 수가 다르므로, 정확도와 함께 실제 epoch, 학습 시간, 메모리 사용량도 기록한다.
- 이 단계에서는 BN/Dropout/augmentation을 1단계와 동일하게 유지한다.
- 각 단계 결과를 검토한 후 다음 단계 실행 범위를 정한다. 계획상의 조합 수가 일괄 실행 승인을 의미하지 않는다.

## 3단계: 유력 후보의 규제·증강 설정 튜닝

- 2단계의 좋은 batch size/learning rate/weight decay 설정을 기준으로 augmentation 사용 여부를 확인한다.
- 이어 Dropout 비율 0 / 0.2 / 0.5를 후보로 비교한다. ResNet에서 Dropout을 사용하면 원본에 없는 FC 직전 계층을 추가하는 실험임을 기록한다.
- 원래 BN이 없는 모델의 BN 추가는 필수로 하지 않는다. 해당 모델이 유력하지만 학습 불안정/성능 부족이 남는 경우 추가 튜닝 후보로 검토한다.
- augmentation/Dropout 변경 후 성능이 크게 달라지면 batch size/learning rate/weight decay의 인접 설정도 다시 확인한다. 순차 탐색이 전체 최적 조합을 보장한다고 주장하지 않는다.
- patience 변경은 기본 조건에서 수렴 문제가 확인될 때 검토한다. batch size는 문제 발생 여부와 무관하게 2단계의 정식 튜닝 항목으로 포함한다.
- 이 단계의 사용/미사용 비교는 좋은 설정을 찾는 튜닝이다. 다른 인자까지 변경한 결과를 최종 ablation의 기여도로 제시하지 않는다.

## 4단계: 최종 후보 재현성 검증과 설정 확정

- 최종 후보의 고정 설정을 seed 42 / 43 / 44에서 확인한다. split_seed는 계속 42로 유지한다.
- 가능하면 최종 두 후보를 같은 seed 목록으로 비교한다. 평균 검증 정확도, 표준편차, 학습 시간으로 선택을 뒷받침한다.
- 이미 동일한 조건으로 실행한 seed 42 결과는 재사용한다. 반복 실행 수는 남은 후보 수에 따라 정한다.
- 시드 하나의 유리한 결과만으로 우수성을 단정하지 않는다.
- 선택된 모델, 입출력 조정, normalization, augmentation, dropout rate, optimizer, learning rate, weight decay, batch size, Early Stopping, seed와 저장 체크포인트를 확정한다.

## 이후 별도 단계

1. 확정된 모델에서 BN/Dropout/augmentation 중 최소 두 기법의 ablation을 설계한다. 비교 대상 외에는 고정하고 각 설정을 새로 학습한다.
2. 문제 1의 입력 정규화 유무 비교를 완성한다. 모델의 BN on/off와 혼동하지 않는다.
3. 검증으로 선택한 최종 체크포인트의 테스트 정확도, 혼동행렬, 클래스별 지표와 예측 사례를 분석한다.

## 현재 진행 상태

- [x] 학습 스크립트 구현 및 기존 스타일로 단순화.
- [x] 모델 20개 BN/Dropout 조합을 포함한 오프라인 회귀 테스트 통과.
- [x] 계획을 모델 선택·튜닝 우선으로 변경.
- [x] 고정 분할의 mean/std 준비: mean=0.2855656942074541, std=0.3527285864947437.
- [ ] 1단계 최초 모델 비교 실행 중.
- [ ] 결과에 근거한 튜닝 및 최종 설정 확정.
- [ ] 최종 설정에서 ablation 설계·실행.

## 실행과 보고서

- `scripts/fashion_mnist_study.py`가 단계별 선택과 순차 실행을 담당한다. 실행 결과가 있는 동일 설정은 재사용한다.
- `scripts/fashion_mnist_experiment.py`가 기존 학습 스크립트를 호출하고 W&B/epoch 로그/저장 체크포인트의 검증 정확도를 수집한다.
- 기존 학습 스크립트와 Trainer는 변경하지 않았다. 제공 로더의 dataset 픽셀만 원본 ToTensor와 일치하는 CPU 변환 후 GPU에 캐시하여 전처리 반복을 줄인다. 분할/샘플 수/정규화는 동일하다.
- 3단계는 augmentation 두 조건과 Dropout 비율 세 조건의 조합을 비교하며, BN이 없던 유력 후보는 BN 추가도 튜닝한다. 이는 최종 ablation과 구분한다.
- 보고서: `output/pdf/fashion_mnist_model_selection_report.pdf`.
- 실행 출력 포함 노트북: `_03_homeworks/homework_3/fashion_mnist_model_selection_report.ipynb`.
- 완료된 실행마다 보고서를 갱신한다. 전체 실험 완료 전에는 중간 보고서라고 명시한다. 중단 시 실패 상태를 보고서에 반영한다.
- 최종 후보 검증은 seed 42/43/44로 수행하고, 테스트 데이터와 최종 ablation은 이번 실행에 포함하지 않는다.
