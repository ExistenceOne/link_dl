# Homework 3 요구사항

출처는 과제 안내 자료 `HW_3.pdf`이다. Fashion MNIST 의류 이미지 분류를 실험하고, 실행한 Jupyter Notebook을 PDF 보고서로 제출한다.

## 문제 1. 입력 정규화 (15점)

- [ ] Fashion MNIST 정규화에 사용할 픽셀 평균과 표준편차를 계산하고 값을 제시한다.
- [ ] 정규화 미적용과 계산한 평균·표준편차로 정규화한 두 조건에서 학습한다.
- [ ] 두 조건의 학습 곡선을 보고서에 넣고 정규화가 학습에 미친 영향을 설명한다.

참고 코드로 [데이터 로더](a_fashion_mnist_data.py), [MNIST 데이터 처리 예제](../../_01_code/_09_fcn_best_practice/a_data_mnist.py), [MNIST CNN 학습 예제](../../_01_code/_11_cnn_architecture/a_mnist_train_cnn.py)가 제시되어 있다.

## 문제 2. CNN 학습 (35점)

- [ ] [지정 데이터 로더](a_fashion_mnist_data.py)를 사용한다. 학습 55,000장, 검증 5,000장, 테스트 10,000장이다.
- [ ] 학습·검증 및 테스트 전처리의 mean/std를 문제 1에서 구한 값으로 바꾼다. Augmentation 등 추가 변환은 허용된다.
- [ ] W&B에 Training Loss, Validation Loss, Training Accuracy, Validation Accuracy를 기록하고 그래프 4개를 보고서에 넣는다.
- [ ] Batch Normalization(BN), Dropout, Augmentation 중 **최소 2개**를 골라 각각 적용·미적용을 비교한다. 한 기법을 비교할 때 나머지 조건은 동일하게 유지한다.
- [ ] 각 ablation 설정의 W&B 그래프 4개를 보고서에 넣고, 기법별 적용·미적용의 정확도 차이를 %p로 기록한다.
- [ ] Early Stopping을 사용한다.
- [ ] 최적 하이퍼파라미터를 Markdown으로 정리한다. Learning Rate, Batch Size, Early Stop Patience, Weight Decay, Normalization Type, Dropout Rate 등이 해당한다.
- [ ] `torchinfo.summary`로 모델 구조를 출력한다.

수업에서 다룬 Optimizers, Regularization, Normalization, Modern CNN 기법을 최대한 활용한다.

## 문제 3. 테스트 평가 (25점)

- [ ] 학습을 완료한 모델로 테스트 데이터 10,000장의 전체 정확도를 계산한다.
- [ ] 혼동행렬을 제시하고 가장 많이 혼동한 클래스쌍을 찾는다.
- [ ] Precision과 recall의 의미를 설명하고, 클래스별 precision과 recall을 보고한다. 지표가 가장 낮은 클래스의 원인을 추정한다.

## 정확도 기준과 점수 상한

| 지표 | 목표 | 중간 구간 및 상한 | 하위 구간 및 상한 |
| --- | --- | --- | --- |
| 문제 2 검증 정확도 | 94.0% 이상 | 91.0% 이상, 94.0% 미만이면 최대 25/35점 | 91.0% 미만이면 최대 10/35점 |
| 문제 3 테스트 정확도 | 93.0% 이상 | 90.0% 이상, 93.0% 미만이면 최대 20/25점 | 90.0% 미만이면 최대 10/25점 |

## 문제 4. 개별 예측 분석 (25점)

- [ ] 테스트 데이터에서 임의로 10장을 선택한다. 각 샘플의 이미지, 실제 레이블, 모델의 예측 레이블, 정분류 여부, 소프트맥스 확신도를 출력한다.
- [ ] 정분류와 오분류 사례 각각을 확신도가 높은 경우와 낮은 경우로 구분해 해석한다.
- [ ] 분류의 애매함을 측정하는 기준을 직접 정의하고 그 이유를 설명한다.
- [ ] 테스트 데이터 10,000장 전체에서 정의한 기준으로 가장 애매한 샘플 1장을 제시한다.

## 보고서 및 제출

- [ ] Jupyter Notebook의 코드 셀을 실행하고 결과가 셀별로 정확히 표시되는지 확인한다.
- [ ] 핵심 코드에 한글 또는 영어 주석을 달고, 실습에서 습득한 기술적 내용과 고찰을 작성한다.
- [ ] 필요한 W&B 결과를 모두 같은 보고서에 넣는다.
- [ ] Notebook 마지막에 **숙제 후기** 섹션을 작성한다. 누락 시 감점된다.
- [ ] 브라우저에서 Notebook을 PDF로 인쇄한다.
- [ ] 파일명은 `딥러닝및실습-3차-홍길동-2020111222-hw3.pdf` 형식을 사용한다. 예시의 이름과 학번을 본인 정보로 바꾼다.
- [ ] [KOREATECH LMS](http://lms.koreatech.ac.kr)의 `homework #3`에 PDF **1개**를 업로드한다.
- [ ] **2026년 11월 21일(토) 23:59**까지 제출한다. 지각 제출은 0점 처리된다.

과제 관련 질문은 [과제 질의응답 문서](https://docs.google.com/document/d/1IHHz9ehpnQ4FrCDciywhweyIlvtpZ_0RZ7w2wNTtWyk/edit?usp=sharing)를 사용한다.
