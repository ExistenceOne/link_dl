"""구버전 생성기 보관본. 현재 제출용 구현은 루트 hw2.ipynb에서 직접 수정한다."""
import argparse
from pathlib import Path
from textwrap import dedent

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]


def build_notebook():
  cells = []

  def md(source):
    cells.append(nbformat.v4.new_markdown_cell(dedent(source).strip()))

  def code(source):
    cells.append(nbformat.v4.new_code_cell(dedent(source).strip()))

  md("""
  # Homework #2: Titanic 생존 예측
  **딥러닝및실습 | 조윤재 | 2021136127**

  승객의 10개 특성으로 생존 여부를 예측하는 MLP를 구현하고 활성화 함수, 배치 크기,
  학습률의 36개 조합을 비교하였다. 검증 손실로 선택한 설정을 재훈련한 뒤
  Early Stopping으로 저장한 모델을 복원하여 Kaggle 제출 파일을 생성한다.

  아래 실험표와 그래프는 실제 학습에서 저장한 epoch별 CSV 및 W&B 기록을 읽어 생성한다.
  노트북을 다시 실행해도 36개 학습을 자동 반복하지 않으며, 최종 모델 복원과 예측은 다시 수행한다.
  Kaggle 제출과 점수 캡처는 별도 수동 단계이다.
  """)
  code("""
  import inspect
  import json
  import sys
  from pathlib import Path
  from types import SimpleNamespace
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import torch
  from torch.utils.data import DataLoader
  from IPython.display import Code, Markdown, display

  ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
              if (p / '_03_homeworks/homework_2/titanic_dataset.py').exists())
  sys.path.insert(0, str(ROOT))
  HOMEWORK = ROOT / '_03_homeworks/homework_2'
  RESULTS = HOMEWORK / 'results'
  from _03_homeworks.homework_2.titanic_dataset import get_preprocessed_dataset
  from _03_homeworks.homework_2.train_titanic_with_argparse_wandb import (
      MyModel, get_data, training_loop, binary_accuracy_f1, build_parser, main)
  from _01_code._99_common_utils.early_stopping import EarlyStopping
  from _03_homeworks.homework_2.predict_titanic import predict_submission
  from _03_homeworks.homework_2.summarize_titanic_runs import summarize_runs
  torch.set_num_threads(1)
  pd.set_option('display.max_columns', 12)
  pd.set_option('display.max_rows', 40)
  plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})
  print('Python:', sys.version.split()[0], '| PyTorch:', torch.__version__)
  print('프로젝트:', ROOT)
  """)
  md("""
  ## 1. 데이터 확인과 전처리

  강의 제공 `titanic_dataset.py`를 기반으로 요금과 나이 결측값을 채우고,
  이름에서 호칭을 추출하며 가족 수 및 혼자 탑승 여부를 생성한다.
  범주형 특성을 정수로 변환한 뒤 입력 10개를 사용한다.
  타깃 `Survived`는 0(사망), 1(생존)이다.

  학습용 891명을 713명/178명으로 나누며, 테스트 418명의 정답은 훈련과 모델 선택에 쓰지 않는다.
  분할과 셔플에 별도 난수 생성기(seed=42)를 사용해 모든 실험에 같은 분할을 적용한다.
  """)
  code("""
  train_set, validation_set, test_set = get_preprocessed_dataset(seed=42)
  raw_train = pd.read_csv(HOMEWORK / 'train.csv')
  raw_test = pd.read_csv(HOMEWORK / 'test.csv')
  columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
             'Embarked', 'title', 'family_num', 'alone']
  display(pd.DataFrame({'분할': ['train', 'validation', 'test'],
                        '표본 수': [len(train_set), len(validation_set), len(test_set)],
                        '입력 특성 수': [10, 10, 10]}))
  display(pd.DataFrame(train_set.dataset.X[:5].numpy(), columns=columns))
  assert torch.isfinite(train_set.dataset.X).all() and torch.isfinite(test_set.X).all()
  assert np.array_equal(train_set.dataset.y.numpy(), raw_train.Survived.to_numpy())
  print('전처리 후 결측값 없음, 원본 타깃 순서 일치')
  print('원본 학습 데이터 생존 비율:', round(raw_train.Survived.mean(), 4))
  """)
  md("""
  ## 2. 모델과 학습 코드

  모델 구조는 `10 -> 20 -> 20 -> 1`이며 두 은닉층에 같은 활성화 함수를 적용한다.
  마지막 출력은 확률이 아닌 logit이다. `BCEWithLogitsLoss`가 sigmoid와 이진 교차 엔트로피를
  함께 계산하므로 출력층에 sigmoid를 중복으로 붙이지 않는다.
  예측은 logit이 0 이상일 때 1이며, sigmoid 확률 0.5 기준과 같다.

  SGD, seed=42, 최대 10,000 epoch, patience=100, delta=0.00001을 고정한다.
  아래에는 실제 실행에 사용하는 모듈의 핵심 코드를 표시하였다.
  """)
  code("""
  display(Code(inspect.getsource(MyModel), language='python'))
  example_run = SimpleNamespace(config=SimpleNamespace(
      activation='relu', n_hidden_unit_list=[20, 20]))
  example_model = MyModel(10, 1, example_run)
  print(example_model)
  print('학습 파라미터 수:', sum(p.numel() for p in example_model.parameters()))
  print('입력 (2, 10) -> 출력', tuple(example_model(torch.zeros(2, 10)).shape))
  """)
  md("""
  ### 학습 손실과 검증 손실 기록

  마지막 배치가 작아도 정확한 전체 평균이 되도록 각 배치 loss에 표본 수를 곱한 후
  전체 표본 수로 나눈다. 학습 손실은 epoch 안에서 갱신되는 모델의 배치별 손실이며,
  검증 손실은 epoch 종료 시점의 고정 모델에서 계산한다.
  검증에는 `eval()`과 `no_grad()`를 사용한다.
  W&B의 두 loss 지표는 모두 `Epoch`를 x축으로 지정한다.
  """)
  code("""
  display(Code(inspect.getsource(get_data), language='python'))
  display(Code(inspect.getsource(training_loop), language='python'))
  """)
  md("""
  ## 3. 기본 모델 훈련 결과

  기본 설정은 ReLU, batch size=128, learning rate=0.001이다.
  기본 실험을 별도로 수행하여 W&B 업로드와 Early Stopping을 확인하였다.
  아래 셀의 `RERUN_BASELINE`은 기본적으로 False이며 저장된 실제 결과를 표시한다.
  재훈련할 경우 새로운 실행 결과가 별도 폴더에 저장된다.
  """)
  code("""
  RERUN_BASELINE = False
  if RERUN_BASELINE:
      baseline = main(build_parser().parse_args([
          '--wandb', '--group', 'hw2-notebook-baseline',
          '--output_dir', str(RESULTS / 'notebook_baseline')]))
  else:
      baseline_paths = sorted((RESULTS / 'baseline').glob('*.json'))
      assert len(baseline_paths) == 1, '기본 실험 기록을 명확하게 지정해야 함'
      baseline = json.loads(baseline_paths[0].read_text())
  display(pd.DataFrame([{k: baseline[k] for k in [
      'activation', 'batch_size', 'learning_rate', 'best_epoch',
      'stopped_epoch', 'best_validation_loss', 'best_validation_accuracy']}]).T)
  display(Markdown('[기본 실험 W&B](' + baseline['wandb_url'] + ')'))

  def show_loss(record, title):
      history = pd.read_csv(ROOT / record['history_path'])
      fig, ax = plt.subplots(figsize=(8, 3.6))
      ax.plot(history['Epoch'], history['Training loss'], label='Training loss', color='#167d8d')
      ax.plot(history['Epoch'], history['Validation loss'], label='Validation loss', color='#b74857')
      ax.axvline(record['best_epoch'], color='#444444', linestyle='--', label='Saved epoch')
      ax.set(xlabel='Epoch', ylabel='Binary cross entropy', title=title)
      ax.legend()
      ax.grid(alpha=0.2)
      fig.tight_layout()
      plt.show()
  show_loss(baseline, 'Baseline: ReLU / batch 128 / learning rate 0.001')
  """)
  md("""
  ## 4. 하이퍼파라미터 36개 조합 비교

  | 변경 항목 | 비교 값 |
  |---|---|
  | 활성화 함수 | Sigmoid, ReLU, ELU, Leaky ReLU |
  | 배치 크기 | 32, 64, 128 |
  | 학습률 | 0.01, 0.001, 0.0001 |

  기존 grid 스크립트로 전체 36개 조합을 학습하였다. 데이터 분할, 초기화 seed,
  모델 크기, optimizer와 종료 기준을 고정했다.
  다음 명령으로 재실험할 수 있다. 새 실험에는 새로운 그룹과 출력 폴더를 지정해야 한다.

  ```bash
  bash _03_homeworks/homework_2/run_titanic_grid.sh --wandb \
      --group hw2-repeat --output_dir _03_homeworks/homework_2/results/repeat
  ```

  표의 accuracy와 F1은 각각의 최대값이 아니라 **저장된 체크포인트와 같은 epoch**의 값이다.
  선택 기준은 저장된 모델의 validation loss 최소이며, 동률이면 해당 모델의 accuracy를 비교한다.
  """)
  code("""
  comparisons = summarize_runs(RESULTS / 'grid', 'hw2-20261012-grid')
  best = comparisons.iloc[0]
  fields = ['activation', 'batch_size', 'learning_rate', 'best_epoch',
            'stopped_epoch', 'best_validation_loss', 'best_validation_accuracy', 'best_validation_f1']
  display(comparisons[fields].round(5))
  print('확인된 조합:', len(comparisons))
  display(Markdown('[최적 탐색 실험 W&B](' + best.wandb_url + ')'))
  """)
  code("""
  activation_order = ['sigmoid', 'relu', 'elu', 'leaky_relu']
  fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
  comparisons['setting'] = comparisons.activation + ' / bs' + comparisons.batch_size.astype(str)
  matrix = comparisons.pivot(index='setting', columns='learning_rate', values='best_validation_loss')
  for ax, (field, selected, title) in zip(axes, [
      ('activation', comparisons[(comparisons.batch_size == best.batch_size) &
          (comparisons.learning_rate == best.learning_rate)].set_index('activation').loc[activation_order],
       f'Batch {best.batch_size}, LR {best.learning_rate:g}'),
      ('batch_size', comparisons[(comparisons.activation == best.activation) &
          (comparisons.learning_rate == best.learning_rate)].sort_values('batch_size').set_index('batch_size'),
       f'{best.activation}, LR {best.learning_rate:g}'),
      ('learning_rate', comparisons[(comparisons.activation == best.activation) &
          (comparisons.batch_size == best.batch_size)].sort_values('learning_rate').set_index('learning_rate'),
       f'{best.activation}, batch {best.batch_size}'),
  ]):
      ax.bar([str(x) for x in selected.index], selected.best_validation_loss,
             color=['#167d8d', '#b74857', '#6a8f44', '#9469a2'][:len(selected)])
      ax.set(xlabel=field, ylabel='Saved validation loss', title=title)
      ax.tick_params(axis='x', labelrotation=20)
      ax.grid(axis='y', alpha=0.2)
  fig.tight_layout()
  plt.show()
  print('각 그래프는 나머지 두 요인을 최적 설정으로 고정한 비교임. 낮은 loss가 좋음.')
  """)
  code("""
  fig, ax = plt.subplots(figsize=(7, 5))
  heat = ax.imshow(matrix.values, cmap='YlGnBu', aspect='auto')
  ax.set_xticks(range(len(matrix.columns)), [f'{x:g}' for x in matrix.columns])
  ax.set_yticks(range(len(matrix.index)), matrix.index)
  for row in range(len(matrix)):
      for col in range(len(matrix.columns)):
          value = matrix.iloc[row, col]
          ax.text(col, row, f'{value:.3f}', ha='center', va='center',
                  color='white' if value > (matrix.values.max()+matrix.values.min())/2 else 'black')
  ax.set(xlabel='Learning rate', title='All 36 runs: saved validation loss')
  fig.colorbar(heat, ax=ax, label='Validation loss')
  fig.tight_layout()
  plt.show()
  reached_limit = int((comparisons.stopped_epoch == comparisons.max_epochs).sum())
  display(Markdown(
      f'최저 검증 손실 설정은 **{best.activation}, batch={best.batch_size}, '
      f'learning rate={best.learning_rate:g}**이며, loss는 **{best.best_validation_loss:.5f}**, '
      f'같은 모델의 검증 정확도는 **{best.best_validation_accuracy:.2f}%**이다. '
      f'총 {reached_limit}개 실험은 최대 epoch에 도달했다. '
      '따라서 비교 결과는 같은 훈련 예산과 현재 분할에서의 결과이며, 모든 설정이 충분히 수렴했다는 뜻은 아니다.'))
  accuracy_best = comparisons.sort_values('best_validation_accuracy', ascending=False).iloc[0]
  display(Markdown(
      f'저장 모델 중 최고 검증 정확도는 {accuracy_best.activation}, '
      f'batch={accuracy_best.batch_size}, learning rate={accuracy_best.learning_rate:g}의 '
      f'**{accuracy_best.best_validation_accuracy:.2f}%**이다. '
      '정확도와 교차 엔트로피는 서로 다른 지표이므로 순위가 같지 않을 수 있다. '
      '본 실험은 미리 정한 검증 손실 기준을 유지해 최종 모델을 선택했다. '
      f'기본 모델 loss {baseline["best_validation_loss"]:.5f}에서 '
      f'선택 모델 loss {best.best_validation_loss:.5f}로 개선되었다.'))
  """)
  md("""
  ## 5. 최적 설정 재훈련과 Early Stopping

  최적 설정을 같은 seed와 분할로 다시 훈련해 별도 final run을 만들었다.
  첫 epoch도 반드시 저장한다. 이후 저장된 loss보다 delta=0.00001 이상 개선되면
  가중치를 저장하고 patience를 초기화한다. 100 epoch 동안 유의한 개선이 없으면 중단한다.
  작은 개선을 무시하는 delta 때문에 저장 epoch와 수치상 절대 최솟값 epoch는 다를 수 있다.
  이 과제에서는 **실제로 저장한 모델의 loss**를 선택과 보고의 일관된 기준으로 사용한다.

  종료 시점 가중치를 그대로 사용하면 검증 성능이 악화된 모델로 예측할 수 있다.
  그러므로 테스트 예측 전에 선택된 final run의 명시적인 checkpoint를 복원한다.
  여러 실험이 덮어쓰는 `latest` 파일은 선택에 사용하지 않는다.
  """)
  code("""
  display(Code(inspect.getsource(EarlyStopping.check_and_save), language='python'))
  final_paths = sorted((RESULTS / 'final').glob('*.json'))
  assert len(final_paths) == 1, '최종 실험 기록을 명확하게 지정해야 함'
  final_path = final_paths[0]
  final = json.loads(final_path.read_text())
  assert all(final[key] == best[key] for key in ['activation', 'batch_size', 'learning_rate'])
  display(pd.DataFrame([{key: final[key] for key in fields}]).T)
  print('예측용 epoch:', final['best_epoch'], '| 학습 종료 epoch:', final['stopped_epoch'])
  display(Markdown('[최종 모델 W&B: 두 loss 그래프](' + final['wandb_url'] + ')'))
  show_loss(final, 'Final run: training and validation loss')
  """)
  code("""
  restored = MyModel(10, 1, SimpleNamespace(config=SimpleNamespace(
      activation=final['activation'], n_hidden_unit_list=final['hidden_sizes'])))
  restored.load_state_dict(torch.load(ROOT / final['checkpoint_path'], map_location='cpu', weights_only=True))
  restored.eval()
  validation = next(iter(DataLoader(validation_set, batch_size=len(validation_set))))
  with torch.no_grad():
      logits = restored(validation['input'])
      targets = validation['target'].float().unsqueeze(1)
      restored_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets).item()
  restored_accuracy, restored_f1 = binary_accuracy_f1(logits, targets)
  assert abs(restored_loss - final['best_validation_loss']) < 1e-6
  assert abs(restored_accuracy - final['best_validation_accuracy']) < 1e-6
  print(f'복원 검증: loss={restored_loss:.6f}, accuracy={restored_accuracy:.2f}%, F1={restored_f1:.4f}')
  print('저장 당시 성능과 복원 후 성능 일치')
  """)
  md("""
  ## 6. 테스트 예측과 submission.csv

  테스트 DataLoader는 섞지 않고 원본 행 순서로 순회한다.
  `PassengerId`는 원본 test.csv에서 가져오고, `Survived`는 0 또는 1로 저장한다.
  `index=False`로 불필요한 인덱스 열을 제거한다.
  테스트 정답이 없으므로 여기에서 테스트 정확도를 계산하지 않는다.
  """)
  code("""
  display(Code(inspect.getsource(predict_submission), language='python'))
  submission_path = HOMEWORK / 'submission.csv'
  submission = predict_submission(final_path, submission_path)
  assert list(submission.columns) == ['PassengerId', 'Survived']
  assert len(submission) == 418
  assert submission.PassengerId.tolist() == raw_test.PassengerId.tolist()
  assert set(submission.Survived.unique()).issubset({0, 1})
  assert not submission.isna().any().any()
  display(submission.head(10))
  display(submission.Survived.value_counts().sort_index().rename('예측 인원').to_frame())
  print('제출 파일:', submission_path)
  print('검증 완료: 418행, 원본 ID 순서, 두 열, 0/1 예측, 결측값 없음')
  """)
  md("""
  ## 7. Kaggle 제출 결과

  **사용자 작성 필요:** 생성된 submission.csv를 Kaggle Titanic 대회에 제출한 후,
  본인의 점수와 순위가 함께 보이는 캡처 이미지를 클라우드에 올리고 이 셀에 이미지 URL을 넣는다.

  - Kaggle 점수: 제출 후 입력
  - Leaderboard 순위: 제출 후 입력
  - 캡처 시점: 제출 후 입력

  이미지 삽입 형식: `![Kaggle 점수 및 순위](이미지_URL)`
  """)
  md("""
  ## 8. 기술적 고찰

  - 배치 크기가 달라지면 epoch당 갱신 횟수도 달라진다. 같은 epoch 수가 같은 SGD 갱신 횟수를 뜻하지는 않는다.
  - 배치 손실의 단순 평균은 마지막 배치가 작은 경우 표본당 평균과 다르다. 표본 수로 가중해 비교 기준을 맞췄다.
  - 최고 accuracy epoch와 최저 loss epoch는 다를 수 있다. 체크포인트와 같은 epoch의 지표를 함께 저장해야 보고와 추론이 일치한다.
  - 첫 epoch 이후 성능이 악화되는 경우에도 예측이 가능하도록 초기 체크포인트를 저장해야 한다.
  - 분할 RNG와 셔플 RNG를 분리하면 노트북에서 다른 난수를 생성해도 실험 분할이 바뀌지 않는다.
  - 강의 제공 전처리는 train/test를 합쳐 결측값 통계와 범주 인코딩을 산출한다. 테스트 정답을 사용하지는 않지만 검증/테스트 입력 분포를 미리 참조한다. 엄격한 일반화 평가에서는 train 분할로만 전처리를 학습하고 나머지에는 적용해야 한다.
  - 이번 실험은 단일 검증 분할로 36개 설정을 선택했으므로 그 분할에 대한 선택 편향이 있다. 반복 seed나 교차 검증을 수행한 결과로 일반화하면 안 된다.
  - 특성 스케일은 강의 코드대로 유지했으므로 Fare와 Age의 크기가 최적화에 영향을 줄 수 있다. 정규화의 효과는 이번 비교에서 검증하지 않았다.
  - 검증 성능은 Kaggle 점수가 아니다. 실제 점수는 제출 후 별도로 확인한다.
  """)
  md("""
  ## 숙제 후기

  **사용자 작성 필요:** 직접 느낀 점, 어려웠던 부분, 새롭게 이해한 점을 본인의 말로 작성한다.
  이 안내 문구를 실제 후기로 교체한 뒤 제출한다.
  """)
  notebook = nbformat.v4.new_notebook(cells=cells)
  notebook.metadata.kernelspec = {
    'display_name': 'Python 3', 'language': 'python', 'name': 'python3',
  }
  notebook.metadata.language_info = {'name': 'python', 'version': '3.9'}
  return notebook


if __name__ == '__main__':
  raise SystemExit(
      '구버전 생성기는 실행하지 않음: 루트 hw2.ipynb를 직접 수정/실행해야 함. '
      '이 파일은 이전 생성 방식의 이력 보존용임.')
