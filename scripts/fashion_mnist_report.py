"""Generate a PDF and notebook from actual recorded tuning results, including partial progress."""
import base64
from datetime import datetime, timezone, timedelta
import html
import io
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '_03_homeworks/homework_3/experiments'
OUTPUT = ROOT / 'output/pdf'
FIGURES = BASE / 'figures'
BUNDLE = Path('/home/link/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
sys.path.append(str(BUNDLE))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, LongTable,
                               TableStyle, PageBreak, Preformatted)


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def build():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = [read(path) for path in BASE.glob('*/result.json')]
    results = sorted([r for r in results if r.get('status') == 'completed'],
                     key=lambda r: (r['configuration']['stage'] != 'baseline',
                                    r['configuration']['stage'], r['args']['model'],
                                    r['args']['batch_size'], r['args']['learning_rate'], r['args']['seed']))
    status = read(BASE / 'study_status.json', {'status': 'running'})
    final = read(BASE / 'final_selection.json')
    evaluation_dir = ROOT / '_03_homeworks/homework_3/evaluation'
    evaluation = read(evaluation_dir / 'metrics.json')
    if evaluation and final:
        if evaluation['checkpoint'] != final['winner']['seed42_configuration']['checkpoint']:
            raise ValueError('Test evaluation belongs to a different selected checkpoint')
    stats = read(BASE / 'statistics.json')
    date = datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST')
    pdfmetrics.registerFont(UnicodeCIDFont('HYSMyeongJo-Medium'))
    styles = getSampleStyleSheet()
    for name in ['Normal', 'BodyText', 'Title', 'Heading1', 'Heading2']:
        styles[name].fontName = 'HYSMyeongJo-Medium'
    styles['Normal'].fontSize = 10
    styles['Normal'].leading = 15
    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 15
    styles['Title'].fontSize = 23
    styles['Title'].leading = 32
    styles.add(ParagraphStyle('SmallNote', parent=styles['Normal'], fontSize=8, leading=12))
    story = []
    notebook = []
    count = 0

    def paragraph(text, style='Normal'):
        story.append(Paragraph(text.replace('·', '/'), styles[style]))
        story.append(Spacer(1, 7))

    def markdown(text):
        notebook.append({'cell_type': 'markdown', 'metadata': {}, 'source': text})

    def code(source, output):
        nonlocal count
        count += 1
        notebook.append({'cell_type': 'code', 'metadata': {}, 'source': source,
                         'execution_count': count, 'outputs': [{'output_type': 'stream',
                         'name': 'stdout', 'text': output}]})

    def table(rows, widths=None):
        formatted = [[Paragraph(html.escape(str(c)).replace('·', '/'), styles['SmallNote']) for c in row] for row in rows]
        t = LongTable(formatted, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#dce8f2')),
            ('VALIGN', (0,0), (-1,-1), 'TOP'), ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#c2cad2')),
            ('LEFTPADDING', (0,0), (-1,-1), 5), ('RIGHTPADDING', (0,0), (-1,-1), 5),
            ('TOPPADDING', (0,0), (-1,-1), 5), ('BOTTOMPADDING', (0,0), (-1,-1), 5)]))
        story.append(t)
        story.append(Spacer(1, 12))

    title = 'Fashion MNIST 모델 선택 및 평가 보고서' if evaluation else 'Fashion MNIST 모델 선택 및 튜닝 보고서'
    markdown('# ' + title)
    paragraph(title, 'Title')
    paragraph(('실험 완료' if final else '진행 중인 실험의 중간 보고서') + ' | ' + date)
    paragraph(f'모델 선택/튜닝 실험: {len(results)}회. 모델과 설정의 선택에는 검증 데이터만 사용했다.')
    if evaluation:
        paragraph(f"선택 완료 후 고정한 대표 체크포인트를 테스트 10,000개로 평가했다. 테스트 정확도는 "
                  f"{evaluation['accuracy']:.2f}%로 93% 기준을 {'충족했다' if evaluation['test_target_met'] else '충족하지 못했다'}. 최종 ablation은 아직 수행하지 않았다.")
    else:
        paragraph('현재 범위는 모델 선택 및 튜닝이다. 최종 ablation, 테스트 정확도, 혼동행렬과 예측 사례는 후속 과제이며 이 보고서에 완료된 결과로 제시하지 않는다.')
    if final:
        winner = final['winner']
        paragraph(f"선택 모델: {winner['model']}. 3개 학습 시드의 검증 정확도 평균 "
                  f"{winner['mean_validation_accuracy']:.2f}%, 표본 표준편차 {winner['std_validation_accuracy']:.2f}%p.", 'Heading2')
        paragraph('검증 정확도 94% 목표: ' + ('3시드 평균 기준 달성.' if final['validation_target_met'] else
                  '3시드 평균 기준 미달. 대표 seed 42 저장 모델은 94.06%이며, 평균과 단일 실행 성능을 구분한다.'))
    else:
        paragraph('아직 최종 모델이나 최종 성능을 확정하지 않았다. 아래 표와 그래프는 완료된 실제 실행만 포함한다.')
        if status.get('status') == 'failed':
            paragraph('실험 실행이 중단되었다: ' + html.escape(status.get('error', '원인 확인 필요')))
        elif status.get('current'):
            paragraph('현재 실행: ' + html.escape(status['current']), 'SmallNote')
    code('import json\nfrom pathlib import Path\nROOT = Path("/home/link/link_dl")\nBASE = ROOT / "_03_homeworks/homework_3/experiments"\nprint((BASE / "statistics.json").read_text())', json.dumps(stats, indent=2))
    paragraph('1. 데이터와 실행 조건', 'Heading1')
    table([['항목', '내용'], ['학습 / 검증 / 테스트', '55,000 / 5,000 / ' + ('10,000 (선택 완료 후 평가)' if evaluation else '미사용')],
           ['분할 시드 / 탐색 시드', '42 / 42; 반복 검증 시 43, 44 추가'],
           ['학습 픽셀 mean / population std', f"{stats['mean']:.10f} / {stats['std']:.10f}"],
           ['평균·표준편차 산정 범위', f"학습 subset의 {stats['pixels']:,}개 픽셀만 사용"],
           ['Optimizer / epoch / Early Stopping', 'Adam / 최대 100 / patience=10, delta=0.00001'],
           ['GPU', 'NVIDIA GeForce RTX 3060 8 GB'],
           ['선택 기준', '검증 loss로 저장된 체크포인트를 재평가한 검증 정확도'],
           ['W&B', 'cnn_fashion_mnist 프로젝트; 각 실행 링크를 아래에 제시']], [145, 365])
    paragraph('기존 b_fashion_mnist_train.py와 원본 Trainer를 사용했다. 별도 실행 스크립트가 기록 수집을 담당한다. '
              '제공 데이터 함수의 분할과 DataLoader를 유지하고, ToTensor의 CPU float32/255 결과를 GPU에 캐시했다. '
              '캐시와 원본 픽셀의 일치를 확인했으며 데이터 수를 줄이거나 배치를 생략하지 않았다.')
    paragraph('모델별 조정: CNN의 입력 채널을 1로, 첫 FC 입력을 256으로 변경했다. VGG는 32×32 resize, '
              'NiN은 기존 GAP, GoogLeNet은 주 경로에 Linear(1024,10) 추가, ResNet은 기존 구조를 사용한다. '
              '원래 BN이 없는 모델의 초기 비교에서는 BN을 추가하지 않았다.')
    paragraph('2. 탐색 방법', 'Heading1')
    paragraph('초기 비교는 다섯 모델을 batch=128, LR=0.001, weight decay=0.0001, augmentation 없음으로 실행한다. '
              '유력한 두 모델에서 batch 64/128/256 × LR 0.0003/0.001/0.003을 탐색하고, '
              '상위 두 조합에서 weight decay 0/0.0001/0.001을 비교한다. '
              '이후 augmentation과 Dropout 0/0.2/0.5를 비교하고, 필요 시 원래 BN이 없는 후보에 BN을 추가해 튜닝한다. '
              '최종 후보는 학습 시드 42/43/44에서 반복 확인한다.')
    paragraph('이는 단계적 탐색이며 모든 조합의 전역 최적값을 보장하지 않는다. 단일 초기 학습률에서 탈락한 모델의 잠재적 성능을 배제하지 않는다. '
              'batch size가 달라도 최대 epoch와 Early Stopping 조건은 동일하므로 optimizer update 수는 다르다.')
    paragraph('3. 실제 실행 결과', 'Heading1')
    rows = [['ID', '모델', 'Batch', 'LR', 'WD', 'BN', 'Drop', 'Aug', 'Seed', 'Val %', 'Epoch']]
    for index, r in enumerate(results, 1):
        a = r['args']
        rows.append([f'E{index:02}', a['model'], a['batch_size'], f"{a['learning_rate']:g}",
                     f"{a['weight_decay']:g}", a['normalization'], a['dropout_rate'] if a['dropout'] else 0,
                     int(a['augment']), a['seed'], f"{r['checkpoint_validation_accuracy']:.2f}", r['epochs']])
    table(rows, [28, 57, 42, 45, 45, 28, 38, 30, 35, 48, 45])
    markdown('## 모델 선택/튜닝 실행 결과\n\n검증 loss로 저장된 체크포인트를 재평가한 검증 정확도를 사용한다. 모델 선택/튜닝에는 테스트 데이터를 사용하지 않았다.')
    code('results = [json.loads(p.read_text()) for p in BASE.glob("*/result.json")]\nfor r in results:\n    print(r["configuration"]["id"], r["checkpoint_validation_accuracy"], r["wandb_url"])',
         '\n'.join(f"{r['configuration']['id']}  {r['checkpoint_validation_accuracy']:.2f}%  {r['wandb_url']}" for r in results))
    if results:
        best = max(results, key=lambda r: r['checkpoint_validation_accuracy'])
        paragraph(f"현재 최고 단일 실행: {best['args']['model']}, 저장 체크포인트 검증 정확도 "
                  f"{best['checkpoint_validation_accuracy']:.2f}%. 학습 중 최고 정확도 "
                  f"{best['best_logged_validation_accuracy']:.2f}%와 구분한다.")
        baseline = sorted([r for r in results if r['configuration']['stage'] == 'baseline'],
                          key=lambda r: -r['checkpoint_validation_accuracy'])
        if len(baseline) >= 2:
            paragraph('초기 모델 비교 해석', 'Heading2')
            paragraph(f"완료된 초기 비교 중 {baseline[0]['args']['model']}이 "
                      f"{baseline[0]['checkpoint_validation_accuracy']:.2f}%로 가장 높고, "
                      f"{baseline[1]['args']['model']}이 {baseline[1]['checkpoint_validation_accuracy']:.2f}%로 뒤를 이었다. "
                      '이는 동일 학습률에서 원본에 가까운 서로 다른 모델 구성을 비교한 결과다. '
                      '구조 하나만의 인과 효과나 각 모델의 최적 성능 순위로 해석하지 않는다.')
        tuned_models = sorted({r['args']['model'] for r in results if r['configuration']['stage'] != 'baseline'})
        if tuned_models:
            paragraph('모델별 튜닝 결과 해석', 'Heading2')
        for model_name in tuned_models:
            candidates = [r for r in results if r['args']['model'] == model_name and r['args']['seed'] == 42]
            chosen = max(candidates, key=lambda r: r['checkpoint_validation_accuracy'])
            a = chosen['args']
            original = next((r for r in baseline if r['args']['model'] == model_name), None)
            text = (f"{model_name}: 현재 좋은 단일 설정은 batch={a['batch_size']}, "
                    f"LR={a['learning_rate']}, WD={a['weight_decay']}, BN={a['normalization']}, "
                    f"Dropout={a['dropout_rate'] if a['dropout'] else 0}, augmentation={a['augment']}이며 "
                    f"저장 모델 검증 정확도는 {chosen['checkpoint_validation_accuracy']:.2f}%다. ")
            if original:
                difference = chosen['checkpoint_validation_accuracy'] - original['checkpoint_validation_accuracy']
                text += f"초기 설정 대비 {difference:+.2f}%p 차이다. 여러 설정이 달라졌으므로 이 차이를 특정 기법 하나의 기여도로 해석하지 않는다."
            paragraph(text)
    if final:
        paragraph('4. 최종 후보의 반복 검증', 'Heading1')
        table([['모델', 'seed 42 / 43 / 44 (%)', '평균 (%)', '표본 표준편차 (%p)']] +
              [[c['model'], ' / '.join(f'{x:.2f}' for x in c['scores']),
                f"{c['mean_validation_accuracy']:.2f}", f"{c['std_validation_accuracy']:.2f}"] for c in final['candidates']],
              [80, 220, 90, 120])
        paragraph('최종 설정과 대표 체크포인트는 다음과 같다. 대표 파일은 선택 모델의 seed 42 실행이며, 가장 잘 나온 시드만 골라 성능을 주장하지 않는다.')
        chosen = final['winner']['seed42_configuration']
        table([['설정', '값']] + [[k, v] for k, v in chosen['args'].items() if k not in ('wandb',)], [150, 360])
        paragraph(html.escape(chosen['checkpoint']), 'SmallNote')
        code('selection = json.loads((BASE / "final_selection.json").read_text())\nprint(json.dumps(selection["winner"], indent=2, ensure_ascii=False))',
             json.dumps(final['winner'], indent=2, ensure_ascii=False))
    paragraph('고찰과 제한', 'Heading1')
    paragraph('모델 선택/튜닝 표의 정확도는 고정된 검증 분할에 대한 값이다. 반복적인 튜닝으로 검증 분할에 과적합될 수 있으므로 '
              '별도로 고정한 대표 체크포인트의 테스트 결과와 구분한다. GoogLeNet의 학습·검증 loss에는 '
              '주 loss + 0.3 × 두 보조 loss가 포함되어 다른 모델과 절대 loss 크기를 직접 비교하지 않는다. '
              '검증 loss는 기존 Trainer의 배치 평균 방식이며 서로 다른 batch size의 절대 loss만으로 설정을 선택하지 않았다.')
    paragraph('다음 단계는 선택된 설정을 고정한 후 최소 두 기법의 ablation을 수행하는 것이다. '
              '문제 1의 입력 정규화 유무 비교와 작성자 본인의 숙제 후기는 아직 별도 작업으로 남아 있다.')
    if evaluation:
        story.append(PageBreak())
        paragraph('5. 고정 모델의 테스트 평가 (문제 3)', 'Heading1')
        paragraph(f"대표 VGG 체크포인트를 strict=True로 복원했으며, 테스트 접근 전에 검증 5,000개 정확도 "
                  f"{evaluation['validation_restore_accuracy']:.2f}%를 재현했다. 입력은 32x32로 resize한 뒤 학습 mean/std로 정규화했다. "
                  f"추론에는 augmentation을 적용하지 않았고 model.eval()/inference_mode()를 사용했다. 제공 테스트 로더의 전체 배치를 {evaluation['evaluation_batch_size']}개씩 나눠 처리했다.")
        table([['항목', '결과'], ['Test accuracy', f"{evaluation['accuracy']:.2f}%"],
               ['정답 / 오답 / 전체', f"{evaluation['num_correct']} / {evaluation['num_incorrect']} / {evaluation['num_samples']}"],
               ['93% 기준', '충족' if evaluation['test_target_met'] else '미충족'],
               ['검증 정확도 - 테스트 정확도', f"{evaluation['validation_restore_accuracy'] - evaluation['accuracy']:.2f}%p"],
               ['추론 전후 BN 통계', '동일함' if evaluation['bn_statistics_unchanged'] else '변경됨']], [210, 300])
        paragraph('Precision은 해당 클래스로 예측한 샘플 중 정답 비율 TP/(TP+FP), recall은 실제 해당 클래스 샘플 중 찾아낸 비율 TP/(TP+FN)이다. '
                  'F1은 precision과 recall의 조화평균이다. 분모가 0인 경우에는 0으로 처리했다.')
        table([['Class', 'Precision %', 'Recall %', 'F1 %', 'Support']] +
              [[c['name'], f"{100*c['precision']:.2f}", f"{100*c['recall']:.2f}", f"{100*c['f1-score']:.2f}", int(c['support'])]
               for c in evaluation['per_class']] +
              [[label, f"{100*evaluation[key]['precision']:.2f}", f"{100*evaluation[key]['recall']:.2f}",
                f"{100*evaluation[key]['f1-score']:.2f}", int(evaluation[key]['support'])]
               for label, key in [('Macro avg', 'macro_average'), ('Weighted avg', 'weighted_average')]], [150, 95, 90, 85, 90])
        paragraph('각 클래스의 support가 1,000개로 동일하므로 macro 평균과 weighted 평균은 이 결과에서 같다.')
        markdown('## 문제 3: 고정한 최종 모델의 테스트 평가\n\n모델/시드 선택에 테스트 데이터를 사용하지 않았다. 평가 전 검증 정확도 재현 및 BN 통계 불변을 확인했다.')
        code('EVALUATION = ROOT / "_03_homeworks/homework_3/evaluation"\nprint((EVALUATION / "evaluation.log").read_text())',
             (evaluation_dir / 'evaluation.log').read_text(errors='replace'))
        code('test_metrics = json.loads((EVALUATION / "metrics.json").read_text())\nprint(json.dumps(test_metrics, indent=2, ensure_ascii=False))',
             json.dumps(evaluation, indent=2, ensure_ascii=False))

        def evaluation_image(filename, width, height):
            nonlocal count
            path = evaluation_dir / filename
            story.append(Image(str(path), width=width, height=height))
            count += 1
            notebook.append({'cell_type': 'code', 'metadata': {}, 'execution_count': count,
                             'source': f'from IPython.display import Image, display\ndisplay(Image(filename=str(EVALUATION / {filename!r})))',
                             'outputs': [{'output_type': 'display_data', 'metadata': {},
                                          'data': {'image/png': base64.b64encode(path.read_bytes()).decode()}}]})

        labels = [c['name'] for c in evaluation['per_class']]
        pair = evaluation['largest_confusion_pair']
        directed = evaluation['largest_directed_confusion']
        for normalized in [False, True]:
            story.append(PageBreak())
            paragraph('테스트 혼동행렬: ' + ('행별 비율' if normalized else '원시 개수'), 'Heading1')
            paragraph(f"양방향 혼동이 가장 많은 쌍은 {labels[pair['labels'][0]]}와 {labels[pair['labels'][1]]}로 {pair['count']}개다. "
                      f"단방향 최대는 {labels[directed['true_label']]}를 {labels[directed['predicted_label']]}로 예측한 {directed['count']}개다.")
            evaluation_image('confusion_matrix_normalized.png' if normalized else 'confusion_matrix_counts.png', 6.8*inch, 6.12*inch)
            paragraph('세로축은 정답, 가로축은 예측 클래스다. 행별 비율 그림의 대각선은 각 클래스 recall에 해당한다.')

        story.append(PageBreak())
        paragraph('낮은 클래스 지표와 실제 오분류 분석', 'Heading1')
        lowest_precision = min(evaluation['per_class'], key=lambda c: c['precision'])
        lowest_recall = min(evaluation['per_class'], key=lambda c: c['recall'])
        paragraph(f"최저 precision은 {lowest_precision['name']} {100*lowest_precision['precision']:.2f}%, "
                  f"최저 recall은 {lowest_recall['name']} {100*lowest_recall['recall']:.2f}%다. "
                  f"{lowest_recall['name']} 정답 샘플 {int(lowest_recall['support'])}개 중 "
                  f"{int(lowest_recall['support']) - evaluation['confusion_matrix'][lowest_recall['label']][lowest_recall['label']]}개를 다른 클래스로 분류했다.")
        paragraph('아래는 가장 낮은 recall 클래스의 오분류 중 테스트 index 순으로 앞의 10개다. 무작위 10개 예측 결과와는 별도의 분석용 사례다.')
        if evaluation['lowest_recall_error_samples']:
            evaluation_image('lowest_recall_errors.png', 7.05*inch, 3.76*inch)
        paragraph('관찰: Shirt를 T-shirt/top, Pullover, Coat 등으로 예측한 사례가 보이고, Dress로 예측한 사례도 있다. '
                  '추정: 상의의 전체 윤곽이 겹치고 28x28 해상도에서는 깃/여밈 등 세부 특징이 약해지는 것이 혼동의 원인일 수 있다. '
                  '이는 사례 관찰을 바탕으로 한 추정이며, 데이터의 레이블 오류나 특정 원인을 확정한 것은 아니다.')
        groups = evaluation['confidence_groups']
        table([['확신도 기준', '정분류', '오분류'], ['0.9 이상', groups['high_correct'], groups['high_incorrect']],
               ['0.9 미만', groups['low_correct'], groups['low_incorrect']]], [250, 130, 130])
        paragraph('고확신 오분류도 존재하므로 큰 softmax 값이 정답을 보장하지 않는다. 0.9는 사전에 정한 해석용 구분이며, 확률 보정을 검증한 기준은 아니다.')

        story.append(PageBreak())
        paragraph('6. 무작위 테스트 이미지 10개 (문제 4)', 'Heading1')
        paragraph(f"예측 결과와 무관하게 sample_seed={evaluation['sample_seed']}로 10개를 중복 없이 추출했다. 정답/오답 개수를 맞추기 위해 재추출하지 않았다.")
        evaluation_image('random_samples.png', 7.05*inch, 3.76*inch)
        table([['Index', '정답', '예측', '정답 여부', '확신도 %', '해석']] +
              [[s['index'], s['label_name'], s['prediction_name'], 'O' if s['correct'] else 'X', f"{100*s['confidence']:.4f}",
                ('고확신' if s['confidence_level'] == 'high' else '저확신') + (' 정분류' if s['correct'] else ' 오분류')]
               for s in evaluation['random_samples']], [48, 115, 115, 65, 80, 87])
        sample_right = sum(s['correct'] for s in evaluation['random_samples'])
        sample_high = sum(s['correct'] and s['confidence_level'] == 'high' for s in evaluation['random_samples'])
        paragraph(f"선택된 10개 중 {sample_right}개가 정분류이며, 그중 고확신 정분류는 {sample_high}개다. "
                  '이 작은 표본의 정답률을 전체 테스트 정확도로 해석하지 않는다. 오분류 해석은 앞의 별도 사례와 전체 확신도 집계를 참고한다.')

        story.append(PageBreak())
        paragraph('가장 애매한 샘플: 상위 두 확률 차이 최소', 'Heading1')
        ambiguous = evaluation['ambiguous_sample']
        paragraph('애매함은 top-1 확률과 top-2 확률의 차이 margin으로 정의했다. 차이가 작을수록 모델이 두 후보를 뚜렷하게 구분하지 못한다고 해석한다. '
                  '테스트 10,000개 전체에서 margin이 최소인 샘플을 골랐고, 동률이면 index가 작은 것을 선택한다.')
        evaluation_image('most_ambiguous.png', 3.8*inch, 3.38*inch)
        table([['항목', '값'], ['Index / 정답', f"{ambiguous['index']} / {ambiguous['label_name']}"],
               ['Top-1', f"{ambiguous['prediction_name']}: {100*ambiguous['confidence']:.4f}%"],
               ['Top-2', f"{ambiguous['second_name']}: {100*ambiguous['second_probability']:.4f}%"],
               ['Margin', f"{ambiguous['margin']:.8f} ({100*ambiguous['margin']:.4f}%p)"],
               ['정답 여부', '정분류' if ambiguous['correct'] else '오분류']], [180, 330])
        paragraph('이 사례는 상위 두 확률이 매우 가깝다. 원본 이미지의 긴 소매와 상의 윤곽만으로 클래스를 구분하기 어려워 보인다는 시각적 해석은 가능하지만, '
                  '선정 자체는 사람의 주관적 선택이 아니라 사전에 정한 margin 기준으로 수행했다.')
        if evaluation.get('wandb_url'):
            paragraph(f'<link href="{html.escape(evaluation["wandb_url"], quote=True)}">W&amp;B 테스트 평가 run</link>', 'SmallNote')
        else:
            paragraph('W&amp;B 온라인 링크 없음. 지표와 이미지는 로컬 평가 결과에 저장했다.', 'SmallNote')
        paragraph('재현 기록: 체크포인트 SHA-256 ' + evaluation['checkpoint_sha256'], 'SmallNote')
        paragraph('남은 과제는 최종 모델의 최소 두 기법 ablation, 입력 정규화 유무 비교와 작성자 본인의 숙제 후기다. '
                  '테스트 결과를 본 뒤 모델이나 시드를 다시 선택하지 않았다.')
        markdown(f"""## 테스트 결과 해석

검증 데이터에서 선택한 seed 42 모델을 그대로 평가했다. 테스트 접근 전에 검증 정확도 {evaluation['validation_restore_accuracy']:.2f}%를 재현했고,
테스트 정확도는 **{evaluation['accuracy']:.2f}%** ({evaluation['num_correct']}/{evaluation['num_samples']})로 93% 기준을 {'충족했다' if evaluation['test_target_met'] else '충족하지 못했다'}.
검증 정확도와의 차이는 {evaluation['validation_restore_accuracy']-evaluation['accuracy']:.2f}%p이며, 이 결과를 보고 모델/시드를 다시 고르지 않았다.

Precision은 TP/(TP+FP), recall은 TP/(TP+FN), F1은 두 지표의 조화평균이다. 분모가 0이면 0으로 처리했다.
""")
        markdown('|Class|Precision|Recall|F1|Support|\n|---|---:|---:|---:|---:|\n' + '\n'.join(
            f"|{c['name']}|{100*c['precision']:.2f}%|{100*c['recall']:.2f}%|{100*c['f1-score']:.2f}%|{int(c['support'])}|"
            for c in evaluation['per_class']))
        markdown(f"""### 낮은 클래스 지표와 혼동

최저 precision은 {lowest_precision['name']} {100*lowest_precision['precision']:.2f}%, 최저 recall은 {lowest_recall['name']} {100*lowest_recall['recall']:.2f}%이다.
양방향 최다 혼동은 {labels[pair['labels'][0]]}와 {labels[pair['labels'][1]]}, 합계 {pair['count']}개다.
단방향으로는 {labels[directed['true_label']]}를 {labels[directed['predicted_label']]}로 예측한 {directed['count']}개가 가장 많다.
실제 Shirt 오분류 이미지에서 T-shirt/top, Pullover, Coat 등과 비슷한 윤곽이 관찰된다.
낮은 해상도에서 깃/여밈 등 세부 특징이 약해지는 것이 한 원인일 수 있다는 해석은 **관찰에 근거한 추정**이다.

### 무작위 10개와 확신도

sample_seed={evaluation['sample_seed']}로 예측 결과와 무관하게 10개를 선택했고 재추출하지 않았다.
{sample_right}개가 정분류이며 고확신 정분류는 {sample_high}개다. 이미지와 JSON에는 각 정답/예측/확신도가 기록되어 있다.
해석용 기준 0.9 이상을 고확신으로 구분했다. 전체 테스트에서는 고확신 정분류 {groups['high_correct']}개,
고확신 오분류 {groups['high_incorrect']}개, 저확신 정분류 {groups['low_correct']}개, 저확신 오분류 {groups['low_incorrect']}개다.
따라서 큰 softmax 값은 정답을 보장하지 않으며, 작은 표본의 정답률은 전체 정확도와 구분한다.

### 가장 애매한 샘플

애매함은 상위 두 softmax 확률 차이 margin의 최소로 정의했다. 차이가 작을수록 두 후보를 명확히 구분하지 못한다고 해석한다.
10,000개 중 index {ambiguous['index']}의 margin **{ambiguous['margin']:.8f}**이 최소다. 동률이면 index가 작은 것을 선택한다.
정답은 {ambiguous['label_name']}, top-1은 {ambiguous['prediction_name']} {100*ambiguous['confidence']:.4f}%,
top-2는 {ambiguous['second_name']} {100*ambiguous['second_probability']:.4f}%로 {'정분류' if ambiguous['correct'] else '오분류'}였다.
이미지에서 긴 소매와 상의 윤곽이 비슷해 보인다는 해석과, 수치 기준으로 이 샘플을 선정했다는 사실을 구분한다.

### 남은 과제

최종 모델의 최소 두 기법 ablation, 입력 정규화 유무 비교와 작성자 본인의 숙제 후기는 아직 수행/작성하지 않았다.
테스트 성능을 보고 추가 튜닝하는 절차는 수행하지 않았다.
""")
    for index, result in enumerate(results, 1):
        folder = Path(result['checkpoint']).parent
        history = [json.loads(line) for line in (folder / 'history.jsonl').read_text().splitlines()]
        figure, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
        for axis, key in zip(axes.flat, ['Training loss', 'Validation loss', 'Training accuracy (%)', 'Validation accuracy (%)']):
            axis.plot([h['Epoch'] for h in history], [h[key] for h in history], linewidth=1.5)
            axis.set_title(key)
            axis.set_xlabel('Epoch')
            axis.grid(alpha=0.25)
        figure.suptitle(f"E{index:02} | {result['configuration']['id']}", fontsize=10)
        filename = FIGURES / f"{result['configuration']['id']}.png"
        figure.savefig(filename, dpi=150)
        plt.close(figure)
        story.append(PageBreak())
        paragraph(f"E{index:02}. {result['args']['model']} 학습 곡선", 'Heading1')
        a = result['args']
        paragraph(f"batch={a['batch_size']}, LR={a['learning_rate']}, WD={a['weight_decay']}, "
                  f"BN={a['normalization']}, Dropout={a['dropout_rate'] if a['dropout'] else 0}, "
                  f"augmentation={a['augment']}, seed={a['seed']}")
        paragraph(f"저장 모델 검증 정확도 {result['checkpoint_validation_accuracy']:.2f}%, "
                  f"총 {result['epochs']} epoch, {result['elapsed_seconds'] / 60:.1f}분, "
                  f"최대 GPU 할당 메모리 {result['peak_gpu_memory_mb']:.0f} MiB.")
        story.append(Image(str(filename), width=7.05*inch, height=4.23*inch))
        if result.get('wandb_url'):
            paragraph(f'<link href="{html.escape(result["wandb_url"], quote=True)}">W&amp;B run: {result["wandb_id"]}</link>', 'SmallNote')
        else:
            paragraph(f"W&amp;B offline run: {result['wandb_id']} (업로드 대기)", 'SmallNote')
        paragraph(html.escape(result['configuration']['id']), 'SmallNote')
        markdown(f"## E{index:02}: {result['configuration']['id']}\n\nW&B: {result.get('wandb_url') or 'offline / 업로드 대기'}")
        code(f'print((BASE / {folder.name!r} / "train.log").read_text())', (folder / 'train.log').read_text(errors='replace'))
        count += 1
        notebook.append({'cell_type': 'code', 'metadata': {}, 'execution_count': count,
                         'source': f'from IPython.display import Image, display\ndisplay(Image(filename=str(BASE / "figures" / {filename.name!r})))',
                         'outputs': [{'output_type': 'display_data', 'metadata': {},
                                      'data': {'image/png': base64.b64encode(filename.read_bytes()).decode()}}]})
    markdown('## 실행 코드\n\n기존 학습 코드와 원본 Trainer를 사용했으며, 별도 실행 스크립트가 GPU 픽셀 캐시와 로그 수집을 담당했다.')
    for source_path in [ROOT / '_03_homeworks/homework_3/b_fashion_mnist_train.py',
                        ROOT / 'scripts/fashion_mnist_experiment.py', ROOT / 'scripts/fashion_mnist_study.py'] + (
                        [ROOT / '_03_homeworks/homework_3/c_fashion_mnist_test.py'] if evaluation else []):
        code(f'print(Path({str(source_path)!r}).read_text())', source_path.read_text())
    ipynb = ROOT / '_03_homeworks/homework_3/fashion_mnist_model_selection_report.ipynb'
    for index, cell in enumerate(notebook):
        cell['id'] = f'cell-{index:04}'
    ipynb.write_text(json.dumps({'cells': notebook, 'metadata': {'kernelspec': {'display_name': 'Python 3',
                     'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}},
                     'nbformat': 4, 'nbformat_minor': 5}, ensure_ascii=False, indent=1))
    pdf = OUTPUT / 'fashion_mnist_model_selection_report.pdf'
    temporary_pdf = OUTPUT / 'fashion_mnist_model_selection_report.tmp.pdf'
    def footer(canvas, document):
        canvas.setFont('Helvetica', 8)
        canvas.drawString(42, 24, date)
        canvas.drawRightString(A4[0]-42, 24, str(document.page))
    SimpleDocTemplate(str(temporary_pdf), pagesize=A4, rightMargin=42, leftMargin=42,
                      topMargin=42, bottomMargin=42, title=title).build(story, onFirstPage=footer, onLaterPages=footer)
    temporary_pdf.replace(pdf)
    # Check every regenerated artifact, including the final unattended update.
    from pypdf import PdfReader
    document = PdfReader(str(pdf))
    assert len(document.pages) >= 2
    assert all(page.extract_text().strip() for page in document.pages)
    assert len(json.loads(ipynb.read_text())['cells']) == len(notebook)
    import subprocess
    subprocess.run(['/usr/bin/pdftoppm', '-f', '1', '-l', '1', '-scale-to', '1400',
                    '-singlefile', '-png', str(pdf), str(FIGURES / 'report_first_page')], check=True)
    write_validation = {'pages': len(document.pages), 'completed_runs': len(results),
                        'notebook_cells': len(notebook), 'verified_at': date,
                        'study_complete': final is not None, 'test_evaluated': evaluation is not None}
    (BASE / 'report_validation.json').write_text(json.dumps(write_validation, indent=2))
    print(json.dumps({'pdf': str(pdf), 'notebook': str(ipynb), 'completed_runs': len(results),
                      'study_status': status.get('status')}, ensure_ascii=False))


if __name__ == '__main__':
    build()
