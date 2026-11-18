"""Regression checks for PDF/notebook export using recorded evaluation data."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class ReportExportTests(unittest.TestCase):
    def render(self, wandb_url):
        from scripts import fashion_mnist_report as report
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            homework = root / '_03_homeworks/homework_3'
            experiments = homework / 'experiments'
            evaluation = homework / 'evaluation'
            experiments.mkdir(parents=True)
            evaluation.mkdir()
            for name in ('statistics.json', 'final_selection.json', 'study_status.json'):
                shutil.copyfile(ROOT / '_03_homeworks/homework_3/experiments' / name, experiments / name)
            source_evaluation = ROOT / '_03_homeworks/homework_3/evaluation'
            metrics = json.loads((source_evaluation / 'metrics.json').read_text())
            metrics['wandb_url'] = wandb_url
            (evaluation / 'metrics.json').write_text(json.dumps(metrics))
            for path in source_evaluation.glob('*.png'):
                shutil.copyfile(path, evaluation / path.name)
            shutil.copyfile(source_evaluation / 'evaluation.log', evaluation / 'evaluation.log')
            for relative in ('_03_homeworks/homework_3/b_fashion_mnist_train.py',
                             '_03_homeworks/homework_3/c_fashion_mnist_test.py',
                             'scripts/fashion_mnist_experiment.py', 'scripts/fashion_mnist_study.py'):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, target)
            with patch.multiple(report, ROOT=root, BASE=experiments, OUTPUT=root / 'output/pdf',
                                FIGURES=experiments / 'figures'), redirect_stdout(io.StringIO()):
                report.build()
            self.assertTrue((root / 'output/pdf/fashion_mnist_model_selection_report.pdf').is_file())
            return json.loads((homework / 'fashion_mnist_model_selection_report.ipynb').read_text())

    def test_report_generation_works_when_wandb_is_disabled(self):
        notebook = self.render(None)
        self.assertGreater(len(notebook['cells']), 0)

    def test_notebook_includes_result_interpretation_as_markdown(self):
        notebook = self.render('https://wandb.ai/example/project/runs/evaluation')
        markdown = '\n'.join(cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'markdown')
        # Numeric findings must appear in readable interpretation, rather than
        # only in a raw JSON/console output or an image.
        for finding in ('93.29%', '75.60%', '0.00189877', '0.9'):
            self.assertIn(finding, markdown)
        self.assertIn('ablation', markdown.lower())


if __name__ == '__main__':
    unittest.main()
