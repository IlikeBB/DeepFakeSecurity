import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from script.bank_data import read_json, write_json
from script.qwen_explanation import (compose_summary, evidence_prompt, grounding_errors,
                                     run, select_rows)


def row(index, label):
    return {
        'image_path': f'image-{index}.jpg',
        'video_id': f'video-{index}',
        'label': label,
        'score': 1.2,
        'threshold': 1.,
        'prediction': 'anomaly',
        'summary_zh': '原始可追溯說明。',
        'regions': [{
            'location': '臉部上方左側', 'patches': 3, 'box': [0, 0, 1, 1],
            'mean_evidence': 1.1, 'peak_evidence': 1.3,
            'dominant_signal': 'both',
        }],
    }


class QwenExplanationTests(unittest.TestCase):
    def test_balanced_fixed_selection_and_prompt(self):
        rows = [row(i, i % 2) for i in range(8)]
        selected = select_rows(rows, 4)
        self.assertEqual([item['label'] for item in selected], [0, 1, 0, 1])
        self.assertEqual(select_rows(rows, 0), rows)
        prompt = evidence_prompt(rows[0])
        self.assertIn('正常特徵庫', prompt[1]['content'][0]['text'])
        self.assertIn('相鄰 patch', prompt[1]['content'][0]['text'])

    def test_grounding_validation_and_fixed_wrapper(self):
        source = row(0, 0)
        good = '較高證據集中於臉部上方左側，正常特徵庫與相鄰 patch 的證據都偏高。'
        self.assertEqual(grounding_errors(good, source), [])
        self.assertIn('unsupported_anatomy', grounding_errors(good + '可能位於眼睛。', source))
        summary = compose_summary(source, good)
        self.assertTrue(summary.startswith('影像分數超過'))
        self.assertTrue(summary.endswith('不是像素級偽造標註。'))

    def test_run_preserves_source_and_falls_back_on_ungrounded_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model'
            model.mkdir()
            (model / 'config.json').write_text('{}')
            (model / 'model.safetensors.index.json').write_text('{}')
            source_path = root / 'explanations.json'
            output = root / 'qwen.json'
            write_json(source_path, [row(0, 0)])
            args = SimpleNamespace(input=source_path, output=output, model_path=model,
                                   device='cpu', max_items=1, max_new_tokens=32,
                                   overwrite=False)
            result = run(args, generator=lambda _messages, _tokens: '眼睛顯示偽造影像。')
            item = result['items'][0]
            self.assertEqual(item['grounding_status'], 'fallback')
            self.assertEqual(item['source_summary_zh'], '原始可追溯說明。')
            self.assertIn('臉部上方左側', item['qwen_summary_zh'])
            self.assertEqual(read_json(output), result)


if __name__ == '__main__':
    unittest.main()
