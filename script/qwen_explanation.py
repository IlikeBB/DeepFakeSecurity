"""Rewrite calibrated patch evidence with a local Qwen3-VL model.

The language model cannot change scores, thresholds, predictions, or regions.  Its
output is stored beside the deterministic explanation so every sentence remains
auditable and a failed grounding check falls back to the original template.
"""
import argparse
import re
from pathlib import Path

import torch
import yaml

from script.bank_data import read_json, sha256, write_json


SIGNAL_TEXT = {
    'nearest': '與正常特徵庫的局部差異偏高',
    'reconstruction': '相鄰 patch 對該處特徵的預測誤差偏高',
    'both': '正常特徵庫距離與相鄰 patch 預測誤差都偏高',
}
ANATOMY_TERMS = ('眼睛', '眼球', '眉毛', '鼻子', '嘴巴', '嘴唇', '牙齒', '耳朵', '皮膚')
LOCATION_PATTERN = re.compile(r'臉部(?:上方|中央|下方)(?:左側|中央|右側)?')


def select_rows(rows, count):
    """Fixed-order, balanced, one-frame-per-video selection; zero means all."""
    if count == 0:
        return list(rows)
    quotas = {0: (count + 1) // 2, 1: count // 2}
    selected, seen = [], set()
    for row in rows:
        label = row.get('label')
        video = row.get('video_id')
        if quotas.get(label, 0) and video not in seen:
            selected.append(row)
            seen.add(video)
            quotas[label] -= 1
    return selected


def validate_source(row):
    required = ('image_path', 'video_id', 'label', 'score', 'threshold', 'prediction',
                'summary_zh', 'regions')
    if any(key not in row for key in required):
        raise ValueError('Explanation row is missing required fields')
    expected = 'anomaly' if row['score'] > row['threshold'] else 'normal'
    if row['prediction'] != expected or row['prediction'] not in ('normal', 'anomaly'):
        raise ValueError(f"Prediction/score mismatch: {row['image_path']}")
    if row['label'] not in (0, 1) or not isinstance(row['regions'], list):
        raise ValueError(f"Invalid explanation row: {row['image_path']}")
    for region in row['regions'][:2]:
        if (region.get('dominant_signal') not in SIGNAL_TEXT
                or not isinstance(region.get('location'), str)
                or not isinstance(region.get('patches'), int)):
            raise ValueError(f"Invalid explanation region: {row['image_path']}")


def evidence_prompt(row):
    """Ask for only the evidence-focused middle sentence, never a new verdict."""
    regions = row['regions'][:2]
    facts = '\n'.join(
        f"- {region['location']}：{region['patches']} 個 patch；{SIGNAL_TEXT[region['dominant_signal']]}"
        for region in regions
    ) or '- 沒有可用的局部證據'
    return [
        {
            'role': 'system',
            'content': [{'type': 'text', 'text': (
                '你是研究報告的文字編輯。只能改寫使用者提供的特徵證據，不可重新判定正常或異常，'
                '不可加入真假、偽造機率、成因、人物身分、眼鼻口等解剖名稱，也不可加入未提供的數字或位置。'
                '請使用繁體中文輸出單一句子，不要標題、條列、Markdown 或額外說明。'
            )}],
        },
        {
            'role': 'user',
            'content': [{'type': 'text', 'text': (
                '請把以下證據合併成一句自然、精確的描述。位置名稱與「正常特徵庫」／「相鄰 patch」'
                '等證據來源必須原樣保留；不要寫分數、門檻、預測結果或像素級結論。\n' + facts
            )}],
        },
    ]


def deterministic_focus(row):
    regions = row['regions'][:2]
    if not regions:
        return '沒有可用的局部證據。'
    details = [f"{r['location']}（{r['patches']} 個 patch，{SIGNAL_TEXT[r['dominant_signal']]}）"
               for r in regions]
    return '較高的模型證據集中在' + '、'.join(details) + '。'


def grounding_errors(text, row):
    text = text.strip()
    errors = []
    if not text:
        errors.append('empty')
    if len(text) > 300:
        errors.append('too_long')
    if re.search(r'[0-9０-９]', text):
        errors.append('added_number')
    if any(term in text for term in ANATOMY_TERMS):
        errors.append('unsupported_anatomy')
    if any(term in text for term in ('deepfake', 'DeepFake', '真實影像', '偽造影像', '偽造機率')):
        errors.append('unsupported_verdict')
    regions = row['regions'][:2]
    allowed_locations = {region['location'] for region in regions}
    mentioned_locations = set(LOCATION_PATTERN.findall(text))
    if mentioned_locations - allowed_locations:
        errors.append('added_location')
    if any(location not in text for location in allowed_locations):
        errors.append('missing_location')
    signals = {region['dominant_signal'] for region in regions}
    if ('nearest' in signals or 'both' in signals) and '正常特徵庫' not in text:
        errors.append('missing_nearest_evidence')
    if ('reconstruction' in signals or 'both' in signals) and '相鄰 patch' not in text:
        errors.append('missing_reconstruction_evidence')
    return sorted(set(errors))


def compose_summary(row, focus):
    lead = ('影像分數超過只用真實校準資料設定的異常門檻。' if row['prediction'] == 'anomaly'
            else '影像分數未超過只用真實校準資料設定的異常門檻。')
    focus = focus.strip()
    if focus and focus[-1] not in '。！？':
        focus += '。'
    return lead + focus + '這些位置代表特徵偏差，不是像素級偽造標註。'


def clean_output(text):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = re.sub(r'^```[^\n]*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
    return re.sub(r'\s+', ' ', text).strip()


class QwenGenerator:
    def __init__(self, model_path, device):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        mapping = 'auto' if device == 'auto' else {'': device}
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path), dtype='auto', device_map=mapping, local_files_only=True,
            low_cpu_mem_usage=True).eval()
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)

    def __call__(self, messages, max_new_tokens):
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors='pt')
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                            do_sample=False)
        trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def run(args, generator=None):
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    rows = read_json(args.input)
    if not isinstance(rows, list) or not rows:
        raise ValueError('Input must be a non-empty explanation list')
    for row in rows:
        validate_source(row)
    selected = select_rows(rows, args.max_items)
    if not selected:
        raise ValueError('Selection contains no explanations')
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f'{args.output} exists; use --overwrite to replace it')
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    generator = generator or QwenGenerator(args.model_path, args.device)
    items = []
    for index, row in enumerate(selected, 1):
        raw = clean_output(generator(evidence_prompt(row), args.max_new_tokens))
        errors = grounding_errors(raw, row)
        accepted = not errors
        focus = raw if accepted else deterministic_focus(row)
        items.append({
            'image_path': row['image_path'],
            'video_id': row['video_id'],
            'label': row['label'],
            'score': row['score'],
            'threshold': row['threshold'],
            'prediction': row['prediction'],
            'regions': row['regions'],
            'source_summary_zh': row['summary_zh'],
            'qwen_summary_zh': compose_summary(row, focus),
            'grounding_status': 'accepted' if accepted else 'fallback',
            'grounding_errors': errors,
            'raw_model_output': raw,
        })
        print(f'[{index}/{len(selected)}] {row["image_path"]}: '
              f'{items[-1]["grounding_status"]}', flush=True)
    config = args.model_path / 'config.json'
    index = args.model_path / 'model.safetensors.index.json'
    result = {
        'schema': 'grounded-qwen-explanation-v1',
        'input_path': str(args.input.resolve()),
        'input_sha256': sha256(args.input),
        'model': {
            'path': str(args.model_path.resolve()),
            'config_sha256': sha256(config),
            'weight_index_sha256': sha256(index),
        },
        'generation': {'do_sample': False, 'max_new_tokens': args.max_new_tokens},
        'selection': {
            'method': 'fixed-order, class-balanced, at most one frame per video',
            'requested': args.max_items,
            'available': len(rows),
            'selected': len(selected),
        },
        'items': items,
    }
    write_json(args.output, result)
    return result


def parse_args(argv=None):
    project = Path(__file__).resolve().parents[1]
    defaults = yaml.safe_load((project / 'utils/config.yaml').read_text())['qwen_explanation']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='Stage 2 explanations.json produced by this project.')
    parser.add_argument('--output', type=Path,
                        help='Default: qwen_explanations.json beside --input.')
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--device')
    parser.add_argument('--max-items', type=int,
                        help='Balanced examples to rewrite; 0 processes every row.')
    parser.add_argument('--max-new-tokens', type=int)
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction)
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args.input = args.input.expanduser().resolve()
    args.output = ((args.output or args.input.with_name('qwen_explanations.json'))
                   .expanduser().resolve())
    args.model_path = args.model_path.expanduser().resolve()
    if args.max_items < 0 or args.max_new_tokens < 1:
        parser.error('max-items must be >= 0 and max-new-tokens must be >= 1')
    if not isinstance(args.device, str) or not args.device:
        parser.error('device must be a non-empty string')
    return args


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
