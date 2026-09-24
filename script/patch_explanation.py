"""Calibrated patch evidence, spatial grouping, and grounded Chinese explanations."""
import math

import numpy as np

from script.bank_retrieval import patch_evidence


def nearest_lookup(rows, expected):
    lookup = {row['image_path']: row for row in rows}
    if len(lookup) != len(rows) or set(lookup) != {row['image_path'] for row in expected}:
        raise ValueError('NN and reconstruction image sets differ')
    for row in expected:
        other = lookup[row['image_path']]
        if any(row[key] != other[key] for key in ('label', 'group_id', 'video_id')):
            raise ValueError('NN split metadata mismatch')
    return lookup


def _component_vectors(row, nearest, boundary_weight):
    grid = tuple(row['grid'])
    ids = np.asarray(row['patch_ids'], dtype=np.int64)
    reconstruction = np.asarray(row['patch_errors'], dtype=np.float32)
    if (len(ids) != len(reconstruction) or ids.ndim != 1 or len(set(ids.tolist())) != len(ids)
            or np.any(ids < 0) or np.any(ids >= np.prod(grid))
            or not np.isfinite(reconstruction).all()):
        raise ValueError(f"Invalid reconstruction patch evidence: {row['image_path']}")
    result = {'reconstruction': patch_evidence(reconstruction, ids, grid, boundary_weight)}
    if nearest is not None:
        with np.load(nearest['patch_matches'], allow_pickle=False) as archive:
            distances = archive['distances']
        if tuple(distances.shape) != grid:
            raise ValueError(f"NN/reconstruction grid mismatch: {row['image_path']}")
        values = distances.reshape(-1)[ids]
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Missing NN evidence for reconstructable patches: {row['image_path']}")
        result['nearest'] = patch_evidence(values, ids, grid, boundary_weight)
    return ids, grid, result


def fit_patch_normalization(rows, lookup, quantile, boundary_weight):
    """Fit component medians/scales using calibration-real patch evidence only."""
    if not rows or any(row['label'] != 0 for row in rows):
        raise ValueError('Patch evidence calibration must contain real images only')
    chunks = {'reconstruction': [], **({'nearest': []} if lookup is not None else {})}
    for row in rows:
        nearest = lookup[row['image_path']] if lookup is not None else None
        _, _, components = _component_vectors(row, nearest, boundary_weight)
        for name, values in components.items():
            chunks[name].append(values)
    statistics = {}
    for name, values in chunks.items():
        values = np.concatenate(values)
        center = float(np.median(values))
        scale = float(np.quantile(values, quantile) - center)
        if not np.isfinite(center) or not np.isfinite(scale) or scale <= 1e-8:
            raise ValueError(f'Degenerate patch calibration scale: {name}')
        statistics[name] = {'center': center, 'scale': scale, 'patches': len(values)}
    return statistics


def evidence_maps(row, nearest, statistics, nearest_weight, boundary_weight):
    ids, grid, components = _component_vectors(row, nearest, boundary_weight)
    normalized = {}
    for name, values in components.items():
        calibration = statistics[name]
        normalized[name] = np.maximum((values - calibration['center']) / calibration['scale'], 0.)
    weight = nearest_weight if 'nearest' in normalized else 0.
    fused = (1 - weight) * normalized['reconstruction'] + weight * normalized.get('nearest', 0.)
    maps = {}
    for name, values in {**normalized, 'fusion': fused}.items():
        image = np.full(grid, np.nan, dtype=np.float32)
        image.reshape(-1)[ids] = values
        maps[name] = image
    return maps


def patch_fusion_score(maps, fraction):
    values = maps['fusion'][np.isfinite(maps['fusion'])]
    if not len(values):
        raise ValueError('No valid fused patch evidence')
    count = max(1, math.ceil(len(values) * fraction))
    return float(np.partition(values, -count)[-count:].mean())


def _location_name(y, x, height, width):
    vertical = ('上方', '中央', '下方')[min(2, int(3 * y / max(height, 1)))]
    horizontal = ('左側', '中央', '右側')[min(2, int(3 * x / max(width, 1)))]
    if vertical == horizontal == '中央':
        return '臉部中央'
    return f'臉部{vertical}{horizontal}'


def _regions(maps, fraction, limit=3):
    fusion = maps['fusion']
    valid = np.flatnonzero(np.isfinite(fusion.reshape(-1)))
    count = max(1, math.ceil(len(valid) * fraction))
    selected = set(valid[np.argpartition(fusion.reshape(-1)[valid], -count)[-count:]].tolist())
    height, width = fusion.shape
    groups = []
    while selected:
        pending = [selected.pop()]
        group = []
        while pending:
            index = pending.pop()
            group.append(index)
            y, x = divmod(index, width)
            for neighbor in (index - width, index + width, index - 1, index + 1):
                ny, nx = divmod(neighbor, width)
                if (neighbor in selected and 0 <= ny < height and 0 <= nx < width
                        and abs(ny - y) + abs(nx - x) == 1):
                    selected.remove(neighbor)
                    pending.append(neighbor)
        groups.append(group)
    result = []
    for group in groups:
        ys, xs = zip(*(divmod(index, width) for index in group))
        reconstruction = float(np.nanmean([maps['reconstruction'][y, x] for y, x in zip(ys, xs)]))
        nearest = (float(np.nanmean([maps['nearest'][y, x] for y, x in zip(ys, xs)]))
                   if 'nearest' in maps else None)
        if nearest is None:
            dominant = 'reconstruction'
        elif nearest > reconstruction * 1.2:
            dominant = 'nearest'
        elif reconstruction > nearest * 1.2:
            dominant = 'reconstruction'
        else:
            dominant = 'both'
        result.append({
            'location': _location_name(float(np.mean(ys)) + .5, float(np.mean(xs)) + .5, height, width),
            'patches': len(group),
            'box': [min(ys), min(xs), max(ys) + 1, max(xs) + 1],
            'mean_evidence': float(np.nanmean([fusion[y, x] for y, x in zip(ys, xs)])),
            'peak_evidence': float(np.nanmax([fusion[y, x] for y, x in zip(ys, xs)])),
            'dominant_signal': dominant,
        })
    return sorted(result, key=lambda item: item['peak_evidence'], reverse=True)[:limit]


def explain(row, maps, threshold, top_fraction):
    regions = _regions(maps, top_fraction)
    abnormal = row['score'] > threshold
    lead = ('影像分數超過只用真實校準資料設定的異常門檻。' if abnormal else
            '影像分數未超過只用真實校準資料設定的異常門檻。')
    descriptions = []
    signal_text = {
        'nearest': '主要與正常特徵庫的局部差異有關',
        'reconstruction': '主要來自相鄰 patch 無法準確預測該處特徵',
        'both': '正常庫距離與鄰域重建誤差都偏高',
    }
    for region in regions[:2]:
        descriptions.append(f"{region['location']}（{region['patches']} 個 patch，{signal_text[region['dominant_signal']]}）")
    focus = '較高的模型證據集中在' + '、'.join(descriptions) + '。' if descriptions else '沒有可用的局部證據。'
    return {
        'image_path': row['image_path'], 'video_id': row['video_id'], 'label': row['label'],
        'score': row['score'], 'threshold': threshold,
        'prediction': 'anomaly' if abnormal else 'normal',
        'summary_zh': lead + focus + '這些位置代表特徵偏差，不是像素級偽造標註。',
        'regions': regions,
    }
