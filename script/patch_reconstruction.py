"""Only-real center-blind reconstruction with optional calibrated NN score fusion."""
import argparse
from collections import defaultdict
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm
import yaml

from models.patch_reconstruction import PatchReconstruction
from script.bank_data import image_records, read_json, sha256, write_json
from script.bank_retrieval import aggregate_patch_score
from script.patch_explanation import (evidence_maps, explain, fit_patch_normalization,
                                      nearest_lookup, patch_fusion_score)
from script.real_patch_bank import _load_observation, _split_real_rows, _source_spec, ensure_cache
from script.retrieval_io import experiment_lock, metrics
from script.reconstruction_heatmap import save_evidence_heatmap


class Features(Dataset):
    def __init__(self, rows, bank, minimum):
        self.rows, self.bank, self.minimum = rows, bank, minimum

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        x, mask = _load_observation(self.rows[index], self.bank, self.bank / 'cache', self.minimum)
        return torch.from_numpy(x), torch.from_numpy(mask), index


def loader(rows, bank, args, shuffle=False):
    return DataLoader(Features(rows, bank, args.foreground_minimum), batch_size=args.batch_size,
                      shuffle=shuffle, num_workers=args.workers)


def epoch(model, batches, device, optimizer=None):
    model.train(optimizer is not None)
    total = count = 0
    for x, mask, _ in tqdm(batches, desc='train' if optimizer else 'validation', leave=False):
        with torch.set_grad_enabled(optimizer is not None):
            _, error, valid = model(x.to(device), mask.to(device))
            n = int(valid.sum())
            if not n:
                continue
            loss = error[valid].mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite reconstruction loss')
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
        total += float(loss.detach()) * n
        count += n
    if not count:
        raise ValueError('No foreground patches with valid neighbors')
    return total / count


def normalized_fusion(calibration, evaluation, weight, quantile):
    """Fit scales on held-out real ONLY, keep scores unclipped for ranking."""
    if not calibration or any(row['label'] != 0 for row in calibration):
        raise ValueError('Calibration must be real only')
    names = ['reconstruction'] + (['nearest'] if weight > 0 else [])
    stats = {}
    for name in names:
        values = np.array([r[name] for r in calibration])
        center = float(np.median(values))
        scale = float(np.quantile(values, quantile) - center)
        if not np.isfinite(scale) or scale <= 1e-8:
            raise ValueError(f'Degenerate calibration scale: {name}')
        stats[name] = {'center': center, 'scale': scale}
    for row in calibration + evaluation:
        norm = {key: (row[key] - v['center']) / v['scale'] for key, v in stats.items()}
        row['score'] = ((1 - weight) * norm['reconstruction'] + weight * norm.get('nearest', 0.))
    threshold = float(np.quantile([r['score'] for r in calibration], quantile))
    return stats, threshold


@torch.no_grad()
def score(model, rows, bank, args):
    model.eval()
    result = []
    for x, mask, indices in tqdm(loader(rows, bank, args), desc='reconstruction scoring'):
        _, errors, valid = model(x.to(args.device), mask.to(args.device))
        for error, keep, index in zip(errors.cpu().numpy(), valid.cpu().numpy(), indices):
            if not keep.any():
                raise ValueError('Image has no reconstructable foreground patches')
            ids = np.flatnonzero(keep)
            value = aggregate_patch_score(error.reshape(-1)[ids], ids, keep.shape,
                                          args.top_fraction, args.boundary_weight)
            row = rows[int(index)]
            result.append({key: row[key] for key in ('image_path', 'sample_id', 'video_id', 'group_id', 'label')})
            result[-1].update(reconstruction=value, valid_patches=int(keep.sum()),
                              patch_errors=error[keep].tolist(), patch_ids=ids.tolist(), grid=list(keep.shape))
    return result


def attach_nearest(scored, path):
    baseline = read_json(path)
    lookup = nearest_lookup(baseline, scored)
    for row in scored:
        other = lookup[row['image_path']]
        if not np.isfinite(other['score']):
            raise ValueError('Invalid NN score')
        row['nearest'] = float(other['score'])
    return lookup


def aggregate_videos(rows, fields):
    """Average frame scores per video while preserving all requested components."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['video_id']].append(row)
    result = []
    for video_id, frames in grouped.items():
        labels = {row['label'] for row in frames}
        if len(labels) != 1:
            raise ValueError(f'Conflicting labels within video: {video_id}')
        item = {'video_id': video_id, 'label': labels.pop(), 'frames': len(frames)}
        for field in fields:
            values = [row[field] for row in frames]
            if not all(np.isfinite(values)):
                raise ValueError(f'Non-finite {field} within video: {video_id}')
            item[field] = float(np.mean(values))
        result.append(item)
    return result


def balanced_heatmap_rows(rows, count):
    """Select fixed-order, one-frame-per-video examples without score cherry-picking."""
    selected = []
    quotas = {0: (count + 1) // 2, 1: count // 2}
    seen = set()
    for row in rows:
        label = row['label']
        if quotas.get(label, 0) and row['video_id'] not in seen:
            selected.append(row)
            seen.add(row['video_id'])
            quotas[label] -= 1
    return selected


def parse_args(argv=None):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / 'utils/config.yaml').read_text())['patch_reconstruction']
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exper', required=True, dest='experiment')
    p.add_argument('--run-name', default='local_transformer_v2')
    p.add_argument('--stage', choices=['train', 'evaluate', 'all'], default='train')
    p.add_argument('--device', default='cuda:0')
    for name in ('epochs', 'batch-size', 'workers', 'patience', 'max-train-images', 'max-validation-images', 'heatmap-count'):
        p.add_argument('--' + name, type=int)
    p.add_argument('--nn-weight', type=float)
    p.set_defaults(**config)
    a = p.parse_args(argv)
    if any(getattr(a, k) < 1 for k in ('epochs', 'batch_size', 'patience', 'hidden_dim', 'radius', 'heads', 'layers')) or a.workers < 0:
        p.error('Invalid integer configuration')
    if a.hidden_dim % a.heads:
        p.error('hidden_dim must be divisible by heads')
    if any(getattr(a, k) < 0 for k in ('max_train_images', 'max_validation_images', 'heatmap_count')):
        p.error('Image limits must be >= 0; 0 means all')
    if not (0 <= a.nn_weight < 1 and 0 < a.threshold_quantile < 1 and 0 < a.validation_fraction < 1
            and 0 < a.top_fraction <= 1 and 0 < a.foreground_minimum <= 1 and 0 <= a.boundary_weight <= 1
            and a.learning_rate > 0):
        p.error('Invalid ratio or learning rate')
    if any(Path(s).name != s or s in ('.', '..') for s in (a.experiment, a.run_name)):
        p.error('Experiment and run names must be single directory names')
    for key in ('bank_dir', 'source_results_dir', 'results_dir'):
        setattr(a, key, str((root / getattr(a, key)).resolve()))
    a.devices = [a.device]
    return a


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    bank = Path(args.bank_dir) / args.experiment
    output = Path(args.results_dir) / args.experiment / args.run_name
    plan = read_json(bank / 'splits.json')
    if _source_spec(bank)['encoder_tuning']['enabled']:
        raise ValueError('This branch requires a frozen Stage 1 encoder')
    groups = [{v['group_id'] for role in roles for v in plan['groups'].get(role, [])}
              for roles in [('bank', 'train_fake'), ('calibration',), ('evaluation',)]]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Source families overlap across partitions')
    fingerprint = {name: sha256(bank / name) for name in ('splits.json', 'bank_config.json')}
    with experiment_lock(output):
        checkpoint = output / 'model.safetensors'
        if args.stage in ('train', 'all'):
            if checkpoint.exists() or (output / 'training.json').exists():
                raise FileExistsError('Use a new --run-name to preserve existing training')
            fit, val = _split_real_rows(image_records(plan['groups']['bank'], 'bank'), args.validation_fraction, args.seed)
            rng = random.Random(args.seed)
            rng.shuffle(fit)
            rng.shuffle(val)
            fit = fit[:args.max_train_images or None]
            val = val[:args.max_validation_images or None]
            first, _ = _load_observation(fit[0], bank, bank / 'cache', args.foreground_minimum)
            architecture = dict(input_dim=first.shape[-1], hidden_dim=args.hidden_dim, radius=args.radius,
                                heads=args.heads, layers=args.layers)
            model = PatchReconstruction(**architecture).to(args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
            history, best, stale = [], float('inf'), 0
            for i in range(args.epochs):
                train_loss = epoch(model, loader(fit, bank, args, True), args.device, optimizer)
                val_loss = epoch(model, loader(val, bank, args), args.device)
                history.append(dict(epoch=i + 1, train_loss=train_loss, validation_loss=val_loss))
                print(history[-1], flush=True)
                if val_loss < best:
                    best, stale = val_loss, 0
                    temporary = checkpoint.with_suffix('.tmp')
                    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(temporary))
                    temporary.replace(checkpoint)
                else:
                    stale += 1
                if stale >= args.patience:
                    break
            metadata = dict(architecture=architecture, fingerprint=fingerprint, config=vars(args), history=history,
                            fit_images=len(fit), validation_images=len(val), fake_training_images=0,
                            fit_families=sorted({r['group_id'] for r in fit}),
                            validation_families=sorted({r['group_id'] for r in val}),
                            checkpoint_sha256=sha256(checkpoint))
            write_json(output / 'training.json', metadata)
        if args.stage in ('evaluate', 'all'):
            meta = read_json(output / 'training.json')
            if meta['fingerprint'] != fingerprint or meta['checkpoint_sha256'] != sha256(checkpoint):
                raise ValueError('Source or checkpoint changed')
            for key in ('foreground_minimum', 'top_fraction', 'boundary_weight', 'threshold_quantile', 'nn_weight'):
                if getattr(args, key) != meta['config'][key]:
                    raise ValueError(f'Evaluation setting differs from training declaration: {key}')
            calibration_rows = image_records(plan['groups']['calibration'], 'calibration')
            if not calibration_rows or any(r['label'] for r in calibration_rows):
                raise ValueError('Calibration must be real only')
            ensure_cache(plan, args, bank, bank / 'cache', ('calibration', 'evaluation'))
            model = PatchReconstruction(**meta['architecture']).to(args.device)
            model.load_state_dict(load_file(str(checkpoint), device=args.device))
            calibration = score(model, calibration_rows, bank, args)
            evaluation = score(model, image_records(plan['groups']['evaluation'], 'evaluation'), bank, args)
            nearest_hashes = {}
            nearest_lookups = {}
            if args.nn_weight > 0:
                source_stage1 = Path(args.source_results_dir) / args.experiment / 'stage1'
                if (read_json(source_stage1 / 'config.json') != _source_spec(bank)
                        or read_json(source_stage1 / 'splits.json') != plan):
                    raise ValueError('NN source configuration or split differs from reconstruction')
                for role, scored in [('calibration', calibration), ('evaluation', evaluation)]:
                    path = Path(args.source_results_dir) / args.experiment / 'stage2' / f'{role}_scores.json'
                    nearest_lookups[role] = attach_nearest(scored, path)
                    nearest_hashes[role] = sha256(path)
            stats, threshold = normalized_fusion(calibration, evaluation, args.nn_weight, args.threshold_quantile)
            report = dict(image=metrics(evaluation, threshold), normalization=stats, nn_weight=args.nn_weight,
                          fingerprint=fingerprint, checkpoint_sha256=meta['checkpoint_sha256'],
                          nearest_scores_sha256=nearest_hashes, fake_training_images=0, components={})
            for name in stats:
                cut = float(np.quantile([r[name] for r in calibration], args.threshold_quantile))
                report['components'][name] = metrics([dict(label=r['label'], score=r[name]) for r in evaluation], cut)
            fields = ['score', *stats]
            calibration_videos = aggregate_videos(calibration, fields)
            evaluation_videos = aggregate_videos(evaluation, fields)
            video_threshold = float(np.quantile([r['score'] for r in calibration_videos], args.threshold_quantile))
            report['video_mean'] = metrics(evaluation_videos, video_threshold)
            report['video_components'] = {}
            for name in stats:
                cut = float(np.quantile([r[name] for r in calibration_videos], args.threshold_quantile))
                report['video_components'][name] = metrics(
                    [dict(label=r['label'], score=r[name]) for r in evaluation_videos], cut)
            for role, scored in [('calibration', calibration), ('evaluation', evaluation)]:
                for row in scored:
                    row['prediction'] = 'anomaly' if row['score'] > threshold else 'normal'
            for row in evaluation_videos:
                row['prediction'] = 'anomaly' if row['score'] > video_threshold else 'normal'
            write_json(output / 'stage2/video_scores.json', evaluation_videos)
            patch_normalization = fit_patch_normalization(
                calibration, nearest_lookups.get('calibration'), args.threshold_quantile,
                args.boundary_weight)
            for row in calibration:
                nearest = (nearest_lookups['calibration'][row['image_path']]
                           if 'calibration' in nearest_lookups else None)
                maps = evidence_maps(row, nearest, patch_normalization,
                                     args.nn_weight, args.boundary_weight)
                row['patch_fusion'] = patch_fusion_score(maps, args.top_fraction)
            patch_fusion_threshold = float(np.quantile(
                [row['patch_fusion'] for row in calibration], args.threshold_quantile))
            selected = balanced_heatmap_rows(evaluation, args.heatmap_count)
            selected_paths = {row['image_path'] for row in selected}
            explanations = []
            heatmaps = {}
            for row in evaluation:
                nearest = (nearest_lookups['evaluation'][row['image_path']]
                           if 'evaluation' in nearest_lookups else None)
                maps = evidence_maps(row, nearest, patch_normalization,
                                     args.nn_weight, args.boundary_weight)
                row['patch_fusion'] = patch_fusion_score(maps, args.top_fraction)
                row['patch_fusion_prediction'] = ('anomaly' if row['patch_fusion'] > patch_fusion_threshold
                                                  else 'normal')
                explanation_row = dict(row, score=row['patch_fusion'])
                explanations.append(explain(explanation_row, maps, patch_fusion_threshold,
                                            args.top_fraction))
                if row['image_path'] in selected_paths:
                    heatmaps[row['image_path']] = maps
            explanation_path = output / 'stage2/explanations.json'
            write_json(explanation_path, explanations)
            write_json(output / 'stage2/calibration_scores.json', calibration)
            write_json(output / 'stage2/evaluation_scores.json', evaluation)
            report['patch_fusion'] = metrics(
                [dict(label=row['label'], score=row['patch_fusion']) for row in evaluation],
                patch_fusion_threshold)
            report['patch_explanations'] = {
                'path': str(explanation_path.resolve()),
                'count': len(explanations),
                'patch_normalization': patch_normalization,
                'language': 'zh-TW deterministic template grounded in calibrated patch evidence',
                'prediction_method': 'mean of highest fused patch evidence; same evidence used by heatmap',
                'limitation': 'Feature evidence, not pixel-level forgery ground truth.',
            }
            write_json(output / 'stage2/metrics.json', report)
            for row in selected:
                save_evidence_heatmap(row, heatmaps[row['image_path']],
                                      output / 'stage2/heatmaps' / f"{row['sample_id']:08d}.png")
            print(report['image'], flush=True)


if __name__ == '__main__':
    main()
