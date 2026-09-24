"""Celeb-DF official-test preparation and frozen DFDC reconstruction transfer evaluation.

RetinaFace crops must first be generated with script/crop_face.py (official test only).
No Celeb-DF images are used for training, validation, normalization, or thresholds.
"""
import argparse
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import os

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file
from tqdm.auto import tqdm
import yaml

from models.patch_reconstruction import PatchReconstruction
from script.bank_data import read_json, write_json, sha256, save_array, image_records
from script.bank_encoder import load_encoder, encode
from script.bank_export import load_feature
from script.face_parser import SegFaceParser
from script.face_crop import expand_mask, crop_masked_face
from script.segment_face import save_image
from script.patch_reconstruction import score
from script.real_patch_bank import _foreground_mask
from script.reconstruction_heatmap import save_heatmap
from script.retrieval_io import experiment_lock, metrics


def official_test(root):
    result = {}
    for line in (root / 'List_of_testing_videos.txt').read_text().splitlines():
        if not line.strip():
            continue
        real_label, name = line.split(maxsplit=1)
        if real_label not in ('0', '1') or name in result:
            raise ValueError('Invalid or duplicate official test entry')
        path = (root / name).resolve()
        if root.resolve() not in path.parents or not path.is_file():
            raise ValueError(f'Missing/invalid official test video: {name}')
        result[name] = 1 - int(real_label)  # Official 1=real; project 1=fake.
    if set(result.values()) != {0, 1}:
        raise ValueError('External test needs both classes')
    return result


def source_hashes(bank):
    plan = read_json(bank / 'splits.json')
    rows = [r for role, videos in plan['groups'].items() for r in image_records(videos, role)]
    if any(not r.get('content_sha256') for r in rows):
        raise ValueError('Source split needs content hashes for external overlap audit')
    return {r['content_sha256'] for r in rows}


def segmented_name(relative, filename):
    """Match DFDC SegFace's flat source-video-frame naming convention."""
    video = Path(relative).with_suffix('')
    if len(video.parts) != 2 or Path(filename).name != filename:
        raise ValueError(f'Invalid Celeb-DF source path: {relative}/{filename}')
    return f'{video.parts[0]}-{video.parts[1]}-{filename}'


def segmented_destination(args, label, relative, filename):
    root = args.normal_output_dir if label == 0 else args.anomaly_output_dir
    return root / segmented_name(relative, filename)


def prepare(args, project, features=True):
    config = yaml.safe_load((project / 'utils/config.yaml').read_text())['segment_face']
    config['model_dir'] = str((project / config['model_dir']).resolve())
    spec = read_json(args.bank / 'bank_config.json')
    if spec['encoder_tuning']['enabled'] or spec['face_source'] != 'segface':
        raise ValueError('External protocol requires frozen SegFace source features')
    for name, digest in spec['files_sha256'].items():
        if sha256(Path(spec['model_path']) / name) != digest:
            raise ValueError('DINO checkpoint/processor changed')
    expected = official_test(args.data_root)
    protocol = dict(dataset='Celeb-DF', partition='official-test-only',
                    official_list_sha256=sha256(args.data_root / 'List_of_testing_videos.txt'),
                    source_bank_config_sha256=sha256(args.bank / 'bank_config.json'),
                    source_splits_sha256=sha256(args.bank / 'splits.json'),
                    segface_config={k: config[k] for k in ('parser_model', 'parser_revision', 'checkpoint',
                        'face_ids', 'background_ids', 'pixel_confidence', 'dilation_ratio', 'closing_ratio',
                        'min_face_area', 'background', 'jpeg_quality')},
                    segface_sha256=sha256(Path(config['model_dir']) / config['checkpoint']))
    config_path = args.output / 'preparation.json'
    if config_path.exists() and read_json(config_path) != protocol:
        raise ValueError('External preparation configuration changed; use another output directory')
    write_json(config_path, protocol)
    manifests = []
    for relative, label in expected.items():
        path = args.crops / Path(relative).with_suffix('') / 'metadata.json'
        data = read_json(path)
        if (data['split'] != 'test' or data['label'] != label
                or Path(data['source']).resolve() != (args.data_root / relative).resolve()):
            raise ValueError(f'Crop metadata mismatch: {relative}')
        if data['settings'] != {'num_frames': 32, 'threshold': .9, 'margin': .2,
                               'image_size': 224, 'detection_size': 640}:
            raise ValueError('External RetinaFace settings differ from DFDC protocol')
        manifests.append((relative, label, path, data))
    excluded, rows, duplicates = [], [], []
    seen = {}
    source = source_hashes(args.bank)
    parser = None
    for relative, label, manifest, data in tqdm(manifests, desc='External SegFace'):
        kept = 0
        valid_frames = [frame for frame in data['frames'] if frame['status'] == 'ok']
        pending = []
        for frame in valid_frames:
            original = (manifest.parent / frame['file']).resolve()
            if original.parent != manifest.parent.resolve():
                raise ValueError('Crop path escapes manifest directory')
            destination = segmented_destination(args, label, relative, original.name)
            if not destination.exists():
                pending.append((frame, original, destination))
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            images = []
            for _, original, _ in batch:
                with Image.open(original) as image:
                    images.append(image.convert('RGB'))
            if batch:
                if parser is None:
                    parser = SegFaceParser(config, args.device)
                masks = parser(images)
            for (frame, original, destination), image, raw in zip(batch, images, masks):
                mask = expand_mask(raw, config['dilation_ratio'], config['closing_ratio'], config['min_face_area'])
                if mask is None:
                    excluded.append(dict(video=relative, frame=frame['frame_index'], reason='no_segface'))
                    continue
                pixels, _ = crop_masked_face(np.asarray(image), mask, config['background'])
                destination.parent.mkdir(parents=True, exist_ok=True)
                save_image(destination, Image.fromarray(pixels), config['jpeg_quality'])
        for frame in valid_frames:
            original = (manifest.parent / frame['file']).resolve()
            destination = segmented_destination(args, label, relative, original.name)
            if not destination.exists():
                continue
            digest = sha256(destination)
            if digest in source:
                raise ValueError(f'Exact external/source image overlap: {destination}')
            if digest in seen:
                duplicates.append(dict(image=str(destination), duplicate_of=seen[digest]))
            seen.setdefault(digest, str(destination))
            stat = destination.stat()
            video = 'Celeb-DF/' + str(Path(relative).with_suffix(''))
            # This entire dataset is test-only; identity grouping is descriptive, never used to split training.
            identity = Path(relative).stem.split('_')[0]
            row = dict(image_path=str(destination.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                       content_sha256=digest, feature_path=str(Path(relative).with_suffix('') / (original.stem + '.npy')),
                       sample_id=len(rows), role='evaluation', video_id=video,
                       group_id='Celeb-DF/' + Path(relative).parts[0] + '/' + identity,
                       label=label, frame_index=frame['frame_index'], crop_settings=data['settings'])
            rows.append(row)
            kept += 1
        if not kept:
            excluded.append(dict(video=relative, reason='no_usable_frames'))
    if not features:
        write_json(args.output / 'data_manifest.json', rows)
        write_json(args.output / 'data_summary.json', dict(
            status='Data preprocessing only; no DINO extraction, training or evaluation',
            dataset='Celeb-DF official test', expected_videos=len(expected),
            expected_real_videos=sum(label == 0 for label in expected.values()),
            expected_fake_videos=sum(label == 1 for label in expected.values()),
            segmented_videos=len({r['video_id'] for r in rows}),
            segmented_images=len(rows), real_images=sum(r['label'] == 0 for r in rows),
            fake_images=sum(r['label'] == 1 for r in rows),
            retinaface_no_face_frames=sum(f['status'] != 'ok' for _, _, _, d in manifests for f in d['frames']),
            excluded=excluded, exact_within_test_duplicates=duplicates, exact_source_overlap=0,
            crops=str(args.crops), normal_output_dir=str(args.normal_output_dir),
            anomaly_output_dir=str(args.anomaly_output_dir)))
        print(f"Data-only preprocessing complete: {len(rows)} images; {args.output / 'data_summary.json'}", flush=True)
        return
    del parser
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    encoder_args = SimpleNamespace(device=args.device, model_path=spec['model_path'],
                                   dtype=spec['dtype'], batch_size=args.batch_size)
    model = processor = None
    for start in tqdm(range(0, len(rows), args.batch_size), desc='External frozen DINO'):
        batch = rows[start:start + args.batch_size]
        pending = []
        for row in batch:
            path = args.output / 'cache/evaluation' / row['feature_path']
            if path.exists() and path.with_suffix('.json').exists():
                load_feature(args.output / 'cache/evaluation', row)
            else:
                pending.append(row)
        if not pending:
            continue
        if model is None:
            model, processor = load_encoder(encoder_args)
        _, features = encode([r['image_path'] for r in pending], model, processor, encoder_args)
        for row, feature in zip(pending, features):
            path = args.output / 'cache/evaluation' / row['feature_path']
            path.parent.mkdir(parents=True, exist_ok=True)
            save_array(path, feature)
            write_json(path.with_suffix('.json'), dict(image=row, shape=list(feature.shape), dtype=str(feature.dtype)))
    usable = []
    for row in rows:
        feature = load_feature(args.output / 'cache/evaluation', row)
        try:
            mask = _foreground_mask(row['image_path'], feature.shape[:2], spec.get('foreground_minimum', .5))
        except ValueError:
            excluded.append(dict(video=row['video_id'], frame=row['frame_index'], reason='no_foreground_tokens'))
            continue
        padded = np.pad(mask.astype(np.int32), 1)
        neighbors = sum(padded[y:y + mask.shape[0], x:x + mask.shape[1]]
                        for y in range(3) for x in range(3) if (y, x) != (1, 1))
        if not (mask & (neighbors > 0)).any():
            excluded.append(dict(video=row['video_id'], frame=row['frame_index'], reason='no_neighbor_context'))
            continue
        usable.append(row)
    rows = usable
    if {r['label'] for r in rows} != {0, 1}:
        raise ValueError('No usable external examples for one or both classes')
    write_json(args.output / 'evaluation_rows.json', rows)
    write_json(args.output / 'coverage.json', dict(expected_videos=len(expected), scored_videos=len({r['video_id'] for r in rows}),
               expected_real_videos=sum(label == 0 for label in expected.values()),
               expected_fake_videos=sum(label == 1 for label in expected.values()),
               retinaface_no_face_frames=sum(f['status'] != 'ok' for _, _, _, d in manifests for f in d['frames']),
               frames=len(rows), excluded=excluded, exact_within_test_duplicates=duplicates,
               exact_source_overlap=0, limitation='Exact segmented JPEG hashes only; not identity or near-duplicate verification.'))


def video_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['video_id']].append(row)
    result = []
    for video, frames in grouped.items():
        if len({r['label'] for r in frames}) != 1:
            raise ValueError('Conflicting labels in a video')
        result.append(dict(video_id=video, label=frames[0]['label'],
                           score=float(np.mean([r['reconstruction'] for r in frames]))))
    return result


def evaluate(args):
    source = args.source_run
    meta = read_json(source / 'training.json')
    checkpoint = source / 'model.safetensors'
    if sha256(checkpoint) != meta['checkpoint_sha256']:
        raise ValueError('Reconstruction checkpoint changed')
    for name, digest in meta['fingerprint'].items():
        if sha256(args.bank / name) != digest:
            raise ValueError('Training source split/config changed')
    prep = read_json(args.output / 'preparation.json')
    if (prep['source_bank_config_sha256'] != meta['fingerprint']['bank_config.json']
            or prep['source_splits_sha256'] != meta['fingerprint']['splits.json']):
        raise ValueError('External features differ from training source encoder or split')
    # Reuse source real calibration ONLY. Never calibrate on external real or fake.
    calibration = read_json(source / 'stage2/calibration_scores.json')
    source_report = read_json(source / 'stage2/metrics.json')
    if source_report['checkpoint_sha256'] != meta['checkpoint_sha256']:
        raise ValueError('Source calibration belongs to another checkpoint')
    if not calibration or any(r['label'] != 0 for r in calibration):
        raise ValueError('Invalid source-only real calibration')
    config = SimpleNamespace(**meta['config'])
    config.device, config.workers, config.batch_size = args.device, args.workers, args.batch_size
    model = PatchReconstruction(**meta['architecture']).to(args.device)
    model.load_state_dict(load_file(str(checkpoint), device=args.device))
    rows = read_json(args.output / 'evaluation_rows.json')
    scored = score(model, rows, args.output, config)
    threshold = float(np.quantile([r['reconstruction'] for r in calibration], config.threshold_quantile))
    for row in scored:
        row['score'] = row['reconstruction']
        row['prediction'] = 'anomaly' if row['score'] > threshold else 'normal'
    videos = video_rows(scored)
    video_threshold = float(np.quantile([r['score'] for r in video_rows(calibration)], config.threshold_quantile))
    result = args.output / 'results' / source.name
    write_json(result / 'evaluation_scores.json', scored)
    write_json(result / 'video_scores.json', videos)
    report = dict(protocol='DFDC-only training/validation/calibration -> Celeb-DF official test',
                  method='reconstruction-only (no external NN fusion)',
                  source_run=str(source), checkpoint_sha256=meta['checkpoint_sha256'],
                  source_calibration_sha256=sha256(source / 'stage2/calibration_scores.json'),
                  external_rows_sha256=sha256(args.output / 'evaluation_rows.json'),
                  image=metrics(scored, threshold), video_mean=metrics(videos, video_threshold),
                  coverage=read_json(args.output / 'coverage.json'),
                  source_image_reconstruction=source_report['components']['reconstruction'],
                  threshold_source='DFDC calibration real q99; video threshold uses per-video mean scores')
    write_json(result / 'metrics.json', report)
    # Deterministic examples spread across classes, not selected by anomaly score.
    for label in (0, 1):
        first_frames = {}
        for row in scored:
            if row['label'] == label:
                first_frames.setdefault(row['video_id'], row)
        examples = list(first_frames.values())[:args.heatmap_count // 2]
        for row in examples:
            save_heatmap(row, result / 'heatmaps' / f"{row['sample_id']:08d}.png")
    print({key: report[key] for key in ('image', 'video_mean')}, flush=True)


def main(argv=None):
    os.environ['USE_TF'] = '0'
    os.environ['USE_TORCH'] = '1'
    project = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['segment', 'prepare', 'evaluate', 'all'], default='segment')
    p.add_argument('--bank', type=Path, default=project / 'RAG/normal/SEGFACE_FROZEN_NN_DEDUP_V2')
    p.add_argument('--source-run', type=Path, default=project / 'outputs/patch_reconstruction/SEGFACE_FROZEN_NN_DEDUP_V2/local_transformer_v1')
    p.add_argument('--data-root', type=Path, default=Path('/ssd2/DeepFakes/celeb-df-video'))
    p.add_argument('--crops', type=Path, default=Path('/ssd8/chihyu/Dataset/DeepFake_Dataset/Celeb-df-Frame-Face'))
    p.add_argument('--output', type=Path, default=Path('/ssd8/chihyu/Dataset/DeepFake_Dataset/Celeb-df-external'))
    p.add_argument('--normal-output-dir', type=Path, default=Path('/ssd8/chihyu/Dataset/DeepFake_Dataset/Celeb-df-Frame-Face-nomral'))
    p.add_argument('--anomaly-output-dir', type=Path, default=Path('/ssd8/chihyu/Dataset/DeepFake_Dataset/Celeb-df-Frame-Face-anomaly'))
    p.add_argument('--device', default='cuda:3')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--heatmap-count', type=int, default=32)
    args = p.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0 or args.heatmap_count < 0:
        p.error('Invalid batch, workers or heatmap count')
    for key in ('bank', 'source_run', 'data_root', 'crops', 'output', 'normal_output_dir', 'anomaly_output_dir'):
        setattr(args, key, getattr(args, key).resolve())
    directories = [args.crops, args.normal_output_dir, args.anomaly_output_dir, args.data_root, args.output]
    if any(a == b or a in b.parents or b in a.parents
           for i, a in enumerate(directories) for b in directories[:i]):
        p.error('Source, crops, class outputs and metadata directories must be separate')
    torch.set_num_threads(4)
    with experiment_lock(args.output):
        if args.stage == 'segment':
            prepare(args, project, features=False)
        if args.stage in ('prepare', 'all'):
            prepare(args, project)
        if args.stage in ('evaluate', 'all'):
            evaluate(args)


if __name__ == '__main__':
    main()
