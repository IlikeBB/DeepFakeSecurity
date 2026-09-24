"""Deterministic real-only image augmentation for the Stage 1 normal bank."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
from tqdm.auto import tqdm

from script.bank_data import image_records, read_json, sha256, write_json


SCHEMA = 'real-bank-augmentation-v1'


def validate_augmentation_config(config):
    if not isinstance(config, dict) or type(config.get('enabled')) is not bool:
        raise ValueError('bank_augmentation.enabled must be bool')
    if not config['enabled']:
        return
    if type(config.get('views_per_image')) is not int or config['views_per_image'] < 1:
        raise ValueError('bank_augmentation.views_per_image must be a positive integer')
    if type(config.get('seed')) is not int:
        raise ValueError('bank_augmentation.seed must be an integer')
    for name, lower_limit, upper_limit in (
            ('brightness', .5, 1.5), ('noise_std', 0., .1),
            ('rotation_degrees', 0., 15.), ('scale', .9, 1.1)):
        values = config.get(name)
        if (not isinstance(values, list) or len(values) != 2
                or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values)
                or not lower_limit <= values[0] <= values[1] <= upper_limit):
            raise ValueError(f'bank_augmentation.{name} has an invalid range')
    shear = config.get('shear_degrees')
    translation = config.get('translate_fraction')
    fraction = config.get('bank_patch_fraction')
    if not isinstance(shear, (int, float)) or not math.isfinite(shear) or not 0 <= shear <= 5:
        raise ValueError('bank_augmentation.shear_degrees must be between 0 and 5')
    if (not isinstance(translation, (int, float)) or not math.isfinite(translation)
            or not 0 <= translation <= .05):
        raise ValueError('bank_augmentation.translate_fraction must be between 0 and 0.05')
    if (not isinstance(fraction, (int, float)) or not math.isfinite(fraction)
            or not 0 < fraction <= 1):
        raise ValueError('bank_augmentation.bank_patch_fraction must be in (0, 1]')


def _row_digest(rows):
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _rng(row, config, view):
    source = row.get('content_sha256', row['image_path'])
    digest = hashlib.sha256(f"{config['seed']}|{source}|{view}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little'))


def augmentation_parameters(row, config, view):
    """Generate stable parameters independently of worker count and execution order."""
    rng = _rng(row, config, view)
    rotation = float(rng.uniform(*config['rotation_degrees']))
    rotation *= -1 if rng.integers(2) else 1
    shear = float(rng.uniform(-config['shear_degrees'], config['shear_degrees']))
    shear_y = float(rng.uniform(-config['shear_degrees'], config['shear_degrees']))
    translation = config['translate_fraction']
    return {
        'brightness': float(rng.uniform(*config['brightness'])),
        'noise_std': float(rng.uniform(*config['noise_std'])),
        'rotation_degrees': rotation,
        'scale_x': float(rng.uniform(*config['scale'])),
        'scale_y': float(rng.uniform(*config['scale'])),
        'shear_x_degrees': shear,
        'shear_y_degrees': shear_y,
        'translate_x_fraction': float(rng.uniform(-translation, translation)),
        'translate_y_fraction': float(rng.uniform(-translation, translation)),
        'noise_seed': int(rng.integers(0, 2**63)),
        'patch_seed': int(rng.integers(0, 2**63)),
    }


def augment_image(image, parameters):
    """Apply one mild affine transform, brightness change and foreground-only noise."""
    image = image.convert('RGB')
    width, height = image.size
    angle = math.radians(parameters['rotation_degrees'])
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]], dtype=np.float64)
    deformation = np.array([
        [parameters['scale_x'], math.tan(math.radians(parameters['shear_x_degrees']))],
        [math.tan(math.radians(parameters['shear_y_degrees'])), parameters['scale_y']],
    ], dtype=np.float64)
    linear = rotation @ deformation
    center = np.array([(width - 1) / 2, (height - 1) / 2], dtype=np.float64)
    shift = np.array([parameters['translate_x_fraction'] * width,
                      parameters['translate_y_fraction'] * height])
    forward = np.eye(3, dtype=np.float64)
    forward[:2, :2] = linear
    forward[:2, 2] = center + shift - linear @ center
    inverse = np.linalg.inv(forward)
    transformed = image.transform(
        image.size, Image.Transform.AFFINE, tuple(inverse[:2].reshape(-1)),
        resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))
    foreground = np.asarray(transformed).max(axis=-1) > 16
    transformed = ImageEnhance.Brightness(transformed).enhance(parameters['brightness'])
    pixels = np.asarray(transformed, dtype=np.float32).copy()
    noise = np.random.default_rng(parameters['noise_seed']).normal(
        0., parameters['noise_std'] * 255., pixels.shape)
    pixels[foreground] += noise[foreground]
    pixels[~foreground] = 0
    return Image.fromarray(np.clip(np.rint(pixels), 0, 255).astype(np.uint8), 'RGB')


def _augmented_record(job, bank, config):
    position, source, view = job
    source_path = Path(source['image_path'])
    source_stat = source_path.stat()
    if (source_stat.st_size, source_stat.st_mtime_ns) != (source['size'], source['mtime_ns']):
        raise ValueError(f'Source image changed: {source_path}')
    parameters = augmentation_parameters(source, config, view)
    relative = Path(source['feature_path'])
    image_path = bank / 'augmentations/images' / f'view_{view:02d}' / relative.with_suffix('.png')
    feature_path = Path('augmentations/features') / f'view_{view:02d}' / relative
    sidecar = image_path.with_suffix('.json')
    source_spec = {
        'image_path': source['image_path'],
        'size': source['size'],
        'mtime_ns': source['mtime_ns'],
        'content_sha256': source.get('content_sha256'),
    }
    spec = {'schema': SCHEMA, 'source': source_spec, 'view': view,
            'parameters': parameters, 'feature_path': feature_path.as_posix(),
            'bank_patch_fraction': config['bank_patch_fraction']}
    if image_path.is_file() and sidecar.is_file():
        data = read_json(sidecar)
        if data.get('spec') == spec:
            stat = image_path.stat()
            row = data.get('row', {})
            if (row.get('size'), row.get('mtime_ns')) == (stat.st_size, stat.st_mtime_ns):
                return row
    with Image.open(source_path) as image:
        augmented = augment_image(image, parameters)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = image_path.with_name(image_path.stem + '.tmp.png')
    augmented.save(temporary, format='PNG')
    temporary.replace(image_path)
    stat = image_path.stat()
    row = dict(source)
    row.update({
        'image_path': str(image_path.resolve()),
        'feature_path': feature_path.as_posix(),
        'sample_id': position,
        'size': stat.st_size,
        'mtime_ns': stat.st_mtime_ns,
        'content_sha256': sha256(image_path),
        'source_image_path': source['image_path'],
        'source_content_sha256': source.get('content_sha256'),
        'augmentation': {
            'schema': SCHEMA,
            'view': view,
            'parameters': parameters,
            'bank_patch_fraction': config['bank_patch_fraction'],
        },
    })
    write_json(sidecar, {'spec': spec, 'row': row})
    return row


def prepare_bank_rows(plan, bank, config, workers):
    """Create augmented real images and return original + augmented bank records."""
    validate_augmentation_config(config)
    original = image_records(plan['groups']['bank'], 'bank')
    if not original or any(row['label'] != 0 for row in original):
        raise ValueError('Stage 1 augmentation accepts bank real images only')
    if not config['enabled']:
        return original
    jobs = []
    for source in original:
        for view in range(config['views_per_image']):
            jobs.append((len(original) + len(jobs), source, view))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        augmented = list(tqdm(
            pool.map(lambda job: _augmented_record(job, bank, config), jobs),
            total=len(jobs), desc='Stage 1：產生 real 增強視圖', unit='image'))
    rows = [dict(row) for row in original] + augmented
    manifest = {
        'schema': SCHEMA,
        'config': config,
        'source_rows_sha256': _row_digest(original),
        'original_images': len(original),
        'augmented_images': len(augmented),
        'rows': augmented,
    }
    write_json(bank / 'augmentations/manifest.json', manifest)
    return rows


def load_bank_rows(plan, bank):
    """Load the exact Stage 1 bank population declared by bank_config.json."""
    original = image_records(plan['groups']['bank'], 'bank')
    config_path = bank / 'bank_config.json'
    config = read_json(config_path).get('bank_augmentation', {'enabled': False}) if config_path.is_file() else {
        'enabled': False}
    if not config.get('enabled', False):
        return original
    validate_augmentation_config(config)
    manifest_path = bank / 'augmentations/manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Missing Stage 1 augmentation manifest: {manifest_path}')
    manifest = read_json(manifest_path)
    if (manifest.get('schema') != SCHEMA or manifest.get('config') != config
            or manifest.get('source_rows_sha256') != _row_digest(original)):
        raise ValueError('Stage 1 augmentation manifest does not match split/config')
    augmented = manifest.get('rows', [])
    expected = len(original) * config['views_per_image']
    if len(augmented) != expected or any(row.get('label') != 0 for row in augmented):
        raise ValueError('Stage 1 augmentation manifest has an invalid population')
    for row in augmented:
        path = Path(row['image_path'])
        if not path.is_file() or (path.stat().st_size, path.stat().st_mtime_ns) != (
                row['size'], row['mtime_ns']):
            raise ValueError(f"Augmented image missing or changed: {path}")
    return original + augmented
