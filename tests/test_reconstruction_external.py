import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import yaml

from script.bank_data import write_json, read_json
from script.reconstruction_external import prepare, segmented_name


class ExternalPreparationTests(unittest.TestCase):
    def test_flat_name_matches_dfdc_style(self):
        self.assertEqual(segmented_name('Celeb-synthesis/id1_id2_0003.mp4', 'frame_000042.jpg'),
                         'Celeb-synthesis-id1_id2_0003-frame_000042.jpg')

    def test_official_only_cache_and_overlap_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'utils').mkdir()
            (root / 'weights').mkdir()
            (root / 'weights/parser.safetensors').write_bytes(b'test')
            config = dict(model_dir='weights', checkpoint='parser.safetensors', parser_model='test',
                          parser_revision='test', face_ids=[1], background_ids=[0], pixel_confidence=.5,
                          dilation_ratio=0., closing_ratio=0., min_face_area=.02, background='black', jpeg_quality=95)
            (root / 'utils/config.yaml').write_text(yaml.safe_dump({'segment_face': config}))
            data = root / 'data'
            data.mkdir()
            paths = ['Celeb-real/one.mp4', 'Celeb-synthesis/two.mp4']
            (data / 'List_of_testing_videos.txt').write_text(f'1 {paths[0]}\n0 {paths[1]}\n')
            crops = root / 'crops'
            for label, relative in enumerate(paths):
                video = data / relative
                video.parent.mkdir(exist_ok=True)
                video.touch()
                destination = crops / Path(relative).with_suffix('')
                destination.mkdir(parents=True)
                Image.fromarray(np.full((32, 32, 3), 180 + 20 * label, dtype=np.uint8)).save(destination / 'frame_000000.jpg')
                write_json(destination / 'metadata.json', dict(label=label, split='test', source=str(video),
                    settings=dict(num_frames=32, threshold=.9, margin=.2, image_size=224, detection_size=640),
                    frames=[dict(file='frame_000000.jpg', status='ok', frame_index=0)]))
            bank = root / 'bank'
            write_json(bank / 'bank_config.json', dict(encoder_tuning={'enabled': False}, face_source='segface',
                       files_sha256={}, model_path=str(root), dtype='float16'))
            source = dict(video_id='source', group_id='source', label=0, crop_settings={},
                          frames=[dict(frame_index=0, content_sha256='not-an-external-image')])
            write_json(bank / 'splits.json', {'groups': {'bank': [source]}})
            args = SimpleNamespace(data_root=data, crops=crops, bank=bank, output=root / 'output',
                                   normal_output_dir=root / 'normal', anomaly_output_dir=root / 'anomaly',
                                   device='cpu', batch_size=2)
            parser = lambda images: [np.ones((im.height, im.width), np.uint8) for im in images]
            def encode(paths, *unused):
                return np.zeros((len(paths), 8)), np.ones((len(paths), 4, 4, 8), np.float16)
            with patch('script.reconstruction_external.SegFaceParser', return_value=parser), \
                    patch('script.reconstruction_external.load_encoder', return_value=(None, None)) as load_encoder, \
                    patch('script.reconstruction_external.encode', side_effect=encode):
                prepare(args, root, features=False)
                load_encoder.assert_not_called()
                self.assertFalse((args.output / 'cache').exists())
                self.assertFalse((args.output / 'evaluation_rows.json').exists())
                self.assertEqual(read_json(args.output / 'data_summary.json')['segmented_images'], 2)
                prepare(args, root)
                # A rerun checks metadata and reuses identical cached features.
                prepare(args, root)
            rows = read_json(args.output / 'evaluation_rows.json')
            self.assertEqual([r['label'] for r in rows], [0, 1])
            self.assertEqual(Path(rows[0]['image_path']).parent, args.normal_output_dir)
            self.assertEqual(Path(rows[1]['image_path']).parent, args.anomaly_output_dir)
            self.assertEqual(read_json(args.output / 'coverage.json')['scored_videos'], 2)
            source['frames'][0]['content_sha256'] = rows[0]['content_sha256']
            write_json(bank / 'splits.json', {'groups': {'bank': [source]}})
            protocol = read_json(args.output / 'preparation.json')
            from script.bank_data import sha256
            protocol['source_splits_sha256'] = sha256(bank / 'splits.json')
            write_json(args.output / 'preparation.json', protocol)
            with self.assertRaisesRegex(ValueError, 'Exact external/source image overlap'):
                prepare(args, root)
