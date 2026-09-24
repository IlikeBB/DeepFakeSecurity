import unittest
import numpy as np
import torch
from models.patch_reconstruction import PatchReconstruction
from script.patch_reconstruction import normalized_fusion, epoch
from script.real_patch_bank import _split_real_rows


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = PatchReconstruction(8, 4, 1).eval()
        self.x = torch.randn(2, 5, 5, 8)
        self.mask = torch.ones(2, 5, 5, dtype=torch.bool)

    def test_center_blind(self):
        before = self.model(self.x, self.mask)[0][:, 2, 2]
        self.x[:, 2, 2] = torch.randn(2, 8) * 100
        after = self.model(self.x, self.mask)[0][:, 2, 2]
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_background_blind_and_no_wraparound(self):
        self.mask[:, 1, 1] = False
        before = self.model(self.x, self.mask)[0]
        self.x[:, 1, 1] *= 100
        torch.testing.assert_close(before, self.model(self.x, self.mask)[0], rtol=0, atol=0)
        before = self.model(self.x, self.mask)[0][:, 0, 0]
        self.x[:, -1, -1] *= 100
        torch.testing.assert_close(before, self.model(self.x, self.mask)[0][:, 0, 0], rtol=0, atol=0)

    def test_isolated_token_and_training(self):
        mask = torch.zeros_like(self.mask)
        mask[:, 2, 2] = True
        self.assertFalse(self.model(self.x, mask)[2].any())
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.03)
        batches = [(self.x, self.mask, None)]
        first = epoch(self.model, batches, 'cpu')
        for _ in range(8):
            epoch(self.model, batches, 'cpu', optimizer)
        self.assertLess(epoch(self.model, batches, 'cpu'), first)

    def test_real_calibration_only_and_fixed_to_eval(self):
        cal = [dict(label=0, reconstruction=float(i), nearest=float(i+1)) for i in range(10)]
        test = [dict(label=1, reconstruction=1e8, nearest=1e8)]
        stats, threshold = normalized_fusion(cal, test, .5, .99)
        other_stats, other_threshold = normalized_fusion(cal, [], .5, .99)
        self.assertEqual(stats, other_stats)
        self.assertEqual(threshold, other_threshold)
        with self.assertRaises(ValueError):
            normalized_fusion(test, [], .5, .99)

    def test_family_split_rejects_fake(self):
        rows = [dict(label=0, group_id=str(i // 2)) for i in range(20)]
        fit, val = _split_real_rows(rows, .2, 42)
        self.assertFalse({r['group_id'] for r in fit} & {r['group_id'] for r in val})
        with self.assertRaises(ValueError):
            _split_real_rows(rows + [dict(label=1, group_id='fake')], .2, 42)

    def test_train_reload_evaluate_and_heatmap(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from PIL import Image
        from script.bank_data import write_json, read_json, image_records
        from script.patch_reconstruction import main
        from script.reconstruction_external import official_test, video_rows
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank = root / 'banks/test'
            bank.mkdir(parents=True)
            groups = {'bank': [], 'calibration': [], 'evaluation': []}
            rng = np.random.default_rng(4)
            for role in groups:
                for i in range(4):
                    path = root / f'{role}{i}.jpg'
                    Image.fromarray(np.full((16, 16, 3), 200, np.uint8)).save(path)
                    stat = path.stat()
                    groups[role].append(dict(video_id=f'{role}/{i}', group_id=f'{role}/{i}',
                        label=int(role == 'evaluation' and i > 1), crop_settings={}, frames=[dict(
                            image_path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                            frame_index=0, feature_path=f'{role}/{i}.npy')]))
                for row in image_records(groups[role], role):
                    cache = bank if role == 'bank' else bank / 'cache' / role
                    path = cache / row['feature_path']
                    path.parent.mkdir(parents=True, exist_ok=True)
                    value = rng.normal(size=(4, 4, 8)).astype(np.float32)
                    np.save(path, value)
                    write_json(path.with_suffix('.json'), dict(image=row, shape=list(value.shape), dtype=str(value.dtype)))
            write_json(bank / 'splits.json', {'groups': groups})
            write_json(bank / 'bank_config.json', dict(schema='retrieval-v1', encoder_tuning={'enabled': False}))
            args = SimpleNamespace(seed=42, bank_dir=str(root / 'banks'), experiment='test',
                results_dir=str(root / 'out'), run_name='smoke', stage='all', device='cpu', devices=['cpu'],
                validation_fraction=.25, max_train_images=0, max_validation_images=0, foreground_minimum=.5,
                hidden_dim=8, heads=2, layers=1, radius=1, learning_rate=.01, epochs=2, patience=2,
                batch_size=2, workers=0, nn_weight=0., threshold_quantile=.99, boundary_weight=.5,
                top_fraction=.1, heatmap_count=1)
            with patch('script.patch_reconstruction.parse_args', return_value=args):
                main([])
            output = root / 'out/test/smoke'
            report = read_json(output / 'stage2/metrics.json')
            self.assertEqual(report['image']['count'], 4)
            self.assertTrue((output / 'stage2/heatmaps/00000000.png').is_file())
            training = read_json(output / 'training.json')
            self.assertEqual(training['fake_training_images'], 0)
            rows = read_json(output / 'stage2/calibration_scores.json')
            self.assertEqual(len(video_rows(rows)), 4)
            # Official Celeb-DF labels are the inverse of this project's labels.
            (root / 'real.mp4').touch()
            (root / 'fake.mp4').touch()
            (root / 'List_of_testing_videos.txt').write_text('1 real.mp4\n0 fake.mp4\n')
            self.assertEqual(official_test(root), {'real.mp4': 0, 'fake.mp4': 1})
