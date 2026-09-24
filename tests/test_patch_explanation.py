import tempfile
import unittest
from pathlib import Path

import numpy as np

from script.patch_explanation import (evidence_maps, explain, fit_patch_normalization,
                                      nearest_lookup, patch_fusion_score)


class PatchExplanationTests(unittest.TestCase):
    def _row(self, path, offset=0.):
        return dict(image_path=path, video_id='video', group_id='family', label=0,
                    score=1.2, grid=[3, 3], patch_ids=list(range(9)),
                    patch_errors=(np.linspace(.1, .9, 9) + offset).tolist())

    def test_calibrated_fusion_and_grounded_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, baseline, rows = [], [], []
            for index, offset in enumerate((0., .1)):
                path = Path(directory) / f'{index}.npz'
                np.savez(path, distances=(np.linspace(.2, 1., 9) + offset).reshape(3, 3),
                         neighbors=np.zeros((3, 3), dtype=np.int64))
                image = f'image-{index}.jpg'
                paths.append(path)
                rows.append(self._row(image, offset))
                baseline.append(dict(image_path=image, video_id='video', group_id='family',
                                     label=0, patch_matches=str(path), score=.5))
            lookup = nearest_lookup(baseline, rows)
            statistics = fit_patch_normalization(rows, lookup, .9, 1.)
            maps = evidence_maps(rows[1], lookup[rows[1]['image_path']], statistics, .5, 1.)
            self.assertEqual(set(maps), {'reconstruction', 'nearest', 'fusion'})
            np.testing.assert_allclose(maps['fusion'],
                                       .5 * maps['reconstruction'] + .5 * maps['nearest'])
            self.assertGreaterEqual(patch_fusion_score(maps, .25), float(np.nanmean(maps['fusion'])))
            result = explain(rows[1], maps, threshold=1., top_fraction=.25)
            self.assertEqual(result['prediction'], 'anomaly')
            self.assertIn('異常門檻', result['summary_zh'])
            self.assertIn('不是像素級偽造標註', result['summary_zh'])
            self.assertTrue(result['regions'])

    def test_calibration_rejects_fake_and_lookup_mismatch(self):
        row = self._row('real.jpg')
        with self.assertRaisesRegex(ValueError, 'real images only'):
            fit_patch_normalization([dict(row, label=1)], None, .9, 1.)
        with self.assertRaisesRegex(ValueError, 'image sets differ'):
            nearest_lookup([], [row])
