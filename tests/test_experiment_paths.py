import tempfile
from pathlib import Path
import unittest

from script.bank_data import read_json, write_json
from script.experiment_paths import stage1_output, stage2_output
from script.retrieval_io import ensure_experiment_config


class ExperimentPathTests(unittest.TestCase):
    def test_stage_directories_are_explicit_and_separate(self):
        root = Path("/tmp/experiment")
        self.assertEqual(stage1_output(root), root / "Stage1_Bank_Build")
        self.assertEqual(stage2_output(root), root / "Stage2_Anomaly_Evaluation")
        self.assertNotEqual(stage1_output(root), stage2_output(root))

    def test_stage1_config_recovery_uses_new_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank, output = root / "bank", root / "output"
            spec = {"schema": "retrieval-v3-fsfm", "method": "topk"}
            write_json(stage1_output(output) / "config.json", spec)

            ensure_experiment_config(bank, output, spec)

            self.assertEqual(read_json(bank / "bank_config.json"), spec)
            self.assertFalse((output / "stage1").exists())


if __name__ == "__main__":
    unittest.main()
