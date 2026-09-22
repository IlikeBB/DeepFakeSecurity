import tempfile
import unittest
from pathlib import Path

from script.bank_data import sha256, write_json
from script.retrieval_io import ensure_experiment_config


class RecoveryTests(unittest.TestCase):
    def fixture(self, root):
        bank, output = root / "bank", root / "outputs"
        bank.mkdir()
        stage1 = output / "stage1"
        stage1.mkdir(parents=True)
        spec = {"experiment": "example", "seed": 42}
        # 非預設格式也必須原樣恢復，不能重新序列化造成雜湊不同。
        (stage1 / "config.json").write_text('{"experiment":"example","seed":42}\n')
        (stage1 / "splits.json").write_text('{"groups":{}}\n')
        feature = bank / "features.npy"
        feature.write_bytes(b"fixture")
        write_json(stage1 / "retrieval.json", {
            "source_config_sha256": sha256(stage1 / "config.json"),
            "plan_sha256": sha256(stage1 / "splits.json"),
            "files": {"features": {"path": str(feature), "sha256": sha256(feature)}}})
        return bank, output, stage1, spec

    def test_recovers_exact_metadata_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank, output, stage1, spec = self.fixture(Path(tmp))
            ensure_experiment_config(bank, output, spec)
            self.assertEqual(sha256(bank / "bank_config.json"), sha256(stage1 / "config.json"))
            self.assertEqual(sha256(bank / "splits.json"), sha256(stage1 / "splits.json"))
            ensure_experiment_config(bank, output, spec)

    def test_missing_features_do_not_get_marked_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank, output, _, spec = self.fixture(Path(tmp))
            (bank / "features.npy").unlink()
            with self.assertRaisesRegex(ValueError, "RAG 索引檔案遺失"):
                ensure_experiment_config(bank, output, spec)
            self.assertFalse((bank / "bank_config.json").exists())

    def test_rejects_changed_config_or_plan_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank, output, stage1, spec = self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, "YAML 設定不同"):
                ensure_experiment_config(bank, output, dict(spec, seed=1))
            (stage1 / "splits.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "雜湊不符"):
                ensure_experiment_config(bank, output, spec)
            self.assertFalse((bank / "bank_config.json").exists())


if __name__ == "__main__":
    unittest.main()
