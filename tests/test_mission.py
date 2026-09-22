import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from script.bank_data import write_json
from script.mission import allocate_cpus, gpu_ids, partition_jobs, run_jobs
from script.segment_face import lock_outputs, select_manifests


class MissionTests(unittest.TestCase):
    def test_resource_allocation(self):
        groups = allocate_cpus([2, 4, 6, 8, 10, 12, 14, 16], 8, 3)
        self.assertEqual(sorted(cpu for group in groups for cpu in group), [2, 4, 6, 8, 10, 12, 14, 16])
        self.assertEqual([len(group) for group in groups], [3, 3, 2])
        for cores, workers in ((0, 1), (2, 3), (9, 1)):
            with self.assertRaises(ValueError):
                allocate_cpus(range(8), cores, workers)
        self.assertEqual(gpu_ids("1,2"), [1, 2])
        self.assertEqual(gpu_ids("none"), [])
        for value in ("1,1", "-1", "", "two"):
            with self.assertRaises(argparse.ArgumentTypeError):
                gpu_ids(value)

    def test_class_limits_and_disjoint_partitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            for index, label in enumerate((0, 1, 1, 0, 1, 0)):
                path = Path(tmp) / f"dfdc_train_part_0/video{index}/metadata.json"
                path.parent.mkdir(parents=True)
                write_json(path, {"label": label})
            args = SimpleNamespace(input_dir=tmp, parts=[0], labels="all", limit=2)
            selected = select_manifests(args)
            self.assertEqual([p.parent.name for p in selected], ["video0", "video1", "video2", "video3"])
            for workers in (1, 2, 3):
                partitions = partition_jobs(selected, workers)
                flat = [job for partition in partitions for job in partition]
                self.assertEqual(len(flat), len(set(flat)))
                self.assertCountEqual(flat, selected)
            args.labels = "real"
            self.assertEqual([p.parent.name for p in select_manifests(args)], ["video0", "video3"])

    def test_shared_output_lock_excludes_other_runs(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as parent:
            outputs = {0: Path(tmp) / "normal"}
            locks = lock_outputs(parent, outputs)
            with ExitStack() as competitor:
                with self.assertRaises(RuntimeError):
                    lock_outputs(competitor, outputs)
            with ExitStack() as child:
                lock_outputs(child, outputs, [os.dup(locks[0])])
            # Closing a worker's duplicate must not release the parent's lock.
            with ExitStack() as competitor:
                with self.assertRaises(RuntimeError):
                    lock_outputs(competitor, outputs)
            self.assertEqual(list(outputs[0].iterdir()), [])

    def test_worker_failure_stops_remaining_processes(self):
        started = []
        original = subprocess.Popen

        def record(*args, **kwargs):
            process = original(*args, **kwargs)
            started.append(process)
            return process

        commands = [[sys.executable, "-c", "import time; time.sleep(60)"],
                    [sys.executable, "-c", "raise SystemExit(3)"]]
        with patch("script.mission.subprocess.Popen", side_effect=record):
            with self.assertRaisesRegex(RuntimeError, "exit 3"):
                run_jobs(commands, [os.environ.copy()] * 2, [])
        self.assertEqual(len(started), 2)
        self.assertTrue(all(process.poll() is not None for process in started))


if __name__ == "__main__":
    unittest.main()
