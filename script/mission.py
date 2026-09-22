"""CPU/GPU resource allocation and supervised parallel segmentation jobs."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import yaml


def gpu_ids(value):
    if value.lower() in ("none", "cpu"):
        return []
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("Use comma-separated GPU IDs, e.g. 1,2, or none") from None
    if any(i < 0 for i in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("GPU IDs must be nonnegative and unique")
    return values


def allocate_cpus(available, cores, workers):
    if not 1 <= workers <= cores <= len(available):
        raise ValueError(f"Require 1 <= workers ({workers}) <= cores ({cores}) <= available CPUs ({len(available)})")
    chosen = sorted(available)[:cores]
    return [chosen[i::workers] for i in range(workers)]


def partition_jobs(jobs, workers):
    return [jobs[i::workers] for i in range(workers)]


def stop_processes(processes):
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_jobs(commands, environments, locks):
    processes = []

    def interrupted(*_):
        raise KeyboardInterrupt

    old_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        for command, environment in zip(commands, environments):
            processes.append(subprocess.Popen(command, env=environment, pass_fds=tuple(locks), start_new_session=True))
        pending = list(processes)
        while pending:
            for process in pending[:]:
                status = process.poll()
                if status is not None:
                    pending.remove(process)
                    if status:
                        raise RuntimeError(f"Segmentation worker {process.pid} failed (exit {status})")
            if pending:
                time.sleep(.1)
    finally:
        stop_processes(processes)
        signal.signal(signal.SIGTERM, old_handler)


def main(argv=None):
    project = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project / "utils/config.yaml").read_text())["mission"]
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("--cores", type=int, default=config["cpu_cores"])
    parser.add_argument("--gpus", type=gpu_ids, default=config["gpu_ids"])
    parser.add_argument("--cpu-workers", type=int, default=config["cpu_workers"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--help", action="store_true")
    resources, remaining = parser.parse_known_args(argv)
    if resources.help:
        parser.print_help()
        print("\nSegmentation options: --parts 0 1 ... --frames-per-video 10 --limit 1 --labels all|real|fake")
        return
    if any(option.split("=")[0] in ("--device", "--cpu-threads", "--manifest-list", "--lock-fds") for option in remaining):
        parser.error("Mission owns device/thread/job allocation; use --cores and --gpus")
    available = sorted(os.sched_getaffinity(0))
    if type(resources.cores) is not int or not 1 <= resources.cores <= len(available):
        parser.error(f"cores must be between 1 and {len(available)}")
    if type(resources.cpu_workers) is not int or resources.cpu_workers < 1:
        parser.error("cpu_workers must be positive")
    if not isinstance(resources.gpus, list) or any(type(i) is not int or i < 0 for i in resources.gpus) or len(set(resources.gpus)) != len(resources.gpus):
        parser.error("gpu_ids must be a list of unique nonnegative integers")
    if len(resources.gpus) > resources.cores:
        parser.error("Assign at least one CPU core per GPU worker")
    if not shutil.which("taskset"):
        parser.error("Linux taskset is required for CPU affinity")
    # Apply the total CPU budget before importing numerical libraries; child workers get disjoint subsets.
    os.sched_setaffinity(0, available[:resources.cores])
    from script.segment_face import lock_outputs, output_directories, parse_args, select_manifests
    import torch

    for gpu in resources.gpus:
        if gpu >= torch.cuda.device_count():
            parser.error(f"cuda:{gpu} is unavailable (visible GPU count: {torch.cuda.device_count()})")
    args = parse_args(remaining)
    jobs = select_manifests(args)
    if not jobs:
        raise ValueError("No matching videos found")
    count = min(len(jobs), len(resources.gpus) if resources.gpus else min(resources.cpu_workers, resources.cores))
    allocations = allocate_cpus(available, resources.cores, count)
    partitions = partition_jobs(jobs, count)
    devices = [f"cuda:{gpu}" for gpu in resources.gpus[:count]] if resources.gpus else ["cpu"] * count
    print(f"Mission: {len(jobs)} videos; {resources.cores} logical CPU cores; {len(set(devices)) if resources.gpus else 0} GPUs; {count} workers", flush=True)
    for i, (device, cpus, partition) in enumerate(zip(devices, allocations, partitions)):
        print(f"  worker {i}: {device}, CPU IDs={cpus}, threads={len(cpus)}, videos={len(partition)}", flush=True)
    if resources.dry_run:
        return
    with ExitStack() as stack:
        locks = lock_outputs(stack, output_directories(args))
        temporary = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="deepfake-mission-")))
        commands, environments = [], []
        for i, (device, cpus, partition) in enumerate(zip(devices, allocations, partitions)):
            job_file = temporary / f"worker_{i}.json"
            job_file.write_text(json.dumps([str(path) for path in partition]))
            commands.append(["taskset", "-c", ",".join(map(str, cpus)), sys.executable, "-u", str(project / "main.py"),
                             "--task", "segment-face", *remaining, "--device", device, "--cpu-threads", str(len(cpus)),
                             "--manifest-list", str(job_file), "--lock-fds", *map(str, locks)])
            environment = os.environ.copy()
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                environment[key] = str(len(cpus))
            environments.append(environment)
        run_jobs(commands, environments, locks)
    print("All segmentation tasks completed.", flush=True)


if __name__ == "__main__":
    main()
