"""Shared cache checks, multi-GPU extraction and result metrics for retrieval."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import numpy as np
from tqdm.auto import tqdm

from script.bank_data import image_records, read_json, save_array, write_json
from script.bank_encoder import encode, load_encoder
from script.bank_export import load_feature


@contextmanager
def experiment_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("w") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("此實驗正在執行，請勿重複啟動") from None
        yield


def load_sample(directory, row):
    stat = Path(row["image_path"]).stat()
    if (stat.st_size, stat.st_mtime_ns) != (row["size"], row["mtime_ns"]):
        raise ValueError(f"Source image changed: {row['image_path']}")
    return None, load_feature(directory, row)


def map_devices(function, items, devices):
    devices = devices or ["cpu"]
    def worker(lane):
        return [(i, function(items[i], devices[lane])) for i in range(lane, len(items), len(devices))]
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        results = [item for lane in pool.map(worker, range(len(devices))) for item in lane]
    return [value for _, value in sorted(results)]


def extract_roles(plan, roles, bank, cache, args):
    jobs = [(bank if role == "bank" else cache / role, row)
            for role in roles for row in image_records(plan["groups"][role], role)]
    devices = args.devices or ["cpu"]
    shards = [jobs[i::len(devices)] for i in range(len(devices))]
    model_lock = Lock()  # Transformers 初始化含全域狀態；序列化載入，推論仍平行。
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        def worker(shard, device):
            local = SimpleNamespace(**dict(vars(args), device=device))
            model = processor = None
            for start in tqdm(range(0, len(shard), args.batch_size),
                              desc=f"DINO {device} ({len(shard)} images)", unit="batch", dynamic_ncols=True):
                pending = []
                for directory, row in shard[start:start + args.batch_size]:
                    stat = Path(row["image_path"]).stat()
                    if (stat.st_size, stat.st_mtime_ns) != (row["size"], row["mtime_ns"]):
                        raise ValueError(f"Source image changed: {row['image_path']}")
                    path = directory / row["feature_path"]
                    if path.exists() and path.with_suffix(".json").exists():
                        load_sample(directory, row)
                    else:
                        pending.append((path, row))
                if pending:
                    if model is None:
                        with model_lock:
                            model, processor = load_encoder(local)
                    _, features = encode([row["image_path"] for _, row in pending], model, processor, local,
                                         image_pool=pool)
                    for (path, row), feature in zip(pending, features):
                        path.parent.mkdir(parents=True, exist_ok=True)
                        save_array(path, feature)
                        write_json(path.with_suffix(".json"), {"image": row, "shape": list(feature.shape),
                                                              "dtype": str(feature.dtype)})
            del model, processor
        map_devices(worker, shards, devices)


def metrics(rows, threshold):
    from sklearn.metrics import average_precision_score, roc_auc_score
    labels = np.array([row["label"] for row in rows])
    scores = np.array([row["score"] for row in rows])
    if set(labels) != {0, 1}:
        raise ValueError("Evaluation needs both real and fake images")
    return {"count": len(rows), "auroc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
            "false_positive_rate": float((scores[labels == 0] > threshold).mean()),
            "true_positive_rate": float((scores[labels == 1] > threshold).mean()), "threshold": threshold}
