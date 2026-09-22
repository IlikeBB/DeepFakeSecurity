"""Shared cache checks, multi-GPU extraction and result metrics for retrieval."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
from pathlib import Path
import shutil
from threading import Lock
from types import SimpleNamespace

import numpy as np
from tqdm.auto import tqdm

from script.bank_data import image_records, read_json, save_array, sha256, write_json
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


def ensure_experiment_config(bank, output, spec):
    """Recover missing metadata only from the matching Stage 1 backup."""
    config_file = bank / "bank_config.json"
    if config_file.exists():
        existing = read_json(config_file)
        if existing != spec:
            raise ValueError("實驗資料／模型設定已變更；請在 utils/config.yaml 的 retrieval.experiment 填入新名稱，或使用 --exper")
        return
    stage1 = output / "stage1"
    backup = stage1 / "config.json"
    completion = stage1 / "retrieval.json"
    if not backup.exists():
        if (bank / "splits.json").exists() or completion.exists():
            raise ValueError(f"實驗缺少 {config_file}，且沒有 {backup} 可恢復；請還原原始 RAG 資料夾")
        write_json(config_file, spec)
        return
    if read_json(backup) != spec:
        raise ValueError(f"{backup} 與目前 YAML 設定不同，請還原該實驗設定或另取 experiment 名稱")
    plan_file = bank / "splits.json"
    plan_backup = stage1 / "splits.json"
    if completion.exists():
        info = read_json(completion)
        missing = [item["path"] for item in info["files"].values() if not Path(item["path"]).is_file()]
        if missing:
            raise ValueError(f"Stage 1 結果存在，但 RAG 索引檔案遺失；請先還原 {bank}。缺少：" + ", ".join(missing))
        plan_source = plan_file if plan_file.exists() else plan_backup
        if (sha256(backup) != info["source_config_sha256"] or not plan_source.is_file()
                or sha256(plan_source) != info["plan_sha256"]):
            raise ValueError("Stage 1 備份與檢索索引的設定／資料切分雜湊不符，無法自動恢復")
    # Copy exact bytes: the completed retrieval index references their hashes.
    if not plan_file.exists() and plan_backup.exists():
        temporary = plan_file.with_suffix(".json.tmp")
        shutil.copyfile(plan_backup, temporary)
        temporary.replace(plan_file)
    temporary = config_file.with_suffix(".json.tmp")
    shutil.copyfile(backup, temporary)
    temporary.replace(config_file)
    print(f"已從 Stage 1 備份恢復設定：{config_file}", flush=True)


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
    with ThreadPoolExecutor(max_workers=args.workers) as pool, tqdm(
            total=len(jobs), desc="DINO 特徵提取／快取檢查", unit="image", dynamic_ncols=True, mininterval=.5) as progress:
        def worker(shard, device):
            local = SimpleNamespace(**dict(vars(args), device=device, quiet=True))
            model = processor = None
            for start in range(0, len(shard), args.batch_size):
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
                with progress.get_lock():
                    progress.update(min(args.batch_size, len(shard) - start))
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
