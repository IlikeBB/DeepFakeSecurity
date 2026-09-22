"""Stage 1 real feature bank; Stage 2 exact cosine retrieval and held-out evaluation."""

import argparse
import math
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits
import yaml

from script.bank_data import read_json, sha256, write_json
from script.bank_flat import prepare_flat_plan
from script.bank_retrieval import build, evaluate, load_index
from script.retrieval_io import ensure_experiment_config, experiment_lock, extract_roles


def parse_args(argv=None):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "utils/config.yaml").read_text(encoding="utf-8"))
    defaults = dict(config["retrieval"], model_path=config["model_path"],
                    normal_dir=config["patch_bank"]["normal_dir"], anomaly_dir=config["patch_bank"]["anomaly_dir"],
                    source_root=config["feature_bank"]["source_root"], label_csv=config["feature_bank"]["label_csv"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "stage1", "stage2", "all"), default="stage1")
    parser.add_argument("--exper", "--experiment", dest="experiment")
    parser.add_argument("--gpus", dest="gpu_ids", type=int, nargs="*", help="GPU IDs; empty list uses CPU")
    parser.add_argument("--batch-size", "--extract-batch-size", dest="batch_size", type=int)
    parser.add_argument("--method", choices=("nearest", "topk", "cross_attention"), default="nearest")
    for key in ("workers", "cpu_threads", "query_chunk_size", "bank_chunk_size", "match_count"):
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if (not isinstance(args.experiment, str) or not args.experiment.strip()
            or args.experiment in (".", "..") or Path(args.experiment).name != args.experiment):
        parser.error("請先在 utils/config.yaml 的 retrieval.experiment 填入你取的實驗名稱，或使用 --exper 指定")
    if type(args.export_previews) is not bool:
        parser.error("utils/config.yaml 的 retrieval.export_previews 必須是 true 或 false")
    if min(args.workers, args.cpu_threads, args.batch_size, args.query_chunk_size, args.bank_chunk_size,
           args.match_count, args.pca_images) < 1:
        parser.error("執行緒、batch 與 chunk 參數必須大於 0")
    if (not 0 < args.foreground_minimum <= 1 or not 0 < args.top_fraction <= 1
            or not 0 < args.threshold_quantile < 1 or args.dtype not in ("float16", "float32")
            or not 1 <= args.jpeg_quality <= 100):
        parser.error("Invalid retrieval configuration")
    if (len(set(args.gpu_ids)) != len(args.gpu_ids)
            or any(gpu < 0 or gpu >= torch.cuda.device_count() for gpu in args.gpu_ids)):
        parser.error("--gpus 請指定可用且不重複的 GPU 編號；空的 --gpus 使用 CPU")
    args.devices = [f"cuda:{gpu}" for gpu in args.gpu_ids]
    for key in ("bank_dir", "results_dir", "model_path", "normal_dir", "anomaly_dir", "source_root", "label_csv"):
        setattr(args, key, str((root / Path(getattr(args, key)).expanduser()).resolve()))
    for destination in (Path(args.bank_dir), Path(args.results_dir)):
        for name in ("normal_dir", "anomaly_dir", "source_root"):
            source = Path(getattr(args, name))
            if destination == source or source in destination.parents:
                parser.error("輸出不能放入來源資料集")
    if args.bank_dir == args.results_dir:
        parser.error("特徵目錄與結果目錄必須分開")
    if args.method not in ("nearest", "topk", "cross_attention"):
        parser.error("retrieval.method 必須是 nearest、topk 或 cross_attention")
    if args.method != "nearest":
        config = args.attention
        integers = ("neighbors", "hidden", "heads", "epochs", "patience", "batch_size", "patches_per_image")
        if (any(type(config[key]) is not int or config[key] < 1 for key in integers)
                or (args.method == "cross_attention" and config["neighbors"] < 2)
                or config["hidden"] % config["heads"] or not 0 < config["validation_fraction"] < 1
                or any(not math.isfinite(config[key]) or config[key] <= 0 for key in ("temperature", "learning_rate"))
                or any(not math.isfinite(config[key]) or config[key] < 0 for key in ("noise_std", "weight_decay"))):
            parser.error("Invalid retrieval.attention configuration")
    return args


def main(argv=None):
    args = parse_args(argv)
    bank = Path(args.bank_dir) / args.experiment
    cache = bank / "cache"  # 輔助查詢特徵；不納入 real 檢索索引。
    output = Path(args.results_dir) / args.experiment
    completion = output / "stage1/retrieval.json"
    if args.stage == "stage2" and not completion.is_file():
        raise ValueError(f"請先執行 bash run.sh stage1 --exper {args.experiment}；找不到 {completion}")
    runtime = {"stage", "gpu_ids", "devices", "workers", "cpu_threads", "batch_size", "query_chunk_size", "bank_chunk_size", "export_previews"}
    spec = {key: value for key, value in vars(args).items() if key not in runtime}
    spec.update(schema="retrieval-v1", method=getattr(args, "method", "nearest"),
                files_sha256={name: sha256(Path(args.model_path) / name)
                              for name in ("model.safetensors", "config.json", "preprocessor_config.json")},
                labels_sha256=sha256(args.label_csv))
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    with experiment_lock(output), experiment_lock(bank), threadpool_limits(limits=args.cpu_threads):
        ensure_experiment_config(bank, output, spec)
        plan_file = bank / "splits.json"
        if plan_file.exists():
            plan = read_json(plan_file)
        elif args.stage in ("prepare", "stage1", "all"):
            plan = prepare_flat_plan(dict(vars(args), full_data=True))
            write_json(plan_file, plan)
        else:
            raise ValueError("請先完成 Stage 1，來源切分不存在")
        family_sets = [{row["group_id"] for role in roles for row in plan["groups"][role]}
                       for roles in (("bank", "train_fake"), ("calibration",), ("evaluation",))]
        if any(family_sets[i] & family_sets[j] for i in range(3) for j in range(i)):
            raise ValueError("Source families overlap between training, calibration and evaluation")
        print(f"Bank: {bank}\nOutput: {output}\nGPUs: {args.devices or ['cpu']}; workers={args.workers}", flush=True)
        print({role: sum(len(v["frames"]) for v in values) for role, values in plan["groups"].items()}, flush=True)
        if args.stage in ("prepare", "stage1", "all"):
            write_json(output / "stage1/config.json", spec)
            write_json(output / "stage1/splits.json", plan)
        if args.stage in ("stage1", "all"):
            if not completion.exists():
                extract_roles(plan, ("bank",), bank, cache, args)
            if args.export_previews:
                from script.bank_export import build_bank
                build_bank(plan, args, bank)  # 選用：以既有 NPY 補齊 PCA／JPG 預覽，不影響索引。
            build(plan, args, bank, output)
            if args.method == "cross_attention":
                from script.bank_attention import train_attention
                _, sources, arrays = load_index(plan, bank, output)
                train_attention(sources, arrays, args, bank, output)
        if args.stage in ("stage2", "all"):
            load_index(plan, bank, output)  # 先驗證 real 索引完整性，再讀取校準／測試圖片。
            if args.method == "cross_attention":
                from script.bank_attention import attention_info
                attention_info(output)
            extract_roles(plan, ("calibration", "evaluation"), bank, cache, args)
            evaluate(plan, args, bank, cache, output)
