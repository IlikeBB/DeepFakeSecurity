"""Stage 1 real feature bank; Stage 2 exact cosine retrieval and held-out evaluation."""

import argparse
import math
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits
import yaml

from script.bank_data import prepare_plan, read_json, sha256, write_json
from script.bank_retrieval import build, evaluate, load_index
from script.retrieval_io import ensure_experiment_config, experiment_lock, extract_roles


def parse_args(argv=None):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "utils/config.yaml").read_text(encoding="utf-8"))
    defaults = dict(config["retrieval"], model_path=config["model_path"],
                    faces_dir=config["crop_face"]["output_dir"],
                    segface_normal_dir=config["segment_face"]["normal_output_dir"],
                    segface_anomaly_dir=config["segment_face"]["anomaly_output_dir"],
                    source_root=config["crop_face"]["data_root"],
                    label_csv=config["crop_face"]["label_csv"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "stage1", "stage2", "ablation", "all"), default="stage1")
    parser.add_argument("--exper", "--experiment", dest="experiment")
    parser.add_argument("--gpus", dest="gpu_ids", type=int, nargs="*", help="GPU IDs; empty list uses CPU")
    parser.add_argument("--batch-size", "--extract-batch-size", dest="batch_size", type=int)
    parser.add_argument("--face-source", choices=("retinaface", "segface"))
    parser.add_argument("--tune-encoder", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--method", choices=("nearest", "topk", "cross_attention"), default="nearest")
    for key in ("workers", "cpu_threads", "query_chunk_size", "bank_chunk_size", "match_count"):
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.tune_encoder is not None:
        args.encoder_tuning = dict(args.encoder_tuning, enabled=args.tune_encoder)
    del args.tune_encoder
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
    for key in ("bank_dir", "results_dir", "model_path", "faces_dir", "segface_normal_dir",
                "segface_anomaly_dir", "source_root", "label_csv"):
        setattr(args, key, str((root / Path(getattr(args, key)).expanduser()).resolve()))
    if args.face_source == "segface":
        for key in ("segface_normal_dir", "segface_anomaly_dir"):
            if not Path(getattr(args, key)).is_dir():
                parser.error(f"SegFace input directory does not exist: {getattr(args, key)}")
    for destination in (Path(args.bank_dir), Path(args.results_dir)):
        for name in ("faces_dir", "segface_normal_dir", "segface_anomaly_dir", "source_root"):
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
    ablation = args.ablation
    if (type(ablation["max_per_family"]) is not int or ablation["max_per_family"] < 1
            or not math.isfinite(ablation["boundary_weight"])
            or not 0 <= ablation["boundary_weight"] <= 1):
        parser.error("Invalid retrieval.ablation configuration")
    tuning = args.encoder_tuning
    compression = tuning.get("compression", {})
    local = tuning.get("local_anomaly", {})
    integers = ("rank", "epochs", "patience", "batch_size", "images_per_epoch", "validation_images",
                "fake_interval", "fake_batch_size")
    positive = ("alpha", "learning_rate", "anchor_weight", "fake_margin", "gradient_clip")
    if (type(tuning.get("enabled")) is not bool
            or any(type(tuning.get(key)) is not int or tuning[key] < 1 for key in integers)
            or any(not math.isfinite(tuning.get(key, 0)) or tuning[key] <= 0 for key in positive)
            or not math.isfinite(tuning.get("weight_decay", -1)) or tuning["weight_decay"] < 0
            or not math.isfinite(tuning.get("fake_weight", -1)) or tuning["fake_weight"] < 0
            or not math.isfinite(tuning.get("minimum_delta", -1)) or tuning["minimum_delta"] < 0
            or not 0 < tuning.get("validation_fraction", 0) < 1
            or any(not isinstance(compression.get(key), list) or len(compression[key]) != 2
                   or any(type(value) is not int for value in compression[key])
                   or not 1 <= compression[key][0] <= compression[key][1] <= 100
                   for key in ("mild_quality", "severe_quality"))
            or any(not math.isfinite(compression.get(key, 0)) or not 0 < compression[key] <= 1
                   for key in ("mild_scale", "severe_scale"))
            or not math.isfinite(compression.get("affinity_weight", -1))
            or compression["affinity_weight"] < 0
            or not isinstance(local.get("quality"), list) or len(local["quality"]) != 2
            or any(type(value) is not int for value in local["quality"])
            or not 1 <= local["quality"][0] <= local["quality"][1] <= 100
            or any(not math.isfinite(local.get(key, -1)) for key in
                   ("scale", "minimum_fraction", "maximum_fraction", "mask_threshold", "margin", "weight",
                    "background_weight"))
            or not 0 < local["scale"] <= 1
            or not 0 < local["minimum_fraction"] <= local["maximum_fraction"] < 1
            or not 0 < local["mask_threshold"] <= 1 or local["margin"] <= 0
            or local["weight"] < 0 or local["background_weight"] < 0):
        parser.error("Invalid retrieval.encoder_tuning configuration")
    return args


def main(argv=None):
    args = parse_args(argv)
    bank = Path(args.bank_dir) / args.experiment
    cache = bank / "cache"  # 輔助查詢特徵；不納入 real 檢索索引。
    output = Path(args.results_dir) / args.experiment
    completion = output / "stage1/retrieval.json"
    if args.stage in ("stage2", "ablation") and not completion.is_file():
        raise ValueError(f"請先執行 bash run.sh stage1 --exper {args.experiment}；找不到 {completion}")
    runtime = {"stage", "gpu_ids", "devices", "workers", "cpu_threads", "batch_size", "query_chunk_size",
               "bank_chunk_size", "export_previews", "ablation", "encoder_checkpoint"}
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
            plan = prepare_plan(dict(vars(args), full_data=True))
            write_json(plan_file, plan)
        else:
            raise ValueError("請先完成 Stage 1，來源切分不存在")
        family_sets = [{row["group_id"] for role in roles for row in plan["groups"][role]}
                       for roles in (("bank", "train_fake"), ("calibration",), ("evaluation",))]
        if any(family_sets[i] & family_sets[j] for i in range(3) for j in range(i)):
            raise ValueError("Source families overlap between training, calibration and evaluation")
        counts = {role: sum(len(video["frames"]) for video in videos)
                  for role, videos in plan["groups"].items()}
        print(f"Experiment={args.experiment}; devices={args.devices or ['cpu']}; "
              f"workers={args.workers}; frames={counts}", flush=True)
        if args.stage in ("prepare", "stage1", "all"):
            write_json(output / "stage1/config.json", spec)
            write_json(output / "stage1/splits.json", plan)
        if args.stage in ("stage1", "all"):
            if args.encoder_tuning["enabled"]:
                from script.dino_lora import train_lora

                train_lora(plan, args, bank, output)
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
            if args.encoder_tuning["enabled"]:
                from script.dino_lora import tuning_info

                tuning_info(output, args, bank)
            load_index(plan, bank, output)  # 先驗證 real 索引完整性，再讀取校準／測試圖片。
            if args.method == "cross_attention":
                from script.bank_attention import attention_info
                attention_info(output)
            extract_roles(plan, ("calibration", "evaluation"), bank, cache, args)
            evaluate(plan, args, bank, cache, output)
        if args.stage == "ablation":
            if args.encoder_tuning["enabled"]:
                from script.dino_lora import tuning_info

                tuning_info(output, args, bank)
            extract_roles(plan, ("calibration", "evaluation"), bank, cache, args)
            from script.bank_retrieval import evaluate_ablation
            evaluate_ablation(plan, args, bank, cache, output)
