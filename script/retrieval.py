"""Stage 1 real feature bank; Stage 2 exact cosine retrieval and held-out evaluation."""

import argparse
from contextlib import ExitStack
import math
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits
import yaml

from Stage1.bank_builder import build, load_index
from Stage2.evaluation import evaluate
from script.bank_data import prepare_plan, read_json, sha256, write_json
from script.experiment_paths import stage1_output
from script.retrieval_io import ensure_experiment_config, experiment_lock, extract_roles


def parse_args(argv=None):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "utils/config.yaml").read_text(encoding="utf-8"))
    parser_config = config["segment_face"]
    defaults = dict(config["retrieval"], model_path=config["model_path"],
                    faces_dir=config["crop_face"]["output_dir"],
                    segface_normal_dir=config["segment_face"]["normal_output_dir"],
                    segface_anomaly_dir=config["segment_face"]["anomaly_output_dir"],
                    source_root=config["crop_face"]["data_root"],
                    label_csv=config["crop_face"]["label_csv"],
                    face_parser={key: parser_config[key] for key in
                                 ("model_dir", "checkpoint", "pixel_confidence", "face_ids", "background_ids",
                                  "dilation_ratio", "closing_ratio", "min_face_area")})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "stage1", "cluster", "stage2", "ablation", "all"),
                        default="stage1")
    parser.add_argument("--exper", "--experiment", dest="experiment")
    parser.add_argument("--gpus", dest="gpu_ids", type=int, nargs="*", help="GPU IDs; empty list uses CPU")
    parser.add_argument("--batch-size", "--extract-batch-size", dest="batch_size", type=int)
    parser.add_argument("--face-source", choices=("retinaface", "segface"))
    parser.add_argument("--tune-encoder", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--unfreeze-blocks", type=int)
    parser.add_argument("--method", choices=("nearest", "topk"), default="nearest")
    parser.add_argument("--cluster-samples", type=int, default=500,
                        help="Stage 1 diagnostic: paired source families to sample")
    parser.add_argument("--cluster-count", type=int, default=8,
                        help="Stage 1 diagnostic: number of K-Means clusters")
    for key in ("workers", "cpu_threads", "query_chunk_size", "bank_chunk_size", "match_count"):
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.tune_encoder is not None or args.unfreeze_blocks is not None:
        args.encoder_tuning = dict(args.encoder_tuning)
    if args.tune_encoder is not None:
        args.encoder_tuning["enabled"] = args.tune_encoder
    if args.unfreeze_blocks is not None:
        args.encoder_tuning["unfreeze_blocks"] = args.unfreeze_blocks
    del args.tune_encoder
    del args.unfreeze_blocks
    if (not isinstance(args.experiment, str) or not args.experiment.strip()
            or args.experiment in (".", "..") or Path(args.experiment).name != args.experiment):
        parser.error("請先在 utils/config.yaml 的 retrieval.experiment 填入你取的實驗名稱，或使用 --exper 指定")
    if type(args.export_previews) is not bool:
        parser.error("utils/config.yaml 的 retrieval.export_previews 必須是 true 或 false")
    if min(args.workers, args.cpu_threads, args.batch_size, args.query_chunk_size, args.bank_chunk_size,
           args.match_count, args.pca_images, args.cluster_samples, args.cluster_count) < 1:
        parser.error("執行緒、batch 與 chunk 參數必須大於 0")
    if args.cluster_samples < 2 or args.cluster_count < 2:
        parser.error("--cluster-samples 與 --cluster-count 必須至少為 2")
    if (not 0 < args.foreground_minimum <= 1 or not 0 < args.top_fraction <= 1
            or not 0 < args.threshold_quantile < 1 or args.dtype not in ("float16", "float32")
            or not 1 <= args.jpeg_quality <= 100):
        parser.error("Invalid retrieval configuration")
    if (len(set(args.gpu_ids)) != len(args.gpu_ids)
            or any(gpu < 0 or gpu >= torch.cuda.device_count() for gpu in args.gpu_ids)):
        parser.error("--gpus 請指定可用且不重複的 GPU 編號；空的 --gpus 使用 CPU")
    args.devices = [f"cuda:{gpu}" for gpu in args.gpu_ids]
    for key in ("bank_dir", "stage2_cache_dir", "results_dir", "model_path", "faces_dir", "segface_normal_dir",
                "segface_anomaly_dir", "source_root", "label_csv"):
        setattr(args, key, str((root / Path(getattr(args, key)).expanduser()).resolve()))
    args.face_parser = dict(args.face_parser)
    args.face_parser["model_dir"] = str((root / Path(args.face_parser["model_dir"]).expanduser()).resolve())
    if args.face_source == "segface":
        for key in ("segface_normal_dir", "segface_anomaly_dir"):
            if not Path(getattr(args, key)).is_dir():
                parser.error(f"SegFace input directory does not exist: {getattr(args, key)}")
    destinations = (Path(args.bank_dir), Path(args.stage2_cache_dir), Path(args.results_dir))
    for destination in destinations:
        for name in ("faces_dir", "segface_normal_dir", "segface_anomaly_dir", "source_root"):
            source = Path(getattr(args, name))
            if destination == source or source in destination.parents:
                parser.error("輸出不能放入來源資料集")
    if len(set(destinations)) != len(destinations):
        parser.error("Stage 1 特徵、Stage 2 查詢快取與評估結果目錄必須分開")
    topk = args.topk
    if (type(topk.get("neighbors")) is not int or topk["neighbors"] < 1
            or not math.isfinite(topk.get("temperature", 0)) or topk["temperature"] <= 0):
        parser.error("Invalid retrieval.topk configuration")
    ablation = args.ablation
    if (type(ablation["max_per_family"]) is not int or ablation["max_per_family"] < 1
            or not math.isfinite(ablation["boundary_weight"])
            or not 0 <= ablation["boundary_weight"] <= 1):
        parser.error("Invalid retrieval.ablation configuration")
    tuning = args.encoder_tuning
    fsfm = tuning.get("fsfm", {})
    integers = ("unfreeze_blocks", "epochs", "patience", "batch_size", "images_per_epoch", "validation_images")
    positive = ("backbone_learning_rate", "decoder_learning_rate", "ema_weight", "gradient_clip")
    if (type(tuning.get("enabled")) is not bool
            or any(type(tuning.get(key)) is not int or tuning[key] < 1 for key in integers)
            or any(not math.isfinite(tuning.get(key, 0)) or tuning[key] <= 0 for key in positive)
            or not math.isfinite(tuning.get("weight_decay", -1)) or tuning["weight_decay"] < 0
            or not math.isfinite(tuning.get("minimum_delta", -1)) or tuning["minimum_delta"] < 0
            or not 0 < tuning.get("ema_start", 0) <= tuning.get("ema_end", 0) <= 1
            or not 0 < tuning.get("validation_fraction", 0) < 1):
        parser.error("Invalid retrieval.encoder_tuning configuration")
    fsfm_integers = ("image_size", "grid_size", "parser_batch_size", "decoder_layers", "decoder_heads")
    if (any(type(fsfm.get(key)) is not int or fsfm[key] < 1 for key in fsfm_integers)
            or fsfm.get("image_size") != 224 or fsfm.get("grid_size") != 14
            or not math.isfinite(fsfm.get("decoder_mlp_ratio", 0)) or fsfm["decoder_mlp_ratio"] <= 0
            or not math.isfinite(fsfm.get("decoder_dropout", -1)) or not 0 <= fsfm["decoder_dropout"] < 1
            or any(not math.isfinite(fsfm.get(key, -1)) or not 0 < fsfm[key] < 1
                   for key in ("mask_ratio", "foreground_minimum"))
            or any(not math.isfinite(fsfm.get(key, -1)) or fsfm[key] < 0
                   for key in ("region_weight", "global_weight"))
            or fsfm["image_size"] % fsfm["grid_size"]
            or 768 % fsfm["decoder_heads"]
            or fsfm["decoder_heads"] > 64):
        parser.error("Invalid retrieval.encoder_tuning.fsfm configuration")
    if tuning["enabled"] and not 2 <= tuning["unfreeze_blocks"] <= 4:
        parser.error("Partial fine-tuning requires encoder_tuning.unfreeze_blocks between 2 and 4")
    if tuning["enabled"] and args.face_source != "segface":
        parser.error("Encoder tuning requires --face-source segface so region maps align with bank images")
    face_parser = args.face_parser
    if (not 0 < face_parser.get("pixel_confidence", 0) <= 1
            or not face_parser.get("face_ids")
            or set(face_parser["face_ids"]) & set(face_parser.get("background_ids", []))
            or any(not 0 <= face_parser.get(key, -1) <= .5 for key in ("dilation_ratio", "closing_ratio"))
            or not 0 < face_parser.get("min_face_area", 0) <= 1):
        parser.error("Invalid SegFace configuration for FSFM region parsing")
    return args


def main(argv=None):
    args = parse_args(argv)
    bank = Path(args.bank_dir) / args.experiment
    cache = Path(args.stage2_cache_dir) / args.experiment
    output = Path(args.results_dir) / args.experiment
    stage1 = stage1_output(output)
    completion = stage1 / "retrieval.json"
    if args.stage in ("cluster", "stage2", "ablation") and not completion.is_file():
        raise ValueError(f"請先執行 bash run.sh stage1 --exper {args.experiment}；找不到 {completion}")
    runtime = {"stage", "gpu_ids", "devices", "workers", "cpu_threads", "batch_size", "query_chunk_size",
               "bank_chunk_size", "export_previews", "ablation", "encoder_checkpoint", "cluster_samples",
               "cluster_count"}
    spec = {key: value for key, value in vars(args).items() if key not in runtime}
    use_fsfm = args.encoder_tuning["enabled"]
    if not use_fsfm:
        spec.pop("face_parser")
    model_files = {f"dinov3/{name}": Path(args.model_path) / name
                   for name in ("model.safetensors", "config.json", "preprocessor_config.json")}
    if use_fsfm:
        model_files["segface/model.safetensors"] = (Path(args.face_parser["model_dir"])
                                                     / args.face_parser["checkpoint"])
    missing = [str(path) for path in model_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少本地模型檔案：" + ", ".join(missing))
    spec.update(schema="retrieval-v4-fsfm", method=getattr(args, "method", "nearest"),
                files_sha256={name: sha256(path) for name, path in model_files.items()},
                labels_sha256=sha256(args.label_csv))
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    with ExitStack() as stack:
        stack.enter_context(experiment_lock(output))
        stack.enter_context(experiment_lock(bank))
        if args.stage in ("stage2", "ablation", "all"):
            stack.enter_context(experiment_lock(cache))
        stack.enter_context(threadpool_limits(limits=args.cpu_threads))
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
            write_json(stage1 / "config.json", spec)
            write_json(stage1 / "splits.json", plan)
        if args.stage in ("stage1", "all"):
            if args.encoder_tuning["enabled"]:
                from Stage1.encoder_tuning import train_encoder

                train_encoder(plan, args, bank, output)
            if not completion.exists():
                extract_roles(plan, ("bank",), bank, cache, args)
            if args.export_previews:
                from script.bank_export import build_bank
                build_bank(plan, args, bank)  # 選用：以既有 NPY 補齊 PCA／JPG 預覽，不影響索引。
            build(plan, args, bank, output)
        if args.stage == "cluster":
            if args.encoder_tuning["enabled"]:
                from Stage1.encoder_tuning import tuning_info

                tuning_info(output, args, bank)
            from Stage1.feature_clustering import analyze
            analyze(plan, args, bank, output)
        if args.stage in ("stage2", "all"):
            if args.encoder_tuning["enabled"]:
                from Stage1.encoder_tuning import tuning_info

                tuning_info(output, args, bank)
            _, _, arrays = load_index(plan, bank, output)  # 先驗證 real 索引完整性與顯存。
            from Stage2.bank_search import check_device_memory
            check_device_memory(arrays["features"], args.devices, args.query_chunk_size, args.bank_chunk_size)
            extract_roles(plan, ("calibration", "evaluation"), bank, cache, args)
            evaluate(plan, args, bank, cache, output)
        if args.stage == "ablation":
            if args.encoder_tuning["enabled"]:
                from Stage1.encoder_tuning import tuning_info

                tuning_info(output, args, bank)
            extract_roles(plan, ("calibration", "evaluation"), bank, cache, args)
            from Stage2.evaluation import evaluate_ablation
            evaluate_ablation(plan, args, bank, cache, output)
