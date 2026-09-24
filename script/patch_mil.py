"""Train and evaluate a patch-relation MIL classifier on cached DINO features."""

import argparse
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
from PIL import Image
from safetensors.torch import load_file, save_file
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm
import yaml

from models.patch_mil import PatchRelationMIL
from script.bank_data import image_records, read_json, sha256, write_json
from script.retrieval_io import extract_roles, metrics


def parse_args(argv=None):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "utils/config.yaml").read_text(encoding="utf-8"))["patch_mil"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cache", "train", "evaluate", "all"), default="train")
    parser.add_argument("--exper", "--experiment", dest="experiment")
    parser.add_argument("--gpus", dest="gpu_ids", type=int, nargs="*",
                        help="Cache extraction uses all listed GPUs; classifier training uses the first")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--replace", action="store_true", help="Replace an existing PatchRelationMIL checkpoint")
    parser.set_defaults(**config)
    args = parser.parse_args(argv)
    if not isinstance(args.experiment, str) or not args.experiment.strip():
        parser.error("請以 --exper 指定已完成 Stage 1 的 feature-bank 實驗名稱")
    if args.gpu_ids is None:
        args.gpu_ids = list(config["gpu_ids"])
    required = (args.batch_size, args.workers, args.epochs, args.images_per_epoch,
                args.validation_images, args.hidden_dim, args.heads, args.layers)
    if any(value < 1 for value in required):
        parser.error("batch、workers、epochs、樣本數與模型維度都必須大於 0")
    if args.hidden_dim % args.heads or not 0 < args.dropout < 1 or not 0 < args.top_fraction <= 1:
        parser.error("hidden_dim 必須可被 heads 整除；dropout/top_fraction 範圍不正確")
    if not 0 < args.validation_fraction < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("validation_fraction、learning_rate 或 weight_decay 不正確")
    if any(gpu < 0 or gpu >= torch.cuda.device_count() for gpu in args.gpu_ids):
        parser.error("--gpus 包含不存在的 GPU")
    args.devices = [f"cuda:{gpu}" for gpu in args.gpu_ids]
    args.device = args.devices[0] if args.devices else "cpu"
    for name in ("bank_dir", "source_results_dir", "model_dir", "results_dir"):
        path = Path(getattr(args, name)).expanduser()
        setattr(args, name, str((root / path).resolve() if not path.is_absolute() else path.resolve()))
    return args


def _paths(args):
    bank = Path(args.bank_dir) / args.experiment
    output = Path(args.results_dir) / args.experiment
    model_dir = Path(args.model_dir) / args.experiment
    return bank, bank / "cache", output, model_dir


def _source_spec(bank):
    spec = read_json(bank / "bank_config.json")
    if spec.get("schema") != "retrieval-v1":
        raise ValueError("PatchRelationMIL 需要 retrieval-v1 Stage 1 feature bank")
    return spec


def _extract_args(args, spec):
    tuning = spec["encoder_tuning"]
    checkpoint = None
    if tuning["enabled"]:
        info = read_json(Path(args.source_results_dir) / args.experiment / "stage1/dino_lora.json")
        checkpoint = info["checkpoint"]
        if info["config"] != tuning or not Path(checkpoint).is_file():
            raise ValueError("Stage 1 LoRA checkpoint 與 feature bank 不一致")
    return SimpleNamespace(devices=args.devices, workers=args.workers,
                           batch_size=args.extract_batch_size, dtype=spec["dtype"],
                           model_path=spec["model_path"], encoder_tuning=tuning, encoder_checkpoint=checkpoint)


def ensure_cache(plan, args, bank, cache, roles):
    """Cache requested non-bank roles; existing entries are checked and reused."""
    rows = [row for role in roles for row in image_records(plan["groups"][role], role)]
    missing = sum(not (cache / row["role"] / row["feature_path"]).is_file()
                  or not (cache / row["role"] / row["feature_path"]).with_suffix(".json").is_file()
                  for row in rows)
    if missing:
        print(f"PatchRelationMIL：需要快取 {missing}/{len(rows)} 張 DINO features", flush=True)
        extract_roles(plan, roles, bank, cache, _extract_args(args, _source_spec(bank)))
    else:
        print(f"PatchRelationMIL：沿用 {len(rows)} 張 feature cache", flush=True)


def _split_by_family(rows, fraction, seed):
    families = sorted({row["group_id"] for row in rows})
    if len(families) < 4:
        raise ValueError("PatchRelationMIL 至少需要四個 source families")
    rng = random.Random(seed)
    for _ in range(32):
        shuffled = list(families)
        random.Random(rng.randrange(1 << 30)).shuffle(shuffled)
        count = min(len(families) - 2, max(2, round(len(families) * fraction)))
        validation = set(shuffled[:count])
        train = [row for row in rows if row["group_id"] not in validation]
        valid = [row for row in rows if row["group_id"] in validation]
        if {row["label"] for row in train} == {0, 1} and {row["label"] for row in valid} == {0, 1}:
            return train, valid
    raise ValueError("無法建立同時含 real/fake 的 family-disjoint validation split")


def _limit_per_class(rows, limit, seed):
    selected = []
    for label in (0, 1):
        values = [row for row in rows if row["label"] == label]
        random.Random(seed + label).shuffle(values)
        selected.extend(values[:min(len(values), limit)])
    return selected


def _feature_path(bank, cache, row):
    root = bank if row["role"] == "bank" else cache / row["role"]
    return root / row["feature_path"]


class PatchDataset(Dataset):
    def __init__(self, rows, bank, cache, grid_size):
        self.rows, self.bank, self.cache, self.grid_size = rows, bank, cache, tuple(grid_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        feature = np.load(_feature_path(self.bank, self.cache, row), allow_pickle=False)
        if feature.ndim != 3 or tuple(feature.shape[:2]) != self.grid_size or not np.isfinite(feature).all():
            raise ValueError(f"Invalid patch feature: {row['feature_path']}")
        with Image.open(row["image_path"]) as image:
            pixels = np.asarray(image.convert("RGB"))
        foreground = pixels.max(axis=-1) > 16
        foreground = np.asarray(Image.fromarray(foreground.astype(np.uint8) * 255).resize(
            (self.grid_size[1], self.grid_size[0]), Image.Resampling.BOX)) > 127
        if not foreground.any():
            raise ValueError(f"No foreground patch: {row['image_path']}")
        return (torch.from_numpy(feature.astype(np.float32, copy=True)), torch.from_numpy(foreground),
                torch.tensor(row["label"], dtype=torch.float32), index)


def _loader(dataset, args, shuffle=False, balanced=False, samples=None):
    sampler = None
    if balanced:
        labels = np.array([row["label"] for row in dataset.rows])
        counts = np.bincount(labels, minlength=2)
        weights = torch.as_tensor([1 / counts[label] for label in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=samples, replacement=True)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle if sampler is None else False, sampler=sampler,
                      num_workers=args.workers, pin_memory=args.device != "cpu", persistent_workers=args.workers > 0)


def _model(args, input_dim, grid_size):
    return PatchRelationMIL(input_dim=input_dim, hidden_dim=args.hidden_dim, heads=args.heads, layers=args.layers,
                            dropout=args.dropout, top_fraction=args.top_fraction, grid_size=grid_size)


def _run_epoch(model, loader, optimizer, device, patch_weight=0., train=False):
    logits, labels, total, count = [], [], 0., 0
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    context = torch.enable_grad if train else torch.no_grad
    with context():
        for features, foreground, target, _ in tqdm(
                loader, desc="PatchRelationMIL 訓練" if train else "PatchRelationMIL 驗證",
                unit="image", dynamic_ncols=True, leave=False):
            features = features.to(device, non_blocking=True)
            foreground = foreground.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = model(features, foreground)
                image_loss = F.binary_cross_entropy_with_logits(output["image_logit"], target)
                real = target == 0
                local_loss = (F.softplus(output["patch_logits"][real][foreground[real]]).mean()
                              if real.any() else image_loss.new_zeros(()))
                loss = image_loss + patch_weight * local_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
            total += float(loss.detach()) * len(target)
            count += len(target)
            logits.extend(output["image_logit"].detach().float().cpu().tolist())
            labels.extend(target.detach().cpu().tolist())
    probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
    return {"loss": total / max(1, count), "auroc": float(roc_auc_score(labels, probabilities)),
            "average_precision": float(average_precision_score(labels, probabilities))}


def _model_info(args, bank, model_dir, input_dim, grid_size):
    return {"method": "frozen DINO patch features + pairwise relation statistics + Transformer attention/top-k MIL",
            "experiment": args.experiment, "input_dim": input_dim, "grid_size": list(grid_size),
            "hidden_dim": args.hidden_dim, "heads": args.heads, "layers": args.layers, "dropout": args.dropout,
            "top_fraction": args.top_fraction, "plan_sha256": sha256(bank / "splits.json"),
            "bank_config_sha256": sha256(bank / "bank_config.json"),
            "checkpoint": str((model_dir / "model.safetensors").resolve())}


def train(plan, args, bank, cache, output, model_dir):
    checkpoint, metadata = model_dir / "model.safetensors", model_dir / "model.json"
    if checkpoint.exists() and not args.replace:
        raise ValueError(f"模型已存在：{checkpoint}；若要重新訓練請加 --replace")
    real = image_records(plan["groups"]["bank"], "bank")
    fake = image_records(plan["groups"]["train_fake"], "train_fake")
    train_rows, valid_rows = _split_by_family(real + fake, args.validation_fraction, args.seed)
    valid_rows = _limit_per_class(valid_rows, math.ceil(args.validation_images / 2), args.seed)
    first = PatchDataset(train_rows[:1], bank, cache, (14, 14))[0]
    grid_size, input_dim = first[0].shape[:2], first[0].shape[-1]
    model = _model(args, input_dim, grid_size).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_set = PatchDataset(train_rows, bank, cache, grid_size)
    valid_set = PatchDataset(valid_rows, bank, cache, grid_size)
    real_count = sum(row["label"] == 0 for row in train_rows)
    fake_count = sum(row["label"] == 1 for row in train_rows)
    per_epoch = min(args.images_per_epoch, max(2, 2 * min(real_count, fake_count)))
    train_loader = _loader(train_set, args, balanced=True, samples=per_epoch)
    valid_loader = _loader(valid_set, args)
    model_dir.mkdir(parents=True, exist_ok=True)
    history, best, stale = [], -float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        training = _run_epoch(model, train_loader, optimizer, torch.device(args.device), args.real_patch_weight, train=True)
        model.eval()
        validation = _run_epoch(model, valid_loader, None, torch.device(args.device), train=False)
        row = {"epoch": epoch, "train": training, "validation": validation}
        history.append(row)
        tqdm.write(f"PatchRelationMIL {epoch}/{args.epochs}: train_auc={training['auroc']:.4f} "
                   f"val_auc={validation['auroc']:.4f} val_ap={validation['average_precision']:.4f}")
        if validation["auroc"] > best + args.minimum_delta:
            best, stale = validation["auroc"], 0
            save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}, str(checkpoint))
            best_epoch = epoch
        else:
            stale += 1
        write_json(output / "train_history.json", history)
        if stale >= args.patience:
            break
    info = _model_info(args, bank, model_dir, input_dim, grid_size) | {
        "best_epoch": best_epoch, "best_validation_auroc": best, "epochs_completed": len(history),
        "train_images": len(train_rows), "validation_images": len(valid_rows),
        "train_families": len({row["group_id"] for row in train_rows}),
        "validation_families": len({row["group_id"] for row in valid_rows}),
    }
    write_json(metadata, info)
    return info


def _load_model(args, bank, model_dir):
    info = read_json(model_dir / "model.json")
    if (info["experiment"] != args.experiment or info["plan_sha256"] != sha256(bank / "splits.json")
            or info["bank_config_sha256"] != sha256(bank / "bank_config.json")):
        raise ValueError("PatchRelationMIL checkpoint 不屬於目前的 feature-bank split")
    model = PatchRelationMIL(input_dim=info["input_dim"], hidden_dim=info["hidden_dim"], heads=info["heads"],
                             layers=info["layers"], dropout=info["dropout"], top_fraction=info["top_fraction"],
                             grid_size=tuple(info["grid_size"])).to(args.device)
    model.load_state_dict(load_file(str(model_dir / "model.safetensors"), device=args.device))
    return model.eval(), info


def evaluate(plan, args, bank, cache, output, model_dir):
    model, info = _load_model(args, bank, model_dir)
    results = {}
    for role in ("calibration", "evaluation"):
        rows = image_records(plan["groups"][role], role)
        loader = _loader(PatchDataset(rows, bank, cache, tuple(info["grid_size"])), args)
        scored = []
        with torch.no_grad():
            for features, foreground, _, indices in tqdm(loader, desc=f"PatchRelationMIL {role}",
                                                          unit="image", dynamic_ncols=True):
                values = model(features.to(args.device), foreground.to(args.device))
                for index, score in zip(indices.tolist(), torch.sigmoid(values["image_logit"]).cpu().tolist()):
                    scored.append(dict(rows[index], score=float(score)))
        results[role] = sorted(scored, key=lambda row: row["sample_id"])
    calibration, tested = results["calibration"], results["evaluation"]
    threshold = float(np.quantile([row["score"] for row in calibration], args.threshold_quantile))
    for row in calibration + tested:
        row["prediction"] = "anomaly" if row["score"] > threshold else "normal"
    stage = output / "stage2"
    write_json(stage / "calibration_scores.json", calibration)
    write_json(stage / "evaluation_scores.json", tested)
    write_json(stage / "thresholds.json", {"threshold": threshold, "quantile": args.threshold_quantile,
                                             "calibration_count": len(calibration), "model": info["checkpoint"]})
    report = {"method": info["method"], "protocol": plan["protocol"], "image": metrics(tested, threshold),
              "checkpoint": info["checkpoint"]}
    write_json(stage / "metrics.json", report)
    image = report["image"]
    print(f"PatchRelationMIL Stage 2 完成：AUROC={image['auroc']:.4f} AP={image['average_precision']:.4f} "
          f"FPR={image['false_positive_rate']:.4f} TPR={image['true_positive_rate']:.4f}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    bank, cache, output, model_dir = _paths(args)
    if not (bank / "splits.json").is_file() or not (bank / "bank_config.json").is_file():
        raise ValueError("請先完成對應實驗的 Stage 1 feature bank")
    plan = read_json(bank / "splits.json")
    if "train_fake" not in plan["groups"]:
        raise ValueError("這個 feature bank 沒有 train_fake split，無法訓練 supervised PatchRelationMIL")
    if args.stage in ("cache", "train", "all"):
        ensure_cache(plan, args, bank, cache, ("train_fake",))
    if args.stage in ("train", "all"):
        train(plan, args, bank, cache, output, model_dir)
    if args.stage in ("evaluate", "all"):
        ensure_cache(plan, args, bank, cache, ("calibration", "evaluation"))
        evaluate(plan, args, bank, cache, output, model_dir)
