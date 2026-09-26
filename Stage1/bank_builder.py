"""Stage 1 real-only patch-bank construction and artifact loading."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from script.bank_data import image_records, read_json, save_array, sha256, write_json
from script.experiment_paths import stage1_output
from script.retrieval_io import load_sample


def foreground_patches(features, image_path, minimum):
    """Estimate valid patch occupancy in a face crop."""
    h, w, _ = features.shape
    with Image.open(image_path) as image:
        mask = (np.asarray(image.convert("RGB")).max(axis=-1) > 16).astype(np.uint8) * 255
    occupancy = np.asarray(Image.fromarray(mask).resize((w, h), Image.Resampling.BOX), dtype=np.float32) / 255
    ids = np.flatnonzero(occupancy.reshape(-1) >= minimum)
    if not len(ids):
        raise ValueError(f"No foreground patches: {image_path}")
    return features.reshape(-1, features.shape[-1])[ids], ids, (h, w)


def build(plan, args, bank, output):
    stage1 = stage1_output(output)
    stage1.mkdir(parents=True, exist_ok=True)
    completion = stage1 / "retrieval.json"
    if completion.exists():
        load_index(plan, bank, output)
        print("Stage 1 已完成，沿用 real patch bank 索引。", flush=True)
        return
    rows = image_records(plan["groups"]["bank"], "bank")
    if not rows or any(row["label"] != 0 for row in rows):
        raise ValueError("Retrieval bank must contain real training images only")

    def read(row):
        _, features = load_sample(bank, row)
        return foreground_patches(features, row["image_path"], args.foreground_minimum)

    vectors, origins, patches = [], [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (values, ids, grid) in enumerate(tqdm(pool.map(read, rows), total=len(rows),
                                                   desc="Stage 1：建立 real 檢索索引", unit="image")):
            rows[i]["grid"] = list(grid)
            vectors.append(values)
            origins.append(np.full(len(ids), i, dtype=np.int32))
            patches.append(ids.astype(np.int32))
    directory = bank / "retrieval"
    directory.mkdir(parents=True, exist_ok=True)
    files = {"features": directory / "features.npy", "origins": directory / "origins.npy",
             "patch_ids": directory / "patch_ids.npy", "sources": stage1 / "sources.json"}
    for key, values in (("features", vectors), ("origins", origins), ("patch_ids", patches)):
        save_array(files[key], np.concatenate(values))
    write_json(files["sources"], rows)
    write_json(completion, {
        "method": f"real {args.face_source} face patches; exact cosine 1-NN; highest-distance patch mean",
        "image_count": len(rows), "patch_count": sum(len(values) for values in vectors),
        "foreground_minimum": args.foreground_minimum, "top_fraction": args.top_fraction,
        "plan_sha256": sha256(bank / "splits.json"),
        "source_config_sha256": sha256(bank / "bank_config.json"),
        "files": {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in files.items()},
    })


def load_index(plan, bank, output):
    info = read_json(stage1_output(output) / "retrieval.json")
    if "alignment" in info:
        raise ValueError("此索引使用已移除的文字對齊分支；請另取 experiment 名稱，重新建立原始 DINO bank")
    if (info["plan_sha256"] != sha256(bank / "splits.json")
            or info["source_config_sha256"] != sha256(bank / "bank_config.json")):
        raise ValueError("Retrieval index does not match source configuration/split")
    for item in tqdm(info["files"].values(), desc="驗證 bank 檔案", unit="file", dynamic_ncols=True):
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Retrieval index changed: {item['path']}")
    rows = read_json(info["files"]["sources"]["path"])
    expected = image_records(plan["groups"]["bank"], "bank")
    if [{k: v for k, v in row.items() if k != "grid"} for row in rows] != expected:
        raise ValueError("Bank contains unexpected sources")
    arrays = {key: np.load(info["files"][key]["path"], mmap_mode="r", allow_pickle=False)
              for key in ("features", "origins", "patch_ids")}
    return info, rows, arrays
