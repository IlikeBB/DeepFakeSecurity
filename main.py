import argparse
from pathlib import Path

import yaml


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args(argv=None):
    root = Path(__file__).resolve().parent
    with (root / "utils/config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config.pop("crop_face", None)
    config.pop("feature_bank", None)
    config.pop("segment_face", None)
    config.pop("mission", None)
    parser = argparse.ArgumentParser(description="Extract DINOv3 features from Celeb-DF videos.")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-frames", type=positive_int)
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--device", help="cpu, cuda:0, cuda:1, etc.")
    parser.add_argument("--split", choices=["all", "train", "test"])
    parser.add_argument("--limit", type=positive_int, help="Maximum number of videos for a small trial.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction)
    fields = {action.dest for action in parser._actions if action.dest != "help"}
    if not isinstance(config, dict) or set(config) != fields:
        parser.error("utils/config.yaml must contain exactly: " + ", ".join(sorted(fields)))
    parser.set_defaults(**config)
    args = parser.parse_args(argv)
    for name in ("num_frames", "batch_size", "limit"):
        value = getattr(args, name)
        if name == "limit" and value is None:
            continue
        if type(value) is not int or value < 1:
            parser.error(f"{name} must be a positive integer")
    if args.split not in ("all", "train", "test"):
        parser.error("split must be all, train, or test")
    if not isinstance(args.overwrite, bool):
        parser.error("overwrite must be true or false")
    if not isinstance(args.device, str) or not args.device.strip():
        parser.error("device must be a non-empty string")
    for name in ("data_root", "model_path", "output_dir"):
        value = getattr(args, name)
        if not isinstance(value, (str, Path)) or not str(value).strip():
            parser.error(f"{name} must be a non-empty path")
        path = Path(value).expanduser()
        setattr(args, name, path if path.is_absolute() else root / path)
    return args


if __name__ == "__main__":
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--task", choices=("feature-bank", "segment-face", "extract"), default="feature-bank")
    task, remaining = selector.parse_known_args()
    if task.task == "segment-face":
        from script.segment_face import main

        main(remaining)
    elif task.task == "feature-bank":
        from script.feature_bank import main

        main(remaining)
    else:
        args = parse_args(remaining)
        from script.extract import extract

        extract(args)
