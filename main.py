"""Command router for retrieval experiments and SegFace preprocessing."""

import argparse
import os


def main():
    os.environ["USE_TF"] = "0"
    os.environ["USE_TORCH"] = "1"
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--task", choices=("bank-retrieval", "segment-face"), default="bank-retrieval")
    task, remaining = selector.parse_known_args()
    if task.task == "bank-retrieval":
        from script.retrieval import main as run
    else:
        from script.segment_face import main as run
    run(remaining)


if __name__ == "__main__":
    main()
