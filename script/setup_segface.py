"""Install pinned official SegFace inference sources and weights under models/segface."""

import hashlib
import json
from pathlib import Path
import urllib.request

from huggingface_hub import hf_hub_download


CODE_REVISION = "9416628e7b5a3e4b9a1068b8467bdc0cf0be7a7d"
WEIGHT_REVISION = "5e093b03c0523f7f32a9845bbbc75ecb027c8bee"
WEIGHT_FILE = "swinb_celeba_512/model_299.pt"


def main():
    root = Path(__file__).resolve().parents[1] / "models/segface"
    source = root / "source"
    source.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/Kartik-3004/SegFace/{CODE_REVISION}/"
    for name, upstream in (("segface_celeb.py", "network/models/segface_celeb.py"),
                           ("transformer.py", "network/models/transformer.py"), ("LICENSE", "LICENSE")):
        content = urllib.request.urlopen(base + upstream, timeout=60).read().decode()
        if name == "segface_celeb.py":
            # Inference-only adaptations: package-local imports and no redundant ImageNet download.
            content = content.replace("sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))\n", "")
            content = content.replace("from network.models.transformer import *", "from .transformer import LayerNorm2d, TwoWayTransformer")
            content = content.replace("from network.models.utils_models import *\n", "")
            content = content.replace("swin_b(weights='IMAGENET1K_V1')", "swin_b(weights=None)")
        (source / name).write_text(content)
    (source / "__init__.py").write_text('"""Pinned SegFace inference implementation; see ../README.md."""\n')
    path = hf_hub_download("kartiknarayan/SegFace", filename=WEIGHT_FILE, revision=WEIGHT_REVISION, local_dir=root)
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    # Convert this pinned, author-published checkpoint once; runtime uses tensor-only safetensors.
    import torch
    from safetensors.torch import save_file

    converted = Path(path).with_name("model.safetensors")
    if not converted.exists():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        tensors = {key: value.contiguous() for key, value in checkpoint["state_dict_backbone"].items()}
        temporary = converted.with_suffix(".safetensors.tmp")
        save_file(tensors, temporary)
        temporary.replace(converted)
    (root / "provenance.json").write_text(json.dumps({"code_revision": CODE_REVISION,
        "weight_revision": WEIGHT_REVISION, "weight_file": WEIGHT_FILE, "weight_sha256": digest.hexdigest(),
        "inference_file": str(converted.relative_to(root))}, indent=2))
    print(f"SegFace is ready: {root}")


if __name__ == "__main__":
    main()
