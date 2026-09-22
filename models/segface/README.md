# SegFace — AAAI 2025

Official Swin-B / CelebAMask-HQ / 512 checkpoint:
https://huggingface.co/kartiknarayan/SegFace

Paper: https://arxiv.org/abs/2412.08647

Code: https://github.com/Kartik-3004/SegFace

Run from the project root, using the existing environment:

```bash
conda run --no-capture-output -n pt230 python -m script.setup_segface
```

The setup script pins the upstream code and weight revisions, downloads the original checkpoint,
and converts its `state_dict_backbone` into `swinb_celeba_512/model.safetensors`.
Runtime strictly loads these tensor-only weights; no ImageNet weights or training state are needed.
`provenance.json` records the original checkpoint hash and revisions.

`source/` contains the official model and transformer implementation, with the upstream license.
The only inference adaptations are package-relative imports, removal of an unused utility import,
and `swin_b(weights=None)` to avoid downloading weights that the checkpoint replaces.
The model architecture and parameters are unchanged. `setup_segface.py` reproduces these changes.

Preprocessing follows the upstream test dataset: RGB, bicubic resize to 512×512, ImageNet normalization.
Output class IDs follow SegFace's CelebAMask-HQ ordering (different from SegFormer face parsing).
