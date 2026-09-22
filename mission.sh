#!/usr/bin/env bash
set -eo pipefail

task="${1:-crop-face}"
case "$task" in
  crop-face|segment-face) if (($#)); then shift; fi ;;
  --help|-h)
    cat <<'HELP'
Usage:
  bash mission.sh segment-face [options]  # configurable CPU/GPU parallel jobs
  bash mission.sh crop-face [options]     # original RetinaFace worker pool

Segmentation defaults: parts 0-10, up to 10 frames/video, JPG only.
Resources: --cores 8 --gpus 1,2 (two GPUs), or --gpus none --cpu-workers 4.
Use --dry-run to inspect allocation without processing images.
GPU assignments and data paths: utils/config.yaml.
Example: bash mission.sh segment-face --limit 1
--limit applies per class before distributing videos across workers.
--parts and --frames-per-video override YAML.
HELP
    exit 0 ;;
  --*) task=crop-face ;;  # Preserve original mission.sh --workers ... usage.
  *) echo "Unknown task: $task (use --help)" >&2; exit 2 ;;
esac

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
conda_base="$(conda info --base)"
source "$conda_base/etc/profile.d/conda.sh"
conda activate pt230

if [[ "$task" == crop-face ]]; then
  exec python -u script/crop_face.py "$@"
fi

# Keep shell task dispatch simple; Python manages core allocation, GPU jobs and cleanup.
exec python -u -m script.mission "$@"
