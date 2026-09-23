#!/usr/bin/env bash
set -eo pipefail

task="${1:-crop-face}"
case "$task" in
  stage1|stage2|ablation|all) if (($#)); then shift; fi ;;
  crop-face|segment-face|patch-bank) if (($#)); then shift; fi ;;
  --help|-h)
    cat <<'HELP'
Usage:
  bash mission.sh stage1 [options]   # train LoRA and build the real feature bank
  bash mission.sh stage2 [options]   # calibrate and evaluate with the Stage 1 bank
  bash mission.sh ablation [options] # compare retrieval scoring variants
  bash mission.sh all [options]      # run Stage 1 followed by Stage 2
  bash mission.sh segment-face [options] # parallel SegFace preprocessing
  bash mission.sh crop-face [options]    # RetinaFace video preprocessing

Set retrieval.experiment and resources in utils/config.yaml first.
Example: bash mission.sh stage1 --gpus 4 5 6 --workers 16 --batch-size 8
The legacy patch-bank Python module is absent in this revision.
HELP
    exit 0 ;;
  --*) task=crop-face ;;  # Preserve original mission.sh --workers ... usage.
  *) echo "Unknown task: $task (use --help)" >&2; exit 2 ;;
esac

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ "$task" =~ ^(stage1|stage2|ablation|all)$ ]]; then
  exec bash run.sh "$task" "$@"
fi

case "$task" in
  crop-face) missing="script/crop_face.py" ;;
  segment-face) missing="script/mission.py" ;;
  patch-bank) missing="script/feature_bank.py" ;;
esac
if [[ ! -f "$missing" ]]; then
  echo "任務 '$task' 目前不可用：缺少 $missing；目前表徵實驗請使用 bash mission.sh stage1|stage2|all。" >&2
  exit 2
fi

conda_base="$(conda info --base)"
source "$conda_base/etc/profile.d/conda.sh"
conda activate pt230

if [[ "$task" == crop-face ]]; then
  exec python -u script/crop_face.py "$@"
fi
if [[ "$task" == patch-bank ]]; then
  exec python -u main.py --task patch-bank "$@"
fi

# Keep shell task dispatch simple; Python manages core allocation, GPU jobs and cleanup.
exec python -u -m script.mission "$@"
