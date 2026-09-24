#!/usr/bin/env bash
# Celeb-DF 官方 test：只做 RetinaFace 與 SegFace，不抽 DINO 特徵或訓練模型。
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

task="${1:-all}"
if (($#)); then shift; fi
case "$task" in
  crop-face|segment-face|all) ;;
  --help|-h)
    cat <<'HELP'
使用方式：
  bash mission_Celeb-df.sh all           # 依序裁切、分割；預設 GPU 3
  bash mission_Celeb-df.sh crop-face --gpus 3
  bash mission_Celeb-df.sh segment-face --device cuda:3

範圍：Celeb-DF 官方 test 清單，每影片最多 32 幀。
原始影片：/ssd2/DeepFakes/celeb-df-video（不修改）
裁切：/ssd8/chihyu/Dataset/DeepFake_Dataset/Celeb-df-Frame-Face
real 分割：/ssd8/chihyu/Dataset/DeepFake_Dataset/Celeb-df-Frame-Face-nomral
fake 分割：/ssd8/chihyu/Dataset/DeepFake_Dataset/Celeb-df-Frame-Face-anomaly
清單與統計：/ssd8/chihyu/Dataset/DeepFake_Dataset/Celeb-df-external
分割檔名：來源類別-影片名稱-frame_XXXXXX.jpg（與 DFDC 相同的扁平結構）

all 不接受額外參數；自訂 GPU／批次時，請分別執行兩個子命令。
已完成且設定相同的資料會重用。此入口不提供模型訓練或評估任務。
HELP
    exit 0 ;;
  *) echo "未知資料任務：$task（請使用 --help）" >&2; exit 2 ;;
esac
if [[ "$task" == all ]] && (($#)); then
  echo 'all 不接受額外參數；請分別使用 crop-face 或 segment-face。' >&2
  exit 2
fi

conda_base="$(conda info --base)"
source "$conda_base/etc/profile.d/conda.sh"
conda activate pt230

crop_defaults=(--data-root /ssd2/DeepFakes/celeb-df-video --label-csv '' --split test
  --output-dir /ssd8/chihyu/Dataset/DeepFake_Dataset/Celeb-df-Frame-Face
  --num-frames 32 --gpus 3)
if [[ "$task" == crop-face ]]; then
  exec python -u script/crop_face.py "${crop_defaults[@]}" "$@"
fi
if [[ "$task" == all ]]; then
  python -u script/crop_face.py "${crop_defaults[@]}"
fi
exec python -u -m script.reconstruction_external "$@" --stage segment
