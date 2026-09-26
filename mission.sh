#!/usr/bin/env bash
set -eo pipefail

# Real-Only DINOv3 PatchBank 的人臉前處理與兩階段實驗入口。
#
# 查看說明
#   cd /ssd8/chihyu/Project/Real-Only-DINOv3-PatchBank-for-Deepfake-Detection
#   bash mission.sh --help
#
# 少量資料執行範例：兩支影片、每支四幀
#   bash mission.sh crop-face --limit 2 --num-frames 4 --gpus 5
#
# 完整 DFDC test.csv：不加 --limit，每支影片最多抽 32 幀
#   bash mission.sh crop-face --num-frames 32
#
# 多 GPU：GPU 編號使用逗號分隔，每張卡固定一個 RetinaFace 程序
#   bash mission.sh crop-face --num-frames 32 --gpus 4,5,6
#
# CPU 模式
#   bash mission.sh crop-face --num-frames 32 --gpus none --device cpu --workers 8 --cpu-threads 2
#
# 已完成且設定一致的影片會直接跳過；不完整輸出會自動重建。
# 只有在抽幀、門檻或裁切設定改變，且確定全部重做時才加入 --overwrite。
# 輸出：/ssd8/chihyu/Dataset/DeepFake_Dataset/DFDC-Frame-Face/<part>/<video>/
#
# Feature bank：Stage 1 依 face_source 讀取 RetinaFace 或 SegFace JPG，並沿用 metadata。
#   bash mission.sh stage1
#   bash mission.sh stage2
#   bash mission.sh all
#   bash mission.sh ablation

task="${1:-crop-face}"
case "$task" in
  stage1|stage2|ablation|all) if (($#)); then shift; fi ;;
  crop-face|segment-face) if (($#)); then shift; fi ;;
  --help|-h)
    cat <<'HELP'
使用方式：
  bash mission.sh crop-face [參數]  # RetinaFace 抽幀與人臉裁切
  bash mission.sh segment-face [參數] # SegFace 語意遮罩與去背景裁切
  bash mission.sh stage1 [參數]     # 建立 real feature bank
  bash mission.sh stage2 [參數]     # 校準與評估
  bash mission.sh all [參數]        # 依序完成 Stage 1、Stage 2
  bash mission.sh ablation [參數]   # 固定檢索消融

RetinaFace 多 GPU：--gpus 4,5,6（逗號分隔，每張 GPU 一個程序）
RetinaFace CPU：--gpus none --device cpu --workers 8
完整資料：bash mission.sh crop-face --num-frames 32
推論批次：--detection-batch-size 2（顯存不足時改為 1）
SegFace 多 GPU：bash mission.sh segment-face --cores 24 --gpus 0,1,2,3,5,6
已完成影片預設跳過；不要在續跑時加入 --overwrite。
設定：utils/config.yaml 的 crop_face／segment_face；輸出位置分別由各區塊設定。
HELP
    exit 0 ;;
  --*) task=crop-face ;;
  *) echo "未知任務：$task（請使用 bash mission.sh --help）" >&2; exit 2 ;;
esac

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ "$task" =~ ^(stage1|stage2|ablation|all)$ ]]; then
  exec bash run.sh "$task" "$@"
fi

conda_base="$(conda info --base)"
source "$conda_base/etc/profile.d/conda.sh"
conda activate pt230

if [[ "$task" == segment-face ]]; then
  exec python -u -m script.mission "$@"
fi
exec python -u script/crop_face.py "$@"
