#!/usr/bin/env bash
set -eo pipefail

# 預設實驗：adapter（real 為主、間歇加入 fake）。可改成 real-only 或 baseline。
# 資料路徑、GPU、epoch、fake 間隔與權重仍集中在 utils/config.yaml 的 patch_bank。
#
# 使用案例（以下都是註解，不會自動執行；請複製需要的指令到終端機）
# 請先切換至專案目錄：cd /ssd8/chihyu/Project/DeepFakeSecurity
#
# 案例 1：依 YAML 預設，執行 real 為主、間歇 fake 輔助的完整實驗。
# bash run.sh
#
# 案例 2：小規模確認流程，每支影片最多 2 張；2 支 real 建庫、1 支 real 校準，
#         另取 real／fake 各 1 支測試，1 支 fake 作訓練輔助。
# bash run.sh adapter --experiment smoke_v1 --device cuda:1 \
#   --bank-videos 2 --calibration-videos 1 --eval-real-videos 1 \
#   --eval-fake-videos 1 --train-fake-videos 1 --max-frames 2 --train-batch-size 2
#
# 案例 3：完全只用 real 訓練；fake 仍保留於獨立測試集。
# bash run.sh real-only --device cuda:1 --experiment real_only_trial_v1
#
# 案例 4：不訓練 adapter，以原始 DINOv3 特徵建立比較基準。
# bash run.sh baseline --device cuda:1 --experiment baseline_trial_v1
#
# 案例 5：降低 fake 加入頻率與影響，每 10 步加入最多 1 張、權重 0.05。
# bash run.sh adapter --experiment sparse_fake_v1 --fake-interval 10 \
#   --fake-batch-size 1 --fake-weight 0.05
#
# 案例 6：改用 GPU 2，訓練 10 個 epoch；推論 batch 與訓練 batch 分開設定。
# bash run.sh adapter --experiment longer_train_v1 --device cuda:2 \
#   --train-epochs 10 --batch-size 8 --train-batch-size 4
#
# 案例 7：分階段執行同一實驗（依序執行，使用相同名稱與相同資料／訓練設定）。
# bash run.sh adapter --experiment staged_trial_v1 --stage prepare
# bash run.sh adapter --experiment staged_trial_v1 --stage extract
# bash run.sh adapter --experiment staged_trial_v1 --stage train
# bash run.sh adapter --experiment staged_trial_v1 --stage evaluate
#
# 案例 8：案例 7 完成校準後，推論一張採用相同 segmentation 流程的人臉圖片。
#         請將路徑換成實際 JPG；輸出分數、距離 NPY 與異常熱圖。
# bash run.sh adapter --experiment staged_trial_v1 --stage predict \
#   --image /absolute/path/to/segmented_face.jpg
#
# 注意：改變資料或訓練參數時，請另取 --experiment 名稱，避免混用舊特徵／權重。
#       若小實驗曾覆寫參數，後續分階段執行與推論也須帶上相同覆寫值。
#       相同設定重跑會沿用已完成結果，不會額外追加訓練 epoch。
default_experiment="adapter"
experiment="${1:-$default_experiment}"
arguments=(--task patch-bank)

case "$experiment" in
  stage1|stage2)
    # 全量實驗使用同一名稱與切分；stage1 建庫／訓練，stage2 校準／驗證。
    # --full-data 使用全部現有 JPG，取代各 *_videos 與 max_frames 的小規模上限。
    # bash run.sh stage1 --device cuda:1
    # bash run.sh stage2 --device cuda:1
    arguments+=(--full-data --experiment dfdc_patch_full_v1 --stage "$experiment")
    if (($#)); then shift; fi
    ;;
  adapter)
    # 提取表徵 → 訓練 adapter → real bank → real 校準 → 獨立測試。
    # 訓練使用 YAML 設定；相同實驗重跑會沿用已完成的權重，不追加 epoch。
    if (($#)); then shift; fi
    ;;
  real-only)
    # 只用 real 訓練；fake 僅用於獨立測試，不參與梯度更新或建庫。
    arguments+=(--train-fake-videos 0 --experiment dfdc_patch_real_only_v1)
    if (($#)); then shift; fi
    ;;
  baseline)
    # 不訓練 adapter，直接以凍結 DINOv3 的 patch 特徵建立基準。
    arguments+=(--no-train-adapter --experiment dfdc_patch_baseline_v1)
    if (($#)); then shift; fi
    ;;
  --help|-h)
    cat <<'HELP'
使用方式：bash run.sh [實驗模式] [其他參數]
  adapter    real 為主、間歇 fake 輔助（預設，設定由 YAML 決定）
  real-only  完全只用 real 訓練，保留獨立 fake 測試
  baseline   不訓練 adapter，使用 DINOv3 原始特徵
  stage1     全部現有資料分組後，提取訓練特徵、訓練與建立 real bank
  stage2     使用 stage1 保留的校準與測試資料驗證 bank，不重新訓練

範例：
  bash run.sh
  bash run.sh real-only --device cuda:1
  bash run.sh adapter --fake-interval 10 --experiment adapter_interval10
  bash run.sh adapter --stage predict --image /absolute/path/to/face.jpg
  bash run.sh --task feature-bank  # 原本只匯出 real 表徵的流程

資料與訓練參數：utils/config.yaml。後方參數可覆寫實驗預設值。
每次執行一個實驗；多任務資源分配仍由 mission.sh 管理。
HELP
    exit 0
    ;;
  --*)
    # 未填模式時使用 patch-bank；明確指定 --task 則沿用原本的 main.py 入口。
    for argument in "$@"; do
      if [[ "$argument" == --task || "$argument" == --task=* ]]; then
        arguments=()
        break
      fi
    done
    ;;
  *) echo "未知實驗模式：$experiment（請使用 bash run.sh --help）" >&2; exit 2 ;;
esac

# 不論從哪個目錄啟動，都以專案根目錄執行 main.py。
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# 非互動式 shell 需先載入 Conda，才能啟用環境。
conda_base="$(conda info --base)"
source "$conda_base/etc/profile.d/conda.sh"
conda activate pt230

# 額外參數放在最後，允許覆寫 GPU、實驗名稱或執行階段。
# exec 讓中斷訊號直接交給 Python；-u 即時顯示訓練進度。
exec python -u main.py "${arguments[@]}" "$@"
