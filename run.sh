#!/usr/bin/env bash
set -eo pipefail

# 透過 pt230 啟動 main.py；先在 utils/config.yaml 的 retrieval 填入 experiment。
# 名稱、export_previews、workers、gpu_ids、batch_size 都讀取 YAML，命令列參數優先。
# bash run.sh stage1
# bash run.sh stage2
# bash run.sh all  # 依序完成兩階段
# YAML encoder_tuning.enabled=true：先訓練 DINOv3 最後一層 LoRA，再重建整套表徵。
# YAML method：nearest／topk／cross_attention；attention 訓練參數集中在 retrieval.attention。
# cross_attention 的 Stage 2 同時比較三種評分方式；只用 real 訓練，patience 預設 5。
# Stage 1 只用 real 建庫，Stage 2 查詢 bank 並評估，兩階段使用相同名稱。
# 也可臨時使用命令列覆寫 YAML，名稱請自行指定。
# bash run.sh stage1 --exper "你取的新名稱" --workers 16 --gpus 4 5 6
# bash run.sh stage2 --exper "你取的新名稱" --workers 16 --gpus 4 5 6
# 只檢查資料切分：bash run.sh stage1 --stage prepare
# GPU 記憶體不足：加上 --batch-size 4；CPU 模式：加上空的 --gpus

stage="${1:-stage1}"
case "$stage" in
  --help|-h)
    cat <<'HELP'
使用方式：bash run.sh [stage1|stage2|ablation|all] [其他參數]
先在 utils/config.yaml 的 retrieval.experiment 填入名稱；其他設定也從 YAML 讀取。
  stage1  合併內容重複家族後切分，提取 real 特徵並建庫；LoRA / attention 依設定選用
  stage2  提取保留組特徵，以 real 校準門檻，輸出測試指標及匹配明細
  ablation  比較 Top-K、邊界降權、來源家族去重及兩者合併
  all     依序完成兩階段
資源參數：--workers 16 --gpus 4 5 6 --batch-size 8
預覽：retrieval.export_previews 設為 true 可輸出 PCA／JPG，預設 false。
LoRA：retrieval.encoder_tuning.enabled=true 時使用第一張 GPU 訓練；其後特徵提取使用全部 GPU。
命令列覆寫：--exper 名稱、--method nearest|topk|cross_attention，以及上述資源參數。
新實驗預設合併重複內容及邊界降權；--boundary-weight 1 可測試未加權基準。
舊實驗重現：--no-deduplicate-content --boundary-weight 1 --no-tune-encoder --method nearest。
特徵：RAG/normal/<名稱>/；結果：outputs/feature_bank/<名稱>/stage1、stage2
設定：utils/config.yaml 的 retrieval；執行環境：pt230。
HELP
    exit 0
    ;;
  stage1|stage2|ablation|all) if (($#)); then shift; fi ;;
  probe-stage1|probe-stage2) stage="${stage#probe-}"; shift ;;
  --*) stage="stage1" ;;
  *) echo "未知階段：$stage，請使用 bash run.sh --help" >&2; exit 2 ;;
esac

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
conda_base="$(conda info --base)"
source "$conda_base/etc/profile.d/conda.sh"
conda activate pt230
exec python -u main.py --task bank-retrieval --stage "$stage" "$@"
