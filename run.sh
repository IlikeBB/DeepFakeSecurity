#!/usr/bin/env bash
set -eo pipefail

# 透過 pt230 啟動 main.py；名稱由你使用 --exper 指定。
# Stage 1 只用 real 建庫，Stage 2 查詢 bank 並評估，兩階段使用相同名稱。
# bash run.sh stage1 --exper FB_01 --workers 16 --gpus 4 5 6
# bash run.sh stage2 --exper FB_01 --workers 16 --gpus 4 5 6
# 只檢查資料切分：bash run.sh stage1 --exper FB_01 --stage prepare
# GPU 記憶體不足：加上 --batch-size 4；CPU 模式：加上空的 --gpus

stage="${1:-stage1}"
case "$stage" in
  --help|-h)
    cat <<'HELP'
使用方式：bash run.sh [stage1|stage2|all] --exper "你取的名稱" [其他參數]
  stage1  多 GPU 提取 real 特徵，建立可追溯的 cosine 最近鄰 bank
  stage2  提取保留組特徵，以 real 校準門檻，輸出測試指標及匹配明細
  all     依序完成兩階段
資源參數：--workers 16 --gpus 4 5 6 --batch-size 8
特徵：RAG/normal/<名稱>/；結果：outputs/feature_bank/<名稱>/stage1、stage2
設定：utils/config.yaml 的 retrieval；執行環境：pt230。
HELP
    exit 0
    ;;
  stage1|stage2|all) if (($#)); then shift; fi ;;
  probe-stage1|probe-stage2) stage="${stage#probe-}"; shift ;;
  --*) stage="stage1" ;;
  *) echo "未知階段：$stage，請使用 bash run.sh --help" >&2; exit 2 ;;
esac

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
conda_base="$(conda info --base)"
source "$conda_base/etc/profile.d/conda.sh"
conda activate pt230
exec python -u main.py --task bank-retrieval --stage "$stage" "$@"
