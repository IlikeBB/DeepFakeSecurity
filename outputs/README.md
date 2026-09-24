# Output layout

此目錄只存放可重建的實驗產物；資料集與 DINO normal feature bank 不在這裡。

```text
outputs/
├── feature_bank/          # Stage 1/2 nearest-neighbor baseline
├── patch_reconstruction/  # Only-real Transformer decoder runs
├── real_patch_bank/       # Only-real statistical baseline
├── diagnostics/           # 分析、稽核與診斷圖
├── reports/               # PDF、圖表及報告建構資料
└── cache/                 # 可重新下載或建立的工具快取
```

## Current formal experiment

- NN baseline：`feature_bank/SEGFACE_FROZEN_NN_DEDUP_V2/`
- Patch reconstruction：
  `patch_reconstruction/SEGFACE_FROZEN_NN_DEDUP_V2/local_transformer_v2/`
- 主要指標：
  `patch_reconstruction/SEGFACE_FROZEN_NN_DEDUP_V2/local_transformer_v2/stage2/metrics.json`
- 研究框架 PDF：
  `reports/framework_20260924/DeepFakeSecurity_Research_Framework_20260924.pdf`

## Run status convention

- 實驗名稱下一層的普通目錄為正式 run。
- `_archive/` 保存中斷或已取代的開發版本，不可當作正式結果。
- `_smoke/` 只保存小資料流程測試，不可拿來比較模型效能。
- `cache/` 不包含實驗指標，刪除後可由工具重新建立。

每個正式 reconstruction run 固定使用以下結構：

```text
<run-name>/
├── model.safetensors
├── training.json
└── stage2/
    ├── metrics.json
    ├── calibration_scores.json
    ├── evaluation_scores.json
    ├── video_scores.json
    ├── explanations.json
    ├── qwen_explanations.json  # 執行 Qwen 說明後才會出現
    └── heatmaps/
```

Celeb-DF 是外部資料集輸出，依資料位置保存在：
`/ssd8/chihyu/Dataset/DeepFake_Dataset/Celeb-df-external/results/`。
