# FSFM-inspired 實驗

本實驗將 FSFM 的 CRFR-P facial masking 與 EMA target 概念移植到 DINOv3 real-only PatchBank。官方 FSFM 同時使用 pixel reconstruction、representation decoder 與對比式 target branch；此版本使用輕量 Transformer decoder 重建 EMA DINOv3 clean patch features，因此屬於 FSFM-inspired 方法。

## Stage 1 訓練

```bash
cd /ssd8/chihyu/Project/Real-Only-DINOv3-PatchBank-for-Deepfake-Detection

bash run.sh stage1 \
  --exper DFDC_REAL_ONLY_FSFM_PARTIAL4_TOPK_V1 \
  --face-source segface \
  --tune-encoder \
  --unfreeze-blocks 4 \
  --method topk \
  --gpus 0 5 6 \
  --batch-size 8 \
  --workers 16
```

Stage 1 開始時會先顯示 `Stage 1 FSFM：快取臉部區域`。快取完成後 SegFace 會從 GPU 卸載，再開始 DINOv3 partial fine-tuning。預設凍結前 8 個 blocks、解凍最後 4 個 blocks 與 final norm；`--unfreeze-blocks 2` 可執行較保守的版本。中斷後可沿用已完成的 region NPY。若重新推論的 SegFace 邊界與既有 JPG 不同，程式會以自適應亮度前景將 JPG 與原始 RetinaFace crop 做像素匹配，找回包含暗場人臉在內的實際裁切座標。確認為錯配的圖片會寫入 `face_regions/excluded_images.json`，並從訓練排程與 Feature Bank split 排除。

CRFR-P mask 會透過 DINOv3 原生 `bool_masked_pos` 傳入 embedding layer，選中位置使用 DINOv3 mask token。影像 pixels 不會被改成黑色。EMA teacher 從 `0.996` 逐步提高至 `1.0`，只在訓練時提供完整臉部 target。

每輪 `encoder_tuning_history.json` 會記錄：

- `masked_reconstruction_loss`
- `region_reconstruction_loss`
- `local_global_loss`
- `ema_consistency_loss`
- `mean_ema_momentum`
- `validation_total_loss`

## Stage 2 評估

```bash
bash run.sh stage2 \
  --exper DFDC_REAL_ONLY_FSFM_PARTIAL4_TOPK_V1 \
  --face-source segface \
  --tune-encoder \
  --unfreeze-blocks 4 \
  --method topk \
  --gpus 0 5 6 \
  --batch-size 8 \
  --workers 16
```

Stage 2 只將 `dino_partial.safetensors` 載入原始 DINOv3 以提取 query features；EMA teacher 與 `fsfm_decoder.safetensors` 不參與異常評分。

## 建議比較

固定資料切分、Top-K、threshold 與 GPU 設定，只改 Stage 1 encoder：

| 實驗 | Encoder 設定 | 目的 |
| --- | --- | --- |
| Frozen baseline | `--no-tune-encoder` | 原始 DINOv3 基準 |
| FSFM partial-2 | `--tune-encoder --unfreeze-blocks 2` | 保守的臉部 masked-feature adaptation |
| FSFM partial-4 | `--tune-encoder --unfreeze-blocks 4` | 預設實驗 |

各實驗必須使用不同的 `--exper` 名稱。比較 Stage 2 的 AUROC、AP、固定 real calibration 門檻下的 FPR/TPR，並檢查眼睛、鼻子與嘴巴的 patch heatmap 是否更集中。
