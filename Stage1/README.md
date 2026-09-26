# Stage 1：正常表徵學習與 Feature Bank

Stage 1 只使用切分為 `bank` role 的真實人臉。它可以直接使用 frozen DINOv3，也可以先執行 real-only partial fine-tuning／FSFM-inspired 訓練，最後建立不可被 Stage 2 修改的正常 patch 索引。

## 程式

- `fsfm.py`：SegFace 臉部區域映射、CRFR-P masking、Transformer feature decoder 與重建損失。
- `encoder_tuning.py`：最後 2～4 個 DINOv3 blocks 微調、EMA teacher、early stopping 與 checkpoint。
- `bank_builder.py`：讀取 Stage 1 real features，移除背景 patch，建立並驗證 cosine retrieval index。

## 輸入與輸出

輸入為固定的資料切分及 real SegFace／RetinaFace 圖片。Feature NPY 與檢索矩陣存入設定中的 `retrieval.bank_dir/<EXPERIMENT>/`；訓練紀錄與 checkpoint 存入 `outputs/DFDC/<EXPERIMENT>/Stage1_Bank_Build/`。

```bash
bash Stage1/run.sh --exper <EXPERIMENT> --face-source segface \
  --tune-encoder --unfreeze-blocks 4 --method topk \
  --gpus 0 5 6 --batch-size 8 --workers 16
```

Stage 1 完成條件是 `Stage1_Bank_Build/retrieval.json` 存在且其中記錄的設定、切分及索引檔 SHA-256 全部吻合。

## 完成後的分群診斷

可從同一來源家族配對抽樣 real/fake，觀察目前 encoder 的影像級特徵分布。此步驟只做後驗分析，
不會把 fake 寫入正常 Feature Bank，也不會更新模型：

```bash
bash run.sh cluster --exper <EXPERIMENT> --gpus 0 \
  --cluster-samples 500 --cluster-count 8 --batch-size 8 --workers 16
```

結果保存於 `outputs/DFDC/<EXPERIMENT>/Stage1_Feature_Distribution/`，包含 PCA 圖、群集組成、
逐樣本座標與 ARI、NMI、silhouette 等統計。這是表徵診斷，不取代 Stage 2 的 AUROC/AP。
