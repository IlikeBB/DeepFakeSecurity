# Stage 2：異常檢索與評估

Stage 2 將 calibration／evaluation 圖片視為 query，只讀取 Stage 1 建立的 real patch bank，不會把 query feature 寫回正常庫。

## 程式

- `bank_search.py`：分塊執行 exact cosine nearest-neighbor 搜尋，並在啟動前檢查 GPU 記憶體。
- `evaluation.py`：前景 patch 選取、Nearest／Top-K 評分、real calibration 門檻、AUROC／AP／FPR／TPR 與消融。

## 輸入與輸出

Query features 存入設定中的 `retrieval.stage2_cache_dir/<EXPERIMENT>/`。評估分數、門檻與 patch matching 證據存入 `outputs/DFDC/<EXPERIMENT>/Stage2_Anomaly_Evaluation/`。
完成後另輸出 `evaluation_analysis.png`，包含 ROC、precision-recall 與 real/fake anomaly-score 分布。

檢索會將同批影像的前景 patch 合併送入 GPU，以填滿 query chunk。若工作中斷，重新執行相同指令會驗證並讀取已完成的 `patch_matches/*.npz`，不會重新計算那些影像。

```bash
bash Stage2/run.sh --exper <EXPERIMENT> --face-source segface \
  --tune-encoder --unfreeze-blocks 4 --method topk \
  --gpus 0 5 6 --batch-size 8 --workers 16
```

Stage 2 必須沿用 Stage 1 的實驗名稱與完整設定。程式會先驗證 Stage 1 checkpoint、資料切分及索引檔，驗證成功後才提取 query features。
