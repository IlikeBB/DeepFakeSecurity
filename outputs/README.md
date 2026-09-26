# 實驗輸出

每個資料集與實驗各自使用一個目錄，Stage 1 與 Stage 2 不共用結果資料夾。

```text
outputs/
└── DFDC/
    └── <EXPERIMENT>/
        ├── Stage1_Bank_Build/
        ├── Stage1_Feature_Distribution/
        └── Stage2_Anomaly_Evaluation/
            └── ablations/
```

- `Stage1_Bank_Build`：保存 Stage 1 設定、切分、索引完成紀錄、partial-finetune 權重與訓練歷史；FSFM 實驗另存只供預訓練使用的 `fsfm_decoder.safetensors`。
- `Stage1_Feature_Distribution`：選用的 Stage 1 後驗分群診斷；保存配對 real/fake 表徵、PCA 圖及群集統計，不修改模型或正常庫。
- `Stage2_Anomaly_Evaluation`：保存校準分數、門檻、evaluation 分數、metrics 與每張圖的 patch matching 證據。
  完成後的 `evaluation_analysis.png` 顯示 ROC、precision-recall 與 anomaly-score 分布。
- 大型 `.npy` normal bank 與 query cache 位於 `/ssd8/chihyu/Dataset/DeepFake_Dataset/RAG/`，不存進專案輸出。

執行方式請見專案根目錄的 `README.md`。
