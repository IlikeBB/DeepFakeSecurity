# DeepFakeSecurity

## DINOv3 最後一層 LoRA 表徵實驗

`retrieval.encoder_tuning.enabled: true` 時，Stage 1 會先在 DINOv3 最後一個 Transformer block
的 Q/K/V 投影加入 LoRA。原始 DINOv3 權重保持凍結。壓縮訓練參考 UMCL 的跨品質概念，
將同一張 real 人臉建立 clean、mild、severe 三個對齊分支，同時約束對應 patch 的 cosine
一致性與 patch-to-patch affinity；JPEG quality 與縮放範圍集中在 `encoder_tuning.compression`。
這裡使用 JPEG 模擬品質層級，數值不宣稱等同 FF++ 的 H.264 c23/c40。feature anchor 限制
表徵漂移；真實 fake margin 預設關閉，只在 `fake_weight` 大於 0 時依 `fake_interval` 啟用。
訓練與 validation 按來源 family 分離，validation loss 連續 5 個 epoch 未改善便停止。

同一張 real 圖片也會產生一個局部 soft discrepancy。遮罩內 patch 使用 margin loss 推離原始
real feature，遮罩外保持一致；`fake_weight: 0.0` 時整個 LoRA 訓練不讀取真實 fake。
每個 epoch 只印一行平均 `total/comp/rel/local/bg/anchor/fake/val` loss，完整數值同步寫入
`dino_lora_history.json`。

LoRA 會改變所有 patch features，因此請先在 `utils/config.yaml` 的 `retrieval.experiment` 填入新名稱，
不可沿用 frozen DINO 的實驗資料夾。啟動指令維持簡單：

```bash
bash run.sh stage1
bash run.sh stage2
```

LoRA 權重、訓練歷史與設定分別寫入 `outputs/feature_bank/<exper>/stage1/dino_lora.safetensors`、
`dino_lora_history.json` 與 `dino_lora.json`。Stage 1 完成訓練後使用多張 GPU 重建 real bank；
Stage 2 會以同一份 LoRA 權重提取 calibration 與 evaluation 特徵。
目前 `method: topk` 直接以 real bank cosine 相似度判斷；不額外訓練 cross-attention。

## ATT_DEMO：real bank 的 cross-attention 實驗

在 `utils/config.yaml` 的 `retrieval` 管理設定，啟動方式不變：

```bash
bash run.sh all
# 或分開執行 bash run.sh stage1、bash run.sh stage2
```

目前實驗名稱為使用者指定的 `ATT_DEMO`，`method: cross_attention`。
三種模式為 `nearest`（原始 1-NN）、`topk`（cosine softmax 加權正常參考）、
`cross_attention`（可訓練的多頭正常參考加權）。更改模式或模型設定時另取實驗名稱。
CLI 可用 `--method` 覆寫；通常只需修改 YAML。

Stage 1 凍結 DINOv3，提取 real 前景 patch 並儲存原始檢索庫。
Cross-attention 只訓練 Q/K 投影：預設 Top-16 候選、128 維投影、4 heads；
每個 head 對完整、固定的 DINO values 分配權重，再平均各 head 的重建。
因此輸出是 real 候選向量的凸組合；沒有 query residual、可訓練 V 或輸出 decoder。
這是受限的 multi-head cross-attention，並非標準完整 Transformer block。

Bank real 內按來源家族分出 20% validation。訓練與驗證的候選 bank **只含 train 家族**，
訓練 query 再排除自身整個來源家族；fake、calibration、evaluation 都不參與訓練。
每張 real 最多抽 32 個 patch 作 query，加入輕微噪聲，最小化與乾淨 DINO 特徵的 cosine 重建誤差。
最高 30 epochs，real validation loss 連續 5 輪未改善即 early stop，Stage 2 載入最佳權重。
可在 `retrieval.attention` 修改 `epochs`、`patience`、`batch_size`、`neighbors` 等設定。

Stage 2 以完整 bank real（包含內部 real validation）作正常參考，
在同一次 Top-K 搜尋同時計算 1-NN、固定 Top-K 加權及 cross-attention 重建誤差。
每張圖取最大 10% patch 誤差平均；每種方法分別以獨立 real calibration 的 99 百分位設門檻。
此比較不自動依 Stage 2 AUC 選模；既有 evaluation 已多次查看，應視為開發比較，非全新最終測試。

多 GPU 共用一列提取／候選搜尋／Stage 2 進度，訓練使用 DataParallel 並覆用同一列 tqdm。
`export_previews: false` 預設略過 PCA／JPG 視覺化，改成 `true` 後重跑 Stage 1 可補齊。
舊 CLIP／ASA 分支仍維持移除，原始 DINO NPY 與過去實驗結果保留。

```text
RAG/normal/ATT_DEMO/
  <來源分包>/<影片>/        # 原始 DINO NPY
  retrieval/               # real 前景特徵、來源編號、patch 編號
  attention/model.safetensors
  cache/calibration/、cache/evaluation/
outputs/feature_bank/ATT_DEMO/
  stage1/attention.json           # 設定、訓練／驗證家族、最佳 epoch、權重及索引雜湊
  stage1/attention_history.json   # train／validation loss、未改善輪數
  stage2/metrics.json             # 當前方法指標及三種方法比較
  stage2/comparison.json          # 各方法 AUROC、AP、FPR、TPR 與獨立校準門檻
  stage2/*/patch_matches/*.npz    # 候選 bank IDs、權重、各方法誤差與 query patch IDs
```

匹配 JSON 的 `cosine_similarity`／`distance` 仍指最相似 real patch；
`anomaly_distance` 才是當前評分方法的 patch 誤差。
`reference_patch_ids`／`reference_weights` 可追溯重建所用的全部候選；NPZ 中權重是當前方法的權重。
Stage 1 允許續接已有 NPY 與索引；完整 attention 權重驗證後沿用，未完成的訓練從頭開始。

完成 Stage 2 後，可以使用相同實驗執行一次固定的瓶頸消融：

```bash
bash run.sh ablation
```

設定位於 `retrieval.ablation`。程式同時比較原始 Top-K、前景邊界降權、每個來源家族限制候選數，
以及兩者合併；四種方法各自使用相同 calibration real 校準門檻。預設邊界權重為 0.5，
並在 Stage 2 已儲存的 Top-16 內讓每個來源家族最多保留 2 個。結果寫入
`outputs/feature_bank/<exper>/ablations/boundary_family/report.json`，不覆寫 Stage 2 結果。

## 目前啟動方式：FB_01 real feature bank 相似度檢索

```bash
cd /ssd8/chihyu/Project/DeepFakeSecurity
# Stage 1：多 GPU 提取 real patch，建立檢索索引；JPG 預覽由 YAML 選用
bash run.sh stage1 --exper FB_01 --method nearest --workers 16 --gpus 4 5 6
# Stage 1 完成後，使用相同實驗名稱執行 Stage 2
bash run.sh stage2 --exper FB_01 --method nearest --workers 16 --gpus 4 5 6
```

入口會自動啟用 `pt230`，再執行 `main.py --task bank-retrieval`。
參數集中在 `utils/config.yaml` 的 `retrieval`；圖片、標籤、模型路徑沿用現有 YAML 設定。
尚未提取特徵也能直接開始。`--batch-size` 是每張 GPU 的提取 batch，預設 8。
每張 GPU 各有模型／搜尋 worker；`--workers` 是所有 GPU 共用的圖片與 NPY 讀取執行緒上限。
CPU 模式加空的 `--gpus`。只檢查原始 FB_01 資料切分可執行 `bash run.sh stage1 --exper FB_01 --method nearest --stage prepare`。

流程固定 DINOv3，不訓練分類器：每個前景 patch 查詢 real bank 的 cosine 最近鄰，
以 `1 - similarity` 作距離，最高 10% 距離平均為圖片異常分數。
閾值只用獨立的 real calibration 資料校準；fake 僅在保留的 evaluation 中評估。
使用原全量家族切分；`train_fake` 仍記錄在 split 中供比較，但本方法不提取、不使用該組。
前景判定為非黑色像素占比，不是新增語意分割；目前搜尋整個 real bank，未限制匹配到相同臉部區域。
分數較高代表偏離 real bank，不是 fake 機率。新方法的 AUC 必須以實際 Stage 2 評估為準。

```text
RAG/normal/FB_01/
  <來源分包>/<影片>/           # real NPY；選用 preview.jpg
  retrieval/                  # real 特徵矩陣、來源編號、patch 編號
  cache/calibration/          # 校準查詢特徵，不放入檢索索引
  cache/evaluation/           # 測試查詢特徵，不放入檢索索引
outputs/feature_bank/FB_01/
  stage1/                     # config.json、splits.json、sources.json、retrieval.json
  stage2/                     # metrics.json、thresholds.json、圖片分數與完整 patch 匹配 NPZ
```

`stage2/evaluation_scores.json` 包含每張圖片的分數、判斷與最多 5 個高異常 patch 的最近鄰證據：
來源 JPG、來源影片／家族、雙方 patch 座標與 cosine similarity。
完整距離及 bank patch ID 在每張圖片的 NPZ；背景位置使用 `-1`，由 Stage 1 索引還原匹配來源。
Stage 1 可重跑以續接已完成 NPY；完成索引後會驗證雜湊並沿用。
兩階段須用同一 `--exper`；修改資料、模型或比對參數時另取名稱，GPU／執行緒／batch 可調整。

以下為舊版實驗紀錄；目前有效入口以上方檢索流程為準。舊線性評分器的 0.6739 不代表此檢索方法的成績。

## AUC 改善實驗：自動建立／沿用 DINO 特徵

```bash
cd /ssd8/chihyu/Project/DeepFakeSecurity
# 將「你取的名稱」換成自己的名稱，兩階段必須一致。
bash run.sh probe-stage1 --exper "你取的名稱" --workers 8 --gpus 1 2
bash run.sh stage2 --exper "你取的名稱" --workers 8 --gpus 1 2
```

輸出位置為 `outputs/feature_bank/<你取的名稱>/`。不會自動取名；未指定時會提示用法並停止。
`stage2` 執行 probe 驗證；原本的 `probe-stage2` 仍可使用。舊版 adapter 驗證入口改為 `adapter-stage2`。
例如 Stage 1 使用 `--exper demo_S1`，Stage 2 也使用 `--exper demo_S1`，不能改成 `demo_S2`。
`--exper` 指定整個實驗的名稱，不是階段名稱；找不到權重時，錯誤訊息會列出檢查路徑與可用實驗。
也可以自行設定 `utils/config.yaml` 的 `bank_probe.experiment`，命令列參數優先。
`source_experiment` 預設為 `null`：從人臉 JPG 自動建立與 `--exper` 同名的來源特徵，
不再依賴 `dfdc_patch_full_v1`。也可用 `--source-experiment "已有來源名稱"` 沿用其他實驗的快取。

設定集中在 `utils/config.yaml` 的 `bank_probe`。預設用 8 個執行緒讀取 NPY/JPG，
使用 GPU 1、2；`--workers`、`--gpus` 可覆寫。DINO 提取時每張卡載入一份模型，
共用最多 `--workers` 個圖片讀取執行緒；`--extract-batch-size` 是每張卡的 batch（預設 8）。
評分器訓練時，每張卡有一個候選模型工作執行緒，
不同候選分卡訓練，最終選定模型只在第一張卡重擬合；Stage 2 將圖片分配到多張卡評分。
GPU 版使用與 CPU 版相同的加權 logistic loss 與 L2 正則化，由 GPU 計算 loss／gradient，
CPU 的 L-BFGS-B 控制最佳化；小型評分器不保證比 CPU 快，實際速度也受磁碟讀取影響。
空的 `--gpus` 可改用 CPU。資源設定適用於 probe 的建庫、訓練與評分。
Stage 1 先固定全量來源家族切分（沿用 `feature_bank.seed`），只提取 bank／train_fake，
然後訓練評分器；Stage 2 才提取 calibration／evaluation 並驗證。
新來源的 real bank 與每支影片的預覽在 `RAG/normal/<名稱>/`；
fake／校準／測試 NPY 分開存於 `RAG/cache/<名稱>/<分組>/`，不混入 normal bank。
評分器與指標存放 `outputs/feature_bank/<名稱>/`。不需要先跑舊 adapter 訓練。
中斷後用相同指令重跑會檢查並沿用完成的 NPY，缺少的檔案才重新提取。
明確指定的 `--source-experiment` 若不存在則報錯；舊版來源沿用原有快取路徑，需已完成所需分組的提取。
兩階段都會顯示圖片進度，Stage 1 另列出每個候選的 validation AUROC。

新評分器先依黑色背景的像素比例降低背景 patch 權重，再計算單張圖片的特徵平均、
標準差與 2×2 區域平均。這個 2×2 是空間池化，不是影片時序；原始 `14×14×768` NPY 保持不變。
亮度只用來估計已分割圖片的前景占比，並非重新執行語意分割。
標準化統計只用 real。從原 training 的來源家族另留 20% 作內部驗證，
比較兩種描述子、real 統計距離與三個固定正則化強度的線性評分器；
同來源 real／fake 一起分派，原 calibration／evaluation 完全不參與選型。
選定後以原 training 重擬合；real bank 不加入 fake。

預設評分器使用 1,997 張 real，加上 40 支不同來源影片的 400 張 fake；
每支 fake 的擬合權重是 real 的 10%。這是少量 fake 輔助的弱監督評分，
不是純 real-only，也不是舊 adapter 的「每五步插入 fake」訓練方式。
內部驗證會使用其保留的 real／fake 標籤；400 張僅指最後擬合使用的 fake 數量。
可用 `--fake-videos 0 --exper probe_real_only_v1` 關閉 fake 擬合，
但內部驗證仍使用 fake 標籤選擇統計方式。
改動參數時請更換實驗名稱，兩階段帶上相同參數；已完成的 Stage 1 重跑會沿用評分器。

2026-09-23 實測，同一組 6,549 張圖片（440 real／6,109 fake）：

| 方法 | Image AUROC | AP | FPR | TPR |
| --- | --- | --- | --- | --- |
| 原 adapter＋最近鄰 bank | 0.5971 | 0.9519 | 0.0705 | 0.1216 |
| 前景／空間統計＋少量 fake 評分器 | **0.6739** | 0.9640 | 0.0250 | 0.1015 |

最佳內部驗證 AUROC 為 0.7617，選定 4,608 維描述子與 `C=0.01`。
FPR／TPR 使用各方法在 real calibration 上的第 99 百分位閾值，並非相同測試 FPR；
AUROC 提升不代表目前門檻的召回率已足夠。fake 約占測試圖片 93.3%，因此 AP 很高不能單獨視為良好辨識能力。
此結果仍非 identity-disjoint 或官方 DFDC test；不代表新身份、新資料集上的效能。

上述數字來自先前 CPU 版的實測（當時輸出名稱為 `dfdc_bank_probe_v1`），並非多 GPU 版重新量測的結果。
新實驗使用你指定的名稱，輸出在 `outputs/feature_bank/<你取的名稱>/`。
輸出中的 `selection.json` 記錄候選、選型及 fake 名單，
`validation_split.json` 記錄內部家族切分，`scorer.npz` 存標準化統計與線性權重，
`metrics.json`／`evaluation_scores.json` 存測試結果。原實驗指標完整保留。
評分輸出是排序分數，不是經機率校準的 fake 機率；此評分器目前提供圖片分數，沒有 patch 定位熱圖。

設計參考正常特徵統計建模的 [PaDiM](https://arxiv.org/abs/2011.08785)，
以及使用異常標籤改善特徵比對的 [DFM（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_DFM_Differentiable_Feature_Matching_for_Anomaly_Detection_CVPR_2025_paper.html)。
這裡實作的是容易驗證的統計描述子／線性評分基準，並非上述論文的完整復現。

feature bank 統一放在 `RAG/normal/`，既有 `dfdc_patch_full_v1` 已由 `RAG/general/` 搬入。
既有 provenance 設定檔可能保留建立時的舊路徑；程式允許搬移 bank 根目錄，
保留原設定檔與雜湊，避免使既有 adapter／校準結果失效。

## 全量實驗：Stage 1 建庫、Stage 2 驗證

```bash
cd /ssd8/chihyu/Project/DeepFakeSecurity

# 可先查看全量切分，只寫入設定與來源清單，不提取特徵或訓練
bash run.sh stage1 --stage prepare

# Stage 1：提取訓練特徵、訓練 adapter、儲存原始及調整後的 real bank
bash run.sh stage1 --device cuda:1

# Stage 1 完成後才執行 Stage 2：提取保留資料、校準門檻、評估與輸出熱圖
bash run.sh adapter-stage2 --device cuda:1
```

Stage 1／Stage 2 都會顯示 `tqdm` 進度條，包含完成數、處理速度與預估剩餘時間：
特徵提取以 batch 計數（同時顯示圖片總數），bank 載入與異常評分以圖片計數，
預覽／熱圖以影片計數。訓練顯示目前 epoch、step、`train_mean_auc`、平均 real loss 與累計 fake 加入次數。
`train_mean_auc` 是本 epoch 中同時有 real／fake 的批次 AUROC 之算術平均，每次 fake 加入時更新；
epoch 開始時歸零，尚無有效批次或純 real 訓練時顯示 `N/A`，JSON 記為 `null`。
分數取自該步權重更新前的訓練樣本；沒有使用校準／測試資料。
這只是訓練監測值，不等於整個 epoch 合併計算的 AUC，也不能取代 Stage 2 完整測試集的 AUROC。
每個 epoch 的 `mean_batch_auc` 與 `auc_batches` 會保存在 `training/adapter.json` 的 `history`。
資料檢查、模型載入與 PCA 擬合也有階段提示；模型載入及 PCA 計算本身沒有細分百分比。
重跑時進度包含已有特徵的快取檢查；若 adapter 已完成，會顯示沿用權重並略過訓練。
新進度顯示適用於重新啟動的程序，已在執行中的 Python 程序不會自動載入程式修改。

DINOv3 使用 PyTorch。`main.py` 在載入 Transformers 前設定 `USE_TF=0`、`USE_TORCH=1`，
避免其影像處理器自動匯入 TensorFlow 而出現 oneDNN／CPU 指令集提示；
`run.sh` 與 `mission.sh patch-bank` 都會套用。這些提示本身不是錯誤，無須重裝 CUDA 或 TensorFlow。
RetinaFace 的 `mission.sh crop-face` 仍在獨立程序使用原本的 TensorFlow 環境。

這兩個入口固定使用 `--full-data` 與新實驗名稱 `dfdc_patch_full_v1`。
「全量」是全部現有 segmentation JPG 納入資料池，包含所有現有幀，
不受 `bank_videos`、`calibration_videos`、`eval_*_videos`、`max_frames` 的抽樣上限限制。
它不會自動從原始影片補抽圖片，也不會把驗證資料放入 bank。

全量切分採固定 seed：real 來源家族約 70% 建庫／訓練、15% 校準，其餘驗證；
校準挑選沒有 fake 衍生圖片進入本資料池的 real 家族，確保校準只使用 real，且所有圖片皆可分派。
同家族 fake 隨其 real 分入訓練或驗證；只有 fake 的家族按 70%／30% 分入訓練／驗證。
訓練內允許 real 與其 fake 衍生影片共存，但訓練、校準、驗證三者的來源家族互不重疊。
因此全量模式與舊版小規模實驗的切分策略不同。

目前資料核對結果（2026-09-23）：

| 分組 | 影片數 | 圖片數 |
| --- | --- | --- |
| real bank／訓練 | 200 | 1,997 |
| fake 訓練候選池 | 1,435 | 14,344 |
| real 校準 | 43 | 429 |
| real＋fake 驗證 | 655 | 6,549 |

共使用 23,319 張現有圖片。fake 候選池全量提取特徵，但訓練仍每 5 步抽最多 2 張，
不是把全部 fake 混入每一批，也不保證每張 fake 都參與梯度更新。
`train_fake_videos` 在全量模式下不限制候選池大小；設為 0 仍可關閉 fake 輔助。
fake 特徵僅在加入訓練時載入當批，避免整個候選池佔滿 GPU。

Stage 1 不提取或評分校準／驗證圖片；完成後會寫入 `stage1_complete.json`。
Stage 2 會檢查完成紀錄與權重，使用凍結後的 bank 與 adapter，不重新訓練。
兩階段要使用相同實驗名稱及資料／訓練參數；如果 Stage 1 覆寫參數，Stage 2 也要帶上相同值。
預設模型、epoch、fake 間隔與權重沿用 YAML 的 `patch_bank`，比例目前固定於 `bank_flat.py`。

bank 位於 `RAG/normal/dfdc_patch_full_v1/`；驗證指標位於
`outputs/feature_bank/dfdc_patch_full_v1/metrics.json`，圖片分數在 `evaluation_scores.json`。
這仍是 DFDC test 清單內重新切分的實驗，不是官方 test 評分；全量結果與改善比較見上方紀錄。

## 局部 feature bank 與 real 為主的訓練

新增 `patch-bank` 任務，直接讀取目前扁平命名的 `DFDC-Frame-CropFace_normal` 與
`DFDC-Frame-CropFace_anomaly`。DINOv3 全程凍結，每張人臉獨立提取最後一層
`14×14×768` patch 特徵並存為 NPY；不做影片特徵平均。
本任務設定集中在 `utils/config.yaml` 的 `patch_bank`，未覆寫的值繼承 `feature_bank`。

```bash
cd /ssd8/chihyu/Project/DeepFakeSecurity

# 啟用 pt230 → main.py → 提取、訓練、建庫與評估；預設單 GPU cuda:1
bash run.sh

# mission.sh 也可派送同一任務；此任務使用 --device，不使用 segmentation 的 --gpus/--cores
bash mission.sh patch-bank --device cuda:1

# 完全不使用 fake 訓練；測試集仍包含 fake
bash run.sh real-only

# 關閉 adapter，取得未訓練 DINOv3 的基準；保留相同來源切分
bash run.sh baseline

# 調整 fake 加入間隔、權重；變更實驗設定需另取名稱
bash run.sh adapter --fake-interval 10 --fake-weight 0.05 --exper dfdc_patch_sparse_fake_v1
```

`run.sh` 內已用中文註解列出三種實驗流程，每次執行一個實驗；`bash run.sh --help`
可查看中文說明。省略模式時預設為 `adapter`，可在腳本的 `default_experiment` 修改。
GPU 與訓練參數仍由 YAML 管理，命令列覆寫優先；原本 `--task ...` 的明確任務指令仍可使用。

預設資料分工如下，每支影片最多均勻抽 10 張：

| 用途 | 影片數 | 參與訓練／建庫 |
| --- | --- | --- |
| real 訓練與 bank | 8 | 訓練 adapter，且只有這些 real 進入 bank |
| fake 訓練輔助 | 2 | 偶爾參與圖片層級的異常排序損失，不進 bank |
| real 校準 | 4 | 不參與訓練，只設定圖片分數門檻與熱圖色階 |
| real／fake 測試 | 各 4 | 不參與訓練或門檻設定 |

依來源 CSV 驗證 normal／anomaly 標籤，並依 DFDC 原始 `metadata.json` 的 `original`
欄位隔離原片與其偽造衍生影片。固定 seed、先分影片家族再抽幀；不是按圖片隨機切分。
這是 DFDC test 清單內的前測實驗，**不是官方 test 結果，也不保證人物身分互斥**。
`train_fake_videos: 0` 會重新分組，因此與保留 fake 訓練組的實驗不一定使用相同 bank。

訓練的只有 `768→128→768` residual adapter，初始為原表徵方向的恆等映射：

- real：向其他來源影片的正常 patch 靠近，搜尋時排除自身影片，避免自我匹配。
- 保留特徵：限制調整後與原始 DINOv3 特徵方向的偏差，降低表徵塌縮風險。
- fake：每 `fake_interval: 5` 個 real 步驟加入最多 `fake_batch_size: 2` 張，
  以 `fake_weight: 0.1` 加入圖片分數排序損失。圖片分數為最高 10% patch 距離平均；
  不把 fake 圖片內的每個 patch 都標成偽造。fake 抽樣由 seed 決定，間隔跨 epoch 累計。

訓練用參考集合只從 bank real 分層抽最多 2,048 個 patch，推論使用完整 real bank。
訓練後使用同一 adapter 轉換 bank 與待測圖片，再做 L2 正規化、精確 cosine 最近鄰搜尋。
門檻來自獨立 real 校準圖片分數的第 99 百分位；少量校準資料不保證實際誤報率為 1%。
異常分數表示偏離 bank 的程度，**不是 fake 機率**。

這是可控制的 metric adaptation 實驗，並非某篇論文的完整重現；設計背景可參考
[AnomalyDINO](https://openaccess.thecvf.com/content/WACV2025/papers/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.pdf)
的正常 patch bank，以及 [Deep SAD](https://arxiv.org/abs/1906.02694) 的少量已知異常輔助概念。
目前保留全部 patch（含黑色背景與裁切邊界），尚未加入語意部位限制或 mask 篩選；
熱圖僅呈現 patch 異常距離，不是經像素標註驗證的偽造區域。

```text
RAG/normal/dfdc_patch_adapter_v1/
  bank_config.json、splits.json、manifest.jsonl、patch_index.json
  visualization.json                       # 僅 real bank 擬合的 PCA 色彩設定
  dfdc_train_part_N/<video>/
    frame_XXXXXX.npy、frame_XXXXXX.json     # 原始 DINOv3 特徵與來源紀錄
    preview.jpg                            # 每支 real bank 影片一張 PCA 預覽
  adapted/dfdc_train_part_N/<video>/
    frame_XXXXXX.npy                        # 同一 adapter 轉換後的 real 特徵

outputs/feature_bank/dfdc_patch_adapter_v1/
  training/adapter.safetensors、adapter.json # 權重、來源雜湊、每 epoch 損失與 fake 次數
  train_fake/、calibration/、evaluation/    # 各組原始 NPY 與來源紀錄
  evaluation/heatmaps/<part>/<video>/preview.jpg
  thresholds.json、metrics.json、evaluation_scores.json
  predictions/<image_hash>/               # 單張推論分數、距離 NPY、最近鄰 ID 與熱圖
```

原始與調整後 NPY 不互相覆寫；PCA 預覽顯示原始特徵，異常熱圖顯示實際搜尋結果。
熱圖共用由 real 校準資料決定的色階，每支測試影片儲存一張代表幀；圖片分數仍逐張計算。
`adapted/` 對應同實驗的 adapter；載入權重與推論門檻時會檢查設定、切分與權重雜湊。
來源 JPG 若變動，會拒絕沿用舊特徵。來源資料夾維持只有圖片，不新增任何紀錄。

可拆開執行 `--stage prepare`、`extract`、`train`、`evaluate`；預設 `all` 依序完成。
相同設定重跑會沿用已完成特徵與 adapter checkpoint，再重新評估；不會額外追加 epoch。
若中途訓練尚未完成，重跑會由固定 seed 重新訓練。改變設定請使用新的 `--exper`。
單張推論需要先完成校準，輸入須採用相同的人臉裁切／segmentation 流程：

```bash
bash run.sh --task patch-bank --stage predict --image /absolute/path/to/segmented_face.jpg
```

程式分工：`bank_flat.py` 管理來源與切分、`bank_encoder.py` 提取凍結 DINOv3 表徵、
`bank_adapter.py` 訓練與載入調整層、`bank_search.py` 搜尋、`bank_heatmap.py` 繪圖；
`feature_bank.py` 串接流程。原本只匯出 real NPY 的流程改用 `bash run.sh --task feature-bank`。

2026-09-23 小規模實測使用上述預設與相同切分，測試共 40 張 real、40 張 fake：

| 實驗 | 圖片 AUROC | real 誤報率 | fake 檢出率 |
| --- | --- | --- | --- |
| 凍結 DINOv3、無 adapter | 0.3956 | 7.5% | 2.5% |
| real 為主＋間歇 fake adapter | 0.3863 | 7.5% | 2.5% |

adapter 共訓練 50 步，fake 批次加入 10 次。此結果表示流程可執行，但**目前表徵與訓練方式
尚未有效區分真假，也未改善基準**；僅 8 支測試影片，不能據此推論跨資料集能力。
保留原始結果供比較，不依測試集調整分數方向或門檻。

## mission.sh 目前狀態與使用方式

目前可以直接用 `mission.sh` 啟動 Stage 1／Stage 2 feature bank 實驗。它會把任務轉交給
`run.sh`，自動切換到專案目錄、啟用 Conda 環境 `pt230`，再執行
`main.py --task bank-retrieval`：

```bash
cd /ssd8/chihyu/Project/DeepFakeSecurity

# 先在 utils/config.yaml 設定 retrieval.experiment
bash mission.sh stage1
bash mission.sh stage2

# 或連續執行兩個階段
bash mission.sh all
```

GPU、讀取執行緒與 batch size 建議直接在 `utils/config.yaml` 的 `retrieval` 區塊設定；
需要臨時覆寫時，可使用 `--gpus 4 5 6 --workers 16 --batch-size 8`。Stage 1 使用第一張
GPU 訓練 LoRA，接著以全部指定 GPU 提取 real feature bank；Stage 2 使用相同實驗名稱與
LoRA 權重進行校準和評估。

`mission.sh` 也可執行已還原的 `segment-face` 與 `crop-face` 資料前處理。先查看所有入口：

```bash
bash mission.sh --help
```

資料已經切好時，不必重跑這兩項；直接執行上方 Stage 1 即可。需要重新處理資料時，建議先做
小規模測試，再依下方舊版紀錄擴大範圍：

```bash
# SegFace：每類先測試一支影片，只檢查資源配置時加上 --dry-run
bash mission.sh segment-face --limit 1 --cores 8 --gpus 4,5 --dry-run
bash mission.sh segment-face --limit 1 --cores 8 --gpus 4,5

# RetinaFace：先測試兩支原始影片，每支抽 4 幀
bash mission.sh crop-face --limit 2 --num-frames 4 --workers 2 --device cpu
```

`patch-bank` 仍依賴已移除的 `script/feature_bank.py`，目前不可用；現行 feature bank 請使用
`stage1`／`stage2`。已產生的 normal／anomaly JPG 可以直接供目前流程使用，不需要重新切臉。

### 舊版 mission.sh 使用紀錄

`mission.sh` 是多任務 shell 入口，會切換至專案目錄、載入 Conda 並啟用 `pt230`。
執行前需確保終端機可使用 `conda`、已建立 `pt230`，並使用 Bash；CPU 綁定使用 Linux `taskset`。

```bash
cd /ssd8/chihyu/Project/DeepFakeSecurity
bash mission.sh --help
```

| 指令 | 用途 | 平行方式 |
| --- | --- | --- |
| `bash mission.sh segment-face` | 對已裁切的人臉做 SegFace 分割與外擴裁切 | 每張指定 GPU 一個程序，或 CPU 多程序 |
| `bash mission.sh crop-face` | 從原始影片抽幀，用 RetinaFace 切臉 | 依 `crop_face.workers` 啟動多個工作程序 |
| `bash mission.sh patch-bank` | 局部表徵、real 為主的 adapter 訓練與評估 | 單 GPU，由 `--device` 指定 |
| `bash mission.sh` | 預設執行 `crop-face` | 同上 |

各任務分別啟動，不會自動串接。`segment-face` 讀取 `segment_face.input_dir` 指定的
現成人臉資料；首次使用 SegFace 前，先下載模型：

```bash
conda run --no-capture-output -n pt230 python -m script.setup_segface
```

### SegFace 小規模測試與抽樣

```bash
# 快速測試：normal、anomaly 各一支影片，每支最多 10 張
bash mission.sh segment-face --limit 1

# 執行目前 YAML 設定：part 0～10，每支影片均勻抽最多 10 張
bash mission.sh segment-face

# 暫時覆寫分包範圍、每支張數與每個程序的推論 batch size
bash mission.sh segment-face --parts 0 1 2 --frames-per-video 10 --batch-size 4

# 調整裁切參數後，重新處理本次選中的圖片
bash mission.sh segment-face --overwrite
```

`--limit` 是**每個類別的影片數上限**，所以 `--limit 1` 最多處理兩支影片。
先選定影片再分配給工作程序，增加 GPU 數量不會增加抽樣數量或重複處理影片。
`--parts` 使用空白分隔的分包編號；0～10 共 11 個分包。
不足指定張數時使用全部有效幀；分割失敗的圖片不輸出，也不補抽其他幀。
改變抽樣範圍不會自動清除之前輸出的 JPG；需要獨立樣本集合時請使用新的輸出目錄。

### CPU 與 GPU 資源設定

在 `utils/config.yaml` 的 `mission` 區塊設定預設值，也可用命令列覆寫：

```yaml
mission:
  cpu_cores: 8       # 整個 segmentation 任務的邏輯 CPU 核心預算
  gpu_ids: [1, 2]    # 使用 GPU 1、2，共兩張；[] 表示只使用 CPU
  cpu_workers: 4    # 純 CPU 模式的最大工作程序數
```

```bash
# 總共 8 個 CPU 核心、2 張 GPU；每個 GPU 程序分配 4 個核心
bash mission.sh segment-face --cores 8 --gpus 1,2

# 總共 12 個 CPU 核心、3 張 GPU
bash mission.sh segment-face --cores 12 --gpus 0,1,2

# 只使用 CPU：8 個核心分給 4 個程序，每個程序 2 個核心
bash mission.sh segment-face --cores 8 --gpus none --cpu-workers 4

# 只查看影片數、GPU 與 CPU 分配，不進行推論或輸出圖片
bash mission.sh segment-face --cores 8 --gpus 1,2 --dry-run
```

`--cores` 是**總邏輯 CPU 核心數**，不是程序數。各工作程序綁定互不重疊的 CPU 集合，
並限制推論執行緒數；這不會獨占 CPU，其他使用者仍可使用相同核心。
每張 GPU 啟動一個工作程序，影片依序輪流分配；每個程序都可處理 normal 與 anomaly。
核心數至少需等於指定的 GPU 數量，且不得超過目前程序可使用的 CPU 數量。
GPU 編號以 PyTorch 可見裝置為準，受 `CUDA_VISIBLE_DEVICES` 影響；清單長度就是指定 GPU 數量。
若影片少於程序數，只啟動有影片可處理的程序；純 CPU 模式也會以核心數限制程序數。

資料路徑、`parts`、`frames_per_video`、遮罩外擴比例及輸出設定位於同檔的 `segment_face`。
`parts: null` 表示全部分包，`frames_per_video: null` 表示全部有效幀；命令列參數優先於 YAML。
normal 與 anomaly 分別寫入 `normal_output_dir`、`anomaly_output_dir`，輸出目錄只儲存 JPG。

`mission.sh segment-face` 預設處理兩類，可用 `--labels real` 或 `--labels fake` 指定類別。
此入口使用 `--gpus` 與 `--cores`，不接受 `--device` 或 `--cpu-threads`；輸出路徑可用
`--normal-output-dir` 與 `--anomaly-output-dir` 覆寫。只跑單一類別時也可用 `--output-dir`。

```bash
bash mission.sh segment-face --labels real --cores 8 --gpus 1,2
```

### RetinaFace 原始影片切臉

```bash
# 先測試兩支影片，每支抽 4 幀
bash mission.sh crop-face --limit 2 --num-frames 4

# 指定 CPU 工作程序數與每個程序的推論執行緒數
bash mission.sh crop-face --workers 4 --cpu-threads 2 --device cpu
```

此任務使用 `crop_face` 區塊，抽幀張數參數是 `--num-frames`；SegFace 的對應參數則是
`--frames-per-video`。舊指令 `bash mission.sh --workers 4` 仍會執行 RetinaFace。
`--cores`、`--gpus`、`--cpu-workers` 是 SegFace 任務的參數；RetinaFace 沿用原本的
`--workers`、`--cpu-threads` 與 `--device`，不套用上述多 GPU 分配與 CPU 綁定。

### 完成、失敗與重跑

`segment-face` 會等待所有程序完成，成功時顯示 `All segmentation tasks completed.`。
任一任務失敗會停止其他任務並回傳非零狀態；按 `Ctrl+C` 會停止這次啟動的工作程序。
重跑時，SegFace 依 JPG 是否已存在決定略過；已有 JPG 不會自動比對模型或裁切設定，
更換設定後請加 `--overwrite` 或改用新輸出目錄。處理統計顯示於終端機。

## SegFace 分割人臉裁切（目前資料準備流程）

使用作者官方 **SegFace Swin-B / CelebAMask-HQ / 512，AAAI 2025**，權重及模型程式位於
`models/segface/`。來源為 `/ssd8/chihyu/Dataset/DeepFake_Dataset/DFDC-Frame`，
輸出至 `/ssd8/chihyu/Dataset/DeepFake_Dataset/` 下的 normal、anomaly 目錄。
此流程只做 segmentation 裁切，不訓練 feature bank 分類器。

輸出依來源標籤分流：real 到 `DFDC-Frame-CropFace_normal`，fake 到 `DFDC-Frame-CropFace_anomaly`。
目前測試範圍為 `dfdc_train_part_0`～`dfdc_train_part_10`（含 0、10），每支影片均勻抽 10 張有效人臉；
不足 10 張則全部使用，不重複補幀。抽樣保留原幀編號。

執行指令、平行任務和參數範例見上方「mission.sh 使用方式」。

模型包含皮膚、五官、眼鏡與耳朵，先做閉運算、遮罩外擴與填洞，再裁切外接框。
預設外擴半徑為來源圖片較短邊的 5%（224 像素時為 12 像素），受圖片邊界限制。
輸出保留裁切後的原生長寬、不強制拉成正方形。分割不一定完全正確，先查看少量結果。
遮罩外預設黑色；設定 `segment_face.background: keep` 可改為只裁外接框並保留框內背景。
外擴不可能恢復前一道 RetinaFace 裁切已丟掉的內容。

圖片直接存於輸出根目錄，以連字號串接來源名稱，保留原幀編號及補零位數：

```text
DFDC-Frame-CropFace_normal/
  dfdc_train_part_0-adwbthsgqb-frame_000000.jpg
  dfdc_train_part_0-adwbthsgqb-frame_000010.jpg
```

輸出目錄只存 JPG；遮罩只在記憶體中用於外擴與裁切，不寫入遮罩、JSON 或 lock 檔案。
處理統計只顯示於終端機。重跑依 JPG 是否存在決定略過；沒有人臉的圖片不輸出，下次仍可重試。
因為不保存設定紀錄，調整模型或裁切參數後請使用 `--overwrite` 或新的 `--normal-output-dir`／`--anomaly-output-dir`。
來源 metadata 仍會讀取，以分辨 real／fake，但不複製到輸出目錄。

資料與裁切參數集中在 `utils/config.yaml` 的 `segment_face`；GPU 任務分配位於 `mission`。
`parts: null` 表示全部分包，`frames_per_video: null` 表示使用全部有效幀。
`script/segment_face.py` 管理資料與命名；`script/face_parser.py` 管理模型推論；
`script/face_crop.py` 管理遮罩外擴與裁切。官方模型原始碼與版本紀錄獨立存於 `models/segface/`。
下方舊版 feature bank 仍讀取巢狀的 DFDC-Frame；目前扁平的 segmentation 輸出請使用上方 `patch-bank` 任務。

參考：[SegFace 官方程式](https://github.com/Kartik-3004/SegFace)、
[作者權重](https://huggingface.co/kartiknarayan/SegFace)、[論文](https://arxiv.org/abs/2412.08647)。

## Real feature bank：NPY 與 JPG 預覽

`bash run.sh --task feature-bank` 自動啟用 `pt230`，透過 `main.py` 執行 `feature_bank.stage: build`。
所有參數集中在 `utils/config.yaml` 的 `feature_bank`；模型共用頂層 `model_path`。
目前只從已裁切的人臉中選 real 建庫，不做校準、評估或影片特徵聚合。

每張人臉的 DINOv3 最後一層 patch 特徵獨立存成 `14×14×768` 的 NPY，預設 float16，
保留所有 patch 通道，排除 CLS 與 register tokens。NPY 不經 PCA 壓縮或額外 L2 正規化。
每個來源資料夾另存一張 `preview.jpg`：以第一張入選的人臉為代表，並排繪製人臉與 PCA 特徵圖。
Matplotlib 圖保留裁切後的人臉比例，座標軸標示寬、高像素；特徵網格放大顯示於相同座標範圍。
此座標僅用於展示，不是原始影片座標，也不表示異常定位。
所有預覽共用由最多 32 張 bank 圖片擬合的 PCA 與色彩範圍，設定存於 `visualization.json`。
預覽使用的那張人臉也會保留 NPY；JPG 僅供查看，完整數值位於 NPY。

```text
RAG/normal/dfdc_real_bank_v2/
  bank_config.json
  splits.json
  visualization.json
  manifest.jsonl
  dfdc_train_part_18/
    <原影片名稱>/
      preview.jpg          # 此資料夾唯一的 JPG：代表人臉與特徵預覽
      frame_000000.npy     # 單張人臉 [14,14,768]
      frame_000000.json    # 對應來源、形狀、精度
      frame_000010.npy
      frame_000010.json
      ...
```

沿用來源資料分包、影片資料夾與幀檔名；影片名稱僅用來整理來源。
預設以 seed=42 選最多 50 個 real 來源資料夾，每個最多 32 張有效人臉，即最多 1,600 個 NPY
與 50 張預覽圖。`bank_videos` 控制來源資料夾數、`max_frames` 控制各資料夾圖片數。
`build` 不讀取 fake 裁切內容，也不要求 calibration／evaluation 資料。
目前使用 DFDC test 清單作前測資料池；未來正式評估需另劃分訓練與測試來源。

```bash
# 使用可用的 GPU；auto 預設用 cuda:0，沒有 CUDA 則用 CPU
bash run.sh --task feature-bank --device cuda:1

# 少量測試：2 個 real 來源資料夾，各 2 張人臉
bash run.sh --task feature-bank --exper bank_trial --bank-videos 2 --max-frames 2 --device cuda:1

# 核心測試
conda run -n pt230 python -m unittest discover -s tests -v
```

同設定重跑會驗證並沿用已有 NPY；缺少的預覽可直接從 NPY 重建。
每張來源與 NPY 路徑記錄於 `manifest.jsonl`，JSON 保存圖片大小、修改時間及陣列形狀。
更換資料、模型或抽樣設定時使用新的 `--exper` 名稱；裝置與 batch size 可直接調整。
目前 `build` 的目錄格式用於建庫，不直接接入舊版 `evaluate/predict` 的索引格式。

程式分工：`script/feature_bank.py` 管理流程；`script/bank_data.py` 負責來源與標籤；
`script/bank_encoder.py` 負責模型提取；`script/bank_export.py` 負責 NPY、來源紀錄與 Matplotlib 預覽。舊版前測流程保留，需另開實驗使用。

## RetinaFace 切臉

`mission.sh crop-face` 會啟用 `pt230`，執行 `script/crop_face.py`，以多個獨立工作程序
平行處理不同影片。每個程序只建立一次 RetinaFace 模型，預設 8 個 CPU 工作，
每個工作使用 2 條推論執行緒。所有工作完成後腳本才結束；任何影片失敗會回傳非零狀態。

```bash
# 安裝切臉所需依賴（目前 pt230 已備妥）
conda run -n pt230 python -m pip install -r requirements-crop.txt

# 先測試兩支影片
bash mission.sh crop-face --limit 2 --num-frames 4

# 依照 YAML 處理完整資料集
bash mission.sh crop-face
```

在 `utils/config.yaml` 的 `crop_face` 區塊設定工作數、CPU 執行緒數、裝置、
偵測門檻、裁切邊界、輸出大小及快取位置；抽幀數、資料路徑、切分、影片數量限制
與覆寫設定共用頂層設定。命令列可覆寫，例如 `--workers 4 --device cpu`。
指定 `cuda:0` 時，所有工作共用該 GPU，每個工作各占一份模型記憶體；建議先以
`--workers 1` 測試。GPU 需 TensorFlow 可用的 CUDA 環境，與 PyTorch 是否可用 GPU 無關。

程式使用 [retina-face](https://github.com/serengil/retinaface)，首次執行會下載權重
至 `crop_face.cache_dir` 下的 `.deepface/weights/`，之後重用快取。
均勻抽幀後，每幀選擇最大人臉，從原始畫面裁切並縮放為 224×224 JPEG。
預設每側保留 20% 邊界；不做人臉對齊或跨幀身分追蹤。
未偵測到人臉時不輸出圖片，會在紀錄中標為 `no_face`。

目前預設處理 DFDC test 清單，輸出至 `/ssd8/chihyu/Dataset/DeepFake_Dataset/DFDC`，
並保留來源影片的子目錄結構，例如：

```text
/ssd8/chihyu/Dataset/DeepFake_Dataset/DFDC/dfdc_train_part_13/xfxfulmigh/
  frame_000000.jpg
  ...
  metadata.json
```

`metadata.json` 包含來源、標籤（real=0、fake=1）、資料切分、FPS、原始幀索引、
原圖座標的偵測框與裁切框、信心分數及設定。已完成影片預設略過，
裁切參數改變時需加 `--overwrite` 或另選輸出目錄。原始影片保持不變。
feature bank 讀取已裁切的人臉；下方保留原始影片 CLS 提取功能。

## 提取影片表徵

所有環境路徑與執行參數集中於 `utils/config.yaml`，`main.py` 會自動讀取。
原始影片提取入口為 `bash run.sh --task extract`：腳本會切換至專案目錄、載入 Conda、
啟用 `pt230`，再執行 `main.py`。需事先安裝 Conda 並建立 `pt230` 環境，
確保終端機能執行 `conda`；環境啟用失敗時腳本會停止。
修改 YAML 後即可執行：

```bash
bash run.sh --task extract
```

命令列參數優先於 YAML；相對路徑一律以專案根目錄為基準。
`overwrite: true` 可用 `--no-overwrite` 暫時覆寫。
YAML 使用 `pt230` 環境中的 PyYAML 讀取。

使用本機 DINOv3 ViT-B/16，均勻抽取完整畫面的 RGB 幀，套用模型內附的
224×224 前處理。模型保持 eval 模式且不計算梯度。
每幀取最後一層 CLS token（768 維），影片表徵為每幀 CLS 的平均，不額外做 L2 正規化。
不包含人臉偵測、裁切或模型訓練。

先測試一支影片（CPU）：

```bash
bash run.sh --task extract --limit 1 --num-frames 4 --batch-size 2
```

提取整個 Celeb-DF（請選擇可用 GPU）：

```bash
bash run.sh --task extract --device cuda:0 --num-frames 32 --batch-size 8
```

可用 `--split train` 或 `--split test` 選取子集；test 依據官方
`List_of_testing_videos.txt`，train 是其餘影片，不另做身分切分。
`--data-root`、`--model-path`、`--output-dir` 可修改預設路徑。
影片不足指定幀數時，提取所有幀，不重複補幀；解碼失敗會停止並顯示影片與幀索引。

輸出預設為 `outputs/features/<原始類別>/<影片名稱>.pt`，來源資料保持不變。
已存在的輸出預設略過；更換模型或抽幀參數時，請使用新的輸出目錄或加上 `--overwrite`。

```python
import torch

data = torch.load("outputs/features/Celeb-real/id0_0000.pt", weights_only=True)
frame_features = data["frame_features"]  # [實際抽幀數, 768]
video_feature = data["video_feature"]    # [768]
```

其他欄位：`frame_indices`（從 0 開始）、`frame_count`、`fps`、`source`、
`label`（本程式定義 real=0、fake=1）、`split` 和 `settings`。
請將範例檔名換成實際輸出檔名。不同影片的表徵逐一寫入，不將整個資料集載入記憶體。
