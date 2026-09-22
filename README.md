# DeepFakeSecurity

## mission.sh 使用方式

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
| `bash mission.sh` | 預設執行 `crop-face` | 同上 |

兩個任務分別啟動，不會自動串接。`segment-face` 讀取 `segment_face.input_dir` 指定的
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
目前下方 feature bank 仍讀取原本巢狀的 DFDC-Frame；扁平資料集接入 bank 的讀取流程尚未變更。

參考：[SegFace 官方程式](https://github.com/Kartik-3004/SegFace)、
[作者權重](https://huggingface.co/kartiknarayan/SegFace)、[論文](https://arxiv.org/abs/2412.08647)。

## Real feature bank：NPY 與 JPG 預覽

`bash run.sh` 自動啟用 `pt230`，透過 `main.py` 執行 `feature_bank.stage: build`。
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
RAG/general/dfdc_real_bank_v2/
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
bash run.sh --device cuda:1

# 少量測試：2 個 real 來源資料夾，各 2 張人臉
bash run.sh --experiment bank_trial --bank-videos 2 --max-frames 2 --device cuda:1

# 核心測試
conda run -n pt230 python -m unittest discover -s tests -v
```

同設定重跑會驗證並沿用已有 NPY；缺少的預覽可直接從 NPY 重建。
每張來源與 NPY 路徑記錄於 `manifest.jsonl`，JSON 保存圖片大小、修改時間及陣列形狀。
更換資料、模型或抽樣設定時使用新的 `--experiment` 名稱；裝置與 batch size 可直接調整。
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
