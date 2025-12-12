# VRD-IU 2024 Track B - YOLO11 Object Detection Approach

## 🎯 概要

**VRD-IU 2024 Track B** の上位解法に基づき、問題を **Object Detection** として再定義し、最新の **YOLO11** で解決するアプローチ。

### なぜこのアプローチか？

| アプローチ | MAP@0.5 | 問題点 |
|-----------|---------|--------|
| VLM (Qwen2.5-VL) fine-tuning | 0.42 | VLMはbbox予測が弱い |
| OCR + テキストマッチング | 0.18 | 空間ヒューリスティックが不正確 |
| DeepSeek-OCR | 0.00 | タスクに不適切 |
| **Object Detection (上位解法)** | **0.98** | ✅ 問題の本質に合致 |

### 上位解法の分析

1. **1位 (MAP 0.989)**: Large Margin Feature Matching + Heuristics
   - [論文: arxiv.org/abs/2502.07442](https://arxiv.org/abs/2502.07442)
   
2. **Data Augmentation + Object Detection**
   - LayoutLMv3/DiT + Faster R-CNN/Mask R-CNN
   - Augraphyで手書き風シミュレーション
   - [論文: arxiv.org/abs/2502.06132](https://arxiv.org/abs/2502.06132)

### 本アプローチ

12種類のキークエリを **12クラスの物体検出問題** として再定義：

| Class ID | キークエリ |
|----------|-----------|
| 0 | company name |
| 1 | company ACN/ARSN |
| 2 | substantial holder name |
| 3 | holder ACN/ARSN |
| 4 | There was a change in... |
| 5 | The previous notice was dated |
| 6 | The previous notice was given... |
| 7 | class of securities |
| 8 | Previous notice Person's notes |
| 9 | Previous notice Voting power |
| 10 | Present notice Person's votes |
| 11 | Present notice Voting power |

## 📁 プロジェクト構成

```
kaggle-vrd-yolo/
├── Dockerfile              # YOLO11 Docker環境
├── docker-compose.yml      # Docker Compose設定
├── README.md               # このファイル
├── workspace/
│   ├── convert_to_yolo.py  # データ変換スクリプト
│   ├── train.py            # トレーニングスクリプト
│   └── evaluate.py         # 評価スクリプト
├── data/
│   ├── train_images/       # トレーニング画像
│   ├── val_images/         # バリデーション画像
│   ├── train_dataframe.csv # トレーニングアノテーション
│   ├── val_dataframe.csv   # バリデーションアノテーション
│   └── yolo_dataset/       # 変換後のYOLOデータセット
└── outputs/                # トレーニング出力
```

## 🚀 使用方法

### 1. Docker環境のビルドと起動

```bash
cd /home/ubuntu/Documents/kaggle-vrd-yolo

# Dockerイメージをビルド
docker build -t vrd-yolo .

# コンテナを起動
docker run --gpus all -it --rm \
  -v $(pwd)/workspace:/workspace \
  -v $(pwd)/data:/data \
  -v $(pwd)/outputs:/outputs \
  --ipc=host \
  vrd-yolo bash
```

### 2. データをYOLO形式に変換

```bash
python /workspace/convert_to_yolo.py \
  --train-csv /data/train_dataframe.csv \
  --val-csv /data/val_dataframe.csv \
  --image-dir /data \
  --output-dir /data/yolo_dataset
```

### 3. YOLO11トレーニング（30分程度）

```bash
python /workspace/train.py \
  --model yolo11m.pt \
  --data /data/yolo_dataset/dataset.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --name vrd_yolo11
```

**モデルサイズの選択肢:**
| モデル | パラメータ | mAP (COCO) | 速度 |
|--------|-----------|------------|------|
| yolo11n.pt | 2.6M | 39.5 | 最速 |
| yolo11s.pt | 9.4M | 47.0 | 高速 |
| **yolo11m.pt** | 20.1M | 51.5 | **推奨** |
| yolo11l.pt | 25.3M | 53.4 | 高精度 |
| yolo11x.pt | 56.9M | 54.7 | 最高精度 |

### 4. トレーニング進捗の確認

バックグラウンド実行時の進捗確認：

```bash
# ターミナル出力を確認（リアルタイム）
tail -f /home/ubuntu/.cursor/projects/home-ubuntu-Documents/terminals/304857.txt

# 実行中のDockerコンテナのログを確認
docker logs -f $(docker ps -q --filter ancestor=vrd-yolo)
```

トレーニング中に生成されるファイル：
```
outputs/vrd_yolo11/
├── weights/
│   ├── best.pt      # 最良モデル
│   └── last.pt      # 最新モデル
├── train_batch0.jpg # トレーニングバッチのサンプル
├── labels.jpg       # クラス分布の可視化
├── results.csv      # エポックごとの結果
└── args.yaml        # トレーニング設定
```

### 5. 評価

トレーニング完了後、以下のコマンドで評価：

```bash
docker run --gpus all --rm \
  -v /home/ubuntu/Documents/kaggle-vrd-yolo/workspace:/workspace \
  -v /home/ubuntu/Documents/kaggle-vrd-yolo/data:/data \
  -v /home/ubuntu/Documents/kaggle-vrd-yolo/outputs:/outputs \
  --ipc=host \
  vrd-yolo \
  python /workspace/evaluate.py \
    --model /outputs/vrd_yolo11/weights/best.pt \
    --val-csv /data/val_dataframe.csv \
    --image-dir /data \
    --debug
```

評価オプション：
| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--model` | best.pt | モデルパス |
| `--conf` | 0.25 | 信頼度閾値 |
| `--iou-threshold` | 0.5 | IoU閾値 |
| `--num-samples` | 0 (全件) | サンプル数 |
| `--debug` | False | デバッグ出力 |

## 🔧 技術スタック

- **モデル**: [Ultralytics YOLO11](https://huggingface.co/Ultralytics/YOLO11) (2024年最新)
- **フレームワーク**: ultralytics>=8.3.0, PyTorch
- **環境**: Docker + NVIDIA GPU

## 📊 実験結果 🎉

### トレーニング結果（50エポック、約2.5分）

| 指標 | 値 |
|------|-----|
| mAP@0.5 | **0.994 (99.4%)** |
| mAP@0.5:0.95 | **0.80 (80.0%)** |
| Precision | 0.994 |
| Recall | 0.988 |

### 評価結果

| 指標 | VLMアプローチ | **YOLO11（全904件）** |
|------|--------------|----------------------|
| MAP@0.5 | 0.42 | **0.9768 (97.7%)** ✅ |
| Average IoU | 0.42 | **0.8667** |
| True Positives | 42% | **883/904 (97.7%)** |
| 推論速度 | ~2秒/画像 | **~0.01秒/画像** |
| VRAM使用量 | ~16GB | ~4GB |

### サンプル予測結果

| キークエリ | IoU |
|-----------|-----|
| company name | 0.837 |
| company ACN/ARSN | 0.927 |
| substantial holder name | 0.881 |
| holder ACN/ARSN | 0.875 |
| change date | 0.791 |
| previous notice dated | 0.961 |
| previous notice given | 0.924 |
| class of securities | 0.872 |
| Previous notice Person's notes | 0.881 |
| Present notice Person's votes | 0.898 |

## 📝 参考文献

- [VRD-IU 2024 Competition](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/DM76.pdf)
- [1位解法: Hierarchical Document Parsing](https://arxiv.org/abs/2502.07442)
- [Data Augmentation Approach](https://arxiv.org/abs/2502.06132)
- [Ultralytics YOLO11](https://docs.ultralytics.com/)

## 📅 更新履歴

### 2025-12-12: YOLO11トレーニング完了 🎉
- **MAP@0.5: 0.9768 (97.7%)** を達成（全904サンプル評価）
- Average IoU: 0.8667
- True Positives: 883/904
- トレーニング時間: 約2.5分（50エポック）
- VLMアプローチ（MAP 0.42）から **+133%改善**

### 2025-12-11: YOLOアプローチ採用
- VLMアプローチ（MAP 0.42）からObject Detectionアプローチへ移行
- 上位解法の分析に基づきYOLO11を選択
- データ変換・トレーニング・評価パイプライン構築
