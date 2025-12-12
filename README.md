# VRD-IU 2024 Track B - YOLO11 Object Detection Approach

## 🏆 Kaggle コンペティション情報

| 項目 | 内容 |
|------|------|
| **コンペ名** | VRD-IU 2024 Track B: Form Key Information Localization |
| **Kaggle URL** | https://www.kaggle.com/competitions/vrd-iu2024-trackb |
| **主催** | IJCAI 2025 (国際人工知能学会) |
| **期間** | 2024年 |
| **達成スコア** | **MAP@0.5: 0.9834 (98.3%)** ✅ |

---

## 📖 コンペの背景と目的

### 何を解決するコンペ？

オーストラリアの金融書類（株主報告書）から、**重要な情報の位置を特定**するタスクです。

```
例：「会社名はどこに書いてある？」→ 画像上の座標（四角形）を回答
```

### なぜ重要？

- 📄 **大量の書類処理を自動化**: 人手で確認していた作業をAIが代替
- 💼 **金融・法務分野でのニーズ**: コンプライアンス、監査、データ入力の効率化
- 🔍 **OCRの次のステップ**: 文字を読むだけでなく「どこに何があるか」を理解

### タスクの具体例

入力画像（金融書類）から、以下12種類の情報の**位置（バウンディングボックス）**を予測：

| # | 検出対象（英語） | 説明（日本語） |
|---|-----------------|---------------|
| 0 | company name | 会社名 |
| 1 | company ACN/ARSN | 会社の登録番号 |
| 2 | substantial holder name | 大株主名 |
| 3 | holder ACN/ARSN | 株主の登録番号 |
| 4 | change date | 持株比率変更日 |
| 5 | previous notice dated | 前回届出日 |
| 6 | previous notice given | 前回届出提出日 |
| 7 | class of securities | 有価証券の種類 |
| 8 | Previous notice Person's votes | 前回の議決権数 |
| 9 | Previous notice Voting power | 前回の議決権比率 |
| 10 | Present notice Person's votes | 今回の議決権数 |
| 11 | Present notice Voting power | 今回の議決権比率 |

---

## 📊 評価指標の解説

### MAP@0.5（Mean Average Precision）とは？

```
MAP@0.5 = 「予測した四角形が、正解の四角形とどれだけ重なっているか」の平均スコア
```

| 値 | 意味 |
|----|------|
| 0.0 | 全く当たっていない |
| 0.5 | 半分程度正解 |
| **0.98** | **ほぼ完璧（今回の結果）** |
| 1.0 | 完全に正解 |

> 💡 **@0.5の意味**: 予測と正解の重なりが50%以上あれば「正解」とみなす基準

### IoU（Intersection over Union）とは？

```
IoU = 予測と正解が重なっている面積 ÷ 両方を合わせた面積
```

```
┌─────────┐
│  正解   │
│    ┌────┼────┐
│    │重複│    │
└────┼────┘    │
     │  予測   │
     └─────────┘

IoU = 重複部分 ÷ 全体（正解＋予測−重複）
```

| 値 | 意味 |
|----|------|
| 0.0 | 全く重なっていない |
| 0.5 | そこそこ重なっている |
| **0.87** | **かなり正確（今回の結果）** |
| 1.0 | 完全に一致 |

### True Positives（正解数）とは？

```
True Positives = 正しく検出できた件数

今回: 889件 / 904件 = 98.3% の成功率
     （失敗はわずか15件）
```

---

## 🎯 採用したアプローチ

### なぜ YOLO11（物体検出）を選んだか？

様々なアプローチを試した結果：

| アプローチ | 説明 | MAP@0.5 | 結果 |
|-----------|------|---------|------|
| VLM (Qwen2.5-VL) | AIに「〇〇はどこ？」と質問 | 0.42 | ❌ 精度不足 |
| OCR + マッチング | 文字認識後にテキスト検索 | 0.18 | ❌ 位置特定が曖昧 |
| DeepSeek-OCR | OCR特化モデル | 0.00 | ❌ タスクに不適合 |
| **YOLO11** | **12種類の「物体」として検出** | **0.98** | **✅ 採用** |

### Kaggle上位解法との比較

| 順位 | アプローチ | MAP@0.5 |
|------|-----------|---------|
| 🥇 1位 | Large Margin Feature Matching | 0.989 |
| 🥈 2位 | LayoutLMv3 + Faster R-CNN | 0.95+ |
| **本実装** | **YOLO11** | **0.983** |

> 📚 参考論文:
> - [1位解法](https://arxiv.org/abs/2502.07442)
> - [Data Augmentation解法](https://arxiv.org/abs/2502.06132)

---

## 📊 実験結果 🎉

### 最終スコア（全904サンプル評価）

| 指標 | 値 | 意味 |
|------|-----|------|
| **MAP@0.5** | **0.9834** | 98.3%の精度で位置を特定 |
| **Average IoU** | **0.8758** | 予測と正解が87.6%重複 |
| **True Positives** | **889/904** | 904件中889件成功 |
| **失敗件数** | **15件** | エラーはわずか1.7% |
| **推論速度** | **0.01秒/画像** | 1秒で100枚処理可能 |

### バージョン比較

| バージョン | モデル | 画像サイズ | 学習回数 | MAP@0.5 | 備考 |
|-----------|--------|----------|---------|---------|------|
| v1 | yolo11m (中) | 640px | 50回 | 0.9768 | 初期版 |
| **v2** | **yolo11l (大)** | **1024px** | **100回** | **0.9834** | **✅ 最終版** |

### 各項目の検出精度（IoU）

| 検出対象 | IoU | 精度 |
|---------|-----|------|
| company name | 0.837 | ⭐⭐⭐⭐ |
| company ACN/ARSN | 0.927 | ⭐⭐⭐⭐⭐ |
| substantial holder name | 0.881 | ⭐⭐⭐⭐ |
| holder ACN/ARSN | 0.875 | ⭐⭐⭐⭐ |
| change date | 0.791 | ⭐⭐⭐⭐ |
| previous notice dated | 0.961 | ⭐⭐⭐⭐⭐ |
| previous notice given | 0.924 | ⭐⭐⭐⭐⭐ |
| class of securities | 0.872 | ⭐⭐⭐⭐ |

---

## 📁 プロジェクト構成

```
kaggle-vrd-yolo/
├── Dockerfile              # 実行環境の定義
├── docker-compose.yml      # Docker設定
├── README.md               # このファイル
├── workspace/
│   ├── convert_to_yolo.py  # データ形式変換
│   ├── train.py            # モデル学習
│   └── evaluate.py         # 精度評価
├── data/
│   ├── train_images/       # 学習用画像（3,616枚）
│   ├── val_images/         # 評価用画像（904枚）
│   ├── train_dataframe.csv # 学習データ
│   ├── val_dataframe.csv   # 評価データ
│   └── yolo_dataset/       # YOLO形式データ
└── outputs/
    └── vrd_yolo11_v2/
        └── weights/
            └── best.pt     # 学習済みモデル（最終版）
```

---

## 🚀 使用方法

### 1. 環境構築

```bash
cd /home/ubuntu/Documents/kaggle-vrd-yolo

# Dockerイメージをビルド
docker build -t vrd-yolo .
```

### 2. データ変換（YOLO形式へ）

```bash
docker run --gpus all --rm \
  -v $(pwd)/workspace:/workspace \
  -v $(pwd)/data:/data \
  --ipc=host \
  vrd-yolo \
  python /workspace/convert_to_yolo.py \
    --train-csv /data/train_dataframe.csv \
    --val-csv /data/val_dataframe.csv \
    --image-dir /data \
    --output-dir /data/yolo_dataset
```

### 3. モデル学習（v2設定）

```bash
docker run --gpus all --rm \
  -v $(pwd)/workspace:/workspace \
  -v $(pwd)/data:/data \
  -v $(pwd)/outputs:/outputs \
  --ipc=host \
  vrd-yolo \
  python /workspace/train.py \
    --model yolo11l.pt \
    --data /data/yolo_dataset/dataset.yaml \
    --epochs 100 \
    --imgsz 1024 \
    --batch 8 \
    --device 0 \
    --name vrd_yolo11_v2
```

**モデルサイズの選択肢:**

| モデル | サイズ | 精度 | 用途 |
|--------|--------|------|------|
| yolo11n | 極小 | ★★☆☆☆ | テスト用 |
| yolo11s | 小 | ★★★☆☆ | 軽量用途 |
| yolo11m | 中 | ★★★★☆ | バランス型 |
| **yolo11l** | **大** | **★★★★★** | **推奨（v2で使用）** |
| yolo11x | 特大 | ★★★★★+ | 最高精度（時間かかる） |

### 4. 精度評価

```bash
docker run --gpus all --rm \
  -v $(pwd)/workspace:/workspace \
  -v $(pwd)/data:/data \
  -v $(pwd)/outputs:/outputs \
  --ipc=host \
  vrd-yolo \
  python /workspace/evaluate.py \
    --model /outputs/vrd_yolo11_v2/weights/best.pt \
    --val-csv /data/val_dataframe.csv \
    --image-dir /data \
    --num-samples 0
```

| オプション | 説明 | デフォルト |
|-----------|------|----------|
| `--model` | 学習済みモデルのパス | best.pt |
| `--conf` | 信頼度の閾値（低いほど多く検出） | 0.25 |
| `--num-samples` | 評価サンプル数（0=全件） | 0 |
| `--debug` | 詳細ログ出力 | False |

---

## 🔧 技術スタック

| 技術 | 説明 |
|------|------|
| [YOLO11](https://docs.ultralytics.com/) | 最新の物体検出モデル（2024年リリース） |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | YOLOの公式ライブラリ |
| PyTorch | ディープラーニングフレームワーク |
| Docker + NVIDIA GPU | 実行環境（再現性確保） |

---

## 📝 参考文献

| タイトル | リンク |
|---------|--------|
| Kaggleコンペページ | https://www.kaggle.com/competitions/vrd-iu2024-trackb |
| VRD-IU 2024 論文 | https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/DM76.pdf |
| 1位解法論文 | https://arxiv.org/abs/2502.07442 |
| Data Augmentation解法 | https://arxiv.org/abs/2502.06132 |
| YOLO11公式ドキュメント | https://docs.ultralytics.com/ |
| Hugging Face YOLO11 | https://huggingface.co/Ultralytics/YOLO11 |

---

## 📅 更新履歴

### 2025-12-12: YOLO11 v2 完成 🎉

- **MAP@0.5: 0.9834 (98.3%)** 達成
- モデル: yolo11l, 画像サイズ: 1024px, 100エポック
- VLMアプローチ（0.42）から **+134%改善**

### 2025-12-12: YOLO11 v1 初回トレーニング

- MAP@0.5: 0.9768 (97.7%)
- モデル: yolo11m, 画像サイズ: 640px, 50エポック

### 2025-12-11: プロジェクト開始

- VLM/OCRアプローチを検証後、物体検出へ方針転換
- Kaggle上位解法を分析し、YOLO11を採用

---

## ✅ まとめ

| 項目 | 結果 |
|------|------|
| **課題** | 金融書類から12種類の情報位置を特定 |
| **解法** | YOLO11による物体検出 |
| **精度** | MAP@0.5: 98.3%（904件中889件成功） |
| **速度** | 1画像あたり0.01秒 |
| **Kaggle順位相当** | 上位レベル（1位: 98.9%） |

---

## 🇯🇵 応用例：印鑑登録申請書の情報抽出

本アプローチは **日本の行政書類** にも適用可能です。以下は印鑑登録申請書への応用例です。

### 対象書類

**印鑑登録申請書**（市区町村役所で使用される公的書類）

### 抽出対象フィールド（検出クラス）

| Class ID | フィールド名 | 説明 |
|----------|-------------|------|
| 0 | 氏名 | 申請者の氏名 |
| 1 | フリガナ | 氏名のフリガナ |
| 2 | 生年月日 | 申請者の生年月日 |
| 3 | 住所 | 申請者の住所 |
| 4 | 電話番号 | 連絡先電話番号 |
| 5 | 届出印 | 登録する印鑑の印影 |
| 6 | 申請日 | 申請書の提出日 |
| 7 | 届出区分 | 新規登録・変更・廃止 等 |
| 8 | 本人確認書類 | 提示した身分証明書の種類 |
| 9 | 委任者情報 | 代理申請時の委任者情報 |

### 導入手順

#### Step 1: アノテーションデータの作成

```bash
# LabelImg等でアノテーション（YOLO形式で出力）
pip install labelImg
labelImg
```

アノテーション形式（1画像につき1つの `.txt` ファイル）：
```
# class_id x_center y_center width height（すべて0-1に正規化）
0 0.25 0.15 0.30 0.04    # 氏名
1 0.25 0.19 0.30 0.03    # フリガナ
2 0.75 0.15 0.20 0.04    # 生年月日
...
```

#### Step 2: dataset.yaml の作成

```yaml
# /data/inkan_dataset/dataset.yaml
path: /data/inkan_dataset
train: images/train
val: images/val

names:
  0: name              # 氏名
  1: furigana          # フリガナ
  2: birth_date        # 生年月日
  3: address           # 住所
  4: phone             # 電話番号
  5: seal_impression   # 届出印
  6: application_date  # 申請日
  7: application_type  # 届出区分
  8: id_document       # 本人確認書類
  9: delegator_info    # 委任者情報

nc: 10  # クラス数
```

#### Step 3: 学習の実行

```bash
docker run --gpus all --rm \
  -v $(pwd)/workspace:/workspace \
  -v $(pwd)/data:/data \
  -v $(pwd)/outputs:/outputs \
  --ipc=host \
  vrd-yolo \
  python /workspace/train.py \
    --model yolo11l.pt \
    --data /data/inkan_dataset/dataset.yaml \
    --epochs 100 \
    --imgsz 1024 \
    --batch 8 \
    --device 0 \
    --name inkan_yolo11
```

#### Step 4: 推論の実行

```python
from ultralytics import YOLO

# モデルの読み込み
model = YOLO("/outputs/inkan_yolo11/weights/best.pt")

# 推論
results = model.predict(
    source="申請書画像.png",
    conf=0.25,
    save=True
)

# 結果の取得
for result in results:
    for box in result.boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"クラス: {class_id}, 信頼度: {confidence:.2f}, 座標: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
```

### 期待される精度

| 条件 | 期待MAP@0.5 |
|------|-------------|
| 学習データ 100枚以上 | 90%+ |
| 学習データ 500枚以上 | 95%+ |
| 学習データ 1000枚以上 | 98%+ |

> 💡 **ポイント**: 印鑑登録申請書は市区町村ごとにフォーマットが異なります。
> 複数の自治体のフォーマットを学習データに含めることで、汎用性が向上します。

### 追加の考慮事項

| 項目 | 対応方法 |
|------|----------|
| **手書き文字** | Data Augmentationで手書き風ノイズを追加 |
| **印影の検出** | 赤色領域の検出精度向上のため、色空間変換を前処理に追加 |
| **複数ページ** | ページごとに処理し、結果を統合 |
| **個人情報保護** | 推論後のデータは暗号化・アクセス制限を実施 |
