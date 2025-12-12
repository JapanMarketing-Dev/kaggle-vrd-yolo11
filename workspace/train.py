#!/usr/bin/env python3
"""
VRD-IU Track B - YOLO11 Training Script
========================================
YOLO11を使用してドキュメントのキー情報検出モデルをトレーニング。

上位解法のアプローチ:
- 問題を「Object Detection」として再定義
- 12種類のキークエリを12クラスとして扱う
- YOLO11の高速・高精度な検出能力を活用
"""

from ultralytics import YOLO
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="YOLO11 Training for VRD-IU Track B")
    parser.add_argument("--model", default="yolo11m.pt", 
                        help="Base model (yolo11n/s/m/l/x.pt)")
    parser.add_argument("--data", default="/data/yolo_dataset/dataset.yaml",
                        help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for training")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--device", default="0",
                        help="Device (0, 0,1, cpu)")
    parser.add_argument("--project", default="/outputs",
                        help="Output project directory")
    parser.add_argument("--name", default="vrd_yolo11",
                        help="Experiment name")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of dataloader workers")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 VRD-IU Track B - YOLO11 Training")
    print("=" * 60)
    print(f"📦 Base Model: {args.model}")
    print(f"📂 Dataset: {args.data}")
    print(f"🔢 Epochs: {args.epochs}")
    print(f"📐 Image Size: {args.imgsz}")
    print(f"📦 Batch Size: {args.batch}")
    print(f"🖥️ Device: {args.device}")
    print()
    
    # モデルをロード
    print("📥 Loading model...")
    model = YOLO(args.model)
    
    # トレーニング
    print("🏃 Starting training...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        # 最適化設定
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        # データ拡張
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,  # ドキュメントは回転しない
        translate=0.1,
        scale=0.5,
        shear=0.0,
        flipud=0.0,  # ドキュメントは上下反転しない
        fliplr=0.0,  # ドキュメントは左右反転しない
        mosaic=0.0,  # ドキュメントはモザイクなし
        mixup=0.0,
        # その他
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )
    
    print()
    print("=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"📊 Best model: {args.project}/{args.name}/weights/best.pt")
    print(f"📊 Last model: {args.project}/{args.name}/weights/last.pt")
    
    # バリデーション結果を表示
    if results:
        print()
        print("📈 Validation Results:")
        print(f"   mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"   mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")


if __name__ == "__main__":
    main()
