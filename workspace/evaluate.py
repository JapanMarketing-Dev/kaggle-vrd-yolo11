#!/usr/bin/env python3
"""
VRD-IU Track B - YOLO11 Evaluation Script
==========================================
トレーニング済みYOLO11モデルを評価し、MAP@0.5を計算。
"""

import os
import ast
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm
import argparse


# キークエリからクラスIDへのマッピング
KEY_TO_CLASS = {
    "company name": 0,
    "company ACN/ARSN": 1,
    "substantial holder name": 2,
    "holder ACN/ARSN": 3,
    "There was a change in the interests of the substantial holder on": 4,
    "The previous notice was dated": 5,
    "The previous notice was given to the company on": 6,
    "class of securities": 7,
    "Previous notice Person's notes": 8,
    "Previous notice Voting power": 9,
    "Present notice Person's votes": 10,
    "Present notice Voting power": 11,
}


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x, y, w, h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to x1, y1, x2, y2 format
    box1_x1, box1_y1 = x1, y1
    box1_x2, box1_y2 = x1 + w1, y1 + h1
    box2_x1, box2_y1 = x2, y2
    box2_x2, box2_y2 = x2 + w2, y2 + h2
    
    # Calculate intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def yolo_to_bbox(yolo_box, img_width, img_height):
    """YOLO形式 [x_center, y_center, w, h] → [x, y, w, h]"""
    x_center, y_center, w, h = yolo_box
    x = (x_center - w / 2) * img_width
    y = (y_center - h / 2) * img_height
    w = w * img_width
    h = h * img_height
    return [x, y, w, h]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/outputs/vrd_yolo11/weights/best.pt",
                        help="Path to trained model")
    parser.add_argument("--val-csv", default="/data/val_dataframe.csv",
                        help="Validation CSV path")
    parser.add_argument("--image-dir", default="/data",
                        help="Image directory")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Number of samples (0=all)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for MAP calculation")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode")
    args = parser.parse_args()
    
    print("=" * 60)
    print("📊 VRD-IU Track B - YOLO11 Evaluation")
    print("=" * 60)
    
    # モデルをロード
    print(f"📥 Loading model: {args.model}")
    model = YOLO(args.model)
    
    # データをロード
    df = pd.read_csv(args.val_csv)
    if args.num_samples > 0:
        df = df.head(args.num_samples)
    print(f"📦 Samples: {len(df)}")
    
    # 評価
    results_list = []
    true_positives = 0
    total_iou = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        # 画像パスを特定
        image_path = None
        for subdir in ['train_images', 'val_images', 'test_images', 'handwritten_images']:
            candidate = Path(args.image_dir) / subdir / row['file']
            if candidate.exists():
                image_path = candidate
                break
        
        if image_path is None:
            continue
        
        # 画像サイズを取得
        with Image.open(image_path) as img:
            img_width, img_height = img.size
        
        # Ground Truth
        gt_bbox = ast.literal_eval(row['label(bbox)'])
        key_query = row['key_fix_text']
        target_class = KEY_TO_CLASS.get(key_query)
        
        if target_class is None:
            continue
        
        # 推論
        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            verbose=False,
        )
        
        # 対象クラスの予測を取得
        best_pred = None
        best_conf = 0
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = boxes.conf[i].item()
                
                if cls_id == target_class and conf > best_conf:
                    # YOLO形式 → 通常形式
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    pred_bbox = [x1, y1, x2 - x1, y2 - y1]
                    best_pred = pred_bbox
                    best_conf = conf
        
        # IoU計算
        if best_pred is not None:
            iou = calculate_iou(best_pred, gt_bbox)
            total_iou += iou
            
            if iou >= args.iou_threshold:
                true_positives += 1
            
            if args.debug and idx < 10:
                print(f"  Query: {key_query}")
                print(f"  Pred: {[f'{x:.1f}' for x in best_pred]}, GT: {gt_bbox}, IoU: {iou:.3f}")
        else:
            if args.debug and idx < 10:
                print(f"  Query: {key_query} - No prediction")
        
        results_list.append({
            'file': row['file'],
            'key': key_query,
            'gt_bbox': gt_bbox,
            'pred_bbox': best_pred,
            'iou': iou if best_pred else 0,
        })
    
    # 結果を表示
    n_samples = len(results_list)
    map_score = true_positives / n_samples if n_samples > 0 else 0
    avg_iou = total_iou / n_samples if n_samples > 0 else 0
    
    print()
    print("=" * 60)
    print("📊 Results")
    print("=" * 60)
    print(f"Samples: {n_samples}")
    print(f"MAP@{args.iou_threshold}: {map_score:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"True Positives: {true_positives} / {n_samples}")


if __name__ == "__main__":
    main()
