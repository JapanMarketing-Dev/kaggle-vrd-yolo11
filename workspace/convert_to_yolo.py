#!/usr/bin/env python3
"""
VRD-IU Track B - YOLO11形式へのデータ変換
==========================================
CSVデータをYOLO形式に変換し、トレーニング用データセットを作成。

12種類のキークエリを12クラスとして扱う：
0: company_name
1: company_acn_arsn
2: substantial_holder_name
3: holder_acn_arsn
4: change_date
5: previous_notice_dated
6: previous_notice_given
7: class_of_securities
8: previous_notice_persons_votes
9: previous_notice_voting_power
10: present_notice_persons_votes
11: present_notice_voting_power
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import yaml


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

CLASS_NAMES = [
    "company_name",
    "company_acn_arsn", 
    "substantial_holder_name",
    "holder_acn_arsn",
    "change_date",
    "previous_notice_dated",
    "previous_notice_given",
    "class_of_securities",
    "previous_notice_persons_votes",
    "previous_notice_voting_power",
    "present_notice_persons_votes",
    "present_notice_voting_power",
]


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    [x, y, width, height] → YOLO形式 [x_center, y_center, width, height] (正規化)
    """
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return x_center, y_center, w_norm, h_norm


def process_dataset(csv_path, image_dir, output_dir, split_name):
    """CSVデータをYOLO形式に変換"""
    df = pd.read_csv(csv_path)
    
    # 出力ディレクトリ
    images_out = Path(output_dir) / "images" / split_name
    labels_out = Path(output_dir) / "labels" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    
    # 画像ごとにアノテーションをグループ化
    grouped = df.groupby('file')
    
    processed = 0
    skipped = 0
    
    for image_file, group in tqdm(grouped, desc=f"Processing {split_name}"):
        # 画像パスを特定
        image_path = None
        for subdir in ['train_images', 'val_images', 'test_images', 'handwritten_images']:
            candidate = Path(image_dir) / subdir / image_file
            if candidate.exists():
                image_path = candidate
                break
        
        if image_path is None:
            skipped += 1
            continue
        
        # 画像サイズを取得
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            skipped += 1
            continue
        
        # 画像をコピー
        dst_image = images_out / image_file
        if not dst_image.exists():
            shutil.copy(image_path, dst_image)
        
        # ラベルファイルを作成
        label_file = labels_out / (Path(image_file).stem + ".txt")
        
        with open(label_file, 'w') as f:
            for _, row in group.iterrows():
                key_query = row['key_fix_text']
                
                # キーをクラスIDに変換
                class_id = KEY_TO_CLASS.get(key_query)
                if class_id is None:
                    continue
                
                # bboxを解析
                try:
                    import ast
                    bbox = ast.literal_eval(row['label(bbox)'])
                    x_center, y_center, w_norm, h_norm = convert_bbox_to_yolo(
                        bbox, img_width, img_height
                    )
                    
                    # 値のバリデーション
                    if all(0 <= v <= 1 for v in [x_center, y_center, w_norm, h_norm]):
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                except Exception as e:
                    continue
        
        processed += 1
    
    print(f"  {split_name}: {processed} images processed, {skipped} skipped")
    return processed


def create_yaml_config(output_dir):
    """YOLO用のデータセット設定ファイルを作成"""
    config = {
        'path': str(Path(output_dir).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES,
    }
    
    yaml_path = Path(output_dir) / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Dataset config saved: {yaml_path}")
    return yaml_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="/data/train_dataframe.csv")
    parser.add_argument("--val-csv", default="/data/val_dataframe.csv")
    parser.add_argument("--image-dir", default="/data")
    parser.add_argument("--output-dir", default="/data/yolo_dataset")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔄 VRD-IU → YOLO11 データ変換")
    print("=" * 60)
    print(f"📂 Train CSV: {args.train_csv}")
    print(f"📂 Val CSV: {args.val_csv}")
    print(f"📂 Image Dir: {args.image_dir}")
    print(f"📂 Output Dir: {args.output_dir}")
    print(f"📊 Classes: {len(CLASS_NAMES)}")
    print()
    
    # トレーニングデータ変換
    print("📦 Training data...")
    process_dataset(args.train_csv, args.image_dir, args.output_dir, "train")
    
    # バリデーションデータ変換
    print("📦 Validation data...")
    process_dataset(args.val_csv, args.image_dir, args.output_dir, "val")
    
    # YAML設定ファイル作成
    print()
    create_yaml_config(args.output_dir)
    
    print()
    print("✅ データ変換完了!")


if __name__ == "__main__":
    main()
