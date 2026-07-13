"""
Prepares a raw, downloaded Kaggle landmark dataset for training.

Expects the source directory to contain one sub-folder per landmark class:
    data/raw/<dataset-folder>/
        Eiffel_Tower/
            img1.jpg
            img2.jpg
        Taj_Mahal/
            img1.jpg
            ...

Splits each class into train/val/test (ratios from configs/config.yaml)
and writes the result into data/processed/ as a standard PyTorch
ImageFolder-compatible layout:
    data/processed/train/<class>/*.jpg
    data/processed/val/<class>/*.jpg
    data/processed/test/<class>/*.jpg

Classes with fewer than `min_images_per_class` images are skipped
(configurable in configs/config.yaml) since they're too sparse to
train or evaluate on reliably.

USAGE:
    python scripts/prepare_data.py --source data/raw/landmarks-dataset
"""

import argparse
import os
import shutil
import sys
import yaml
from sklearn.model_selection import train_test_split

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_class_folders(source_dir):
    """
    Return {class_name: [image_paths]}.

    Only the immediate sub-folders of `source_dir` are treated as classes
    (e.g. "Gothic", "Mughal"). All images found anywhere underneath each
    one are pooled together, even if the dataset nests specific examples
    inside it (e.g. Gothic/Notre_Dame/*.jpg, Gothic/Cologne_Cathedral/*.jpg
    all count as class "Gothic"). This matches datasets organized by
    category/style rather than one flat folder per class.
    """
    classes = {}
    for entry in sorted(os.listdir(source_dir)):
        class_dir = os.path.join(source_dir, entry)
        if not os.path.isdir(class_dir):
            continue

        images = []
        for root, dirs, files in os.walk(class_dir):
            images.extend(
                os.path.join(root, f) for f in files if f.lower().endswith(IMG_EXTENSIONS)
            )

        if images:
            classes[entry] = images

    return classes


def main():
    parser = argparse.ArgumentParser(description="Split a raw landmark dataset into train/val/test")
    parser.add_argument("--source", type=str, required=True,
                         help="Path to the raw dataset folder (contains one sub-folder per class)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]
    processed_dir = ds_cfg["processed_dir"]
    min_images = ds_cfg["min_images_per_class"]
    train_r, val_r, test_r = ds_cfg["train_split"], ds_cfg["val_split"], ds_cfg["test_split"]

    if not os.path.isdir(args.source):
        print(f"ERROR: source folder not found: {args.source}")
        sys.exit(1)

    print(f"Scanning '{args.source}' for class folders...")
    classes = find_class_folders(args.source)
    print(f"Found {len(classes)} raw class folders.")

    kept, dropped = 0, 0
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(processed_dir, split), exist_ok=True)

    for class_name, images in sorted(classes.items()):
        if len(images) < min_images:
            dropped += 1
            continue
        kept += 1

        train_imgs, temp_imgs = train_test_split(images, train_size=train_r, random_state=42)
        rel_val = val_r / (val_r + test_r)
        val_imgs, test_imgs = train_test_split(temp_imgs, train_size=rel_val, random_state=42)

        for split, split_imgs in (("train", train_imgs), ("val", val_imgs), ("test", test_imgs)):
            dst_dir = os.path.join(processed_dir, split, class_name)
            os.makedirs(dst_dir, exist_ok=True)
            for src_path in split_imgs:
                shutil.copy2(src_path, os.path.join(dst_dir, os.path.basename(src_path)))

    print(f"\nDone.")
    print(f"  Classes kept:    {kept}")
    print(f"  Classes dropped: {dropped} (fewer than {min_images} images)")
    print(f"  Output written to: {processed_dir}/train, /val, /test")
    print(f"\nNext step:\n    python src/train.py\n")


if __name__ == "__main__":
    main()
