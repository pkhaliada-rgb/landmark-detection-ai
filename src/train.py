"""
Trains the landmark classifier.

USAGE:
    python src/train.py
    python src/train.py --config configs/config.yaml
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloaders
from src.model import build_model, save_checkpoint


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc="train" if train else "val ")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            pbar.set_postfix(loss=total_loss / total, acc=correct / total)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train the landmark classifier")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg, model_cfg, train_cfg = cfg["dataset"], cfg["model"], cfg["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        processed_dir=ds_cfg["processed_dir"],
        image_size=model_cfg["image_size"],
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
    )
    print(f"Classes ({len(class_names)}): {class_names[:10]}{' ...' if len(class_names) > 10 else ''}")

    model, _ = build_model(
        num_classes=len(class_names),
        architecture=model_cfg["architecture"],
        pretrained=model_cfg["pretrained"],
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    os.makedirs(train_cfg["checkpoint_dir"], exist_ok=True)
    best_path = os.path.join(train_cfg["checkpoint_dir"], "best.pt")
    last_path = os.path.join(train_cfg["checkpoint_dir"], "last.pt")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, train_cfg["epochs"] + 1):
        start = time.time()
        print(f"\nEpoch {epoch}/{train_cfg['epochs']}")

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        elapsed = time.time() - start
        print(f"  train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s")

        save_checkpoint(model, class_names, model_cfg["architecture"], last_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, class_names, model_cfg["architecture"], best_path)
            print(f"  -> new best model saved to {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["early_stopping_patience"]:
                print(f"  Early stopping (no improvement for {patience_counter} epochs).")
                break

    print("\nEvaluating best model on the held-out test set...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")
    print(f"\nTraining complete. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
