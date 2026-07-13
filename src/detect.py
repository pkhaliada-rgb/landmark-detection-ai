"""
Runs the trained landmark detector on a single image: classifies the
landmark and draws a bounding box around it using Grad-CAM localization.

USAGE:
    python src/detect.py --image path/to/photo.jpg
    python src/detect.py --image path/to/photo.jpg --checkpoint outputs/checkpoints/best.pt
    python src/detect.py --image path/to/photo.jpg --show-heatmap
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import yaml
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_checkpoint
from src.gradcam import GradCAM, cam_to_bbox, overlay_heatmap
from src.dataset import IMAGENET_MEAN, IMAGENET_STD


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_image(image_bgr, image_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (image_size, image_size))
    tensor = transform(resized).unsqueeze(0)
    return tensor


def scale_bbox(bbox, from_size, to_w, to_h):
    x1, y1, x2, y2 = bbox
    sx, sy = to_w / from_size, to_h / from_size
    return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)


def draw_prediction(image_bgr, bbox, label, confidence):
    output = image_bgr.copy()
    color = (1, 171, 248)  # BGR — matches the Coincent gold/blue brand accent

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        text = f"{label} ({confidence * 100:.1f}%)"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(output, (x1, max(0, y1 - th - 12)), (x1 + tw + 10, y1), color, -1)
        cv2.putText(output, text, (x1 + 5, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2, cv2.LINE_AA)
    else:
        text = f"{label} ({confidence * 100:.1f}%) — no strong localization region found"
        cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    return output


def main():
    parser = argparse.ArgumentParser(description="Classify + localize a landmark in an image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to model checkpoint (defaults to outputs/checkpoints/best.pt)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--show-heatmap", action="store_true",
                         help="Also save a Grad-CAM heatmap overlay alongside the boxed result")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint_path = args.checkpoint or os.path.join(cfg["train"]["checkpoint_dir"], "best.pt")
    image_size = cfg["model"]["image_size"]
    threshold = cfg["gradcam"]["heatmap_threshold"]
    conf_threshold = cfg["inference"]["confidence_threshold"]
    output_dir = cfg["inference"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: checkpoint not found at '{checkpoint_path}'. Train the model first:\n"
              f"    python src/train.py")
        sys.exit(1)

    if not os.path.exists(args.image):
        print(f"ERROR: image not found at '{args.image}'")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, target_layer_name = load_checkpoint(checkpoint_path, device=device)

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(f"ERROR: could not read image '{args.image}' (unsupported format?)")
        sys.exit(1)
    orig_h, orig_w = image_bgr.shape[:2]

    input_tensor = preprocess_image(image_bgr, image_size).to(device)

    cam_extractor = GradCAM(model, target_layer_name)
    cam, class_idx, confidence = cam_extractor.generate(input_tensor)
    label = class_names[class_idx]

    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")

    if confidence < conf_threshold:
        print(f"(Below confidence threshold of {conf_threshold * 100:.0f}% — result may be unreliable.)")

    bbox_model_res = cam_to_bbox(cam, threshold=threshold)
    bbox_orig = None
    if bbox_model_res is not None:
        bbox_orig = scale_bbox(bbox_model_res, image_size, orig_w, orig_h)

    result_image = draw_prediction(image_bgr, bbox_orig, label, confidence)
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    out_path = os.path.join(output_dir, f"{base_name}_detected.jpg")
    cv2.imwrite(out_path, result_image)
    print(f"\nSaved annotated result to: {out_path}")

    if args.show_heatmap:
        heatmap_overlay = overlay_heatmap(image_bgr, cam)
        heatmap_path = os.path.join(output_dir, f"{base_name}_heatmap.jpg")
        cv2.imwrite(heatmap_path, heatmap_overlay)
        print(f"Saved Grad-CAM heatmap to: {heatmap_path}")


if __name__ == "__main__":
    main()
