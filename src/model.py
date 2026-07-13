"""
Builds the landmark classifier backbone. The same backbone is reused by
Grad-CAM (src/gradcam.py) to produce the localization heatmap, so
classification and localization come from a single trained model.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes, architecture="resnet50", pretrained=True):
    architecture = architecture.lower()

    if architecture == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layer_name = "layer4"

    elif architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        target_layer_name = "features"

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model, target_layer_name


def save_checkpoint(model, class_names, architecture, path):
    torch.save({
        "model_state": model.state_dict(),
        "class_names": class_names,
        "architecture": architecture,
    }, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model, target_layer_name = build_model(
        num_classes=len(ckpt["class_names"]),
        architecture=ckpt["architecture"],
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt["class_names"], target_layer_name
