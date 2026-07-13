"""
Grad-CAM (Gradient-weighted Class Activation Mapping).

Kaggle landmark datasets only provide a class label per image — no
bounding boxes. Grad-CAM lets us get a bounding box anyway: it looks at
which pixels the trained classifier "looked at" most to make its
decision, producing a heatmap we can threshold into a box. This is a
well-established weakly-supervised localization technique, so the model
both classifies AND localizes the landmark without ever needing
box-annotated training data.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization", ICCV 2017.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer = dict([*self.model.named_modules()])[target_layer_name]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: (1, C, H, W) preprocessed image tensor, requires_grad not needed.
        Returns: (heatmap [H, W] in 0..1, predicted_class_idx, confidence)
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        confidence = probs[0, class_idx].item()

        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients[0]          # (C, H, W)
        activations = self.activations[0]      # (C, H, W)

        weights = gradients.mean(dim=(1, 2))   # (C,) global average pool of gradients
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cam.cpu().numpy()
        input_size = input_tensor.shape[-1]
        cam = cv2.resize(cam, (input_size, input_size))

        return cam, class_idx, confidence


def cam_to_bbox(cam, threshold=0.5):
    """
    Thresholds the CAM heatmap and returns the bounding box (x1, y1, x2, y2)
    around the largest connected high-activation region, in the same
    pixel coordinates as the CAM (i.e. the model's input resolution).
    Returns None if no region exceeds the threshold.
    """
    mask = (cam >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, x + w, y + h)


def overlay_heatmap(image_bgr, cam, alpha=0.4):
    """Blends the CAM heatmap over the original image for visualization."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    return cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)
