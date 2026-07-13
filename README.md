# Landmark Detection (Classification + Localization)

Recognizes famous physical landmarks (buildings, monuments) in a photo
**and** draws a bounding box around where the landmark is in the image —
trained on a Kaggle landmark classification dataset.

## How it works

Kaggle's landmark datasets label each *whole image* with a landmark name —
they don't include bounding box annotations. So this project uses the
standard practical approach for that situation:

1. **Classification** — a ResNet50 (or EfficientNet-B0) backbone is
   fine-tuned on your chosen Kaggle dataset to recognize each landmark class.
2. **Localization** — [Grad-CAM](https://arxiv.org/abs/1610.02391) inspects
   which pixels the trained classifier relied on most, producing a heatmap
   that's thresholded into a bounding box. No box-labeled data required.

One trained model → both the label and the box.

```
Photo  →  ResNet50 classifier  →  "Eiffel Tower" (96.4%)
                │
                └→ Grad-CAM heatmap → threshold → bounding box
```

## Project structure

```
landmark_detection/
├── configs/
│   ├── config.yaml          # all settings live here
│   └── classes.json         # auto-generated: index -> class name
├── data/
│   ├── raw/                 # Kaggle download lands here
│   └── processed/           # train/val/test split (ImageFolder format)
├── scripts/
│   ├── download_dataset.py  # pulls the dataset from Kaggle
│   └── prepare_data.py      # splits it into train/val/test
├── src/
│   ├── dataset.py           # DataLoader construction
│   ├── model.py             # backbone + checkpoint save/load
│   ├── train.py             # training loop
│   ├── gradcam.py           # Grad-CAM + bounding-box extraction
│   └── detect.py            # inference on a single image
├── outputs/
│   ├── checkpoints/         # best.pt / last.pt land here after training
│   └── predictions/         # annotated output images land here
├── app.py                   # convenience entry point (calls src/detect.py)
├── requirements.txt
└── README.md
```

## Setup (VS Code)

1. **Open the folder in VS Code** (`File -> Open Folder...`).

2. **Create a virtual environment** (open a terminal in VS Code, `` Ctrl+` ``):

   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   ```

   When VS Code prompts "Select interpreter", choose the `.venv` one.

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   If you have an NVIDIA GPU and want CUDA acceleration, install the
   matching PyTorch build first from https://pytorch.org/get-started/locally/
   before running the command above.

4. **Set up Kaggle API credentials** (one-time):
   - Create a free account at https://www.kaggle.com
   - Go to **Account settings → API → Create New Token** — this downloads `kaggle.json`
   - Place it at:
     - Windows: `C:\Users\<you>\.kaggle\kaggle.json`
     - Mac/Linux: `~/.kaggle/kaggle.json`

## Usage

### 1. Download the dataset

```bash
python scripts/download_dataset.py
```

This uses the dataset slug set in `configs/config.yaml`
(`dataset.kaggle_slug`, defaults to a general world-landmarks
classification set). To use a different Kaggle landmark dataset instead:

```bash
python scripts/download_dataset.py --slug "some-user/some-dataset"
```

**Any** Kaggle dataset organized as one folder per landmark class works —
just point `--slug` (or `configs/config.yaml`) at it.

### 2. Prepare train/val/test splits

Inspect what landed in `data/raw/` first, then point this at the folder
that actually contains the per-class sub-folders:

```bash
python scripts/prepare_data.py --source data/raw/<inner-folder-name>
```

This writes `data/processed/train`, `/val`, `/test` and drops any class
with fewer than `min_images_per_class` images (set in `config.yaml`,
default 20 — these are usually too sparse to learn reliably).

### 3. Train the model

```bash
python src/train.py
```

- Trains with early stopping and saves the best checkpoint to
  `outputs/checkpoints/best.pt`.
- Progress (loss/accuracy per epoch) prints to the terminal.
- Adjust epochs, batch size, learning rate, or architecture in
  `configs/config.yaml` — no code changes needed.
- **No GPU?** It'll still run on CPU, just slower — consider lowering
  `epochs` or using a smaller image size while testing.

### 4. Run detection on a photo

```bash
python app.py --image path/to/your_photo.jpg
```

Optional: also save the raw Grad-CAM heatmap (useful for debugging why a
box looks off):

```bash
python app.py --image path/to/your_photo.jpg --show-heatmap
```

Output lands in `outputs/predictions/<name>_detected.jpg`, with the
predicted landmark name, confidence score, and bounding box drawn on it.

## Tuning tips

- **Box too loose/tight?** Adjust `gradcam.heatmap_threshold` in
  `config.yaml` (higher = tighter box, lower = looser box).
- **Low accuracy?** Try `model.architecture: efficientnet_b0`, increase
  `train.epochs`, or check that `min_images_per_class` isn't dropping too
  many useful classes.
- **Confusable landmarks** (e.g. similar-looking towers) usually improve
  with more training images per class rather than more epochs.

## Notes

- This is a **weakly-supervised** localization approach — the bounding box
  is a strong visual estimate, not pixel-perfect ground truth. For
  research requiring precise IoU-evaluated boxes, you'd need a
  box-annotated dataset (e.g. Google Landmark Boxes) and an actual
  detector (Faster R-CNN / YOLO) instead of Grad-CAM.
- All file paths and hyperparameters are centralized in
  `configs/config.yaml` — you shouldn't need to touch the Python files to
  change settings.
