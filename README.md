# Breast cancer screening

---

## Model

The model (`model.py`) is a two-stage pipeline:

### Stage 1 — Mass Detection (Mask R-CNN)
A `maskrcnn_resnet50_fpn_v2` with a pretrained ResNet-50-FPN backbone. The box and mask predictor heads are replaced with 2-class versions (background / mass). Given a full mammogram image it outputs:
- Bounding boxes around detected masses
- Instance segmentation masks per detected mass
- Confidence scores per detection

### Stage 2 — Pathology Classification (ConvNeXt)
For each detected mass, the bounding box is cropped from the image, padded to square, resized to 224×224, and passed through a pretrained `convnext_small` feature extractor followed by a linear head. It outputs a benign / malignant prediction per mass.

During training, GT boxes are used for cropping (not predicted boxes) to avoid training the classifier on noisy early detections.

Only the last two blocks of the ConvNeXt backbone are fine-tuned (`features[6]` downsampling + `features[7]` stage 4). The earlier layers are frozen to prevent overfitting on the small dataset size (~1000 training images).

The classification head is:
```
Linear(768 → 256) → ReLU → Dropout(0.5) → Linear(256 → 2)
```
The hidden layer gives the model capacity to learn a non-linear decision boundary between benign and malignant. Dropout regularises against overfitting on the small number of labelled patches.


ref:
https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html#torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights
https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html#torchvision.models.detection.maskrcnn_resnet50_fpn
https://docs.pytorch.org/vision/stable/models/convnext.html
https://medium.com/@deepvisionkararhaider/the-brain-behind-object-segmentation-a-complete-guide-to-mask-r-cnn-77f5016140d8

---

## Training losses

| Loss | Component | What it measures |
|---|---|---|
| `loss_objectness` | RPN (stage 1) | Does this anchor contain anything at all |
| `loss_rpn_box_reg` | RPN (stage 1) | Coarse box coordinate regression |
| `loss_classifier` | Detection head (stage 2) | Background vs mass classification per ROI |
| `loss_box_reg` | Detection head (stage 2) | Refined bounding box coordinate regression |
| `loss_mask` | Mask head (stage 2) | Per-pixel segmentation accuracy on positive ROIs |
| `loss_pathology` | ConvNeXt classifier | Benign vs malignant cross-entropy on GT-cropped patches |

All losses are summed and optimised jointly with AdamW.

---

## Validation split

The official CBIS-DDSM train CSV is split **at the patient level** (15% of patients held out) so no patient appears in both train and val. The official test set is **not used during training** and is reserved for the final evaluation script.

The following metrics are computed on the val split after each epoch:

| Metric | Description |
|---|---|
| `box_iou` | Mean IoU between each predicted box and its best-matching GT box |
| `mask_iou` | Mean pixel-level IoU between predicted and GT masks, same matching |
| `cls_acc` | Pathology accuracy — only counted for predictions with box IoU ≥ 0.5 |

Best model checkpoint is saved based on `box_iou`.

---

## Training

All runs are logged to wandb. Make sure you are logged in first:
```bash
wandb login
```

---

### Detector training (`train_detector.py`)

The detector uses a **ResNet-101 FPN** backbone with Mask R-CNN, trained at **1024×1024** resolution. Because CBIS-DDSM is small (~1000 images), we recommend pre-training on the Balloon dataset first to warm up the detection heads before fine-tuning on mammograms.

#### Step 1 — Pre-train on Balloon dataset

The Balloon dataset (~74 images) teaches the model general instance segmentation before it sees any mammograms.

```bash
python pretrain_balloon.py --epochs 30 --batch_size 2
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--data_root` | `./data/balloon` | Path to balloon images (`train/` and `test/` subfolders) |
| `--meta_root` | `./meta/balloon` | Path to `train.json` and `test.json` |
| `--output_dir` | `./models/pretrain_balloon` | Where checkpoints are saved |
| `--image_size` | `1024` | Resolution (match your fine-tuning resolution) |
| `--epochs` | `30` | |
| `--batch_size` | `2` | Use 1 if OOM on Apple Silicon |

The best checkpoint is saved to `./models/pretrain_balloon/balloon_best.pth`.

Data structure expected:
```
data/balloon/
    train/   ← jpg images
    test/    ← jpg images
meta/balloon/
    train.json
    test.json
```

#### Step 2 — Fine-tune on CBIS-DDSM

```bash
python train_detector.py --pretrain_weights ./models/pretrain_balloon/balloon_best.pth
```

Without pre-training (train from ImageNet weights only):
```bash
python train_detector.py
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--data_root` | `./data/cbis-ddsm` | Path to dataset root |
| `--train_csv` | `./meta/cbis-ddsm/mass_case_description_train_set.csv` | Train metadata CSV |
| `--test_csv` | `./meta/cbis-ddsm/mass_case_description_test_set.csv` | Test metadata CSV |
| `--pretrain_weights` | `None` | Path to balloon pre-trained checkpoint |
| `--epochs` | `20` | |
| `--lr` | `5e-4` | Learning rate |
| `--weight_decay` | `1e-4` | AdamW weight decay |
| `--batch_size` | `16` | Reduce to 1–2 on Apple Silicon |
| `--val_split` | `0.15` | Fraction of patients held out for validation |
| `--early_stopping_patience` | `3` | Epochs without val loss improvement before stopping |
| `--augment` | flag | Enable random flip/rotation augmentation |
| `--output_dir` | `./models` | Local folder for checkpoints |

#### Wandb sweep (fine-tuning)

The sweep varies `lr`, `weight_decay`, `batch_size`, and `trainable_backbone_layers`. Pre-trained weights are shared across all sweep runs — pass them via `--pretrain_weights` as normal:

```bash
python train_detector.py --pretrain_weights ./models/pretrain_balloon/balloon_best.pth --sweep
```

#### Output structure

```
models/
├── pretrain_balloon/
│   ├── balloon_epoch001.pth
│   └── balloon_best.pth          ← use this for --pretrain_weights
└── <wandb_run_id>/
    ├── hyperparams.json
    ├── detector_resnet101_epoch001.pth
    └── detector_resnet101_best.pth
```

---

### Full model training (`train.py`)

```bash
python train.py
```

With custom hyperparams:
```bash
python train.py --lr 1e-4 --batch_size 2 --epochs 20 --weight_decay 1e-3
```

Full list of arguments:

| Argument | Default | Description |
|---|---|---|
| `--data_root` | `./data/cbis-ddsm` | Path to dataset root |
| `--train_csv` | `./meta/cbis-ddsm/mass_case_description_train_set.csv` | Train metadata CSV |
| `--test_csv` | `./meta/cbis-ddsm/mass_case_description_test_set.csv` | Test metadata CSV |
| `--epochs` | `10` | Number of training epochs |
| `--lr` | `5e-4` | Learning rate |
| `--weight_decay` | `1e-4` | AdamW weight decay |
| `--batch_size` | `2` | Batch size |
| `--lr_step_size` | `3` | LR scheduler step size (epochs) |
| `--lr_gamma` | `0.5` | LR scheduler decay factor |
| `--num_workers` | `0` | DataLoader workers |
| `--output_dir` | `./models` | Local folder for checkpoints |

### Wandb sweep
```bash
python train.py --sweep
```

This launches a Bayesian sweep over `lr`, `weight_decay`, `batch_size`, `lr_step_size`, and `lr_gamma`. Alternatively, define your own sweep config on the wandb dashboard and run:
```bash
wandb agent <sweep_id>
```

### Output structure

Each run saves to its own subfolder named by the wandb run ID:
```
models/
└── <run_id>/
    ├── hyperparams.json        ← all hyperparams for this run
    ├── maskrcnn_epoch001.pth   ← checkpoint per epoch (includes metrics)
    ├── maskrcnn_epoch002.pth
    └── maskrcnn_best.pth       ← best checkpoint by val box_iou
```

Model files are **not** uploaded to wandb — only metrics and hyperparams are logged there.

---

## Docker (sweep)

Build:
```bash
docker build -t hadamlab-sweep .
```

Run a sweep — mounts a local `models/` folder and injects the wandb key from `.env`:
```bash
docker run --gpus all \
  --env-file .env \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/meta:/app/meta \
  hadamlab-sweep
```

`WANDB_API_KEY` in `.env` is picked up automatically by wandb — no manual login needed. **Never commit `.env` to git.**

---

## Web inference (Gradio)

A Gradio web app (`gradio_app.py`) lets you upload any mammogram image (PNG, JPG, DICOM) and get back an annotated image with bounding boxes and a per-lesion results table.

**Green box** = benign &nbsp;|&nbsp; **Red box** = malignant

### Quickstart with Docker Compose

Make sure your model weights are in the right place first:

```
models/
└── detector_resnet101_best.pth   ← detector checkpoint
classifier_model/
├── classifier_best.pth           ← classifier checkpoint
└── hyperparams.json              ← saved during classifier training
```

Then build and start:

```bash
docker compose -f docker-compose.inference.yml up --build
```

Open **http://localhost:7860** in a browser, upload a mammogram image, hit **Elemzés indítása**.

To stop:
```bash
docker compose -f docker-compose.inference.yml down
```

### CPU-only (no NVIDIA GPU)

Remove or comment out the `deploy` block in `docker-compose.inference.yml`, then:

```bash
docker compose -f docker-compose.inference.yml up --build
```

Or set the env var directly:

```bash
DEVICE=cpu docker compose -f docker-compose.inference.yml up --build
```

### Run without Docker

```bash
pip install gradio pydicom
python gradio_app.py
```

The app reads config from environment variables:

| Variable | Default | Description |
|---|---|---|
| `DETECTOR_CHECKPOINT` | `./models/detector_resnet101_best.pth` | Path to detector `.pth` |
| `CLASSIFIER_DIR` | `./classifier_model` | Directory with `classifier_best.pth` + `hyperparams.json` |
| `SCORE_THRESHOLD` | `0.05` | Detector confidence threshold |
| `CLS_THRESHOLD` | `0.5` | Malignant probability threshold |
| `DEVICE` | auto | `cpu`, `cuda`, or `mps` |
| `GRADIO_SERVER_PORT` | `7860` | Port to listen on |

Example with custom paths:

```bash
DETECTOR_CHECKPOINT=./models/my_run/detector_resnet101_best.pth \
CLASSIFIER_DIR=./models_cls/my_cls_run \
CLS_THRESHOLD=0.4 \
python gradio_app.py
```
