"""
Gradio web interface for the breast cancer screening pipeline.

Upload any mammogram image (PNG, JPG, DICOM) and the model will:
  1. Preprocess it (MMS pseudo-colour, 1024×1024 letterbox)
  2. Detect lesions with Mask R-CNN
  3. Classify each detected lesion (benign / malignant)
  4. Return an annotated image + per-lesion results table

Environment variables (required):
  DETECTOR_CHECKPOINT   path to detector .pth file
  CLASSIFIER_DIR        directory containing classifier_best.pth + hyperparams.json

Optional:
  SCORE_THRESHOLD       detector confidence threshold (default 0.05)
  CLS_THRESHOLD         malignant probability threshold (default 0.5)
  DEVICE                cpu / cuda / mps (auto-detected if unset)
  GRADIO_SERVER_PORT    port to listen on (default 7860)
"""

from __future__ import annotations

import os
import io
import traceback

import cv2
import numpy as np
import torch
import gradio as gr

from config import PreprocessConfig, Representation
from preprocess import MammogramPreprocessor
from full_model import load_full_model

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

DETECTOR_CHECKPOINT = os.environ.get("DETECTOR_CHECKPOINT", "./models/detector_resnet101_best.pth")
CLASSIFIER_DIR      = os.environ.get("CLASSIFIER_DIR",      "./classifier_model")
SCORE_THRESHOLD     = float(os.environ.get("SCORE_THRESHOLD", "0.05"))
CLS_THRESHOLD       = float(os.environ.get("CLS_THRESHOLD",   "0.5"))

if os.environ.get("DEVICE"):
    DEVICE = torch.device(os.environ["DEVICE"])
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"[app] device={DEVICE}  score_thr={SCORE_THRESHOLD}  cls_thr={CLS_THRESHOLD}")

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------

_model = None

def _get_model():
    global _model
    if _model is None:
        print(f"[app] loading model  detector={DETECTOR_CHECKPOINT}  clf={CLASSIFIER_DIR}")
        _model = load_full_model(
            detector_checkpoint=DETECTOR_CHECKPOINT,
            classifier_dir=CLASSIFIER_DIR,
            inference_score_threshold=SCORE_THRESHOLD,
            device=DEVICE,
        )
        _model.eval()
        print("[app] model loaded")
    return _model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_preprocessor = MammogramPreprocessor(PreprocessConfig())


def _load_image(filepath: str) -> np.ndarray:
    """Load any image file to uint8 grayscale numpy array."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".dcm":
        try:
            import pydicom
            ds = pydicom.dcmread(filepath)
            arr = ds.pixel_array.astype(np.float32)
            arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            return arr
        except ImportError:
            raise RuntimeError("pydicom is required for DICOM files: pip install pydicom")

    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {filepath}")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return img


def _preprocess(gray: np.ndarray) -> torch.Tensor:
    """Preprocess grayscale → 3-channel float tensor [0,1] (1, 3, H, W)."""
    processed, _ = _preprocessor(gray, view="CC")          # (H, W, 3) float32 [0,1]
    tensor = torch.from_numpy(processed).permute(2, 0, 1)  # (3, H, W)
    return tensor


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

_BENIGN_COLOR    = (0,  200,  0)    # green  (BGR)
_MALIGNANT_COLOR = (0,   0, 220)    # red    (BGR)
_UNKNOWN_COLOR   = (200, 200, 0)    # yellow (BGR)


def _draw_results(
    processed_tensor: torch.Tensor,
    detection: dict,
    cls_threshold: float,
) -> np.ndarray:
    """
    Draw bounding boxes + labels on the preprocessed image.

    Returns an RGB uint8 numpy array.
    """
    img_np = processed_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3) float [0,1]
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    boxes  = detection.get("boxes",            torch.empty((0, 4)))
    scores = detection.get("scores",           torch.empty((0,)))
    probs  = detection.get("pathology_probs",  torch.empty((0, 2)))
    valid  = detection.get("pathology_valid",  torch.ones(len(boxes), dtype=torch.bool))

    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(v) for v in boxes[i].tolist()]
        det_score = float(scores[i]) if i < len(scores) else 0.0

        if i < len(probs) and valid[i]:
            mal_prob = float(probs[i, 1])
            is_mal   = mal_prob >= cls_threshold
            color    = _MALIGNANT_COLOR if is_mal else _BENIGN_COLOR
            label    = f"{'MAL' if is_mal else 'BEN'} {mal_prob:.2f} | det {det_score:.2f}"
        else:
            color = _UNKNOWN_COLOR
            label = f"det {det_score:.2f}"

        thickness = 3
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)

        font_scale = max(0.5, min(img_bgr.shape[0], img_bgr.shape[1]) / 1024)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        bg_y1 = max(0, y1 - th - 6)
        cv2.rectangle(img_bgr, (x1, bg_y1), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img_bgr, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Inference entry point
# ---------------------------------------------------------------------------

def run_inference(image_path: str | None) -> tuple:
    """
    Gradio inference function.

    Returns: (annotated_image, summary_markdown, detections_table)
    """
    if image_path is None:
        return None, "Tölts fel egy képet.", []

    try:
        model = _get_model()

        gray = _load_image(image_path)
        tensor = _preprocess(gray).to(DEVICE)

        with torch.no_grad():
            outputs = model([tensor])

        det = outputs[0]

        boxes  = det.get("boxes",           torch.empty((0, 4))).cpu()
        scores = det.get("scores",          torch.empty((0,))).cpu()
        probs  = det.get("pathology_probs", torch.empty((0, 2))).cpu()
        valid  = det.get("pathology_valid", torch.ones(len(boxes), dtype=torch.bool)).cpu()

        n_det = len(boxes)

        # Build results table  [#, det_score, class, mal_prob, x1, y1, x2, y2]
        rows = []
        n_mal = 0
        n_ben = 0
        for i in range(n_det):
            x1, y1, x2, y2 = [int(v) for v in boxes[i].tolist()]
            det_score = float(scores[i]) if i < len(scores) else 0.0

            if i < len(probs) and valid[i]:
                mal_prob = float(probs[i, 1])
                cls = "Malignus" if mal_prob >= CLS_THRESHOLD else "Benignus"
                if mal_prob >= CLS_THRESHOLD:
                    n_mal += 1
                else:
                    n_ben += 1
            else:
                mal_prob = float("nan")
                cls = "N/A"

            rows.append([
                i + 1,
                f"{det_score:.3f}",
                cls,
                f"{mal_prob:.3f}" if not (mal_prob != mal_prob) else "N/A",
                f"({x1}, {y1}, {x2}, {y2})",
            ])

        # Summary text
        summary_lines = [
            f"**Detektált elváltozások:** {n_det}",
            f"- Malignus: **{n_mal}**",
            f"- Benignus: **{n_ben}**",
            "",
            f"Detektor küszöb: `{SCORE_THRESHOLD}` | Osztályozó küszöb: `{CLS_THRESHOLD}`",
            f"Eszköz: `{DEVICE}`",
        ]
        summary = "\n".join(summary_lines)

        annotated = _draw_results(tensor.cpu(), {k: v.cpu() for k, v in det.items()
                                                  if isinstance(v, torch.Tensor)},
                                  CLS_THRESHOLD)
        return annotated, summary, rows

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return None, f"**Hiba:** {e}\n\n```\n{tb}\n```", []


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Emlőrák szűrő – mammogram elemző") as demo:
    gr.Markdown(
        """
        # Emlőrák szűrő – mammogram elemző
        Tölts fel egy mammogram képet (PNG, JPG, DICOM). A pipeline detektálja a tömör
        elváltozásokat és benignus/malignus osztályozást végez.

        **Zöld doboz** = benignus &nbsp;|&nbsp; **Piros doboz** = malignus
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="filepath",
                label="Mammogram kép",
                sources=["upload"],
            )
            run_btn = gr.Button("Elemzés indítása", variant="primary")

        with gr.Column(scale=2):
            image_output = gr.Image(label="Annotált kép", type="numpy")

    summary_output = gr.Markdown(label="Összefoglaló")

    table_output = gr.Dataframe(
        headers=["#", "Det. score", "Osztály", "Malignus prob.", "Box (x1,y1,x2,y2)"],
        label="Detektált elváltozások",
        interactive=False,
    )

    run_btn.click(
        fn=run_inference,
        inputs=[image_input],
        outputs=[image_output, summary_output, table_output],
    )

    image_input.upload(
        fn=run_inference,
        inputs=[image_input],
        outputs=[image_output, summary_output, table_output],
    )

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
