#!/bin/bash
#SBATCH --job-name=hadamlab_eval_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs_eval_full/eval_full_%j.out
#SBATCH --error=logs_eval_full/eval_full_%j.err

set -euo pipefail

module load singularity

# Resolve project paths from submit directory (not Slurm spool copy).
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR" || { echo "Failed to cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$PROJECT_DIR/logs_eval_full"

# Allow override with:
# sbatch --export=ALL,EVAL_IMAGE=/path/to/image.sif eval_full_pipeline.slurm
if [[ -n "${EVAL_IMAGE:-}" ]]; then
  IMAGE="$EVAL_IMAGE"
elif [[ -f "$PROJECT_DIR/hadamlab-sweep.sif" ]]; then
  IMAGE="$PROJECT_DIR/hadamlab-sweep.sif"
else
  IMAGE="$(ls -1t "$PROJECT_DIR"/*.sif 2>/dev/null | head -n1)"
fi

if [[ -z "$IMAGE" || ! -f "$IMAGE" ]]; then
  echo "Could not find a .sif image. Set EVAL_IMAGE when submitting."
  exit 1
fi

BINDS="$PROJECT_DIR:/app"

MODEL_ID="${MODEL_ID:-nb7u0jeg}"
CLASSIFIER_DIR="${CLASSIFIER_DIR:-/app/models_cls_finetune/vhhjav2f}"
CLASSIFIER_CHECKPOINT="${CLASSIFIER_CHECKPOINT:-}"
FP_TP_CLASSIFIER_DIR="${FP_TP_CLASSIFIER_DIR:-/app/models_fp_tp/yiobv2cs}"
FP_TP_THRESHOLD="${FP_TP_THRESHOLD:-0.5}"
FP_TP_POST_NMS_IOU="${FP_TP_POST_NMS_IOU:-0.30}"
FP_TP_THRESHOLDS="${FP_TP_THRESHOLDS:-}"
FP_TP_POST_NMS_IOUS="${FP_TP_POST_NMS_IOUS:-}"
DETECTOR_BACKBONE="${DETECTOR_BACKBONE:-resnet101}"

if [[ -n "$CLASSIFIER_CHECKPOINT" ]]; then
  CLASSIFIER_DIR="$(dirname "$CLASSIFIER_CHECKPOINT")"
fi

HOST_CLASSIFIER_DIR="$CLASSIFIER_DIR"
if [[ "$CLASSIFIER_DIR" == /app/* ]]; then
  HOST_CLASSIFIER_DIR="$PROJECT_DIR/${CLASSIFIER_DIR#/app/}"
fi

if [[ ! -d "$CLASSIFIER_DIR" && ! -d "$HOST_CLASSIFIER_DIR" ]]; then
  echo "Classifier directory not found."
  echo "Container path: $CLASSIFIER_DIR"
  echo "Host path: $HOST_CLASSIFIER_DIR"
  echo "Set CLASSIFIER_DIR or CLASSIFIER_CHECKPOINT explicitly when submitting if needed."
  exit 1
fi

HOST_CLASSIFIER_CKPT="$HOST_CLASSIFIER_DIR/classifier_best.pth"
if [[ ! -f "$CLASSIFIER_DIR/classifier_best.pth" && ! -f "$HOST_CLASSIFIER_CKPT" ]]; then
  echo "classifier_best.pth not found under classifier directory."
  echo "Checked: $CLASSIFIER_DIR/classifier_best.pth"
  echo "Checked: $HOST_CLASSIFIER_CKPT"
  exit 1
fi

FP_TP_ARGS=()
if [[ -n "${FP_TP_CLASSIFIER_DIR:-}" ]]; then
  HOST_FP_TP_CLASSIFIER_DIR="$FP_TP_CLASSIFIER_DIR"
  if [[ "$FP_TP_CLASSIFIER_DIR" == /app/* ]]; then
    HOST_FP_TP_CLASSIFIER_DIR="$PROJECT_DIR/${FP_TP_CLASSIFIER_DIR#/app/}"
  fi

  if [[ ! -d "$FP_TP_CLASSIFIER_DIR" && ! -d "$HOST_FP_TP_CLASSIFIER_DIR" ]]; then
    echo "FP/TP classifier directory not found."
    echo "Container path: $FP_TP_CLASSIFIER_DIR"
    echo "Host path: $HOST_FP_TP_CLASSIFIER_DIR"
    echo "Set FP_TP_CLASSIFIER_DIR explicitly when submitting if needed."
    exit 1
  fi

  HOST_FP_TP_CKPT="$HOST_FP_TP_CLASSIFIER_DIR/fp_tp_classifier_best.pth"
  if [[ ! -f "$FP_TP_CLASSIFIER_DIR/fp_tp_classifier_best.pth" && ! -f "$HOST_FP_TP_CKPT" ]]; then
    echo "fp_tp_classifier_best.pth not found under FP/TP classifier directory."
    echo "Checked: $FP_TP_CLASSIFIER_DIR/fp_tp_classifier_best.pth"
    echo "Checked: $HOST_FP_TP_CKPT"
    exit 1
  fi

  FP_TP_ARGS+=(--fp_tp_classifier_dir "$FP_TP_CLASSIFIER_DIR")
  FP_TP_ARGS+=(--fp_tp_threshold "$FP_TP_THRESHOLD")
  FP_TP_ARGS+=(--fp_tp_post_nms_iou "$FP_TP_POST_NMS_IOU")
fi

if [[ -n "${DETECTOR_CHECKPOINT:-}" ]]; then
  DETECTOR_CKPT="$DETECTOR_CHECKPOINT"
else
  DETECTOR_CKPT="/app/models/${MODEL_ID}/detector_${DETECTOR_BACKBONE}_best.pth"
fi

HOST_DETECTOR_CKPT="$DETECTOR_CKPT"
if [[ "$DETECTOR_CKPT" == /app/* ]]; then
  HOST_DETECTOR_CKPT="$PROJECT_DIR/${DETECTOR_CKPT#/app/}"
fi

if [[ ! -f "$DETECTOR_CKPT" && ! -f "$HOST_DETECTOR_CKPT" ]]; then
  echo "Detector checkpoint not found."
  echo "Container path: $DETECTOR_CKPT"
  echo "Host path: $HOST_DETECTOR_CKPT"
  echo "Set DETECTOR_CHECKPOINT explicitly when submitting if needed."
  exit 1
fi

SPLIT="${SPLIT:-test}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.10}"
IOU_MATCH_THR="${IOU_MATCH_THR:-0.4}"
CLS_THRESHOLD="${CLS_THRESHOLD:-0.42}"
TARGET_SENS="${TARGET_SENS:-}"
HEAVY_METRICS_MODE="${HEAVY_METRICS_MODE:-all}"
SCORE_THRESHOLDS="${SCORE_THRESHOLDS:-}"
IOU_MATCH_THRS="${IOU_MATCH_THRS:-}"
CLS_THRESHOLDS="${CLS_THRESHOLDS:-}"

OUTPUT_JSON="${OUTPUT_JSON:-/app/logs_eval_full/eval_full_pipeline_metrics_${MODEL_ID}_${SLURM_JOB_ID}.json}"
FROC_CSV="${FROC_CSV:-/app/logs_eval_full/eval_full_pipeline_froc_${MODEL_ID}_${SLURM_JOB_ID}.csv}"
FROC_PNG="${FROC_PNG:-/app/logs_eval_full/eval_full_pipeline_froc_${MODEL_ID}_${SLURM_JOB_ID}.png}"

TARGET_SENS_ARGS=()
if [[ -n "$TARGET_SENS" ]]; then
  TARGET_SENS_ARGS=(--target_sens "$TARGET_SENS")
fi

GRID_ARGS=()
if [[ -n "$SCORE_THRESHOLDS" ]]; then
  GRID_ARGS+=(--score_thresholds "$SCORE_THRESHOLDS")
fi
if [[ -n "$IOU_MATCH_THRS" ]]; then
  GRID_ARGS+=(--iou_match_thrs "$IOU_MATCH_THRS")
fi
if [[ -n "$CLS_THRESHOLDS" ]]; then
  GRID_ARGS+=(--cls_thresholds "$CLS_THRESHOLDS")
fi
if [[ -n "$FP_TP_THRESHOLDS" ]]; then
  GRID_ARGS+=(--fp_tp_thresholds "$FP_TP_THRESHOLDS")
fi
if [[ -n "$FP_TP_POST_NMS_IOUS" ]]; then
  GRID_ARGS+=(--fp_tp_post_nms_ious "$FP_TP_POST_NMS_IOUS")
fi
GRID_ARGS+=(--heavy_metrics_mode "$HEAVY_METRICS_MODE")

echo "[eval_full] image: $IMAGE"
echo "[eval_full] detector_checkpoint: $DETECTOR_CKPT"
echo "[eval_full] classifier_dir: $CLASSIFIER_DIR"
if [[ -n "${FP_TP_CLASSIFIER_DIR:-}" ]]; then
  echo "[eval_full] fp_tp_classifier_dir: $FP_TP_CLASSIFIER_DIR"
  echo "[eval_full] fp_tp_threshold: $FP_TP_THRESHOLD | fp_tp_post_nms_iou: $FP_TP_POST_NMS_IOU"
  if [[ -n "$FP_TP_THRESHOLDS" || -n "$FP_TP_POST_NMS_IOUS" ]]; then
    echo "[eval_full] fp_tp threshold grid: tp_prob=[$FP_TP_THRESHOLDS] post_nms_iou=[$FP_TP_POST_NMS_IOUS]"
  fi
fi
echo "[eval_full] split: $SPLIT"
echo "[eval_full] score_threshold: $SCORE_THRESHOLD"
echo "[eval_full] iou_match_thr: $IOU_MATCH_THR"
echo "[eval_full] cls_threshold: $CLS_THRESHOLD"
echo "[eval_full] heavy_metrics_mode: $HEAVY_METRICS_MODE"
if [[ -n "$SCORE_THRESHOLDS" || -n "$IOU_MATCH_THRS" || -n "$CLS_THRESHOLDS" ]]; then
  echo "[eval_full] threshold grid: score=[$SCORE_THRESHOLDS] iou=[$IOU_MATCH_THRS] cls=[$CLS_THRESHOLDS]"
fi
echo "[eval_full] output_json: $OUTPUT_JSON"
echo "[eval_full] froc_csv: $FROC_CSV"
echo "[eval_full] froc_png: $FROC_PNG"

srun singularity exec --nv --bind "$BINDS" --pwd /app "$IMAGE" \
  python /app/eval_full_pipeline.py \
  --detector_checkpoint "$DETECTOR_CKPT" \
  --classifier_dir "$CLASSIFIER_DIR" \
  --detector_backbone "$DETECTOR_BACKBONE" \
  --split "$SPLIT" \
  --data_root /app/data/cbis-ddsm \
  --train_csv /app/meta/cbis-ddsm/mass_case_description_train_set.csv \
  --test_csv /app/meta/cbis-ddsm/mass_case_description_test_set.csv \
  --num_workers "$NUM_WORKERS" \
  --score_threshold "$SCORE_THRESHOLD" \
  --iou_match_thr "$IOU_MATCH_THR" \
  --cls_threshold "$CLS_THRESHOLD" \
  --output_json "$OUTPUT_JSON" \
  --froc_csv "$FROC_CSV" \
  --froc_png "$FROC_PNG" \
  "${FP_TP_ARGS[@]}" \
  "${GRID_ARGS[@]}" \
  "${TARGET_SENS_ARGS[@]}"
