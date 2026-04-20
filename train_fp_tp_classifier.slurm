#!/bin/bash
#SBATCH --job-name=hadamlab_fp_tp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs_fp_tp/train_fp_tp_%j.out
#SBATCH --error=logs_fp_tp/train_fp_tp_%j.err

set -euo pipefail

module load singularity

# Resolve project paths from submit directory (not Slurm spool copy).
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR" || { echo "Failed to cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$PROJECT_DIR/logs_fp_tp" "$PROJECT_DIR/models_fp_tp"

# Allow override with:
# sbatch --export=ALL,TRAIN_IMAGE=/path/to/image.sif train_fp_tp_classifier.slurm
if [[ -n "${TRAIN_IMAGE:-}" ]]; then
  IMAGE="$TRAIN_IMAGE"
elif [[ -f "$PROJECT_DIR/hadamlab-sweep.sif" ]]; then
  IMAGE="$PROJECT_DIR/hadamlab-sweep.sif"
else
  IMAGE="$(ls -1t "$PROJECT_DIR"/*.sif 2>/dev/null | head -n1)"
fi

if [[ -z "$IMAGE" || ! -f "$IMAGE" ]]; then
  echo "Could not find a .sif image. Set TRAIN_IMAGE when submitting."
  exit 1
fi

BINDS="$PROJECT_DIR:/app"
export PYTHONUNBUFFERED=1

MODEL_ID="${MODEL_ID:-nb7u0jeg}"
DETECTOR_BACKBONE="${DETECTOR_BACKBONE:-resnet101}"

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

# Core training hyperparameters
EPOCHS="${EPOCHS:-20}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_SPLIT="${VAL_SPLIT:-0.15}"
SEED="${SEED:-42}"

# Classifier model hyperparameters
BACKBONE="${BACKBONE:-resnet50}"
FREEZE_STAGES="${FREEZE_STAGES:-2}"
HEAD_HIDDEN="${HEAD_HIDDEN:-256}"
DROPOUT="${DROPOUT:-0.3}"
PATCH_SIZE="${PATCH_SIZE:-224}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"
CLASS_WEIGHT="${CLASS_WEIGHT:-auto}"

# FP/TP label mining hyperparameters
DET_SCORE_THRESHOLD="${DET_SCORE_THRESHOLD:-0.10}"
TP_IOU_THR="${TP_IOU_THR:-0.50}"
FP_IOU_MAX="${FP_IOU_MAX:-0.20}"
LABEL_ASSIGNMENT="${LABEL_ASSIGNMENT:-greedy}"
CONTEXT_FACTOR="${CONTEXT_FACTOR:-1.1}"
MAX_PATCHES_PER_IMAGE="${MAX_PATCHES_PER_IMAGE:-0}"
MAX_NEG_POS_RATIO="${MAX_NEG_POS_RATIO:-0}"
CLS_THRESHOLD="${CLS_THRESHOLD:-0.50}"

OUTPUT_DIR="${OUTPUT_DIR:-/app/models_fp_tp}"

# Toggle augment with AUGMENT=1
AUGMENT_ARGS=()
case "${AUGMENT:-0}" in
  1|true|TRUE|yes|YES|y|Y)
    AUGMENT_ARGS=(--augment)
    ;;
esac

# Bayesian sweep mode over the discrete hyperparameter grid in train_fp_tp_classifier.py
# Example:
# sbatch --export=ALL,SWEEP=1,SWEEP_COUNT=24,WANDB_PROJECT=hadamlab-fp-tp-classifier train_fp_tp_classifier.slurm
SWEEP_ARGS=()
case "${SWEEP:-0}" in
  1|true|TRUE|yes|YES|y|Y)
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
      ENV_FILE="$PROJECT_DIR/.env"
      if [[ -f "$ENV_FILE" ]]; then
        WANDB_API_KEY="$(grep -E '^WANDB_API_KEY[[:space:]]*=' "$ENV_FILE" | head -n1 | sed -E 's/^[^=]*=//; s/\r$//; s/^[[:space:]]+|[[:space:]]+$//g')"
        WANDB_API_KEY="${WANDB_API_KEY#\"}"
        WANDB_API_KEY="${WANDB_API_KEY%\"}"
        WANDB_API_KEY="${WANDB_API_KEY#\'}"
        WANDB_API_KEY="${WANDB_API_KEY%\'}"
        WANDB_API_KEY="${WANDB_API_KEY#=}"
        export WANDB_API_KEY
      fi
    fi

    if [[ -z "${WANDB_API_KEY:-}" ]]; then
      echo "[fp_tp][sweep] WANDB_API_KEY not found."
      echo "Set it via sbatch --export=ALL,WANDB_API_KEY=<key>,..."
      echo "or put WANDB_API_KEY=<key> in $PROJECT_DIR/.env"
      exit 1
    fi

    if [[ ! "$WANDB_API_KEY" =~ ^[A-Za-z0-9_]+$ ]]; then
      echo "[fp_tp][sweep] WANDB_API_KEY has invalid characters."
      echo "Use only letters, numbers, and underscores."
      exit 1
    fi

    if (( ${#WANDB_API_KEY} < 32 )); then
      echo "[fp_tp][sweep] WANDB_API_KEY looks too short (${#WANDB_API_KEY} chars)."
      exit 1
    fi

    SWEEP_ARGS+=(--sweep)
    if [[ -n "${SWEEP_COUNT:-}" ]]; then
      SWEEP_ARGS+=(--sweep_count "$SWEEP_COUNT")
    fi
    if [[ -n "${WANDB_PROJECT:-}" ]]; then
      SWEEP_ARGS+=(--wandb_project "$WANDB_PROJECT")
    fi
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
      SWEEP_ARGS+=(--wandb_entity "$WANDB_ENTITY")
    fi
    echo "[fp_tp][sweep] enabled (bayesian + discrete grid in train_fp_tp_classifier.py)"
    ;;
esac

echo "[fp_tp] image: $IMAGE"
echo "[fp_tp] detector_checkpoint: $DETECTOR_CKPT"
echo "[fp_tp] detector_backbone: $DETECTOR_BACKBONE"
echo "[fp_tp] model_id: $MODEL_ID"
echo "[fp_tp] output_dir: $OUTPUT_DIR"
echo "[fp_tp] epochs: $EPOCHS | lr: $LR | batch_size: $BATCH_SIZE"
echo "[fp_tp] label_assignment: $LABEL_ASSIGNMENT | det_score: $DET_SCORE_THRESHOLD"
echo "[fp_tp] tp_iou_thr: $TP_IOU_THR | fp_iou_max: $FP_IOU_MAX"
echo "[fp_tp] max_patches_per_image: $MAX_PATCHES_PER_IMAGE | max_neg_pos_ratio: $MAX_NEG_POS_RATIO"

srun singularity exec --nv --bind "$BINDS" --pwd /app "$IMAGE" \
  python /app/train_fp_tp_classifier.py \
  --detector_checkpoint "$DETECTOR_CKPT" \
  --detector_backbone "$DETECTOR_BACKBONE" \
  --data_root /app/data/cbis-ddsm \
  --train_csv /app/meta/cbis-ddsm/mass_case_description_train_set.csv \
  --test_csv /app/meta/cbis-ddsm/mass_case_description_test_set.csv \
  --val_split "$VAL_SPLIT" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --backbone "$BACKBONE" \
  --freeze_stages "$FREEZE_STAGES" \
  --head_hidden "$HEAD_HIDDEN" \
  --dropout "$DROPOUT" \
  --patch_size "$PATCH_SIZE" \
  --label_smoothing "$LABEL_SMOOTHING" \
  --class_weight "$CLASS_WEIGHT" \
  --det_score_threshold "$DET_SCORE_THRESHOLD" \
  --tp_iou_thr "$TP_IOU_THR" \
  --fp_iou_max "$FP_IOU_MAX" \
  --label_assignment "$LABEL_ASSIGNMENT" \
  --context_factor "$CONTEXT_FACTOR" \
  --max_patches_per_image "$MAX_PATCHES_PER_IMAGE" \
  --max_neg_pos_ratio "$MAX_NEG_POS_RATIO" \
  --cls_threshold "$CLS_THRESHOLD" \
  --seed "$SEED" \
  --output_dir "$OUTPUT_DIR" \
  "${AUGMENT_ARGS[@]}" \
  "${SWEEP_ARGS[@]}"
