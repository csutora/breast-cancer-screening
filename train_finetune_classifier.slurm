#!/bin/bash
#SBATCH --job-name=hadamlab_ft_cls
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs_finetune/train_finetune_%j.out
#SBATCH --error=logs_finetune/train_finetune_%j.err

set -euo pipefail

module load singularity

# Resolve project paths from submit directory (not Slurm spool copy).
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR" || { echo "Failed to cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$PROJECT_DIR/logs_finetune" "$PROJECT_DIR/models_cls_finetune"

# Allow override with:
# sbatch --export=ALL,TRAIN_IMAGE=/path/to/image.sif train_finetune_classifier.slurm
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
CLASSIFIER_DIR="${CLASSIFIER_DIR:-/app/classifier_model}"
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

EPOCHS="${EPOCHS:-5}"
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"
CLASS_WEIGHT="${CLASS_WEIGHT:-auto}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-6}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_SPLIT="${VAL_SPLIT:-0.15}"

SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.05}"
IOU_MATCH_THR="${IOU_MATCH_THR:-0.3}"
MAX_PATCHES_PER_IMAGE="${MAX_PATCHES_PER_IMAGE:-32}"
CURRICULUM_EPOCHS="${CURRICULUM_EPOCHS:-2}"
STRICT_SCORE_THRESHOLD="${STRICT_SCORE_THRESHOLD:-0.20}"
STRICT_IOU_MATCH_THR="${STRICT_IOU_MATCH_THR:-0.50}"
MIN_PATCHES_PER_BATCH="${MIN_PATCHES_PER_BATCH:-8}"
MAX_GT_FALLBACK_PATCHES_PER_BATCH="${MAX_GT_FALLBACK_PATCHES_PER_BATCH:-16}"
MAX_GT_FALLBACK_PATCHES_PER_IMAGE="${MAX_GT_FALLBACK_PATCHES_PER_IMAGE:-8}"

PATCH_SIZE="${PATCH_SIZE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-/app/models_cls_finetune}"

# Bayesian sweep mode over a discrete hyperparameter grid (configured in finetune_classifier.py).
# Example:
#   sbatch --export=ALL,SWEEP=1,SWEEP_COUNT=24,WANDB_PROJECT=hadamlab-classifier-finetune train_finetune_classifier.slurm
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
        # Handle common typo: WANDB_API_KEY==<key>
        WANDB_API_KEY="${WANDB_API_KEY#=}"
        export WANDB_API_KEY
      fi
    fi

    if [[ -z "${WANDB_API_KEY:-}" ]]; then
      echo "[finetune][sweep] WANDB_API_KEY not found."
      echo "Set it via sbatch --export=ALL,WANDB_API_KEY=<key>,..."
      echo "or put WANDB_API_KEY=<key> in $PROJECT_DIR/.env"
      exit 1
    fi

    if [[ ! "$WANDB_API_KEY" =~ ^[A-Za-z0-9_]+$ ]]; then
      echo "[finetune][sweep] WANDB_API_KEY has invalid characters."
      echo "Use only letters, numbers, and underscores."
      exit 1
    fi

    if (( ${#WANDB_API_KEY} < 32 )); then
      echo "[finetune][sweep] WANDB_API_KEY looks too short (${#WANDB_API_KEY} chars)."
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
    echo "[finetune][sweep] enabled (bayesian + discrete grid in finetune_classifier.py)"
    ;;
esac

AUGMENT_ARGS=()
case "${AUGMENT:-0}" in
  1|true|TRUE|yes|YES|y|Y)
    AUGMENT_ARGS=(--augment)
    ;;
esac

FALLBACK_ARGS=()
case "${DISABLE_GT_FALLBACK:-0}" in
  1|true|TRUE|yes|YES|y|Y)
    FALLBACK_ARGS+=(--disable_gt_fallback)
    ;;
esac
case "${GT_FALLBACK_IN_VAL:-0}" in
  1|true|TRUE|yes|YES|y|Y)
    FALLBACK_ARGS+=(--gt_fallback_in_val)
    ;;
esac

echo "[finetune] image: $IMAGE"
echo "[finetune] detector_checkpoint: $DETECTOR_CKPT"
echo "[finetune] classifier_dir: $CLASSIFIER_DIR"
echo "[finetune] detector_backbone: $DETECTOR_BACKBONE"
echo "[finetune] output_dir: $OUTPUT_DIR"
echo "[finetune] epochs: $EPOCHS | lr: $LR | batch_size: $BATCH_SIZE"
echo "[finetune] score_threshold: $SCORE_THRESHOLD | iou_match_thr: $IOU_MATCH_THR"
echo "[finetune] curriculum_epochs: $CURRICULUM_EPOCHS | strict_score: $STRICT_SCORE_THRESHOLD | strict_iou: $STRICT_IOU_MATCH_THR"
echo "[finetune] min_patches_per_batch: $MIN_PATCHES_PER_BATCH | gt_fallback_batch: $MAX_GT_FALLBACK_PATCHES_PER_BATCH"

srun singularity exec --nv --bind "$BINDS" --pwd /app "$IMAGE" \
  python /app/finetune_classifier.py \
  --detector_checkpoint "$DETECTOR_CKPT" \
  --classifier_dir "$CLASSIFIER_DIR" \
  --detector_backbone "$DETECTOR_BACKBONE" \
  --data_root /app/data/cbis-ddsm \
  --train_csv /app/meta/cbis-ddsm/mass_case_description_train_set.csv \
  --test_csv /app/meta/cbis-ddsm/mass_case_description_test_set.csv \
  --val_split "$VAL_SPLIT" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --label_smoothing "$LABEL_SMOOTHING" \
  --class_weight "$CLASS_WEIGHT" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --score_threshold "$SCORE_THRESHOLD" \
  --iou_match_thr "$IOU_MATCH_THR" \
  --max_patches_per_image "$MAX_PATCHES_PER_IMAGE" \
  --curriculum_epochs "$CURRICULUM_EPOCHS" \
  --strict_score_threshold "$STRICT_SCORE_THRESHOLD" \
  --strict_iou_match_thr "$STRICT_IOU_MATCH_THR" \
  --min_patches_per_batch "$MIN_PATCHES_PER_BATCH" \
  --max_gt_fallback_patches_per_batch "$MAX_GT_FALLBACK_PATCHES_PER_BATCH" \
  --max_gt_fallback_patches_per_image "$MAX_GT_FALLBACK_PATCHES_PER_IMAGE" \
  --patch_size "$PATCH_SIZE" \
  --output_dir "$OUTPUT_DIR" \
  "${SWEEP_ARGS[@]}" \
  "${AUGMENT_ARGS[@]}" \
  "${FALLBACK_ARGS[@]}"
