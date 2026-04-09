#!/bin/bash
#SBATCH --job-name=hadamlab_eval_det
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs_eval_detector/eval_detector_%j.out
#SBATCH --error=logs_eval_detector/eval_detector_%j.err

module load singularity

# Resolve project paths from submit directory (not Slurm spool copy).
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR" || { echo "Failed to cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$PROJECT_DIR/logs_eval_detector"

# Allow override with: sbatch --export=ALL,EVAL_IMAGE=/path/to/image.sif evaluate_detector.slurm
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
MODEL_DIR="${MODEL_DIR:-/app/models}"
BACKBONE="${BACKBONE:-resnet101}"
SPLIT="${SPLIT:-test}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.3}"
IOU_MATCH_THR="${IOU_MATCH_THR:-0.5}"
NUM_EXAMPLES="${NUM_EXAMPLES:-20}"
NUM_GOOD_EXAMPLES="${NUM_GOOD_EXAMPLES:-20}"
MAX_IMAGES="${MAX_IMAGES:-0}"
INCLUDE_EMPTY_IN_GOOD="${INCLUDE_EMPTY_IN_GOOD:-1}"

if [[ -n "${CHECKPOINT:-}" ]]; then
  CHECKPOINT_PATH="$CHECKPOINT"
else
  if [[ -f "$PROJECT_DIR/models/$MODEL_ID/detector_resnet101_best.pth" ]]; then
    CHECKPOINT_PATH="/app/models/$MODEL_ID/detector_resnet101_best.pth"
  elif [[ -f "$PROJECT_DIR/models/$MODEL_ID/detector_resnet152_best.pth" ]]; then
    CHECKPOINT_PATH="/app/models/$MODEL_ID/detector_resnet152_best.pth"
  else
    HOST_CANDIDATE="$(ls -1 "$PROJECT_DIR/models/$MODEL_ID"/detector_*_best.pth 2>/dev/null | head -n1)"
    if [[ -z "$HOST_CANDIDATE" ]]; then
      echo "Could not find detector checkpoint under $PROJECT_DIR/models/$MODEL_ID"
      echo "Set CHECKPOINT explicitly when submitting if needed."
      exit 1
    fi
    CHECKPOINT_PATH="/app/models/$MODEL_ID/$(basename "$HOST_CANDIDATE")"
  fi
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-/app/logs_eval_detector}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/examples_${MODEL_ID}_${SLURM_JOB_ID}}"
OUTPUT_JSON="${OUTPUT_JSON:-$OUTPUT_ROOT/eval_detector_metrics_${MODEL_ID}_${SLURM_JOB_ID}.json}"

INCLUDE_EMPTY_ARGS=()
case "$INCLUDE_EMPTY_IN_GOOD" in
  1|true|TRUE|yes|YES|y|Y)
    INCLUDE_EMPTY_ARGS=(--include_empty_images_in_good)
    ;;
esac

echo "[eval_detector] image: $IMAGE"
echo "[eval_detector] model_id: $MODEL_ID"
echo "[eval_detector] checkpoint: $CHECKPOINT_PATH"
echo "[eval_detector] model_dir: $MODEL_DIR"
echo "[eval_detector] split: $SPLIT"
echo "[eval_detector] backbone: $BACKBONE"
echo "[eval_detector] score_threshold: $SCORE_THRESHOLD"
echo "[eval_detector] iou_match_thr: $IOU_MATCH_THR"
echo "[eval_detector] num_examples: $NUM_EXAMPLES"
echo "[eval_detector] num_good_examples: $NUM_GOOD_EXAMPLES"
echo "[eval_detector] max_images: $MAX_IMAGES"
echo "[eval_detector] include_empty_images_in_good: $INCLUDE_EMPTY_IN_GOOD"
echo "[eval_detector] output_dir: $OUTPUT_DIR"
echo "[eval_detector] output_json: $OUTPUT_JSON"

srun singularity exec --nv --bind "$BINDS" --pwd /app "$IMAGE" \
  python /app/eval_detector.py \
  --model_dir "$MODEL_DIR" \
  --checkpoint "$CHECKPOINT_PATH" \
  --split "$SPLIT" \
  --data_root /app/data/cbis-ddsm \
  --train_csv /app/meta/cbis-ddsm/mass_case_description_train_set.csv \
  --test_csv /app/meta/cbis-ddsm/mass_case_description_test_set.csv \
  --num_workers "$NUM_WORKERS" \
  --backbone "$BACKBONE" \
  --score_threshold "$SCORE_THRESHOLD" \
  --iou_match_thr "$IOU_MATCH_THR" \
  --num_examples "$NUM_EXAMPLES" \
  --num_good_examples "$NUM_GOOD_EXAMPLES" \
  --max_images "$MAX_IMAGES" \
  --output_dir "$OUTPUT_DIR" \
  --output_json "$OUTPUT_JSON" \
  "${INCLUDE_EMPTY_ARGS[@]}"