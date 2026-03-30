#!/bin/bash
#SBATCH --job-name=hadamlab_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs_eval/eval_%j.out
#SBATCH --error=logs_eval/eval_%j.err

module load singularity

# Resolve project paths from submit directory (not Slurm spool copy).
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR" || { echo "Failed to cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$PROJECT_DIR/logs_eval"

# Allow override with: sbatch --export=ALL,EVAL_IMAGE=/path/to/image.sif evaluate.slurm
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
SPLIT="${SPLIT:-test}"
IOU_MATCH_THR="${IOU_MATCH_THR:-0.5}"
NUM_WORKERS="${NUM_WORKERS:-0}"

OUTPUT_JSON="/app/logs_eval/eval_metrics_${SLURM_JOB_ID}.json"
FROC_CSV="/app/logs_eval/froc_curve_${SLURM_JOB_ID}.csv"
FROC_PNG="/app/logs_eval/froc_curve_${SLURM_JOB_ID}.png"

CHECKPOINT_ARGS=()
if [[ -n "${CHECKPOINT:-}" ]]; then
  CHECKPOINT_ARGS=(--checkpoint "$CHECKPOINT")
fi

MODEL_DIR="${MODEL_DIR:-/app/models}"

echo "[eval] image: $IMAGE"
echo "[eval] split: $SPLIT"
echo "[eval] model_dir: $MODEL_DIR"
echo "[eval] output_json: $OUTPUT_JSON"
echo "[eval] froc_csv: $FROC_CSV"
echo "[eval] froc_png: $FROC_PNG"

srun singularity exec --nv --bind "$BINDS" --pwd /app "$IMAGE" \
  python /app/eval.py \
  --model_dir "$MODEL_DIR" \
  --split "$SPLIT" \
  --data_root /app/data/cbis-ddsm \
  --train_csv /app/meta/cbis-ddsm/mass_case_description_train_set.csv \
  --test_csv /app/meta/cbis-ddsm/mass_case_description_test_set.csv \
  --num_workers "$NUM_WORKERS" \
  --iou_match_thr "$IOU_MATCH_THR" \
  --output_json "$OUTPUT_JSON" \
  --froc_csv "$FROC_CSV" \
  --froc_png "$FROC_PNG" \
  "${CHECKPOINT_ARGS[@]}"
