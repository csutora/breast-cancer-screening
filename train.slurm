#!/bin/bash
#SBATCH --job-name=hadamlab_train
#SBATCH --partition=gpu             # Target the GPU nodes
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           # Good baseline for PyTorch data loaders
#SBATCH --mem=64G
#SBATCH --time=48:00:00             # Max run time (adjust if needed)
#SBATCH --output=logs/train_%j.out  # Standard logs
#SBATCH --error=logs/train_%j.err   # Crash logs

# 1. Load Singularity
module load singularity

# Resolve everything from the original submit directory (not Slurm spool copy).
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR" || { echo "Failed to cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/models"

# 2. Inject the WandB API Key into the environment
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
else
	echo "Missing .env at $ENV_FILE"
	exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
	echo "WANDB_API_KEY missing in $ENV_FILE"
	exit 1
fi

if [[ ! "$WANDB_API_KEY" =~ ^[A-Za-z0-9_]+$ ]]; then
	echo "WANDB_API_KEY has invalid characters; use only letters, numbers, and underscores"
	exit 1
fi

if (( ${#WANDB_API_KEY} < 32 )); then
	echo "WANDB_API_KEY looks too short (${#WANDB_API_KEY} chars). Recopy from wandb account settings."
	exit 1
fi

# 3. Bind the full project into /app so code files (including train.py) are always available.
BINDS="$PROJECT_DIR:/app"
IMAGE="$PROJECT_DIR/hadamlab-sweep.sif"
if [[ ! -f "$IMAGE" ]]; then
	echo "Missing Singularity image at $IMAGE"
	exit 1
fi

export PYTHONUNBUFFERED=1

# 4. Launch the Sweep!
srun singularity exec --nv --bind "$BINDS" --pwd /app "$IMAGE" \
	python -u /app/train_detector.py --sweep \
	--output_dir /app/models \
	--data_root /app/data/cbis-ddsm \
	--train_csv /app/meta/cbis-ddsm/mass_case_description_train_set.csv \
	--test_csv /app/meta/cbis-ddsm/mass_case_description_test_set.csv