FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    opencv-python-headless \
    pandas \
    matplotlib \
    wandb \
    torchvision==0.21.0

# Copy source
COPY config.py dataset.py model.py preprocess.py train.py eval.py train_detector.py eval_detector.py balloon.py dataset.py pretrain_balloon.py .

# Models folder will be mounted at runtime — create it so it exists if not mounted
RUN mkdir -p /app/models

# WANDB_API_KEY is injected at runtime via --env-file .env
# wandb login is handled automatically when WANDB_API_KEY is set in the environment

CMD ["python", "train.py", "--sweep", "--output_dir", "/app/models"]
