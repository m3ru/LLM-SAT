#!/bin/bash
#SBATCH -J sat-dpo                    # Job name
#SBATCH -N 1                          # Number of nodes
#SBATCH --ntasks-per-node=1          # Tasks per node
#SBATCH --gres=gpu:A100:1            # Request 1 A100 GPU (adjust based on availability)
#SBATCH --mem=64G                     # Memory
#SBATCH -t 24:00:00                   # Time limit (24 hours)
#SBATCH -o logs/dpo_%j.out           # Standard output
#SBATCH -e logs/dpo_%j.err           # Standard error
#SBATCH --mail-type=BEGIN,END,FAIL    # Email notifications
#SBATCH --mail-user=<your_email>@gatech.edu

# Load modules
module load anaconda3
module load cuda/11.8  # Or cuda/12.1 depending on ICE-PACE setup

# Activate environment
source activate dpo-training

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY="<your_wandb_key>"  # Optional

# Run training
python train_dpo.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --train_data "data/dpo_formatted" \
    --output_dir "output/sat-solver-dpo-run1" \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --num_epochs 3 \
    --beta 0.1 \
    --max_length 2048 \
    --max_prompt_length 1024 \
    --use_wandb \
    --run_name "sat-solver-dpo-run1"

echo "Job completed at $(date)"
