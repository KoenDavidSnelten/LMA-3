#!/bin/bash
#SBATCH --job-name=prompting
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --array=0-49
#SBATCH --output=logs/output/%x_%A_%a.out
#SBATCH --error=logs/error/%x_%A_%a.err

# One array task per model/prompt combination (5 models x 10 prompts = 50 tasks)

module purge
module load Python/3.10.4-GCCcore-11.3.0

export HF_HOME=/scratch/$USER/hf_cache

source $HOME/venvs/LMA/bin/activate

set -a
source .env
set +a

MODELS=(Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-9B mistralai/Ministral-8B-Instruct-2410 meta-llama/Llama-3.1-8B-Instruct google/gemma-4-E4B-it)
PROMPTS=(0-shot 4-shot 8-shot 16-shot 32-shot
         0-shot-expert 4-shot-expert 8-shot-expert 16-shot-expert 32-shot-expert)

MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / ${#PROMPTS[@]}))]}
PROMPT=${PROMPTS[$((SLURM_ARRAY_TASK_ID % ${#PROMPTS[@]}))]}

echo "=========================================="
echo "Running: ${MODEL##*/} | Prompt: $PROMPT"
echo "Model: $MODEL"
echo "=========================================="

python3 prompting.py \
    --model $MODEL \
    --prompt prompts/${PROMPT}.txt \
    --output_file predictions/${MODEL##*/}_${PROMPT}.csv \
    --batch_size 8 \
    --seed 42

deactivate
