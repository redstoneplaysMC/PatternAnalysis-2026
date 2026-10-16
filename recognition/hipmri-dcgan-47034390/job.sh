#!/bin/bash
#SBATCH --job-name=cuda-check
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00 # Time limit
#SBATCH --output=cuda_check_%j.out
#SBATCH --error=cuda_check_%j.err
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
module load Anaconda3

# Activate your conda environment
source $HOME/miniconda3/bin/activate
conda activate torch
echo "Running on:"
hostname
echo "GPU:"
nvidia-smi
echo "Python:"
which python
echo "=== Environment ==="
which python
python --version

echo "=== PyTorch ==="
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

# Run the script
python -u train2.py
python -u predict.py