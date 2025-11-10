#!/bin/bash
#SBATCH --job-name=qrdqn_run1
#SBATCH --output=slurm_qrdqn_run1.out

# Activate conda environment
source ~/.bashrc
conda activate crafter-sb3-2x

export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg

# Prefer conda libs; avoid system CUDA
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
unset CUDA_HOME

# If you used pip cu121 wheels for torch, add:
SITE_PKGS="$(python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="$SITE_PKGS/torch/lib:$SITE_PKGS/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"

# Set headless mode
export DISPLAY=

RUN_NUM=1

echo "========================================="
echo "QR-DQN RUN $RUN_NUM"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "========================================="
nvidia-smi || echo "No GPU found"
echo ""

# Training
echo "Training QR-DQN run $RUN_NUM..."
python3 training/train.py \
    --algorithm qrdqn \
    --config configs/qrdqn_baseline_config.yaml \
    --n_envs 1 \
    --total_timesteps 1000000 \
    --model_dir models/cluster/qrdqn/run_$RUN_NUM \
    --log_dir logs/cluster/qrdqn/run_$RUN_NUM

# Evaluation
echo ""
echo "Evaluating QR-DQN run $RUN_NUM..."
python3 training/evaluate.py \
    --algorithm qrdqn \
    --model_path models/cluster/qrdqn/run_$RUN_NUM/qrdqn_crafter_final.zip \
    --config configs/qrdqn_baseline_config.yaml \
    --n_episodes 50 \
    --log_dir logs/cluster/qrdqn/run_$RUN_NUM/eval

echo ""
echo "========================================="
echo "QR-DQN run $RUN_NUM complete!"
echo "End time: $(date)"
echo "========================================="
