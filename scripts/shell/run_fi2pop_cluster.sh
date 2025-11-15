#!/bin/bash
#SBATCH --job-name=fi2pop_gen
#SBATCH --output=slurm_fi2pop_gen.out

# Activate conda environment
source ~/.bashrc
conda activate crafter-sb3-2x

export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg

# Prefer conda libs; avoid system CUDA
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
unset CUDA_HOME

# Set headless mode
export DISPLAY=

echo "========================================="
echo "FI-2POP LEVEL GENERATOR"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "========================================="
echo ""

# Check dependencies first
echo "Checking dependencies..."
python3 check_dependencies.py
DEPS_OK=$?

if [ $DEPS_OK -ne 0 ]; then
    echo ""
    echo "ERROR: Missing dependencies!"
    echo "Installing required packages..."
    conda install -y numpy networkx tqdm
    echo ""
fi

# Verify pre-generated levels exist
echo ""
echo "Verifying pre-generated level libraries..."
for DIR in captured_levels constructive_levels random_levels; do
    if [ -d "$DIR" ]; then
        COUNT=$(ls -1 "$DIR"/*.txt 2>/dev/null | wc -l)
        echo "[OK] $DIR: $COUNT levels found"
    else
        echo "[FAIL] $DIR: directory not found!"
    fi
done
echo ""

# Run FI-2POP generator
echo "========================================="
echo "Running FI-2POP Generator..."
echo "========================================="
echo ""

python3 run_fi2pop.py

echo ""
echo "========================================="
echo "Checking generated outputs..."
echo "========================================="

if [ -d "generated_levels" ]; then
    GEN_COUNT=$(ls -1 generated_levels/*.txt 2>/dev/null | wc -l)
    echo "[OK] Generated $GEN_COUNT levels in generated_levels/"
    echo ""
    echo "Generated files:"
    ls -lh generated_levels/*.txt
else
    echo "[FAIL] No generated_levels directory found!"
fi

echo ""
echo "========================================="
echo "FI-2POP Generation Complete!"
echo "End time: $(date)"
echo "========================================="
