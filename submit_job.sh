#!/bin/bash
# submit_job.sh - Fashion CBIR on CeSViMa Magerit
# Usage: sbatch submit_job.sh
#
# conda setup (run once interactively):
#   export PATH=~/conda/envs/framework/bin:~/conda/bin:$PATH
#   source activate framework
#   pip install tensorflow opencv-python opencv-contrib-python \
#       scikit-image scikit-learn matplotlib tqdm scipy

#SBATCH --job-name=fashion_cbir
#SBATCH --output=cbir_%j.out
#SBATCH --error=cbir_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G

module purge
module load CUDA/12.1.1

export PATH=~/conda/envs/framework/bin:~/conda/bin:$PATH
source activate framework

export MPLBACKEND=Agg
export TF_CPP_MIN_LOG_LEVEL=2
#export TF_USE_LEGACY_KERAS=1

echo "=== Fashion CBIR — started $(date) ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Python: $(python --version)"
echo ""

# step 1: build index (one-time)
srun python -u ToyCBIR.py \
    --mode index \
    --image-dir ./images \
    --metadata ./archive/styles.csv \
    --index-dir ./index \
    --batch-size 64

# step 2: ablation — evaluate each retrieval strategy
for STRATEGY in cascaded resnet_only color_only; do
    echo ""
    echo "=== Evaluating strategy: $STRATEGY ==="
    srun python -u ToyCBIR.py \
        --mode evaluate \
        --strategy $STRATEGY \
        --metadata ./archive/styles.csv \
        --index-dir ./index \
        --results-dir ./results/$STRATEGY \
        --num-eval 200 --top-k 10
done

# step 3: demo queries with the cascaded strategy
srun python -u ToyCBIR.py \
    --mode demo \
    --strategy cascaded \
    --metadata ./archive/styles.csv \
    --index-dir ./index \
    --results-dir ./results/demo

echo ""
echo "=== Finished $(date) ==="
echo "Results:"
ls -lhR results/
