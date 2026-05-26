#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=slurm_logs/train-%j.out
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --account=dsci410_510

cd /gpfs/home/khousto4/movie-poster-genre
source venv/bin/activate

if [ -f ".env" ]; then
    source .env
    export HF_TOKEN
fi

python train.py --config configs/default.yaml