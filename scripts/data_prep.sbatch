#!/bin/bash
#SBATCH --job-name=data_prep
#SBATCH --output=slurm_logs/data_prep-%j.out
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --account=dsci410_510

cd /home/khousto4/movie-poster-genre
source venv/bin/activate
source .env
export HF_TOKEN

python data/data_prep.py --config configs/default.yaml