#!/usr/bin/env bash
#SBATCH -A gftabor
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -C H100
#SBATCH -t 12:00:00
#SBATCH --mem 8G
#SBATCH --job-name="P3"

conda init --all
source activate myenv
python main.py --train_dqn