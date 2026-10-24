#!/usr/bin/env bash
#SBATCH -A gftabor
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -C H100
#SBATCH -t 12:00:00
#SBATCH --mem 12G
#SBATCH --job-name="P3"

source activate myenv
python main.py --train_dqn