#!/usr/bin/env bash
#SBATCH -A cs551
#SBATCH -p academic
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -C A30
#SBATCH -t 24:00:00
#SBATCH --mem 12G
#SBATCH --job-name="kh-cs551"

wget -qO- https://astral.sh/uv/install.sh | sh
cd WPI-DS551-Fall26
uv sync
python Project3/main.py --train_dqn
