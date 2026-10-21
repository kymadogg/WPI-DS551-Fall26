#!/bin/bash
#!/usr/bin/env bash
#SBATCH -A gftabor
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -C A30
#SBATCH -t 12:00:00
#SBATCH --mem 12G
#SBATCH --job-name="P3"

# install uv
wget -qO- https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# move into directory
cd WPI-DS551-Fall26

# sync dependencies
uv sync

# train dqn
uv run Project3/main.py --train_dqn
