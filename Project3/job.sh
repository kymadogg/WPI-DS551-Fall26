#!/usr/bin/env bash
#SBATCH -A gftabor
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -C A30
#SBATCH -t 12:00:00
#SBATCH --mem 8G
#SBATCH --job-name="P3"

conda init --all
conda activate myenv

# pip install opencv-python-headless "gymnasium[atari]==1.2.2" autorom[accept-rom-license]
# pip install -U "ray[rllib]" ipywidgets
# conda install -c conda-forge moviepy, conda install -c conda-forge ffmpeg, conda update ffmpeg
# conda install -c conda-forge tqdm

# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132

python main.py --train_dqn