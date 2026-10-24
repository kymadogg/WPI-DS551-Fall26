#!/bin/bash
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

# conda setup
# curl -O https://repo.anaconda.com/archive/Anaconda3-2026.07-1-Linux-x86_64.sh
# bash ./Anaconda3-2026.07-1-Linux-x86_64.sh -b -p $HOME/anaconda3
# source ~/anaconda3/bin/activate

# miniconda setup
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source ~/miniconda3/bin/activate
conda init --all

# make environment
conda create -n myenv python=3.11
conda activate myenv

# install depencencies 
pip install opencv-python-headless "gymnasium[atari]==1.2.2" autorom[accept-rom-license]
pip install -U "ray[rllib]" ipywidgets
conda install -c conda-forge moviepy, conda install -c conda-forge ffmpeg, conda update ffmpeg
conda install -c conda-forge tqdm

# install pytorch (CUDA 13.2)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
python main.py --train_dqn