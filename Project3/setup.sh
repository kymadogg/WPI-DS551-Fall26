#!/usr/bin/env bash

# conda setup
# curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
# source ~/miniconda3/bin/activate
conda init --all

# make environment
conda create -n myenv python=3.11
conda activate myenv

pip install opencv-python-headless "gymnasium[atari]==1.2.2" autorom[accept-rom-license]
pip install -U "ray[rllib]" ipywidgets
conda install -c conda-forge moviepy, conda install -c conda-forge ffmpeg, conda update ffmpeg
conda install -c conda-forge tqdm

# install pytorch (CUDA 13.2)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
