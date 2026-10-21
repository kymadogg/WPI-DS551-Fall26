#!/bin/bash
#SBATCH --nodes 1        
#SBATCH --cpus-per-task 2 
#SBATCH --mem=8g      
#SBATCH --job-name "project 3"  
#SBATCH --partition short 
#SBATCH --time 0-12:00:00 

wget -qO- https://astral.sh/uv/install.sh | sh
cd WPI-DS551-Fall26
uv sync
python Project3/main.py --train_dqn
