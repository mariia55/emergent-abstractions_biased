#!/bin/bash

#SBATCH -J gs_DS_3_16
#SBATCH --time=48:00:00
#SBATCH --mem=400gb
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=gpu
#SBATCH --gpu-bind=single:1


# Source the .bashrc file
source $HOME/.bashrc

# Activate the emergab environment using conda
spack load miniconda3
conda activate eggfix

# Navigate to the emergent-abstractions directory
cd "$HOME/emergent-abstractions/"

# Run the Python script using srun with the absolute path to nest_local.py
srun /home/student/r/rverdugo/miniconda3/envs/eggfix/bin/python -m egg.nest.nest_local --game train --sweep grid_search/parameters_by_DS.json --n_workers=29 --root_dir "grid_search/DS_3_16" --name "emergent-abstractions"

# Make sure the system has time to save all the models and stuff
srun sleep 10