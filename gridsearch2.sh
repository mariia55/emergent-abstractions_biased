#!/bin/bash

# Set the PATH for the Slurm job environment
export PATH="/home/student/r/rverdugo/miniconda3/bin:$PATH"

# Set the CONDA_PREFIX for the Slurm job environment
export CONDA_PREFIX="/home/student/r/rverdugo/miniconda3/envs/emergab"

#SBATCH -J standard-grid-search
#SBATCH --time=35:00:00
#SBATCH --mem=400gb
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=gpu
#SBATCH --gpu-bind=single:1

# Source the .bashrc file
source $HOME/.bashrc

# Activate the emergab environment using conda
conda activate emergab

# Set the PYTHONPATH to include the egg module directory
export PYTHONPATH="/home/student/r/rverdugo/miniconda3/envs/emergab/lib/python3.9/site-packages:$PYTHONPATH"

# Navigate to the emergent-abstractions directory
cd "$HOME/emergent-abstractions/"

# Run the Python script using srun with the absolute path to nest_local.py
srun python /home/student/r/rverdugo/miniconda3/envs/emergab/lib/python3.9/site-packages/egg/nest/nest_local.py --game train --sweep grid_search/parameters.json --n_workers=25 --root_dir "grid_search/29-03-2-sv" --name "emergent-abstractions"

# Make sure the system has time to save all the models and stuff
srun sleep 10