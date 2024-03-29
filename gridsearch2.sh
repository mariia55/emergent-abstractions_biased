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

# Add the Conda installation in your home directory to the PATH
export PATH="/home/student/r/rverdugo/miniconda3/bin:$PATH"

# Activate the emergab environment
source "/home/student/r/rverdugo/miniconda3/etc/profile.d/conda.sh"
conda activate emergab

# Navigate to the emergent-abstractions directory
cd "$HOME/emergent-abstractions/"

# Run the Python script using srun
srun python -m egg.nest.nest_local --game train --sweep grid_search/parameters.json --n_workers=25 --root_dir "grid_search/29-03-2-sv" --name "emergent-abstractions"

# Make sure the system has time to save all the models and stuff
srun sleep 10