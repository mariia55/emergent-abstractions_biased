#!/bin/bash

#SBATCH -J DS_4_4
#SBATCH --time=8:00:00
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

# Navigate to the directory
cd "$HOME/emergent-abstractions/train.py"

# Run the Python script using srun with the absolute path to nest_local.py
srun /home/student/r/rverdugo/miniconda3/envs/eggfix/bin/python --batch_size 64 --n_epochs 300 --dimensions 4 4 4 4 --learning_rate 0.0005 --game_size 10 --hidden_size 128 --temp_update 0.99 --temperature 2 --save True --num_of_runs 5 --path "$HOME/emergent-abstractions/results/vague_ds_results

# Make sure the system has time to save all the models and stuff
srun sleep 10