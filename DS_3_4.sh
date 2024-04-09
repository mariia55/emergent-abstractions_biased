#!/bin/bash
#SBATCH -J DS_3_4
#SBATCH --time=10:00:00
#SBATCH --mem=400gb
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=gpu
#SBATCH --gpu-bind=single:1
source $HOME/.bashrc
spack load miniconda3
conda activate eggfix 

srun python $HOME/emergent-abstractions/train.py --batch_size 64 --n_epochs 300 --dimensions 4 4 4 --learning_rate 0.0005 --game_size 10 --hidden_size 128 --temp_update 0.99 --temperature 2 --save True --num_of_runs 5 --path "$HOME/emergent-abstractions/results/vague_ds_results

srun sleep 10
