#!/bin/bash
#SBATCH -J standard-grid-search
#SBATCH --time=35:00:00
#SBATCH --mem=400gb
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=gpu
#SBATCH --gpu-bind=single:1
source $HOME/.bashrc

spack load miniconda3@4.10.3
conda activate emergab 

cd $HOME/emergent-abstractions/

srun python -m egg.nest.nest_local --game train --sweep grid_search/parameters.json --n_workers=25 --root_dir "grid_search/29-03-2" --name "emergent-abstractions"
#just make sure system got time to save all the models and stuff...
srun sleep 10
