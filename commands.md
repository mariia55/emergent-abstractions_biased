## Grid search for #82

```
python -m egg.nest.nest_local --game train --sweep grid_search/parameters_by_DS.json --n_workers=1 --root_dir "grid_search/82" --name "emergent-abstractions"
```

## Grid search command (with egg)

```
python -m egg.nest.nest_local --game train --sweep grid_search/parameters.json --n_workers=1 --root_dir "grid_search/balh" --name "emergent-abstractions"
```

## To run model w/ my grid search params

```
python train.py --dimensions 4 4 4 --game_size 10 --vocab_size_factor 3 --n_epochs 60 --batch_size 32 --learning_rate 0.001 --hidden_size 256 --temp_update 0.99 --temperature 2 --save True
```
## 17-02 grid search
```
python train.py --dimensions 4 4 4 --game_size 10 --vocab_size_factor 3 --n_epochs 400 --batch_size 32 --learning_rate 0.0005 --hidden_size 256 --temp_update 0.99 --temperature 2 --num_of_runs 5 --save True
```
## 18-02 grid search
```
python train.py --dimensions 4 4 4 --game_size 10 --vocab_size_factor 3 --n_epochs 300 --batch_size 32 --learning_rate 0.001 --hidden_size 256 --temp_update 0.99 --temperature 2 --num_of_runs 5 --save True
```

## To run model w/ Kristina's grid search params

```
python train.py --dimensions 4 4 4 --game_size 10 --vocab_size_factor 3 --n_epochs 300 --batch_size 32 --learning_rate 0.001 --hidden_size 128 --temp_update 0.99 --temperature 1.5
```

## Cluster grid search w/ srun--pty/bin/bash
```
python -m egg.nest.nest_local --game train --sweep grid_search/parameters.json --n_workers=25 --root_dir "grid_search/29-03" --name "emergent-abstractions"
```

## Poetry path
```bash
/home/student/r/rverdugo/.local/bin/poetry
```

## pyenv
# Load pyenv automatically by appending
# the following to
# ~/.bash_profile if it exists, otherwise ~/.profile (for login shells)
# and ~/.bashrc (for interactive shells) :

export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Restart your shell for the changes to take effect.

# Load pyenv-virtualenv automatically by adding
# the following to ~/.bashrc:

eval "$(pyenv virtualenv-init -)"

[rverdugo@mgmt01 emergent-abstractions]$