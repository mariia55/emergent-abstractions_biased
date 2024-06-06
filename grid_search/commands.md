## Input this Grid search command (with egg) into the terminal to conduct gridsearch. 

```
python -m egg.nest.nest_local --game train --sweep grid_search/parameters.json --n_workers=1 --root_dir "grid_search/{folder name}" --name "emergent-abstractions"
```

## Poetry path to access poetry env
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