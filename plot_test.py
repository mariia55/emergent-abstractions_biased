from utils.load_results import *
from utils.plot_helpers import *

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import random
from seaborn.algorithms import bootstrap


datasets = ['(3,4)', '(3,8)', '(3,16)', '(4,4)', '(4,8)', '(5,4)', 'shapes3d']
n_values = [4, 8, 16, 4, 8, 4, 4]
n_attributes = [3, 3, 3, 4, 4, 5, 3]
n_epochs = 300
n_datasets = len(datasets)
paths = ['results/' + d + '_game_size_4_vsf_3' for d in datasets]

# paths.append('results/shapes3d_game_size_4_vsf_3')
# n_values.append(20)
# n_attributes.append(64)
# n_datasets += 1

context_unaware = False

if context_unaware:
    setting = 'context_unaware'
else:
    setting = 'standard' # context-aware

all_accuracies = load_accuracies(paths, n_runs=5, n_epochs=300, val_steps=1, zero_shot=False, context_unaware=context_unaware)
plot_training_trajectory(all_accuracies['train_acc'], all_accuracies['val_acc'], ylim=(0.5, 1), steps=(1, 1), plot_indices=(1, 2, 3, 4, 5, 7, 8), 
                         titles=('D(3,4)', 'D(3,8)', 'D(3,16)', 'D(4,4)', 'D(4,8)', 'D(5,4)', 'Shapes3d'))

plt.show()

plot_training_trajectory(all_accuracies['train_acc'], all_accuracies['val_acc'], ylim=(0.5, 1), xlim=(0, 10), steps=(1, 1), train_only=True, plot_indices=(1, 2, 3, 4, 5, 7, 8),
                         titles=('D(3,4)', 'D(3,8)', 'D(3,16)', 'D(4,4)', 'D(4,8)', 'D(5,4)', 'Shapes3d'))

plt.show()

accuracies = [all_accuracies['train_acc'], all_accuracies['val_acc'], all_accuracies['test_acc']]

plot_heatmap(accuracies, 'mean', plot_dims=(1,3), ylims=(0.4, 1.0), figsize=(13, 3.5), titles=('standard \ntrain', 'standard \nval', 'standard \ntest'), suptitle='accuracies', 
             matrix_indices=((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 2)), fontsize=17)

plt.show()