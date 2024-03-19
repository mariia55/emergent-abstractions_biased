from utils.load_results import *
from utils.plot_helpers import *

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import random
from seaborn.algorithms import bootstrap

# 3: number of attributes
# 4: number of values
# example: attribute: shape, values: how many shapes are there?
datasets = [
    "(3,4)",
]
n_values = [
    4,
]
n_attributes = [
    3,
]
n_epochs = 400
n_datasets = len(datasets)
paths = [f"results/{d}_game_size_10_vsf_3/" for d in datasets]

context_unaware = False  # whether original or context_unaware simulations are evaluated
if context_unaware:
    setting = "context_unaware"
else:
    setting = "standard"  # context-aware

entropy_scores = load_entropies(paths, context_unaware=context_unaware)  #
entropies = [
    entropy_scores["NMI"],
    entropy_scores["effectiveness"],
    entropy_scores["consistency"],
]
print(entropies)

# from generic to specific
entropies_hierarchical = [
    entropy_scores["NMI_hierarchical"],
    entropy_scores["effectiveness_hierarchical"],
    entropy_scores["consistency_context_dep"],
]
print(entropies_hierarchical)
entropy_dict_context_dep = {}
for i, score in enumerate(list(entropy_scores.keys())[3:6]):
    results = entropies_hierarchical[i]
    mean = np.mean(results, axis=-1)
    # sd = results.std(axis=-1)
    for idx, d in enumerate(datasets):
        entropy_dict_context_dep[d + score] = mean[idx]  # (mean[idx], sd[idx])

# from coarse to fine context
entropies_context_dep = [
    entropy_scores["NMI_context_dep"],
    entropy_scores["effectiveness_context_dep"],
    entropy_scores["consistency_context_dep"],
]

entropy_dict_context_dep = {}
for i, score in enumerate(list(entropy_scores.keys())[6:9]):
    results = entropies_context_dep[i]
    mean = np.mean(results, axis=-1)
    # sd = results.std(axis=-1)
    for idx, d in enumerate(datasets):
        entropy_dict_context_dep[d + score] = mean[idx]  # (mean[idx], sd[idx])

entropies_concept_x_context = [
    entropy_scores["NMI_concept_x_context"],
    entropy_scores["effectiveness_concept_x_context"],
    entropy_scores["consistency_concept_x_context"],
]

entropy_dict_conc_x_cont = {}
for i, score in enumerate(list(entropy_scores.keys())[9:]):
    results = entropies_context_dep[i]
    mean = np.mean(results, axis=-1)
    # sd = results.std(axis=-1)
    for idx, d in enumerate(datasets):
        entropy_dict_conc_x_cont[d + score] = mean[idx]  # (mean[idx], sd[idx])

plot_heatmap_concept_x_context(
    entropies_concept_x_context,
    score="effectiveness",
    mode="mean",
    plot_dims=(1, 1),
    heatmap_size=(3, 3),
    figsize=(10.5, 7),
    titles=("D(3,4)"),
    fontsize=12,
)
