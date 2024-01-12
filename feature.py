"""
Feature processing backbones
"""

import torch
import torch.nn as nn
import math


def new_parameter(size):
    p = torch.zeros(size)
    stdv = 1.0 / math.sqrt(size)
    p.data.uniform_(-stdv, stdv)
    return nn.Parameter(p)


def reset_sequential(seq):
    for layer in seq:
        if isinstance(layer, nn.Linear):
            layer.reset_parameters()


class FeatureMLP(nn.Module):
    def __init__(self, input_size=16, output_size=16, n_layers=2):
        super().__init__()
        assert n_layers >= 2, "Need at least 2 layers"

        layers = [nn.Linear(input_size, output_size)]
        print("feature MLP", input_size, output_size)

        for _ in range(n_layers - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(output_size, output_size))

        self.trunk = nn.Sequential(*layers)
        self.n_layers = n_layers
        self.input_size = input_size
        self.output_size = output_size
        self.final_feat_dim = self.output_size

    def forward(self, x, threshold=0.09):
        output = self.trunk(x)
        # apply threshold
        activated_output = torch.where(output > threshold, 1.0, 0.0)
        return activated_output

    def reset_parameters(self):
        reset_sequential(self.trunk)
