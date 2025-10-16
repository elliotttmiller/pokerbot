"""
Neural Network Builder for DeepStack Poker (Python)
Builds neural net architectures for bucketed poker ranges and counterfactual values.
"""
import torch
import torch.nn as nn

class NetBuilder(nn.Module):
    def __init__(self, num_buckets, hidden_sizes=[128, 128], activation='relu', use_gpu=False, input_size: int | None = None):
        super().__init__()
        # Default input layout: [{p1_range}, {p2_range}, pot_size]
        inferred_input = 2 * num_buckets + 1
        input_size = int(input_size) if input_size is not None else inferred_input
        output_size = 2 * num_buckets     # [{p1_cfvs}, {p2_cfvs}]
        layers = []
        last_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(last_size, h))
            if activation == 'prelu':
                layers.append(nn.PReLU())
            else:
                layers.append(nn.ReLU())
            last_size = h
        layers.append(nn.Linear(last_size, output_size))
        self.model = nn.Sequential(*layers)
        if use_gpu:
            self.model = self.model.cuda()

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def build_net(num_buckets, hidden_sizes=[128, 128], activation='relu', use_gpu=False, input_size: int | None = None):
        return NetBuilder(num_buckets, hidden_sizes, activation, use_gpu, input_size)
