import torch
import torch.nn as nn
import utils

class MLP(nn.Module):
    def __init__(self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = [256, 256],
        activation: nn.Module = nn.ReLU(),
        output_activation: nn.Module = nn.Identity(),
    ):
        super().__init__()
        self.layers = [nn.Linear(input_dim, hidden_dims[0]), activation]
        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers.append(output_activation)
        self.net = nn.Sequential(*self.layers)

        self.apply(utils.orthogonal_init)
        
    def forward(self, x):
        return self.net(x)