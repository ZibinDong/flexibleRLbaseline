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
        use_ln: bool = False,
    ):
        super().__init__()
        self.layers = [nn.Linear(input_dim, hidden_dims[0])]
        if use_ln:
            self.layers += [nn.LayerNorm((hidden_dims[0],)), nn.Tanh()]
        else:
            self.layers += [activation]
        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers.append(output_activation)
        self.net = nn.Sequential(*self.layers)

        self.apply(utils.orthogonal_init)
        
    def forward(self, x):
        return self.net(x)
    
class Reshape(nn.Module):
    def __init__(self, shape: tuple):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)
    
class HighwayPerceptionLayer(nn.Module):
    def __init__(self, 
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 1,
        output_dim: int = 128,
    ):
        super().__init__()
        self.input_layer = nn.Linear(n_features, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, n_heads, d_model * 4, batch_first=True, dropout=0.0,
            ), num_layers
        )
        self.output_layer = nn.Linear(d_model, output_dim)
        self.fusion_emb = nn.Parameter(torch.randn(1,1,d_model), requires_grad=True)
        self.ego_emb = nn.Parameter(torch.randn(1,1,d_model), requires_grad=True)
        self.neighbor_emb = nn.Parameter(torch.randn(1,1,d_model), requires_grad=True)
    def forward(self, obs):
        h = self.input_layer(obs)
        h[:, :1] += self.ego_emb
        h[:, 1:] += self.neighbor_emb
        h = torch.cat([h, self.fusion_emb.expand(h.shape[0], -1, -1)], dim=1)
        h = self.output_layer(self.transformer(h)[:,-1])
        return h