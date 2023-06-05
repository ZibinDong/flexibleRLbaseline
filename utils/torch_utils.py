import torch
import numpy as np
import torch.nn as nn
from typing import List, Union

def to_torch(x, dtype=torch.float32, device="cpu"):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).type(dtype).to(device)
    elif isinstance(x, torch.Tensor):
        return x.type(dtype).to(device)
    elif isinstance(x, (int, float, bool)):
        return torch.tensor(x, dtype=dtype, device=device)
    else:
        raise ValueError("Unknown type {}".format(type(x)))
 
@torch.no_grad()   
def orthogonal_init(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        nn.init.zeros_(m.bias)
    
def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data*(1.0-tau) + s.data*tau)

def hard_update(target: nn.Module, source: nn.Module):
    soft_update(target, source, 1.0)
    
def freeze_params(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False

def unfreeze_params(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = True
        
def str2activation(activation: Union[nn.Module, str]):
    if isinstance(activation, nn.Module):
        return activation
    elif isinstance(activation, str):
        if activation == 'relu': return nn.ReLU()
        elif activation == 'elu': return nn.ELU()
        elif activation == 'leaky_relu': return nn.LeakyReLU()
        else: raise NotImplementedError("Unsupport activation {}".format(activation))
    else:
        raise TypeError("Unknown type {}".format(type(activation)))
        
class FreezeParameters():
    def __init__(self, modules: List[nn.Module]):
        self.modules = modules
    
    def __enter__(self):
        for module in self.modules:
            if isinstance(module, nn.Module):
                for p in module.parameters(): 
                    p.requires_grad = False
            elif isinstance(module, nn.Parameter):
                module.requires_grad = False
            else:
                raise TypeError("Unknown type {}".format(type(module)))
    
    def __exit__(self, type, value, traceback):
        for module in self.modules:
            if isinstance(module, nn.Module):
                for p in module.parameters(): 
                    p.requires_grad = True
            elif isinstance(module, nn.Parameter):
                module.requires_grad = True
            else:
                raise TypeError("Unknown type {}".format(type(module)))
                
class EvalModules():
    def __init__(self, modules: List[nn.Module]):
        self.modules = modules
    
    def __enter__(self):
        for module in self.modules:
            module.eval()
    
    def __exit__(self, type, value, traceback):
        for module in self.modules:
            module.train()