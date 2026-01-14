


import torch
import torch.nn as nn

from config.model import LLMconfig
from models.layers.mlp import MLP

class Expert(nn.Module):
    """ A single feed-forward network expert. """
    def __init__(self, config:LLMconfig):
        super().__init__()
        self.expert = MLP(config, tp_group =None , enable_tp=False)
        
    def forward(self, x):
        return self.expert(x)

