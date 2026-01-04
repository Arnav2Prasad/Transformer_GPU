



import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model import LLMconfig
from config.cli import parallel_flag

from parallel.tp import _get_group_and_ranks , ColumnParallelLinear , RowParallelLinear




class MLP(nn.Module):
    """ A simple feed-forward network block. """
    def __init__(self, config: LLMconfig, tp_group=None, enable_tp=True):
        super().__init__()
        self.config = config
        self.non_linearity = config.non_linearity.lower()
        
        # Common dropout layer
        self.dropout = nn.Dropout(config.dropout)
        
        # Set up TP if applicable
        if parallel_flag == 5:
            self.tp_group, self.tp_size, self.tp_rank = _get_group_and_ranks(tp_group)
            self.enable_tp = (
                enable_tp and 
                (tp_group is not None) and 
                (self.tp_size > 1) and 
                dist.is_initialized()
            )
        else:
            self.enable_tp = False
        
        # Common activation function mapping
        def get_activation_func(non_linearity):
            activation_map = {
                'relu': nn.ReLU(), 'gelu': nn.GELU(), 'swish': nn.SiLU(), 'mish': nn.Mish(),
                'silu': nn.SiLU(), 'selu': nn.SELU(), 'celu': nn.CELU(), 'elu': nn.ELU(),
                'glu': nn.GLU(), 'sigmoid': nn.Sigmoid(), 
                'lrelu': nn.LeakyReLU(negative_slope=0.01), 'tanh': nn.Tanh()
            }
            return activation_map.get(non_linearity, nn.GELU())
        
        # Setup layers based on configuration
        self.setup_layers(config, get_activation_func)
    
    def setup_layers(self, config, get_activation_func):
        """Setup the MLP layers based on parallel configuration and activation type"""
        
        # For SwiGLU activation
        if self.non_linearity == 'swiglu':
            self.setup_swiglu_layers(config)
        else:
            # For other activations
            if parallel_flag == 5 and self.enable_tp:
                # Tensor Parallel path for standard activations
                self.c_fc = ColumnParallelLinear(
                    config.n_embd, config.up_dim, bias=False,
                    gather_output=False, group=self.tp_group
                )
                self.non_linearity_func = get_activation_func(self.non_linearity)
                self.c_proj = RowParallelLinear(
                    config.up_dim, config.n_embd, bias=False,
                    input_is_parallel=True, group=self.tp_group
                )
            elif parallel_flag == 5 and not self.enable_tp:
                # Non-TP fallback in parallel_flag==5
                self.c_fc = nn.Linear(config.n_embd, config.up_dim, bias=False)
                self.non_linearity_func = get_activation_func(self.non_linearity)
                self.c_proj = nn.Linear(config.up_dim, config.n_embd, bias=False)
            else:
                # Standard non-parallel path
                self.c_fc = nn.Linear(config.n_embd, config.up_dim, bias=False)
                self.non_linearity_func = get_activation_func(self.non_linearity)
                self.c_proj = nn.Linear(config.up_dim, config.n_embd, bias=False)
    
    def setup_swiglu_layers(self, config):
        """Setup layers specifically for SwiGLU activation"""
        if parallel_flag == 5 and self.enable_tp:
            # Tensor Parallel path for SwiGLU
            self.c_fc = ColumnParallelLinear(
                config.n_embd, 2 * config.up_dim, bias=False,
                gather_output=False, group=self.tp_group
            )
            self.c_proj = RowParallelLinear(
                config.up_dim, config.n_embd, bias=False,
                input_is_parallel=True, group=self.tp_group
            )
        else:
            # Non-TP or standard path for SwiGLU
            self.c_fc = nn.Linear(config.n_embd, 2 * config.up_dim, bias=False)
            self.c_proj = nn.Linear(config.up_dim, config.n_embd, bias=False)
    
    def forward(self, x):
        """Forward pass with common logic for all configurations"""
        
        # Common forward logic for SwiGLU
        if self.non_linearity == 'swiglu':
            return self.forward_swiglu(x)
        else:
            return self.forward_standard(x)
    
    def forward_swiglu(self, x):
        """Forward pass for SwiGLU activation"""
        # Chunk the output of c_fc into two parts for SwiGLU
        x1, x2 = self.c_fc(x).chunk(2, dim=-1)
        x = F.silu(x1) * x2
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
    def forward_standard(self, x):
        """Forward pass for standard activations"""
        x = self.c_fc(x)
        x = self.non_linearity_func(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
