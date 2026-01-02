

import torch
import torch.nn as nn

from config.model import LLMconfig

from models.attention.base import Attention
from train import parallel_flag
from models.moe.moe import MoE

class Block(nn.Module):
    """ A single Transformer block combining attention and MLP. """
    def __init__(self, config: LLMconfig, tp_group=None):
        super().__init__()
        self.config = config
        self.is_moe = config.moe
        self.act_recomp = config.act_recomp
        
        # Common components for all configurations
        self.attn = Attention(config, tp_group=tp_group)  # Pass tp_group to Attention
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        # Initialize EP attributes if needed
        if parallel_flag == 6:
            self._init_expert_parallel_attributes(config)
        
        # Initialize FFN (MLP or MoE)
        self._init_ffn(config, tp_group)
    
    def _init_expert_parallel_attributes(self, config):
        """Initialize expert parallel attributes if they don't exist"""
        if not hasattr(config, 'ep_rank'):
            config.ep_rank = 0
        if not hasattr(config, 'ep_size'):
            config.ep_size = 1
        if not hasattr(config, 'ep_group'):
            config.ep_group = None
    
    def _init_ffn(self, config, tp_group):
        """Initialize the feed-forward network (MLP or MoE)"""
        if config.moe:
            self.moe = MoE(config)
            self.mlp = None
        else:
            self.mlp = MLP(config, tp_group=tp_group, enable_tp=True)
            self.moe = None
    
    def _attention_forward(self, x_norm, freqs_cis, kv_cache, VAL_RUN):
        """Common attention forward pass logic"""
        if parallel_flag == 7:
            # Special handling for parallel_flag 7
            def _attn_call(z, freqs, kv):
                return self.attn(z, freqs, kv, VAL_RUN)
            
            if self.act_recomp:
                return checkpoint(_attn_call, x_norm, freqs_cis, kv_cache, use_reentrant=False)
            else:
                return self.attn(x_norm, freqs_cis, kv_cache, VAL_RUN)
        
        elif parallel_flag in [4, 5, 8]:
            # Tensor Parallel, Sequence Parallel, or Context Parallel
            if self.act_recomp:
                return checkpoint(self.attn, x_norm, freqs_cis, kv_cache, VAL_RUN, use_reentrant=False)
            else:
                return self.attn(x_norm, freqs_cis, kv_cache, VAL_RUN)
        
        else:
            # Standard forward pass
            return self.attn.forward(x_norm, freqs_cis, kv_cache, VAL_RUN)
    
    def _ffn_forward(self, x_norm):
        """Common feed-forward network forward pass logic"""
        if self.is_moe:
            moe_output, aux_loss = self.moe(x_norm)
            return moe_output, aux_loss
        else:
            mlp_output = self.mlp(x_norm)
            return mlp_output, 0.0
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor | None = None, 
                kv_cache=None, VAL_RUN=False):
        """
        Forward pass for the Transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, hidden_size)
            freqs_cis: Rotary position embeddings (optional)
            kv_cache: Key-value cache for attention (optional)
            VAL_RUN: Validation run flag (optional)
            
        Returns:
            x: Output tensor
            updated_kv_cache: Updated key-value cache
            aux_loss: Auxiliary loss (for MoE) or 0.0
        """
        # Save residual for first sub-layer
        residual = x
        
        # Layer Norm + Attention
        x_norm = self.ln1(x)
        attn_output, updated_kv_cache = self._attention_forward(x_norm, freqs_cis, kv_cache, VAL_RUN)
        
        # Add residual connection
        x = residual + attn_output
        
        # Save residual for second sub-layer
        residual = x
        
        # Layer Norm + FFN (MLP or MoE)
        x_norm = self.ln2(x)
        ffn_output, aux_loss = self._ffn_forward(x_norm)
        
        # Add residual connection
        x = residual + ffn_output
        
        return x, updated_kv_cache, aux_loss
