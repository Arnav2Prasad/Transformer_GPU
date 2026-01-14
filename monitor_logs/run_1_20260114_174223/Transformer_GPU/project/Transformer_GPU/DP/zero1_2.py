'''
Complete LLM Training Framework with Data Parallelism, ZeRO-1, and ZeRO-2
================================================================================

This implementation provides:
- Data Parallel training with DDP
- ZeRO-1: Optimizer state sharding across GPUs
- ZeRO-2: Gradient sharding across GPUs
- Mixed precision training
- Gradient accumulation
- Advanced attention mechanisms (MHA, MQA, GQA, MLA)
- Mixture of Experts (MoE) with load balancing

Credits:
- Inspired by Andrej Karpathy's nanoGPT
- ZeRO optimization inspired by DeepSpeed
- MLA implementation based on DeepSeek-V2
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import requests
import os
import argparse
import tiktoken
import numpy as np
import warnings
from typing import Literal, Optional, Dict, List, Tuple
from time import perf_counter
from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

warnings.filterwarnings("ignore")



# ============================================================================
# ZERO-2 IMPLEMENTATION: Custom Gradient Sharding
# ============================================================================


'''
In standard DDP, every GPU holds a full copy of the model, optimizer states, and gradients. ZeRO-2 eliminates the memory redundancy of optimizer states and gradients.
'''


class ZeRO2GradientHandler:
    """
    Implements ZeRO-2 gradient sharding and reduction.
    Each rank owns a shard of gradients and is responsible for updating
    the corresponding parameters.
    """
    def __init__(self, model: nn.Module, process_group=None):
        # Purpose : Distributed Setup
        # Working
        # Initializes standard DDP properties: the model being trained, 
        # the communication group (defaults to the entire WORLD), the total number of GPUs, and the current GPU's identifier.
        # 4 GPUs → world_size=4, ranks: 0,1,2,3
        self.model = model
        self.process_group = process_group if process_group else dist.group.WORLD
        self.world_size = dist.get_world_size(self.process_group)
        self.rank = dist.get_rank(self.process_group)
        
        # Partition parameters across ranks
        # Filters the model to include only trainable parameters.
        self.param_to_rank = {}             ## Maps parameter → owning GPU
        self.rank_to_params = {i: [] for i in range(self.world_size)}       # Maps GPU → parameters it owns

        '''
        Sharding Logic: This is the core partitioning algorithm. It iterates through all trainable parameters and assigns them to ranks in a round-robin fashion.
        Example : If world_size=4, parameters 1,5,9... go to rank=1, and 4,8,12... go to rank=0.
        '''
        
        params_with_grad = [p for p in model.parameters() if p.requires_grad]
        for idx, param in enumerate(params_with_grad):
            rank = idx % self.world_size        # Round-robin assignment


            # Mapping Storage
            '''
            Stores two essential dictionaries: 
            1) Which rank owns a specific parameter. 
            2) A list of parameters owned by each rank. 
            A rank is designated as the "owner" of a parameter's optimizer state and gradient.
            '''
            self.param_to_rank[param] = rank
            self.rank_to_params[rank].append(param)

        '''
        What this does:

        Divides ALL model parameters across all GPUs
        Each parameter is assigned to exactly ONE GPU (its "owner")
        Uses round-robin: param0→GPU0, param1→GPU1, param2→GPU2, param3→GPU3, param4→GPU0, etc.
        Visual Example (4 GPUs, 8 parameters):

        text
        GPU0 (Rank 0): [param0, param4]  # Owner of these parameters
        GPU1 (Rank 1): [param1, param5]  
        GPU2 (Rank 2): [param2, param6]
        GPU3 (Rank 3): [param3, param7]
        '''
        

        # Bucket Setup (Placeholder)
        '''
        Initializes structures for managing gradient transfers. 
        This specific implementation is simplified and doesn't fully utilize gradient buckling for communication overlap, 
        but it acknowledges the necessity of grouping gradients.
        '''
        # Storage for gradient buckets
        self.grad_buckets = {i: [] for i in range(self.world_size)}
        self.bucket_size = 25 * 1024 * 1024  # 25MB buckets



        
    def reduce_gradients(self):
        """
        Reduce gradients using reduce-scatter operation.
        Each rank will own a shard of the full gradients.
        """

        # Step A: Prepare Gradients by Owner
        # Group parameters by owning rank
        for rank in range(self.world_size):
            params = self.rank_to_params[rank]          # Get parameters owned by this rank
            if not params:
                continue
                
            # Flatten gradients for this rank's parameters
            grads = []
            for p in params:
                if p.grad is not None:
                    grads.append(p.grad.data.flatten())         # Convert to 1D
                else:
                    grads.append(torch.zeros_like(p.data.flatten()))             # Zero if no grad
            '''
            What happens:
                    For each GPU rank, gather ALL gradients for parameters it OWNS
                    Flatten each gradient tensor into 1D arrays
                    If a parameter has no gradient (rare), use zeros


            Example:
                GPU0 preparing gradients for parameters it OWNS:
                - param0.grad: [0.1, 0.2, 0.3] → flattened
                - param4.grad: [0.4, 0.5] → flattened  
                - Concatenated: [0.1, 0.2, 0.3, 0.4, 0.5]
            '''
            
            if grads:
                
                # Concatenate all gradients for this rank
                grad_buffer = torch.cat(grads)          # Single large 1D tensor
                
                # Reduce-scatter: sum gradients across all ranks, 
                # but each rank only keeps its shard
                '''
                dist.reduce() Operation:

                All GPUs send their local gradients for these parameters to the owner GPU
                Owner GPU sums all received gradients
                Result: Owner GPU has the globally averaged gradients

                Example:
                    Before reduce:
                        - GPU0: param0.grad = [0.1, 0.2]  (from its local batch)
                        - GPU1: param0.grad = [0.3, 0.4]  (from its local batch) 
                        - GPU2: param0.grad = [0.5, 0.6]  (from its local batch)
                        - GPU3: param0.grad = [0.7, 0.8]  (from its local batch)

                        After dist.reduce() to GPU0:
                        - GPU0: param0.grad = [1.6, 2.0]  # Sum of all GPUs: [0.1+0.3+0.5+0.7, 0.2+0.4+0.6+0.8]
                        - Other GPUs: discard their param0 gradients
                '''
                if self.world_size > 1:
                    dist.reduce(grad_buffer, dst=rank, 
                              op=dist.ReduceOp.SUM, 
                              group=self.process_group)
                
                # Unflatten back to parameters (only on owning rank)
                if self.rank == rank:           # If I am the owner of these parameters
                    offset = 0
                    for p in params:
                        numel = p.numel()           # Number of elements in this parameter
                        p.grad.data.copy_(
                            grad_buffer[offset:offset + numel].view_as(p.data)
                        )
                        offset += numel
                else:
                    # Non-owning ranks can discard gradients
                    for p in params:
                        p.grad = None

            '''
            What happens:
                Owner GPU: Takes the summed gradients and distributes them back to individual parameters
                Non-owner GPUs: Discard gradients (they don't need them for optimization)
            '''
    
    def sync_parameters(self):
        """
        Broadcast parameters from their owning rank to all other ranks.
        Called after optimizer step to synchronize model parameters.
        """
        '''
        What this does:

        After optimizer updates parameters on owner GPU...
        Owner GPU broadcasts updated parameters to all other GPUs
        Ensures all GPUs have the same model parameters
        '''
        for rank in range(self.world_size):
            params = self.rank_to_params[rank]
            for p in params:
                if self.world_size > 1:
                    dist.broadcast(p.data, src=rank, group=self.process_group)



# This ZeRO2Optimizer class is the orchestrator that coordinates ZeRO-1 and ZeRO-2
class ZeRO2Optimizer:
    """
    Wrapper around optimizer that implements ZeRO-1 and ZeRO-2.
    - ZeRO-1: Optimizer states are sharded (via ZeroRedundancyOptimizer)
    - ZeRO-2: Gradients are sharded (via custom reduce-scatter)
    """
    def __init__(self, optimizer, gradient_handler: ZeRO2GradientHandler):
        '''
        self.optimizer: A ZeRO-1 enabled optimizer (shards optimizer states)
        self.gradient_handler: A ZeRO-2 enabled gradient handler (shards gradients)
        '''
        self.optimizer = optimizer
        self.gradient_handler = gradient_handler
        
    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
    

    # The Core ZeRO-2 Algorithm
    def step(self):
        # First, reduce and shard gradients (ZeRO-2)
        self.gradient_handler.reduce_gradients()
        '''
        What happens:
        Collect gradients from all GPUs for each parameter
        Average them across all GPUs
        Shard results - each GPU keeps only gradients for parameters it owns

        Before reduce_gradients():

            
            GPU0: [∇A₁, ∇A₂, ∇A₃, ∇A₄]  # Full local gradients
            GPU1: [∇B₁, ∇B₂, ∇B₃, ∇B₄]  
            GPU2: [∇C₁, ∇C₂, ∇C₃, ∇C₄]
            GPU3: [∇D₁, ∇D₂, ∇D₃, ∇D₄]
            After reduce_gradients():

            text
            GPU0: [∇₁_avg, -, -, -]    # Only keeps averaged ∇₁
            GPU1: [-, ∇₂_avg, -, -]    # Only keeps averaged ∇₂  
            GPU2: [-, -, ∇₃_avg, -]    # Only keeps averaged ∇₃
            GPU3: [-, -, -, ∇₄_avg]    # Only keeps averaged ∇₄
        '''


        '''
        What happens:
        The underlying ZeroRedundancyOptimizer performs the update
        ZeRO-1: Each GPU only has optimizer states for parameters it owns
        ZeRO-2: Each GPU only has gradients for parameters it owns
        '''
        # Step optimizer (only updates parameters owned by this rank)
        self.optimizer.step()

        '''
        What happens:

            Each GPU broadcasts the parameters it updated to all other GPUs
            Ensures all GPUs have identical model parameters

            Before sync:

            text
            GPU0: [updated_param1, old_param2, old_param3, old_param4]
            GPU1: [old_param1, updated_param2, old_param3, old_param4]  
            GPU2: [old_param1, old_param2, updated_param3, old_param4]
            GPU3: [old_param1, old_param2, old_param3, updated_param4]
            After sync:
            ALL GPUs: [updated_param1, updated_param2, updated_param3, updated_param4]
        '''
        
        # Sync parameters across all ranks
        self.gradient_handler.sync_parameters()
    
    @property
    '''
    Purpose: Makes ZeRO2Optimizer behave like a regular optimizer

        Learning rate schedulers can access optimizer.param_groups
        Training loops can modify learning rates normally
    '''
    def param_groups(self):
        return self.optimizer.param_groups


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

@dataclass
class LLMconfig:
    # Token params
    vocab_size: int
    block_size: int
    n_embd: int
    pos_emb: str | Literal['learn', 'sin', 'rope']
    
    # Neural Network
    up_dim: int
    non_linearity: str | Literal['elu', 'lrelu', 'relu', 'gelu', 'swish', 
                                  'mish', 'silu', 'selu', 'celu', 'tanh', 
                                  'sigmoid', 'swiglu']
    dropout: float
    n_layer: int
    
    # MoE
    moe: bool
    n_exp: int
    n_shared: int
    n_act: int
    coeff: float
    aux_free: bool
    alpha: float
    gamma: float
    
    # Attention
    attn: str | Literal['mha', 'mqa', 'gqa', 'mla']
    n_head: int
    n_kv_heads: int
    q_latent_dim: int | None
    kv_latent_dim: int | None
    rope_head_dim: int | None
    
    act_recomp: bool
    
    @staticmethod
    def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Applies RoPE to query or key embeddings."""
        B, T, H, _ = x.size()
        x_ = x.float().reshape(B, T, H, -1, 2)
        x_re, x_im = x_.unbind(-1)
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
        
        x_re_out = x_re * freqs_cis.real - x_im * freqs_cis.imag
        x_im_out = x_re * freqs_cis.imag + x_im * freqs_cis.real
        
        x_out = torch.stack([x_re_out, x_im_out], dim=-1).flatten(3)
        return x_out.type_as(x)


# ============================================================================
# ATTENTION MECHANISMS
# ============================================================================

class GQA(nn.Module):
    """Grouped-Query Attention with optional RoPE"""
    
    def __init__(self, config: LLMconfig):
        super().__init__()
        if config.attn == 'mha':
            config.n_kv_heads = config.n_head
        elif config.attn == 'mqa':
            config.n_kv_heads = 1
        else:
            assert config.n_head % config.n_kv_heads == 0
        
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.head_size = config.n_embd // config.n_head
        
        self.c_attn = nn.Linear(config.n_embd, 
                               config.n_embd + 2 * config.n_kv_heads * self.head_size)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor | None, 
                kv_cache=None, VAL_RUN=False):
        B, T, C = x.size()
        nh, nkvh, hs = self.config.n_head, self.config.n_kv_heads, self.head_size
        
        q_proj_size = C
        kv_proj_size = nkvh * hs
        q, k, v = self.c_attn(x).split([q_proj_size, kv_proj_size, kv_proj_size], dim=2)
        
        q = q.view(B, T, nh, hs)
        k = k.view(B, T, nkvh, hs)
        v = v.view(B, T, nkvh, hs).transpose(1, 2)
        
        if self.config.pos_emb == 'rope':
            q = LLMconfig.apply_rotary_emb(q, freqs_cis)
            k = LLMconfig.apply_rotary_emb(k, freqs_cis)
        
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        
        updated_kv_cache = (k, v)
        
        if nkvh != nh:
            num_repeats = nh // nkvh
            k = k.repeat_interleave(num_repeats, dim=1)
            v = v.repeat_interleave(num_repeats, dim=1)
        
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.config.dropout if self.training else 0,
            is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        
        return y, updated_kv_cache


class NaiveMHLA(nn.Module):
    """Multi-Head Latent Attention without RoPE"""
    
    def __init__(self, config: LLMconfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_size = config.n_embd // config.n_head
        self.config = config
        
        self.W_dq = nn.Linear(config.n_embd, config.q_latent_dim, bias=False)
        self.W_uq = nn.Linear(config.q_latent_dim, config.n_embd, bias=False)
        self.W_dkv = nn.Linear(config.n_embd, config.kv_latent_dim, bias=False)
        self.W_uk = nn.Linear(config.kv_latent_dim, config.n_embd, bias=False)
        self.W_uv = nn.Linear(config.kv_latent_dim, config.n_embd, bias=False)
        self.W_o = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
        self.register_buffer('_k_absorbed_inference', None)
        self.register_buffer('_v_absorbed_inference', None)
    
    def _precompute_absorbed_matrices(self):
        if self._k_absorbed_inference is not None:
            return
        
        nh, n_kvl, hs = self.config.n_head, self.config.kv_latent_dim, self.head_size
        with torch.no_grad():
            self._k_absorbed_inference = (
                self.W_dq.weight.T @ self.W_uq.weight.T @ self.W_uk.weight
            ).view(nh, hs, n_kvl).unsqueeze(0)
            self._v_absorbed_inference = (
                self.W_uv.weight.T @ self.W_o.weight.T
            ).view(n_kvl, nh, hs).transpose(0, 1).unsqueeze(0)
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor | None,
                kv_cache=None, VAL_RUN=False):
        B, T, C = x.size()
        nh, n_kvl, hs = self.config.n_head, self.config.kv_latent_dim, self.head_size
        
        if self.training or VAL_RUN:
            k_eff = (
                self.W_dq.weight.T @ self.W_uq.weight.T @ self.W_uk.weight
            ).view(nh, hs, n_kvl).unsqueeze(0)
            v_eff = (
                self.W_uv.weight.T @ self.W_o.weight.T
            ).view(n_kvl, nh, hs).transpose(0, 1).unsqueeze(0)
        else:
            if self._k_absorbed_inference is None:
                self._precompute_absorbed_matrices()
            k_eff = self._k_absorbed_inference
            v_eff = self._v_absorbed_inference
        
        new_c_kv = self.W_dkv(x)
        
        if kv_cache is None:
            c_kv = new_c_kv
        else:
            c_kv = torch.cat([kv_cache, new_c_kv], dim=1)
        
        updated_kv_cache = c_kv
        T_full = c_kv.size(1)
        
        q = self.W_uq(self.W_dq(x))
        q = q.view(B, T, nh, hs).transpose(1, 2)
        
        attn = (q @ k_eff @ c_kv.transpose(1, 2).unsqueeze(1)) / math.sqrt(hs)
        
        mask = torch.triu(torch.ones(T, T_full, device=x.device, dtype=torch.bool),
                         diagonal=T_full - T + 1)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = self.dropout(F.softmax(attn, dim=-1))
        y = attn @ c_kv.unsqueeze(1) @ v_eff
        y = self.dropout(y.transpose(1, 2).contiguous().view(B, T, C))
        
        return y, updated_kv_cache


class FullMHLA(nn.Module):
    """Multi-Head Latent Attention with Decoupled RoPE (DeepSeek-V2 style)"""
    
    def __init__(self, config: LLMconfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.head_size = config.n_embd // config.n_head
        
        self.W_dq = nn.Linear(config.n_embd, config.q_latent_dim, False)
        self.W_uq = nn.Linear(config.q_latent_dim, config.n_embd, False)
        self.W_dkv = nn.Linear(config.n_embd, config.kv_latent_dim, False)
        self.W_uk = nn.Linear(config.kv_latent_dim, config.n_embd, False)
        self.W_uv = nn.Linear(config.kv_latent_dim, config.n_embd, False)
        self.W_qr = nn.Linear(config.q_latent_dim, 
                             config.n_head * config.rope_head_dim, False)
        self.W_kr = nn.Linear(config.n_embd, config.rope_head_dim, False)
        self.W_o = nn.Linear(config.n_embd, config.n_embd, False)
        
        self.dropout = nn.Dropout(config.dropout)
        
        self.register_buffer('_k_absorbed_inference', None, persistent=False)
        self.register_buffer('_v_absorbed_inference', None, persistent=False)
    
    def _precompute_absorbed_matrices(self):
        if self._k_absorbed_inference is not None:
            return
        
        nh, nlkv, hs, nlq = (self.config.n_head, self.config.kv_latent_dim,
                            self.config.n_embd // self.config.n_head,
                            self.config.q_latent_dim)
        with torch.no_grad():
            self._k_absorbed_inference = (
                self.W_uq.weight.view(1, nlq, nh, hs).transpose(1, 2) @
                self.W_uk.weight.view(1, nh, hs, nlkv)
            )
            self._v_absorbed_inference = (
                self.W_uv.weight.T @ self.W_o.weight.T
            ).view(nlkv, nh, hs).transpose(0, 1).unsqueeze(0)
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor | None,
                kv_cache=None, VAL_RUN=False):
        B, T, C = x.size()
        nh, nlkv, nlq = (self.config.n_head, self.config.kv_latent_dim,
                        self.config.q_latent_dim)
        hs = C // nh
        dhr = self.config.rope_head_dim
        
        c_q = self.W_dq(x)
        
        # NoPE component
        if self.training or VAL_RUN:
            k_eff = (
                self.W_uq.weight.view(1, nlq, nh, hs).transpose(1, 2) @
                self.W_uk.weight.view(1, nh, hs, nlkv)
            )
            v_eff = (
                self.W_uv.weight.T @ self.W_o.weight.T
            ).view(nlkv, nh, hs).transpose(0, 1).unsqueeze(0)
        else:
            if self._k_absorbed_inference is None:
                self._precompute_absorbed_matrices()
            k_eff = self._k_absorbed_inference
            v_eff = self._v_absorbed_inference
        
        new_c_kv = self.W_dkv(x)
        
        if kv_cache is None:
            c_kv = new_c_kv
        else:
            c_kv = torch.cat([kv_cache['c_kv'], new_c_kv], dim=1)
        
        T_full = c_kv.size(1)
        attn_c = c_q.unsqueeze(1) @ k_eff @ c_kv.transpose(-1, -2).unsqueeze(1)
        
        # RoPE component
        c_kr = self.W_kr(x).unsqueeze(2)
        k_r = LLMconfig.apply_rotary_emb(c_kr, freqs_cis).transpose(1, 2)
        
        if kv_cache is not None:
            k_r = torch.cat([kv_cache['k_r'], k_r], dim=2)
        
        c_qr = self.W_qr(c_q).view(B, T, nh, dhr)
        q_r = LLMconfig.apply_rotary_emb(c_qr, freqs_cis).transpose(1, 2)
        
        attn_r = q_r @ k_r.transpose(-1, -2)
        
        # Combine and mask
        attn = (attn_c + attn_r) / math.sqrt(hs + dhr)
        
        mask = torch.triu(torch.ones(T, T_full, device=x.device, dtype=torch.bool),
                         diagonal=T_full - T + 1)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = self.dropout(F.softmax(attn, dim=-1))
        y = attn @ c_kv.unsqueeze(1) @ v_eff
        y = self.dropout(y.transpose(1, 2).contiguous().view(B, T, C))
        
        updated_kv_cache = {'c_kv': c_kv, 'k_r': k_r}
        
        return y, updated_kv_cache


class Attention(nn.Module):
    """Attention router"""
    
    def __init__(self, config: LLMconfig):
        super().__init__()
        self.config = config
        if config.attn in ('mha', 'mqa', 'gqa'):
            self.attn = GQA(config)
        elif config.attn == 'mla':
            if config.pos_emb != 'rope':
                self.attn = NaiveMHLA(config)
            else:
                self.attn = FullMHLA(config)
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor | None = None,
                kv_cache=None, VAL_RUN=False):
        return self.attn(x, freqs_cis, kv_cache, VAL_RUN)


# ============================================================================
# FEED-FORWARD NETWORKS
# ============================================================================

class MLP(nn.Module):
    """Simple feed-forward network"""
    
    def __init__(self, config: LLMconfig):
        super().__init__()
        self.non_linearity = config.non_linearity.lower()
        
        if self.non_linearity == 'swiglu':
            self.c_fc = nn.Linear(config.n_embd, 2 * config.up_dim, bias=False)
            self.c_proj = nn.Linear(config.up_dim, config.n_embd, bias=False)
        else:
            non_linearity_map = {
                'relu': nn.ReLU(), 'gelu': nn.GELU(), 'swish': nn.SiLU(),
                'mish': nn.Mish(), 'silu': nn.SiLU(), 'selu': nn.SELU(),
                'celu': nn.CELU(), 'elu': nn.ELU(), 'glu': nn.GLU(),
                'sigmoid': nn.Sigmoid(), 'lrelu': nn.LeakyReLU(0.01),
                'tanh': nn.Tanh()
            }
            self.c_fc = nn.Linear(config.n_embd, config.up_dim, bias=False)
            self.non_linearity_func = non_linearity_map.get(
                self.non_linearity, nn.GELU()
            )
            self.c_proj = nn.Linear(config.up_dim, config.n_embd, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        if self.non_linearity == 'swiglu':
            x1, x2 = self.c_fc(x).chunk(2, dim=-1)
            x = F.silu(x1) * x2
        else:
            x = self.c_fc(x)
            x = self.non_linearity_func(x)
        
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Expert(nn.Module):
    """Single expert for MoE"""
    
    def __init__(self, config: LLMconfig):
        super().__init__()
        self.expert = MLP(config)
    
    def forward(self, x):
        return self.expert(x)


class MoE(nn.Module):
    """
    Mixture of Experts with DeepSeek-style load balancing.
    Supports both auxiliary loss and aux-loss-free balancing.
    """
    
    def __init__(self, config: LLMconfig):
        super().__init__()
        self.config = config
        
        self.n_shared = config.n_shared
        self.n_routed = config.n_exp - config.n_shared
        self.n_act_routed = config.n_act - config.n_shared
        assert self.n_act_routed > 0
        
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_exp)])
        self.gate = nn.Linear(config.n_embd, self.n_routed, bias=False)
        
        if config.aux_free:
            self.register_buffer('expert_bias', torch.zeros(self.n_routed))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        n_tokens = x_flat.shape[0]
        
        # Shared experts
        shared_output = torch.zeros_like(x_flat)
        if self.n_shared > 0:
            for i in range(self.n_shared):
                shared_output += self.experts[i](x_flat)
        
        # Routed experts
        router_logits = self.gate(x_flat)
        
        if self.config.aux_free:
            biased_router_logits = router_logits + self.expert_bias
            topk_biased_logits, topk_indices = torch.topk(
                biased_router_logits, self.n_act_routed, dim=1
            )
            
            topk_original_logits = torch.gather(router_logits, 1, topk_indices)
            topk_gates = F.softmax(topk_original_logits, dim=1)
            
            with torch.no_grad():
                ones = torch.ones_like(topk_indices, dtype=x_flat.dtype)
                fi_counts = torch.zeros(self.n_routed, device=x.device).scatter_add_(
                    0, topk_indices.flatten(), ones.flatten()
                )
                fi = fi_counts / n_tokens
            
            if self.training:
                with torch.no_grad():
                    ideal_load = 1.0 / self.n_routed
                    delta = ideal_load - fi
                    self.expert_bias += (self.config.gamma * delta)
            
            router_probs = F.softmax(router_logits, dim=1)
            pi = router_probs.mean(dim=0)
            aux_loss = self.config.alpha * self.n_routed * torch.sum(pi * fi)
        else:
            router_probs = F.softmax(router_logits, dim=1)
            pi = router_probs.mean(dim=0)
            
            topk_logits, topk_indices = torch.topk(
                router_logits, self.n_act_routed, dim=1
            )
            ones = torch.ones_like(topk_indices, dtype=torch.float)
            fi_counts = torch.zeros(self.n_routed, device=x.device).scatter_add_(
                0, topk_indices.flatten(), ones.flatten()
            )
            fi = fi_counts / n_tokens
            
            aux_loss = self.config.coeff * self.n_routed * torch.sum(pi * fi)
            topk_gates = F.softmax(topk_logits, dim=1)
        
        # Dispatch
        routed_output = torch.zeros_like(x_flat)
        
        for i in range(self.n_routed):
            token_indices, topk_slot = (topk_indices == i).nonzero(as_tuple=True)
            if token_indices.numel() > 0:
                tokens_for_expert = x_flat[token_indices]
                gates_for_expert = topk_gates[token_indices, topk_slot].unsqueeze(1)
                
                expert_output = self.experts[i + self.n_shared](tokens_for_expert)
                
                weighted_output = expert_output * gates_for_expert
                routed_output.index_add_(0, token_indices, weighted_output)
        
        y = (shared_output + routed_output).view(B, T, C)
        return y, aux_loss


# ============================================================================
# TRANSFORMER BLOCK
# ============================================================================

class Block(nn.Module):
    """Single Transformer block"""
    
    def __init__(self, config: LLMconfig):
        super().__init__()
        self.is_moe = config.moe
        self.attn = Attention(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        if config.moe:
            self.moe = MoE(config)
        else:
            self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor | None = None,
                kv_cache=None, VAL_RUN=False):
        attn_output, updated_kv_cache = self.attn.forward(
            self.ln1(x), freqs_cis, kv_cache, VAL_RUN
        )
        x = x + attn_output
        
        if self.is_moe:
            moe_output, aux_loss = self.moe(self.ln2(x))
            x = x + moe_output
        else:
            aux_loss = 0.0
            x = x + self.mlp(self.ln2(x))
        
        return x, updated_kv_cache, aux_loss


# ============================================================================
# MAIN LLM MODEL
# ============================================================================

class LLM(nn.Module):
    """Large Language Model with ZeRO-2 support"""
    
    def __init__(self, config: LLMconfig):
        super().__init__()
        self.config = config
        self.head_size = config.n_embd // config.n_head
        self.tkn_emb = nn.Embedding(config.vocab_size, config.n_embd)
        
        if config.pos_emb == 'learn':
            self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        elif config.pos_emb == 'sin':
            pos_emb = torch.zeros(config.block_size, config.n_embd)
            position = torch.arange(0, config.block_size, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, config.n_embd, 2).float() *
                (-math.log(10000.0) / config.n_embd)
            )
            pos_emb[:, 0::2] = torch.sin(position * div_term)
            pos_emb[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pos_emb', pos_emb)
        elif config.pos_emb == 'rope':
            self.register_buffer("freqs_cis", self._precompute_freqs_cis())
        
        self.transformer = nn.ModuleDict(dict(
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd)
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tkn_emb.weight = self.lm_head.weight  # Weight tying
        self.apply(self._init_weights)
        
        self.VAL_RUN = False
        self.print_act_recomp = config.act_recomp
        self.print_fused_adamw = False
    
    def _precompute_freqs_cis(self):
        """Precomputes rotary frequencies for RoPE"""
        d = (self.config.rope_head_dim if self.config.attn == 'mla'
             else self.head_size)
        assert d % 2 == 0
        
        theta = 1.0 / (10000.0 ** (torch.arange(0, d, 2).float() / d))
        seq = torch.arange(self.config.block_size)
        freqs = torch.outer(seq, theta)
        
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis
    
    def _init_weights(self, module):
        """Initializes model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_num_params(self):
        """Returns total and active parameters"""
        n_params = sum(p.numel() for p in self.parameters())
        
        active_params = 0
        active_params += self.tkn_emb.weight.numel()
        if self.config.pos_emb == 'learn':
            active_params += self.pos_emb.weight.numel()
        active_params += (self.transformer.ln_f.weight.numel() +
                         self.transformer.ln_f.bias.numel())
        
        for block in self.transformer.h:
            active_params += sum(p.numel() for p in block.attn.parameters())
            active_params += sum(p.numel() for p in block.ln1.parameters())
            active_params += sum(p.numel() for p in block.ln2.parameters())
            
            if block.is_moe:
                active_params += sum(p.numel() for p in block.moe.gate.parameters())
                for i in range(block.moe.n_shared):
                    active_params += sum(
                        p.numel() for p in block.moe.experts[i].parameters()
                    )
                
                if block.moe.n_routed > 0:
                    params_per_routed_expert = sum(
                        p.numel() for p in 
                        block.moe.experts[block.moe.n_shared].parameters()
                    )
                    active_params += block.moe.n_act_routed * params_per_routed_expert
            else:
                active_params += sum(p.numel() for p in block.mlp.parameters())
        
        return n_params, active_params
    
    def configure_optimizers(self, weight_decay, learning_rate, device, use_zero2=True):
        """
        Configures optimizer with ZeRO-1 and optionally ZeRO-2.
        
        Args:
            weight_decay: Weight decay for regularization
            learning_rate: Learning rate
            device: Device to place optimizer on
            use_zero2: If True, wrap optimizer with ZeRO-2 gradient sharding
        """
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # ZeRO-1: Optimizer state sharding
        from torch.distributed.optim import ZeroRedundancyOptimizer
        
        optimizer = ZeroRedundancyOptimizer(
            optim_groups,
            optimizer_class=torch.optim.AdamW,
            lr=learning_rate,
        )
        
        # ZeRO-2: Gradient sharding (optional)
        if use_zero2 and dist.is_initialized():
            gradient_handler = ZeRO2GradientHandler(self)
            optimizer = ZeRO2Optimizer(optimizer, gradient_handler)
        
        return optimizer
    
    def forward(self, idx: torch.Tensor, targets=None, kv_caches=None):
        B, T = idx.size()
        start_pos = 0
        
        if kv_caches is not None and kv_caches[0] is not None:
            if self.config.attn in ('mha', 'mqa', 'gqa'):
                start_pos = kv_caches[0][0].shape[-2]
            elif self.config.attn == 'mla':
                if self.config.pos_emb == 'rope':
                    start_pos = kv_caches[0]['c_kv'].shape[1]
                else:
                    start_pos = kv_caches[0].shape[1]
        
        tkn_emb = self.tkn_emb(idx)
        x = tkn_emb
        freqs_cis = None
        
        if self.config.pos_emb == 'rope':
            freqs_cis = self.freqs_cis[start_pos:start_pos + T]
        elif self.config.pos_emb == 'learn':
            pos = torch.arange(start_pos, start_pos + T, dtype=torch.long,
                             device=idx.device)
            x = tkn_emb + self.pos_emb(pos)
        elif self.config.pos_emb == 'sin':
            pos = torch.arange(start_pos, start_pos + T, dtype=torch.long,
                             device=idx.device)
            x = tkn_emb + self.pos_emb[pos]
        
        x = self.transformer.drop(x)
        
        if kv_caches is None:
            kv_caches = [None] * self.config.n_layer
        
        updated_kv_caches = []
        total_aux_loss = 0.0
        for i, block in enumerate(self.transformer.h):
            if not self.config.act_recomp:
                x, updated_kv_cache, aux_loss = block(
                    x, freqs_cis, kv_caches[i], self.VAL_RUN
                )
            else:
                x, updated_kv_cache, aux_loss = checkpoint(
                    block, x, freqs_cis, kv_caches[i], self.VAL_RUN
                )
            
            updated_kv_caches.append(updated_kv_cache)
            total_aux_loss += aux_loss
        
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            main_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            loss = main_loss + total_aux_loss / self.config.n_layer
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss, updated_kv_caches
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, topk: int | None = None):
        """Generates text autoregressively"""
        self.eval()
        kv_caches = [None] * self.config.n_layer
        
        for i in range(max_new_tokens):
            if i == 0:
                idx_cond = (idx if idx.size(1) <= self.config.block_size
                           else idx[:, -self.config.block_size:])
                input_for_forward = idx_cond
            else:
                input_for_forward = idx[:, -1:]
            
            if kv_caches[0] is not None:
                if self.config.attn in ('mha', 'mqa', 'gqa'):
                    cache_len = kv_caches[0][0].shape[-2]
                elif self.config.attn == 'mla':
                    cache_len = (kv_caches[0]['c_kv'].shape[1]
                               if self.config.pos_emb == 'rope'
                               else kv_caches[0].shape[1])
                
                if cache_len >= self.config.block_size:
                    keep_len = self.config.block_size - 1
                    for layer_idx in range(self.config.n_layer):
                        layer_cache = kv_caches[layer_idx]
                        if self.config.attn in ('mha', 'mqa', 'gqa'):
                            k, v = layer_cache
                            kv_caches[layer_idx] = (
                                k[..., -keep_len:, :],
                                v[..., -keep_len:, :]
                            )
                        elif self.config.attn == 'mla':
                            if self.config.pos_emb == 'rope':
                                layer_cache['c_kv'] = layer_cache['c_kv'][:, -keep_len:, :]
                                layer_cache['k_r'] = layer_cache['k_r'][:, :, -keep_len:, :]
                            else:
                                kv_caches[layer_idx] = layer_cache[:, -keep_len:, :]
            
            logits, _, kv_caches = self.forward(input_for_forward, kv_caches=kv_caches)
            logits = logits[:, -1, :]
            
            if temperature > 0:
                logits = logits / temperature
            if topk is not None:
                v, _ = torch.topk(logits, min(topk, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        self.train()
        return idx


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

@dataclass
class Trainconfig:
    dataset: str | Literal['shakespeare', 'tinystories', 'fineweb']
    total_batch_size: int
    batch_size: int
    max_iters: int
    eval: bool
    eval_interval: int
    eval_iters: int
    learning_rate: float
    warmup_steps: int
    grad_clip: float
    compile: bool
    save_model: bool
    file_name: str
    act_recomp: bool
    use_zero2: bool  # NEW: Enable ZeRO-2


# ============================================================================
# DATA LOADING
# ============================================================================

def tokenize_and_save():
    """Downloads and tokenizes Shakespeare dataset"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()
        text = response.text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the dataset: {e}")
        return
    
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    tokens = np.array(tokens, dtype=np.uint16)
    
    n = int(0.9 * len(tokens))
    train_data = tokens[:n]
    val_data = tokens[n:]
    
    data_splits = {'train': train_data, 'val': val_data}
    for split, data in data_splits.items():
        file_path = f'{split}.bin'
        with open(file_path, 'wb') as f:
            f.write(data.tobytes())


class DataLoader:
    """Memory-mapped data loader for efficient loading"""
    
    def __init__(self, B, T, file_path, device):
        self.B = B
        self.T = T
        self.file_path = file_path
        self.device = device
        self.device_type = 'cuda'
        
        self.tokens = np.memmap(self.file_path, dtype=np.uint16, mode='r')
        self.N = len(self.tokens)
        if self.B * self.T + 1 > self.N:
            raise ValueError(
                f"Batch size {B} and block size {T} are too large "
                f"for dataset of length {self.N}"
            )
    
    def next_batch(self):
        """Returns (x, y) batch"""
        B, T = self.B, self.T
        
        start_indices = torch.randint(0, self.N - T - 1, (B,))
        
        x_list = []
        y_list = []
        for start in start_indices:
            seq = self.tokens[start:start + T + 1].astype(np.int64)
            x_list.append(seq[:-1])
            y_list.append(seq[1:])
        
        x = torch.from_numpy(np.stack(x_list)).long()
        y = torch.from_numpy(np.stack(y_list)).long()
        
        if self.device_type == 'cuda':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def get_lr(iter, config: Trainconfig):
    """Cosine learning rate schedule with warmup"""
    max_lr = config.learning_rate
    min_lr = max_lr * 0.1
    max_decay_steps = config.max_iters + 2
    
    if iter < config.warmup_steps:
        return max_lr * (iter + 1) / config.warmup_steps
    elif iter > max_decay_steps:
        return min_lr
    else:
        decay_ratio = ((iter - config.warmup_steps) /
                      (max_decay_steps - config.warmup_steps))
        decay_ratio = min(decay_ratio, 1.0)
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(model: LLM, config: Trainconfig,
                  train_loader: DataLoader, val_loader: DataLoader):
    """Estimates train and validation loss"""
    out = {}
    model.eval()
    model.VAL_RUN = True
    
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = loader.next_batch()
            # Use autocast context if available
            if hasattr(model, 'ctx'):
                with model.ctx:
                    _, loss, _ = model(X, Y)
            else:
                _, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    model.VAL_RUN = False
    return out


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Train LLM with DP + ZeRO-1 + ZeRO-2'
    )
    
    # Training parameters
    parser.add_argument('--dataset', type=str, default='shakespeare')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--total_batch_size_str', type=str, default='4096')
    parser.add_argument('--max_iters', type=int, default=2500)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--eval_iters', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--act_recomp', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--file_name', type=str, default='llm_model')
    parser.add_argument('--use_zero2', action='store_true',
                       help='Enable ZeRO-2 gradient sharding')
    
    # Model parameters
    parser.add_argument('--vocab_size', type=int, default=50304)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--pos_emb', type=str, default='rope')
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--up_dim', type=int, default=512)
    parser.add_argument('--non_linearity', type=str, default='swiglu')
    
    # MoE parameters
    parser.add_argument('--moe', action='store_true')
    parser.add_argument('--n_exp', type=int, default=8)
    parser.add_argument('--n_shared', type=int, default=1)
    parser.add_argument('--n_act', type=int, default=4)
    parser.add_argument('--coeff', type=float, default=0.01)
    parser.add_argument('--aux_free', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.001)
    
    # Attention parameters
    parser.add_argument('--attn', type=str, default='mla')
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_kv_heads', type=int, default=4)
    parser.add_argument('--q_latent_dim', type=int, default=32)
    parser.add_argument('--kv_latent_dim', type=int, default=32)
    parser.add_argument('--rope_head_dim', type=int, default=16)
    
    return parser.parse_args()


def main():
    """Main training function"""
    # Check CUDA availability
    assert torch.cuda.is_available(), "CUDA is required for training"
    
    # Initialize distributed training
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f"cuda:{ddp_local_rank}"
    master_process = ddp_rank == 0
    
    if master_process:
        print(f"=== Distributed Training Setup ===")
        print(f"World Size: {ddp_world_size}")
        print(f"Using device: {device}")
    
    torch.cuda.set_device(device)
    torch.manual_seed(1729 + ddp_rank)
    torch.cuda.manual_seed(1729 + ddp_rank)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Mixed precision setup
    dtype = 'float16'
    ctx = torch.amp.autocast(device_type="cuda", dtype=getattr(torch, dtype))
    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
    
    # Parse arguments
    args = parse_args()
    
    # Create configurations
    TrainingConfig = Trainconfig(
        dataset=args.dataset,
        total_batch_size=eval(args.total_batch_size_str),
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        eval=args.eval,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        compile=False if os.name != 'posix' else True,
        save_model=args.save_model,
        file_name=args.file_name,
        act_recomp=args.act_recomp,
        use_zero2=args.use_zero2
    )
    
    ModelConfig = LLMconfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        pos_emb=args.pos_emb,
        up_dim=args.up_dim,
        non_linearity=args.non_linearity,
        dropout=args.dropout,
        n_layer=args.n_layer,
        moe=args.moe,
        n_exp=args.n_exp,
        n_shared=args.n_shared,
        n_act=args.n_act,
        coeff=args.coeff,
        aux_free=args.aux_free,
        alpha=args.alpha,
        gamma=args.gamma,
        attn=args.attn,
        n_head=args.n_head,
        n_kv_heads=args.n_kv_heads,
        q_latent_dim=args.q_latent_dim,
        kv_latent_dim=args.kv_latent_dim,
        rope_head_dim=args.rope_head_dim,
        act_recomp=args.act_recomp
    )
    
    # Validate attention configuration
    if ModelConfig.attn == 'mha':
        ModelConfig.n_kv_heads = ModelConfig.n_head
    elif ModelConfig.attn == 'mqa':
        ModelConfig.n_kv_heads = 1
    elif ModelConfig.attn == 'mla':
        assert ModelConfig.q_latent_dim is not None
        assert ModelConfig.kv_latent_dim is not None
        if ModelConfig.pos_emb == 'rope':
            assert ModelConfig.rope_head_dim is not None
    
    # Prepare dataset
    if master_process:
        print("\n=== Preparing Dataset ===")
        tokenize_and_save()
        print("Dataset prepared successfully")
    
    # Wait for dataset preparation
    if dist.is_initialized():
        dist.barrier()
    
    # Create data loaders
    train_loader = DataLoader(
        B=TrainingConfig.batch_size,
        T=ModelConfig.block_size,
        file_path="train.bin",
        device=device
    )
    val_loader = DataLoader(
        B=TrainingConfig.batch_size,
        T=ModelConfig.block_size,
        file_path="val.bin",
        device=device
    )
    
    # Calculate gradient accumulation steps
    total_batch_size = TrainingConfig.total_batch_size
    B = TrainingConfig.batch_size
    T = ModelConfig.block_size
    assert total_batch_size % (B * T * ddp_world_size) == 0
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    
    if master_process:
        print(f"\n=== Training Configuration ===")
        print(f"Total batch size: {total_batch_size}")
        print(f"Micro batch size: {B}")
        print(f"Sequence length: {T}")
        print(f"Gradient accumulation steps: {grad_accum_steps}")
        print(f"ZeRO-2 enabled: {TrainingConfig.use_zero2}")
    
    # Create model
    model = LLM(ModelConfig).to(device)
    model.ctx = ctx  # Attach context for evaluation
    
    if master_process:
        total, active = model.get_num_params()
        print(f"\n=== Model Statistics ===")
        print(f"Total parameters: {total:,}")
        print(f"Active parameters: {active:,}")
        print(f"Activation recomputation: {model.print_act_recomp}")
    
    # Wrap with DDP
    model = DDP(model, device_ids=[ddp_local_rank],
                find_unused_parameters=ModelConfig.moe)
    
    # Compile model (if supported)
    if TrainingConfig.compile:
        if master_process:
            print("Compiling model...")
        model = torch.compile(model)
    
    raw_model: LLM = model.module
    
    # Configure optimizer with ZeRO-1 and ZeRO-2
    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=TrainingConfig.learning_rate,
        device=device,
        use_zero2=TrainingConfig.use_zero2
    )
    
    if master_process:
        print(f"\n=== Optimizer Configuration ===")
        if TrainingConfig.use_zero2:
            print("Using ZeRO-1 (optimizer state sharding) + ZeRO-2 (gradient sharding)")
        else:
            print("Using ZeRO-1 (optimizer state sharding) only")
        mem_after = torch.cuda.memory_allocated(device) / 1024**3
        print(f"GPU memory after setup: {mem_after:.2f} GB")
    
    # Training loop
    if master_process:
        print(f"\n=== Starting Training ===")
        print(f"Training for {TrainingConfig.max_iters} iterations")
    
    x, y = train_loader.next_batch()
    loss_stats = []
    
    for iter in range(TrainingConfig.max_iters + 1):
        t0 = perf_counter()
        
        # Learning rate schedule
        lr = get_lr(iter, TrainingConfig)
        for param_grp in optimizer.param_groups:
            param_grp['lr'] = lr
        
        optimizer.zero_grad(set_to_none=True)
        
        # Evaluation
        eval_time = 0
        if (TrainingConfig.eval and 
            (iter % TrainingConfig.eval_interval == 0 or 
             iter == TrainingConfig.max_iters) and iter != 0):
            eval_start = perf_counter()
            losses = estimate_loss(model, TrainingConfig, train_loader, val_loader)
            eval_time = perf_counter() - eval_start
            if master_process:
                print(f"\n{'='*60}")
                print(f"Evaluation at step {iter}")
                print(f"Train loss: {losses['train']:.4f}")
                print(f"Val loss: {losses['val']:.4f}")
                print(f"Eval time: {eval_time*1000:.2f}ms")
                print(f"{'='*60}\n")
            t0 = perf_counter()
        
        # Training step with gradient accumulation
        for micro_step in range(grad_accum_steps):
            # Sync gradients only on the last micro-step
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            
            with ctx:
                _, loss, _ = model(x, y)
                loss = loss / grad_accum_steps
            
            # Prefetch next batch asynchronously
            x, y = train_loader.next_batch()
            loss_stats.append(loss.cpu())
            scaler.scale(loss).backward()
        
        # Gradient clipping
        if TrainingConfig.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                          TrainingConfig.grad_clip)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Logging
        if master_process:
            torch.cuda.synchronize()
            dt = (perf_counter() - t0) * 1000
            tokens_per_sec = (B * T * grad_accum_steps * ddp_world_size) / (dt / 1000)
            
            if iter % 10 == 0:
                print(f"Step {iter:5d} | "
                      f"Loss: {loss*grad_accum_steps:.4f} | "
                      f"LR: {lr:.6f} | "
                      f"Time: {dt:.2f}ms | "
                      f"Tokens/sec: {tokens_per_sec:.0f}")
    
    # Cleanup
    destroy_process_group()
    
    # Save model
    if TrainingConfig.save_model and master_process:
        checkpoint = {
            'config': ModelConfig,
            'model_state': raw_model.state_dict(),
            'iter_num': iter,
            'train_losses': loss_stats
        }
        checkpoint_path = f'{TrainingConfig.file_name}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"\nCheckpoint saved to {checkpoint_path}")
    
    if master_process:
        print("\n=== Training Complete ===")


if __name__ == '__main__':
    main()












# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

"""
HOW TO RUN:

1. Single GPU (No distributed training, only DDP wrapper):
   python train.py --batch_size 4 --max_iters 1000

2. Multi-GPU with ZeRO-1 (Optimizer state sharding):
   torchrun --nproc_per_node=4 train.py --batch_size 2 --max_iters 2500

3. Multi-GPU with ZeRO-1 + ZeRO-2 (Optimizer + Gradient sharding):
   torchrun --nproc_per_node=4 train.py --batch_size 2 --max_iters 2500 --use_zero2

4. With MoE:
   torchrun --nproc_per_node=4 train.py --moe --n_exp 8 --n_shared 1 --n_act 4 --use_zero2

5. With activation recomputation (saves memory):
   torchrun --nproc_per_node=4 train.py --act_recomp --use_zero2

6. Full example with all features:
   torchrun --nproc_per_node=4 train.py \
       --batch_size 2 \
       --total_batch_size_str "2**12" \
       --max_iters 5000 \
       --learning_rate 3e-4 \
       --warmup_steps 100 \
       --grad_clip 1.0 \
       --moe \
       --n_exp 8 \
       --n_shared 1 \
       --n_act 4 \
       --aux_free \
       --attn mla \
       --n_head 8 \
       --n_kv_heads 4 \
       --q_latent_dim 32 \
       --kv_latent_dim 32 \
       --rope_head_dim 16 \
       --act_recomp \
       --use_zero2 \
       --eval \
       --eval_interval 100 \
       --save_model

MEMORY COMPARISON:
- Standard DDP: Each GPU holds full model + optimizer state + gradients
- DDP + ZeRO-1: Optimizer state sharded across GPUs (~4x memory reduction)
- DDP + ZeRO-1 + ZeRO-2: Optimizer + gradients sharded (~8x memory reduction)

KEY FEATURES:
1. ZeRO-1: Optimizer state partitioning via ZeroRedundancyOptimizer
2. ZeRO-2: Custom gradient sharding with reduce-scatter operations
3. Mixed precision training (FP16/BF16)
4. Gradient accumulation for large effective batch sizes
5. Activation recomputation for memory efficiency
6. KV caching for efficient inference
7. Multiple attention mechanisms (MHA, MQA, GQA, MLA)
8. Mixture of Experts with load balancing
9. Cosine learning rate schedule with warmup
10. Async data loading with memory mapping

ARCHITECTURE HIGHLIGHTS:
- Supports MHA, MQA, GQA (standard attention variants)
- Supports MLA (Multi-Head Latent Attention from DeepSeek-V2)
- Supports RoPE, learnable, and sinusoidal positional encodings
- Supports MoE with shared experts and aux-loss-free balancing
- Weight tying between embedding and output layers
- Layer normalization with residual connections

PERFORMANCE OPTIMIZATIONS:
- TF32 enabled for faster matrix multiplications
- Gradient checkpointing (activation recomputation)
- Async gradient prefetching
- Pinned memory for faster CPU-GPU transfers
- Compiled model with torch.compile (PyTorch 2.0+)
- Efficient KV cache management for generation

ZERO-2 IMPLEMENTATION DETAILS:
The ZeRO2GradientHandler class implements gradient sharding by:
1. Partitioning parameters across ranks (round-robin)
2. Using reduce-scatter to sum gradients across ranks
3. Each rank only keeps gradients for its partition
4. Broadcasting updated parameters after optimizer step
5. This reduces memory by ~2x compared to ZeRO-1 alone

LIMITATIONS:
- ZeRO-3 (parameter sharding) is NOT implemented
- Requires NCCL backend (CUDA/GPU only)
- MoE requires find_unused_parameters=True (slightly slower)
- ZeRO-2 adds communication overhead vs pure ZeRO-1

TROUBLESHOOTING:
- Out of memory: Reduce batch_size, enable act_recomp, or use more GPUs
- Slow training: Check grad_accum_steps, disable compile if issues
- NaN loss: Reduce learning_rate, increase warmup_steps, check grad_clip
- MoE not routing: Adjust alpha/gamma for aux-free or coeff for standard
"""