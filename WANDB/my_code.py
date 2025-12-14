
'''
This script builds an LLM model based on the user's CLI inputs.
Credits:
    - This code is highly inspired by Andrej Karpathy's work on his nanoGPT : https://github.com/karpathy/nanoGPT/
    - Thanks to Vizuara AI Labs for their detailed explanations of MLA : https://youtu.be/m1x8vA_Tscc

Available settings to choose from : 
1. Attention Type (with  KV caching): 
   - Multi Head Attention (mha)
   - Multi Query Attention (mqa)
   - Grouped Query Attention (gqa)
   - Multi Head Latent Attention (mla)
   - (Work in progress) Flash Multi Head Latent Attention (fmla)

2. Positional Encodings:
   - Learnable PE
   - Sinusoidal PE
   - Rotary PE (RoPE)

3. Feed Forward Layers:
   - Dense Network Layer (moe=False): Fully Connected, MLP layer
   - Sparse Network Layer (moe=True): Mixture of Exprts
        - Load Balancing with Auxilary Loss function (aux_free = False) 
        - Shared Expert Isolation                    (n_shared = 0) 
        - Fine Grained Expert Segmentation           (set up_dim, n_exp, n_act accordingly)
        - Aux Loss Free Load Balancing               (aux_free = True)  
'''
import wandb
from datetime import datetime
import glob  # <-- MISSING
import gc    # <-- MISSING
# 1 for plain DP
# 2 for zero1
# 3 for zero2
merging_code = 3
print('1 for plain DP')
print('2 for zero1')
print('3 for zero2')
print('------')
print('merging_code : ',merging_code)


    
# 1 for DP
# 2 for FSDP
ddp_flag = 2
print('1 for DP')
print('2 for FSDP')
print('------')


print('ddp_flag : ', ddp_flag)
print('------')


tp_code = 2
print('1 for TP')
print('2 for No TP')


ep_code = 2
print('1 for EP')
print('2 for No EP')


cp_code = 2
print('1 for CP')
print('2 for No CP')


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import requests
import os
import argparse
import tiktoken
import requests

from packaging import version


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import warnings ; warnings.filterwarnings("ignore")
import os
import math
import torch
import argparse
import numpy as np

from typing import Literal
from time import perf_counter
from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

assert torch.cuda.is_available()

from math import ceil




from typing import Literal
from dataclasses import dataclass 
from torch.distributed.optim import ZeroRedundancyOptimizer

from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp.api  import ShardingStrategy, CPUOffload

try:
    ZERO_OPTIMIZER_AVAILABLE = True
except ImportError:
    ZERO_OPTIMIZER_AVAILABLE = False
    print("Warning: ZeroRedundancyOptimizer not available in this PyTorch version")

import os
os.environ['WANDB_API_KEY'] = 'c78410b3a816898642987ae3c3899430080b89d1'





from tensor_parallel_code import RowParallelLinear, ColumnParallelLinear, _get_group_and_ranks, init_distributed


from zero2_code import ZeRO2GradientHandler, ZeRO2Optimizer

from context_parallel import all_gather_sequence

from expert_parallel import EPLayout, create_worker_model, find_latest_checkpoint, save_checkpoint, load_checkpoint, finalize_training, setup_ep_groups, main_worker


@dataclass
class LLMconfig:
    # token params
    vocab_size : int
    block_size : int
    n_embd : int
    pos_emb : str | Literal['learn','sin','rope']

    # Neural Network
    up_dim  : int
    non_linearity : str | Literal['elu','lrelu','relu', 'gelu', 'swish', 'mish', 'silu', 'selu','celu','tanh','sigmoid']
    dropout : float
    n_layer : int

    # MoE
    moe : bool

    n_exp : int
    n_shared : int  
    n_act : int      ### INCLUDES THE SHARED EXPERTS
    coeff : float

    aux_free : bool
    alpha : float   # complementry aux loss coeff
    gamma: float    # bias update speed
    
    # Attention
    attn : str | Literal['mha', 'mqa', 'gqa', 'mla']
    # kv_cache : bool
    n_head : int
    n_kv_heads : int 
        # Only for mla 
    q_latent_dim  : int | None
    kv_latent_dim : int | None
    rope_head_dim : int | None

    act_recomp : bool  # more of a training param, but the best way to integrate that is to just add it here


    context_parallel_size: int = 1       
    context_parallel_rank: int = 0
    context_parallel_group: bool = None

    

    # Expert Parallelism - ADD THESE FIELDS
    ep_size: int = 1
    ep_rank: int = 0
    ep_group: any = None

    

    @staticmethod
    def apply_rotary_emb(x:torch.Tensor, freqs_cis:torch.Tensor)->torch.Tensor:
        ''' Applies RoPE to either the query or the key whose embeddings are to be rotated two at a time.'''

        # H below is either the number of total query heads(nh)
        # hs is the embedding dimension for the query/key, given by n_embd//nh
        B,T,H,_ = x.size()
        x_ = x.float().reshape(B, T, H, -1, 2)          # (B, T, H, hs)       -> (B, T, H, hs//2, 2)    -> creates the two pairs in the embd dim
        x_re, x_im = x_.unbind(-1)                      # (B, T, H, hs//2, 2) -> (B, T, H, hs//2)       -> splits those two pairs
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # (T, hs//2)          -> (1, T, 1, hs//2)       -> this has dtype complex64, so last dim has two parts, real and imaginary
        # freqs_cis has two parts : real and imaginary (cosθ, sinθ)
        # import code ; code.interact(local=locals())
        # Perform the rotation (vector * rotation matrix)
        x_re_out = x_re*freqs_cis.real - x_im*freqs_cis.imag    # (B, T, H, hs//2) * (1, T, 1, hs//2) - (B, T, H, hs//2) * (1, T, 1, hs//2) -> (B, T, H, hs//2)
        x_im_out = x_re*freqs_cis.imag + x_im*freqs_cis.real    # (B, T, H, hs//2) * (1, T, 1, hs//2) + (B, T, H, hs//2) * (1, T, 1, hs//2) -> (B, T, H, hs//2)
        
        # Stack the real and imaginary parts back together
        x_out = torch.stack([x_re_out, x_im_out], dim=-1).flatten(3) # (B, T, H, hs//2), (B, T, H, hs//2) -> (B, T, H, hs)

        return x_out.type_as(x)

    



class GQA(nn.Module):
    """ Grouped-Query Attention with or without RoPE """

    def __init__(self, config:LLMconfig , tp_group = None):
        super().__init__()

        if config.attn == 'mha' : config.n_kv_heads = config.n_head
        elif config.attn == 'mqa' : config.n_kv_heads = 1
        else : assert config.n_head % config.n_kv_heads == 0, "n_head must be divisible by n_kv_heads"
        
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.config = config

        if tp_code == 1:
            self.tp_group, self.tp_size, self.tp_rank = _get_group_and_ranks(tp_group)
            assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
            assert config.n_head % self.tp_size == 0, "n_head must be divisible by tp_size"
            assert config.n_embd % self.tp_size == 0, "n_embd must be divisible by tp_size"
        
        self.head_size = config.n_embd // config.n_head

        if tp_code == 1:
            self.n_head_per_rank = config.n_head // self.tp_size
        else:
            # k,q,v in a btach
            self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * config.n_kv_heads * self.head_size)
            # output projection
            self.c_proj = nn.Linear(config.n_embd, config.n_embd)
            # regularization
            self.attn_dropout  = nn.Dropout(config.dropout)
            self.resid_dropout = nn.Dropout(config.dropout)

        if cp_code == 1:
            self.context_parallel_group = getattr(config, 'context_parallel_group', None)


        if tp_code ==1 :
            # Critical: Handle KV head partitioning correctly
            self.partition_kv = (config.n_kv_heads % self.tp_size == 0)
            if self.partition_kv:
                self.n_kv_heads_per_rank = config.n_kv_heads // self.tp_size
                # Additional safety check for KV projection divisibility
                kv_out_features = 2 * config.n_kv_heads * self.head_size
                assert kv_out_features % self.tp_size == 0, \
                    "KV out features must be divisible by tp_size when partitioning KV"
            else:
                self.n_kv_heads_per_rank = config.n_kv_heads  # Replicated on all ranks
            
            # Q projection: Always TP-sharded
            self.q_proj = ColumnParallelLinear(
                config.n_embd, config.n_embd, 
                bias=True, gather_output=False, group=self.tp_group
            )
            
            # KV projection: Sharded only if divisible, else replicated
            kv_out_features = 2 * config.n_kv_heads * self.head_size
            if self.partition_kv:
                self.kv_proj = ColumnParallelLinear(
                    config.n_embd, kv_out_features,
                    bias=True, gather_output=False, group=self.tp_group
                )
            else:
                self.kv_proj = nn.Linear(config.n_embd, kv_out_features, bias=True)
            
            # Output projection: RowParallel with all-reduce
            self.c_proj = RowParallelLinear(
                config.n_embd, config.n_embd,
                bias=True, input_is_parallel=True, group=self.tp_group
            )
            
            self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor|None, kv_cache=None, VAL_RUN=False):
        B, T, C = x.size()

        if tp_code == 1:
            hs = self.head_size
        else:
            nh, nkvh, hs = self.config.n_head , self.config.n_kv_heads, self.head_size

        if tp_code == 1:
            q_local = self.q_proj(x)  # [B, T, C/tp_size]
            q = q_local.contiguous().view(B, T, self.n_head_per_rank, hs)
        else:
            q_proj_size = C # n_embd
            kv_proj_size = nkvh * hs
            q, k, v = self.c_attn(x).split([q_proj_size, kv_proj_size, kv_proj_size], dim=2)
            q:torch.Tensor = q.view(B, T, nh, hs) # (B, T, nh, hs)
            k:torch.Tensor = k.view(B, T, nkvh, hs) # (B, T, n_kvh, hs)
            v:torch.Tensor = v.view(B, T, nkvh, hs).transpose(1, 2) # (B, n_kvh, T, hs)


        if tp_code==2 and cp_code != 1:

            if self.config.pos_emb == 'rope' and freqs_cis is not None:
            # Apply RoPE
                q = LLMconfig.apply_rotary_emb(q, freqs_cis)
                k = LLMconfig.apply_rotary_emb(k, freqs_cis)

            q,k = q.transpose(1, 2), k.transpose(1, 2) # (B, nh, T, hs) # (B, n_kvh, T, hs)

            if kv_cache is not None:
                past_k, past_v = kv_cache
                k = torch.cat((past_k, k), dim=-2)
                v = torch.cat((past_v, v), dim=-2)

            updated_kv_cache = (k, v)

            if nkvh != nh:
                num_repeats = nh // nkvh
                k = k.repeat_interleave(num_repeats, dim=1)
                v = v.repeat_interleave(num_repeats, dim=1)

        if cp_code == 1:
            B, T_local, C = x.size()  # CHANGE: T_local instead of T

            q, k, v = self.c_attn(x).split([q_proj_size, kv_proj_size, kv_proj_size], dim=2)
            q = q.view(B, T_local, nh, hs)
            k = k.view(B, T_local, nkvh, hs)
            v = v.view(B, T_local, nkvh, hs)

            # Apply RoPE BEFORE gathering
            if self.config.pos_emb == 'rope':
                q = LLMconfig.apply_rotary_emb(q, freqs_cis)
                k = LLMconfig.apply_rotary_emb(k, freqs_cis)

            q = q.transpose(1, 2)  # (B, nh, T_local, hs)
            k = k.transpose(1, 2)  # (B, nkvh, T_local, hs) 
            v = v.transpose(1, 2)  # (B, nkvh, T_local, hs)

            # CONTEXT PARALLEL: Gather K and V along sequence dimension
            if self.config.context_parallel_size > 1:
                k = all_gather_sequence(k, dim=2, group=self.context_parallel_group)
                v = all_gather_sequence(v, dim=2, group=self.context_parallel_group)

            # Universal KV cache disabling for context parallelism
            use_cp = (self.config.context_parallel_size > 1)
            if use_cp:
                updated_kv_cache = None
            else:
                if kv_cache is not None:
                    past_k, past_v = kv_cache
                    k = torch.cat((past_k, k), dim=-2)
                    v = torch.cat((past_v, v), dim=-2)
                updated_kv_cache = (k, v)

            if nkvh != nh:
                num_repeats = nh // nkvh
                k = k.repeat_interleave(num_repeats, dim=1)
                v = v.repeat_interleave(num_repeats, dim=1)

            # Manual attention with rectangular causal mask
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(hs)

            
            # Rectangular causal mask for local Q × global K
            T_global = k.size(-2)
            shard_start = self.config.context_parallel_rank * T_local
            q_pos = shard_start + torch.arange(T_local, device=x.device)
            k_pos = torch.arange(T_global, device=x.device)
            causal_mask = (k_pos.unsqueeze(0) > q_pos.unsqueeze(1))  # (T_local, T_global)

            # Dtype-safe masking
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(attn.dtype).min)

            attn = self.attn_dropout(F.softmax(attn, dim=-1))
            y = attn @ v
            
            y = y.transpose(1, 2).contiguous().view(B, T_local, C)
            y = self.resid_dropout(self.c_proj(y))

            return y, updated_kv_cache

            

        if tp_code ==1:
            # Key-Value projection (sharded or replicated)
            kv = self.kv_proj(x)
            
            # Shape safety check for KV projection output
            if self.partition_kv:
                expected_kv_dim = 2 * self.n_kv_heads_per_rank * hs
            else:
                expected_kv_dim = 2 * self.config.n_kv_heads * hs
            
            assert kv.shape[-1] == expected_kv_dim, \
                f"KV projection output dim {kv.shape[-1]} != expected {expected_kv_dim}"
            
            # Calculate split sizes based on actual output
            kv_split_size = expected_kv_dim // 2
            k, v = kv.split([kv_split_size, kv_split_size], dim=2)
            
            # Ensure contiguity before view operations
            k = k.contiguous().view(B, T, self.n_kv_heads_per_rank, hs)
            v = v.contiguous().view(B, T, self.n_kv_heads_per_rank, hs)
            
            # Apply rotary embeddings if needed (expects [B, T, heads, hs])
            if self.config.pos_emb == 'rope' and freqs_cis is not None:
                # q = self.apply_rotary_emb(q, freqs_cis)
                # k = self.apply_rotary_emb(k, freqs_cis)
                q = LLMconfig.apply_rotary_emb(q, freqs_cis)  # ✅ Static method call
                k = LLMconfig.apply_rotary_emb(k, freqs_cis)  # ✅ Static method call
            
            # Transpose for attention: [B, heads, T, hs]
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            
            # Handle KV cache
            if kv_cache is not None:
                past_k, past_v = kv_cache
                k = torch.cat((past_k, k), dim=-2)
                v = torch.cat((past_v, v), dim=-2)
            updated_kv_cache = (k, v)
            
            # Repeat KV heads to match Q heads if needed - WITH SAFETY ASSERT
            if self.n_kv_heads_per_rank != self.n_head_per_rank:
                assert self.n_head_per_rank % self.n_kv_heads_per_rank == 0, \
                    "Local n_head must be a multiple of local n_kv_heads when repeating."
                num_repeats = self.n_head_per_rank // self.n_kv_heads_per_rank
                k = k.repeat_interleave(num_repeats, dim=1)
                v = v.repeat_interleave(num_repeats, dim=1)


        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.config.dropout if self.training else 0, is_causal=True)
        
        if tp_code == 1:
            y = y.transpose(1,2).contiguous().view(B,T,-1)
        else:
            y = y.transpose(1,2).contiguous().view(B,T,C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y, updated_kv_cache



class Attention(nn.Module):
    """ Routes the attention mechanism according to the config"""

    def __init__(self, config:LLMconfig, tp_group = None):
        super().__init__()
        self.config = config
        if config.attn in ('mha','mqa','gqa'):
            if tp_code == 1:
                self.attn = GQA(config, tp_group=tp_group)
            else:
                # Add this line to handle non-TP case
                self.attn = GQA(config)
        else:
            # Add handling for other attention types if needed
            raise ValueError(f"Unsupported attention type: {config.attn}")
    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor|None = None, kv_cache=None, VAL_RUN=False):
        return self.attn(x, freqs_cis, kv_cache, VAL_RUN)



class MLP(nn.Module):
    """ A simple feed-forward network block. """
    def __init__(self, config: LLMconfig , tp_group=None , enable_tp = True):
        super().__init__()
        self.non_linearity = config.non_linearity.lower()

        if tp_code == 1:
            self.tp_group, self.tp_size, self.tp_rank = _get_group_and_ranks(tp_group)
            self.enable_tp = (
                enable_tp and (tp_group is not None) and (self.tp_size > 1) and dist.is_initialized()
            )
        
        # if self.non_linearity == 'swiglu':
        #     # One projection, then split into two halves
        #     self.c_fc = nn.Linear(config.n_embd, 2 * config.up_dim, bias=False)
        #     self.c_proj = nn.Linear(config.up_dim, config.n_embd, bias=False)
        # else:
        #     non_linearity_map = {
        #         'relu': nn.ReLU(), 'gelu': nn.GELU(), 'swish': nn.SiLU(), 'mish': nn.Mish(),
        #         'silu': nn.SiLU(), 'selu': nn.SELU(), 'celu': nn.CELU(), 'elu': nn.ELU(),
        #         'glu' : nn.GLU(), 'sigmoid': nn.Sigmoid(),
        #         'lrelu': nn.LeakyReLU(negative_slope=0.01), 'tanh': nn.Tanh()
        #     }
        #     self.c_fc = nn.Linear(config.n_embd, config.up_dim, bias=False)
        #     self.non_linearity_func = non_linearity_map.get(self.non_linearity, nn.GELU())
        #     self.c_proj = nn.Linear(config.up_dim, config.n_embd, bias=False)

        if tp_code==1:
            # TP path: ColumnParallel (gather_output=False) then RowParallel (input_is_parallel=True)
            if self.non_linearity == 'swiglu':
                self.c_fc = ColumnParallelLinear(
                    config.n_embd, 2 * config.up_dim, bias=False,  # ✅ 2*up_dim for SwiGLU
                    gather_output=False, group=self.tp_group
                )
                self.c_proj = RowParallelLinear(
                    config.up_dim, config.n_embd, bias=False,
                    input_is_parallel=True, group=self.tp_group
                )
            else:
                self.c_fc = ColumnParallelLinear(
                    config.n_embd, config.up_dim, bias=False,
                    gather_output=False, group=self.tp_group
                )
                self.c_proj = RowParallelLinear(
                    config.up_dim, config.n_embd, bias=False,
                    input_is_parallel=True, group=self.tp_group
                )
        else:
            # Replicated fallback (no TP)
            if self.non_linearity == 'swiglu':
                self.c_fc = nn.Linear(config.n_embd, 2 * config.up_dim, bias=False)  # ✅ 2*up_dim
                self.c_proj = nn.Linear(config.up_dim, config.n_embd, bias=False)
            else:
                self.c_fc = nn.Linear(config.n_embd, config.up_dim, bias=False)
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

class Block(nn.Module):
    """ A single Transformer block combining attention and MLP. """
    def __init__(self, config:LLMconfig , tp_group = None):
        super().__init__()
        self.is_moe = config.moe
        self.act_recomp = config.act_recomp
        self.attn = Attention(config , tp_group=tp_group)
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.ln2  = nn.LayerNorm(config.n_embd)

        if ep_code == 1:
            # Initialize EP attributes if they don't exist
            if not hasattr(config, 'ep_rank'):
                config.ep_rank = 0
            if not hasattr(config, 'ep_size'):
                config.ep_size = 1
            if not hasattr(config, 'ep_group'):
                config.ep_group = None


        if config.moe:
            self.moe = MoE(config)
        else:
            # self.mlp = MLP(config)
            # ✅ MLP gets TP support
            self.mlp = MLP(config, tp_group=tp_group, enable_tp=True)

    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor|None = None, kv_cache=None, VAL_RUN=False):
        if self.act_recomp:
            attn_output, updated_kv_cache = checkpoint(self.attn, self.ln1(x), freqs_cis, kv_cache, VAL_RUN, use_reentrant=False)
        else:
            attn_output, updated_kv_cache = self.attn(self.ln1(x), freqs_cis, kv_cache, VAL_RUN)
        
        x = x + attn_output

        # NO checkpointing the MoE/MLP part -> memory grows O(T^2) for attn, O(T) for MoE, +scary looking error when we add MoE in checkpoint  
        if self.is_moe: 
            moe_output, aux_loss = self.moe(self.ln2(x))
            x = x + moe_output
        else:
            aux_loss = 0.0
            x = x + self.mlp(self.ln2(x))

        return x, updated_kv_cache, aux_loss







class Expert(nn.Module):
    """ A single feed-forward network expert. """
    def __init__(self, config:LLMconfig):
        super().__init__()
        
        self.expert = MLP(config,tp_group= None , enable_tp=False)
        
    def forward(self, x):
        return self.expert(x)

class MoE(nn.Module):
    '''
    This class implements the DeepSeekMoE layer, featuring shared and routed experts.
    It uses an Auxiliary-Loss-Free load balancing strategy with a dynamic bias term.
    Ref: https://arxiv.org/pdf/2412.19437
    '''

    def __init__(self, config: LLMconfig):
        super().__init__()
        self.config = config

        if ep_code == 1:
            if not hasattr(config, 'ep_rank'):
                config.ep_rank = 0
            if not hasattr(config, 'ep_size'):
                config.ep_size = 1
            if not hasattr(config, 'ep_group'):
                config.ep_group = None
                
            self.rank = config.ep_rank
            self.world_size = config.ep_size
            self.ep_group = config.ep_group
        
        # first `n_shared` are shared, the rest are routed
        self.n_shared = config.n_shared
        self.n_routed = config.n_exp - config.n_shared

        # Store whether we're using CP
        self.use_cp = (cp_code == 1)
        
        # Number of experts to activate from the ROUTED pool
        self.n_act_routed = config.n_act - config.n_shared

        if ep_code == 1:
            # Early return for shared-only layers
            if self.n_routed == 0:
                self.shared_only = True
                if self.rank == 0:
                    self.shared_experts = nn.ModuleList([Expert(config) for _ in range(self.n_shared)])
                else:
                    self.shared_experts = nn.ModuleList()
                return
            else:
                self.shared_only = False



        assert self.n_act_routed > 0, "Number of active experts must be greater than shared experts"


        if ep_code == 1:
            # Expert layout management
            self.layout = EPLayout(self.n_routed, self.world_size, self.rank)
            
            # Short-circuit for world size = 1
            self.use_ep = self.world_size > 1 and self.ep_group is not None
            
            # Gate and shared experts only on rank 0
            if self.rank == 0:
                self.gate = nn.Linear(config.n_embd, self.n_routed, bias=False)
                if config.aux_free:
                    self.register_buffer('expert_bias', torch.zeros(self.n_routed))
                self.shared_experts = nn.ModuleList([Expert(config) for _ in range(self.n_shared)])
            else:
                self.gate = None
                self.shared_experts = nn.ModuleList()
                if config.aux_free:
                    self.register_buffer('expert_bias', torch.zeros(1), persistent=False)
            
            # Local routed experts only
            self.local_routed_experts = nn.ModuleList([
                Expert(config) for _ in self.layout.local_global_ids
            ])
            
            # Pre-allocated buffers for performance
            self._buffers_allocated = False
            self._capacity = 0
            self._recv_tokens = None
            self._recv_gates = None
            self._recv_local_indices = None
            self._got_back = None

        else:
            self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_exp)])
            self.gate = nn.Linear(config.n_embd, self.n_routed, bias=False)
            
            if config.aux_free:
                self.register_buffer('expert_bias', torch.zeros(self.n_routed))

    def _assert(self, condition: bool, message: str):
        """Safe assertion that aborts distributed process group immediately on failure"""
        if not condition:
            if dist.is_initialized():
                dist.abort(1)  # Immediate teardown to avoid hangs
            raise AssertionError(message)

    def _allocate_buffers(self, needed: int, C: int, device: torch.device, dtype: torch.dtype):
        """Pre-allocate buffers with growth-on-demand"""
        if not self._buffers_allocated or needed > self._capacity:
            # Grow capacity by 2x or use needed, whichever is larger
            new_capacity = max(needed, self._capacity * 2) if self._buffers_allocated else needed
            self._recv_tokens = torch.empty(new_capacity, C, device=device, dtype=dtype)
            self._recv_gates = torch.empty(new_capacity, device=device, dtype=dtype)
            self._recv_local_indices = torch.empty(new_capacity, device=device, dtype=torch.long)
            self._got_back = torch.empty(new_capacity, C, device=device, dtype=dtype)
            self._capacity = new_capacity
            self._buffers_allocated = True


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        if self.use_cp:
            return self.forward_single_gpu(x)

        # Early return for shared-only layers
        if self.shared_only:
            return self._forward_shared_only(x)
        
        # Short-circuit for single GPU
        if not self.use_ep:
            return self._forward_single_gpu(x)
        
        B, T, C = x.shape
        device = x.device
        dtype = x.dtype
        
        # 1) Rank 0 computes routing and shared experts
        if self.rank == 0:
            x_flat = x.view(-1, C)  # (B*T, C)
            n_tokens = x_flat.shape[0]
            
            # Shared experts
            shared_out = torch.zeros_like(x_flat)
            if self.n_shared > 0:
                for expert in self.shared_experts:
                    shared_out += expert(x_flat)
            
            # Routing logic with fp32 for numerical stability
            router_logits = self.gate(x_flat).float()  # Compute in fp32
            
            if self.config.aux_free:
                biased_logits = router_logits + self.expert_bias
                topk_logits, topk_indices = torch.topk(biased_logits, self.n_act_routed, dim=1)
                topk_original = torch.gather(router_logits, 1, topk_indices)
                topk_gates = F.softmax(topk_original, dim=1).to(x_flat.dtype)  # Convert back to model dtype
                
                # Aux loss calculation
                with torch.no_grad():
                    ones = torch.ones_like(topk_indices, dtype=torch.float)
                    fi_counts = torch.zeros(self.n_routed, device=device).scatter_add_(0, topk_indices.flatten(), ones.flatten())
                    fi = fi_counts / n_tokens
                
                if self.training:
                    with torch.no_grad():
                        ideal_load = 1.0 / self.n_routed
                        delta = ideal_load - fi
                        # Clamp bias updates to prevent drift
                        self.expert_bias.add_(self.config.gamma * delta).clamp_(-5.0, 5.0)
                
                router_probs = F.softmax(router_logits, dim=1)
                pi = router_probs.mean(dim=0)
                aux_loss = self.config.alpha * self.n_routed * torch.sum(pi * fi)
            else:
                router_probs = F.softmax(router_logits, dim=1)
                topk_logits, topk_indices = torch.topk(router_logits, self.n_act_routed, dim=1)
                topk_gates = F.softmax(topk_logits, dim=1).to(x_flat.dtype)  # Convert back to model dtype
                
                with torch.no_grad():
                    ones = torch.ones_like(topk_indices, dtype=torch.float)
                    fi_counts = torch.zeros(self.n_routed, device=device).scatter_add_(0, topk_indices.flatten(), ones.flatten())
                    fi = fi_counts / n_tokens
                
                pi = router_probs.mean(dim=0)
                aux_loss = self.config.coeff * self.n_routed * torch.sum(pi * fi)
            
            # For top-1 routing (simplified - extend to top-k as needed)
            expert_choices = topk_indices[:, 0]  # (n_tokens,)
            gate_values = topk_gates[:, 0]       # (n_tokens,)
            
            # CORRECTED: Vectorized destination and local index computation
            n_local = self.layout.n_local
            dest_ranks = torch.div(expert_choices, n_local, rounding_mode='floor').clamp_max(self.world_size - 1)
            
            # Sort by destination for efficient all-to-all
            dest_sorted, perm = torch.sort(dest_ranks)
            xs_sorted = x_flat[perm]
            gs_sorted = gate_values[perm]
            es_sorted = expert_choices[perm]
            restore_idx = torch.argsort(perm)
            
            # CORRECTED: Local indices relative to owner
            local_indices = es_sorted - dest_sorted * n_local
            
            # Calculate split sizes with guard for empty batches
            if n_tokens > 0:
                counts = torch.bincount(dest_sorted, minlength=self.world_size)
            else:
                counts = torch.zeros(self.world_size, device=device, dtype=torch.long)
        else:
            xs_sorted = torch.empty(0, C, device=device, dtype=dtype)
            gs_sorted = torch.empty(0, device=device, dtype=dtype)
            es_sorted = torch.empty(0, device=device, dtype=torch.long)
            local_indices = torch.empty(0, device=device, dtype=torch.long)
            restore_idx = torch.empty(0, device=device, dtype=torch.long)
            counts = torch.zeros(self.world_size, device=device, dtype=torch.long)
            shared_out = torch.zeros(0, C, device=device, dtype=dtype)  # Empty tensor
            aux_loss = torch.tensor(0.0, device=device)

        # 2) CORRECTED: Build send-counts matrix S (sender x dest)
        S = torch.zeros(self.world_size, self.world_size, device=device, dtype=torch.long)
        if self.rank == 0:
            S[0] = counts  # row 0: rank 0 sends to all dests
        
        dist.broadcast(S, src=0, group=self.ep_group)

        # Safe assertion with distributed abort
        if self.rank == 0:
            self._assert(int(S.sum().item()) == B * T, 
                        f"Sum of send counts {int(S.sum().item())} must equal number of tokens {B * T}")

        # CORRECTED: Split sizes for dispatch (avoid device→host sync in loops)
        in_splits = S[self.rank].cpu().tolist()       # what I send
        out_splits = S[:, self.rank].cpu().tolist()   # what I receive

        # CRITICAL FIX: Allocate buffers based on actual received sizes, not B*T
        needed_tokens = sum(out_splits)
        self._allocate_buffers(needed_tokens, C, device, dtype)

        # 3) All-to-all: dispatch tokens to expert owners
        # Use pre-allocated buffers
        recv_tokens = self._recv_tokens[:needed_tokens]
        recv_gates = self._recv_gates[:needed_tokens]
        recv_local_indices = self._recv_local_indices[:needed_tokens]
        
        # Sanity check sizes on every rank (avoid device→host sync in loops)
        self._assert(int(S[:, self.rank].sum().item()) == recv_tokens.size(0), 
                    f"Split size mismatch: {int(S[:, self.rank].sum().item())} != {recv_tokens.size(0)}")
        
        # CRITICAL FIX: Never skip collectives - always call them even with zero sizes
        # Dispatch phase
        dist.all_to_all_single(
            recv_tokens, xs_sorted, 
            output_split_sizes=out_splits,
            input_split_sizes=in_splits, 
            group=self.ep_group
        )
        dist.all_to_all_single(
            recv_gates, gs_sorted,
            output_split_sizes=out_splits,
            input_split_sizes=in_splits, 
            group=self.ep_group
        )
        dist.all_to_all_single(
            recv_local_indices, local_indices,
            output_split_sizes=out_splits,
            input_split_sizes=in_splits, 
            group=self.ep_group
        )

        # 4) Process local experts with safety check and optimized bucketing
        y_local = torch.zeros_like(recv_tokens)
        if recv_tokens.size(0) > 0:
            # Safety check for local indices
            self._assert((recv_local_indices < len(self.local_routed_experts)).all(),
                        f"Local index out of bounds: {recv_local_indices.max()} >= {len(self.local_routed_experts)}")
            
            # OPTIMIZED: Bucket by local expert using sorting for better cache locality
            sorted_indices = torch.argsort(recv_local_indices)
            recv_lidx_sorted = recv_local_indices[sorted_indices]
            recv_tokens_sorted = recv_tokens[sorted_indices]
            recv_gates_sorted = recv_gates[sorted_indices]
            
            # Process contiguous ranges
            unique_experts, uc_counts = torch.unique_consecutive(recv_lidx_sorted, return_counts=True)
            start_idx = 0
            for expert_id, count in zip(unique_experts, uc_counts):
                end_idx = start_idx + count
                expert_tokens = recv_tokens_sorted[start_idx:end_idx]
                expert_gates = recv_gates_sorted[start_idx:end_idx]
                expert_output = self.local_routed_experts[expert_id](expert_tokens)
                y_local[sorted_indices[start_idx:end_idx]] = expert_output * expert_gates.unsqueeze(1)
                start_idx = end_idx

        # 5) CORRECTED: Return trip with proper split matrix
        # Return matrix R = S.T (owners send back to sources)
        R = S.t().contiguous()
        in_splits_back = R[self.rank].cpu().tolist()       # what I send back
        out_splits_back = R[:, self.rank].cpu().tolist()   # what I receive back

        # CRITICAL FIX: Allocate return buffers based on actual sizes
        needed_back = sum(out_splits_back)
        if needed_back > self._capacity:
            self._allocate_buffers(max(needed_tokens, needed_back), C, device, dtype)

        # CORRECTED: Allocate on ALL ranks with proper sizes using pre-allocated buffer
        send_back = y_local
        got_back = self._got_back[:needed_back]
        
        # CRITICAL FIX: Never skip collectives - always call them even with zero sizes
        dist.all_to_all_single(
            got_back, send_back,
            output_split_sizes=out_splits_back,
            input_split_sizes=in_splits_back, 
            group=self.ep_group
        )

        # 6) Rank 0 combines outputs and returns final result
        if self.rank == 0:
            # Restore original order
            y_routed = got_back[restore_idx]
            # Combine with shared output
            y_combined = (shared_out + y_routed).view(B, T, C)
            return y_combined, aux_loss
        else:
            # Other ranks return zeros (they don't contribute to final output)
            return torch.zeros(B, T, C, device=device, dtype=dtype), torch.tensor(0.0, device=device)




    def _forward_shared_only(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for shared-only MoE layers"""
        if self.rank == 0:
            B, T, C = x.shape
            x_flat = x.view(-1, C)
            shared_out = torch.zeros_like(x_flat)
            for expert in self.shared_experts:
                shared_out += expert(x_flat)
            return shared_out.view(B, T, C), torch.tensor(0.0, device=x.device)
        else:
            return torch.zeros_like(x), torch.tensor(0.0, device=x.device)



    def forward_single_gpu(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass for the DeepSeekMoE layer with Aux-Loss-Free Balancing. """
        

        B, T, C = x.shape
        x_flat = x.view(-1, C)  # Shape: (B*T, C)
        n_tokens = x_flat.shape[0]

        # ___________ SHARED EXPERT PATH ___________

        shared_output = torch.zeros_like(x_flat)
        if self.n_shared > 0:
            for i in range(self.n_shared):
                shared_output += self.experts[i](x_flat) # bypass the router

        #  ___________ ROUTED EXPERT PATH ___________

        router_logits = self.gate(x_flat)

        if self.config.aux_free:        
            # Add Bias and then select topk
            biased_router_logits = router_logits + self.expert_bias
            topk_biased_logits, topk_indices = torch.topk(biased_router_logits, self.n_act_routed, dim=1)

            # Gating weights are based on un-biased logits
            topk_original_logits = torch.gather(router_logits, 1, topk_indices) 
            topk_gates = F.softmax(topk_original_logits, dim=1)

            # Calculate expert load and update bias during training only
            with torch.no_grad():
                ones = torch.ones_like(topk_indices, dtype=x_flat.dtype)
                fi_counts = torch.zeros(self.n_routed, device=x.device).scatter_add_(0, topk_indices.flatten(), ones.flatten())
                fi = fi_counts / n_tokens

            if self.training:
                with torch.no_grad():
                    ideal_load = 1.0 / self.n_routed
                    delta = ideal_load - fi 
                    self.expert_bias += (self.config.gamma*delta)

            router_probs = F.softmax(router_logits, dim=1)
            pi = router_probs.mean(dim=0)
            aux_loss = self.config.alpha * self.n_routed * torch.sum(pi*fi)

        else:
            router_probs = F.softmax(router_logits, dim=1)
            pi = router_probs.mean(dim=0)
            
            topk_logits, topk_indices = torch.topk(router_logits, self.n_act_routed, dim=1)
            ones = torch.ones_like(topk_indices, dtype=torch.float)
            fi_counts = torch.zeros(self.n_routed, device=x.device).scatter_add_(0, topk_indices.flatten(), ones.flatten())
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

                # access the expert using an offset of `n_shared`
                expert_output = self.experts[i + self.n_shared](tokens_for_expert)
                
                weighted_output = expert_output * gates_for_expert
                routed_output.index_add_(0, token_indices, weighted_output)
        
        # combine to output
        y = (shared_output + routed_output).view(B, T, C)
        return y, aux_loss


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Custom state_dict that includes only local experts for workers"""
        state_dict = super().state_dict(destination, prefix, keep_vars)
        
        # On workers, only keep local routed experts to save space
        if self.rank != 0:
            keys_to_remove = []
            for key in list(state_dict.keys()):
                if '.local_routed_experts.' not in key:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del state_dict[key]
        
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        """Custom load_state_dict that handles distributed expert loading"""
        # Filter state_dict for current rank's experts
        if self.rank != 0:
            # Workers only load their local experts
            filtered_state_dict = {k: v for k, v in state_dict.items() if '.local_routed_experts.' in k}
            return super().load_state_dict(filtered_state_dict, strict=False)
        else:
            # Rank 0 loads everything
            return super().load_state_dict(state_dict, strict=strict)

     



class LLM(nn.Module):
    """ A simple Large language model """
    def __init__(self, config:LLMconfig , tp_group=None):
        super().__init__()
        self.config = config

        if tp_code == 1:
            self.tp_group = tp_group
        
        self.head_size = config.n_embd//config.n_head
        self.tkn_emb = nn.Embedding(config.vocab_size, config.n_embd)


        if config.pos_emb == 'rope':
            self.register_buffer("freqs_cis", self._precompute_freqs_cis())


        if tp_code == 1:
            self.transformer = nn.ModuleDict(dict(
                drop = nn.Dropout(config.dropout),
                h    = nn.ModuleList([Block(config , tp_group = tp_group) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd)))
        else:
    
            self.transformer = nn.ModuleDict(dict(
                drop = nn.Dropout(config.dropout),
                h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd)))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tkn_emb.weight = self.lm_head.weight # weight tying
        self.apply(self._init_weights)

        self.VAL_RUN=False
        self.print_act_recomp=config.act_recomp
        self.print_fused_adamw=False 

    def _precompute_freqs_cis(self):
        """Precomputes the rotary frequencies for RoPE."""
        d = self.config.rope_head_dim if self.config.attn=='mla' else self.head_size
        assert d % 2 == 0, "head dimension must be even"
        
        theta = 1.0 / (10000.0 ** (torch.arange(0, d, 2).float() / d)) # 1.0 / (base^(2i/d))
        seq = torch.arange(self.config.block_size)
        freqs = torch.outer(seq, theta)

        # Convert to complex numbers: r * e^(i*theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis
        
    def _init_weights(self, module):
        """Initializes model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # def get_num_params(self):
    #     """Returns the total number of parameters and active parameters in the model."""
    #     n_params = sum(p.numel() for p in self.parameters())
        
    #     active_params = 0

    #     active_params += self.tkn_emb.weight.numel()      # embeddings
    #     if self.config.pos_emb == 'learn': active_params += self.pos_emb.weight.numel()
    #     active_params += self.transformer.ln_f.weight.numel() + self.transformer.ln_f.bias.numel()

    #     for block in self.transformer.h:
    #         active_params += sum(p.numel() for p in block.attn.parameters())   # ----|
    #         active_params += sum(p.numel() for p in block.ln1.parameters())    #     |---> Always active
    #         active_params += sum(p.numel() for p in block.ln2.parameters())    # ----|

    #         if hasattr(block,'is_moe') and block.is_moe:

    #             active_params += sum(p.numel() for p in block.moe.gate.parameters())                # ----|
    #             for i in range(block.moe.n_shared):                                                 #     |---> Always active
    #                 active_params += sum(p.numel() for p in block.moe.experts[i].parameters())      # ----|

    #             if block.moe.n_routed > 0:
    #                 # Calculate params for one routed expert, multiply by the number of active ones
    #                 params_per_routed_expert = sum(p.numel() for p in block.moe.experts[block.moe.n_shared].parameters())
    #                 active_params += block.moe.n_act_routed * params_per_routed_expert
            
    #         else: # In case a block is not MoE
    #             active_params += sum(p.numel() for p in block.mlp.parameters())

    #     return n_params, active_params

    def get_num_params(self):
        """Returns the total number of parameters and active parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        
        active_params = 0

        active_params += self.tkn_emb.weight.numel()      # embeddings
        if self.config.pos_emb == 'learn': 
            active_params += self.pos_emb.weight.numel()
        active_params += self.transformer.ln_f.weight.numel() + self.transformer.ln_f.bias.numel()

        for block in self.transformer.h:
            active_params += sum(p.numel() for p in block.attn.parameters())   # ----|
            active_params += sum(p.numel() for p in block.ln1.parameters())    #     |---> Always active
            active_params += sum(p.numel() for p in block.ln2.parameters())    # ----|

            if block.is_moe:
                # Gate parameters
                if hasattr(block.moe, 'gate') and block.moe.gate is not None:
                    active_params += sum(p.numel() for p in block.moe.gate.parameters())
                
                # Shared experts (always active)
                if hasattr(block.moe, 'shared_experts'):
                    for expert in block.moe.shared_experts:
                        active_params += sum(p.numel() for p in expert.parameters())
                elif hasattr(block.moe, 'experts'):
                    # Fallback for non-EP case
                    for i in range(block.moe.n_shared):
                        active_params += sum(p.numel() for p in block.moe.experts[i].parameters())
                
                # Routed experts (only active ones)
                if block.moe.n_routed > 0:
                    if hasattr(block.moe, 'local_routed_experts'):
                        # EP case: calculate params per local routed expert
                        if block.moe.local_routed_experts and len(block.moe.local_routed_experts) > 0:
                            params_per_routed_expert = sum(p.numel() for p in block.moe.local_routed_experts[0].parameters())
                            active_params += block.moe.n_act_routed * params_per_routed_expert
                    elif hasattr(block.moe, 'experts'):
                        # Non-EP case
                        params_per_routed_expert = sum(p.numel() for p in block.moe.experts[block.moe.n_shared].parameters())
                        active_params += block.moe.n_act_routed * params_per_routed_expert
            
            else: # In case a block is not MoE
                active_params += sum(p.numel() for p in block.mlp.parameters())

        return n_params, active_params

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}]

        if merging_code == 1 or tp_code == 1 or ep_code == 1:

            # Create AdamW optimizer and use the fused version if it is available
            try:
                optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, fused=True)
                self.print_fused_adamw = True
            except:
                optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
            print("Using plain DP or tp_code = 1 ; this is an OR condition")

        elif merging_code == 2:
            if not ZERO_OPTIMIZER_AVAILABLE:
                raise ImportError("ZeroRedundancyOptimizer not available. Upgrade PyTorch or use merging_code=1")
            optimizer = ZeroRedundancyOptimizer(
                optim_groups,
                optimizer_class=torch.optim.AdamW,
                lr=learning_rate,
                # Removed parameters_as_bucket_view for better compatibility
            )
            print("Using ZeRO Stage 1: Optimizer State Sharding Only")
            
        elif merging_code == 3:
            # ZeRO-2: Both optimizer state AND gradient sharding

            # ZeRO-2: Both optimizer state AND gradient sharding
            if not ZERO_OPTIMIZER_AVAILABLE:
                raise ImportError("ZeroRedundancyOptimizer not available. Upgrade PyTorch or use merging_code=1")
            
            optimizer = ZeroRedundancyOptimizer(
                optim_groups,
                optimizer_class=torch.optim.AdamW,
                lr=learning_rate,
            )
            
            # ADD ZeRO-2 gradient sharding
            if dist.is_initialized():
                gradient_handler = ZeRO2GradientHandler(self)
                optimizer = ZeRO2Optimizer(optimizer, gradient_handler)
                print("Using ZeRO Stage 2: Optimizer State + Gradient Sharding")
            else:
                print("Using ZeRO Stage 1 (fallback): Distributed not initialized")

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

        tkn_emb = self.tkn_emb(idx)  # Shape: (B, T, n_embd)
        
        x = tkn_emb # Default value for x
        freqs_cis = None
        if self.config.pos_emb == 'rope':
            freqs_cis = self.freqs_cis[start_pos : start_pos + T]
            
        # if self.config.pos_emb == 'sin':
        #     pos = torch.arange(start_pos, start_pos + T, dtype=torch.long, device=idx.device)
        #     x = tkn_emb + self.pos_emb[pos]

        x = self.transformer.drop(x)

        if kv_caches is None:
            kv_caches = [None] * self.config.n_layer
        
        updated_kv_caches = []
        total_aux_loss = 0.0
        for i, block in enumerate(self.transformer.h):
            # The block now returns an auxiliary loss from the MoE layer
            if not self.config.act_recomp: 
                x, updated_kv_cache, aux_loss = block(x, freqs_cis, kv_caches[i], self.VAL_RUN)
            else : 
                x, updated_kv_cache, aux_loss = checkpoint(block, x, freqs_cis, kv_caches[i], self.VAL_RUN)

            updated_kv_caches.append(updated_kv_cache)
            total_aux_loss += aux_loss

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            main_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            # Add the accumulated auxiliary loss to the main loss
            # We divide by the number of layers because loss is accumulated from each MoE block
            loss = main_loss + total_aux_loss / self.config.n_layer
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, updated_kv_caches
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, topk: int | None = None):
        self.eval()
        kv_caches = [None] * self.config.n_layer

        for i in range(max_new_tokens):
            if i == 0:
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                input_for_forward = idx_cond
            else:
                input_for_forward = idx[:, -1:]

            if kv_caches[0] is not None:
                if self.config.attn in ('mha', 'mqa', 'gqa'):
                    cache_len = kv_caches[0][0].shape[-2]
                elif self.config.attn == 'mla':
                     cache_len = kv_caches[0]['c_kv'].shape[1] if self.config.pos_emb == 'rope' else kv_caches[0].shape[1]

                if cache_len >= self.config.block_size:
                    # Keep the most recent (block_size - 1) tokens to make space for the new one
                    keep_len = self.config.block_size - 1
                    for layer_idx in range(self.config.n_layer):
                        layer_cache = kv_caches[layer_idx]
                        if self.config.attn in ('mha', 'mqa', 'gqa'):
                            k, v = layer_cache
                            kv_caches[layer_idx] = (k[..., -keep_len:, :], v[..., -keep_len:, :])
                        elif self.config.attn == 'mla':
                            if self.config.pos_emb == 'rope':
                                layer_cache['c_kv'] = layer_cache['c_kv'][:, -keep_len:, :]
                                layer_cache['k_r']  = layer_cache['k_r'][:, :, -keep_len:, :] # Seq len is dim 2
                            else: # c_kv
                                kv_caches[layer_idx] = layer_cache[:, -keep_len:, :]

            # The forward pass now returns three items; we only need logits and caches for generation
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





if tp_code == 1:
    assert torch.cuda.device_count() > 1

# ______________DEVICE, DTYPE, DDP SETUP_________________

init_process_group(backend='nccl')


rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
device = f"cuda:{local_rank}"
master_process = rank == 0
if master_process : print(f"world_size = {world_size}")

torch.cuda.set_device(device)
torch.manual_seed(1729 + rank)         # offset the seed
torch.cuda.manual_seed(1729 + rank)    # offset the seed
torch.set_float32_matmul_precision('high') # Not sure if this has any effect when used with Auto Mixed Precision
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

dtype = 'float16' # if not torch.cuda.is_bf16_supported else 'bfloat16'
ctx = torch.amp.autocast(device_type="cuda", dtype=getattr(torch, dtype))
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

# ____________PARAMS-CONFIG_________________

@dataclass
class Trainconfig:
    dataset : str | Literal['shakespeare', 'tinystories', 'fineweb']
    total_batch_size : int
    batch_size : int
    max_iters : int
    eval : bool
    eval_interval : int
    eval_iters : int
    learning_rate : float
    warmup_steps : int
    grad_clip : int
    compile : bool #= False if os.name != 'posix' else True
    save_model : bool
    file_name : str
    act_recomp : bool

TrainingConfig = Trainconfig(
    dataset='shakespeare',
    total_batch_size = 2**12,
    batch_size = 2**1, # how many independent sequences will we process in parallel?
    max_iters = 2500,
    eval = False,
    eval_interval=100,
    eval_iters=100,
    learning_rate = 3e-4,
    warmup_steps = 100,
    grad_clip = 1.0,    
    compile = False if os.name != 'posix' else True,
    save_model = True,
    file_name='llm_model',
    act_recomp=False)   # Default to False

ModelConfig = LLMconfig(
    # token params
    vocab_size = 50304, 
    block_size = 2**10,
    n_embd = 256, 
    pos_emb = 'rope',
    
    # MoE
    moe = True,

    up_dim = 512, 
    non_linearity = 'swiglu',  
    dropout=0.0,
    n_layer = 6,

    n_exp = 16,
    n_shared = 2,
    n_act = 8,        ### INCLUDES THE SHARED EXPERTS

    coeff=0.01,
    aux_free=True,
    alpha = 0.0001,
    gamma = 0.001,

    # Attention
    attn = 'mla', 
    n_head = 8,
    n_kv_heads=4,
    # MHLA
    q_latent_dim = 32, 
    kv_latent_dim = 32,
    rope_head_dim = 16,

    ep_size= 1,
    ep_rank=0,
    ep_group=None,
    
    act_recomp=TrainingConfig.act_recomp)



def parse_args():
    parser = argparse.ArgumentParser(description='Train a simple LLM model')
    # Training Parameters
    parser.add_argument('--dataset',       type=str,   default=TrainingConfig.dataset,       help='The data set to be used for training')
    parser.add_argument('--batch_size',    type=int,   default=TrainingConfig.batch_size,    help='Batch size for training')
    parser.add_argument('--max_iters',     type=int,   default=TrainingConfig.max_iters,     help='Maximum number of iterations for training')
    parser.add_argument('--eval_interval', type=int,   default=TrainingConfig.eval_interval, help='Interval for evaluation')
    parser.add_argument('--eval_iters',    type=int,   default=TrainingConfig.eval_iters,    help='Number of iterations for evaluation')
    parser.add_argument('--learning_rate', type=float, default=TrainingConfig.learning_rate, help='Learning rate for training')
    parser.add_argument('--warmup_steps',  type=int,   default=TrainingConfig.warmup_steps,  help='Number of warmup steps for learning rate')
    parser.add_argument('--grad_clip',     type=float,  default=TrainingConfig.grad_clip,    help='Gradient Clip value')
    parser.add_argument('--act_recomp', action='store_true', help='Whether to use (selective) activation recomputation')
    
    # Model Parameters
    parser.add_argument('--vocab_size',  type=int,   default=ModelConfig.vocab_size,  help='Vocabulary size for the model')
    parser.add_argument('--block_size',  type=int,   default=ModelConfig.block_size,  help='Block size for the model')
    parser.add_argument('--n_embd',      type=int,   default=ModelConfig.n_embd,      help='Embedding dimension for the model')
    parser.add_argument('--pos_emb',     type=str,   default=ModelConfig.pos_emb,     help='Type of positional encoding (learn, sin, rope)')
    parser.add_argument('--n_layer',     type=int,   default=ModelConfig.n_layer,     help='Number of layers in the model')
    parser.add_argument('--dropout',     type=float, default=ModelConfig.dropout,     help='Dropout rate for the model')
    # MLP Params
    parser.add_argument('--up_dim',      type=int,   default=ModelConfig.up_dim,      help='Up dimension for the Expert in the model')
    parser.add_argument('--non_linearity',type=str,   default=ModelConfig.non_linearity,help='Non-linearity for the Expert in the model')
    # MoE Params
    parser.add_argument('--n_exp',       type=int,   default=ModelConfig.n_exp,       help='Number of Experts in the model')
    parser.add_argument('--n_shared',    type=int,   default=ModelConfig.n_shared,    help='Number of Shared Experts in the model')
    parser.add_argument('--n_act',       type=int,   default=ModelConfig.n_act,       help='Number of Active Experts in the model')
    parser.add_argument('--coeff',       type=float, default=ModelConfig.coeff,       help='Aux Loss Coefficient for the MoE if not using Aux Free')
    parser.add_argument('--alpha',       type=float, default=ModelConfig.alpha,       help='Complementry Loss Coefficient for the MoE if using Aux Free')
    parser.add_argument('--gamma',       type=float, default=ModelConfig.gamma,       help='Bias Update speed in Aux loss free MoE if using Aux Free')
    # Attention Params
    parser.add_argument('--attn',        type=str,   default=ModelConfig.attn,        help='Type of attention mechanism (mha, mqa, gqa, mla)')
    parser.add_argument('--n_head',      type=int,   default=ModelConfig.n_head,      help='Number of attention heads in the model')
    parser.add_argument('--n_kv_heads',  type=int,   default=ModelConfig.n_kv_heads,  help='Number of KV heads in the model (only for gqa)')
    parser.add_argument('--q_latent_dim',  type=int, default=ModelConfig.q_latent_dim,help='Query latent dimension (only for mla)')
    parser.add_argument('--kv_latent_dim', type=int, default=ModelConfig.kv_latent_dim,help='KV latent dimension (only for mla)')
    parser.add_argument('--rope_head_dim', type=int, default=ModelConfig.rope_head_dim,help='RoPE head dimension (only for mla)')
    
    parser.add_argument('--total_batch_size_str', type=str, default=str(TrainingConfig.total_batch_size), help='Total batch size for training passed in as a string expression')
    parser.add_argument('--moe',        action='store_true', help='Whether to use Mixture of Experts in the model')
    parser.add_argument('--aux_free',   action='store_true', help='Whether to use Aux Loss Free MoE')
    parser.add_argument('--eval',       action='store_true', help='Wheter to perform Evalutions once a while')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the model after training')
    parser.add_argument('--file_name', type=str, default=TrainingConfig.file_name, help='Name of the checkpoint to be saved')
    # WandB arguments
    parser.add_argument('--wandb_project', type=str, default='llm-training', 
                       help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, 
                       help='WandB entity/username')
    parser.add_argument('--wandb_run_name', type=str, default=None, 
                       help='WandB run name')
    parser.add_argument('--wandb_notes', type=str, default='', 
                       help='Notes for WandB run')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=[], 
                       help='Tags for WandB run')
    parser.add_argument('--no_wandb', action='store_true', 
                       help='Disable wandb logging')

    return parser.parse_args()

import os


if tp_code == 1:
    # Initialize TP
    ddp_info = init_distributed()
    rank = ddp_info["rank"]
    world_size = ddp_info["world_size"] 
    local_rank = ddp_info["local_rank"]
    master_process = ddp_info["is_master"]
    device = ddp_info["device"]
    tp_group = ddp_info["tp_group"]



args = parse_args()
# Initialize wandb only on master process

for key, value in vars(args).items():
    # need to eval the total_batch_size to get the grad_accum_steps
    if key == 'total_batch_size_str':
        value = eval(value)
        setattr(TrainingConfig, 'total_batch_size', value)
    elif key == 'act_recomp':
        setattr(ModelConfig, key, value)
    else:
        if isinstance(value, str) and key !='non_linearity':
            value = value.lower().strip()
        if hasattr(TrainingConfig, key):
            setattr(TrainingConfig, key, value)
        else:
            setattr(ModelConfig, key, value)
# Initialize wandb only on master process


if tp_code == 1:
    # Add TP config to ModelConfig
    ModelConfig.tp_size = world_size
    ModelConfig.tp_rank = rank


if ModelConfig.attn == 'mha':
    ModelConfig.n_kv_heads = ModelConfig.n_head
elif ModelConfig.attn == 'mqa':
    ModelConfig.n_kv_heads = 1
elif ModelConfig.attn == 'mla':
    req = ModelConfig.q_latent_dim is not None and ModelConfig.kv_latent_dim is not None
    assert req, "Either q_latent_dim or kv_latent_dim is missing"
    if ModelConfig.pos_emb == 'rope':
        assert ModelConfig.rope_head_dim is not None, "Need dim of Rotary heads"



if cp_code == 1:
    # After dist.init_process_group()
    ModelConfig.context_parallel_size = world_size
    ModelConfig.context_parallel_rank = rank
    ModelConfig.context_parallel_group = torch.distributed.group.WORLD

    # Validation
    if torch.distributed.is_initialized():
        assert ModelConfig.context_parallel_size == torch.distributed.get_world_size(), \
            f"context_parallel_size ({ModelConfig.context_parallel_size}) must equal world_size ({torch.distributed.get_world_size()})"


# _______________ DATASET _________________
def tokenize_and_save():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()  # This will raise an error for bad responses (4xx or 5xx)
        text = response.text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the dataset: {e}")
        return # Exit the function if download fails

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

tokenize_and_save() # Using The Tiny Shakespeare dataset for demo

if tp_code==1:
    # Only download dataset on master process
    if rank == 0:
        tokenize_and_save()
    if dist.is_initialized():
        dist.barrier()
else:
    tokenize_and_save() # Using The Tiny Shakespeare dataset for demo


class DataLoader:
    def __init__(self, B, T, file_path, device ,context_parallel_size=1, context_parallel_rank=0):
        self.B = B
        self.T = T
        self.file_path = file_path
        self.device = device
        # self.device_type = 'cuda'
        # FIX: Convert device string to torch.device object
        self.device = torch.device(device)
        self.device_type = self.device.type  # Store device type separately if needed

        if cp_code == 1:
            self.context_parallel_size = context_parallel_size
            self.context_parallel_rank = context_parallel_rank
            
            # Calculate local sequence length
            self.local_T = T // context_parallel_size
            assert T % context_parallel_size == 0, "Sequence length must be divisible by context parallel size"

        # Keep the memory-mapped file open persistently
        self.tokens = np.memmap(self.file_path, dtype=np.uint16, mode='r')
        self.N = len(self.tokens)
        if self.B * self.T + 1 > self.N:
            raise ValueError(f"Batch size {B} and block size {T} are too large for dataset of length {self.N}")

    def next_batch(self):
        """
        Returns (x, y) where:
        - x is (B, T) input tokens
        - y is (B, T) target tokens (shifted by one)
        """
        B, T = self.B, self.T

        if cp_code == 1:
            local_T = self.local_T 

        # Sample B random starting positions independently
        start_indices = torch.randint(0, self.N - T - 1, (B,))

        # Gather sequences
        x_list = []
        y_list = []
        for start in start_indices:
            if cp_code == 1:
                full_seq = self.tokens[start : start + self.T + 1].astype(np.int64)
            
                # Extract local chunk for this context parallel rank
                local_start = self.context_parallel_rank * local_T
                local_end = local_start + local_T
                x_local = full_seq[local_start:local_end]
                y_local = full_seq[local_start + 1:local_end + 1]
                
                x_list.append(x_local)
                y_list.append(y_local)

            else:
                seq = self.tokens[start : start + T + 1].astype(np.int64)
                x_list.append(seq[:-1])
                y_list.append(seq[1:])

        # Stack into tensors
        x = torch.from_numpy(np.stack(x_list)).long()
        y = torch.from_numpy(np.stack(y_list)).long()

        if cp_code == 1:
            # Verify local sequence length
            assert x.shape[1] == self.local_T, f"Expected local_T={self.local_T}, got {x.shape[1]}"

        # Move to device (with pinned memory if CUDA)
        if self.device_type == 'cuda':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y

    def close(self):
        """Close memory-mapped file to release resources"""
        try:
            # Access the underlying mmap object and close it
            if hasattr(self.tokens, '_mmap'):
                self.tokens._mmap.close()
            # Also try to delete the reference
            del self.tokens
        except Exception as e:
            # Silently fail if closing doesn't work
            pass



from tensor_parallel_code import broadcast_batch


if cp_code == 1:
    train_loader = DataLoader(
        B=TrainingConfig.batch_size, 
        T=ModelConfig.block_size,  # GLOBAL sequence length
        file_path="train.bin", 
        device=device,
        context_parallel_size=world_size,
        context_parallel_rank=rank
    )
    val_loader = DataLoader(
        B=TrainingConfig.batch_size, 
        T=ModelConfig.block_size,  # GLOBAL sequence length  
        file_path="val.bin", 
        device=device,
        context_parallel_size=world_size,
        context_parallel_rank=rank
    )

else:
    train_loader = DataLoader(B=TrainingConfig.batch_size, T=ModelConfig.block_size, file_path= "train.bin", device=device)
    val_loader = DataLoader(B=TrainingConfig.batch_size, T=ModelConfig.block_size, file_path="val.bin", device=device)

# ____________ UTIL FUNCTIONS _________________

def get_lr(iter, TrainingConfig:Trainconfig):
    max_lr = TrainingConfig.learning_rate
    min_lr = max_lr*0.1
    max_decay_steps = TrainingConfig.max_iters + 2 # avoid division by zero
    # 1) linear warump for warmup_steps:
    if iter < TrainingConfig.warmup_steps:
        return max_lr * (iter+1)/TrainingConfig.warmup_steps
    #2) if iter > lr_decay_iters, return min_lr
    elif iter > max_decay_steps:
        return min_lr
    #3) in between, use cosine decay
    else:
        decay_ratio = (iter - TrainingConfig.warmup_steps) / (max_decay_steps - TrainingConfig.warmup_steps)
        decay_ratio = min(decay_ratio, 1.0)  # ensure it does
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

@torch.no_grad()
def estimate_loss(model:LLM, TrainingConfig:Trainconfig, train_loader:DataLoader, val_loader:DataLoader):
    out = {}
    model.eval() ; model.VAL_RUN = True
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(TrainingConfig.eval_iters)
        for k in range(TrainingConfig.eval_iters):
            X, Y = loader.next_batch() # Data is now moved to device in next_batch()
            with ctx:
                _, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train(); model.VAL_RUN = False
    return out

#___________GRAD_ACCUM SETUP_____________


torch_dtype = getattr(torch, dtype)

if ep_code == 1:
    #__________GRAD_ACCUM SETUP____________

    total_batch_size = TrainingConfig.total_batch_size
    B = TrainingConfig.batch_size    # microbatch size
    T = ModelConfig.block_size       # sequence length
    assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
    grad_accum_steps = total_batch_size // (B * T)

    #__________CREATE YOUR MODEL____________
    model = LLM(ModelConfig).to(device)
    total, active = model.get_num_params()
    print(f"total parameters = {total:,}, acitive parameters = {active:,}")

    if TrainingConfig.compile :  
        print("Using compiled model")
        model = torch.compile(model)


    # Set NCCL environment variables for stability
    os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')


    '''
    What it does: Controls the verbosity of NCCL logging
    Levels:
        WARN: Only shows warnings and errors (balanced)
        INFO: More detailed information
        VERSION: Just version info
        TRACE: Maximum verbosity (for deep debugging)
        Why WARN: Enough info to diagnose issues without log spam
    '''
    os.environ.setdefault('NCCL_DEBUG', 'WARN') 
    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    # Make P2P configurable rather than default disabled
    if os.getenv('NCCL_P2P_DISABLE') is None:
        os.environ.setdefault('NCCL_P2P_DISABLE', '0')  # Enable by default for better perf on NVLink

    # Version check at program start
    assert version.parse(torch.__version__) >= version.parse("2.1.0"), \
        "EP MoE requires PyTorch >= 2.1.0 for autograd on all_to_all_single"



elif tp_code == 1:
    #__________GRAD_ACCUM SETUP____________

    total_batch_size = TrainingConfig.total_batch_size
    B = TrainingConfig.batch_size    # microbatch size
    T = ModelConfig.block_size       # sequence length
    # assert total_batch_size % (B * T *world_size) == 0, "make sure total_batch_size is divisible by B * T * world_size"
    # grad_accum_steps = total_batch_size // (B * T *world_size)

    # TP-only: batch size is NOT multiplied by world_size
    assert total_batch_size % (B * T) == 0, \
        f"total_batch_size {total_batch_size} must be divisible by B*T = {B}*{T}"
    grad_accum_steps = total_batch_size // (B * T)

    if master_process:
        print(f"Grad accum steps: {grad_accum_steps}")



    model = LLM(ModelConfig , tp_group=tp_group).to(device)
    use_wandb = not args.no_wandb and master_process
    if use_wandb:
        total, active = model.get_num_params()  # Get params first
        print(f"total parameters = {total:,}, active parameters = {active:,}")
        
        if not args.wandb_run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.wandb_run_name = f"{args.dataset}_{args.attn}_{timestamp}"
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            notes=args.wandb_notes,
            tags=args.wandb_tags,
            config={
                **vars(TrainingConfig),
                **vars(ModelConfig),
                "total_params": total,
                "active_params": active,
                "world_size": world_size,
                "device": device,
                "dtype": dtype,
                "grad_accum_steps": grad_accum_steps,
                "tp_code": tp_code,
                "ep_code": ep_code,
                "cp_code": cp_code,
                "ddp_flag": ddp_flag,
                "merging_code": merging_code,
            }
        )
        
        wandb.watch(model, log="all", log_freq=100)
    if master_process : 
        total, active = model.get_num_params()
        print(f"total parameters = {total:,}, acitive parameters = {active:,}")
        if model.print_fused_adamw: print("Using Fused AdamW")
        if model.print_act_recomp: print("Using Activation Recomputation")

    if master_process : print("Using compiled model")
    model = torch.compile(model)

    raw_model:LLM = model

    optimizer = model.configure_optimizers(weight_decay=0.1,learning_rate=TrainingConfig.learning_rate,device=device)

    # Initialize scaler (add this if missing)
    scaler = torch.cuda.amp.GradScaler()



elif ddp_flag == 1:
    print('inside ddp_flag = 1')
    total_batch_size = TrainingConfig.total_batch_size
    B = TrainingConfig.batch_size    # microbatch size
    T = ModelConfig.block_size       # sequence length
    assert total_batch_size % (B * T *world_size) == 0, "make sure total_batch_size is divisible by B * T * world_size"
    grad_accum_steps = total_batch_size // (B * T *world_size)

    #___________CREATE YOUR MODEL_____________
    model = LLM(ModelConfig).to(device)
    use_wandb = not args.no_wandb and master_process
    if use_wandb:
        total, active = model.get_num_params()  # Get params first
        print(f"total parameters = {total:,}, active parameters = {active:,}")
        
        if not args.wandb_run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.wandb_run_name = f"{args.dataset}_{args.attn}_{timestamp}"
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            notes=args.wandb_notes,
            tags=args.wandb_tags,
            config={
                **vars(TrainingConfig),
                **vars(ModelConfig),
                "total_params": total,
                "active_params": active,
                "world_size": world_size,
                "device": device,
                "dtype": dtype,
                "grad_accum_steps": grad_accum_steps,
                "tp_code": tp_code,
                "ep_code": ep_code,
                "cp_code": cp_code,
                "ddp_flag": ddp_flag,
                "merging_code": merging_code,
            }
        )
        
        wandb.watch(model, log="all", log_freq=100)
    if master_process : 
        total, active = model.get_num_params()
        print(f"total parameters = {total:,}, acitive parameters = {active:,}")
        if model.print_fused_adamw: print("Using Fused AdamW")
        if model.print_act_recomp: print("Using Activation Recomputation")



    model = DDP(model, device_ids=[local_rank], find_unused_parameters=ModelConfig.moe)


    if master_process : print("Using compiled model")
    model = torch.compile(model)

    raw_model:LLM = model.module

    #______________________________________________ TRAINING ______________________________________________

    optimizer = raw_model.configure_optimizers(weight_decay=0.1,learning_rate=TrainingConfig.learning_rate,device=device)
elif ddp_flag == 2:
    print('inside ddp_flag = 2')
    total_batch_size = TrainingConfig.total_batch_size
    B = TrainingConfig.batch_size    # microbatch size
    T = ModelConfig.block_size       # sequence length
    world_size = world_size
    assert total_batch_size % (B * T *world_size) == 0, "make sure total_batch_size is divisible by B * T * world_size"
    grad_accum_steps = total_batch_size // (B * T *world_size)

    #___________CREATE YOUR MODEL_____________=
    fsdp_wrap_policy = ModuleWrapPolicy({Block})

    dtype = 'float16' # if not torch.cuda.is_bf16_supported() else 'bfloat16'
    torch_dtype = getattr(torch, dtype)

    mp_policy = MixedPrecision(
        param_dtype=torch_dtype,
        reduce_dtype=torch_dtype,
        buffer_dtype=torch_dtype,
    )

    model = LLM(ModelConfig).to(device)
    use_wandb = not args.no_wandb and master_process
    if use_wandb:
        total, active = model.get_num_params()  # Get params first
        print(f"total parameters = {total:,}, active parameters = {active:,}")
        
        if not args.wandb_run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.wandb_run_name = f"{args.dataset}_{args.attn}_{timestamp}"
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            notes=args.wandb_notes,
            tags=args.wandb_tags,
            config={
                **vars(TrainingConfig),
                **vars(ModelConfig),
                "total_params": total,
                "active_params": active,
                "world_size": world_size,
                "device": device,
                "dtype": dtype,
                "grad_accum_steps": grad_accum_steps,
                "tp_code": tp_code,
                "ep_code": ep_code,
                "cp_code": cp_code,
                "ddp_flag": ddp_flag,
                "merging_code": merging_code,
            }
        )
        
        wandb.watch(model, log="all", log_freq=100)
    if master_process : 
        total, active = model.get_num_params()
        print(f"total parameters = {total:,}, acitive parameters = {active:,}")
        if model.print_fused_adamw: print("Using Fused AdamW")
        if model.print_act_recomp: print("Using Activation Recomputation")

    model = FSDP(
        model,
        auto_wrap_policy=fsdp_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD, # This is ZeRO-3
        device_id=torch.cuda.current_device(),
        # cpu_offload=CPUOffload(offload_params=True), # Optional: to save even more GPU memory
        limit_all_gathers=True, # Recommended for performance
        use_orig_params=True, # Important for optimizers like AdamW and for getting original parameters
        sync_module_states=True,
    )

    if master_process : print("Using compiled model")
    model = torch.compile(model)

    raw_model:LLM = model.module

    #______________________________________________ TRAINING ______________________________________________

    optimizer = raw_model.configure_optimizers(weight_decay=0.1,learning_rate=TrainingConfig.learning_rate,device=device)

elif cp_code == 1:
    fsdp_wrap_policy = ModuleWrapPolicy({Block})

    mp_policy = MixedPrecision(
        param_dtype=torch_dtype,
        reduce_dtype=torch_dtype,
        buffer_dtype=torch_dtype,
    )

    model = LLM(ModelConfig).to(device)
    use_wandb = not args.no_wandb and master_process
    if use_wandb:
        total, active = model.get_num_params()  # Get params first
        print(f"total parameters = {total:,}, active parameters = {active:,}")
        
        if not args.wandb_run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.wandb_run_name = f"{args.dataset}_{args.attn}_{timestamp}"
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            notes=args.wandb_notes,
            tags=args.wandb_tags,
            config={
                **vars(TrainingConfig),
                **vars(ModelConfig),
                "total_params": total,
                "active_params": active,
                "world_size": world_size,
                "device": device,
                "dtype": dtype,
                "grad_accum_steps": grad_accum_steps,
                "tp_code": tp_code,
                "ep_code": ep_code,
                "cp_code": cp_code,
                "ddp_flag": ddp_flag,
                "merging_code": merging_code,
            }
        )
        
        wandb.watch(model, log="all", log_freq=100)
    if master_process : 
        total, active = model.get_num_params()
        print(f"total parameters = {total:,}, acitive parameters = {active:,}")
        if model.print_fused_adamw: print("Using Fused AdamW")
        if model.print_act_recomp: print("Using Activation Recomputation")



    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)


    if master_process : print("Using compiled model")
    model = torch.compile(model)

    raw_model:LLM = model.module

    #______________________________________________ TRAINING ______________________________________________

    optimizer = raw_model.configure_optimizers(weight_decay=0.1,learning_rate=TrainingConfig.learning_rate,device=device)



# ===============
# EP code
# =========
'''
The class handles the mapping between global expert IDs and their local placement across different GPU ranks in an Expert Parallel setup.
'''







def save_checkpoint_with_wandb(model, optimizer, iter, rank, checkpoint_dir="checkpoints", use_wandb=False):
    """Save checkpoint and log as wandb artifact"""
    save_checkpoint(model, optimizer, iter, rank, checkpoint_dir)
    
    # Log as artifact on rank 0
    if rank == 0 and use_wandb:
        artifact = wandb.Artifact(
            name=f"model-checkpoint-iter-{iter}",
            type="model",   
            description=f"Model checkpoint at iteration {iter}"
        )
        
        # Add all checkpoint files
        for file in os.listdir(checkpoint_dir):
            if f"_iter_{iter}" in file:
                artifact.add_file(os.path.join(checkpoint_dir, file))
        
        wandb.log_artifact(artifact)



from expert_parallel import main


def print_all_gpu_memory(prefix=""):
    # Get memory for this process's GPU
    local_allocated = torch.cuda.memory_allocated(local_rank) / 1024**3
    local_reserved = torch.cuda.memory_reserved(local_rank) / 1024**3
    
    # Convert to tensors for gathering
    allocated_tensor = torch.tensor(local_allocated).to(device)
    reserved_tensor = torch.tensor(local_reserved).to(device)
    
    # Create lists to gather into
    allocated_list = [torch.zeros(1).to(device) for _ in range(world_size)]
    reserved_list = [torch.zeros(1).to(device) for _ in range(world_size)]
    
    # Gather from all processes
    torch.distributed.all_gather(allocated_list, allocated_tensor)
    torch.distributed.all_gather(reserved_list, reserved_tensor)
    
    # Only master process prints the complete picture
    if master_process:
        print(f"\n{prefix} GPU Memory Usage:")
        for i in range(world_size):
            print(f"  GPU {i}: {allocated_list[i].item():.2f} GB allocated, {reserved_list[i].item():.2f} GB reserved")

# Print initial memory
print('inital GPU memory')
print_all_gpu_memory("Initial")

if ep_code == 1:
    if __name__ == "__main__":
        main()

else:
    if tp_code == 1:
        if master_process:
            x, y = train_loader.next_batch()
        else:
            x = torch.empty(B, T, dtype=torch.long, device=device)
            y = torch.empty(B, T, dtype=torch.long, device=device)

        x, y = broadcast_batch(x, y, src=0)

    else:
        x,y = train_loader.next_batch() # get the first batch of training data


    loss_stats = []
    for iter in range(TrainingConfig.max_iters+1):
        t0 = perf_counter()

        if iter % 100 == 0:
            print_all_gpu_memory(f"Iteration {iter}")

        lr = get_lr(iter, TrainingConfig) 
        for param_grp in optimizer.param_groups:
            param_grp['lr'] = lr
        
        optimizer.zero_grad(set_to_none=True)

        a,b = 0,0
        if TrainingConfig.eval and (iter % TrainingConfig.eval_interval == 0 or iter == TrainingConfig.max_iters) and iter!=0:
            a = perf_counter()
            losses = estimate_loss(model, TrainingConfig, train_loader, val_loader)
            b = perf_counter()
            if master_process:
                print(f"--------val run-------- train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | dt {1000*(b-a):.4f}ms")
            t0 = b
        
        
        # Add evaluation logging
        if TrainingConfig.eval and (iter % TrainingConfig.eval_interval == 0 or iter == TrainingConfig.max_iters) and iter != 0:
            losses = estimate_loss(model, TrainingConfig, train_loader, val_loader)
            
            if master_process:
                print(f"--------val run-------- train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | dt {1000*(b-a):.4f}ms")
                
                if use_wandb:
                    wandb.log({
                        "eval/train_loss": losses['train'],
                        "eval/val_loss": losses['val'],
                        "eval/step": iter,
                    })

        for micro_step in range(grad_accum_steps):

            if tp_code == 1 or cp_code == 2:
                if master_process:
                    x, y = train_loader.next_batch()
                else:
                    x = torch.empty(B, T, dtype=torch.long, device=device)
                    y = torch.empty(B, T, dtype=torch.long, device=device)
                
                # 2. Ensure all GPUs have the same data BEFORE the forward pass
                x, y = broadcast_batch(x, y, src=0)
                
                # Use autocast for mixed precision
                with torch.cuda.amp.autocast(dtype=torch_dtype): # Make sure torch_dtype is defined
                    _, loss, _ = model(x, y)
                    loss = loss / grad_accum_steps

                # Scale the loss and call backward
                scaler.scale(loss).backward()

            else:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

                with ctx:
                    _, loss, _ = model(x,y)
                    loss:torch.Tensor = loss/grad_accum_steps

                x,y = train_loader.next_batch() # Async prefetch the next batch of data
                loss_stats.append(loss.cpu())
                scaler.scale(loss).backward()

        if TrainingConfig.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.grad_clip)

        scaler.step(optimizer)
        scaler.update()    

        if master_process:
            torch.cuda.synchronize()
            dt  = (perf_counter()-t0)*1000
            print(f"step: {iter} | train loss:{loss*grad_accum_steps:.4f} | dt: {dt:.2f}ms")

            if use_wandb:
                log_data = {
                    "train/loss": loss.item() * grad_accum_steps if 'loss' in locals() else 0,
                    "train/lr": lr,
                    "train/step": iter,
                    "perf/iteration_time_ms": dt if 'dt' in locals() else 0,
                    "perf/throughput_tokens_per_sec": (B * T * grad_accum_steps * world_size) / (dt / 1000) if 'dt' in locals() else 0,
                }
                
                # Add memory usage
                if torch.cuda.is_available():
                    log_data.update({
                        "memory/allocated_gb": torch.cuda.memory_allocated(device) / (1024**3),
                        "memory/reserved_gb": torch.cuda.memory_reserved(device) / (1024**3),
                        "memory/max_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
                    })
                
                wandb.log(log_data)
        

    destroy_process_group()
    if use_wandb and master_process:
        wandb.finish()

    if TrainingConfig.save_model and master_process and False: # For now lets not save the trash model
        if ddp_flag==1:
            print('inside ddp_flag = 1')
            checkpoint = {'config': ModelConfig, 'model_state': raw_model.state_dict(), 'iter_num':iter, 'last_loss':losses, 'train_losses':loss_stats} 
            torch.save(checkpoint, 'llm_model.pt')
            print("checkpoint saved to llm_model.pt")
        elif ddp_flag == 2 or cp_code == 1:
            print('inside ddp_flag = 2 or cp_code = 1')
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state_dict = model.state_dict()

            checkpoint = {'model_config': ModelConfig, 'train_config': TrainingConfig, 'model_state': cpu_state_dict}  # Use the gathered state dict
            torch.save(checkpoint, TrainingConfig.file_name + '_ckpt.pt')
            print("Model checkpoint saved to {}.pt".format(TrainingConfig.file_name + '_ckpt'))

            loss_stats = {'train':train_loss_stats, 'valrun_val':valrun_val_loss_stats, 'valrun_train':valrun_train_loss_stats}
            stats      = {'model_config':ModelConfig, 'train_config':TrainingConfig, 'losses':loss_stats, 'total_params':total, 'active_params':active}
            torch.save(stats, TrainingConfig.file_name+'_stats.pt')
            print("Stats and config saved to {}.pt".format(TrainingConfig.file_name + '_stats'))


    if tp_code == 1:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()