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

from datetime import datetime
import wandb
from datetime import datetime
import glob  # <-- MISSING
import gc    # <-- MISSING



from typing import Literal
from dataclasses import dataclass 
from torch.distributed.optim import ZeroRedundancyOptimizer

from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp.api  import ShardingStrategy, CPUOffload

from config_code import LLMconfig, merging_code, ddp_flag , tp_code, ep_code, cp_code, EPLayout, DataLoader





class LLM(nn.Module):
    """ A simple Large language model """
    # def __init__(self, config:LLMconfig , tp_group=None, tp_code):
    def __init__(self, config: LLMconfig, tp_group=None, tp_code=0, merging_code=1, ep_code=0):
    
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

        if self.use_cp or ddp_flag == 2:
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

 

