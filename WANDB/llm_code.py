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

from config_code import LLMconfig, merging_code, ddp_flag , tp_code, ep_code, cp_code




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
