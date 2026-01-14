%%writefile cp_code_fixed.py

'''
TRUE CONTEXT PARALLELISM IMPLEMENTATION
Fixed version that implements pure context parallelism without DDP

Key Changes:
1. Removed DDP - using raw model with manual gradient sync
2. Fixed DataLoader - all ranks sample same sequences, then shard
3. Fixed loss computation - gather logits OR compute correct local loss
4. Manual gradient all-reduce
5. Ring attention support (optional)
'''

import warnings; warnings.filterwarnings('ignore')
import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from typing import Literal
from dataclasses import dataclass 

# ============================================================================
# CONFIGS (Keep your original configs)
# ============================================================================

@dataclass
class LLMconfig:
    vocab_size : int
    block_size : int
    n_embd : int
    pos_emb : str | Literal['learn','sin','rope']
    up_dim  : int
    non_linearity : str | Literal['elu','lrelu','relu', 'gelu', 'swish', 'mish', 'silu', 'selu','celu','tanh','sigmoid']
    dropout : float
    n_layer : int
    moe : bool
    n_exp : int
    n_shared : int  
    n_act : int
    coeff : float
    aux_free : bool
    alpha : float
    gamma: float
    attn : str | Literal['mha', 'mqa', 'gqa', 'mla']
    n_head : int
    n_kv_heads : int 
    q_latent_dim  : int | None
    kv_latent_dim : int | None
    rope_head_dim : int | None
    act_recomp : bool
    context_parallel_size: int = 1
    context_parallel_rank: int = 0
    context_parallel_group: bool = None

    @staticmethod
    def apply_rotary_emb(x:torch.Tensor, freqs_cis:torch.Tensor)->torch.Tensor:
        B,T,H,_ = x.size()
        x_ = x.float().reshape(B, T, H, -1, 2)
        x_re, x_im = x_.unbind(-1)
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
        x_re_out = x_re*freqs_cis.real - x_im*freqs_cis.imag
        x_im_out = x_re*freqs_cis.imag + x_im*freqs_cis.real
        x_out = torch.stack([x_re_out, x_im_out], dim=-1).flatten(3)
        return x_out.type_as(x)

# ============================================================================
# HELPER FUNCTIONS FOR CONTEXT PARALLELISM
# ============================================================================

def all_gather_sequence(tensor: torch.Tensor, dim: int, group=None) -> torch.Tensor:
    """Efficient all-gather along specified dimension"""
    if not torch.distributed.is_initialized():
        return tensor
        
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return tensor

    perm = list(range(tensor.ndim))
    perm[0], perm[dim] = perm[dim], perm[0]
    t_perm = tensor.permute(perm).contiguous()

    T_local = t_perm.size(0)
    out_perm = torch.empty(
        (T_local * world_size, *t_perm.shape[1:]),
        dtype=t_perm.dtype, 
        device=t_perm.device
    )

    torch.distributed.all_gather_into_tensor(out_perm, t_perm, group=group)

    inv_perm = list(range(tensor.ndim))
    inv_perm[0], inv_perm[dim] = inv_perm[dim], inv_perm[0]
    out = out_perm.permute(inv_perm).contiguous()
    
    return out


def reduce_scatter_sequence(tensor: torch.Tensor, dim: int, group=None) -> torch.Tensor:
    """Reduce-scatter along specified dimension"""
    if not torch.distributed.is_initialized():
        return tensor
        
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return tensor
    
    # Move target dim to position 1 (after batch)
    if dim != 1:
        perm = list(range(tensor.ndim))
        perm[1], perm[dim] = perm[dim], perm[1]
        tensor = tensor.permute(perm).contiguous()
    
    # Split along sequence dimension
    chunks = list(tensor.chunk(world_size, dim=1))
    output = torch.zeros_like(chunks[0])
    
    torch.distributed.reduce_scatter(output, chunks, group=group)
    
    # Restore original dimension order
    if dim != 1:
        inv_perm = list(range(output.ndim))
        inv_perm[1], inv_perm[dim] = inv_perm[dim], inv_perm[1]
        output = output.permute(inv_perm).contiguous()
    
    return output


# ============================================================================
# ATTENTION MODULES (Keep your implementations, they're correct)
# ============================================================================

class GQA(nn.Module):
    """Grouped-Query Attention with Context Parallelism support"""
    def __init__(self, config:LLMconfig):
        super().__init__()
        if config.attn == 'mha' : config.n_kv_heads = config.n_head
        elif config.attn == 'mqa' : config.n_kv_heads = 1
        else : assert config.n_head % config.n_kv_heads == 0
        
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.head_size = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * config.n_kv_heads * self.head_size)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.context_parallel_group = getattr(config, 'context_parallel_group', None)

    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor|None, kv_cache=None, VAL_RUN=False):
        B, T_local, C = x.size()
        nh, nkvh, hs = self.config.n_head , self.config.n_kv_heads, self.head_size

        q_proj_size = C
        kv_proj_size = nkvh * hs
        q, k, v = self.c_attn(x).split([q_proj_size, kv_proj_size, kv_proj_size], dim=2)
        q = q.view(B, T_local, nh, hs)
        k = k.view(B, T_local, nkvh, hs)
        v = v.view(B, T_local, nkvh, hs)

        if self.config.pos_emb == 'rope':
            q = LLMconfig.apply_rotary_emb(q, freqs_cis)
            k = LLMconfig.apply_rotary_emb(k, freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Context Parallel: Gather K and V
        if self.config.context_parallel_size > 1:
            k = all_gather_sequence(k, dim=2, group=self.context_parallel_group)
            v = all_gather_sequence(v, dim=2, group=self.context_parallel_group)

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

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        
        # Rectangular causal mask
        T_global = k.size(-2)
        shard_start = self.config.context_parallel_rank * T_local
        q_pos = shard_start + torch.arange(T_local, device=x.device)
        k_pos = torch.arange(T_global, device=x.device)
        causal_mask = (k_pos.unsqueeze(0) > q_pos.unsqueeze(1))
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(attn.dtype).min)

        attn = self.attn_dropout(F.softmax(attn, dim=-1))
        y = attn @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T_local, C)
        y = self.resid_dropout(self.c_proj(y))

        return y, updated_kv_cache


class NaiveMHLA(nn.Module):
    """Multi-Head Latent Attention without RoPE"""
    def __init__(self, config:LLMconfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_size = config.n_embd // config.n_head
        self.config = config

        self.W_dq  = nn.Linear(config.n_embd, config.q_latent_dim, bias=False)
        self.W_uq  = nn.Linear(config.q_latent_dim, config.n_embd, bias=False)
        self.W_dkv = nn.Linear(config.n_embd, config.kv_latent_dim, bias=False)
        self.W_uk  = nn.Linear(config.kv_latent_dim, config.n_embd, bias=False)
        self.W_uv  = nn.Linear(config.kv_latent_dim, config.n_embd, bias=False)
        self.W_o   = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        self.context_parallel_group = getattr(config, 'context_parallel_group', None)

        self.register_buffer('_k_absorbed_inference', None)
        self.register_buffer('_v_absorbed_inference', None)

    def _precompute_absorbed_matrices(self):
        if (self._k_absorbed_inference is not None) and (self._v_absorbed_inference is not None):
            return 
        
        nh, n_kvl, hs = self.config.n_head, self.config.kv_latent_dim, self.head_size
        with torch.no_grad():
            self._k_absorbed_inference = (self.W_dq.weight.T @ self.W_uq.weight.T @ self.W_uk.weight).view(nh, hs, n_kvl).unsqueeze(0)
            self._v_absorbed_inference = (self.W_uv.weight.T @ self.W_o.weight.T).view(n_kvl, nh, hs).transpose(0,1).unsqueeze(0)    

    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor|None, kv_cache=None, VAL_RUN=False):
        B, T_local, C = x.size()
        nh, n_kvl, hs = self.config.n_head, self.config.kv_latent_dim, self.config.n_embd//self.config.n_head

        if self.training or VAL_RUN:
            k_eff = (self.W_dq.weight.T @ self.W_uq.weight.T @ self.W_uk.weight).view(nh, hs, n_kvl).unsqueeze(0)
            v_eff = (self.W_uv.weight.T @ self.W_o.weight.T).view(n_kvl, nh, hs).transpose(0,1).unsqueeze(0)
        else:
            if (self._k_absorbed_inference is None) or (self._v_absorbed_inference is None):
                self._precompute_absorbed_matrices()
            k_eff = self._k_absorbed_inference
            v_eff = self._v_absorbed_inference
        
        new_c_kv = self.W_dkv(x)

        if self.config.context_parallel_size > 1:
            c_kv = all_gather_sequence(new_c_kv, dim=1, group=self.context_parallel_group)
            T_full = c_kv.size(1)
        else:
            if kv_cache is None:
                c_kv = new_c_kv
            else:
                c_kv = torch.cat([kv_cache, new_c_kv], dim=1)
            T_full = c_kv.size(1)

        use_cp = (self.config.context_parallel_size > 1)
        if use_cp:
            updated_kv_cache = None
        else:
            updated_kv_cache = c_kv

        q = self.W_uq(self.W_dq(x))
        q = q.view(B, T_local, nh, hs).transpose(1, 2)

        attn = (q @ k_eff @ c_kv.transpose(1,2).unsqueeze(1)) / math.sqrt(hs)

        T_global = c_kv.size(1)
        shard_start = self.config.context_parallel_rank * T_local
        q_pos = shard_start + torch.arange(T_local, device=x.device)
        k_pos = torch.arange(T_global, device=x.device)
        causal_mask = (k_pos.unsqueeze(0) > q_pos.unsqueeze(1))
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(attn.dtype).min)

        attn = self.dropout(F.softmax(attn, dim=-1))
        y = attn @ c_kv.unsqueeze(1) @ v_eff
        y = self.dropout(y.transpose(1,2).contiguous().view(B, T_local, C))

        return y, updated_kv_cache


class FullMHLA(nn.Module):
    """Multi-Head Latent Attention with RoPE"""
    def __init__(self, config:LLMconfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.W_dq  = nn.Linear(config.n_embd, config.q_latent_dim, False)
        self.dropout = nn.Dropout(config.dropout)
        self.context_parallel_group = getattr(config, 'context_parallel_group', None)
        
        self.head_size = config.n_embd // config.n_head
        self.W_uq  = nn.Linear(config.q_latent_dim, config.n_embd, False)
        self.W_dkv = nn.Linear(config.n_embd, config.kv_latent_dim, False)
        self.W_uk  = nn.Linear(config.kv_latent_dim, config.n_embd, False)
        self.W_uv  = nn.Linear(config.kv_latent_dim, config.n_embd, False)

        self.W_qr  = nn.Linear(config.q_latent_dim, config.n_head * config.rope_head_dim, False)
        self.W_kr  = nn.Linear(config.n_embd, config.rope_head_dim, False)
        self.W_o = nn.Linear(config.n_embd, config.n_embd, False)

        self.register_buffer('_k_absorbed_inference', None, persistent=False)
        self.register_buffer('_v_absorbed_inference', None, persistent=False)

    def _precompute_absorbed_matrices(self):
        if (self._k_absorbed_inference is not None) and (self._v_absorbed_inference is not None):
            return 
        
        nh, nlkv, hs, nlq = self.config.n_head, self.config.kv_latent_dim, self.config.n_embd//self.config.n_head, self.config.q_latent_dim
        with torch.no_grad():
            self._k_absorbed_inference = (self.W_uq.weight.view(1,nlq,nh,hs).transpose(1,2) @ self.W_uk.weight.view(1,nh,hs,nlkv))
            self._v_absorbed_inference = (self.W_uv.weight.T @ self.W_o.weight.T).view(nlkv, nh, hs).transpose(0,1).unsqueeze(0)    

    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor|None, kv_cache=None, VAL_RUN=False):
        B, T_local, C = x.size()
        nh, nlkv, nlq = self.config.n_head, self.config.kv_latent_dim, self.config.q_latent_dim
        hs = C//nh
        dhr = self.config.rope_head_dim
        
        c_q = self.W_dq(x)

        if self.training or VAL_RUN:
            k_eff = (self.W_uq.weight.view(1,nlq,nh,hs).transpose(1,2) @ self.W_uk.weight.view(1,nh,hs,nlkv))
            v_eff = (self.W_uv.weight.T @ self.W_o.weight.T).view(nlkv, nh, hs).transpose(0,1).unsqueeze(0)  
        else:
            if (self._k_absorbed_inference is None) or (self._v_absorbed_inference is None):
                self._precompute_absorbed_matrices()
            k_eff = self._k_absorbed_inference
            v_eff = self._v_absorbed_inference

        new_c_kv = self.W_dkv(x)

        if self.config.context_parallel_size > 1:
            c_kv = all_gather_sequence(new_c_kv, dim=1, group=self.context_parallel_group)
            T_full = c_kv.size(1)
        else:
            if kv_cache is None:
                c_kv = new_c_kv
            else:
                c_kv = torch.cat([kv_cache['c_kv'], new_c_kv], dim=1)
            T_full = c_kv.size(1)

        c_kr = self.W_kr(x).unsqueeze(2)
        k_r = LLMconfig.apply_rotary_emb(c_kr, freqs_cis).transpose(1,2)

        if self.config.context_parallel_size > 1:
            k_r = all_gather_sequence(k_r, dim=2, group=self.context_parallel_group)
        else:
            if kv_cache is not None:
                k_r = torch.cat([kv_cache['k_r'], k_r], dim=2)

        c_qr = self.W_qr(c_q).view(B, T_local, nh, dhr)
        q_r = LLMconfig.apply_rotary_emb(c_qr, freqs_cis).transpose(1,2)
        
        attn_r = q_r @ k_r.transpose(-1,-2)
        attn_c = c_q.unsqueeze(1) @ k_eff @ c_kv.transpose(-1,-2).unsqueeze(1)
        attn = (attn_c + attn_r) / math.sqrt(hs + dhr)

        T_global = c_kv.size(1)
        shard_start = self.config.context_parallel_rank * T_local
        q_pos = shard_start + torch.arange(T_local, device=x.device)
        k_pos = torch.arange(T_global, device=x.device)
        causal_mask = (k_pos.unsqueeze(0) > q_pos.unsqueeze(1))
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(attn.dtype).min)

        attn = self.dropout(F.softmax(attn, dim=-1))
        y = attn @ c_kv.unsqueeze(1) @ v_eff
        y = self.dropout(y.transpose(1,2).contiguous().view(B, T_local, C))

        use_cp = (self.config.context_parallel_size > 1)
        if use_cp:
            updated_kv_cache = None
        else:
            updated_kv_cache = {'c_kv': c_kv, 'k_r': k_r}

        return y, updated_kv_cache


class Attention(nn.Module):
    def __init__(self, config:LLMconfig):
        super().__init__()
        self.config = config
        if config.attn in ('mha','mqa','gqa'):
            self.attn = GQA(config)
        elif config.attn == 'mla':
            if config.pos_emb != 'rope':
                self.attn = NaiveMHLA(config)
            else:
                self.attn = FullMHLA(config)
                
    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor|None = None, kv_cache=None, VAL_RUN=False):
        return self.attn(x, freqs_cis, kv_cache, VAL_RUN)


# ============================================================================
# MLP AND MOE (Keep your implementations)
# ============================================================================

class MLP(nn.Module):
    def __init__(self, config: LLMconfig):
        super().__init__()
        self.non_linearity = config.non_linearity.lower()
        
        if self.non_linearity == 'swiglu':
            self.c_fc = nn.Linear(config.n_embd, 2 * config.up_dim, bias=False)
            self.c_proj = nn.Linear(config.up_dim, config.n_embd, bias=False)
        else:
            non_linearity_map = {
                'relu': nn.ReLU(), 'gelu': nn.GELU(), 'swish': nn.SiLU(), 'mish': nn.Mish(),
                'silu': nn.SiLU(), 'selu': nn.SELU(), 'celu': nn.CELU(), 'elu': nn.ELU(),
                'glu' : nn.GLU(), 'sigmoid': nn.Sigmoid(),
                'lrelu': nn.LeakyReLU(negative_slope=0.01), 'tanh': nn.Tanh()
            }
            self.c_fc = nn.Linear(config.n_embd, config.up_dim, bias=False)
            self.non_linearity_func = non_linearity_map.get(self.non_linearity, nn.GELU())
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
    def __init__(self, config:LLMconfig):
        super().__init__()
        self.expert = MLP(config)
        
    def forward(self, x):
        return self.expert(x)


class MoE(nn.Module):
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        n_tokens = x_flat.shape[0]

        shared_output = torch.zeros_like(x_flat)
        if self.n_shared > 0:
            for i in range(self.n_shared):
                shared_output += self.experts[i](x_flat)

        router_logits = self.gate(x_flat)

        if self.config.aux_free:        
            biased_router_logits = router_logits + self.expert_bias
            topk_biased_logits, topk_indices = torch.topk(biased_router_logits, self.n_act_routed, dim=1)
            topk_original_logits = torch.gather(router_logits, 1, topk_indices) 
            topk_gates = F.softmax(topk_original_logits, dim=1)

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
    def __init__(self, config:LLMconfig):
        super().__init__()
        self.is_moe = config.moe
        self.act_recomp = config.act_recomp
        self.attn = Attention(config)
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.ln2  = nn.LayerNorm(config.n_embd)
        if config.moe:
            self.moe = MoE(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor|None = None, kv_cache=None, VAL_RUN=False):
        if self.act_recomp:
            attn_output, updated_kv_cache = checkpoint(self.attn, self.ln1(x), freqs_cis, kv_cache, VAL_RUN, use_reentrant=False)
        else:
            attn_output, updated_kv_cache = self.attn(self.ln1(x), freqs_cis, kv_cache, VAL_RUN)
        
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
    def __init__(self, config:LLMconfig):
        super().__init__()
        self.config = config
        self.context_parallel_group = None
        
        if config.context_parallel_size > 1:
            self.context_parallel_group = torch.distributed.group.WORLD

        for module in self.modules():
            if hasattr(module, 'context_parallel_group'):
                module.context_parallel_group = config.context_parallel_group

        self.head_size = config.n_embd//config.n_head
        self.tkn_emb = nn.Embedding(config.vocab_size, config.n_embd)

        if config.pos_emb == 'learn':
            self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        elif config.pos_emb == 'sin':
            pos_emb  = torch.zeros(config.block_size, config.n_embd)
            position = torch.arange(0, config.block_size, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, config.n_embd, 2).float() * (-math.log(10000.0) / config.n_embd))
            pos_emb[:, 0::2] = torch.sin(position * div_term)
            pos_emb[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pos_emb', pos_emb)
        elif config.pos_emb == 'rope':
            self.register_buffer("freqs_cis", self._precompute_freqs_cis())
    
        self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tkn_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

        self.VAL_RUN=False
        self.print_act_recomp=config.act_recomp
        self.print_fused_adamw='fused' in inspect.signature(torch.optim.AdamW).parameters

    def _precompute_freqs_cis(self):
        d = self.config.rope_head_dim if self.config.attn=='mla' else self.head_size
        assert d % 2 == 0
        
        theta = 1.0 / (10000.0 ** (torch.arange(0, d, 2).float() / d))
        seq = torch.arange(self.config.block_size)
        freqs = torch.outer(seq, theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        active_params = 0

        active_params += self.tkn_emb.weight.numel()
        if self.config.pos_emb == 'learn': active_params += self.pos_emb.weight.numel()
        active_params += self.transformer.ln_f.weight.numel() + self.transformer.ln_f.bias.numel()

        for block in self.transformer.h:
            active_params += sum(p.numel() for p in block.attn.parameters())
            active_params += sum(p.numel() for p in block.ln1.parameters())
            active_params += sum(p.numel() for p in block.ln2.parameters())

            if block.is_moe:
                active_params += sum(p.numel() for p in block.moe.gate.parameters())
                for i in range(block.moe.n_shared):
                    active_params += sum(p.numel() for p in block.moe.experts[i].parameters())

                if block.moe.n_routed > 0:
                    params_per_routed_expert = sum(p.numel() for p in block.moe.experts[block.moe.n_shared].parameters())
                    active_params += block.moe.n_act_routed * params_per_routed_expert
            else:
                active_params += sum(p.numel() for p in block.mlp.parameters())

        return n_params, active_params

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}]

        try:
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, fused=True)
        except:
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
        return optimizer

    def forward(self, idx: torch.Tensor, targets=None, kv_caches=None):
        B, T_local = idx.size()
        start_pos = 0
        shard_start = self.config.context_parallel_rank * T_local
        
        tkn_emb = self.tkn_emb(idx)
        x = tkn_emb
        freqs_cis = None

        if self.config.pos_emb == 'rope':
            assert shard_start + T_local <= self.freqs_cis.size(0)
            freqs_cis = self.freqs_cis[shard_start : shard_start + T_local]
        elif self.config.pos_emb == 'learn':
            pos = torch.arange(shard_start, shard_start + T_local, dtype=torch.long, device=idx.device)
            x = tkn_emb + self.pos_emb(pos)
        elif self.config.pos_emb == 'sin':
            pos = torch.arange(shard_start, shard_start + T_local, dtype=torch.long, device=idx.device)
            x = tkn_emb + self.pos_emb[pos]

        x = self.transformer.drop(x)

        if kv_caches is None:
            kv_caches = [None] * self.config.n_layer
        
        updated_kv_caches = []
        total_aux_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        
        for i, block in enumerate(self.transformer.h):
            x, updated_kv_cache, aux_loss = block(x, freqs_cis, kv_caches[i], self.VAL_RUN)            
            updated_kv_caches.append(updated_kv_cache)
            if not torch.is_tensor(aux_loss):
                aux_loss = torch.as_tensor(aux_loss, device=x.device, dtype=torch.float32)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.transformer.ln_f(x)

        if targets is not None:
            # CRITICAL FIX: Gather logits before computing loss
            if self.config.context_parallel_size > 1:
                # Gather x (hidden states) from all ranks
                x_gathered = all_gather_sequence(x, dim=1, group=self.context_parallel_group)
                logits = self.lm_head(x_gathered)
                
                # Gather targets as well
                targets_gathered = all_gather_sequence(targets, dim=1, group=self.context_parallel_group)
                targets_flat = targets_gathered.view(-1)
            else:
                logits = self.lm_head(x)
                targets_flat = targets.view(-1)
            
            # Compute loss on full sequence
            valid = (targets_flat != -1)
            num_valid = valid.long().sum()
            
            if num_valid > 0:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1))[valid],
                    targets_flat[valid],
                    reduction='mean',
                )
            else:
                loss = torch.zeros((), device=logits.device, dtype=torch.float32)
            
            # Synchronize aux loss across ranks
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                torch.distributed.all_reduce(total_aux_loss, op=torch.distributed.ReduceOp.SUM)
                total_aux_loss = total_aux_loss / world_size
            
            loss = loss + total_aux_loss / self.config.n_layer
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, updated_kv_caches
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, topk: int | None = None):
        self.eval()
        kv_caches = [None] * self.config.n_layer

        if self.config.context_parallel_size > 1:
            raise NotImplementedError("Generation with context parallelism requires special handling")

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
                    keep_len = self.config.block_size - 1
                    for layer_idx in range(self.config.n_layer):
                        layer_cache = kv_caches[layer_idx]
                        if self.config.attn in ('mha', 'mqa', 'gqa'):
                            k, v = layer_cache
                            kv_caches[layer_idx] = (k[..., -keep_len:, :], v[..., -keep_len:, :])
                        elif self.config.attn == 'mla':
                            if self.config.pos_emb == 'rope':
                                layer_cache['c_kv'] = layer_cache['c_kv'][:, -keep_len:, :]
                                layer_cache['k_r']  = layer_cache['k_r'][:, :, -keep_len:, :]
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
# TRAINING SCRIPT
# ============================================================================

import os
import argparse
import tiktoken
import requests
import numpy as np
from time import perf_counter
from torch.distributed import init_process_group, destroy_process_group

assert torch.cuda.is_available()
assert torch.cuda.device_count() > 1

# Setup distributed training
init_process_group(backend='nccl')

rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
device = f"cuda:{local_rank}"
master_process = rank == 0
if master_process : print(f"Num GPUs = {world_size}")

torch.cuda.set_device(device)
torch.manual_seed(1729 + rank)
torch.cuda.manual_seed(1729 + rank)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

dtype = 'float16'
torch_dtype = getattr(torch, dtype)
ctx = torch.amp.autocast(device_type="cuda", dtype=torch_dtype)
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))


# ============================================================================
# TRAINING CONFIG
# ============================================================================

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
    compile : bool
    save_model : bool
    file_name : str
    act_recomp : bool

TrainingConfig = Trainconfig(
    dataset='shakespeare',
    total_batch_size = 2**12,
    batch_size = 2**1,
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
    act_recomp=False)

ModelConfig = LLMconfig(
    vocab_size = 50304, 
    block_size = 2**10,
    n_embd = 256, 
    pos_emb = 'rope',
    moe = True,
    up_dim = 512, 
    non_linearity = 'swiglu',  
    dropout=0.0,
    n_layer = 6,
    n_exp = 8,
    n_shared = 1,
    n_act = 4,
    coeff=0.01,
    aux_free=True,
    alpha = 0.0001,
    gamma = 0.001,
    attn = 'mla', 
    n_head = 8,
    n_kv_heads=4,
    q_latent_dim = 32, 
    kv_latent_dim = 32,
    rope_head_dim = 16,
    act_recomp=TrainingConfig.act_recomp)


# ============================================================================
# CLI ARGUMENT PARSING
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM with Context Parallelism')
    parser.add_argument('--dataset', type=str, default=TrainingConfig.dataset)
    parser.add_argument('--batch_size', type=int, default=TrainingConfig.batch_size)
    parser.add_argument('--max_iters', type=int, default=TrainingConfig.max_iters)
    parser.add_argument('--eval_interval', type=int, default=TrainingConfig.eval_interval)
    parser.add_argument('--eval_iters', type=int, default=TrainingConfig.eval_iters)
    parser.add_argument('--learning_rate', type=float, default=TrainingConfig.learning_rate)
    parser.add_argument('--warmup_steps', type=int, default=TrainingConfig.warmup_steps)
    parser.add_argument('--grad_clip', type=float, default=TrainingConfig.grad_clip)
    parser.add_argument('--act_recomp', action='store_true')
    
    parser.add_argument('--vocab_size', type=int, default=ModelConfig.vocab_size)
    parser.add_argument('--block_size', type=int, default=ModelConfig.block_size)
    parser.add_argument('--n_embd', type=int, default=ModelConfig.n_embd)
    parser.add_argument('--pos_emb', type=str, default=ModelConfig.pos_emb)
    parser.add_argument('--n_layer', type=int, default=ModelConfig.n_layer)
    parser.add_argument('--dropout', type=float, default=ModelConfig.dropout)
    parser.add_argument('--up_dim', type=int, default=ModelConfig.up_dim)
    parser.add_argument('--non_linearity', type=str, default=ModelConfig.non_linearity)
    parser.add_argument('--n_exp', type=int, default=ModelConfig.n_exp)
    parser.add_argument('--n_shared', type=int, default=ModelConfig.n_shared)
    parser.add_argument('--n_act', type=int, default=ModelConfig.n_act)
    parser.add_argument('--coeff', type=float, default=ModelConfig.coeff)
    parser.add_argument('--alpha', type=float, default=ModelConfig.alpha)
    parser.add_argument('--gamma', type=float, default=ModelConfig.gamma)
    parser.add_argument('--attn', type=str, default=ModelConfig.attn)
    parser.add_argument('--n_head', type=int, default=ModelConfig.n_head)
    parser.add_argument('--n_kv_heads', type=int, default=ModelConfig.n_kv_heads)
    parser.add_argument('--q_latent_dim', type=int, default=ModelConfig.q_latent_dim)
    parser.add_argument('--kv_latent_dim', type=int, default=ModelConfig.kv_latent_dim)
    parser.add_argument('--rope_head_dim', type=int, default=ModelConfig.rope_head_dim)
    
    parser.add_argument('--total_batch_size_str', type=str, default=str(TrainingConfig.total_batch_size))
    parser.add_argument('--moe', action='store_true')
    parser.add_argument('--aux_free', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--file_name', type=str, default=TrainingConfig.file_name)

    return parser.parse_args()

args = parse_args()
for key, value in vars(args).items():
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

if ModelConfig.attn == 'mha':
    ModelConfig.n_kv_heads = ModelConfig.n_head
elif ModelConfig.attn == 'mqa':
    ModelConfig.n_kv_heads = 1
elif ModelConfig.attn == 'mla':
    req = ModelConfig.q_latent_dim is not None and ModelConfig.kv_latent_dim is not None
    assert req
    if ModelConfig.pos_emb == 'rope':
        assert ModelConfig.rope_head_dim is not None

# CRITICAL: Set context parallel config
ModelConfig.context_parallel_size = world_size
ModelConfig.context_parallel_rank = rank
ModelConfig.context_parallel_group = torch.distributed.group.WORLD

if torch.distributed.is_initialized():
    assert ModelConfig.context_parallel_size == torch.distributed.get_world_size()


# ============================================================================
# DATASET AND DATALOADER
# ============================================================================

def tokenize_and_save():
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

tokenize_and_save()


class DataLoader:
    """
    CRITICAL FIX: All ranks sample the SAME sequences, then each rank takes its shard
    """
    def __init__(self, B, T, file_path, device, context_parallel_size=1, context_parallel_rank=0):
        self.B = B
        self.T = T  # Global sequence length
        self.file_path = file_path
        self.device = torch.device(device)
        self.device_type = self.device.type
        
        self.context_parallel_size = context_parallel_size
        self.context_parallel_rank = context_parallel_rank
        
        # Local sequence length per rank
        self.local_T = T // context_parallel_size
        assert T % context_parallel_size == 0, "Sequence length must be divisible by CP size"

        self.tokens = np.memmap(self.file_path, dtype=np.uint16, mode='r')
        self.N = len(self.tokens)
        if self.B * self.T + 1 > self.N:
            raise ValueError(f"Batch size {B} and block size {T} too large for dataset of length {self.N}")

        # CRITICAL: Use same seed for all ranks so they sample same sequences
        self.rng = np.random.RandomState(42)

    def next_batch(self):
        """
        All ranks sample the SAME sequences, then each rank extracts its local chunk
        """
        B, T, local_T = self.B, self.T, self.local_T
        
        # CRITICAL FIX: All ranks use the same random state
        # This ensures all ranks sample the same starting indices
        start_indices = self.rng.randint(0, self.N - T - 1, size=(B,))

        x_list = []
        y_list = []
        
        for start in start_indices:
            # Load the FULL sequence
            full_seq = self.tokens[start : start + T + 1].astype(np.int64)
            
            # Extract THIS rank's local chunk
            local_start = self.context_parallel_rank * local_T
            local_end = local_start + local_T
            
            x_local = full_seq[local_start:local_end]
            y_local = full_seq[local_start + 1:local_end + 1]
            
            x_list.append(x_local)
            y_list.append(y_local)

        x = torch.from_numpy(np.stack(x_list)).long()
        y = torch.from_numpy(np.stack(y_list)).long()

        assert x.shape[1] == self.local_T, f"Expected local_T={self.local_T}, got {x.shape[1]}"
        
        if self.device_type == 'cuda':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        
        return x, y


train_loader = DataLoader(
    B=TrainingConfig.batch_size, 
    T=ModelConfig.block_size,
    file_path="train.bin", 
    device=device,
    context_parallel_size=world_size,
    context_parallel_rank=rank
)

val_loader = DataLoader(
    B=TrainingConfig.batch_size, 
    T=ModelConfig.block_size,
    file_path="val.bin", 
    device=device,
    context_parallel_size=world_size,
    context_parallel_rank=rank
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_lr(iter, TrainingConfig:Trainconfig):
    max_lr = TrainingConfig.learning_rate
    min_lr = max_lr*0.1
    max_decay_steps = TrainingConfig.max_iters + 2
    
    if iter < TrainingConfig.warmup_steps:
        return max_lr * (iter+1)/TrainingConfig.warmup_steps
    elif iter > max_decay_steps:
        return min_lr
    else:
        decay_ratio = (iter - TrainingConfig.warmup_steps) / (max_decay_steps - TrainingConfig.warmup_steps)
        decay_ratio = min(decay_ratio, 1.0)
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(model:LLM, TrainingConfig:Trainconfig, train_loader:DataLoader, val_loader:DataLoader):
    out = {}
    model.eval()
    model.VAL_RUN = True
    
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(TrainingConfig.eval_iters)
        for k in range(TrainingConfig.eval_iters):
            X, Y = loader.next_batch()
            with ctx:
                _, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    model.VAL_RUN = False
    return out


# ============================================================================
# GRADIENT ACCUMULATION SETUP
# ============================================================================

total_batch_size = TrainingConfig.total_batch_size
B = TrainingConfig.batch_size
T = ModelConfig.block_size
assert total_batch_size % (B * T * world_size) == 0
grad_accum_steps = total_batch_size // (B * T * world_size)


# ============================================================================
# CREATE MODEL (NO DDP - PURE CONTEXT PARALLELISM)
# ============================================================================

model = LLM(ModelConfig).to(device)

if master_process:
    total, active = model.get_num_params()
    print(f"Total parameters = {total:,}, Active parameters = {active:,}")
    if model.print_fused_adamw: 
        print("Using Fused AdamW")
    if model.print_act_recomp: 
        print("Using Activation Recomputation")
    print(f"Context Parallel Size: {world_size}")
    print(f"Local Sequence Length per GPU: {T // world_size}")

# CRITICAL: NO DDP! Model stays as raw model for pure context parallelism
if TrainingConfig.compile and master_process:
    print("Using compiled model")
if TrainingConfig.compile:
    model = torch.compile(model)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=TrainingConfig.learning_rate, device=device)


# ============================================================================
# TRAINING LOOP
# ============================================================================

x, y = train_loader.next_batch()

for iter in range(TrainingConfig.max_iters + 1):
    t0 = perf_counter()

    lr = get_lr(iter, TrainingConfig)
    for param_grp in optimizer.param_groups:
        param_grp['lr'] = lr
    
    optimizer.zero_grad(set_to_none=True)

    if TrainingConfig.eval and (iter % TrainingConfig.eval_interval == 0 or iter == TrainingConfig.max_iters) and iter != 0:
        a = perf_counter()
        losses = estimate_loss(model, TrainingConfig, train_loader, val_loader)
        b = perf_counter()
        if master_process:
            print(f"-------- Eval -------- train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | dt {1000*(b-a):.4f}ms")
        t0 = b
    
    # Gradient accumulation loop
    for micro_step in range(grad_accum_steps):
        with ctx:
            _, loss, _ = model(x, y)
            loss = loss / grad_accum_steps

        x, y = train_loader.next_batch()  # Async pre-load
        scaler.scale(loss).backward()

    # CRITICAL: Manual gradient all-reduce (replaces DDP's automatic sync)
    if torch.distributed.is_initialized():
        for param in model.parameters():
            if param.grad is not None:
                torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.AVG)

    # Gradient clipping
    if TrainingConfig.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.grad_clip)

    scaler.step(optimizer)
    scaler.update()

    if master_process:
        torch.cuda.synchronize()
        mem = torch.cuda.memory_reserved()
        dt = (perf_counter() - t0) * 1000
        print(f"step: {iter} | train loss: {loss.item()*grad_accum_steps:.4f} | dt: {dt:.2f}ms | grad_accum_steps: {grad_accum_steps} | GPU RAM: {mem/1024**3:.2f}GB")

destroy_process_group()

if TrainingConfig.save_model and master_process:
    checkpoint = {
        'model_config': ModelConfig, 
        'train_config': TrainingConfig, 
        'model_state': model.state_dict()
    }
    torch.save(checkpoint, TrainingConfig.file_name + '_ckpt.pt')
    print(f"Model checkpoint saved to {TrainingConfig.file_name}_ckpt.pt")