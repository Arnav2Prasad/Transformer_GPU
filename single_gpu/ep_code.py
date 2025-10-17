%%writefile moe_code.py

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
import os
import argparse
import tiktoken
import requests
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from typing import Literal
from dataclasses import dataclass 

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

    def __init__(self, config:LLMconfig):
        super().__init__()
        if config.attn == 'mha' : config.n_kv_heads = config.n_head
        elif config.attn == 'mqa' : config.n_kv_heads = 1
        else : assert config.n_head % config.n_kv_heads == 0, "n_head must be divisible by n_kv_heads"
        
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.config = config
        self.head_size = config.n_embd // config.n_head

        # k,q,v in a btach
        self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * config.n_kv_heads * self.head_size)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor|None, kv_cache=None, VAL_RUN=False):
        B, T, C = x.size()
        nh, nkvh, hs = self.config.n_head , self.config.n_kv_heads, self.head_size

        q_proj_size = C # n_embd
        kv_proj_size = nkvh * hs
        q, k, v = self.c_attn(x).split([q_proj_size, kv_proj_size, kv_proj_size], dim=2)
        q:torch.Tensor = q.view(B, T, nh, hs) # (B, T, nh, hs)
        k:torch.Tensor = k.view(B, T, nkvh, hs) # (B, T, n_kvh, hs)
        v:torch.Tensor = v.view(B, T, nkvh, hs).transpose(1, 2) # (B, n_kvh, T, hs)

        if self.config.pos_emb == 'rope':
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

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.config.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y, updated_kv_cache

class NaiveMHLA(nn.Module):
    """ A fully parallel implementation of the MHLA algorithm without the RoPE. No for loops."""
    def __init__(self, config:LLMconfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "num of heads must be a divisor of n_embd"
        self.head_size = config.n_embd // config.n_head
        self.config = config

        self.W_dq  = nn.Linear(config.n_embd,        config.q_latent_dim,  bias=False)
        self.W_uq  = nn.Linear(config.q_latent_dim,  config.n_embd,        bias=False)
        self.W_dkv = nn.Linear(config.n_embd,        config.kv_latent_dim, bias=False)
        self.W_uk  = nn.Linear(config.kv_latent_dim, config.n_embd,        bias=False)
        self.W_uv  = nn.Linear(config.kv_latent_dim, config.n_embd,        bias=False)
        self.W_o   = nn.Linear(config.n_embd,        config.n_embd,        bias=False)
        
        # self.ln  = nn.LayerNorm(config.kv_latent_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer('_k_absorbed_inference', None)
        self.register_buffer('_v_absorbed_inference', None)

    def _precompute_absorbed_matrices(self):
        """Precomputes k_absorbed and v_absorbed for efficient inference."""
        # Just to be safe
        if (self._k_absorbed_inference is not None) and (self._v_absorbed_inference is not None):
            return 
        
        nh , n_kvl, hs = self.config.n_head, self.config.kv_latent_dim, self.head_size
        with torch.no_grad():
            self._k_absorbed_inference = (self.W_dq.weight.T @ self.W_uq.weight.T  @ self.W_uk.weight).view(nh, hs, n_kvl).unsqueeze(0)
            self._v_absorbed_inference = (self.W_uv.weight.T @ self.W_o.weight.T).view(n_kvl, nh, hs).transpose(0,1).unsqueeze(0)    

    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor|None, kv_cache=None, VAL_RUN=False) -> tuple[torch.Tensor, torch.Tensor]:

        B, T, C = x.size()
        nh, n_kvl, hs = self.config.n_head, self.config.kv_latent_dim, self.config.n_embd//self.config.n_head

        # k_eff and v_eff based on training or inference
        if self.training or VAL_RUN: # HIDDEN IN PLAIN SIGHT : THIS BUG TOOK ~16 HRS TO DEBUG
            k_eff = (self.W_dq.weight.T @ self.W_uq.weight.T  @ self.W_uk.weight).view(nh, hs, n_kvl).unsqueeze(0)
            v_eff = (self.W_uv.weight.T @ self.W_o.weight.T).view(n_kvl, nh, hs).transpose(0,1).unsqueeze(0)
        else:
            if (self._k_absorbed_inference is None) or (self._v_absorbed_inference is None):
                self._precompute_absorbed_matrices()
            k_eff = self._k_absorbed_inference
            v_eff = self._v_absorbed_inference
        
        new_c_kv = self.W_dkv(x) # down projection : (B,T,C) -> (B,T,n_kvl)

        if kv_cache is None:
            c_kv = new_c_kv # (B,T,n_kvl) ; initiate cache
        else:
            c_kv = torch.cat([kv_cache, new_c_kv], dim=1) # append cache
        
        updated_kv_cache = c_kv

        T_full = c_kv.size(1) # Current total sequence length (including cache)

        q:torch.Tensor = self.W_uq(self.W_dq(x)) # query projection : (B,T,C) -> (B,T,n_ql) -> (B,T,C)
        q = q.view(B, T, nh, hs).transpose(1, 2) # (B,T,C) -> (B,T,nh,hs) -> (B, nh, T, hs)

        attn:torch.Tensor = (q @ k_eff @ c_kv.transpose(1,2).unsqueeze(1)) / math.sqrt(hs)

        # query_indices = torch.arange(T, device=x.device).unsqueeze(1) + (T_full - T)
        # key_indices   = torch.arange(T_full, device=x.device).unsqueeze(0)
        # mask = (query_indices >= key_indices).unsqueeze(0).unsqueeze(0) # (1,1,T,T_full)
        # attn = attn.masked_fill(mask == 0, float('-inf'))
        
        mask = torch.triu(torch.ones(T, T_full, device=x.device, dtype=torch.bool), diagonal=T_full - T + 1)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = self.dropout(F.softmax(attn, dim=-1)) # (B, nh, T, T_full)

        # final output : attn @ C_kv @ v_abs 
        # (B, nh, T, T) * (B, 1, T, n_kvl) * (1, nh, n_kvl, hs) = (B, nh, T, hs)
        y:torch.Tensor = attn @ c_kv.unsqueeze(1) @ v_eff #(B, nh, T, hs)
        y = self.dropout(y.transpose(1,2).contiguous().view(B,T,C))

        return y, updated_kv_cache

class FullMHLA(nn.Module):
    """
    A fully parallel implementation of Multi-Head Latent Attention (MLA)
    with Decoupled Rotary Position Embeddings (RoPE), as described in DeepSeek-V2.
    """
     
    def __init__(self, config:LLMconfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "num of heads must be a divisor of n_embd"
        self.config = config
        self.W_dq  = nn.Linear(config.n_embd, config.q_latent_dim , False)
        self.dropout = nn.Dropout(config.dropout)
        
        # (NoPE)
        self.head_size = config.n_embd // config.n_head
        self.W_uq  = nn.Linear(config.q_latent_dim , config.n_embd, False)
        self.W_dkv = nn.Linear(config.n_embd, config.kv_latent_dim, False)
        self.W_uk  = nn.Linear(config.kv_latent_dim, config.n_embd, False)
        self.W_uv  = nn.Linear(config.kv_latent_dim, config.n_embd, False)

        # (RoPE)
        self.W_qr  = nn.Linear(config.q_latent_dim, config.n_head * config.rope_head_dim,  False)
        self.W_kr  = nn.Linear(config.n_embd, config.rope_head_dim, False)

        # (Out)
        self.W_o = nn.Linear(config.n_embd, config.n_embd ,False)

        # Absroption during inference
        self.register_buffer('_k_absorbed_inference', None, persistent=False)
        self.register_buffer('_v_absorbed_inference', None, persistent=False)

    def _precompute_absorbed_matrices(self):
        """Precomputes k_absorbed and v_absorbed for efficient inference."""
        # Just to be safe
        if (self._k_absorbed_inference is not None) and (self._v_absorbed_inference is not None):
            return 
        
        nh, nlkv, hs, nlq = self.config.n_head, self.config.kv_latent_dim, self.config.n_embd//self.config.n_head, self.config.q_latent_dim
        with torch.no_grad():
            self._k_absorbed_inference = (self.W_uq.weight.view(1,nlq,nh,hs).transpose(1,2) @ self.W_uk.weight.view(1,nh,hs,nlkv))
            self._v_absorbed_inference = (self.W_uv.weight.T @ self.W_o.weight.T).view(nlkv, nh, hs).transpose(0,1).unsqueeze(0)    

    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor|None, kv_cache=None, VAL_RUN=False):
        B,T,C = x.size()
        nh,nlkv,nlq = self.config.n_head, self.config.kv_latent_dim, self.config.q_latent_dim
        hs = C//nh
        dhr = self.config.rope_head_dim
        
        c_q:torch.Tensor = self.W_dq(x)  # (B,T,nlq)

 #------------ NoPE--------------

        # Define the absorbed matrices
        if self.training or VAL_RUN:  # HIDDEN IN PLAIN SIGHT : THIS BUG TOOK ~16 HRS TO DEBUG
            k_eff = (self.W_uq.weight.view(1,nlq,nh,hs).transpose(1,2) @ self.W_uk.weight.view(1,nh,hs,nlkv))
            v_eff = (self.W_uv.weight.T @ self.W_o.weight.T).view(nlkv, nh, hs).transpose(0,1).unsqueeze(0)  
        else:
            if (self._k_absorbed_inference is None) or (self._v_absorbed_inference is None):
                self._precompute_absorbed_matrices()
            k_eff = self._k_absorbed_inference
            v_eff = self._v_absorbed_inference

        new_c_kv = self.W_dkv(x) # down projection : (B,T,C) -> (B,T,n_kvl)

        if kv_cache is None: # first pass
            c_kv = new_c_kv # (B,T,n_kvl) ; initiate cache
        else:
            c_kv = torch.cat([kv_cache['c_kv'], new_c_kv], dim=1) # append cache

        T_full = c_kv.size(1) # Current total sequence length (including cache)

        attn_c = c_q.unsqueeze(1) @ k_eff @ c_kv.transpose(-1,-2).unsqueeze(1)

 #------------ RoPE--------------

        c_kr:torch.Tensor = self.W_kr(x).unsqueeze(2)        # (B,T,1,dhr)
        k_r = LLMconfig.apply_rotary_emb(c_kr, freqs_cis).transpose(1,2)  # (B,1,T,dhr), to be cached

        # initate KV cache
        if kv_cache is not None:
            k_r = torch.cat([kv_cache['k_r'], k_r], dim=2)

        c_qr:torch.Tensor = self.W_qr(c_q).view(B,T,nh,dhr) # (B,T,nh,dhr) # because rope expects (B,T,H,dh)
        q_r = LLMconfig.apply_rotary_emb(c_qr, freqs_cis).transpose(1,2) # (B,nh,T,dhr)
        
        attn_r = q_r @ k_r.transpose(-1,-2)

 #------------ Out--------------

        attn = (attn_c + attn_r)/math.sqrt(hs+dhr)

        # query_indices = torch.arange(T, device=x.device).unsqueeze(1) + (T_full - T)
        # key_indices = torch.arange(T_full, device=x.device).unsqueeze(0)
        # mask = (query_indices >= key_indices).unsqueeze(0).unsqueeze(0) # (1,1,T,T_full)
        # attn = attn.masked_fill(mask == 0, float('-inf')) 

        mask = torch.triu(torch.ones(T, T_full, device=x.device, dtype=torch.bool), diagonal=T_full - T + 1)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = self.dropout(F.softmax(attn, dim=-1)) # (B, nh, T, T_full)

        # final output : attn @ C_kv @ v_abs 
        # (B, nh, T, T) * (B, 1, T, n_kvl) * (1, nh, n_kvl, hs) = (B, nh, T, hs)
        y:torch.Tensor = attn @ c_kv.unsqueeze(1) @ v_eff #(B, nh, T, hs)
        y = self.dropout(y.transpose(1,2).contiguous().view(B,T,C))

        updated_kv_cache = {'c_kv': c_kv, 'k_r': k_r}

        return y, updated_kv_cache

class Attention(nn.Module):
    """ Routes the attention mechanism according to the config"""

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

class MLP(nn.Module):
    """ A simple feed-forward network block. """
    def __init__(self, config: LLMconfig):
        super().__init__()
        self.non_linearity = config.non_linearity.lower()
        
        if self.non_linearity == 'swiglu':
            # One projection, then split into two halves
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
    """ A single feed-forward network expert. """
    def __init__(self, config:LLMconfig):
        super().__init__()
        self.expert = MLP(config)
        
    def forward(self, x):
        return self.expert(x)

'''
class MoE(nn.Module):
    
    This class implements the DeepSeekMoE layer, featuring shared and routed experts.
    It uses an Auxiliary-Loss-Free load balancing strategy with a dynamic bias term.
    Ref: https://arxiv.org/pdf/2412.19437
    

    def __init__(self, config: LLMconfig):
        super().__init__()
        self.config = config
        
        # first `n_shared` are shared, the rest are routed
        self.n_shared = config.n_shared
        self.n_routed = config.n_exp - config.n_shared
        
        # Number of experts to activate from the ROUTED pool
        self.n_act_routed = config.n_act - config.n_shared
        assert self.n_act_routed > 0, "Number of active experts must be greater than shared experts"

        self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_exp)])
        self.gate = nn.Linear(config.n_embd, self.n_routed, bias=False)
        
        if config.aux_free:
            self.register_buffer('expert_bias', torch.zeros(self.n_routed))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

'''

from math import ceil
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

class EPLayout:
    """Manages expert distribution across EP ranks"""
    def __init__(self, n_routed, world_size, rank):
        self.n_routed = n_routed
        self.world_size = world_size
        self.rank = rank
        self.n_local = ceil(n_routed / world_size)
        self.start = self.n_local * rank
        self.end = min(self.start + self.n_local, n_routed)
        self.local_global_ids = list(range(self.start, self.end))

    def owner_rank(self, gid: int) -> int:
        return min(gid // self.n_local, self.world_size - 1)

    def local_index(self, gid: int) -> int:
        return gid - self.start

class MoE(nn.Module):
    '''
    Pure Expert Parallelism implementation with correct all-to-all token routing.
    NOTE: This implementation uses top-1 routing. For top-k, flatten (token, slot)
    into a single dimension and dispatch once, then sum slot contributions after gather.
    '''
    def __init__(self, config: LLMconfig):
        super().__init__()
        self.config = config
        
        # Initialize EP attributes if they don't exist
        if not hasattr(config, 'ep_rank'):
            config.ep_rank = 0
        if not hasattr(config, 'ep_size'):
            config.ep_size = 1
        if not hasattr(config, 'ep_group'):
            config.ep_group = None
            
        self.rank = config.ep_rank
        self.world_size = config.ep_size
        self.ep_group = config.ep_group
        
        self.n_shared = config.n_shared
        self.n_routed = config.n_exp - config.n_shared
        self.n_act_routed = config.n_act - config.n_shared
        
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

    def _forward_single_gpu(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fallback for single GPU (no EP)"""
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        n_tokens = x_flat.shape[0]
        
        # Shared experts
        shared_out = torch.zeros_like(x_flat)
        if self.n_shared > 0:
            for expert in self.shared_experts:
                shared_out += expert(x_flat)
        
        # Routing with fp32 for numerical stability
        router_logits = self.gate(x_flat).float()  # Compute in fp32
        
        if self.config.aux_free:
            biased_logits = router_logits + self.expert_bias
            topk_logits, topk_indices = torch.topk(biased_logits, self.n_act_routed, dim=1)
            topk_original = torch.gather(router_logits, 1, topk_indices)
            topk_gates = F.softmax(topk_original, dim=1).to(x_flat.dtype)  # Convert back to model dtype
            
            # Aux loss calculation
            with torch.no_grad():
                ones = torch.ones_like(topk_indices, dtype=torch.float)
                fi_counts = torch.zeros(self.n_routed, device=x.device).scatter_add_(0, topk_indices.flatten(), ones.flatten())
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
                fi_counts = torch.zeros(self.n_routed, device=x.device).scatter_add_(0, topk_indices.flatten(), ones.flatten())
                fi = fi_counts / n_tokens
            
            pi = router_probs.mean(dim=0)
            aux_loss = self.config.coeff * self.n_routed * torch.sum(pi * fi)
        
        # Process all experts locally
        routed_output = torch.zeros_like(x_flat)
        for i in range(self.n_routed):
            token_indices, topk_slot = (topk_indices == i).nonzero(as_tuple=True)
            if token_indices.numel() > 0:
                tokens_for_expert = x_flat[token_indices]
                gates_for_expert = topk_gates[token_indices, topk_slot].unsqueeze(1)
                expert_output = self.local_routed_experts[i](tokens_for_expert)
                weighted_output = expert_output * gates_for_expert
                routed_output.index_add_(0, token_indices, weighted_output)
        
        # Combine outputs
        y = (shared_out + routed_output).view(B, T, C)
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



class Block(nn.Module):
    """ A single Transformer block combining attention and MLP. """
    def __init__(self, config:LLMconfig):
        super().__init__()
        self.is_moe = config.moe
        self.attn = Attention(config)
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.ln2  = nn.LayerNorm(config.n_embd)

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
            self.mlp = MLP(config)

    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor|None = None, kv_cache=None, VAL_RUN=False):
        # Layer Norm + Attention
        attn_output, updated_kv_cache = self.attn.forward(self.ln1(x), freqs_cis, kv_cache, VAL_RUN)
        x = x + attn_output

        if self.is_moe: 
            moe_output, aux_loss = self.moe(self.ln2(x))
            x = x + moe_output
        else:
            aux_loss = 0.0
            x = x + self.mlp(self.ln2(x))

        return x, updated_kv_cache, aux_loss

class LLM(nn.Module):
    """ A simple Large language model """
    def __init__(self, config: LLMconfig):
        super().__init__()
        self.config = config
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
        self.tkn_emb.weight = self.lm_head.weight # weight tying
        self.apply(self._init_weights)

        self.VAL_RUN=False
        if config.act_recomp: print("Using Activation Recomputation")

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

            if hasattr(block, 'is_moe') and block.is_moe:
                # MoE block - count gate and experts
                active_params += sum(p.numel() for p in block.moe.gate.parameters()) if block.moe.gate is not None else 0
                
                # Count shared experts (always active)
                for i in range(len(block.moe.shared_experts)):
                    active_params += sum(p.numel() for p in block.moe.shared_experts[i].parameters())
                
                # Count routed experts (only active ones)
                if hasattr(block.moe, 'n_act_routed') and block.moe.n_act_routed > 0:
                    # Calculate params for one routed expert, multiply by the number of active ones
                    if len(block.moe.local_routed_experts) > 0:
                        params_per_routed_expert = sum(p.numel() for p in block.moe.local_routed_experts[0].parameters())
                        active_params += block.moe.n_act_routed * params_per_routed_expert
            else:
                # Regular MLP block
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

        # Create AdamW optimizer and use the fused version if it is available
        try:
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, fused=True)
            print("Using Fused AdamW")
        except:
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
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
        
        elif self.config.pos_emb == 'learn':
            pos = torch.arange(start_pos, start_pos + T, dtype=torch.long, device=idx.device)
            x = tkn_emb + self.pos_emb(pos)

        elif self.config.pos_emb == 'sin':
            pos = torch.arange(start_pos, start_pos + T, dtype=torch.long, device=idx.device)
            x = tkn_emb + self.pos_emb[pos]

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







# TRAINING SCRIPT

import warnings ; warnings.filterwarnings("ignore")
import os
import math
import torch
import argparse
import numpy as np

from pathlib import Path
from typing import Literal
from time import perf_counter
from dataclasses import dataclass
from contextlib import nullcontext


# ______________DEVICE and DTYPE SETUP_________________
torch.manual_seed(1729)
torch.cuda.manual_seed(1729)
torch.set_float32_matmul_precision('medium')   # Not sure if this has any effect when used with Auto Mixed Precision

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ctx = torch.amp.autocast(device_type=device_type, dtype=getattr(torch, dtype))
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16')) if device == 'cuda' else nullcontext()

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
    dataset='tinystories',
    total_batch_size = 2**11,
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

    up_dim = 384, 
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
    # ADD EP CONFIG
    ep_size=1,  # Will be updated in training script
    ep_rank=0,  # Will be updated in training script
    ep_group=None,  # Will be updated in training script  
    
    act_recomp=TrainingConfig.act_recomp # Link the activation recomputation from the TRaining params     
)    

        

# ___________ CLI-OVERRIDE__________________

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

    return parser.parse_args()

args = parse_args()
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
if ModelConfig.attn == 'mha':
    ModelConfig.n_kv_heads = ModelConfig.n_head
elif ModelConfig.attn == 'mqa':
    ModelConfig.n_kv_heads = 1
elif ModelConfig.attn == 'mla':
    req = ModelConfig.q_latent_dim is not None and ModelConfig.kv_latent_dim is not None
    assert req, "Either q_latent_dim or kv_latent_dim is missing"
    if ModelConfig.pos_emb == 'rope':
        assert ModelConfig.rope_head_dim is not None, "Need dim of Rotary heads"

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


class DataLoader:
    def __init__(self, B, T, file_path, device):
        self.B = B
        self.T = T
        self.file_path = file_path
        self.device = device
        self.device_type = 'cuda' if 'cuda' in device else 'cpu'

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

        # Sample B random starting positions independently
        start_indices = torch.randint(0, self.N - T - 1, (B,))

        # Gather sequences
        x_list = []
        y_list = []
        for start in start_indices:
            seq = self.tokens[start : start + T + 1].astype(np.int64)
            x_list.append(seq[:-1])
            y_list.append(seq[1:])

        # Stack into tensors
        x = torch.tensor(np.stack(x_list), dtype=torch.long)
        y = torch.tensor(np.stack(y_list), dtype=torch.long)

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

# data_dir = os.path.join('..','data', TrainingConfig.dataset)
# print(f"Using Dataset {Path(data_dir).stem}")
train_loader = DataLoader(B=TrainingConfig.batch_size, T=ModelConfig.block_size, file_path="train.bin", device=device)
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

total_batch_size = TrainingConfig.total_batch_size
B = TrainingConfig.batch_size    # microbatch size
T = ModelConfig.block_size       # sequence length
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)

#___________CREATE YOUR MODEL_____________
model = LLM(ModelConfig).to(device)
total, active = model.get_num_params()
print(f"total parameters = {total:,}, acitive parameters = {active:,}")

if TrainingConfig.compile :  
    print("Using compiled model")
    model = torch.compile(model)

'''
#______________________________________________ TRAINING ______________________________________________

optimizer = model.configure_optimizers(weight_decay=0.1,learning_rate=TrainingConfig.learning_rate,device=device)

x,y = train_loader.next_batch() # get the first batch of training data
train_loss_stats = []
valrun_val_loss_stats = []
valrun_train_loss_stats = []
for iter in range(TrainingConfig.max_iters+1):
    t0 = perf_counter()

    lr = get_lr(iter, TrainingConfig) 
    for param_grp in optimizer.param_groups:
        param_grp['lr'] = lr
    
    optimizer.zero_grad(set_to_none=True)
    a,b = 0,0
    if TrainingConfig.eval and (iter % TrainingConfig.eval_interval == 0 or iter == TrainingConfig.max_iters) and iter!=0:
        a = perf_counter()
        losses = estimate_loss(model, TrainingConfig, train_loader, val_loader)
        valrun_val_loss_stats.append(losses['val'])
        valrun_train_loss_stats.append(losses['train'])
        b = perf_counter()
        print(f"--------val run-------- train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | dt {1000*(b-a):.4f}ms")
        t0 = b
    
    for micro_step in range(grad_accum_steps):
        with ctx:
            _, loss, _ = model(x,y) #logits, loss, kv cache
            loss:torch.Tensor = loss/grad_accum_steps

        x,y = train_loader.next_batch() # Async prefetch the next batch of data
        train_loss_stats.append(loss.item())
        scaler.scale(loss).backward()

    if TrainingConfig.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.grad_clip)

    scaler.step(optimizer)
    scaler.update()    
    mem = 0
    if "cuda" in device : 
        torch.cuda.synchronize()
        mem = torch.cuda.memory_reserved()
    
    dt  = (perf_counter()-t0)*1000
    print(f"step: {iter} | train loss:{loss*grad_accum_steps:.4f} | dt: {dt:.2f}ms | grad_accum_steps: {grad_accum_steps} | GPU RAM: {mem/1024**3:.2f}GB")

if TrainingConfig.save_model:
    # might do in-training checkpointing later
    loss_stats = {'train':train_loss_stats, 'valrun_val':valrun_val_loss_stats, 'valrun_train':valrun_train_loss_stats}

    checkpoint = {'model_config':ModelConfig, 'train_config':TrainingConfig, 'model_state': model.state_dict()}
    stats      = {'model_config':ModelConfig, 'train_config':TrainingConfig, 'losses':loss_stats, 'total_params':total, 'active_params':active}

    torch.save(checkpoint, TrainingConfig.file_name+'_ckpt.pt')
    torch.save(stats, TrainingConfig.file_name+'_stats.pt')

    print("Model and config saved to {}.pt".format(TrainingConfig.file_name + '_ckpt'))
    print("Stats and config saved to {}.pt".format(TrainingConfig.file_name + '_stats'))

'''


import torch.multiprocessing as mp
from contextlib import nullcontext
from packaging import version
import os
import glob
import torch.distributed as dist  # CRITICAL FIX: Added missing import

# Set NCCL environment variables for stability
os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
os.environ.setdefault('NCCL_DEBUG', 'WARN') 
os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
# Make P2P configurable rather than default disabled
if os.getenv('NCCL_P2P_DISABLE') is None:
    os.environ.setdefault('NCCL_P2P_DISABLE', '0')  # Enable by default for better perf on NVLink

# Version check at program start
assert version.parse(torch.__version__) >= version.parse("2.1.0"), \
    "EP MoE requires PyTorch >= 2.1.0 for autograd on all_to_all_single"

# def setup_ep_groups(ep_size: int):
#     """Initialize expert parallelism groups"""
#     if not dist.is_initialized():
#         dist.init_process_group(backend='nccl')
    
#     world_size = dist.get_world_size()
#     rank = dist.get_rank()
    
#     if ep_size > world_size:
#         raise ValueError(f"EP size ({ep_size}) cannot be larger than world size ({world_size})")
    
#     # For pure EP, use all GPUs in one EP group
#     ep_group = dist.new_group(list(range(world_size)))
#     ep_rank = rank
    
#     return ep_group, ep_rank, world_size

import sys
import gc
import datetime
import torch
import torch.distributed as dist

# def finalize_training(local_rank, train_loader=None, val_loader=None):
#     """Robust cleanup function that ensures proper termination"""
#     try:
#         # Finish all device work
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#     except Exception:
#         pass

#     # Close dataset memory maps (rank-0 created them)
#     if local_rank == 0:
#         try:
#             if train_loader is not None and hasattr(train_loader, "close"):
#                 train_loader.close()
#             if val_loader is not None and hasattr(val_loader, "close"):
#                 val_loader.close()
#         except Exception:
#             pass

#     # Rendezvous and teardown process group
#     if dist.is_available() and dist.is_initialized():
#         try:
#             dist.barrier()  # Ensure all ranks finish before teardown
#         except Exception:
#             pass
#         try:
#             dist.destroy_process_group()
#         except Exception:
#             pass

#     # Free memory
#     try:
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#     except Exception:
#         pass

#     if local_rank == 0:
#         print("✅ Training resources cleaned up successfully")

# def setup_ep_groups(ep_size: int, local_rank: int, world_size: int):
#     """Initialize expert parallelism groups with proper error handling"""
#     # Use updated environment variable
#     os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    
#     if not dist.is_initialized():
#         try:
#             dist.init_process_group(
#                 backend='nccl' if torch.cuda.is_available() else 'gloo',
#                 init_method='env://',
#                 world_size=world_size,
#                 rank=local_rank,
#                 timeout=datetime.timedelta(seconds=180)
#             )
#         except Exception as e:
#             print(f"Rank {local_rank}: Failed to initialize process group: {e}")
#             raise
    
#     # Create EP group with all ranks
#     ep_group = dist.new_group(list(range(world_size)))
    
#     return ep_group, local_rank, world_size


def create_worker_model(config: LLMconfig, device: str, moe_layer_mask: list[bool]):
    """Create a lightweight model for worker ranks (experts only)"""
    class WorkerMoEBlock(nn.Module):
        """Minimal block containing only MoE layers for worker ranks"""
        def __init__(self, config):
            super().__init__()
            self.moe = MoE(config)
        
        def forward(self, x):
            return self.moe(x)
    
    class WorkerLLM(nn.Module):
        """Lightweight model for worker ranks containing only MoE layers"""
        def __init__(self, config, moe_layer_mask):
            super().__init__()
            self.config = config
            self.moe_layer_mask = moe_layer_mask
            
            # Only create MoE blocks for layers that are actually MoE in the full model
            self.moe_blocks = nn.ModuleList()
            for i, is_moe in enumerate(moe_layer_mask):
                if is_moe:
                    self.moe_blocks.append(WorkerMoEBlock(config))
                # Don't create anything for non-MoE layers
            
            # Freeze all parameters initially, we'll unfreeze only experts
            for param in self.parameters():
                param.requires_grad = False
            
            # Unfreeze only the local routed experts
            for block in self.moe_blocks:
                if hasattr(block, 'moe') and hasattr(block.moe, 'local_routed_experts'):
                    for expert in block.moe.local_routed_experts:
                        for param in expert.parameters():
                            param.requires_grad = True
        
        def forward(self, x, targets=None, kv_caches=None):
            # Workers only participate in MoE computation via all_to_all
            # They don't compute loss or final outputs
            if kv_caches is None:
                kv_caches = [None] * self.config.n_layer
            
            total_aux_loss = 0.0
            moe_block_idx = 0
            for i in range(self.config.n_layer):
                if self.moe_layer_mask[i]:
                    # Only process MoE layers that exist in this worker model
                    if moe_block_idx < len(self.moe_blocks):
                        x, aux_loss = self.moe_blocks[moe_block_idx](x)
                        total_aux_loss += aux_loss
                        moe_block_idx += 1
                # Skip non-MoE layers entirely on workers
            
            # Workers don't compute final output or loss
            return None, None, kv_caches
    
    return WorkerLLM(config, moe_layer_mask).to(device)

def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Find the latest checkpoint iteration across all ranks"""
    if not os.path.exists(checkpoint_dir):
        return 0
    
    checkpoint_files = glob.glob(f"{checkpoint_dir}/rank_*_iter_*.pt")
    if not checkpoint_files:
        return 0
    
    # Extract iteration numbers and find the maximum
    iterations = []
    for file in checkpoint_files:
        try:
            iter_num = int(file.split('_iter_')[-1].split('.pt')[0])
            iterations.append(iter_num)
        except (ValueError, IndexError):
            continue
    
    return max(iterations) if iterations else 0

def save_checkpoint(model, optimizer, iter, rank, checkpoint_dir="checkpoints"):
    """Save checkpoint with distributed expert handling"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'iteration': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rank': rank,
    }
    
    # Each rank saves its own checkpoint
    torch.save(checkpoint, f"{checkpoint_dir}/rank_{rank}_iter_{iter}.pt")
    
    # Rank 0 also saves a metadata file
    if rank == 0:
        metadata = {
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'iteration': iter,
            'timestamp': torch.tensor(torch.timestamp()),
        }
        torch.save(metadata, f"{checkpoint_dir}/metadata_iter_{iter}.pt")

def load_checkpoint(model, optimizer, checkpoint_dir="checkpoints", resume_iter=None):
    """Load checkpoint with distributed expert handling"""
    if resume_iter is None:
        # Find the latest checkpoint
        resume_iter = find_latest_checkpoint(checkpoint_dir)
    
    if resume_iter == 0:
        return 0  # No checkpoint found
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    checkpoint_path = f"{checkpoint_dir}/rank_{rank}_iter_{resume_iter}.pt"
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Rebuild optimizer with current parameters before loading state
        if rank != 0:
            # Workers need to rebuild optimizer with current local experts
            local_params = []
            for module in model.modules():
                if hasattr(module, 'local_routed_experts'):
                    for expert in module.local_routed_experts:
                        local_params.extend([p for p in expert.parameters() if p.requires_grad])
            optimizer = torch.optim.AdamW(local_params, lr=optimizer.param_groups[0]['lr'])
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Rank {rank}: Loaded checkpoint from iteration {resume_iter}")
        return checkpoint['iteration']
    else:
        print(f"Rank {rank}: Checkpoint not found: {checkpoint_path}")
        return 0

def main_worker(local_rank, world_size, TrainingConfig, ModelConfig):
    """Worker function for expert parallelism"""

    train_loader = None
    val_loader = None

    try:
        # CRITICAL FIX: Use local_rank for device assignment (supports multi-node)
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(local_rank)
        
        # Set deterministic seeds (same across ranks for reproducibility)
        torch.manual_seed(42 + local_rank)  # Different expert params per rank
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + local_rank)
        
        # Initialize distributed
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
        
        # Setup expert parallelism - use ALL GPUs for EP
        # ep_group, ep_rank, ep_size = setup_ep_groups(world_size)
        ep_group, ep_rank, ep_size = setup_ep_groups(world_size, local_rank, world_size)
        
        # Verify world size matches EP size
        if world_size != ep_size:
            raise ValueError(f"World size {world_size} must equal EP size {ep_size} for pure EP")
        
        # Update config with EP info
        # ModelConfig.ep_size = ep_size
        # ModelConfig.ep_rank = ep_rank
        # ModelConfig.ep_group = ep_group
        # Create a copy of ModelConfig to avoid modifying the original
        model_config_copy = LLMconfig(
            vocab_size=ModelConfig.vocab_size,
            block_size=ModelConfig.block_size,
            n_embd=ModelConfig.n_embd,
            pos_emb=ModelConfig.pos_emb,
            up_dim=ModelConfig.up_dim,
            non_linearity=ModelConfig.non_linearity,
            dropout=ModelConfig.dropout,
            n_layer=ModelConfig.n_layer,
            moe=ModelConfig.moe,
            n_exp=ModelConfig.n_exp,
            n_shared=ModelConfig.n_shared,
            n_act=ModelConfig.n_act,
            coeff=ModelConfig.coeff,
            aux_free=ModelConfig.aux_free,
            alpha=ModelConfig.alpha,
            gamma=ModelConfig.gamma,
            attn=ModelConfig.attn,
            n_head=ModelConfig.n_head,
            n_kv_heads=ModelConfig.n_kv_heads,
            q_latent_dim=ModelConfig.q_latent_dim,
            kv_latent_dim=ModelConfig.kv_latent_dim,
            rope_head_dim=ModelConfig.rope_head_dim,
            act_recomp=ModelConfig.act_recomp,
            # Set EP attributes
            ep_size=ep_size,
            ep_rank=ep_rank,
            ep_group=ep_group
        )
        

        # Setup AMP
        device_type = 'cuda'
        amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast(device_type=device_type, dtype=amp_dtype)
        
        # Different model creation for rank 0 vs workers
        if local_rank == 0:
            # Rank 0: full model
            model = LLM(model_config_copy).to(device)
            train_loader = DataLoader(B=TrainingConfig.batch_size, T=model_config_copy.block_size, 
                                    file_path="train.bin", device=device)
            total, active = model.get_num_params()
            print(f"total parameters = {total:,}, active parameters = {active:,}")
            
            # Full optimizer for rank 0
            optimizer = model.configure_optimizers(
                weight_decay=0.1, 
                learning_rate=TrainingConfig.learning_rate, 
                device=device
            )
            scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))
            
            # CRITICAL FIX: Create MoE layer mask to mirror full model structure
            moe_layer_mask = []
            for block in model.transformer.h:
                moe_layer_mask.append(hasattr(block, 'moe') and block.is_moe)
        else:
            # Worker ranks need the MoE layer mask from rank 0
            moe_layer_mask = [False] * model_config_copy.n_layer  # Placeholder
            # Worker ranks get minimal loaders for cleanup consistency
            train_loader = None
            val_loader = None
            
        # Broadcast MoE layer mask from rank 0 to all workers
        if world_size > 1:
            moe_layer_mask_tensor = torch.tensor(moe_layer_mask, dtype=torch.bool, device=device)
            dist.broadcast(moe_layer_mask_tensor, src=0)
            moe_layer_mask = moe_layer_mask_tensor.cpu().tolist()
        
        if local_rank != 0:
            # Worker ranks: lightweight model with only experts, mirroring rank 0's structure
            model = create_worker_model(model_config_copy, device, moe_layer_mask)
            train_loader = None
            
            # Local optimizer for worker experts only
            local_params = []
            for module in model.modules():
                if hasattr(module, 'local_routed_experts'):
                    for expert in module.local_routed_experts:
                        local_params.extend([p for p in expert.parameters() if p.requires_grad])
            
            optimizer = torch.optim.AdamW(
                local_params, 
                lr=TrainingConfig.learning_rate,
                weight_decay=0.1
            )
            # Workers don't use GradScaler
            scaler = None
        
        # Get parameter dtype for dummy inputs
        param_dtype = next(model.parameters()).dtype
        
        # Set model to training mode
        model.train()
        
        # Training loop with corrected synchronization
        start_iter = 0
        
        # Optional: load checkpoint
        if TrainingConfig.save_model:
            start_iter = load_checkpoint(model, optimizer)
            # Ensure all ranks resume consistently and validate iteration
            if world_size > 1:
                start_iter_tensor = torch.tensor(start_iter, device=device)
                dist.broadcast(start_iter_tensor, src=0)
                start_iter = start_iter_tensor.item()
        
        for iter in range(start_iter, TrainingConfig.max_iters + 1):

            t0 = perf_counter()

            # Learning rate scheduling
            lr = get_lr(iter, TrainingConfig) 
            for param_grp in optimizer.param_groups:
                param_grp['lr'] = lr
            

            # Zero gradients on all ranks
            optimizer.zero_grad(set_to_none=True)
            
            if local_rank == 0:
                # Get batch and forward/backward on rank 0
                x, y = train_loader.next_batch()
                
                # Forward pass (includes EP communication)
                with ctx:
                    _, loss, _ = model(x, y)
                
                # Backward pass
                scaler.scale(loss).backward()
            else:
                # Worker ranks: dummy forward to trigger EP communication and gradients
                # Use a small dummy input to minimize memory usage
                dummy_x = torch.zeros(1, 1, ModelConfig.n_embd, device=device, dtype=param_dtype)
                with ctx:
                    _, _, _ = model(dummy_x)
                # No backward on workers - gradients flow via autograd through collectives
            
            # CRITICAL: Ensure backward is finished before stepping
            dist.barrier()
            
            # Optimization step
            if local_rank == 0:
                if TrainingConfig.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()

                # Synchronize and measure time
                if "cuda" in device:
                    torch.cuda.synchronize()
                dt = (perf_counter() - t0) * 1000  # Convert to milliseconds
                
                # Memory usage
                if torch.cuda.is_available():
                    mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)  # Convert to GB
                    torch.cuda.reset_peak_memory_stats(device)
                else:
                    mem_gb = 0.0
                
                if local_rank == 0:
                    print(f"step: {iter} | train loss:{total_loss * grad_accum_steps:.4f} | "
          f"dt: {dt:.2f}ms | grad_accum_steps: {grad_accum_steps} | "
          f"GPU RAM: {mem_gb:.2f}GB")

                # Final iteration - save checkpoint and prepare for termination
                if iter == TrainingConfig.max_iters:
                    if local_rank == 0:
                        print(f"🎉 Training completed all {TrainingConfig.max_iters} iterations!")

                    # Save final checkpoint on rank 0
                    if TrainingConfig.save_model and local_rank == 0:
                        save_checkpoint(model, optimizer, TrainingConfig.max_iters, local_rank)
                        print("💾 Final checkpoint saved")

                    # Ensure all ranks finish before cleanup
                    if dist.is_initialized():
                        dist.barrier()

            else:
                # Worker optimization (no scaler)
                if TrainingConfig.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(local_params, TrainingConfig.grad_clip)
                
                optimizer.step()
            
            # Save checkpoint periodically
            if TrainingConfig.save_model and iter % TrainingConfig.eval_interval == 0 and iter > 0:
                save_checkpoint(model, optimizer, iter, local_rank)
            
            # Final synchronization (keep for correctness, can optimize later)
            dist.barrier()
    except KeyboardInterrupt:
        if local_rank == 0:
            print("⏹️ Training interrupted by user")
        # Save partial checkpoint if desired
        if TrainingConfig.save_model and local_rank == 0:
            save_checkpoint(model, optimizer, iter, local_rank)
            print("💾 Partial checkpoint saved")
        raise
        
    except Exception as e:
        print(f"Rank {local_rank}: Training error: {e}")
        raise
        
    finally:
        # GUARANTEED cleanup - this will always run
        finalize_training(local_rank, train_loader, val_loader)
        
        if local_rank == 0:
            print("🏁 Worker process completed and cleaned up")

# def main():
    

#     world_size = int(os.environ["WORLD_SIZE"])
#     local_rank = int(os.environ["LOCAL_RANK"])
#     main_worker(local_rank, world_size, TrainingConfig, ModelConfig)

# if __name__ == "__main__":
#     main()

# def main():
#     """Main function for torchrun launch method"""
#     try:
#         # Get distributed setup from environment variables (torchrun sets these)
#         world_size = int(os.environ["WORLD_SIZE"])
#         local_rank = int(os.environ["LOCAL_RANK"])
        
#         print(f"🚀 Starting torchrun training - Rank {local_rank}/{world_size-1}")
#         print(f"📊 Target: {TrainingConfig.max_iters} iterations")
        
#         # Call the worker function
#         main_worker(local_rank, world_size, TrainingConfig, ModelConfig)
        
#         # If we reach here, training completed successfully
#         if local_rank == 0:
#             print("✅ All training iterations completed successfully")
            
#     except KeyboardInterrupt:
#         if 'local_rank' in locals() and local_rank == 0:
#             print("⏹️ Training interrupted by user")
#         # Re-raise to ensure torchrun sees the interruption
#         raise
        
#     except Exception as e:
#         print(f"❌ Training failed: {e}")
#         # Re-raise to ensure torchrun propagates the error
#         raise
        
#     finally:
#         # Final cleanup in the main process
#         if 'local_rank' in locals() and local_rank == 0:
#             print("🧹 Final cleanup completed")
        
#         # Force garbage collection
#         import gc
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

# if __name__ == "__main__":
#     main()
#     # torchrun will handle process termination, no need for sys.exit() here






import sys
import gc
import datetime
import torch
import torch.distributed as dist
from time import perf_counter

def finalize_training(local_rank, train_loader=None, val_loader=None):
    """Robust cleanup function that ensures proper termination"""
    try:
        # Finish all device work
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

    # Close dataset memory maps (rank-0 created them)
    if local_rank == 0:
        try:
            if train_loader is not None and hasattr(train_loader, "close"):
                train_loader.close()
            if val_loader is not None and hasattr(val_loader, "close"):
                val_loader.close()
        except Exception:
            pass

    # Rendezvous and teardown process group
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()  # Ensure all ranks finish before teardown
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    # Free memory
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if local_rank == 0:
        print("✅ Training resources cleaned up successfully")

def setup_ep_groups(ep_size: int, local_rank: int, world_size: int):
    """Initialize expert parallelism groups with proper error handling"""
    # Use updated environment variable
    import os
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    
    if not dist.is_initialized():
        try:
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                init_method='env://',
                world_size=world_size,
                rank=local_rank,
                timeout=datetime.timedelta(seconds=180)
            )
        except Exception as e:
            print(f"Rank {local_rank}: Failed to initialize process group: {e}")
            raise
    
    # Create EP group with all ranks
    ep_group = dist.new_group(list(range(world_size)))
    
    return ep_group, local_rank, world_size

def main_worker(local_rank, world_size, TrainingConfig, ModelConfig):
    """Worker function with detailed step printing and proper termination"""
    train_loader = None
    val_loader = None
    
    try:
        # CRITICAL FIX: Use local_rank for device assignment (supports multi-node)
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(local_rank)
        
        # Set deterministic seeds (same across ranks for reproducibility)
        torch.manual_seed(42 + local_rank)  # Different expert params per rank
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + local_rank)
        
        # Initialize distributed
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
        
        # Setup expert parallelism - use ALL GPUs for EP
        ep_group, ep_rank, ep_size = setup_ep_groups(world_size, local_rank, world_size)
        
        # Verify world size matches EP size
        if world_size != ep_size:
            raise ValueError(f"World size {world_size} must equal EP size {ep_size} for pure EP")
        
        # Create a copy of ModelConfig to avoid modifying the original
        model_config_copy = LLMconfig(
            vocab_size=ModelConfig.vocab_size,
            block_size=ModelConfig.block_size,
            n_embd=ModelConfig.n_embd,
            pos_emb=ModelConfig.pos_emb,
            up_dim=ModelConfig.up_dim,
            non_linearity=ModelConfig.non_linearity,
            dropout=ModelConfig.dropout,
            n_layer=ModelConfig.n_layer,
            moe=ModelConfig.moe,
            n_exp=ModelConfig.n_exp,
            n_shared=ModelConfig.n_shared,
            n_act=ModelConfig.n_act,
            coeff=ModelConfig.coeff,
            aux_free=ModelConfig.aux_free,
            alpha=ModelConfig.alpha,
            gamma=ModelConfig.gamma,
            attn=ModelConfig.attn,
            n_head=ModelConfig.n_head,
            n_kv_heads=ModelConfig.n_kv_heads,
            q_latent_dim=ModelConfig.q_latent_dim,
            kv_latent_dim=ModelConfig.kv_latent_dim,
            rope_head_dim=ModelConfig.rope_head_dim,
            act_recomp=ModelConfig.act_recomp,
            # Set EP attributes
            ep_size=ep_size,
            ep_rank=ep_rank,
            ep_group=ep_group
        )

        # Setup AMP
        device_type = 'cuda'
        amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast(device_type=device_type, dtype=amp_dtype)
        
        # Calculate gradient accumulation steps
        total_batch_size = TrainingConfig.total_batch_size
        B = TrainingConfig.batch_size
        T = model_config_copy.block_size
        assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
        grad_accum_steps = total_batch_size // (B * T)
        
        if local_rank == 0:
            print(f"📈 Training with gradient accumulation: {grad_accum_steps} steps")
        
        # Different model creation for rank 0 vs workers
        if local_rank == 0:
            # Rank 0: full model
            model = LLM(model_config_copy).to(device)
            train_loader = DataLoader(B=TrainingConfig.batch_size, T=model_config_copy.block_size, 
                                    file_path="train.bin", device=device)
            total, active = model.get_num_params()
            print(f"total parameters = {total:,}, active parameters = {active:,}")
            
            # Full optimizer for rank 0
            optimizer = model.configure_optimizers(
                weight_decay=0.1, 
                learning_rate=TrainingConfig.learning_rate, 
                device=device
            )
            scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))
            
            # CRITICAL FIX: Create MoE layer mask to mirror full model structure
            moe_layer_mask = []
            for block in model.transformer.h:
                moe_layer_mask.append(hasattr(block, 'moe') and block.is_moe)
                
            # Get initial batch
            x, y = train_loader.next_batch()
        else:
            # Worker ranks need the MoE layer mask from rank 0
            moe_layer_mask = [False] * model_config_copy.n_layer  # Placeholder
            # Worker ranks get minimal loaders for cleanup consistency
            train_loader = None
            val_loader = None
            
        # Broadcast MoE layer mask from rank 0 to all workers
        if world_size > 1:
            moe_layer_mask_tensor = torch.tensor(moe_layer_mask, dtype=torch.bool, device=device)
            dist.broadcast(moe_layer_mask_tensor, src=0)
            moe_layer_mask = moe_layer_mask_tensor.cpu().tolist()
        
        if local_rank != 0:
            # Worker ranks: lightweight model with only experts, mirroring rank 0's structure
            model = create_worker_model(model_config_copy, device, moe_layer_mask)
            train_loader = None
            
            # Local optimizer for worker experts only
            local_params = []
            for module in model.modules():
                if hasattr(module, 'local_routed_experts'):
                    for expert in module.local_routed_experts:
                        local_params.extend([p for p in expert.parameters() if p.requires_grad])
            
            optimizer = torch.optim.AdamW(
                local_params, 
                lr=TrainingConfig.learning_rate,
                weight_decay=0.1
            )
            # Workers don't use GradScaler
            scaler = None
        
        # Get parameter dtype for dummy inputs
        param_dtype = next(model.parameters()).dtype
        
        # Set model to training mode
        model.train()
        
        # Training loop with corrected synchronization
        start_iter = 0
        
        # Optional: load checkpoint
        if TrainingConfig.save_model:
            start_iter = load_checkpoint(model, optimizer)
            # Ensure all ranks resume consistently and validate iteration
            if world_size > 1:
                start_iter_tensor = torch.tensor(start_iter, device=device)
                dist.broadcast(start_iter_tensor, src=0)
                start_iter = start_iter_tensor.item()
        
        # MAIN TRAINING LOOP
        for iter in range(start_iter, TrainingConfig.max_iters + 1):
            t0 = perf_counter()

            # Learning rate scheduling
            lr = get_lr(iter, TrainingConfig) 
            for param_grp in optimizer.param_groups:
                param_grp['lr'] = lr

            # Zero gradients on all ranks
            optimizer.zero_grad(set_to_none=True)
            
            # Track loss for printing (only on rank 0)
            current_loss = 0.0
            
            if local_rank == 0:
                # Get batch and forward/backward on rank 0
                x, y = train_loader.next_batch()
                
                # Forward pass (includes EP communication)
                with ctx:
                    _, loss, _ = model(x, y)
                    current_loss = loss.item()
                
                # Backward pass
                scaler.scale(loss).backward()
            else:
                # Worker ranks: dummy forward to trigger EP communication and gradients
                # Use a small dummy input to minimize memory usage
                dummy_x = torch.zeros(1, 1, ModelConfig.n_embd, device=device, dtype=param_dtype)
                with ctx:
                    _, _, _ = model(dummy_x)
                # No backward on workers - gradients flow via autograd through collectives
            
            # CRITICAL: Ensure backward is finished before stepping
            if world_size > 1:
                dist.barrier()
            
            # Optimization step
            if local_rank == 0:
                if TrainingConfig.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Worker optimization (no scaler)
                if TrainingConfig.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(local_params, TrainingConfig.grad_clip)
                
                optimizer.step()
            
            # Synchronize and measure time
            if "cuda" in device:
                torch.cuda.synchronize()
            dt = (perf_counter() - t0) * 1000  # Convert to milliseconds
            
            # Memory usage
            if torch.cuda.is_available():
                mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)  # Convert to GB
                torch.cuda.reset_peak_memory_stats(device)
            else:
                mem_gb = 0.0
            
            # Print step information (only on rank 0 to avoid duplicate output)
            if local_rank == 0:
                print(f"step: {iter} | train loss:{current_loss:.4f} | "
                      f"dt: {dt:.2f}ms | grad_accum_steps: {grad_accum_steps} | "
                      f"GPU RAM: {mem_gb:.2f}GB")

            # VALIDATION RUN - Every eval_interval steps and on final iteration
            # if (TrainingConfig.eval and 
            #     local_rank == 0 and 
            #     (iter % TrainingConfig.eval_interval == 0 or iter == TrainingConfig.max_iters) and 
            #     iter != 0):
                
            #     eval_start = perf_counter()
            #     losses = estimate_loss(model, TrainingConfig, train_loader, val_loader)
            #     eval_time = (perf_counter() - eval_start) * 1000
            #     print(f"--------val run-------- train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | dt {eval_time:.2f}ms")

            # a,b = 0,0
            # if TrainingConfig.eval and (iter % TrainingConfig.eval_interval == 0 or iter == TrainingConfig.max_iters) and iter!=0:
            #     a = perf_counter()
            #     losses = estimate_loss(model, TrainingConfig, train_loader, val_loader)
            #     valrun_val_loss_stats.append(losses['val'])
            #     valrun_train_loss_stats.append(losses['train'])
            #     b = perf_counter()
            #     print(f"--------val run-------- train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | dt {1000*(b-a):.4f}ms")
            #     t0 = b
            
            # Save checkpoint periodically
            if TrainingConfig.save_model and iter % TrainingConfig.eval_interval == 0 and iter > 0 and local_rank == 0:
                save_checkpoint(model, optimizer, iter, local_rank)
            
            # FINAL ITERATION - CRITICAL: Break out and cleanup
            if iter == TrainingConfig.max_iters:
                if local_rank == 0:
                    print(f"🎉 Training completed all {TrainingConfig.max_iters} iterations!")
                
                # Save final checkpoint on rank 0
                if TrainingConfig.save_model and local_rank == 0:
                    save_checkpoint(model, optimizer, TrainingConfig.max_iters, local_rank)
                    print("💾 Final checkpoint saved")
                
                # BREAK OUT OF THE LOOP - THIS PREVENTS INFINITE LOOP
                break
            
            # Final synchronization for non-final iterations
            if world_size > 1:
                dist.barrier()
                
    except KeyboardInterrupt:
        if local_rank == 0:
            print("⏹️ Training interrupted by user")
        # Save partial checkpoint if desired
        if TrainingConfig.save_model and local_rank == 0:
            save_checkpoint(model, optimizer, iter, local_rank)
            print("💾 Partial checkpoint saved")
        
    except Exception as e:
        print(f"Rank {local_rank}: Training error: {e}")
        raise
        
    finally:
        # GUARANTEED cleanup - this will always run and ensure termination
        finalize_training(local_rank, train_loader, val_loader)
        
        if local_rank == 0:
            print("🏁 Worker process completed and cleaned up")

def main():
    """Main function for torchrun launch method"""
    import os
    try:
        # Get distributed setup from environment variables (torchrun sets these)
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        print(f"🚀 Starting torchrun training - Rank {local_rank}/{world_size-1}")
        print(f"📊 Target: {TrainingConfig.max_iters} iterations")
        
        # Call the worker function
        main_worker(local_rank, world_size, TrainingConfig, ModelConfig)
        
        # If we reach here, training completed successfully
        if local_rank == 0:
            print("✅ All training iterations completed successfully")
            
    except KeyboardInterrupt:
        if 'local_rank' in locals() and local_rank == 0:
            print("⏹️ Training interrupted by user")
        # Re-raise to ensure torchrun sees the interruption
        raise
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        # Re-raise to ensure torchrun propagates the error
        raise
        
    finally:
        # Final cleanup in the main process
        if 'local_rank' in locals() and local_rank == 0:
            print("🧹 Final cleanup completed")
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
    # torchrun will handle process termination, no need for sys.exit() here