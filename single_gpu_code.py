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

# class MoE(nn.Module):
#     '''
#     This class implements the DeepSeekMoE layer, featuring shared and routed experts.
#     It uses an Auxiliary-Loss-Free load balancing strategy with a dynamic bias term.
#     Ref: https://arxiv.org/pdf/2412.19437
#     '''

#     def __init__(self, config: LLMconfig):
#         super().__init__()
#         self.config = config
        
#         # first `n_shared` are shared, the rest are routed
#         self.n_shared = config.n_shared
#         self.n_routed = config.n_exp - config.n_shared
        
#         # Number of experts to activate from the ROUTED pool
#         self.n_act_routed = config.n_act - config.n_shared
#         assert self.n_act_routed > 0, "Number of active experts must be greater than shared experts"

#         self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_exp)])
#         self.gate = nn.Linear(config.n_embd, self.n_routed, bias=False)
        
#         if config.aux_free:
#             self.register_buffer('expert_bias', torch.zeros(self.n_routed))

#     def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         """ Forward pass for the DeepSeekMoE layer with Aux-Loss-Free Balancing. """
#         B, T, C = x.shape
#         x_flat = x.view(-1, C)  # Shape: (B*T, C)
#         n_tokens = x_flat.shape[0]

#         # ___________ SHARED EXPERT PATH ___________

#         shared_output = torch.zeros_like(x_flat)
#         if self.n_shared > 0:
#             for i in range(self.n_shared):
#                 shared_output += self.experts[i](x_flat) # bypass the router

#         #  ___________ ROUTED EXPERT PATH ___________

#         router_logits = self.gate(x_flat)

#         if self.config.aux_free:        
#             # Add Bias and then select topk
#             biased_router_logits = router_logits + self.expert_bias
#             topk_biased_logits, topk_indices = torch.topk(biased_router_logits, self.n_act_routed, dim=1)

#             # Gating weights are based on un-biased logits
#             topk_original_logits = torch.gather(router_logits, 1, topk_indices) 
#             topk_gates = F.softmax(topk_original_logits, dim=1)

#             # Calculate expert load and update bias during training only
#             with torch.no_grad():
#                 ones = torch.ones_like(topk_indices, dtype=x_flat.dtype)
#                 fi_counts = torch.zeros(self.n_routed, device=x.device).scatter_add_(0, topk_indices.flatten(), ones.flatten())
#                 fi = fi_counts / n_tokens

#             if self.training:
#                 with torch.no_grad():
#                     ideal_load = 1.0 / self.n_routed
#                     delta = ideal_load - fi 
#                     self.expert_bias += (self.config.gamma*delta)

#             router_probs = F.softmax(router_logits, dim=1)
#             pi = router_probs.mean(dim=0)
#             aux_loss = self.config.alpha * self.n_routed * torch.sum(pi*fi)

#         else:
#             router_probs = F.softmax(router_logits, dim=1)
#             pi = router_probs.mean(dim=0)
            
#             topk_logits, topk_indices = torch.topk(router_logits, self.n_act_routed, dim=1)
#             ones = torch.ones_like(topk_indices, dtype=torch.float)
#             fi_counts = torch.zeros(self.n_routed, device=x.device).scatter_add_(0, topk_indices.flatten(), ones.flatten())
#             fi = fi_counts / n_tokens

#             aux_loss = self.config.coeff * self.n_routed * torch.sum(pi * fi)

#             topk_gates = F.softmax(topk_logits, dim=1)  

#         # Dispatch
#         routed_output = torch.zeros_like(x_flat)

#         for i in range(self.n_routed):
#             token_indices, topk_slot = (topk_indices == i).nonzero(as_tuple=True)
#             if token_indices.numel() > 0:
#                 tokens_for_expert = x_flat[token_indices]
#                 gates_for_expert = topk_gates[token_indices, topk_slot].unsqueeze(1)

#                 # access the expert using an offset of `n_shared`
#                 expert_output = self.experts[i + self.n_shared](tokens_for_expert)
                
#                 weighted_output = expert_output * gates_for_expert
#                 routed_output.index_add_(0, token_indices, weighted_output)
        
#         # combine to output
#         y = (shared_output + routed_output).view(B, T, C)
#         return y, aux_loss

class Block(nn.Module):
    """ A single Transformer block combining attention and MLP. """
    def __init__(self, config:LLMconfig):
        super().__init__()
        self.is_moe = config.moe
        self.attn = Attention(config)
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.ln2  = nn.LayerNorm(config.n_embd)
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

def setup_expert_parallel():
    """Initialize expert parallel groups"""
    global expert_parallel_group
    
    # Create a group for expert parallelism (all GPUs)
    if torch.distributed.is_initialized():
        # expert_parallel_group = torch.distributed.new_group(backend='nccl')
        expert_parallel_group = torch.distributed.group.WORLD
    else:
        expert_parallel_group = None
    return expert_parallel_group

def get_expert_parallel_group():
    """Get the expert parallel group"""
    return expert_parallel_group

# Initialize expert parallel group at the start
expert_parallel_group = setup_expert_parallel()


def sync_moe_gradients(model):
    """Sync gradients for replicated MoE parameters (only when using expert parallelism)"""
    if hasattr(model, 'module'):  # FSDP wrapped
        raw_model = model.module
    else:
        raw_model = model

    # Only sync if we're using expert parallelism and world_size > 1
    for block in raw_model.transformer.h:
        if block.is_moe and hasattr(block.moe, 'expert_parallel') and block.moe.expert_parallel:
            # Synchronize shared expert and gate gradients across expert parallel group
            for param in block.moe.shared_experts.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.AVG, group=block.moe.ep_group)
            
            for param in block.moe.gate.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.AVG, group=block.moe.ep_group)




class MoE(nn.Module):
    '''
    Expert Parallel MoE layer - Production-ready implementation
    '''
    
    def __init__(self, config: LLMconfig):
        super().__init__()
        self.config = config
        self._owner_map_cache = {}  # device -> tensor
        
        # Handle distributed initialization safely
        if not torch.distributed.is_initialized():
            self.ep_group = None
            self.world_size, self.rank = 1, 0
            self.single_gpu_fallback = True
            self.expert_parallel = False  # No EP in single GPU mode
        else:
            self.ep_group = get_expert_parallel_group()
            if self.ep_group is None:
                self.ep_group = torch.distributed.group.WORLD
            self.world_size = torch.distributed.get_world_size(group=self.ep_group)
            self.rank = torch.distributed.get_rank(group=self.ep_group)
            self.single_gpu_fallback = (self.world_size == 1)
            self.expert_parallel = not self.single_gpu_fallback
        
        # Calculate expert distribution with robust partitioning
        self.n_shared = config.n_shared
        self.n_routed = config.n_exp - config.n_shared
        self.n_act_routed = config.n_act - self.n_shared
        
        # Robust expert partitioning (handles uneven distribution)
        self.experts_per_rank = (self.n_routed + self.world_size - 1) // self.world_size
        self.local_expert_start = self.rank * self.experts_per_rank
        self.local_expert_end = min(self.local_expert_start + self.experts_per_rank, self.n_routed)
        self.local_expert_indices = list(range(self.local_expert_start, self.local_expert_end))
        self.num_local_experts = len(self.local_expert_indices)
        
        # Build owner map for global expert to rank mapping
        owner_map = torch.full((self.n_routed,), self.world_size - 1, dtype=torch.int64)
        for global_expert_idx in range(self.n_routed):
            owner_map[global_expert_idx] = min(global_expert_idx // self.experts_per_rank, self.world_size - 1)
        
        # Register as non-persistent buffer (won't be saved in checkpoint)
        # self.register_buffer("owner_map", owner_map.to(torch.cuda.current_device()), persistent=False)
        # NOW (device-agnostic; cheap to keep on CPU)
        self.register_buffer("owner_map", owner_map, persistent=False)
        
        # Create local experts (routed experts only) - handle empty case
        # self.local_experts = nn.ModuleList([
        #     Expert(config) for _ in range(self.num_local_experts)
        # ]) if self.num_local_experts > 0 else nn.ModuleList()
        self.local_experts = nn.ModuleList([
            Expert(config) for _ in range(self.num_local_experts)
        ]) if self.num_local_experts > 0 else nn.ModuleList()

        
        # Single GPU sanity check
        if self.single_gpu_fallback and len(self.local_experts) != self.n_routed:
            raise RuntimeError("EP: single-GPU mode expects all routed experts local.")
        
        # Shared experts are replicated on all GPUs
        self.shared_experts = nn.ModuleList([
            Expert(config) for _ in range(self.n_shared)
        ])
        
        # Gate network - replicated on all GPUs
        self.gate = nn.Linear(config.n_embd, self.n_routed, bias=False)
        
        if config.aux_free:
            self.register_buffer('expert_bias', torch.zeros(self.n_routed))
        
        # Capacity configuration
        self.capacity_factor = getattr(config, 'capacity_factor', 1.25)
        self.capacity_policy = getattr(config, 'capacity_policy', 'topk')  # 'topk', 'random', 'drop'
        self.bias_clamp_min = getattr(config, 'bias_clamp_min', -2.0)
        self.bias_clamp_max = getattr(config, 'bias_clamp_max', 2.0)
        
        # Debug mode
        self.debug_asserts = getattr(config, 'moe_debug_asserts', False)

    def _owner_map_on(self, device: torch.device) -> torch.Tensor:
        """
        Return owner_map on the requested device. Cache per device to avoid
        re-copying each forward. Assumes owner_map is a 1D int32 tensor.
        """
        if device not in self._owner_map_cache:
            self._owner_map_cache[device] = self.owner_map.to(device, non_blocking=True)
        return self._owner_map_cache[device]

    def _build_send_buffers_vectorized(self, x_flat, topk_indices, topk_gates):
        """Vectorized implementation of send buffer building with two-pass for efficiency"""
        n_tokens = x_flat.shape[0]
        
        # Flatten all top-k assignments
        assign_flat = topk_indices.reshape(-1)  # (n_tokens * n_act_routed,)
        origin_flat = torch.arange(n_tokens, device=x_flat.device).repeat_interleave(self.n_act_routed)
        gates_flat = topk_gates.reshape(-1)
        
        # Get destination ranks for each assignment - VECTORIZED
        # dest_ranks = self.owner_map[assign_flat]

        owner_map_dev = self._owner_map_on(x_flat.device)
        dest_ranks = owner_map_dev[assign_flat]
        
        # Count tokens per rank - VECTORIZED with proper dtype
        send_counts = torch.bincount(dest_ranks, minlength=self.world_size).to(torch.int64)
        
        # Early return if single GPU or no tokens to send
        if self.single_gpu_fallback or send_counts.sum() == 0:
            return send_counts, None, None, None
        
        # Precompute displacements for scattering with proper dtype
        send_displacements = torch.cumsum(send_counts, 0, dtype=torch.int64) - send_counts
        
        # Create contiguous send buffers using two-pass approach
        total_send = send_counts.sum().item()
        send_tensor = torch.zeros((total_send, x_flat.size(1)), device=x_flat.device, dtype=x_flat.dtype)
        send_gates = torch.zeros((total_send,), device=x_flat.device, dtype=torch.float32)  # gates in fp32
        send_metadata = torch.zeros((total_send, 2), device=x_flat.device, dtype=torch.int64)
        
        # Scatter data into buffers
        for rank_idx in range(self.world_size):
            rank_mask = (dest_ranks == rank_idx)
            if rank_mask.any():
                # Get data for this rank
                rank_origins = origin_flat[rank_mask]
                rank_assigns = assign_flat[rank_mask]
                rank_gates = gates_flat[rank_mask]
                rank_tokens = x_flat[rank_origins]
                
                # Sort by global_expert_idx then origin_index (stable two-stage sort)
                # First sort by origin for stability
                sort_indices_1 = torch.argsort(rank_origins, stable=True)
                rank_origins = rank_origins[sort_indices_1]
                rank_assigns = rank_assigns[sort_indices_1]
                rank_gates = rank_gates[sort_indices_1]
                rank_tokens = rank_tokens[sort_indices_1]
                
                # Then sort by expert assignment
                sort_indices_2 = torch.argsort(rank_assigns, stable=True)
                rank_origins = rank_origins[sort_indices_2]
                rank_assigns = rank_assigns[sort_indices_2]
                rank_gates = rank_gates[sort_indices_2]
                rank_tokens = rank_tokens[sort_indices_2]
                
                # Write to contiguous buffers
                start_idx = send_displacements[rank_idx]
                end_idx = start_idx + len(rank_origins)
                
                send_tensor[start_idx:end_idx] = rank_tokens
                send_gates[start_idx:end_idx] = rank_gates
                send_metadata[start_idx:end_idx, 0] = rank_origins
                send_metadata[start_idx:end_idx, 1] = rank_assigns
        
        return send_counts, send_tensor, send_gates, send_metadata

    def _process_local_experts_segmented(self, recv_tensor, recv_gates, recv_metadata, n_tokens_global):
        """Process local experts with segmented batching for better performance"""
        if recv_tensor.size(0) == 0:
            return torch.zeros_like(recv_tensor)
        
        # Cast gates to match expert output dtype once (performance optimization)
        recv_gates_typed = recv_gates.to(recv_tensor.dtype)
        
        # Sort all received data by global expert ID for better locality
        sort_indices = torch.argsort(recv_metadata[:, 1])
        recv_tensor = recv_tensor[sort_indices]
        recv_gates_typed = recv_gates_typed[sort_indices]
        recv_metadata = recv_metadata[sort_indices]
        
        # Find segment boundaries per expert
        unique_experts, segment_lengths = torch.unique_consecutive(recv_metadata[:, 1], return_counts=True)
        segment_starts = torch.cat([torch.tensor([0], device=recv_tensor.device), torch.cumsum(segment_lengths, 0)[:-1]])
        
        # Calculate capacity per expert
        capacity = max(1, math.ceil(self.capacity_factor * n_tokens_global / self.n_routed))
        
        # Use scratch buffer for outputs
        local_output = torch.zeros_like(recv_tensor)
        
        # Process each segment
        for expert_global, segment_start, segment_length in zip(unique_experts, segment_starts, segment_lengths):
            # Check if this expert is local to this rank
            if not (self.local_expert_start <= expert_global < self.local_expert_end):
                continue
                
            local_expert_idx = expert_global - self.local_expert_start
            segment_end = segment_start + segment_length
            
            # Get the slice for this expert
            segment_slice = slice(segment_start, segment_end)
            segment_gates = recv_gates_typed[segment_slice]
            segment_tokens = recv_tensor[segment_slice]
            segment_indices = torch.arange(segment_start, segment_end, device=recv_tensor.device)
            
            # Apply capacity limiting
            if segment_length > capacity:
                if self.capacity_policy == 'topk':
                    # Select tokens with highest gate values (deterministic)
                    _, topk_indices = torch.topk(segment_gates, capacity, largest=True)
                    selected_indices = segment_indices[topk_indices]
                elif self.capacity_policy == 'random':
                    # Random selection (non-deterministic)
                    perm = torch.randperm(segment_length, device=segment_gates.device)
                    selected_indices = segment_indices[perm[:capacity]]
                else:  # 'drop' - use first `capacity` tokens after sorting
                    selected_indices = segment_indices[:capacity]
            else:
                selected_indices = segment_indices
            
            # Process through local expert
            expert_output = self.local_experts[local_expert_idx](recv_tensor[selected_indices])
            weighted_output = expert_output * segment_gates[selected_indices].unsqueeze(1)
            
            local_output[selected_indices] = weighted_output
        
        return local_output

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with production-ready expert parallelism"""
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # (B*T, C)
        n_tokens = x_flat.shape[0]

        # Single GPU fallback - no communication needed
        if self.single_gpu_fallback:
            return self._forward_single_gpu(x_flat, B, T, C)

        # ___________ ROUTED EXPERT PATH (expert parallel) ___________
        
        # Step 1: Compute router logits on all GPUs
        router_logits = self.gate(x_flat)  # (B*T, n_routed)
        
        # Router computation
        if self.config.aux_free:
            biased_router_logits = router_logits + self.expert_bias
            topk_biased_logits, topk_indices = torch.topk(biased_router_logits, self.n_act_routed, dim=1, largest=True)
            topk_original_logits = torch.gather(router_logits, 1, topk_indices)
            topk_gates = F.softmax(topk_original_logits, dim=1)
        else:
            router_probs = F.softmax(router_logits, dim=1)
            topk_logits, topk_indices = torch.topk(router_logits, self.n_act_routed, dim=1, largest=True)
            topk_gates = F.softmax(topk_logits, dim=1)

        # Step 2: Build send buffers with vectorized implementation
        send_counts, send_tensor, send_gates, send_metadata = self._build_send_buffers_vectorized(
            x_flat, topk_indices, topk_gates
        )

        # Step 3: Exchange token counts
        recv_counts = torch.zeros_like(send_counts)
        torch.distributed.all_to_all_single(recv_counts, send_counts, group=self.ep_group)

        # Step 4: Prepare buffers (use empty to avoid memset)
        total_send = send_counts.sum().item() if send_tensor is not None else 0
        total_recv = recv_counts.sum().item()
        
        if total_send > 0:
            # We already have contiguous buffers from _build_send_buffers_vectorized
            pass
        else:
            # Handle case where no tokens to send
            send_tensor = torch.empty((0, C), device=x_flat.device, dtype=x_flat.dtype)
            send_gates = torch.empty((0,), device=x_flat.device, dtype=torch.float32)
            send_metadata = torch.empty((0, 2), device=x_flat.device, dtype=torch.int64)

        # Prepare receive buffers (use empty for tensors that will be overwritten)
        recv_tensor = torch.empty((total_recv, C), device=x_flat.device, dtype=x_flat.dtype)
        recv_gates = torch.empty((total_recv,), device=x_flat.device, dtype=torch.float32)
        recv_metadata = torch.empty((total_recv, 2), device=x_flat.device, dtype=torch.int64)

        # Step 5: Asynchronous all-to-all communication with compute overlap
        # Start communication
        work_tensor = torch.distributed.all_to_all_single(
            recv_tensor, send_tensor,
            output_split_sizes=recv_counts.tolist() if not self.single_gpu_fallback else None,
            input_split_sizes=send_counts.tolist() if not self.single_gpu_fallback else None,
            group=self.ep_group,
            async_op=True
        )
        
        work_gates = torch.distributed.all_to_all_single(
            recv_gates, send_gates,
            output_split_sizes=recv_counts.tolist() if not self.single_gpu_fallback else None,
            input_split_sizes=send_counts.tolist() if not self.single_gpu_fallback else None,
            group=self.ep_group,
            async_op=True
        )
        
        work_metadata = torch.distributed.all_to_all_single(
            recv_metadata, send_metadata,
            output_split_sizes=recv_counts.tolist() if not self.single_gpu_fallback else None,
            input_split_sizes=send_counts.tolist() if not self.single_gpu_fallback else None,
            group=self.ep_group,
            async_op=True
        )

        # OVERLAP: Compute shared experts while communication is in flight
        shared_output = torch.zeros_like(x_flat)
        if self.n_shared > 0:
            for i in range(self.n_shared):
                shared_output += self.shared_experts[i](x_flat)
            shared_output /= self.n_shared

        # Wait for all communication to complete
        work_tensor.wait()
        work_gates.wait()
        work_metadata.wait()

        # Debug assertions
        if self.debug_asserts:
            assert recv_tensor.size(0) == total_recv, f"Recv tensor size mismatch: {recv_tensor.size(0)} != {total_recv}"
            assert recv_gates.size(0) == total_recv, f"Recv gates size mismatch: {recv_gates.size(0)} != {total_recv}"
            assert recv_metadata.size(0) == total_recv, f"Recv metadata size mismatch: {recv_metadata.size(0)} != {total_recv}"

        # Step 6: Get global token count for capacity calculation
        n_tokens_local = torch.tensor([n_tokens], dtype=torch.float32, device=x_flat.device)
        torch.distributed.all_reduce(n_tokens_local, op=torch.distributed.ReduceOp.SUM, group=self.ep_group)
        n_tokens_global = n_tokens_local.item()

        # Step 7: Process local experts with segmented batching
        local_output = self._process_local_experts_segmented(recv_tensor, recv_gates, recv_metadata, n_tokens_global)

        # Step 8: All-to-all communication of results back (ALWAYS CALL)
        send_back_output = torch.empty_like(send_tensor)
        
        # Reverse all-to-all - ALWAYS CALL EVEN WITH EMPTY TENSORS
        torch.distributed.all_to_all_single(
            send_back_output, local_output,
            output_split_sizes=send_counts.tolist() if not self.single_gpu_fallback else None,
            input_split_sizes=recv_counts.tolist() if not self.single_gpu_fallback else None,
            group=self.ep_group
        )

        # Debug assertions for reverse pass
        if self.debug_asserts:
            assert send_back_output.size(0) == total_send, f"Send back size mismatch: {send_back_output.size(0)} != {total_send}"
            assert send_back_output.size(0) == send_metadata.size(0), "Send back and metadata size mismatch"

        # Step 9: Reconstruct final output using metadata
        routed_output = torch.zeros_like(x_flat)
        
        if total_send > 0:
            # Use origin indices from original metadata to scatter back
            origin_indices = send_metadata[:, 0].long()
            routed_output.index_add_(0, origin_indices, send_back_output)

        # Step 10: Combine shared and routed outputs
        y = (shared_output + routed_output).view(B, T, C)
        
        # Step 11: Calculate auxiliary loss with pre-computed global token count
        aux_loss = self._calculate_aux_loss(router_logits, topk_indices, n_tokens_global)

        return y, aux_loss

    def _forward_single_gpu(self, x_flat, B, T, C):
        """Single GPU fallback - no communication needed"""
        # Shared experts (same computation pattern as distributed path)
        shared_output = torch.zeros_like(x_flat)
        if self.n_shared > 0:
            for i in range(self.n_shared):
                shared_output += self.shared_experts[i](x_flat)
            shared_output /= self.n_shared

        # Local expert computation (all experts are local on single GPU)
        router_logits = self.gate(x_flat)
        
        if self.config.aux_free:
            biased_router_logits = router_logits + self.expert_bias
            topk_biased_logits, topk_indices = torch.topk(biased_router_logits, self.n_act_routed, dim=1, largest=True)
            topk_original_logits = torch.gather(router_logits, 1, topk_indices)
            topk_gates = F.softmax(topk_original_logits, dim=1)
        else:
            router_probs = F.softmax(router_logits, dim=1)
            topk_logits, topk_indices = torch.topk(router_logits, self.n_act_routed, dim=1, largest=True)
            topk_gates = F.softmax(topk_logits, dim=1)

        # Calculate auxiliary loss
        aux_loss = self._calculate_aux_loss_single_gpu(router_logits, topk_indices, x_flat.shape[0])

        # Local expert computation
        routed_output = torch.zeros_like(x_flat)
        for i in range(self.n_routed):
            expert_mask = (topk_indices == i).any(dim=1)
            if expert_mask.any():
                token_indices = expert_mask.nonzero(as_tuple=True)[0]
                slot_positions = (topk_indices[token_indices] == i).nonzero(as_tuple=True)[1]
                gates = topk_gates[token_indices].gather(1, slot_positions.unsqueeze(1)).squeeze(1)
                
                expert_output = self.local_experts[i](x_flat[token_indices])
                weighted_output = expert_output * gates.to(expert_output.dtype).unsqueeze(1)
                routed_output[token_indices] += weighted_output

        y = (shared_output + routed_output).view(B, T, C)
        return y, aux_loss

    def _calculate_aux_loss(self, router_logits, topk_indices, n_tokens_global):
        """Calculate auxiliary loss with pre-computed global token count"""
        if self.config.aux_free:
            # Calculate fi (expert utilization) with proper global reduction
            ones = torch.ones_like(topk_indices, dtype=torch.float32, device=router_logits.device)
            fi_counts = torch.zeros(self.n_routed, dtype=torch.float32, device=router_logits.device)
            fi_counts.scatter_add_(0, topk_indices.flatten(), ones.flatten())
            
            # Global reduction
            torch.distributed.all_reduce(fi_counts, op=torch.distributed.ReduceOp.SUM, group=self.ep_group)
            fi = fi_counts / n_tokens_global

            # Calculate pi (router probability) with global reduction
            router_probs = F.softmax(router_logits, dim=1).to(torch.float32)
            pi_sum = router_probs.sum(dim=0).detach()
            torch.distributed.all_reduce(pi_sum, op=torch.distributed.ReduceOp.SUM, group=self.ep_group)
            pi = pi_sum / n_tokens_global

            # Update bias (replicated on all GPUs)
            if self.training:
                with torch.no_grad():
                    ideal_load = 1.0 / self.n_routed
                    delta = ideal_load - fi
                    self.expert_bias.add_(self.config.gamma * delta.to(self.expert_bias.dtype))
                    # Clamp bias to prevent explosion
                    self.expert_bias.clamp_(self.bias_clamp_min, self.bias_clamp_max)

            aux_loss = self.config.alpha * self.n_routed * torch.sum(pi * fi)
        else:
            # Similar logic for non-aux-free case
            router_probs = F.softmax(router_logits, dim=1).to(torch.float32)
            pi_sum = router_probs.sum(dim=0).detach()
            torch.distributed.all_reduce(pi_sum, op=torch.distributed.ReduceOp.SUM, group=self.ep_group)
            
            ones = torch.ones_like(topk_indices, dtype=torch.float32, device=router_logits.device)
            fi_counts = torch.zeros(self.n_routed, dtype=torch.float32, device=router_logits.device)
            fi_counts.scatter_add_(0, topk_indices.flatten(), ones.flatten())
            torch.distributed.all_reduce(fi_counts, op=torch.distributed.ReduceOp.SUM, group=self.ep_group)
            
            pi = pi_sum / n_tokens_global
            fi = fi_counts / n_tokens_global
            
            aux_loss = self.config.coeff * self.n_routed * torch.sum(pi * fi)

        return aux_loss

    def _calculate_aux_loss_single_gpu(self, router_logits, topk_indices, n_tokens):
        """Single GPU version of auxiliary loss calculation"""
        n_tokens_global = n_tokens  # No reduction needed for single GPU
        
        if self.config.aux_free:
            ones = torch.ones_like(topk_indices, dtype=torch.float32, device=router_logits.device)
            fi_counts = torch.zeros(self.n_routed, dtype=torch.float32, device=router_logits.device)
            fi_counts.scatter_add_(0, topk_indices.flatten(), ones.flatten())
            fi = fi_counts / n_tokens_global

            router_probs = F.softmax(router_logits, dim=1).to(torch.float32)
            pi = router_probs.mean(dim=0)

            if self.training:
                with torch.no_grad():
                    ideal_load = 1.0 / self.n_routed
                    delta = ideal_load - fi
                    self.expert_bias.add_(self.config.gamma * delta.to(self.expert_bias.dtype))
                    self.expert_bias.clamp_(self.bias_clamp_min, self.bias_clamp_max)

            aux_loss = self.config.alpha * self.n_routed * torch.sum(pi * fi)
        else:
            router_probs = F.softmax(router_logits, dim=1).to(torch.float32)
            pi = router_probs.mean(dim=0)
            
            ones = torch.ones_like(topk_indices, dtype=torch.float32, device=router_logits.device)
            fi_counts = torch.zeros(self.n_routed, dtype=torch.float32, device=router_logits.device)
            fi_counts.scatter_add_(0, topk_indices.flatten(), ones.flatten())
            fi = fi_counts / n_tokens_global
            
            aux_loss = self.config.coeff * self.n_routed * torch.sum(pi * fi)

        return aux_loss

class LLM(nn.Module):
    """ A simple Large language model """
    def __init__(self, config:LLMconfig):
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
        if self.config.pos_emb == 'learn': active_params += self.pos_emb.weight.numel()
        active_params += self.transformer.ln_f.weight.numel() + self.transformer.ln_f.bias.numel()

        for block in self.transformer.h:
            active_params += sum(p.numel() for p in block.attn.parameters())   # ----|
            active_params += sum(p.numel() for p in block.ln1.parameters())    #     |---> Always active
            active_params += sum(p.numel() for p in block.ln2.parameters())    # ----|

            # if block.is_moe:

            #     active_params += sum(p.numel() for p in block.moe.gate.parameters())                # ----|
            #     for i in range(block.moe.n_shared):                                                 #     |---> Always active
            #         active_params += sum(p.numel() for p in block.moe.experts[i].parameters())      # ----|

            #     if block.moe.n_routed > 0:
            #         # Calculate params for one routed expert, multiply by the number of active ones
            #         params_per_routed_expert = sum(p.numel() for p in block.moe.experts[block.moe.n_shared].parameters())
            #         active_params += block.moe.n_act_routed * params_per_routed_expert
            
            # else: # In case a block is not MoE
            #     active_params += sum(p.numel() for p in block.mlp.parameters())
            # --- in LLM.get_num_params ---
            if block.is_moe:
                active_params += sum(p.numel() for p in block.moe.gate.parameters())
                for se in block.moe.shared_experts:
                    active_params += sum(p.numel() for p in se.parameters())
                if block.moe.num_local_experts > 0:
                    per_expert = sum(p.numel() for p in block.moe.local_experts[0].parameters())
                    active_params += block.moe.n_act_routed * per_expert
            else:
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



# --TRAINING SCRIPT--



assert torch.cuda.is_available()
assert torch.cuda.device_count() > 1

# ______________DEVICE, DTYPE, DDP SETUP_________________
import os
import argparse
import tiktoken
import requests
import numpy as np

from time import perf_counter

from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp.api  import ShardingStrategy, CPUOffload

init_process_group(backend='nccl')

rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
device = f"cuda:{local_rank}"
master_process = rank == 0
if master_process : print(f"Num GPUs = {world_size}")

torch.cuda.set_device(device)
torch.manual_seed(1729 + rank)         # offset the seed
torch.cuda.manual_seed(1729 + rank)    # offset the seed
torch.set_float32_matmul_precision('high') # Not sure if this has any effect when used with Auto Mixed Precision
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

dtype = 'float16' # if not torch.cuda.is_bf16_supported() else 'bfloat16'
torch_dtype = getattr(torch, dtype)
ctx = torch.amp.autocast(device_type="cuda", dtype=torch_dtype)
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

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

# from model import LLM

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

# @dataclass
# class LLMconfig:
#     # token params
#     vocab_size : int
#     block_size : int
#     n_embd : int
#     pos_emb : str | Literal['learn','sin','rope']

#     # Neural Network
#     up_dim  : int
#     non_linearity : str | Literal['elu','lrelu','relu', 'gelu', 'glu', 'swiglu', 'swish', 'mish', 'silu', 'selu','celu','tanh','sigmoid']
#     dropout : float
#     n_layer : int

#     # MoE
#     moe : bool

#     n_exp : int
#     n_shared : int  
#     n_act : int      ### INCLUDES THE SHARED EXPERTS
#     coeff : float

#     aux_free : bool
#     alpha : float   # complementry aux loss coeff
#     gamma: float    # bias update speed
    
#     # Attention
#     attn : str | Literal['mha', 'mqa', 'gqa', 'mla']
#     n_head : int
#     n_kv_heads : int 
#         # Only for mla 
#     q_latent_dim  : int | None
#     kv_latent_dim : int | None
#     rope_head_dim : int | None

#     act_recomp : bool

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
    
    act_recomp=TrainingConfig.act_recomp)    # Link the activation recomputation from the TRaining params           

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
    print('inside the func -> tokenize_and_save()')
    print('dataset has been download and saved to the files!')

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
    raw_model = model

# raw_model:LLM = model.module

#______________________________________________ TRAINING ______________________________________________

# optimizer = model.configure_optimizers(weight_decay=0.1,learning_rate=TrainingConfig.learning_rate,device=device)

# x,y = train_loader.next_batch() # get the first batch of training data
# train_loss_stats = []
# valrun_val_loss_stats = []
# valrun_train_loss_stats = []
# for iter in range(TrainingConfig.max_iters+1):
#     t0 = perf_counter()

#     lr = get_lr(iter, TrainingConfig) 
#     for param_grp in optimizer.param_groups:
#         param_grp['lr'] = lr
    
#     optimizer.zero_grad(set_to_none=True)
#     a,b = 0,0
#     if TrainingConfig.eval and (iter % TrainingConfig.eval_interval == 0 or iter == TrainingConfig.max_iters) and iter!=0:
#         a = perf_counter()
#         losses = estimate_loss(model, TrainingConfig, train_loader, val_loader)
#         valrun_val_loss_stats.append(losses['val'])
#         valrun_train_loss_stats.append(losses['train'])
#         b = perf_counter()
#         print(f"--------val run-------- train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | dt {1000*(b-a):.4f}ms")
#         t0 = b
    
#     for micro_step in range(grad_accum_steps):
#         with ctx:
#             _, loss, _ = model(x,y) #logits, loss, kv cache
#             loss:torch.Tensor = loss/grad_accum_steps

#         x,y = train_loader.next_batch() # Async prefetch the next batch of data
#         train_loss_stats.append(loss.item())
#         scaler.scale(loss).backward()

#     if TrainingConfig.grad_clip != 0.0:
#         scaler.unscale_(optimizer)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.grad_clip)

#     scaler.step(optimizer)
#     scaler.update()    
#     mem = 0
#     if "cuda" in device : 
#         torch.cuda.synchronize()
#         mem = torch.cuda.memory_reserved()
    
#     dt  = (perf_counter()-t0)*1000
#     print(f"step: {iter} | train loss:{loss*grad_accum_steps:.4f} | dt: {dt:.2f}ms | grad_accum_steps: {grad_accum_steps} | GPU RAM: {mem/1024**3:.2f}GB")

# if TrainingConfig.save_model:
#     # might do in-training checkpointing later
#     loss_stats = {'train':train_loss_stats, 'valrun_val':valrun_val_loss_stats, 'valrun_train':valrun_train_loss_stats}

#     checkpoint = {'model_config':ModelConfig, 'train_config':TrainingConfig, 'model_state': model.state_dict()}
#     stats      = {'model_config':ModelConfig, 'train_config':TrainingConfig, 'losses':loss_stats, 'total_params':total, 'active_params':active}

#     torch.save(checkpoint, TrainingConfig.file_name+'_ckpt.pt')
#     torch.save(stats, TrainingConfig.file_name+'_stats.pt')

#     print("Model and config saved to {}.pt".format(TrainingConfig.file_name + '_ckpt'))
#     print("Stats and config saved to {}.pt".format(TrainingConfig.file_name + '_stats'))


#______________________________________________ TRAINING ______________________________________________

optimizer = raw_model.configure_optimizers(weight_decay=0.1,learning_rate=TrainingConfig.learning_rate,device=device)
x,y = train_loader.next_batch()

print('before for loop -- in training code')
for iter in range(TrainingConfig.max_iters+1):
    t0 = perf_counter()

    lr = get_lr(iter, TrainingConfig)
    for param_grp in optimizer.param_groups:
        param_grp['lr'] = lr
    
    optimizer.zero_grad(set_to_none=True)

    print('before if condition')
    if TrainingConfig.eval and (iter % TrainingConfig.eval_interval == 0 or iter == TrainingConfig.max_iters) and iter!=0:
        a = perf_counter()
        losses = estimate_loss(model, TrainingConfig, train_loader, val_loader)
        b = perf_counter()
        if master_process:
            print(f"--------val run-------- train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | dt {1000*(b-a):.4f}ms")
        t0 = b
    
    print('before for micro_step in range(grad_accum_steps)')
    for micro_step in range(grad_accum_steps):
        print('just inside for loop -- in the next line -- micro_step->',micro_step)
        model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        # sync_context = model.no_sync() if micro_step < (grad_accum_steps-1) else nullcontext()
        # with sync_context:
        print('before -> with ctx')
        with ctx:
            print('after with ctx')
            _, loss, _ = model(x,y)
            print('after loss:torch.Tensor = loss/grad_accum_steps')
            loss:torch.Tensor = loss/grad_accum_steps

        print('before scaler.scale(loss).backward')
        scaler.scale(loss).backward()

        print('before x,y = train_loader.next_batch')
        x,y = train_loader.next_batch() # async pre-load next batch
        # scaler.scale(loss).backward()

    # if TrainingConfig.grad_clip != 0.0:
    #     scaler.unscale_(optimizer)
    #     model.clip_grad_norm_(TrainingConfig.grad_clip)

    # scaler.step(optimizer)
    # scaler.update()  
    # Now once per step (after all micro-steps):
        scaler.unscale_(optimizer)
        sync_moe_gradients(model)  # Average shared-expert & gate grads
        print('before if TrainingConfig.grad_clip > 0...')
        print('TrainingConfig.grad_clip->',TrainingConfig.grad_clip)
        if TrainingConfig.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.grad_clip)
        print('before scaler.step(optimizer)')
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)  

    if master_process:
        torch.cuda.synchronize()
        mem = torch.cuda.memory_reserved()
        dt  = (perf_counter()-t0)*1000
        print(f"step: {iter} | train loss:{loss.item()*grad_accum_steps:.4f} | dt: {dt:.2f}ms | grad_accum_steps: {grad_accum_steps} | GPU RAM: {mem/1024**3:.2f}GB")

destroy_process_group()

if TrainingConfig.save_model and master_process and False:
    # might do in-training checkpointing later by defining a save_model() function
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