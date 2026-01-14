'''
This script builds an LLM model based on the user's CLI inputs.

This script is meant for a demo multi-GPU run, perhaphs on Kaggle which provides free access to 2 GPUs.
eg: !torchrun --standalone --nproc_per_node=2 kaggle-train.py --moe --aux_free --eval --max_iters=250 --eval_interval=50

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
import warnings; warnings.filterwarnings('ignore')
import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from typing import Literal
from dataclasses import dataclass 

import os
os.environ['WANDB_API_KEY'] = 'c78410b3a816898642987ae3c3899430080b89d1'

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

class MoE(nn.Module):
    '''
    This class implements the DeepSeekMoE layer, featuring shared and routed experts.
    It uses an Auxiliary-Loss-Free load balancing strategy with a dynamic bias term.
    Ref: https://arxiv.org/pdf/2412.19437
    '''

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

class Block(nn.Module):
    """ A single Transformer block combining attention and MLP. """
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

        # NO checkpointing the MoE/MLP part -> memory grows O(T^2) for attn, O(T) for MoE, +scary looking error when we add MoE in checkpoint  
        if self.is_moe: 
            moe_output, aux_loss = self.moe(self.ln2(x))
            x = x + moe_output
        else:
            aux_loss = 0.0
            x = x + self.mlp(self.ln2(x))

        return x, updated_kv_cache, aux_loss

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
        self.print_act_recomp=config.act_recomp
        self.print_fused_adamw='fused' in inspect.signature(torch.optim.AdamW).parameters

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

            if block.is_moe:

                active_params += sum(p.numel() for p in block.moe.gate.parameters())                # ----|
                for i in range(block.moe.n_shared):                                                 #     |---> Always active
                    active_params += sum(p.numel() for p in block.moe.experts[i].parameters())      # ----|

                if block.moe.n_routed > 0:
                    # Calculate params for one routed expert, multiply by the number of active ones
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

        # Create AdamW optimizer and use the fused version if it is available
        try:
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, fused=True)
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
            x, updated_kv_cache, aux_loss = block(x, freqs_cis, kv_caches[i], self.VAL_RUN)            
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
 
### ----------- Training Script -----------

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

assert torch.cuda.is_available()
assert torch.cuda.device_count() > 1

import torch.distributed as dist


# ______________DEVICE, DTYPE, DDP SETUP_________________

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
    total_batch_size = 2**13,
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
    n_embd = 768, 
    pos_emb = 'rope',
    
    # MoE
    moe = True,

    up_dim = 1536, 
    non_linearity = 'swiglu',  
    dropout=0.0,
    n_layer = 12,

    n_exp = 8,
    n_shared = 1,
    n_act = 4,        ### INCLUDES THE SHARED EXPERTS

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
    
    act_recomp=TrainingConfig.act_recomp)


import wandb

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
'''
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
'''

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
        self.device_type = 'cuda'

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
        x = torch.from_numpy(np.stack(x_list)).long()
        y = torch.from_numpy(np.stack(y_list)).long()

        # Move to device (with pinned memory if CUDA)
        if self.device_type == 'cuda':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y



'''
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
assert total_batch_size % (B * T *world_size) == 0, "make sure total_batch_size is divisible by B * T * world_size"
grad_accum_steps = total_batch_size // (B * T *world_size)

#___________CREATE YOUR MODEL_____________

fsdp_wrap_policy = ModuleWrapPolicy({Block})

mp_policy = MixedPrecision(
    param_dtype=torch_dtype,
    reduce_dtype=torch_dtype,
    buffer_dtype=torch_dtype,
)

model = LLM(ModelConfig).to(device)
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
x,y = train_loader.next_batch()

for iter in range(TrainingConfig.max_iters+1):
    t0 = perf_counter()

    lr = get_lr(iter, TrainingConfig)
    for param_grp in optimizer.param_groups:
        param_grp['lr'] = lr
    
    optimizer.zero_grad(set_to_none=True)

    if TrainingConfig.eval and (iter % TrainingConfig.eval_interval == 0 or iter == TrainingConfig.max_iters) and iter!=0:
        a = perf_counter()
        losses = estimate_loss(model, TrainingConfig, train_loader, val_loader)
        b = perf_counter()
        if master_process:
            print(f"--------val run-------- train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | dt {1000*(b-a):.4f}ms")
        t0 = b
    
    for micro_step in range(grad_accum_steps):
        model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        # sync_context = model.no_sync() if micro_step < (grad_accum_steps-1) else nullcontext()
        # with sync_context:
        with ctx:
            _, loss, _ = model(x,y)
            loss:torch.Tensor = loss/grad_accum_steps

        x,y = train_loader.next_batch() # async pre-load next batch
        scaler.scale(loss).backward()

    if TrainingConfig.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        model.clip_grad_norm_(TrainingConfig.grad_clip)

    scaler.step(optimizer)
    scaler.update()    

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

'''


import torch
import torch.profiler as profiler
import math
from time import perf_counter
from contextlib import nullcontext
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any
from enum import Enum

# ... [Your existing imports] ...

# ==================== PROFILER CONFIGURATION ====================

class ProfilerActivity(Enum):
    """Wrapper for profiler activities for easier configuration"""
    CPU = profiler.ProfilerActivity.CPU
    CUDA = profiler.ProfilerActivity.CUDA
    XPU = profiler.ProfilerActivity.XPU
    MTIA = profiler.ProfilerActivity.MTIA

@dataclass
class ProfilerConfig:
    """
    Comprehensive configuration for PyTorch Profiler with all features.
    """
    # Basic profiling settings
    enabled: bool = True
    wait: int = 1
    warmup: int = 1
    active: int = 3
    repeat: int = 1
    
    # Recording options
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = True
    with_flops: bool = True
    with_modules: bool = True
    
    # Activities to profile
    activities: List[ProfilerActivity] = field(default_factory=lambda: [
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA
    ])
    
    # Advanced profiling options
    use_kineto: bool = True
    record_concurrent: bool = True
    record_io: bool = False
    skip_first: int = 0
    profile_communication: bool = False
    profile_autograd: bool = False
    
    # Export options
    export_chrome_trace: bool = True
    export_memory_timeline: bool = True
    export_stacks: bool = True
    export_perfetto_trace: bool = False
    export_operator_stats: bool = True
    export_module_stats: bool = True
    
    # Output configuration
    output_dir: str = "./profiler_logs"
    trace_file_name: str = "pytorch_trace"
    detailed_report: bool = True
    
    # Internal fields (set programmatically)
    schedule: Any = None
    on_trace_ready: Optional[Callable] = None
    
    @property
    def use_cpu(self) -> bool:
        return ProfilerActivity.CPU in self.activities
    
    @property
    def use_cuda(self) -> bool:
        return ProfilerActivity.CUDA in self.activities
    
    def get_torch_activities(self) -> List[profiler.ProfilerActivity]:
        """Convert to PyTorch profiler activities"""
        torch_activities = []
        for activity in self.activities:
            torch_activities.append(activity.value)
        return torch_activities
    
    def create_schedule(self):
        """Create profiler schedule based on configuration"""
        if self.schedule is None:
            self.schedule = profiler.schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
                skip_first=self.skip_first
            )
        return self.schedule
    
    def get_profiler_kwargs(self) -> dict:
        """Get keyword arguments for torch.profiler.profile"""
        kwargs = {
            "activities": self.get_torch_activities(),
            "schedule": self.create_schedule(),
            "on_trace_ready": self.on_trace_ready,
            "record_shapes": self.record_shapes,
            "profile_memory": self.profile_memory,
            "with_stack": self.with_stack,
            "with_flops": self.with_flops,
            "with_modules": self.with_modules,
        }
        
        try:
            if hasattr(profiler, 'record_concurrent'):
                kwargs["record_concurrent"] = self.record_concurrent
            if hasattr(profiler, 'record_io'):
                kwargs["record_io"] = self.record_io
        except AttributeError:
            pass
        
        return kwargs

# Add profiler config to TrainingConfig
TrainingConfig.profiler_config = ProfilerConfig()

def get_event_time(event, time_type="cuda"):
    """Safely extract timing information from profiler events"""
    if time_type == "cuda":
        attrs = [f'{time_type}_time_total', f'{time_type}_time', f'{time_type}_total_time']
    else:
        attrs = [f'self_{time_type}_time_total', f'self_{time_type}_time', f'self_{time_type}_total_time']
    
    for attr in attrs:
        if hasattr(event, attr):
            return getattr(event, attr)
    return 0

# ==================== PROFILER HELPER FUNCTIONS ====================

def setup_profiler(config: ProfilerConfig) -> Optional[profiler.profile]:
    """Initialize and return profiler instance"""
    if not config.enabled:
        return None

    # Create output directory only on master process
    if master_process:
        try:
            os.makedirs(config.output_dir, exist_ok=True)
            print(f"Created profiler output directory: {config.output_dir}")
        except Exception as e:
            print(f"Warning: Failed to create profiler directory {config.output_dir}: {e}")
            # Disable profiler if directory creation fails
            config.enabled = False
            return None
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Create default trace handler
    def trace_handler(profiler_instance):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{config.trace_file_name}_{timestamp}"
        
        # # Export Chrome trace
        # if config.export_chrome_trace:
        #     trace_file = os.path.join(config.output_dir, f"{base_name}.json")
        #     profiler_instance.export_chrome_trace(trace_file)
        #     if master_process:
        #         print(f"Chrome trace saved to: {trace_file}")
        if config.export_chrome_trace:
            trace_file = os.path.join(config.output_dir, f"{base_name}.json")
            try:
                profiler_instance.export_chrome_trace(trace_file)
                if master_process:
                    print(f"Chrome trace saved to: {trace_file}")
            except Exception as e:
                if master_process:
                    print(f"Warning: Failed to export Chrome trace: {e}")
        
        # Export memory timeline
        # if config.export_memory_timeline and config.profile_memory:
        #     memory_file = os.path.join(config.output_dir, f"{base_name}_memory.html")
        #     device = "cuda" if torch.cuda.is_available() else "cpu"
        #     if hasattr(profiler_instance, 'export_memory_timeline'):
        #         profiler_instance.export_memory_timeline(memory_file, device=device)
        #         if master_process:
        #             print(f"Memory timeline saved to: {memory_file}")

        if config.export_memory_timeline and config.profile_memory:
            memory_file = os.path.join(config.output_dir, f"{base_name}_memory.html")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if hasattr(profiler_instance, 'export_memory_timeline'):
                # Check if there's actual memory data before exporting
                try:
                    # First check if memory timeline has data
                    if (hasattr(profiler_instance, 'mem_tl') and 
                        profiler_instance.mem_tl and 
                        hasattr(profiler_instance.mem_tl, 'times') and 
                        profiler_instance.mem_tl.times):
                        profiler_instance.export_memory_timeline(memory_file, device=device)
                        if master_process:
                            print(f"Memory timeline saved to: {memory_file}")
                    else:
                        if master_process:
                            print(f"No memory timeline data collected, skipping export")
                except Exception as e:
                    if master_process:
                        print(f"Warning: Failed to export memory timeline: {e}")
        
        # Generate detailed report
        if config.detailed_report:
            generate_detailed_report(profiler_instance, config, base_name)
    
    config.on_trace_ready = trace_handler
    
    # Create and start profiler
    prof = profiler.profile(**config.get_profiler_kwargs())
    prof.start()
    
    if master_process:
        print("\n" + "="*60)
        print("PYTORCH PROFILER INITIALIZED")
        print("="*60)
        print(f"Schedule: wait={config.wait}, warmup={config.warmup}, active={config.active}, repeat={config.repeat}")
        print(f"Output directory: {config.output_dir}")
        print("="*60 + "\n")
    
    return prof

def generate_detailed_report(profiler_instance, config: ProfilerConfig, base_name: str):
    """Generate detailed analysis reports"""
    import json
    
    # Key averages table
    # sort_by = "self_cuda_time_total" if config.use_cuda else "self_cpu_time_total"
    # FIXED CODE (safer approach):
    try:
        # Try to get available attributes
        if config.use_cuda:
            sort_by = "self_cuda_time_total"
        else:
            sort_by = "self_cpu_time_total"
    except:
        sort_by = "cpu_time_total"  # Fallback


    table = profiler_instance.key_averages().table(
        sort_by=sort_by,
        row_limit=50,
        top_level_events_only=False
    )
    
    table_file = os.path.join(config.output_dir, f"{base_name}_table.txt")
    with open(table_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("PYTORCH PROFILER DETAILED REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(table)
    
    # Operator statistics
    if config.export_operator_stats:
        key_avgs = profiler_instance.key_averages(group_by_input_shape=True, group_by_stack_n=5)
        op_stats = []
        for avg in key_avgs:
            if avg.count > 0:
                op_stats.append({
                    "name": avg.key,
                    "count": avg.count,
                    # "cpu_time_us": avg.cpu_time_total,
                    "cpu_time_us": getattr(avg, 'cpu_time_total', 
                      getattr(avg, 'cpu_time', 
                             getattr(avg, 'cpu_total_time', 0))),
                    # "cuda_time_us": avg.cuda_time_total,
                    "cuda_time_us": getattr(avg, 'cuda_time_total', 
                       getattr(avg, 'cuda_time', 
                              getattr(avg, 'cuda_total_time', 0))),
                    # "self_cpu_time_us": avg.self_cpu_time_total,
                    "self_cpu_time_us": getattr(avg, 'self_cpu_time_total', 
                           getattr(avg, 'self_cpu_time', 
                                  getattr(avg, 'self_cpu_total_time', 0))),
                    # "self_cuda_time_us": avg.self_cuda_time_total,
                    "self_cuda_time_us": getattr(avg, 'self_cuda_time_total', 
                            getattr(avg, 'self_cuda_time', 
                                   getattr(avg, 'self_cuda_total_time', 0))),
                    "input_shapes": str(avg.input_shapes),
                    "flops": getattr(avg, 'flops', 0) if config.with_flops else 0,
                    "cuda_time_us": get_event_time(avg, "cuda"),
                    "self_cuda_time_us": get_event_time(avg, "self_cuda"),
                    "cpu_time_us": get_event_time(avg, "cpu"),
                    "self_cpu_time_us": get_event_time(avg, "self_cpu"),
                })
        
        op_file = os.path.join(config.output_dir, f"{base_name}_operators.json")
        with open(op_file, "w") as f:
            json.dump(op_stats, f, indent=2, default=str)
    
    # Module statistics
    if config.export_module_stats and config.with_modules:
        events_by_module = {}
        for event in profiler_instance.events():
            module_name = event.name.split("::")[0] if "::" in event.name else "Other"
            if module_name not in events_by_module:
                events_by_module[module_name] = {
                    "total_cuda_time_us": 0,
                    "total_cpu_time_us": 0,
                    "count": 0
                }
            events_by_module[module_name]["total_cuda_time_us"] += event.self_cuda_time_total
            events_by_module[module_name]["total_cpu_time_us"] += event.self_cpu_time_total
            events_by_module[module_name]["count"] += 1
        
        module_file = os.path.join(config.output_dir, f"{base_name}_modules.json")
        with open(module_file, "w") as f:
            json.dump(events_by_module, f, indent=2)
    
    if master_process:
        print(f"Detailed report saved with base name: {base_name}")

# ==================== MODIFIED TRAINING CODE ====================

train_loader = DataLoader(B=TrainingConfig.batch_size, T=ModelConfig.block_size, file_path="train.bin", device=device)
val_loader = DataLoader(B=TrainingConfig.batch_size, T=ModelConfig.block_size, file_path="val.bin", device=device)

# ____________ UTIL FUNCTIONS _________________

def get_lr(iter, TrainingConfig:TrainingConfig):
    max_lr = TrainingConfig.learning_rate
    min_lr = max_lr*0.1
    max_decay_steps = TrainingConfig.max_iters + 2 # avoid division by zero
    # 1) linear warmup for warmup_steps:
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
assert total_batch_size % (B * T * world_size) == 0, "make sure total_batch_size is divisible by B * T * world_size"
grad_accum_steps = total_batch_size // (B * T * world_size)

#___________CREATE YOUR MODEL_____________




'''
fsdp_wrap_policy = ModuleWrapPolicy({Block})

mp_policy = MixedPrecision(
    param_dtype=torch_dtype,
    reduce_dtype=torch_dtype,
    buffer_dtype=torch_dtype,
)

model = LLM(ModelConfig).to(device)
if master_process : 
    total, active = model.get_num_params()
    print(f"total parameters = {total:,}, active parameters = {active:,}")
    if model.print_fused_adamw: print("Using Fused AdamW")
    if model.print_act_recomp: print("Using Activation Recomputation")

model = FSDP(
    model,
    auto_wrap_policy=fsdp_wrap_policy,
    mixed_precision=mp_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD, # This is ZeRO-3
    device_id=torch.cuda.current_device(),
    limit_all_gathers=True, # Recommended for performance
    use_orig_params=True, # Important for optimizers like AdamW and for getting original parameters
    sync_module_states=True,
)

if master_process : print("Using compiled model")
model = torch.compile(model)

raw_model:LLM = model.module

#___________SETUP PROFILER______________

profiler_instance = setup_profiler(TrainingConfig.profiler_config)

#______________________________________________ TRAINING ______________________________________________

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=TrainingConfig.learning_rate, device=device)
x,y = train_loader.next_batch()

# Training metrics tracking
train_losses = []
train_times = []
val_losses = []
val_times = []

for iter in range(TrainingConfig.max_iters + 1):
    t0 = perf_counter()
    
    # Learning rate scheduling
    lr = get_lr(iter, TrainingConfig)
    for param_grp in optimizer.param_groups:
        param_grp['lr'] = lr
    
    optimizer.zero_grad(set_to_none=True)
    
    # Evaluation phase
    if TrainingConfig.eval and (iter % TrainingConfig.eval_interval == 0 or iter == TrainingConfig.max_iters) and iter != 0:
        eval_start = perf_counter()
        losses = estimate_loss(model, TrainingConfig, train_loader, val_loader)
        eval_time = (perf_counter() - eval_start) * 1000
        
        if master_process:
            print(f"\n{'='*50}")
            print(f"EVALUATION at iter {iter}")
            print(f"{'='*50}")
            print(f"Train loss: {losses['train']:.4f}")
            print(f"Val loss: {losses['val']:.4f}")
            print(f"Evaluation time: {eval_time:.2f}ms")
            print(f"{'='*50}\n")
        
        val_losses.append((iter, losses['val'].item()))
        val_times.append(eval_time)
        t0 = perf_counter()
    
    # Training phase with profiler integration
    for micro_step in range(grad_accum_steps):
        model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        
        # Profile forward pass if profiler is active
        if TrainingConfig.profiler_config.enabled and profiler_instance is not None:
            with profiler.record_function(f"FORWARD_PASS_microstep_{micro_step}"):
                with ctx:
                    _, loss, _ = model(x, y)
                    loss = loss / grad_accum_steps
        else:
            with ctx:
                _, loss, _ = model(x, y)
                loss = loss / grad_accum_steps
        
        # Async load next batch
        x, y = train_loader.next_batch()
        
        # Profile backward pass if profiler is active
        if TrainingConfig.profiler_config.enabled and profiler_instance is not None:
            with profiler.record_function(f"BACKWARD_PASS_microstep_{micro_step}"):
                scaler.scale(loss).backward()
        else:
            scaler.scale(loss).backward()
    
    # Gradient clipping with profiling
    if TrainingConfig.grad_clip != 0.0:
        if TrainingConfig.profiler_config.enabled and profiler_instance is not None:
            with profiler.record_function("GRADIENT_CLIPPING"):
                scaler.unscale_(optimizer)
                model.clip_grad_norm_(TrainingConfig.grad_clip)
        else:
            scaler.unscale_(optimizer)
            model.clip_grad_norm_(TrainingConfig.grad_clip)
    
    # Optimizer step with profiling
    if TrainingConfig.profiler_config.enabled and profiler_instance is not None:
        with profiler.record_function("OPTIMIZER_STEP"):
            scaler.step(optimizer)
            scaler.update()
    else:
        scaler.step(optimizer)
        scaler.update()
    
    # Step the profiler
    if TrainingConfig.profiler_config.enabled and profiler_instance is not None:
        profiler_instance.step()
    
    # Logging and metrics
    if master_process:
        torch.cuda.synchronize()
        dt = (perf_counter() - t0) * 1000
        
        # Memory statistics
        mem_reserved = torch.cuda.memory_reserved() / 1024**3
        mem_allocated = torch.cuda.memory_allocated() / 1024**3
        max_mem_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        # Training metrics
        train_loss = loss.item() * grad_accum_steps
        train_losses.append(train_loss)
        train_times.append(dt)
        
        print(f"\n[Iter {iter}]")
        print(f"  Loss: {train_loss:.4f} | LR: {lr:.6f}")
        print(f"  Time: {dt:.2f}ms | Grad accum steps: {grad_accum_steps}")
        print(f"  Memory: {mem_allocated:.2f}GB alloc | {mem_reserved:.2f}GB res | {max_mem_allocated:.2f}GB max")
        
        # Periodic detailed logging
        if iter % 100 == 0 and TrainingConfig.profiler_config.enabled:
            print(f"  Profiler active: cycle {profiler_instance.step_num if hasattr(profiler_instance, 'step_num') else 'N/A'}")
        
        # Save checkpoint if needed
        if TrainingConfig.save_model and iter % TrainingConfig.checkpoint_interval == 0 and iter != 0:
            save_checkpoint(model, TrainingConfig, iter, train_losses, val_losses)

# Stop profiler if active
if TrainingConfig.profiler_config.enabled and profiler_instance is not None:
    profiler_instance.stop()
    if master_process:
        print("\n" + "="*60)
        print("PROFILING COMPLETED")
        print("="*60)
        print(f"Profiler outputs saved to: {TrainingConfig.profiler_config.output_dir}")
        print("="*60 + "\n")

# Cleanup
destroy_process_group()

# ==================== CHECKPOINT FUNCTIONS ====================

def save_checkpoint(model, config: TrainConfig, iteration: int, train_losses, val_losses):
    """Save model checkpoint with profiler metadata"""
    if not master_process:
        return
    
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_iter_{iteration}_{timestamp}.pt")
    
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state_dict = model.state_dict()
    
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': cpu_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_times': train_times,
        'val_times': val_times,
        'model_config': ModelConfig,
        'train_config': TrainingConfig,
        'profiler_config': TrainingConfig.profiler_config.__dict__,
        'timestamp': timestamp
    }
    
    torch.save(checkpoint, checkpoint_file)
    print(f"Checkpoint saved to: {checkpoint_file}")
    
    # Also save a summary
    summary_file = os.path.join(checkpoint_dir, f"summary_iter_{iteration}_{timestamp}.txt")
    with open(summary_file, "w") as f:
        f.write(f"Checkpoint Summary - Iteration {iteration}\n")
        f.write("="*60 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Train loss: {train_losses[-1] if train_losses else 'N/A':.4f}\n")
        f.write(f"Average iteration time: {np.mean(train_times[-100:]) if len(train_times) > 100 else np.mean(train_times):.2f}ms\n")
        f.write(f"Memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f}GB\n")
        f.write(f"Profiler enabled: {TrainingConfig.profiler_config.enabled}\n")
        if TrainingConfig.profiler_config.enabled:
            f.write(f"Profiler output dir: {TrainingConfig.profiler_config.output_dir}\n")
        f.write("="*60 + "\n")

# ==================== POST-TRAINING SUMMARY ====================

if master_process:
    print("\n" + "="*80)
    print("TRAINING COMPLETED - SUMMARY")
    print("="*80)
    
    if train_losses:
        print(f"Final training loss: {train_losses[-1]:.4f}")
        if len(train_losses) > 10:
            avg_last_10 = np.mean(train_losses[-10:])
            print(f"Average of last 10 losses: {avg_last_10:.4f}")
    
    if train_times:
        avg_time = np.mean(train_times)
        std_time = np.std(train_times)
        print(f"Average iteration time: {avg_time:.2f}ms (±{std_time:.2f}ms)")
    
    if val_losses:
        print(f"Final validation loss: {val_losses[-1][1] if val_losses else 'N/A':.4f}")
    
    # Memory summary
    if torch.cuda.is_available():
        print(f"\nGPU Memory Summary:")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
        print(f"  Current allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        print(f"  Current reserved: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
    
    # Profiler summary
    if TrainingConfig.profiler_config.enabled:
        print(f"\nProfiler Summary:")
        print(f"  Outputs saved to: {TrainingConfig.profiler_config.output_dir}")
        print(f"  Available reports:")
        if os.path.exists(TrainingConfig.profiler_config.output_dir):
            files = os.listdir(TrainingConfig.profiler_config.output_dir)
            trace_files = [f for f in files if f.endswith('.json')]
            report_files = [f for f in files if f.endswith('.txt')]
            print(f"    - Chrome traces: {len(trace_files)} files")
            print(f"    - Text reports: {len(report_files)} files")
    
    print("="*80)

# Optional: Save final model if configured
if TrainingConfig.save_model and master_process:
    save_checkpoint(model, TrainingConfig, TrainingConfig.max_iters, train_losses, val_losses)

'''



fsdp_wrap_policy = ModuleWrapPolicy({Block})

mp_policy = MixedPrecision(
    param_dtype=torch_dtype,
    reduce_dtype=torch_dtype,
    buffer_dtype=torch_dtype,
)

model = LLM(ModelConfig).to(device)
if master_process : 
    total, active = model.get_num_params()
    print(f"total parameters = {total:,}, active parameters = {active:,}")
    if model.print_fused_adamw: print("Using Fused AdamW")
    if model.print_act_recomp: print("Using Activation Recomputation")


use_wandb = not args.no_wandb and master_process
if use_wandb:
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
        }
    )

model = FSDP(
    model,
    auto_wrap_policy=fsdp_wrap_policy,
    mixed_precision=mp_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD, # This is ZeRO-3
    device_id=torch.cuda.current_device(),
    limit_all_gathers=True, # Recommended for performance
    use_orig_params=True, # Important for optimizers like AdamW and for getting original parameters
    sync_module_states=True,
)

if master_process : print("Using compiled model")
model = torch.compile(model)

raw_model:LLM = model.module

#___________SETUP PROFILER______________

profiler_instance = setup_profiler(TrainingConfig.profiler_config)

#___________MFU CALCULATION SETUP FOR A40______________



# Replace the MFU calculation setup section with:

#___________MFU CALCULATION SETUP FOR A40______________

def calculate_mfu(model_flops_per_iter, dt_ms, peak_flops_per_gpu, num_gpus=1):
    """
    Calculate Model FLOPs Utilization (MFU)
    
    Args:
        model_flops_per_iter: FLOPs performed by model per iteration
        dt_ms: Time taken for iteration in milliseconds
        peak_flops_per_gpu: Peak FLOPS of the GPU
        num_gpus: Number of GPUs used
    
    Returns:
        MFU as a percentage
    """
    dt_s = dt_ms / 1000.0  # Convert to seconds
    flops_achieved = model_flops_per_iter / dt_s
    flops_promised = peak_flops_per_gpu * num_gpus
    mfu = flops_achieved / flops_promised
    return mfu * 100

def estimate_model_flops(num_params, batch_size, seq_length):
    """
    More accurate transformer FLOPs estimation.
    Standard formula: 6 * N * B * S per forward pass
    Training (forward + backward): ~2 * 6 * N * B * S = 12 * N * B * S
    Including attention: additional ~4 * n_layer * B * S^2 * d_model
    
    For simplicity, we use the 12*N*B*S approximation.
    """
    # Forward pass FLOPs
    flops_per_token = 6 * num_params  # 6 * N per token
    flops_forward = flops_per_token * batch_size * seq_length
    
    # Training FLOPs (forward + backward ≈ 2x forward)
    flops_training = 2 * flops_forward
    
    return flops_training

# A40 Specifications
if torch_dtype == torch.float16 or torch_dtype == torch.bfloat16:
    PEAK_FLOPS_A40 = 149.7e12  # 149.7 TFLOPS for FP16/BF16
elif torch_dtype == torch.float32:
    if torch.backends.cuda.matmul.allow_tf32:
        PEAK_FLOPS_A40 = 149.7e12  # 149.7 TFLOPS for TF32
    else:
        PEAK_FLOPS_A40 = 37.4e12   # 37.4 TFLOPS for FP32
else:
    PEAK_FLOPS_A40 = 149.7e12  # Default to tensor core performance

# Calculate model FLOPs per iteration - ONLY INITIALIZE HERE, CALCULATE PROPERLY IN LOOP
if master_process:
    num_gpus = dist.get_world_size() if dist.is_initialized() else 1
    print(f"\n{'='*60}")
    print(f"MFU CALCULATION SETUP")
    print(f"{'='*60}")
    print(f"GPU: NVIDIA A40")
    print(f"Dtype: {torch_dtype}")
    print(f"Peak FLOPS per A40: {PEAK_FLOPS_A40/1e12:.1f} TFLOPS")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Total Peak FLOPS: {(PEAK_FLOPS_A40 * num_gpus)/1e12:.1f} TFLOPS")
    print(f"Batch size: {TrainingConfig.batch_size}, Seq length: {ModelConfig.block_size}")
    print(f"{'='*60}\n")



'''
def calculate_mfu(model_flops_per_iter, dt_ms, peak_flops_per_gpu, num_gpus=1):
    """
    Calculate Model FLOPs Utilization (MFU)
    
    Args:
        model_flops_per_iter: FLOPs performed by model per iteration
        dt_ms: Time taken for iteration in milliseconds
        peak_flops_per_gpu: Peak FLOPS of the GPU
        num_gpus: Number of GPUs used
    
    Returns:
        MFU as a percentage
    """
    dt_s = dt_ms / 1000.0  # Convert to seconds
    flops_achieved = model_flops_per_iter / dt_s
    flops_promised = peak_flops_per_gpu * num_gpus
    mfu = flops_achieved / flops_promised
    return mfu * 100




def estimate_model_flops(num_params, batch_size, seq_length, num_gpus):
    """
    Estimate FLOPs per training iteration for a transformer model.
    Uses the approximation: 6 * N * B * S per forward pass
    (where N = params, B = batch size, S = sequence length)
    
    Training does 3x forward pass FLOPs (forward + backward = 3x forward)
    """
    # Forward pass FLOPs
    flops_forward = 6 * num_params * batch_size * seq_length
    
    # Training FLOPs (forward + backward ≈ 3x forward)
    flops_per_iter = 3 * flops_forward
    
    # Adjust for gradient accumulation if needed
    # (this is per micro-batch, so multiply by grad_accum_steps for total)
    
    return flops_per_iter
'''
# A40 Specifications
# - Peak FP32: 37.4 TFLOPS
# - Peak TF32 (Tensor Core): 149.7 TFLOPS
# - Peak FP16/BF16 (Tensor Core): 149.7 TFLOPS

# Determine peak FLOPS based on dtype
# if torch_dtype == torch.float16 or torch_dtype == torch.bfloat16:
#     PEAK_FLOPS_A40 = 149.7e12  # 149.7 TFLOPS for FP16/BF16
# elif torch_dtype == torch.float32:
#     # Check if TF32 is enabled (default in PyTorch)
#     if torch.backends.cuda.matmul.allow_tf32:
#         PEAK_FLOPS_A40 = 149.7e12  # 149.7 TFLOPS for TF32
#     else:
#         PEAK_FLOPS_A40 = 37.4e12   # 37.4 TFLOPS for FP32
# else:
#     PEAK_FLOPS_A40 = 149.7e12  # Default to tensor core performance

# # Calculate model FLOPs per iteration
# if master_process:
#     num_gpus = dist.get_world_size() if dist.is_initialized() else 1
    
#     # Get actual batch size and sequence length from your config
#     batch_size = TrainingConfig.batch_size  # Total batch size
#     seq_length = ModelConfig.block_size  # Sequence length
    
#     # Calculate FLOPs per iteration (per gradient accumulation step)
#     model_flops_per_iter = estimate_model_flops(
#         active,  # Use active parameters
#         batch_size // (num_gpus * grad_accum_steps),  # Per-GPU micro-batch size
#         seq_length,
#     )   
    
#     print(f"\n{'='*60}")
#     print(f"MFU CALCULATION SETUP")
#     print(f"{'='*60}")
#     print(f"GPU: NVIDIA A40")
#     print(f"Dtype: {torch_dtype}")
#     print(f"Peak FLOPS per A40: {PEAK_FLOPS_A40/1e12:.1f} TFLOPS")
#     print(f"Number of GPUs: {num_gpus}")
#     print(f"Total Peak FLOPS: {(PEAK_FLOPS_A40 * num_gpus)/1e12:.1f} TFLOPS")
#     print(f"Model FLOPs per iteration: {model_flops_per_iter/1e12:.2f} TFLOPS")
#     print(f"Batch size: {batch_size}, Seq length: {seq_length}")
#     print(f"{'='*60}\n")


# Tesla T4 Specifications (YOUR GPU)
# T4 has 65 TFLOPS for FP16 with Tensor Cores
# 8.1 TFLOPS for FP32 without Tensor Cores

if torch_dtype == torch.float16 or torch_dtype == torch.bfloat16:
    PEAK_FLOPS_T4 = 65.0e12  # 65 TFLOPS for FP16/BF16 with Tensor Cores
elif torch_dtype == torch.float32:
    if torch.backends.cuda.matmul.allow_tf32:
        # T4 doesn't support TF32, so use FP32 performance
        PEAK_FLOPS_T4 = 8.1e12   # 8.1 TFLOPS for FP32
    else:
        PEAK_FLOPS_T4 = 8.1e12   # 8.1 TFLOPS for FP32
else:
    PEAK_FLOPS_T4 = 65.0e12  # Default to FP16 with Tensor Cores

PEAK_FLOPS_PER_GPU = PEAK_FLOPS_T4
NUM_GPUS = world_size

# Use the value from your dashboard for accurate MFU
PEAK_FLOPS_PER_T4 = 26.1e12  # 26.1 TFLOPS from your dashboard

if master_process:
    print(f"\n{'='*80}")
    print(f"TESLA T4 CLUSTER CONFIGURATION")
    print(f"{'='*80}")
    print(f"Number of Tesla T4 GPUs: {world_size}")
    print(f"Dashboard reported: 26.1 TFLOPS per GPU")
    print(f"Total theoretical peak: {PEAK_FLOPS_PER_T4 * world_size / 1e12:.1f} TFLOPS")
    print(f"Dtype being used: {dtype}")
    print(f"Tensor Core enabled: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"Note: MFU will use dashboard value for accurate comparison")
    print(f"{'='*80}\n")



#______________________________________________ TRAINING ______________________________________________

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=TrainingConfig.learning_rate, device=device)
x,y = train_loader.next_batch()

# Training metrics tracking
train_losses = []
train_times = []
val_losses = []
val_times = []
mfu_values = []

for iter in range(TrainingConfig.max_iters + 1):
    t0 = perf_counter()
    
    # Learning rate scheduling
    lr = get_lr(iter, TrainingConfig)
    for param_grp in optimizer.param_groups:
        param_grp['lr'] = lr
    
    optimizer.zero_grad(set_to_none=True)
    
    # Evaluation phase
    if TrainingConfig.eval and (iter % TrainingConfig.eval_interval == 0 or iter == TrainingConfig.max_iters) and iter != 0:
        eval_start = perf_counter()
        losses = estimate_loss(model, TrainingConfig, train_loader, val_loader)
        eval_time = (perf_counter() - eval_start) * 1000
        
        if master_process:
            print(f"\n{'='*50}")
            print(f"EVALUATION at iter {iter}")
            print(f"{'='*50}")
            print(f"Train loss: {losses['train']:.4f}")
            print(f"Val loss: {losses['val']:.4f}")
            print(f"Evaluation time: {eval_time:.2f}ms")
            if mfu_values:
                avg_mfu = np.mean(mfu_values[-100:]) if len(mfu_values) > 100 else np.mean(mfu_values)
                print(f"Average MFU: {avg_mfu:.2f}%")
            print(f"{'='*50}\n")
        
        val_losses.append((iter, losses['val'].item()))
        val_times.append(eval_time)
        t0 = perf_counter()
    
    # Training phase with profiler integration
    for micro_step in range(grad_accum_steps):
        model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        
        # Profile forward pass if profiler is active
        if TrainingConfig.profiler_config.enabled and profiler_instance is not None:
            with profiler.record_function(f"FORWARD_PASS_microstep_{micro_step}"):
                with ctx:
                    _, loss, _ = model(x, y)
                    loss = loss / grad_accum_steps
        else:
            with ctx:
                _, loss, _ = model(x, y)
                loss = loss / grad_accum_steps
        
        # Async load next batch
        x, y = train_loader.next_batch()
        
        # Profile backward pass if profiler is active
        if TrainingConfig.profiler_config.enabled and profiler_instance is not None:
            with profiler.record_function(f"BACKWARD_PASS_microstep_{micro_step}"):
                scaler.scale(loss).backward()
        else:
            scaler.scale(loss).backward()

        if use_wandb and master_process:
            wandb.log({
                "train/loss": loss.item() * grad_accum_steps,
                "train/lr": lr,
                "train/step": iter,
                "perf/iteration_time_ms": dt if 'dt' in locals() else 0,
                "perf/throughput_tokens_per_sec": (B * T * grad_accum_steps * world_size) / (dt / 1000) if 'dt' in locals() else 0,
                "memory/allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "memory/reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "memory/max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
                # "train/loss": current_loss,
                # "train/lr": lr,
                # "train/step": iter,
                # "perf/iteration_time_ms": dt,
                # "perf/throughput_tokens_per_sec": throughput,
                # "memory/allocated_gb": mem_gb,
            })

        # Evaluation logging
        if TrainingConfig.eval and (iter % TrainingConfig.eval_interval == 0) and iter != 0:
            if use_wandb and master_process:
                wandb.log({
                    "eval/train_loss": losses['train'].item(),
                    "eval/val_loss": losses['val'].item(),
                    "eval/step": iter,
                    # "perf/iteration_time_ms": dt if 'dt' in locals() else 0,
                    "perf/throughput_tokens_per_sec": (B * T * grad_accum_steps * world_size) / (dt / 1000) if 'dt' in locals() else 0,
                })
    
    # Gradient clipping with profiling
    if TrainingConfig.grad_clip != 0.0:
        if TrainingConfig.profiler_config.enabled and profiler_instance is not None:
            with profiler.record_function("GRADIENT_CLIPPING"):
                scaler.unscale_(optimizer)
                model.clip_grad_norm_(TrainingConfig.grad_clip)
        else:
            scaler.unscale_(optimizer)
            model.clip_grad_norm_(TrainingConfig.grad_clip)
    
    # Optimizer step with profiling
    if TrainingConfig.profiler_config.enabled and profiler_instance is not None:
        with profiler.record_function("OPTIMIZER_STEP"):
            scaler.step(optimizer)
            scaler.update()
    else:
        scaler.step(optimizer)
        scaler.update()
    
    # Step the profiler
    if TrainingConfig.profiler_config.enabled and profiler_instance is not None:
        profiler_instance.step()
    
    # Logging and metrics
        # Logging and metrics
    if master_process:
        torch.cuda.synchronize()
        dt = (perf_counter() - t0) * 1000
        
        # # CORRECTED: Calculate actual tokens processed
        # # Per GPU micro-batch size
        # micro_batch_size = TrainingConfig.batch_size  # This is per GPU batch size
        # seq_length = ModelConfig.block_size
        
        # # Total tokens processed in this iteration (across all gradient accumulation steps)
        # # Each micro-step processes B*T tokens, and we have grad_accum_steps micro-steps
        # total_tokens_processed = micro_batch_size * seq_length * grad_accum_steps
        
        # # CORRECTED: Calculate FLOPs for this iteration
        # # Use active parameters (sparse) for MoE models
        # model_flops_per_iter = estimate_model_flops(
        #     active,  # Use ACTIVE parameters (not total)
        #     total_tokens_processed,  # Total tokens in this iteration
        #     1  # Sequence length already accounted for in total_tokens
        # )
        
        # # Multiply by world_size to get total FLOPs across all GPUs
        # total_flops_per_iter = model_flops_per_iter * world_size
        
        # # Calculate throughput
        # tokens_per_iter = micro_batch_size * seq_length * grad_accum_steps * world_size
        # throughput = tokens_per_iter / (dt / 1000)  # tokens per second
        
        # # Calculate MFU - use total FLOPs across all GPUs
        # mfu = calculate_mfu(
        #     total_flops_per_iter,  # Total FLOPs for this iteration across all GPUs
        #     dt,  # Time for this iteration
        #     PEAK_FLOPS_A40,
        #     world_size  # Number of GPUs
        # )
        # mfu_values.append(mfu)
        
        # # Memory statistics
        # mem_reserved = torch.cuda.memory_reserved() / 1024**3
        # mem_allocated = torch.cuda.memory_allocated() / 1024**3
        # max_mem_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        # # Training metrics
        # train_loss = loss.item() * grad_accum_steps
        # train_losses.append(train_loss)
        # train_times.append(dt)
        
        # print(f"\n[Iter {iter}]")
        # print(f"  Loss: {train_loss:.4f} | LR: {lr:.6f}")
        # print(f"  Time: {dt:.2f}ms | MFU: {mfu:.2f}%")
        # print(f"  Throughput: {throughput:,.0f} tokens/sec")
        # print(f"  Tokens/iter: {tokens_per_iter:,} (per GPU: {micro_batch_size * seq_length * grad_accum_steps:,})")
        # print(f"  Grad accum steps: {grad_accum_steps}")
        # print(f"  Memory: {mem_allocated:.2f}GB alloc | {mem_reserved:.2f}GB res | {max_mem_allocated:.2f}GB max")

        dt_s = (perf_counter() - t0)  # Time in seconds
    
        # 2. Calculate tokens processed
        tokens_per_gpu = TrainingConfig.batch_size * ModelConfig.block_size * grad_accum_steps
        tokens_total = tokens_per_gpu * world_size
        
        # 3. Calculate FLOPs - USE ACTIVE PARAMS FOR MoE
        # Standard: 6 * params per token for forward, ~6 for backward = 12 * params
        # For MoE: Only active experts compute, so use ACTIVE parameters
        flops_per_token = 12 * active  # ACTIVE params, not total!
        
        # Total FLOPs this iteration
        total_flops = flops_per_token * tokens_total
        
        # 4. Calculate achieved FLOPS rate
        achieved_flops = total_flops / dt_s
        
        # 5. Calculate peak FLOPS (using dashboard value)
        peak_flops_per_gpu = 26.1e12  # From your dashboard
        peak_flops_total = peak_flops_per_gpu * world_size
        
        # 6. Calculate MFU
        mfu = (achieved_flops / peak_flops_total) * 100
        mfu_values.append(mfu)
        
        # 7. Calculate throughput
        throughput = tokens_total / dt_s  # tokens per second
        
        # =========================================
        
        # Training metrics
        train_loss = loss.item() * grad_accum_steps
        train_losses.append(train_loss)
        train_times.append(dt_s * 1000)  # Convert to ms
        
        if master_process:
            print(f"\n[Iter {iter}] - 4x Tesla T4")
            print(f"{'-'*50}")
            print(f"  Loss: {train_loss:.4f} | LR: {lr:.6f}")
            print(f"  Time: {dt_s*1000:.2f}ms | Throughput: {throughput:,.0f} tokens/sec")
            print(f"  Tokens: {tokens_total:,} (per GPU: {tokens_per_gpu:,})")
            print(f"  Grad accum steps: {grad_accum_steps}")
            print(f"\n  MFU CALCULATION:")
            print(f"    Active params: {active:,}")
            print(f"    FLOPs per token: {flops_per_token/1e9:.1f} GFLOPs")
            print(f"    Total FLOPs: {total_flops/1e12:.2f} TFLOPS")
            print(f"    Achieved FLOPS: {achieved_flops/1e12:.1f} TFLOPS")
            print(f"    Peak FLOPS (4x T4): {peak_flops_total/1e12:.1f} TFLOPS")
            print(f"    MFU: {mfu:.2f}%")
            print(f"{'-'*50}")

# Stop profiler if active
if TrainingConfig.profiler_config.enabled and profiler_instance is not None:
    profiler_instance.stop()
    if master_process:
        print("\n" + "="*60)
        print("PROFILING COMPLETED")
        print("="*60)
        print(f"Profiler outputs saved to: {TrainingConfig.profiler_config.output_dir}")
        print("="*60 + "\n")

# Cleanup
destroy_process_group()

# ==================== CHECKPOINT FUNCTIONS ====================

def save_checkpoint(model, config: Trainconfig, iteration: int, train_losses, val_losses, mfu_values=None):
    """Save model checkpoint with profiler and MFU metadata"""
    if not master_process:
        return
    
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_iter_{iteration}_{timestamp}.pt")
    
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state_dict = model.state_dict()
    
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': cpu_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_times': train_times,
        'val_times': val_times,
        'mfu_values': mfu_values,
        'model_config': ModelConfig,
        'train_config': TrainingConfig,
        'profiler_config': TrainingConfig.profiler_config.__dict__,
        'timestamp': timestamp
    }
    
    torch.save(checkpoint, checkpoint_file)
    print(f"Checkpoint saved to: {checkpoint_file}")
    
    # Also save a summary
    summary_file = os.path.join(checkpoint_dir, f"summary_iter_{iteration}_{timestamp}.txt")
    with open(summary_file, "w") as f:
        f.write(f"Checkpoint Summary - Iteration {iteration}\n")
        f.write("="*60 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Train loss: {train_losses[-1] if train_losses else 'N/A':.4f}\n")
        f.write(f"Average iteration time: {np.mean(train_times[-100:]) if len(train_times) > 100 else np.mean(train_times):.2f}ms\n")
        
        if mfu_values:
            avg_mfu = np.mean(mfu_values[-100:]) if len(mfu_values) > 100 else np.mean(mfu_values)
            f.write(f"Average MFU: {avg_mfu:.2f}%\n")
        
        f.write(f"Memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f}GB\n")
        f.write(f"Profiler enabled: {TrainingConfig.profiler_config.enabled}\n")
        if TrainingConfig.profiler_config.enabled:
            f.write(f"Profiler output dir: {TrainingConfig.profiler_config.output_dir}\n")
        f.write("="*60 + "\n")

# ==================== POST-TRAINING SUMMARY ====================

if master_process:
    print("\n" + "="*80)
    print("TRAINING COMPLETED - SUMMARY")
    print("="*80)
    
    if train_losses:
        print(f"Final training loss: {train_losses[-1]:.4f}")
        if len(train_losses) > 10:
            avg_last_10 = np.mean(train_losses[-10:])
            print(f"Average of last 10 losses: {avg_last_10:.4f}")
    
    if train_times:
        avg_time = np.mean(train_times)
        std_time = np.std(train_times)
        print(f"Average iteration time: {avg_time:.2f}ms (±{std_time:.2f}ms)")
    
    if val_losses:
        print(f"Final validation loss: {val_losses[-1][1] if val_losses else 'N/A':.4f}")
    
    # MFU Summary
    # if mfu_values:
    #     print(f"\nModel FLOPs Utilization (MFU) Summary:")
    #     print(f"  Average MFU: {np.mean(mfu_values):.2f}%")
    #     print(f"  Max MFU: {np.max(mfu_values):.2f}%")
    #     print(f"  Min MFU: {np.min(mfu_values):.2f}%")
    #     print(f"  Std Dev: {np.std(mfu_values):.2f}%")
    #     if len(mfu_values) > 100:
    #         print(f"  Average MFU (last 100 iters): {np.mean(mfu_values[-100:]):.2f}%")
    # In the post-training summary, update MFU reporting:
    if mfu_values:
        print(f"\nModel FLOPs Utilization (MFU) Summary:")
        print(f"  Average MFU: {np.mean(mfu_values):.2f}%")
        print(f"  Max MFU: {np.max(mfu_values):.2f}%")
        print(f"  Min MFU: {np.min(mfu_values):.2f}%")
        print(f"  Std Dev: {np.std(mfu_values):.2f}%")
        
        # Show detailed breakdown
        if len(mfu_values) > 20:
            first_10_avg = np.mean(mfu_values[:10])
            last_10_avg = np.mean(mfu_values[-10:])
            print(f"  First 10 iters avg MFU: {first_10_avg:.2f}%")
            print(f"  Last 10 iters avg MFU: {last_10_avg:.2f}%")
        
        # Calculate theoretical max
        print(f"\n  Theoretical Analysis:")
        print(f"  Model Active Params: {active:,}")
        print(f"  Peak FLOPS per A40: {PEAK_FLOPS_A40/1e12:.1f} TFLOPS")
    
    # Memory summary
    if torch.cuda.is_available():
        print(f"\nGPU Memory Summary:")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
        print(f"  Current allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        print(f"  Current reserved: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
    
    # Profiler summary
    if TrainingConfig.profiler_config.enabled:
        print(f"\nProfiler Summary:")
        print(f"  Outputs saved to: {TrainingConfig.profiler_config.output_dir}")
        print(f"  Available reports:")
        if os.path.exists(TrainingConfig.profiler_config.output_dir):
            files = os.listdir(TrainingConfig.profiler_config.output_dir)
            trace_files = [f for f in files if f.endswith('.json')]
            report_files = [f for f in files if f.endswith('.txt')]
            print(f"    - Chrome traces: {len(trace_files)} files")
            print(f"    - Text reports: {len(report_files)} files")
    
    print("="*80)

# Optional: Save final model if configured
if TrainingConfig.save_model and master_process:
    save_checkpoint(model, TrainingConfig, TrainingConfig.max_iters, train_losses, val_losses, mfu_values)


# Finish WandB run
if use_wandb and master_process:
    wandb.finish()
    print("WandB run completed")