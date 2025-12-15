
import warnings; warnings.filterwarnings('ignore')
import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from typing import Literal
from dataclasses import dataclass 

from torch.nn.parallel import DistributedDataParallel as DDP



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

    # Context parallelism splits the sequence dimension across multiple GPUs. 
    # Instead of each GPU processing the full sequence, each GPU processes a chunk of the sequence.
   
    context_parallel_size: int = 1  

         
    context_parallel_rank: int = 0

    
    context_parallel_group: bool = None


    @staticmethod
    def apply_rotary_emb(x:torch.Tensor, freqs_cis:torch.Tensor)->torch.Tensor:
        ''' Applies RoPE to either the query or the key whose embeddings are to be rotated two at a time.'''

        
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

        # Add context parallel group
        self.context_parallel_group = getattr(config, 'context_parallel_group', None)

    '''
    Concept: Split sequence dimension across GPUs, but each GPU needs global context for attention.

    Without Context Parallel:

        Single GPU: [Q0 Q1 Q2 Q3] × [K0 K1 K2 K3] → Full attention

    With Context Parallel (4 GPUs):

        GPU0: [Q0] × [K0 K1 K2 K3] → Output0
        GPU1: [Q1] × [K0 K1 K2 K3] → Output1  
        GPU2: [Q2] × [K0 K1 K2 K3] → Output2
        GPU3: [Q3] × [K0 K1 K2 K3] → Output3
    '''


    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor|None, kv_cache=None, VAL_RUN=False):


        '''
        Global Sequence: [Token0 Token1 Token2 Token3 Token4 Token5 Token6 Token7]
            GPU0 (rank=0): [Token0 Token1]           # T_local = 2
            GPU1 (rank=1): [Token2 Token3]           # T_local = 2  
            GPU2 (rank=2): [Token4 Token5]           # T_local = 2
            GPU3 (rank=3): [Token6 Token7]           # T_local = 2
        '''
        B, T_local, C = x.size()  # CHANGE: T_local instead of T

        nh, nkvh, hs = self.config.n_head , self.config.n_kv_heads, self.head_size

        q_proj_size = C
        kv_proj_size = nkvh * hs

        '''
        Each GPU computes locally:
            GPU0: Q0, K0, V0  (from Token0-1)
            GPU1: Q1, K1, V1  (from Token2-3)
            GPU2: Q2, K2, V2  (from Token4-5) 
            GPU3: Q3, K3, V3  (from Token6-7)
        '''
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


        '''
        KEY STEP: All-Gather K and V
            BEFORE all-gather:
                GPU0: K0, V0
                GPU1: K1, V1  
                GPU2: K2, V2
                GPU3: K3, V3

            AFTER all-gather:
                GPU0: [K0 K1 K2 K3], [V0 V1 V2 V3]
                GPU1: [K0 K1 K2 K3], [V0 V1 V2 V3]
                GPU2: [K0 K1 K2 K3], [V0 V1 V2 V3] 
                GPU3: [K0 K1 K2 K3], [V0 V1 V2 V3]
        '''
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

        '''
            In language modeling, we use causal masking to ensure each token can only attend to previous tokens (and itself), 
            not future tokens. This prevents the model from "cheating" by looking ahead.


            Standard Causal Mask (single GPU):
                Tokens:  T0   T1   T2   T3
                    T0 can see: [T0]
                    T1 can see: [T0, T1]  
                    T2 can see: [T0, T1, T2]
                    T3 can see: [T0, T1, T2, T3]

            The Problem in Context Parallel
            When we split the sequence across GPUs, each GPU only has local queries but global keys. 
            We need to ensure causal relationships are preserved across GPU boundaries.

            Example with 4 GPUs, 2 tokens each:
            Global Sequence: [T0 T1 T2 T3 T4 T5 T6 T7]
                GPU0: [T0 T1]  ← Queries Q0, Q1
                GPU1: [T2 T3]  ← Queries Q2, Q3  
                GPU2: [T4 T5]  ← Queries Q4, Q5
                GPU3: [T6 T7]  ← Queries Q6, Q7

                After all-gather, ALL GPUs have ALL Keys: [K0 K1 K2 K3 K4 K5 K6 K7]


            The Masking Challenge

                For GPU1 (handling tokens T2, T3):

                Q2 (token T2) should only attend to: T0, T1, T2 (NOT T3-T7)
                Q3 (token T3) should only attend to: T0, T1, T2, T3 (NOT T4-T7)
                But GPU1 has ALL keys! So we need to mask out the invalid ones.
        '''


        '''
            Mask Visualization for GPU1 (rank=1):
            Global positions:   0   1   2   3   4   5   6   7
            GPU1 Q positions:         2   3
            Causal Mask:
            Q2 can attend to: [0, 1, 2]        → Mask: [F, F, F, T, T, T, T, T]
            Q3 can attend to: [0, 1, 2, 3]     → Mask: [F, F, F, F, T, T, T, T]
        '''
        
        # Rectangular causal mask for local Q × global K
        T_global = k.size(-2)
        shard_start = self.config.context_parallel_rank * T_local
        q_pos = shard_start + torch.arange(T_local, device=x.device)
        k_pos = torch.arange(T_global, device=x.device)
        causal_mask = (k_pos.unsqueeze(0) > q_pos.unsqueeze(1))  # (T_local, T_global)

        '''
            Query Positions (q_pos): [2, 3]
            Key Positions   (k_pos): [0, 1, 2, 3, 4, 5, 6, 7]

            Mask Calculation:
            For Q2 (position 2): 
            Keys at [0,1,2] → False (can attend)  
            Keys at [3,4,5,6,7] → True (MASKED OUT) ✓

            For Q3 (position 3):
            Keys at [0,1,2,3] → False (can attend)
            Keys at [4,5,6,7] → True (MASKED OUT) ✓
        '''


        '''
        Visualization for GPU1:
            GPU1 Local Q: [Q2, Q3] 
            Global K:     [K0, K1, K2, K3, K4, K5, K6, K7]

            Attention Matrix for GPU1:
                K0  K1  K2  K3  K4  K5  K6  K7
            Q2:  ✓   ✓   ✓   ✗   ✗   ✗   ✗   ✗   (causal mask applied)
            Q3:  ✓   ✓   ✓   ✓   ✗   ✗   ✗   ✗
        '''
        
        # Dtype-safe masking
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(attn.dtype).min)

        attn = self.attn_dropout(F.softmax(attn, dim=-1))
        y = attn @ v
        
        # NO reduce_scatter - output is already local
        '''
        Each GPU produces its local output:
            GPU0: Output0 (from Token0-1)
            GPU1: Output1 (from Token2-3)  
            GPU2: Output2 (from Token4-5)
            GPU3: Output3 (from Token6-7)
        '''
        y = y.transpose(1, 2).contiguous().view(B, T_local, C)
        y = self.resid_dropout(self.c_proj(y))

        '''
        Complete Data Flow Visualization

            SEQUENCE DISTRIBUTION:
                Global: [T0 T1 T2 T3 T4 T5 T6 T7]
                GPU0:   [T0 T1] → Q0, K0, V0
                GPU1:   [T2 T3] → Q1, K1, V1  
                GPU2:   [T4 T5] → Q2, K2, V2
                GPU3:   [T6 T7] → Q3, K3, V3

                ALL-GATHER PHASE:
                Each GPU exchanges K,V → All get [K0 K1 K2 K3], [V0 V1 V2 V3]

                ATTENTION PHASE:
                GPU0: Q0 × [K0 K1 K2 K3] × [V0 V1 V2 V3] → Output0
                GPU1: Q1 × [K0 K1 K2 K3] × [V0 V1 V2 V3] → Output1  
                GPU2: Q2 × [K0 K1 K2 K3] × [V0 V1 V2 V3] → Output2
                GPU3: Q3 × [K0 K1 K2 K3] × [V0 V1 V2 V3] → Output3

                FINAL OUTPUTS:
                GPU0: Output0 (local to T0-T1)
                GPU1: Output1 (local to T2-T3)
                GPU2: Output2 (local to T4-T5) 
                GPU3: Output3 (local to T6-T7)
        '''

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

        self.context_parallel_group = None
        if config.context_parallel_size > 1:
            # Create context parallel group (all GPUs in this case)
            self.context_parallel_group = torch.distributed.group.WORLD

        # After all modules are created, propagate context parallel group
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
        B, T_local = idx.size()  # CHANGE: T_local instead of T
        start_pos = 0

        # Calculate shard start position for this rank
        shard_start = self.config.context_parallel_rank * T_local
        
        tkn_emb = self.tkn_emb(idx)
        x = tkn_emb
        freqs_cis = None

        # RoPE buffer length check
        if self.config.pos_emb == 'rope':
            assert shard_start + T_local <= self.freqs_cis.size(0), \
                f"RoPE buffer too short: need {shard_start + T_local}, have {self.freqs_cis.size(0)}"
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
        # Initialize aux_loss as tensor
        total_aux_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        
        for i, block in enumerate(self.transformer.h):
            x, updated_kv_cache, aux_loss = block(x, freqs_cis, kv_caches[i], self.VAL_RUN)            
            updated_kv_caches.append(updated_kv_cache)
            # Ensure aux_loss is always a tensor
            if not torch.is_tensor(aux_loss):
                aux_loss = torch.as_tensor(aux_loss, device=x.device, dtype=torch.float32)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            
            # Valid token aware loss calculation
            targets_flat = targets.view(-1)
            valid = (targets_flat != -1)
            num_valid_local = valid.long().sum()
            
            if num_valid_local > 0:
                loss_sum = F.cross_entropy(
                    logits.view(-1, logits.size(-1))[valid],
                    targets_flat[valid],
                    reduction='sum',
                )
            else:
                loss_sum = torch.zeros((), device=logits.device, dtype=torch.float32)
            
            # Synchronize across all ranks
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(num_valid_local, op=torch.distributed.ReduceOp.SUM)
            
            num_valid_global = num_valid_local.clamp_min(1)
            loss = (loss_sum / num_valid_global).to(torch.float32)
            
            # Synchronize aux loss
            if torch.distributed.is_initialized():
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
            raise NotImplementedError("Generation with context parallelism requires ring attention (not implemented)")

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


class DataLoader:
    def __init__(self, B, T, file_path, device, context_parallel_size=1, context_parallel_rank=0):
        self.B = B
        self.T = T
        self.file_path = file_path
        self.device = device
        # self.device_type = 'cuda'

        # FIX: Convert device string to torch.device object
        self.device = torch.device(device)
        self.device_type = self.device.type  # Store device type separately if needed

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
        
        B, local_T = self.B, self.local_T
        start_indices = torch.randint(0, self.N - self.T - 1, (B,))

        x_list = []
        y_list = []
        for start in start_indices:
            full_seq = self.tokens[start : start + self.T + 1].astype(np.int64)
            
            # Extract local chunk for this context parallel rank
            local_start = self.context_parallel_rank * local_T
            local_end = local_start + local_T
            x_local = full_seq[local_start:local_end]
            y_local = full_seq[local_start + 1:local_end + 1]
            
            x_list.append(x_local)
            y_list.append(y_local)

        x = torch.from_numpy(np.stack(x_list)).long()
        y = torch.from_numpy(np.stack(y_list)).long()

        # Verify local sequence length
        assert x.shape[1] == self.local_T, f"Expected local_T={self.local_T}, got {x.shape[1]}"
        
        # Proper device handling
        if self.device.type == 'cuda':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y





def all_gather_sequence(tensor: torch.Tensor, dim: int, group=None) -> torch.Tensor:
    """Efficient all-gather along specified dimension using all_gather_into_tensor"""
    if not torch.distributed.is_initialized():
        return tensor
        
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return tensor

    # Move target dimension to front for all_gather_into_tensor
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




# train_loader = DataLoader(B=TrainingConfig.batch_size, T=ModelConfig.block_size, file_path="train.bin", device=device , context_parallel_size=world_size,
#     context_parallel_rank=rank)
# val_loader = DataLoader(B=TrainingConfig.batch_size, T=ModelConfig.block_size, file_path="val.bin", device=device)

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



model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)


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
        # model.clip_grad_norm_(TrainingConfig.grad_clip)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), TrainingConfig.grad_clip)

        


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