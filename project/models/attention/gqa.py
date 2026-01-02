
import math

from config.model import LLMconfig

from typing import Literal, Optional, Dict, List, Tuple

from train import parallel_flag
import torch
import torch.nn as nn
import torch.nn.functional as F

from parallel.utils import all_gather_sequence

from parallel.tp import _get_group_and_ranks


class GQA(nn.Module):
    """ Grouped-Query Attention with or without RoPE """

    def __init__(self, config: LLMconfig, tp_group=None):
        super().__init__()
        # Validate attention configuration
        if config.attn == 'mha':
            config.n_kv_heads = config.n_head
        elif config.attn == 'mqa':
            config.n_kv_heads = 1
        else:
            assert config.n_head % config.n_kv_heads == 0, "n_head must be divisible by n_kv_heads"
        
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.config = config
        self.head_size = config.n_embd // config.n_head
        
        # Common dropout layers
        self.resid_dropout = nn.Dropout(config.dropout)

        if parallel_flag == 5:
            # Tensor Parallelism setup
            self.tp_group, self.tp_size, self.tp_rank = _get_group_and_ranks(tp_group)
            
            # Critical divisibility assertions
            assert config.n_head % self.tp_size == 0, "n_head must be divisible by tp_size"
            assert config.n_embd % self.tp_size == 0, "n_embd must be divisible by tp_size"
            
            self.n_head_per_rank = config.n_head // self.tp_size
            
            # Handle KV head partitioning
            self.partition_kv = (config.n_kv_heads % self.tp_size == 0)
            if self.partition_kv:
                self.n_kv_heads_per_rank = config.n_kv_heads // self.tp_size
                kv_out_features = 2 * config.n_kv_heads * self.head_size
                assert kv_out_features % self.tp_size == 0, \
                    "KV out features must be divisible by tp_size when partitioning KV"
            else:
                self.n_kv_heads_per_rank = config.n_kv_heads  # Replicated on all ranks
            
            # Projection layers for tensor parallelism
            kv_out_features = 2 * config.n_kv_heads * self.head_size
            self.q_proj = ColumnParallelLinear(
                config.n_embd, config.n_embd, 
                bias=True, gather_output=False, group=self.tp_group
            )
            
            if self.partition_kv:
                self.kv_proj = ColumnParallelLinear(
                    config.n_embd, kv_out_features,
                    bias=True, gather_output=False, group=self.tp_group
                )
            else:
                self.kv_proj = nn.Linear(config.n_embd, kv_out_features, bias=True)
            
            self.c_proj = RowParallelLinear(
                config.n_embd, config.n_embd,
                bias=True, input_is_parallel=True, group=self.tp_group
            )
            
        elif parallel_flag == 8:
            # Context Parallelism setup
            self.context_parallel_group = getattr(config, 'context_parallel_group', None)
            self.attn_dropout = nn.Dropout(config.dropout)
            self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * config.n_kv_heads * self.head_size)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd)
            
        else:
            # Standard single GPU setup
            self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * config.n_kv_heads * self.head_size)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd)
            self.attn_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None, 
                kv_cache=None, VAL_RUN=False) -> Tuple[torch.Tensor, Optional[Tuple]]:
        
        B = x.size(0)
        C = x.size(2)
        nh = self.config.n_head
        nkvh = self.config.n_kv_heads
        hs = self.head_size
        
        # Common projection and reshape operations
        def project_and_reshape(x, kv_only=False):
            """Common projection and reshape logic"""
            if parallel_flag == 5:
                if kv_only:
                    kv = self.kv_proj(x)
                    expected_kv_dim = 2 * (self.n_kv_heads_per_rank if self.partition_kv else nkvh) * hs
                    assert kv.shape[-1] == expected_kv_dim, \
                        f"KV projection output dim {kv.shape[-1]} != expected {expected_kv_dim}"
                    kv_split_size = expected_kv_dim // 2
                    k, v = kv.split([kv_split_size, kv_split_size], dim=2)
                    
                    k = k.contiguous().view(B, -1, self.n_kv_heads_per_rank, hs)
                    v = v.contiguous().view(B, -1, self.n_kv_heads_per_rank, hs)
                    
                    # Query projection
                    q_local = self.q_proj(x)
                    q = q_local.contiguous().view(B, -1, self.n_head_per_rank, hs)
                else:
                    q_local = self.q_proj(x)
                    q = q_local.contiguous().view(B, -1, self.n_head_per_rank, hs)
                    kv = self.kv_proj(x)
                    expected_kv_dim = 2 * (self.n_kv_heads_per_rank if self.partition_kv else nkvh) * hs
                    kv_split_size = expected_kv_dim // 2
                    k, v = kv.split([kv_split_size, kv_split_size], dim=2)
                    k = k.contiguous().view(B, -1, self.n_kv_heads_per_rank, hs)
                    v = v.contiguous().view(B, -1, self.n_kv_heads_per_rank, hs)
                return q, k, v
            else:
                q_proj_size = C
                kv_proj_size = nkvh * hs
                q, k, v = self.c_attn(x).split([q_proj_size, kv_proj_size, kv_proj_size], dim=2)
                q = q.view(B, -1, nh, hs)
                k = k.view(B, -1, nkvh, hs)
                v = v.view(B, -1, nkvh, hs)
                return q, k, v

        # Common RoPE application
        def apply_rope_if_needed(q, k):
            if self.config.pos_emb == 'rope' and freqs_cis is not None:
                q = LLMconfig.apply_rotary_emb(q, freqs_cis)
                k = LLMconfig.apply_rotary_emb(k, freqs_cis)
            return q, k

        # Common KV cache handling
        def handle_kv_cache(k, v, kv_cache):
            if kv_cache is not None:
                past_k, past_v = kv_cache
                k = torch.cat((past_k, k), dim=-2)
                v = torch.cat((past_v, v), dim=-2)
            updated_kv_cache = (k, v)
            return k, v, updated_kv_cache

        # Common KV head repetition
        def repeat_kv_heads_if_needed(k, v, nkvh_local, nh_local):
            if nkvh_local != nh_local:
                num_repeats = nh_local // nkvh_local
                k = k.repeat_interleave(num_repeats, dim=1)
                v = v.repeat_interleave(num_repeats, dim=1)
            return k, v

        if parallel_flag == 5:
            # Tensor Parallelism forward pass
            T = x.size(1)
            q, k, v = project_and_reshape(x)
            
            # Apply RoPE
            q, k = apply_rope_if_needed(q, k)
            
            # Transpose for attention
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            
            # Handle KV cache
            k, v, updated_kv_cache = handle_kv_cache(k, v, kv_cache)
            
            # Repeat KV heads
            k, v = repeat_kv_heads_if_needed(k, v, self.n_kv_heads_per_rank, self.n_head_per_rank)
            
            # Scaled dot-product attention
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.config.dropout if self.training else 0,
                is_causal=True
            )
            
            # Reshape and project
            y = y.transpose(1, 2).contiguous().view(B, T, -1)
            y = self.resid_dropout(self.c_proj(y))
            
            return y, updated_kv_cache

        elif parallel_flag == 8:
            # Context Parallelism forward pass
            T_local = x.size(1)
            q, k, v = project_and_reshape(x)
            
            # Apply RoPE before gathering
            q, k = apply_rope_if_needed(q, k)
            
            # Transpose
            q = q.transpose(1, 2)  # (B, nh, T_local, hs)
            k = k.transpose(1, 2)  # (B, nkvh, T_local, hs)
            v = v.transpose(1, 2)  # (B, nkvh, T_local, hs)
            
            # Context Parallel: Gather K and V
            if self.config.context_parallel_size > 1:
                k = all_gather_sequence(k, dim=2, group=self.context_parallel_group)
                v = all_gather_sequence(v, dim=2, group=self.context_parallel_group)
            
            # Handle KV cache (disabled for context parallelism)
            use_cp = (self.config.context_parallel_size > 1)
            if use_cp:
                updated_kv_cache = None
            else:
                k, v, updated_kv_cache = handle_kv_cache(k, v, kv_cache)
            
            # Repeat KV heads
            k, v = repeat_kv_heads_if_needed(k, v, nkvh, nh)
            
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
            
            # Reshape and project
            y = y.transpose(1, 2).contiguous().view(B, T_local, C)
            y = self.resid_dropout(self.c_proj(y))
            
            return y, updated_kv_cache

        else:
            # Standard forward pass
            T = x.size(1)
            q, k, v = project_and_reshape(x)
            
            # Apply RoPE
            q, k = apply_rope_if_needed(q, k)
            
            # Transpose
            q = q.transpose(1, 2)  # (B, nh, T, hs)
            k = k.transpose(1, 2)  # (B, nkvh, T, hs)
            v = v.transpose(1, 2)  # (B, nkvh, T, hs)
            
            # Handle KV cache
            k, v, updated_kv_cache = handle_kv_cache(k, v, kv_cache)
            
            # Repeat KV heads
            k, v = repeat_kv_heads_if_needed(k, v, nkvh, nh)
            
            # Scaled dot-product attention
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.config.dropout if self.training else 0,
                is_causal=True
            )
            
            # Reshape and project
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.resid_dropout(self.c_proj(y))
            
            return y, updated_kv_cache

