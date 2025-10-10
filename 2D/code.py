
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
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from typing import Literal
from dataclasses import dataclass 

from torch.nn.parallel import DistributedDataParallel as DDP




# =====2D====
def verify_data_broadcasting(topology, train_loader, B, T, device):
    """Verify that TP groups get identical batches and return success status"""
    success = True
    
    if topology["is_global_leader"]:
        print("Verifying data broadcasting topology...")
    
    # Test batch broadcasting
    with torch.no_grad():
        if topology["is_tp_leader"]:
            x, y = train_loader.next_batch()
        else:
            x = torch.empty(B, T, dtype=torch.long, device=device)
            y = torch.empty(B, T, dtype=torch.long, device=device)
        
        x, y = broadcast_batch_tp(x, y, topology["tp_group"], 0)

        # GPU-only checksum: device-side, 64-bit (no Python int round-trip)
        batch_hash = x.to(torch.int64).sum().unsqueeze(0)  # shape [1], device=cuda

        # All shards in a TP group must match
        h_min = batch_hash.clone()
        h_max = batch_hash.clone()
        dist.all_reduce(h_min, op=dist.ReduceOp.MIN, group=topology["tp_group"])
        dist.all_reduce(h_max, op=dist.ReduceOp.MAX, group=topology["tp_group"])
        
        if h_min.item() != h_max.item():
            print(f"❌ TP shards disagree on input batch! min={h_min.item()}, max={h_max.item()}")
            success = False
        else:
            if topology["is_global_leader"]:
                print("✅ TP group broadcasting verified: all shards see identical data")

        '''
        Why only TP rank 0?

        Within each TP group, all shards have identical data (verified earlier)
        No need for all TP shards to participate - they'd provide duplicate information
        TP rank 0 acts as the "representative" for its entire TP group

        TP Group 0: [GPU0, GPU1] → Both have same data, GPU0 represents
        TP Group 1: [GPU2, GPU3] → Both have same data, GPU2 represents  
        TP Group 2: [GPU4, GPU5] → Both have same data, GPU4 represents


        '''        
        
        # Optional: Verify DP groups see different data (GPU-only gather)
        if topology["tp_rank"] == 0:
            gather_buf = [torch.zeros_like(batch_hash, device=device) for _ in range(topology["dp_size"])]
            dist.all_gather(gather_buf, batch_hash, group=topology["dp_group"])
            unique_hashes = len(set([t.item() for t in gather_buf]))
            
            if topology["is_global_leader"]:
                if unique_hashes == topology["dp_size"]:
                    print(f"✅ DP groups: all {topology['dp_size']} replicas have unique batches")
                else:
                    print(f"⚠️  DP groups: {unique_hashes} unique batches out of {topology['dp_size']} replicas")
                    # Not a failure - this can happen with small datasets
    
    return success




class DPDataLoader:
    def __init__(self, B, T, file_path, device, dp_rank, dp_size):
        self.B = B
        self.T = T
        self.file_path = file_path
        self.device = device
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        
        self.tokens = np.memmap(self.file_path, dtype=np.uint16, mode='r')
        self.N = len(self.tokens)
        
        # Improved shard assignment with remainder handling
        base_shard_size = self.N // self.dp_size
        remainder = self.N % self.dp_size
        
        # Distribute remainder across first few DP ranks
        extra = 1 if self.dp_rank < remainder else 0
        offset = min(self.dp_rank, remainder)
        
        self.shard_size = base_shard_size + extra
        self.start_idx = self.dp_rank * base_shard_size + offset
        self.end_idx = self.start_idx + self.shard_size
        
        # Validate shard can accommodate sequence length
        if self.shard_size < T + 1:
            raise ValueError(
                f"Shard too small for sequence length T={T}: "
                f"shard_size={self.shard_size}, need at least {T+1}"
            )
        
        # Use separate generator for data sampling
        self.generator = torch.Generator(device='cpu')
        self.generator.manual_seed(1729 + dp_rank)
        self.epoch = 0
        
        if dp_rank == 0:  # Only log from one DP leader
            print(f"DPDataLoader: DP_rank={dp_rank}, shard [{self.start_idx}:{self.end_idx}] "
                  f"(size={self.shard_size})")  

    def set_epoch(self, epoch):
        """Reset generator for new epoch to avoid same samples each epoch"""
        self.epoch = epoch
        self.generator.manual_seed(1729 + self.dp_rank + epoch * 1000)

    def next_batch(self):
        B, T = self.B, self.T
        
        # Correct bounds: torch.randint low (inclusive) to high (exclusive)
        low = self.start_idx
        high = self.end_idx - T  # exclusive upper bound
        
        if high <= low:
            raise ValueError(
                f"Shard too small for T={T}: "
                f"[{low}, {self.end_idx}) length={self.end_idx - low}"
            )
        
        # Sample using isolated generator
        start_indices = torch.randint(
            low, high, (B,), generator=self.generator
        )
        
        x_list, y_list = [], []
        for start in start_indices:
            seq = self.tokens[start:start + T + 1].astype(np.int64)
            x_list.append(seq[:-1])
            y_list.append(seq[1:])
        
        x = torch.from_numpy(np.stack(x_list)).long()
        y = torch.from_numpy(np.stack(y_list)).long()
        
        x = x.pin_memory().to(self.device, non_blocking=True)
        y = y.pin_memory().to(self.device, non_blocking=True)
        
        return x, y



# After model creation and DDP wrapping - OPTIONAL FEATURE
ENABLE_DP_TORCH_RNG_DIVERSITY = True  # Configurable flag

def setup_dp_runtime_rng_diversity(topology, base_seed=1729):
    """
    OPTIONAL: Diversify PyTorch's runtime RNG per DP replica for dropout diversity
    
    This should be called AFTER model initialization to keep weights identical,
    but BEFORE any forward passes to get different dropout masks per DP replica.
    """
    if not ENABLE_DP_TORCH_RNG_DIVERSITY:
        return
        
    seed = base_seed + topology["dp_rank"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    if topology["is_global_leader"]:
        print(f"Runtime RNG: DP replicas have different torch RNG (base={base_seed} + DP_rank)")







# we added the below code
def _get_group_and_ranks(tp_group = None):
    """Get TP group, world size, and rank - safer version""" 
    if not dist.is_initialized():
        print('inside if not dist.is_initialized() ')
        return None ,1, 0
    
    tp_group = tp_group or dist.group.WORLD

    return tp_group , dist.get_world_size(tp_group) , dist.get_rank(tp_group)


def verify_model_consistency(model, topology, rtol=1e-5, atol=1e-8):
    """Verify all DP replicas have identical model weights using full checksum"""
    with torch.no_grad():
        # Build a comprehensive checksum across ALL parameters
        checksum = torch.tensor(0.0, device=next(model.parameters()).device, dtype=torch.float32)
        
        for param in model.parameters():
            # Sum all parameters in float32 for numerical stability
            checksum += param.float().sum()
        
        # Each DP group should be identical → SUM across DP gives dp_size * checksum
        checksum_sum = checksum.clone()
        dist.all_reduce(checksum_sum, op=dist.ReduceOp.SUM, group=topology["dp_group"])
        
        expected = checksum * topology["dp_size"]
        is_consistent = torch.allclose(checksum_sum, expected, rtol=rtol, atol=atol)
        
        if topology["is_global_leader"]:
            if is_consistent:
                print(f"✅ Model weights: All {topology['dp_size']} DP replicas are identical")
                print(f"   Checksum: {checksum.item():.6f} (consistent across replicas)")
            else:
                print(f"❌ Model weights: DP replicas have diverged!")
                print(f"   Expected: {expected.item():.6f}, Got: {checksum_sum.item():.6f}")
                print(f"   Difference: {torch.abs(checksum_sum - expected).item():.6f}")
        
        return is_consistent





def run_integration_checks(model, topology, train_loader, B, T, device):
    """Run comprehensive integration checks for DP×TP setup"""
    if topology["is_global_leader"]:
        print("\n" + "="*60)
        print("DP×TP INTEGRATION CHECKS")
        print("="*60)
    

    # -> FOR DP : checks all model onstance are the same or not 
    # 1. Check model weight consistency across DP replicas
    model_consistent = verify_model_consistency(model, topology)
    

    # for TP 
    # 2. Check data broadcasting within TP groups (now with GPU-only checksum)
    data_consistent = verify_data_broadcasting(topology, train_loader, B, T, device)

    
    # 3. Final integration status
    all_checks_passed = model_consistent and data_consistent
    
    if topology["is_global_leader"]:
        print("\n" + "="*60)
        print("INTEGRATION SUMMARY:")
        print(f"   - Model weight consistency: {'PASS' if model_consistent else 'FAIL'}")
        print(f"   - Data broadcasting: {'PASS' if data_consistent else 'FAIL'}")
        print(f"   - Topology: DP={topology['dp_size']} × TP={topology['tp_size']}")
        print(f"   - Runtime RNG diversity: {'ENABLED' if ENABLE_DP_TORCH_RNG_DIVERSITY else 'DISABLED'}")
        
        if all_checks_passed:
            print("✅ ALL CHECKS PASSED - DP×TP setup is correct!")
        else:
            print("❌ SOME CHECKS FAILED - Review configuration")
        print("="*60)
    
    return all_checks_passed



class ColumnParallelLinear(nn.Module):
    """Shard the weight matrix along output dimension (column-wise)"""

    def __init__(self , in_features , out_features , bias = True , gather_output=True , group = None):
        super().__init__()

        self.group  , self.world_size , self.rank  = _get_group_and_ranks(group)

        assert out_features % self.world_size == 0, \
            f"out_features={out_features} not divisible by TP world_size={self.world_size}"

        self.local_out_features = out_features // self.world_size

        self.linear = nn.Linear(in_features, self.local_out_features, bias=bias)
        self.gather_output = gather_output


        # TP-aware initialization for better training parity
        self._apply_tp_aware_init()


    def _apply_tp_aware_init(self):
        """Scale initialization for TP parity"""

        with torch.no_grad():
            # Scale weights by 1/sqrt(tp_size) for variance preservation
            if self.world_size > 1:

                self.linear.weight.data.div_(self.world_size ** 0.5)

                if self.linear.bias is not None:
                    self.linear.bias.data.div_(self.world_size ** 0.5)

    def forward(self , x):
        local_output = self.linear(x)

        if self.world_size > 1 and self.gather_output:
            # Faster all-gather with pre-allocated tensor
            full_output = torch.empty(
                *local_output.shape[:-1],
                local_output.shape[-1] * self.world_size,
                dtype=local_output.dtype,
                device=local_output.device
            )
            dist.all_gather_into_tensor(full_output, local_output, group=self.group)
            return full_output
        return local_output

class RowParallelLinear(nn.Module):
    """Shard the weight matrix along input dimension (row-wise)"""
    def __init__(self, in_features, out_features, bias=True, input_is_parallel=False, group=None):
        super().__init__()
        self.group, self.world_size, self.rank = _get_group_and_ranks(group)
        
        assert in_features % self.world_size == 0, \
            f"in_features={in_features} not divisible by TP world_size={self.world_size}"
        
        self.local_in_features = in_features // self.world_size
        self.linear = nn.Linear(self.local_in_features, out_features, bias=bias)
        self.input_is_parallel = input_is_parallel
        
    def forward(self, x):
        if not self.input_is_parallel and self.world_size > 1:
            # Split input along feature dimension
            x = x.chunk(self.world_size, dim=-1)[self.rank]
        
        local_output = self.linear(x)
        
        if self.world_size > 1:
            dist.all_reduce(local_output, op=dist.ReduceOp.SUM, group=self.group)
            
        return local_output


# done implementation



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

    def __init__(self, config:LLMconfig, tp_group = None):
        super().__init__()

        if config.attn == 'mha' : config.n_kv_heads = config.n_head
        elif config.attn == 'mqa' : config.n_kv_heads = 1
        else : assert config.n_head % config.n_kv_heads == 0, "n_head must be divisible by n_kv_heads"
        
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.config = config

        # for TP -> added the below line
        self.tp_group, self.tp_size, self.tp_rank = _get_group_and_ranks(tp_group)
        # Critical divisibility assertions
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        assert config.n_head % self.tp_size == 0, "n_head must be divisible by tp_size"
        assert config.n_embd % self.tp_size == 0, "n_embd must be divisible by tp_size"



        self.head_size = config.n_embd // config.n_head

        # Distribute heads across TP ranks
        self.n_head_per_rank = config.n_head // self.tp_size

        '''
        # k,q,v in a btach
        self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * config.n_kv_heads * self.head_size)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        '''


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
        '''
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
        '''

        B, T, C = x.size()
        hs = self.head_size
        
        # Query projection (always TP-sharded) - with contiguity guarantee
        q_local = self.q_proj(x)  # [B, T, C/tp_size]
        q = q_local.contiguous().view(B, T, self.n_head_per_rank, hs)
        
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
        
        # Scaled dot-product attention
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.config.dropout if self.training else 0,
            is_causal=True
        )
        
        # Reshape back to [B, T, local_dim]
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        
        # Output projection with all-reduce
        y = self.resid_dropout(self.c_proj(y))
        
        return y, updated_kv_cache




class NaiveMHLA(nn.Module):
    """ A fully parallel implementation of the MHLA algorithm without the RoPE. No for loops."""
    """TP-enabled MHLA (no RoPE). Shards across heads only; latent dims replicated."""

    def __init__(self, config:LLMconfig):
        super().__init__()

        assert config.n_embd % config.n_head == 0, "num of heads must be a divisor of n_embd"


        self.head_size = config.n_embd // config.n_head
        self.config = config

        # TP Plumbing
        self.tp_group, self.tp_size, self.tp_rank = _get_group_and_ranks(tp_group)
        assert config.n_head % self.tp_size == 0, "n_head must be divisible by tp_size"
        assert config.n_embd % self.tp_size == 0, "n_embd must be divisible by tp_size"
        self.nh_local = config.n_head // self.tp_size
        self.c_local = self.nh_local * self.head_size
        self.embd_start = self.c_local * self.tp_rank
        self.embd_end   = self.embd_start + self.c_local
        self.gather_output = gather_output



        # self.W_dq  = nn.Linear(config.n_embd,        config.q_latent_dim,  bias=False)
        # self.W_uq  = nn.Linear(config.q_latent_dim,  config.n_embd,        bias=False)
        # self.W_dkv = nn.Linear(config.n_embd,        config.kv_latent_dim, bias=False)
        # self.W_uk  = nn.Linear(config.kv_latent_dim, config.n_embd,        bias=False)
        # self.W_uv  = nn.Linear(config.kv_latent_dim, config.n_embd,        bias=False)
        # self.W_o   = nn.Linear(config.n_embd,        config.n_embd,        bias=False)

        # ✅ Keep replicated layers (like FullMHLA)
        self.W_dq  = nn.Linear(config.n_embd, config.q_latent_dim, bias=False)
        self.W_dkv = nn.Linear(config.n_embd, config.kv_latent_dim, bias=False)
        self.W_uk  = nn.Linear(config.kv_latent_dim, config.n_embd, bias=False)
        self.W_uv  = nn.Linear(config.kv_latent_dim, config.n_embd, bias=False)
        self.W_o   = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # ✅ Only shard W_uq
        self.W_uq = ColumnParallelLinear(
            config.q_latent_dim, config.n_embd, bias=False,
            gather_output=False, group=self.tp_group
        )
        
        # ✅ Add RowParallel output
        self.output_proj = RowParallelLinear(
            config.n_embd, config.n_embd, bias=False,
            input_is_parallel=True, group=self.tp_group
        )
        
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer('_k_absorbed_inference', None)
        self.register_buffer('_v_absorbed_inference', None)



    def _precompute_absorbed_matrices(self):
        """Precomputes k_absorbed and v_absorbed for efficient inference."""
        # Just to be safe
        if (self._k_absorbed_inference is not None) and (self._v_absorbed_inference is not None):
            return 

        if (self._k_abs_local is not None) and (self._v_abs_local is not None):
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
    TP-enabled Multi-Head Latent Attention with Decoupled Rotary Position Embeddings (RoPE).
    Shards across heads only; latent dims replicated.
    """
     
    def __init__(self, config: LLMconfig, tp_group=None, gather_output=True):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "num of heads must be a divisor of n_embd"
        self.config = config
        self.hs = config.n_embd // config.n_head
        self.dhr = config.rope_head_dim

        # ---- TP plumbing ----
        self.tp_group, self.tp_size, self.tp_rank = _get_group_and_ranks(tp_group)
        assert config.n_head % self.tp_size == 0, "n_head must be divisible by tp_size"
        assert config.n_embd % self.tp_size == 0, "n_embd must be divisible by tp_size"
        self.nh_local = config.n_head // self.tp_size

        self.c_local = self.nh_local * self.hs

        self.embd_start = self.c_local * self.tp_rank
        self.embd_end = self.embd_start + self.c_local
        self.gather_output = gather_output

        # Keep latent/KV layers REPLICATED (as in original)
        self.W_dq  = nn.Linear(config.n_embd, config.q_latent_dim, bias=False)      # REPLICATED
        self.W_dkv = nn.Linear(config.n_embd, config.kv_latent_dim, bias=False)     # REPLICATED  
        self.W_uk  = nn.Linear(config.kv_latent_dim, config.n_embd, bias=False)     # REPLICATED
        self.W_uv  = nn.Linear(config.kv_latent_dim, config.n_embd, bias=False)     # REPLICATED
        self.W_o   = nn.Linear(config.n_embd, config.n_embd, bias=False)            # REPLICATED
        self.W_kr  = nn.Linear(config.n_embd, config.rope_head_dim, bias=False)     # REPLICATED

        

        # Only shard the "up to C" projections
        self.W_uq = ColumnParallelLinear(
            config.q_latent_dim, config.n_embd, bias=False,
            gather_output=False, group=self.tp_group  # ✅ gather_output=False to keep sharded
        )
        self.W_qr = ColumnParallelLinear(
            config.q_latent_dim, config.n_head * config.rope_head_dim, bias=False,
            gather_output=False, group=self.tp_group  # ✅ gather_output=False
        )
        
        # ✅ ADD RowParallel tail instead of manual all-gather
        self.output_proj = RowParallelLinear(
            config.n_embd, config.n_embd, bias=False,
            input_is_parallel=True, group=self.tp_group  # ✅ input_is_parallel=True
        )

        # ---- Replicated RoPE projections ----
        self.W_kr = nn.Linear(config.n_embd, config.rope_head_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        # rank-local absorbed matrices for inference
        self.register_buffer('_k_abs_local', None, persistent=False)
        self.register_buffer('_v_abs_local', None, persistent=False)


    def _precompute_absorbed_local(self):
        """Precomputes local absorbed matrices using the rank's head slice."""
        if (self._k_abs_local is not None) and (self._v_abs_local is not None):
            return
        
        nlq, n_kvl, C = self.config.q_latent_dim, self.config.kv_latent_dim, self.config.n_embd
        nh, hs = self.config.n_head, self.hs
        c0, c1 = self.embd_start, self.embd_end

        with torch.no_grad():
            # Local k_eff: (W_dq^T @ W_uq_local^T @ W_uk[local])
            W_dq_T = self.W_dq.weight.t()                    # [C, nlq]
            W_uq_loc_T = self.W_uq.linear.weight.t()         # [nlq, C_local]
            A = W_dq_T @ W_uq_loc_T                          # [C, C_local]
            
            # Extract local head slice
            A_loc_rows = A[c0:c1, :]                         # [C_local, C_local]
            W_uk_rows = self.W_uk.weight[c0:c1, :]           # [C_local, n_kvl]
            
            # k_abs_local: [C_local, n_kvl] -> [1, nh_local, hs, n_kvl]
            k_abs_local = (A_loc_rows @ W_uk_rows).view(self.nh_local, hs, n_kvl).unsqueeze(0)

            # Local v_eff: (W_uv^T @ W_o^T[:, local])
            v_abs_local = self.W_uv.weight.t() @ self.W_o.weight.t()[:, c0:c1]  # [n_kvl, C_local]
            v_abs_local = v_abs_local.view(n_kvl, self.nh_local, hs).transpose(0, 1).unsqueeze(0)  # [1, nh_local, n_kvl, hs]

            self._k_abs_local = k_abs_local
            self._v_abs_local = v_abs_local

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor | None, kv_cache=None, VAL_RUN=False):
        B, T, C = x.size()
        # if self.tp_rank == 0:  # Only print from rank 0 to avoid clutter
        #     print(f"FullMHLA forward: input shape {x.shape}, tp_size={self.tp_size}")

        n_kvl, nlq = self.config.kv_latent_dim, self.config.q_latent_dim
        dhr = self.dhr

        # ---------- NoPE Path (Content-based attention) ----------
        
        # KV latent (replicated across TP ranks)
        new_c_kv = self.W_dkv(x)  # [B, T, n_kvl]
        
        # Handle KV cache
        if kv_cache is None:
            c_kv = new_c_kv
            k_r_cache = None
        else:
            c_kv = torch.cat([kv_cache['c_kv'], new_c_kv], dim=1)
            k_r_cache = kv_cache['k_r']
        
        T_full = c_kv.size(1)
        updated_kv_cache = {'c_kv': c_kv}

        # Query path: replicated down, sharded up
        c_q = self.W_dq(x)  # [B, T, nlq] - replicated
        
        # NoPE query projection (sharded)
        q_local = self.W_uq(c_q)  # [B, T, C_local]
        q_local = q_local.view(B, T, self.nh_local, self.hs).transpose(1, 2)  # [B, nh_local, T, hs]

        # RoPE query projection (sharded)
        qr_local = self.W_qr(c_q)  # [B, T, nh_local * dhr]
        qr_local = qr_local.view(B, T, self.nh_local, dhr)  # [B, T, nh_local, dhr]
        qr_local = qr_local.transpose(1, 2)  # [B, nh_local, T, dhr]

        # ---------- RoPE Path (Position-based attention) ----------
        
        # RoPE key projection (replicated)
        c_kr = self.W_kr(x).unsqueeze(2)  # [B, T, 1, dhr]
        k_r = LLMconfig.apply_rotary_emb(c_kr, freqs_cis).transpose(1, 2)  # [B, 1, T, dhr]
        
        # Handle RoPE KV cache
        if k_r_cache is not None:
            k_r = torch.cat([k_r_cache, k_r], dim=2)  # [B, 1, T_full, dhr]
        updated_kv_cache['k_r'] = k_r

        # Apply RoPE to local query
        if freqs_cis is not None:
            # Reshape for RoPE application
            qr_rope_ready = qr_local.transpose(1, 2)  # [B, T, nh_local, dhr]
            qr_rope_ready = qr_rope_ready.contiguous().view(B, T, self.nh_local * dhr)
            qr_rope_ready = qr_rope_ready.view(B, T, self.nh_local, dhr)  # Ensure correct shape
            
            qr_local = LLMconfig.apply_rotary_emb(qr_rope_ready, freqs_cis).transpose(1, 2)  # [B, nh_local, T, dhr]

        # ---------- Attention Computation ----------
        
        # Get or compute local absorbed matrices
        if self.training or VAL_RUN:
            with torch.no_grad():
                self._k_abs_local = None
                self._v_abs_local = None
            self._precompute_absorbed_local()
            k_abs = self._k_abs_local
            v_abs = self._v_abs_local
        else:
            if self._k_abs_local is None or self._v_abs_local is None:
                self._precompute_absorbed_local()
            k_abs = self._k_abs_local
            v_abs = self._v_abs_local

        # NoPE attention (content-based)
        attn_c = (q_local @ k_abs @ c_kv.transpose(1, 2).unsqueeze(1))  # [B, nh_local, T, T_full]

        # RoPE attention (position-based)
        attn_r = qr_local @ k_r.transpose(-1, -2)  # [B, nh_local, T, T_full]

        # Combined attention
        attn = (attn_c + attn_r) / math.sqrt(self.hs + dhr)

        # Causal masking
        mask = torch.triu(torch.ones(T, T_full, device=x.device, dtype=torch.bool), 
                         diagonal=T_full - T + 1)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = self.dropout(F.softmax(attn, dim=-1))

        # ---------- Output Computation ----------
        
        # Output via latent
        y_local = attn @ c_kv.unsqueeze(1) @ v_abs  # [B, nh_local, T, hs]
        y_local = y_local.transpose(1, 2).contiguous().view(B, T, self.c_local)  # [B, T, C_local]
        y_local = self.dropout(y_local)

        y = self.output_proj(y_local)  # Automatically handles all-reduce

        return  y, updated_kv_cache



class Attention(nn.Module):
    """ Routes the attention mechanism according to the config"""

    def __init__(self, config:LLMconfig , tp_group=None):
        super().__init__()
        self.config = config
        if config.attn in ('mha','mqa','gqa'):
            self.attn = GQA(config , tp_group=tp_group)
        
        elif config.attn == 'mla':
            if config.pos_emb != 'rope':
                self.attn = NaiveMHLA(config , tp_group=tp_group, gather_output=True)
            else:
                self.attn = FullMHLA(config , tp_group=tp_group, gather_output=True)
                
    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor|None = None, kv_cache=None, VAL_RUN=False):
        return self.attn(x, freqs_cis, kv_cache, VAL_RUN)




class MLP(nn.Module):
    """TP-aware feed-forward block. ColumnParallel -> act -> RowParallel.
       Falls back to replicated when TP is disabled/unavailable.
    """
    def __init__(self, config: LLMconfig, tp_group=None, enable_tp=True):
        super().__init__()
        self.non_linearity = config.non_linearity.lower()
        self.tp_group, self.tp_size, self.tp_rank = _get_group_and_ranks(tp_group)

        # Enable TP only if group is valid and dist is initialized
        self.enable_tp = (
            enable_tp and (tp_group is not None) and (self.tp_size > 1) and dist.is_initialized()
        )

        # Activation lookup
        nl_map = {
            'relu': nn.ReLU(), 'gelu': nn.GELU(), 'swish': nn.SiLU(), 'mish': nn.Mish(),
            'silu': nn.SiLU(), 'selu': nn.SELU(), 'celu': nn.CELU(), 'elu': nn.ELU(),
            'glu': nn.GLU(), 'sigmoid': nn.Sigmoid(), 'lrelu': nn.LeakyReLU(0.01), 'tanh': nn.Tanh()
        }
        self.non_linearity_func = nl_map.get(self.non_linearity, nn.GELU())

        if self.enable_tp:
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
            x1, x2 = self.c_fc(x).chunk(2, dim=-1)  # ✅ Local split on sharded tensor
            x = F.silu(x1) * x2
        else:
            x = self.non_linearity_func(self.c_fc(x))
        
        x = self.c_proj(x)  # ✅ RowParallel automatically handles all-reduce when TP is on
        x = self.dropout(x)
        return x

class Expert(nn.Module):
    """ A single feed-forward network expert. """
    def __init__(self, config:LLMconfig):
        super().__init__()
        # self.expert = MLP(config)
        self.expert = MLP(config, tp_group=None, enable_tp=False)  # ✅ Force replicated
        
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
    def __init__(self, config:LLMconfig , tp_group = None):
        super().__init__()
        self.is_moe = config.moe
        self.act_recomp = config.act_recomp
        self.attn = Attention(config , tp_group=tp_group)
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.ln2  = nn.LayerNorm(config.n_embd)
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

class LLM(nn.Module):
    """ A simple Large language model """
    def __init__(self, config:LLMconfig , tp_group=None):
        super().__init__()
        self.config = config
        self.tp_group = tp_group  # Store TP group


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
            h    = nn.ModuleList([Block(config , tp_group = tp_group) for _ in range(config.n_layer)]),
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
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from torch.distributed.fsdp import MixedPrecision
# from torch.distributed.fsdp import StateDictType, FullStateDictConfig
# from torch.distributed.fsdp.wrap import ModuleWrapPolicy
# from torch.distributed.fsdp.api  import ShardingStrategy, CPUOffload

assert torch.cuda.is_available()
assert torch.cuda.device_count() > 1


# ----------

def init_dp_tp_topology(tp_size: int = None):
    """Initialize DP × TP topology with orthogonal groups"""
    assert dist.is_initialized(), "Distributed must be initialized first"
    
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    
    # Set TP size with better defaults
    if tp_size is None:
        # Sensible default: use 2-way TP for multi-GPU, fallback to world_size for single GPU
        tp_size = min(world_size, 2) if world_size > 1 else 1
    
    assert world_size % tp_size == 0, \
        f"world_size={world_size} must be divisible by tp_size={tp_size}"
    
    dp_size = world_size // tp_size
    
    # Calculate ranks
    tp_rank = global_rank % tp_size
    dp_rank = global_rank // tp_size
    
    # Initialize group variables with None for safety
    my_tp_group = None
    my_dp_group = None
    
    # Create TP groups (contiguous ranks: [0,1,2,3], [4,5,6,7], ...)
    tp_groups = []
    for dp_idx in range(dp_size):
        tp_ranks = list(range(dp_idx * tp_size, (dp_idx + 1) * tp_size))
        tp_group = dist.new_group(ranks=tp_ranks)
        tp_groups.append(tp_group)
        if dp_idx == dp_rank:
            my_tp_group = tp_group
    
    # Create DP groups (strided ranks: [0,4,8,...], [1,5,9,...], ...)
    dp_groups = []
    for tp_idx in range(tp_size):
        dp_ranks = list(range(tp_idx, world_size, tp_size))
        dp_group = dist.new_group(ranks=dp_ranks)
        dp_groups.append(dp_group)
        if tp_idx == tp_rank:
            my_dp_group = dp_group
    
    # CRITICAL: Verify groups were created successfully
    assert my_tp_group is not None, "Failed to create local TP group"
    assert my_dp_group is not None, "Failed to create local DP group"
    
    # Leadership flags
    is_global_leader = (global_rank == 0)
    is_dp_leader = (dp_rank == 0)      
    is_tp_leader = (tp_rank == 0)      
    
    # Barrier for cluster stability
    dist.barrier()
    
    if is_global_leader:
        print(f"DP×TP Topology: {dp_size} × {tp_size} (DP × TP)")
        print(f"  - TP groups: {dp_size} groups of {tp_size} devices")
        print(f"  - DP groups: {tp_size} groups of {dp_size} devices")
        print(f"  - World size: {world_size}, Global rank: {global_rank}")
    
    return {
        "world_size": world_size,
        "global_rank": global_rank,
        "tp_size": tp_size,
        "tp_rank": tp_rank, 
        "tp_group": my_tp_group,
        "dp_size": dp_size,
        "dp_rank": dp_rank,
        "dp_group": my_dp_group,
        "is_global_leader": is_global_leader,
        "is_dp_leader": is_dp_leader,
        "is_tp_leader": is_tp_leader,
    }


# ______________DEVICE, DTYPE, DDP SETUP_________________

init_process_group(backend='nccl')

rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])


# Set device
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


master_process = rank == 0
if master_process: 
    print(f"Num GPUs = {world_size}")

# 🔧 FIX: Define tp_size BEFORE using it
tp_size_env = os.environ.get("TP_SIZE")
if tp_size_env is not None:
    tp_size = int(tp_size_env)
else:
    # For 2 GPUs, default to 2-way tensor parallelism
    tp_size = min(world_size, 2)

if master_process:
    print(f"Using TP size: {tp_size}")

# NOW initialize DP×TP topology
topology = init_dp_tp_topology(tp_size=tp_size)

# Set leadership flags
master_process = topology["is_global_leader"]

if master_process: 
    print(f"DP×TP: {topology['dp_size']} data parallel × {topology['tp_size']} tensor parallel")



# 3. THEN set up RNG diversity
if ENABLE_DP_TORCH_RNG_DIVERSITY:
    # Call this after model creation but before training
    setup_dp_runtime_rng_diversity(topology, base_seed=1729)


rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])


# device = f"cuda:{local_rank}"

# Correct approach
torch.cuda.set_device(local_rank)  # Integer
device = torch.device("cuda", local_rank)  # Proper device object

master_process = rank == 0
if master_process : print(f"Num GPUs = {world_size}")



# 🔧 FIX: Identical seeding for model initialization
base_seed = 1729
torch.manual_seed(base_seed)
torch.cuda.manual_seed(base_seed)
import random
import numpy as np
random.seed(base_seed)
np.random.seed(base_seed)

# torch.cuda.set_device(device)
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


# for TP -> added this code
# TP initialization
import os





def init_distributed():
    """Initialize distributed training for TP"""
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device correctly
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    return {
        "rank": rank,
        "world_size": world_size, 
        "local_rank": local_rank,
        "is_master": (rank == 0),
        "device": device,
        "tp_group": dist.group.WORLD
    }
# Initialize TP
ddp_info = init_distributed()
rank = ddp_info["rank"]
world_size = ddp_info["world_size"] 
local_rank = ddp_info["local_rank"]
master_process = ddp_info["is_master"]
device = ddp_info["device"]
tp_group = ddp_info["tp_group"]



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

# tokenize_and_save() # Using The Tiny Shakespeare dataset for demo

# Only download dataset on master process
if rank == 0:
    tokenize_and_save()
if dist.is_initialized():
    dist.barrier()

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


# Broadcast function to ensure all ranks have same data
def broadcast_batch(x, y, src=0):
    """Ensure all TP ranks have the same batch"""
    if dist.is_initialized():
        dist.broadcast(x, src=src)
        dist.broadcast(y, src=src)
    return x, y



# train_loader = DataLoader(B=TrainingConfig.batch_size, T=ModelConfig.block_size, file_path="train.bin", device=device)

train_loader = DPDataLoader(
    B=TrainingConfig.batch_size, 
    T=ModelConfig.block_size, 
    file_path="train.bin", 
    device=device,
    dp_rank=topology["dp_rank"],
    dp_size=topology["dp_size"]
)

# 🔧 ADD THESE FUNCTIONS HERE:
def broadcast_batch_tp(x, y, tp_group, src_in_group=0):
    """Broadcast batch within TP group (one model replica)"""
    if dist.is_initialized():
        dist.broadcast(x, src=src_in_group, group=tp_group)
        dist.broadcast(y, src=src_in_group, group=tp_group)
    return x, y

def get_next_batch(train_loader, topology, B, T, device):
    """Get next batch with TP-aware broadcasting"""
    if topology["is_tp_leader"]:  # Only TP leader loads data per model replica
        x, y = train_loader.next_batch()
    else:
        # Other TP shards allocate empty tensors
        x = torch.empty(B, T, dtype=torch.long, device=device)
        y = torch.empty(B, T, dtype=torch.long, device=device)
    
    # Broadcast within TP group so all shards of same model replica get same data
    x, y = broadcast_batch_tp(x, y, topology["tp_group"], src_in_group=0)
    return x, y


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
# assert total_batch_size % (B * T *world_size) == 0, "make sure total_batch_size is divisible by B * T * world_size"
# grad_accum_steps = total_batch_size // (B * T *world_size)

# TP-only: batch size is NOT multiplied by world_size
assert total_batch_size % (B * T) == 0, \
    f"total_batch_size {total_batch_size} must be divisible by B*T = {B}*{T}"
grad_accum_steps = total_batch_size // (B * T)

if master_process:
    print(f"Grad accum steps: {grad_accum_steps}")

#___________CREATE YOUR MODEL_____________

# fsdp_wrap_policy = ModuleWrapPolicy({Block})

# mp_policy = MixedPrecision(
#     param_dtype=torch_dtype,
#     reduce_dtype=torch_dtype,
#     buffer_dtype=torch_dtype,
# )

# model = LLM(ModelConfig , tp_group=tp_group).to(device)

# 4. Continue with model creation and training
model = LLM(ModelConfig, tp_group=topology["tp_group"]).to(device)

if topology["is_global_leader"]: 
    total, active = model.get_num_params()
    print(f"total parameters = {total:,}, active parameters = {active:,}")
    if model.print_fused_adamw: print("Using Fused AdamW")
    if model.print_act_recomp: print("Using Activation Recomputation")

# Compile model
# if topology["is_global_leader"]: print("Using compiled model")
# model = torch.compile(model)
# Compile model
if topology["is_global_leader"]: 
    print("Using compiled model")
    model = torch.compile(model)


# (optional) compile
# model = torch.compile(model)


# Wrap with DDP
from torch.nn.parallel import DistributedDataParallel as DDP
import contextlib


print(f"Before DDP - Model type: {type(model)}")
print(f"Before DDP - Model device: {next(model.parameters()).device}")


# Wrap with DDP **only** if dp_size > 1
is_ddp = topology["dp_size"] > 1
print('is_dp = ',is_ddp)
print('is_dp = ',is_ddp)
print('is_dp = ',is_ddp)
print('is_dp = ',is_ddp)
print('is_dp = ',is_ddp)
print('is_dp = ',is_ddp)
use_ddp = topology["dp_size"] > 1


if use_ddp:
    print(f'Using DDP with dp_size={topology["dp_size"]}')
    from torch.nn.parallel import DistributedDataParallel as DDP
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        process_group=topology["dp_group"],
        find_unused_parameters=True,   # Keep True for MoE
        static_graph=False,
        gradient_as_bucket_view=True,
    )
else:
    print(f'Running without DDP (dp_size={topology["dp_size"]})')

# 🔧 FIX: Create optimizer based on whether DDP is used
if use_ddp:
    # Model is DDP-wrapped - use model.module
    optimizer = model.module.configure_optimizers(
        weight_decay=0.1,
        learning_rate=TrainingConfig.learning_rate, 
        device=device
    )
else:
    # Model is not DDP-wrapped - use model directly
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=TrainingConfig.learning_rate, 
        device=device
    )


print(f"After DDP - Model type: {type(model)}")
print(f"After DDP - Has no_sync: {hasattr(model, 'no_sync')}")
print(f"After DDP - Has module: {hasattr(model, 'module')}")


model_to_opt = model.module if is_ddp else model


# Verify DDP wrapping worked
print(f"Model type after DDP: {type(model)}")
print(f"Has no_sync: {hasattr(model, 'no_sync')}")



# 4) Get a safe handle to the underlying module (NO NAME COLLISIONS)
#    Use a fresh, unique variable name that you haven't used as a function anywhere.
base_model = model.module if isinstance(model, DDP) else model




# 5) Sanity checks before calling methods
if not hasattr(base_model, "configure_optimizers"):
    # Fallback: create the optimizer directly from params to avoid attribute issues
    if master_process:
        print("Warning: base_model has no 'configure_optimizers'; creating AdamW directly.")
    decay_params   = [p for p in base_model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in base_model.parameters() if p.requires_grad and p.dim() <  2]
    optim_groups = [
        {"params": decay_params,   "weight_decay": 0.1},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    try:
        optimizer = torch.optim.AdamW(optim_groups, lr=TrainingConfig.learning_rate, fused=True)
    except TypeError:
        optimizer = torch.optim.AdamW(optim_groups, lr=TrainingConfig.learning_rate)
else:
    # optimizer = base_model.configure_optimizers(
    optimizer = model_to_opt.configure_optimizers(
        weight_decay=0.1,
        learning_rate=TrainingConfig.learning_rate,
        device=device,
    )



# 6) Use base_model for model-specific attributes/logging
total, active = base_model.get_num_params()
if master_process:
    print(f"total parameters = {total:,}, acitive parameters = {active:,}")
    if getattr(base_model, "print_fused_adamw", False): print("Using Fused AdamW")
    if getattr(base_model, "print_act_recomp", False):  print("Using Activation Recomputation")

# 7) Use DDP no_sync only when wrapped
for micro_step in range(grad_accum_steps):
    # # sync_context = model.no_sync() if micro_step < grad_accum_steps - 1 else contextlib.nullcontext()
    # if is_ddp and micro_step < grad_accum_steps - 1:
    #     sync_context = model.no_sync()
    # else:
    #     sync_context = contextlib.nullcontext()
    # 🔧 Use simple sync context (no DDP optimization)
    sync_context = contextlib.nullcontext()

    # sync_context = model.no_sync() if use_ddp and micro_step < grad_accum_steps - 1 else contextlib.nullcontext()
    with sync_context:
        x, y = get_next_batch(train_loader, topology, B, T, device)
        with torch.cuda.amp.autocast(dtype=torch_dtype):
            _, loss, _ = model(x, y)
            loss = loss / grad_accum_steps
        scaler.scale(loss).backward()




if master_process : 
    total, active = model.get_num_params()
    print(f"total parameters = {total:,}, acitive parameters = {active:,}")
    if model.print_fused_adamw: print("Using Fused AdamW")
    if model.print_act_recomp: print("Using Activation Recomputation")

# model = FSDP(
#     model,
#     auto_wrap_policy=fsdp_wrap_policy,
#     mixed_precision=mp_policy,
#     sharding_strategy=ShardingStrategy.FULL_SHARD, # This is ZeRO-3
#     device_id=torch.cuda.current_device(),
#     # cpu_offload=CPUOffload(offload_params=True), # Optional: to save even more GPU memory
#     limit_all_gathers=True, # Recommended for performance
#     use_orig_params=True, # Important for optimizers like AdamW and for getting original parameters
#     sync_module_states=True,
# )

if master_process : print("Using compiled model")
# model = torch.compile(model)

# raw_model:LLM = model
# Use a unified handle for optimizer creation and later .module access
# raw_model = model.module if isinstance(model, DDP) else model
# Always get a handle to the underlying nn.Module (works for DDP and non-DDP)

# 🔧 CORRECT: Use model.module.configure_optimizers directly
# optimizer = model.module.configure_optimizers(
#     weight_decay=0.1,
#     learning_rate=TrainingConfig.learning_rate, 
#     device=device
# )




#______________________________________________ TRAINING ______________________________________________

# Optimizer - use underlying TP model
# optimizer = raw_model.module.configure_optimizers(
#     weight_decay=0.1,
#     learning_rate=TrainingConfig.learning_rate, 
#     device=device
# )

# Initialize scaler
scaler = torch.cuda.amp.GradScaler()

# Import contextlib for no_sync
import contextlib

for iter in range(TrainingConfig.max_iters+1):
    t0 = perf_counter()

    lr = get_lr(iter, TrainingConfig)
    for param_grp in optimizer.param_groups:
        param_grp['lr'] = lr
    
    optimizer.zero_grad(set_to_none=True)

    # Evaluation
    if TrainingConfig.eval and (iter % TrainingConfig.eval_interval == 0 or iter == TrainingConfig.max_iters) and iter != 0:
        a = perf_counter()
        losses = estimate_loss(model, TrainingConfig, train_loader, val_loader)
        b = perf_counter()
        if topology["is_global_leader"]:
            print(f"--------val run-------- train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | dt {1000*(b-a):.4f}ms")
        t0 = b

    # Training step with DP×TP optimization
    for micro_step in range(grad_accum_steps):
        # sync_context = model.no_sync() if micro_step < grad_accum_steps - 1 else contextlib.nullcontext()

        if use_ddp:
            sync_context = model.no_sync() if micro_step < grad_accum_steps - 1 else contextlib.nullcontext()
        else:
            sync_context = contextlib.nullcontext()  # No DDP optimization
        
        with sync_context:
            x, y = get_next_batch(train_loader, topology, B, T, device)
            
            with torch.cuda.amp.autocast(dtype=torch_dtype):
                _, loss, _ = model(x, y)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

    if TrainingConfig.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.grad_clip)

    scaler.step(optimizer)
    scaler.update()    

    if topology["is_global_leader"]:
        torch.cuda.synchronize()
        mem = torch.cuda.memory_reserved()
        dt  = (perf_counter()-t0)*1000
        print(f"step: {iter} | train loss:{loss.item()*grad_accum_steps:.4f} | dt: {dt:.2f}ms | grad_accum_steps: {grad_accum_steps} | GPU RAM: {mem/1024**3:.2f}GB")

# Cleanup
if dist.is_initialized():
    dist.destroy_process_group()
