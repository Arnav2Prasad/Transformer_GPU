%%writefile my_code.py




parallel_flag = 1
print('parallel_flag : ', parallel_flag)

# 1 for plain DP
#  2 for zero1
#  3 for zero1 and 2
# 4 for fsdp 
# 5 for TP
# 6 for EP
# 7 for PP 
#  8 for cp

import wandb
from datetime import datetime


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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import time

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
import torch.multiprocessing as mp
from contextlib import nullcontext
from packaging import version
import os
import glob
import torch.distributed as dist  
import sys
import gc
import torch
import torch.distributed as dist

import sys
import gc
import torch
import torch.distributed as dist
from time import perf_counter
from math import ceil
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import torch
import numpy as np




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist



from packaging import version

import glob

from math import ceil

from typing import Literal
from dataclasses import dataclass 

from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.amp import autocast, GradScaler




import warnings
from typing import Literal, Optional, Dict, List, Tuple
from time import perf_counter
from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

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


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

from torch.profiler import profile, record_function, ProfilerActivity, schedule

os.environ['WANDB_API_KEY'] = 'c78410b3a816898642987ae3c3899430080b89d1'



warnings.filterwarnings("ignore")

def create_profiler(output_dir="./profiler_logs", trace_handler=None):
    """
    Create a PyTorch profiler with customizable settings
    
    Args:
        output_dir: Directory to save profiler traces
        trace_handler: Custom trace handler function (optional)
    """

    os.makedirs(output_dir, exist_ok=True)
    
    # Default trace handler if none provided
    if trace_handler is None:
        def default_trace_handler(prof):
            # Export Chrome trace for visualization
            prof.export_chrome_trace(f"{output_dir}/trace_{prof.step_num}.json")
            
            # Print table summary
            print(prof.key_averages().table(
                sort_by="cuda_time_total", 
                row_limit=10
            ))
        
        trace_handler = default_trace_handler
    
    # Create profiler with schedule
    return profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=schedule(
            wait=1,      # Skip first iteration (warmup)
            warmup=1,    # Warmup for 1 iteration
            active=3,    # Profile for 3 iterations
            repeat=2     # Repeat the cycle 2 times
        ),
        on_trace_ready=trace_handler,
        record_shapes=True,       # Record tensor shapes
        profile_memory=True,      # Track memory allocations
        with_stack=True,          # Record stack traces
        with_flops=True,          # Estimate FLOPs
    )


def save_checkpoint_with_wandb(model, optimizer, iter, loss, config, use_wandb, rank=0):
    """Save checkpoint and optionally log as WandB artifact"""
    if not master_process:
        return
    
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_iter_{iter}_{timestamp}.pt")
    
    # Save checkpoint
    if parallel_flag == 5 or parallel_flag == 6:
        raw_model = model
    else:
        raw_model = model.module
    
    checkpoint = {
        'iteration': iter,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'loss': loss.detach().item() if loss else 0.0,
        'timestamp': timestamp,
    }
    
    torch.save(checkpoint, checkpoint_file)
    print(f"Checkpoint saved to: {checkpoint_file}")
    
    # Log as WandB artifact
    if use_wandb:
        artifact = wandb.Artifact(
            name=f"model-checkpoint-iter-{iter}",
            type="model",
            description=f"Model checkpoint at iteration {iter}, loss: {loss.item()*grad_accum_steps:.4f}" if loss else f"Model checkpoint at iteration {iter}"
        )
        artifact.add_file(checkpoint_file)
        wandb.log_artifact(artifact)



# ============================================================================
# ZERO-2 IMPLEMENTATION: Custom Gradient Sharding
# ============================================================================


'''
In standard DDP, every GPU holds a full copy of the model, optimizer states, and gradients. ZeRO-2 eliminates the memory redundancy of optimizer states and gradients.
'''




class ZeRO2GradientHandler:
    """
    Implements ZeRO-2 gradient sharding and reduction.
    Each rank owns a shard of gradients and is responsible for updating
    the corresponding parameters.
    """
    def __init__(self, model: nn.Module, process_group=None):
        # Purpose : Distributed Setup
        # Working
        # Initializes standard DDP properties: the model being trained, 
        # the communication group (defaults to the entire WORLD), the total number of GPUs, and the current GPU's identifier.
        # 4 GPUs → world_size=4, ranks: 0,1,2,3
        self.model = model
        self.process_group = process_group if process_group else dist.group.WORLD
        self.world_size = dist.get_world_size(self.process_group)
        self.rank = dist.get_rank(self.process_group)
        
        # Partition parameters across ranks
        # Filters the model to include only trainable parameters.
        self.param_to_rank = {}             ## Maps parameter → owning GPU
        self.rank_to_params = {i: [] for i in range(self.world_size)}       # Maps GPU → parameters it owns

        '''
        Sharding Logic: This is the core partitioning algorithm. It iterates through all trainable parameters and assigns them to ranks in a round-robin fashion.
        Example : If world_size=4, parameters 1,5,9... go to rank=1, and 4,8,12... go to rank=0.
        '''
        
        params_with_grad = [p for p in model.parameters() if p.requires_grad]
        for idx, param in enumerate(params_with_grad):
            rank = idx % self.world_size        # Round-robin assignment


            # Mapping Storage
            '''
            Stores two essential dictionaries: 
            1) Which rank owns a specific parameter. 
            2) A list of parameters owned by each rank. 
            A rank is designated as the "owner" of a parameter's optimizer state and gradient.
            '''
            self.param_to_rank[param] = rank
            self.rank_to_params[rank].append(param)

        '''
        What this does:

        Divides ALL model parameters across all GPUs
        Each parameter is assigned to exactly ONE GPU (its "owner")
        Uses round-robin: param0→GPU0, param1→GPU1, param2→GPU2, param3→GPU3, param4→GPU0, etc.
        Visual Example (4 GPUs, 8 parameters):

        text
        GPU0 (Rank 0): [param0, param4]  # Owner of these parameters
        GPU1 (Rank 1): [param1, param5]  
        GPU2 (Rank 2): [param2, param6]
        GPU3 (Rank 3): [param3, param7]
        '''
        

        # Bucket Setup (Placeholder)
        '''
        Initializes structures for managing gradient transfers. 
        This specific implementation is simplified and doesn't fully utilize gradient buckling for communication overlap, 
        but it acknowledges the necessity of grouping gradients.
        '''
        # Storage for gradient buckets
        self.grad_buckets = {i: [] for i in range(self.world_size)}
        self.bucket_size = 25 * 1024 * 1024  # 25MB buckets



        
    def reduce_gradients(self):
        """
        Reduce gradients using reduce-scatter operation.
        Each rank will own a shard of the full gradients.
        """

        # Step A: Prepare Gradients by Owner
        # Group parameters by owning rank
        for rank in range(self.world_size):
            params = self.rank_to_params[rank]          # Get parameters owned by this rank
            if not params:
                continue
                
            # Flatten gradients for this rank's parameters
            grads = []
            for p in params:
                if p.grad is not None:
                    grads.append(p.grad.data.flatten())         # Convert to 1D
                else:
                    grads.append(torch.zeros_like(p.data.flatten()))             # Zero if no grad
            '''
            What happens:
                    For each GPU rank, gather ALL gradients for parameters it OWNS
                    Flatten each gradient tensor into 1D arrays
                    If a parameter has no gradient (rare), use zeros


            Example:
                GPU0 preparing gradients for parameters it OWNS:
                - param0.grad: [0.1, 0.2, 0.3] → flattened
                - param4.grad: [0.4, 0.5] → flattened  
                - Concatenated: [0.1, 0.2, 0.3, 0.4, 0.5]
            '''
            
            if grads:
                
                # Concatenate all gradients for this rank
                grad_buffer = torch.cat(grads)          # Single large 1D tensor
                
                # Reduce-scatter: sum gradients across all ranks, 
                # but each rank only keeps its shard
                '''
                dist.reduce() Operation:

                All GPUs send their local gradients for these parameters to the owner GPU
                Owner GPU sums all received gradients
                Result: Owner GPU has the globally averaged gradients

                Example:
                    Before reduce:
                        - GPU0: param0.grad = [0.1, 0.2]  (from its local batch)
                        - GPU1: param0.grad = [0.3, 0.4]  (from its local batch) 
                        - GPU2: param0.grad = [0.5, 0.6]  (from its local batch)
                        - GPU3: param0.grad = [0.7, 0.8]  (from its local batch)

                        After dist.reduce() to GPU0:
                        - GPU0: param0.grad = [1.6, 2.0]  # Sum of all GPUs: [0.1+0.3+0.5+0.7, 0.2+0.4+0.6+0.8]
                        - Other GPUs: discard their param0 gradients
                '''
                if self.world_size > 1:
                    dist.reduce(grad_buffer, dst=rank, 
                              op=dist.ReduceOp.SUM, 
                              group=self.process_group)
                
                # Unflatten back to parameters (only on owning rank)
                if self.rank == rank:           # If I am the owner of these parameters
                    offset = 0
                    for p in params:
                        if p.requires_grad:
                            numel = p.numel()
                            
                            # Ensure gradient exists
                            if p.grad is None:
                                p.grad = torch.zeros_like(p.data)
                            
                            # Copy from buffer
                            p.grad.data.copy_(
                                grad_buffer[offset:offset + numel].view_as(p.data)
                            )
                            offset += numel
                        else:
                            # For non-trainable parameters, still count their size in buffer
                            offset += p.numel()
                else:
                    # Non-owning ranks
                    for p in params:
                        # Only clear gradients for trainable parameters
                        if p.requires_grad:
                            p.grad = None

            '''
            What happens:
                Owner GPU: Takes the summed gradients and distributes them back to individual parameters
                Non-owner GPUs: Discard gradients (they don't need them for optimization)
            '''
    
    def sync_parameters(self):
        """
        Broadcast parameters from their owning rank to all other ranks.
        Called after optimizer step to synchronize model parameters.
        """
        '''
        What this does:

        After optimizer updates parameters on owner GPU...
        Owner GPU broadcasts updated parameters to all other GPUs
        Ensures all GPUs have the same model parameters
        '''
        for rank in range(self.world_size):
            params = self.rank_to_params[rank]
            for p in params:
                if self.world_size > 1:
                    dist.broadcast(p.data, src=rank, group=self.process_group)



# This ZeRO2Optimizer class is the orchestrator that coordinates ZeRO-1 and ZeRO-2
class ZeRO2Optimizer:
    """
    Wrapper around optimizer that implements ZeRO-1 and ZeRO-2.
    - ZeRO-1: Optimizer states are sharded (via ZeroRedundancyOptimizer)
    - ZeRO-2: Gradients are sharded (via custom reduce-scatter)
    """
    def __init__(self, optimizer, gradient_handler: ZeRO2GradientHandler):
        '''
        self.optimizer: A ZeRO-1 enabled optimizer (shards optimizer states)
        self.gradient_handler: A ZeRO-2 enabled gradient handler (shards gradients)
        '''
        self.optimizer = optimizer
        self.gradient_handler = gradient_handler
        
    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
    

    # The Core ZeRO-2 Algorithm
    def step(self):
        # First, reduce and shard gradients (ZeRO-2)
        self.gradient_handler.reduce_gradients()
        '''
        What happens:
        Collect gradients from all GPUs for each parameter
        Average them across all GPUs
        Shard results - each GPU keeps only gradients for parameters it owns

        Before reduce_gradients():

            
            GPU0: [∇A₁, ∇A₂, ∇A₃, ∇A₄]  # Full local gradients
            GPU1: [∇B₁, ∇B₂, ∇B₃, ∇B₄]  
            GPU2: [∇C₁, ∇C₂, ∇C₃, ∇C₄]
            GPU3: [∇D₁, ∇D₂, ∇D₃, ∇D₄]
            After reduce_gradients():

            text
            GPU0: [∇₁_avg, -, -, -]    # Only keeps averaged ∇₁
            GPU1: [-, ∇₂_avg, -, -]    # Only keeps averaged ∇₂  
            GPU2: [-, -, ∇₃_avg, -]    # Only keeps averaged ∇₃
            GPU3: [-, -, -, ∇₄_avg]    # Only keeps averaged ∇₄
        '''


        '''
        What happens:
        The underlying ZeroRedundancyOptimizer performs the update
        ZeRO-1: Each GPU only has optimizer states for parameters it owns
        ZeRO-2: Each GPU only has gradients for parameters it owns
        '''
        # Step optimizer (only updates parameters owned by this rank)
        self.optimizer.step()

        '''
        What happens:

            Each GPU broadcasts the parameters it updated to all other GPUs
            Ensures all GPUs have identical model parameters

            Before sync:

            text
            GPU0: [updated_param1, old_param2, old_param3, old_param4]
            GPU1: [old_param1, updated_param2, old_param3, old_param4]  
            GPU2: [old_param1, old_param2, updated_param3, old_param4]
            GPU3: [old_param1, old_param2, old_param3, updated_param4]
            After sync:
            ALL GPUs: [updated_param1, updated_param2, updated_param3, updated_param4]
        '''
        
        # Sync parameters across all ranks
        self.gradient_handler.sync_parameters()
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups
        '''
        Purpose: Makes ZeRO2Optimizer behave like a regular optimizer

            Learning rate schedulers can access optimizer.param_groups
            Training loops can modify learning rates normally
        '''


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================





# we added the below code
def _get_group_and_ranks(tp_group = None):
    """Get TP group, world size, and rank - safer version""" 
    if not dist.is_initialized():
        print('inside if not dist.is_initialized() ')
        return None ,1, 0
    
    tp_group = tp_group or dist.group.WORLD

    return tp_group , dist.get_world_size(tp_group) , dist.get_rank(tp_group)



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
        
        
        x_re_out = x_re*freqs_cis.real - x_im*freqs_cis.imag    # (B, T, H, hs//2) * (1, T, 1, hs//2) - (B, T, H, hs//2) * (1, T, 1, hs//2) -> (B, T, H, hs//2)
        x_im_out = x_re*freqs_cis.imag + x_im*freqs_cis.real    # (B, T, H, hs//2) * (1, T, 1, hs//2) + (B, T, H, hs//2) * (1, T, 1, hs//2) -> (B, T, H, hs//2)
        
        # Stack the real and imaginary parts back together
        x_out = torch.stack([x_re_out, x_im_out], dim=-1).flatten(3) # (B, T, H, hs//2), (B, T, H, hs//2) -> (B, T, H, hs)

        return x_out.type_as(x)





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










class Attention(nn.Module):
    """ Routes the attention mechanism according to the config"""

    def __init__(self, config:LLMconfig, tp_group=None):  # ← ADDED tp_group parameter
        super().__init__()
        self.config = config
        if config.attn in ('mha','mqa','gqa'):
            self.attn = GQA(config, tp_group=tp_group)  # ← Pass tp_group
        else:
            raise NotImplementedError("Only GQA supported")
        
        
        # elif config.attn == 'mla':
        #     if config.pos_emb != 'rope':
        #         self.attn = NaiveMHLA(config)
        #     else:
        #         self.attn = FullMHLA(config)

                
    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor|None = None, kv_cache=None, VAL_RUN=False):
        return self.attn(x, freqs_cis, kv_cache, VAL_RUN)





class MLP(nn.Module):
    """ A simple feed-forward network block. """
    def __init__(self, config: LLMconfig, tp_group=None, enable_tp=True):
        super().__init__()
        self.config = config
        self.non_linearity = config.non_linearity.lower()
        
        # Common dropout layer
        self.dropout = nn.Dropout(config.dropout)
        
        # Set up TP if applicable
        if parallel_flag == 5:
            self.tp_group, self.tp_size, self.tp_rank = _get_group_and_ranks(tp_group)
            self.enable_tp = (
                enable_tp and 
                (tp_group is not None) and 
                (self.tp_size > 1) and 
                dist.is_initialized()
            )
        else:
            self.enable_tp = False
        
        # Common activation function mapping
        def get_activation_func(non_linearity):
            activation_map = {
                'relu': nn.ReLU(), 'gelu': nn.GELU(), 'swish': nn.SiLU(), 'mish': nn.Mish(),
                'silu': nn.SiLU(), 'selu': nn.SELU(), 'celu': nn.CELU(), 'elu': nn.ELU(),
                'glu': nn.GLU(), 'sigmoid': nn.Sigmoid(), 
                'lrelu': nn.LeakyReLU(negative_slope=0.01), 'tanh': nn.Tanh()
            }
            return activation_map.get(non_linearity, nn.GELU())
        
        # Setup layers based on configuration
        self.setup_layers(config, get_activation_func)
    
    def setup_layers(self, config, get_activation_func):
        """Setup the MLP layers based on parallel configuration and activation type"""
        
        # For SwiGLU activation
        if self.non_linearity == 'swiglu':
            self.setup_swiglu_layers(config)
        else:
            # For other activations
            if parallel_flag == 5 and self.enable_tp:
                # Tensor Parallel path for standard activations
                self.c_fc = ColumnParallelLinear(
                    config.n_embd, config.up_dim, bias=False,
                    gather_output=False, group=self.tp_group
                )
                self.non_linearity_func = get_activation_func(self.non_linearity)
                self.c_proj = RowParallelLinear(
                    config.up_dim, config.n_embd, bias=False,
                    input_is_parallel=True, group=self.tp_group
                )
            elif parallel_flag == 5 and not self.enable_tp:
                # Non-TP fallback in parallel_flag==5
                self.c_fc = nn.Linear(config.n_embd, config.up_dim, bias=False)
                self.non_linearity_func = get_activation_func(self.non_linearity)
                self.c_proj = nn.Linear(config.up_dim, config.n_embd, bias=False)
            else:
                # Standard non-parallel path
                self.c_fc = nn.Linear(config.n_embd, config.up_dim, bias=False)
                self.non_linearity_func = get_activation_func(self.non_linearity)
                self.c_proj = nn.Linear(config.up_dim, config.n_embd, bias=False)
    
    def setup_swiglu_layers(self, config):
        """Setup layers specifically for SwiGLU activation"""
        if parallel_flag == 5 and self.enable_tp:
            # Tensor Parallel path for SwiGLU
            self.c_fc = ColumnParallelLinear(
                config.n_embd, 2 * config.up_dim, bias=False,
                gather_output=False, group=self.tp_group
            )
            self.c_proj = RowParallelLinear(
                config.up_dim, config.n_embd, bias=False,
                input_is_parallel=True, group=self.tp_group
            )
        else:
            # Non-TP or standard path for SwiGLU
            self.c_fc = nn.Linear(config.n_embd, 2 * config.up_dim, bias=False)
            self.c_proj = nn.Linear(config.up_dim, config.n_embd, bias=False)
    
    def forward(self, x):
        """Forward pass with common logic for all configurations"""
        
        # Common forward logic for SwiGLU
        if self.non_linearity == 'swiglu':
            return self.forward_swiglu(x)
        else:
            return self.forward_standard(x)
    
    def forward_swiglu(self, x):
        """Forward pass for SwiGLU activation"""
        # Chunk the output of c_fc into two parts for SwiGLU
        x1, x2 = self.c_fc(x).chunk(2, dim=-1)
        x = F.silu(x1) * x2
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
    def forward_standard(self, x):
        """Forward pass for standard activations"""
        x = self.c_fc(x)
        x = self.non_linearity_func(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x





class Expert(nn.Module):
    """ A single feed-forward network expert. """
    def __init__(self, config:LLMconfig):
        super().__init__()
        self.expert = MLP(config, tp_group =None , enable_tp=False)
        
    def forward(self, x):
        return self.expert(x)


'''
The class handles the mapping between global expert IDs and their local placement across different GPU ranks in an Expert Parallel setup.
'''

class EPLayout:
    """Manages expert distribution across EP ranks"""
    def __init__(self, n_routed, world_size, rank):
       

        # Total number of routed experts
        self.n_routed = n_routed


        # Total number of GPUs
        self.world_size = world_size


        # Current GPU rank (0, 1, 2, ...)
        self.rank = rank

        '''
        Uses ceil() to ensure all experts are assigned, even if not perfectly divisible
        Example: 10 experts, 3 GPUs → ceil(10/3) = 4 experts per GPU
        '''
        self.n_local = ceil(n_routed / world_size)

        # First expert on this GPU
        self.start = self.n_local * rank


        # Last expert (+1)
        self.end = min(self.start + self.n_local, n_routed)

        # Local expert IDs
        self.local_global_ids = list(range(self.start, self.end))


    
    # Find Expert Owner
    # Given a global expert ID, find which GPU owns it
    def owner_rank(self, gid: int) -> int:
        return min(gid // self.n_local, self.world_size - 1)


    # Convert to Local Index
    # Convert global expert ID to local index within GPU
    def local_index(self, gid: int) -> int:
        return gid - self.start





class MoE(nn.Module):
    '''
    This class implements the DeepSeekMoE layer, featuring shared and routed experts.
    It uses an Auxiliary-Loss-Free load balancing strategy with a dynamic bias term.
    Ref: https://arxiv.org/pdf/2412.19437
    '''

    def __init__(self, config: LLMconfig):
        super().__init__()
        self.config = config
        
        # Common initialization for all configurations
        self.n_shared = config.n_shared
        self.n_routed = config.n_exp - config.n_shared
        self.n_act_routed = config.n_act - config.n_shared
        
        # Common validation
        assert self.n_act_routed > 0, "Number of active experts must be greater than shared experts"
        
        # Setup based on parallel configuration
        if parallel_flag == 6:
            self.setup_expert_parallel(config)
        else:
            self.setup_single_gpu(config)
    
    def setup_expert_parallel(self, config):
        """Setup for expert parallel configuration"""
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
        
        # Scenario: When all experts are shared (no routing needed)
        if self.n_routed == 0:
            self.shared_only = True
            if self.rank == 0:
                self.shared_experts = nn.ModuleList([Expert(config) for _ in range(self.n_shared)])
            else:
                self.shared_experts = nn.ModuleList()
            return
        else:
            self.shared_only = False
        
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
    
    def setup_single_gpu(self, config):
        """Setup for single GPU configuration"""
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
        """Main forward method that routes to appropriate implementation"""
        if parallel_flag == 6:
            return self.forward_expert_parallel(x)
        else:
            return self.forward_single_gpu_fallback(x)
    
    def forward_expert_parallel(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for expert parallel configuration"""
        # Early return for shared-only layers
        if self.shared_only:
            return self._forward_shared_only(x)
        
        # Short-circuit for single GPU in EP mode
        if not self.use_ep:
            return self._forward_single_gpu_fallback_ep(x)
        
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
            
            # Compute routing with common routing logic
            router_logits, topk_gates, topk_indices, aux_loss = self.compute_routing(
                x_flat, n_tokens, device, self.rank == 0
            )
            
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
            router_logits = None
        
        # Perform all-to-all communication and expert processing
        y_combined, aux_loss = self._perform_all_to_all_communication(
            x, xs_sorted, gs_sorted, local_indices, restore_idx, counts, 
            shared_out, aux_loss, router_logits, self.rank == 0
        )
        
        return y_combined, aux_loss
    
    def _perform_all_to_all_communication(self, x, xs_sorted, gs_sorted, local_indices, 
                                        restore_idx, counts, shared_out, aux_loss, 
                                        router_logits, is_rank_0):
        """Handle all-to-all communication and expert processing"""
        B, T, C = x.shape
        device = x.device
        dtype = x.dtype
        
        # 2) CORRECTED: Build send-counts matrix S (sender x dest)
        S = torch.zeros(self.world_size, self.world_size, device=device, dtype=torch.long)
        if is_rank_0:
            S[0] = counts  # row 0: rank 0 sends to all dests
        
        dist.broadcast(S, src=0, group=self.ep_group)

        # Safe assertion with distributed abort
        if is_rank_0:
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
            y_local = self._process_local_experts(recv_tokens, recv_gates, recv_local_indices)

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
        if is_rank_0:
            # Restore original order
            y_routed = got_back[restore_idx]
            # Combine with shared output
            y_combined = (shared_out + y_routed).view(B, T, C)
            return y_combined, aux_loss
        else:
            # Other ranks return zeros (they don't contribute to final output)
            return torch.zeros(B, T, C, device=device, dtype=dtype), torch.tensor(0.0, device=device)
    
    def _process_local_experts(self, tokens, gates, local_indices):
        """Process tokens through local experts with optimized bucketing"""
        y_local = torch.zeros_like(tokens)
        
        # Safety check for local indices
        self._assert((local_indices < len(self.local_routed_experts)).all(),
                    f"Local index out of bounds: {local_indices.max()} >= {len(self.local_routed_experts)}")
        
        # OPTIMIZED: Bucket by local expert using sorting for better cache locality
        sorted_indices = torch.argsort(local_indices)
        recv_lidx_sorted = local_indices[sorted_indices]
        recv_tokens_sorted = tokens[sorted_indices]
        recv_gates_sorted = gates[sorted_indices]
        
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
        
        return y_local
    
    def compute_routing(self, x_flat, n_tokens, device, is_rank_0=True):
        """Common routing logic used by both EP and single GPU"""
        if not is_rank_0:
            return None, None, None, torch.tensor(0.0, device=device)
        
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
        
        return router_logits, topk_gates, topk_indices, aux_loss
    
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
    
    def _forward_single_gpu_fallback_ep(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fallback for single GPU in EP mode"""
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        n_tokens = x_flat.shape[0]
        
        # Shared experts
        shared_out = torch.zeros_like(x_flat)
        if self.n_shared > 0:
            for expert in self.shared_experts:
                shared_out += expert(x_flat)
        
        # Compute routing
        router_logits, topk_gates, topk_indices, aux_loss = self.compute_routing(
            x_flat, n_tokens, x.device, self.rank == 0
        )
        
        if router_logits is None:
            return torch.zeros_like(x), torch.tensor(0.0, device=x.device)
        
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
    
    def forward_single_gpu_fallback(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for single GPU configuration"""
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # Shape: (B*T, C)
        n_tokens = x_flat.shape[0]

        # ___________ SHARED EXPERT PATH ___________
        shared_output = torch.zeros_like(x_flat)
        if self.n_shared > 0:
            for i in range(self.n_shared):
                shared_output += self.experts[i](x_flat)  # bypass the router

        # ___________ ROUTED EXPERT PATH ___________
        router_logits = self.gate(x_flat).float()

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
                    self.expert_bias += (self.config.gamma * delta)

            router_probs = F.softmax(router_logits, dim=1)
            pi = router_probs.mean(dim=0)
            aux_loss = self.config.alpha * self.n_routed * torch.sum(pi * fi)

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
        if parallel_flag == 6 and self.rank != 0:
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
        if parallel_flag == 6 and self.rank != 0:
            # Workers only load their local experts
            filtered_state_dict = {k: v for k, v in state_dict.items() if '.local_routed_experts.' in k}
            return super().load_state_dict(filtered_state_dict, strict=False)
        else:
            # Rank 0 or single GPU loads everything
            return super().load_state_dict(state_dict, strict=strict)




class Block(nn.Module):
    """ A single Transformer block combining attention and MLP. """
    def __init__(self, config: LLMconfig, tp_group=None):
        super().__init__()
        self.config = config
        self.is_moe = config.moe
        self.act_recomp = config.act_recomp
        
        # Common components for all configurations
        self.attn = Attention(config, tp_group=tp_group)  # Pass tp_group to Attention
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        # Initialize EP attributes if needed
        if parallel_flag == 6:
            self._init_expert_parallel_attributes(config)
        
        # Initialize FFN (MLP or MoE)
        self._init_ffn(config, tp_group)
    
    def _init_expert_parallel_attributes(self, config):
        """Initialize expert parallel attributes if they don't exist"""
        if not hasattr(config, 'ep_rank'):
            config.ep_rank = 0
        if not hasattr(config, 'ep_size'):
            config.ep_size = 1
        if not hasattr(config, 'ep_group'):
            config.ep_group = None
    
    def _init_ffn(self, config, tp_group):
        """Initialize the feed-forward network (MLP or MoE)"""
        if config.moe:
            self.moe = MoE(config)
            self.mlp = None
        else:
            self.mlp = MLP(config, tp_group=tp_group, enable_tp=True)
            self.moe = None
    
    def _attention_forward(self, x_norm, freqs_cis, kv_cache, VAL_RUN):
        """Common attention forward pass logic"""
        if parallel_flag == 7:
            # Special handling for parallel_flag 7
            def _attn_call(z, freqs, kv):
                return self.attn(z, freqs, kv, VAL_RUN)
            
            if self.act_recomp:
                return checkpoint(_attn_call, x_norm, freqs_cis, kv_cache, use_reentrant=False)
            else:
                return self.attn(x_norm, freqs_cis, kv_cache, VAL_RUN)
        
        elif parallel_flag in [4, 5, 8]:
            # Tensor Parallel, Sequence Parallel, or Context Parallel
            if self.act_recomp:
                return checkpoint(self.attn, x_norm, freqs_cis, kv_cache, VAL_RUN, use_reentrant=False)
            else:
                return self.attn(x_norm, freqs_cis, kv_cache, VAL_RUN)
        
        else:
            # Standard forward pass
            return self.attn.forward(x_norm, freqs_cis, kv_cache, VAL_RUN)
    
    def _ffn_forward(self, x_norm):
        """Common feed-forward network forward pass logic"""
        if self.is_moe:
            moe_output, aux_loss = self.moe(x_norm)
            return moe_output, aux_loss
        else:
            mlp_output = self.mlp(x_norm)
            return mlp_output, 0.0
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor | None = None, 
                kv_cache=None, VAL_RUN=False):
        """
        Forward pass for the Transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, hidden_size)
            freqs_cis: Rotary position embeddings (optional)
            kv_cache: Key-value cache for attention (optional)
            VAL_RUN: Validation run flag (optional)
            
        Returns:
            x: Output tensor
            updated_kv_cache: Updated key-value cache
            aux_loss: Auxiliary loss (for MoE) or 0.0
        """
        # Save residual for first sub-layer
        residual = x
        
        # Layer Norm + Attention
        x_norm = self.ln1(x)
        attn_output, updated_kv_cache = self._attention_forward(x_norm, freqs_cis, kv_cache, VAL_RUN)
        
        # Add residual connection
        x = residual + attn_output
        
        # Save residual for second sub-layer
        residual = x
        
        # Layer Norm + FFN (MLP or MoE)
        x_norm = self.ln2(x)
        ffn_output, aux_loss = self._ffn_forward(x_norm)
        
        # Add residual connection
        x = residual + ffn_output
        
        return x, updated_kv_cache, aux_loss




class LLM(nn.Module):
    """ A simple Large Language Model """
    
    def __init__(self, config: LLMconfig, tp_group=None):
        super().__init__()
        self.config = config
        self.head_size = config.n_embd // config.n_head
        
        # Initialize parallel configuration
        self._init_parallel_config(tp_group)
        
        # Initialize embeddings
        self.tkn_emb = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Initialize positional embeddings
        self._init_positional_embeddings()
        
        # Initialize transformer blocks
        self.transformer = nn.ModuleDict(dict(
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config, tp_group=tp_group) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd)
        ))
        
        # Initialize output head with weight tying
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tkn_emb.weight = self.lm_head.weight
        
        # Apply weight initialization
        self.apply(self._init_weights)
        
        # Initialize flags and settings
        self.VAL_RUN = False
        self.print_act_recomp = config.act_recomp
        self.print_fused_adamw = False
    
    def _init_parallel_config(self, tp_group):
        """Initialize parallel configuration settings"""
        if parallel_flag == 5:
            self.tp_group = tp_group
        elif parallel_flag == 8:
            self.context_parallel_group = None
            if self.config.context_parallel_size > 1:
                self.context_parallel_group = torch.distributed.group.WORLD
            
            # Propagate context parallel group to all modules
            for module in self.modules():
                if hasattr(module, 'context_parallel_group'):
                    module.context_parallel_group = self.config.context_parallel_group
    
    def _init_positional_embeddings(self):
        """Initialize positional embeddings based on configuration"""
        if parallel_flag == 7:
            self._init_positional_embeddings_flag7()
        else:
            self._init_positional_embeddings_standard()
    
    def _init_positional_embeddings_flag7(self):
        """Initialize positional embeddings for parallel_flag == 7"""
        config = self.config
        if config.pos_emb == 'learn':
            self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        elif config.pos_emb == 'sin':
            pos_emb = torch.zeros(config.block_size, config.n_embd)
            position = torch.arange(0, config.block_size, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, config.n_embd, 2).float() * (-math.log(10000.0) / config.n_embd))
            pos_emb[:, 0::2] = torch.sin(position * div_term)
            pos_emb[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pos_emb', pos_emb)
        elif config.pos_emb == 'rope':
            d = config.rope_head_dim if config.attn == 'mla' else config.n_embd // config.n_head
            assert d % 2 == 0
            theta = 1.0 / (10000.0 ** (torch.arange(0, d, 2).float() / d))
            seq = torch.arange(config.block_size)
            freqs = torch.outer(seq, theta)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
            self.register_buffer("freqs_cis", freqs_cis)
    
    def _init_positional_embeddings_standard(self):
        """Initialize positional embeddings for standard configuration"""
        config = self.config
        if config.pos_emb == 'learn':
            self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        elif config.pos_emb == 'sin':
            pos_emb = torch.zeros(config.block_size, config.n_embd)
            position = torch.arange(0, config.block_size, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, config.n_embd, 2).float() * (-math.log(10000.0) / config.n_embd))
            pos_emb[:, 0::2] = torch.sin(position * div_term)
            pos_emb[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pos_emb', pos_emb)
        elif config.pos_emb == 'rope':
            self.register_buffer("freqs_cis", self._precompute_freqs_cis())
    
    def _precompute_freqs_cis(self):
        """Precomputes the rotary frequencies for RoPE."""
        d = self.config.rope_head_dim if self.config.attn == 'mla' else self.head_size
        assert d % 2 == 0, "head dimension must be even"
        
        theta = 1.0 / (10000.0 ** (torch.arange(0, d, 2).float() / d))
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
        if parallel_flag == 6:
            return self._get_num_params_expert_parallel()
        else:
            return self._get_num_params_standard()
    
    def _get_num_params_expert_parallel(self):
        """Calculate parameters for expert parallel configuration"""
        n_params = sum(p.numel() for p in self.parameters())
        active_params = 0
        
        # Embeddings and layer norm
        active_params += self.tkn_emb.weight.numel()
        if self.config.pos_emb == 'learn':
            active_params += self.pos_emb.weight.numel()
        active_params += self.transformer.ln_f.weight.numel() + self.transformer.ln_f.bias.numel()
        
        # Transformer blocks
        for block in self.transformer.h:
            active_params += sum(p.numel() for p in block.attn.parameters())
            active_params += sum(p.numel() for p in block.ln1.parameters())
            active_params += sum(p.numel() for p in block.ln2.parameters())
            
            if hasattr(block, 'is_moe') and block.is_moe:
                # MoE block parameters
                active_params += self._calculate_moe_params_expert_parallel(block)
            else:
                # Regular MLP block
                active_params += sum(p.numel() for p in block.mlp.parameters())
        
        return n_params, active_params
    
    def _calculate_moe_params_expert_parallel(self, block):
        """Calculate MoE parameters for expert parallel configuration"""
        active_params = 0
        active_params += sum(p.numel() for p in block.moe.gate.parameters()) if block.moe.gate is not None else 0
        
        # Shared experts (always active)
        for i in range(len(block.moe.shared_experts)):
            active_params += sum(p.numel() for p in block.moe.shared_experts[i].parameters())
        
        # Routed experts (only active ones)
        if hasattr(block.moe, 'n_act_routed') and block.moe.n_act_routed > 0:
            if len(block.moe.local_routed_experts) > 0:
                params_per_routed_expert = sum(p.numel() for p in block.moe.local_routed_experts[0].parameters())
                active_params += block.moe.n_act_routed * params_per_routed_expert
        
        return active_params
    
    def _get_num_params_standard(self):
        """Calculate parameters for standard configuration"""
        n_params = sum(p.numel() for p in self.parameters())
        active_params = 0
        
        # Embeddings and layer norm
        active_params += self.tkn_emb.weight.numel()
        if self.config.pos_emb == 'learn':
            active_params += self.pos_emb.weight.numel()
        active_params += self.transformer.ln_f.weight.numel() + self.transformer.ln_f.bias.numel()
        
        # Transformer blocks
        for block in self.transformer.h:
            active_params += sum(p.numel() for p in block.attn.parameters())
            active_params += sum(p.numel() for p in block.ln1.parameters())
            active_params += sum(p.numel() for p in block.ln2.parameters())
            
            if block.is_moe:
                active_params += self._calculate_moe_params_standard(block)
            else:
                active_params += sum(p.numel() for p in block.mlp.parameters())
        
        return n_params, active_params
    
    def _calculate_moe_params_standard(self, block):
        """Calculate MoE parameters for standard configuration"""
        active_params = 0
        active_params += sum(p.numel() for p in block.moe.gate.parameters())
        
        # Shared experts (always active)
        for i in range(block.moe.n_shared):
            active_params += sum(p.numel() for p in block.moe.experts[i].parameters())
        
        # Routed experts (only active ones)
        if block.moe.n_routed > 0:
            params_per_routed_expert = sum(p.numel() for p in block.moe.experts[block.moe.n_shared].parameters())
            active_params += block.moe.n_act_routed * params_per_routed_expert
        
        return active_params
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        """Configure optimizer with appropriate parallel settings"""
        # Collect trainable parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Separate parameters by dimension for weight decay
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        return self._create_optimizer(optim_groups, learning_rate)
    
    def _create_optimizer(self, optim_groups, learning_rate):
        """Create optimizer based on parallel configuration"""
        try:
            if parallel_flag in [1, 4, 5, 6, 8]:
                # Use fused AdamW for certain parallel configurations
                optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, fused=True)
                self.print_fused_adamw = True
                return optimizer
            elif parallel_flag in [2, 3]:
                # Zero Redundancy Optimizer for distributed training
                optimizer = ZeroRedundancyOptimizer(
                    optim_groups,
                    optimizer_class=torch.optim.AdamW,
                    lr=learning_rate,
                )
                
                # Gradient sharding for ZeRO-2
                if parallel_flag == 3 and dist.is_initialized():
                    gradient_handler = ZeRO2GradientHandler(self)
                    optimizer = ZeRO2Optimizer(optimizer, gradient_handler)
                
                return optimizer
        except:
            # Fallback to standard AdamW
            pass
        
        return torch.optim.AdamW(optim_groups, lr=learning_rate)
    
    def forward(self, idx: torch.Tensor, targets=None, kv_caches=None):
        """Forward pass through the language model"""
        if parallel_flag == 8:
            return self._forward_context_parallel(idx, targets, kv_caches)
        else:
            return self._forward_standard(idx, targets, kv_caches)
    
    def _forward_context_parallel(self, idx: torch.Tensor, targets=None, kv_caches=None):
        """Forward pass for context parallel configuration"""
        B, T_local = idx.size()
        shard_start = self.config.context_parallel_rank * T_local
        
        # Token embeddings
        tkn_emb = self.tkn_emb(idx)
        
        # Apply positional embeddings
        x, freqs_cis = self._apply_positional_embeddings_context_parallel(
            tkn_emb, T_local, shard_start
        )
        
        # Dropout
        x = self.transformer.drop(x)
        
        # Initialize KV caches if needed
        if kv_caches is None:
            kv_caches = [None] * self.config.n_layer
        
        # Process transformer blocks
        updated_kv_caches, total_aux_loss = self._process_transformer_blocks_standard(
            x, freqs_cis, kv_caches
        )
        
        # Final layer norm
        x = self.transformer.ln_f(x)
        
        # Calculate loss if targets are provided
        if targets is not None:
            logits = self.lm_head(x)
            loss = self._calculate_loss_context_parallel(
                logits, targets, total_aux_loss, x.device
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss, updated_kv_caches
    
    def _apply_positional_embeddings_context_parallel(self, tkn_emb, T_local, shard_start):
        """Apply positional embeddings for context parallel"""
        x = tkn_emb
        freqs_cis = None
        
        if self.config.pos_emb == 'rope':
            # Check RoPE buffer length
            assert shard_start + T_local <= self.freqs_cis.size(0), \
                f"RoPE buffer too short: need {shard_start + T_local}, have {self.freqs_cis.size(0)}"
            freqs_cis = self.freqs_cis[shard_start: shard_start + T_local]
        elif self.config.pos_emb == 'learn':
            pos = torch.arange(shard_start, shard_start + T_local, dtype=torch.long, device=tkn_emb.device)
            x = tkn_emb + self.pos_emb(pos)
        elif self.config.pos_emb == 'sin':
            pos = torch.arange(shard_start, shard_start + T_local, dtype=torch.long, device=tkn_emb.device)
            x = tkn_emb + self.pos_emb[pos]
        
        return x, freqs_cis
    
    def _calculate_loss_context_parallel(self, logits, targets, total_aux_loss, device):
        """Calculate loss for context parallel configuration"""
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
            loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        
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
        
        # Combine main loss and auxiliary loss
        loss = loss + total_aux_loss / self.config.n_layer
        return loss
    
    def _forward_standard(self, idx: torch.Tensor, targets=None, kv_caches=None):
        """Forward pass for standard configuration"""
        B, T = idx.size()
        
        # Calculate start position from KV cache
        start_pos = self._get_start_position(kv_caches)
        
        # Token embeddings
        tkn_emb = self.tkn_emb(idx)
        
        # Apply positional embeddings
        x, freqs_cis = self._apply_positional_embeddings_standard(
            tkn_emb, T, start_pos, idx.device
        )
        
        # Dropout
        x = self.transformer.drop(x)
        
        # Initialize KV caches if needed
        if kv_caches is None:
            kv_caches = [None] * self.config.n_layer
        
        # Process transformer blocks
        updated_kv_caches, total_aux_loss = self._process_transformer_blocks_standard(
            x, freqs_cis, kv_caches
        )
        
        # Final layer norm
        x = self.transformer.ln_f(x)
        
        # Calculate loss if targets are provided
        if targets is not None:
            logits = self.lm_head(x)
            loss = self._calculate_loss_standard(logits, targets, total_aux_loss)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss, updated_kv_caches
    
    def _get_start_position(self, kv_caches):
        """Get start position from KV cache"""
        start_pos = 0
        if kv_caches is not None and kv_caches[0] is not None:
            if self.config.attn in ('mha', 'mqa', 'gqa'):
                start_pos = kv_caches[0][0].shape[-2]
            elif self.config.attn == 'mla':
                if self.config.pos_emb == 'rope':
                    start_pos = kv_caches[0]['c_kv'].shape[1]
                else:
                    start_pos = kv_caches[0].shape[1]
        return start_pos
    
    def _apply_positional_embeddings_standard(self, tkn_emb, T, start_pos, device):
        """Apply positional embeddings for standard configuration"""
        x = tkn_emb
        freqs_cis = None
        
        if self.config.pos_emb == 'rope':
            freqs_cis = self.freqs_cis[start_pos: start_pos + T]
        elif self.config.pos_emb == 'learn':
            pos = torch.arange(start_pos, start_pos + T, dtype=torch.long, device=device)
            x = tkn_emb + self.pos_emb(pos)
        elif self.config.pos_emb == 'sin':
            pos = torch.arange(start_pos, start_pos + T, dtype=torch.long, device=device)
            x = tkn_emb + self.pos_emb[pos]
        
        return x, freqs_cis
    
    def _process_transformer_blocks_standard(self, x, freqs_cis, kv_caches):
        """Process all transformer blocks"""
        updated_kv_caches = []
        total_aux_loss = 0.0
        
        for i, block in enumerate(self.transformer.h):
            if not self.config.act_recomp:
                x, updated_kv_cache, aux_loss = block(x, freqs_cis, kv_caches[i], self.VAL_RUN)
            else:
                x, updated_kv_cache, aux_loss = checkpoint(
                    block, x, freqs_cis, kv_caches[i], self.VAL_RUN
                )
            
            updated_kv_caches.append(updated_kv_cache)
            total_aux_loss += aux_loss
        
        return updated_kv_caches, total_aux_loss
    
    def _calculate_loss_standard(self, logits, targets, total_aux_loss):
        """Calculate loss for standard configuration"""
        main_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1
        )
        
        # Combine main loss and auxiliary loss
        loss = main_loss + total_aux_loss / self.config.n_layer
        return loss
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, topk: int | None = None):
        """Generate text using the language model"""
        self.eval()
        kv_caches = [None] * self.config.n_layer
        
        for i in range(max_new_tokens):
            # Prepare input for current step
            input_for_forward = self._prepare_generation_input(idx, i, kv_caches)
            
            # Forward pass
            logits, _, kv_caches = self.forward(input_for_forward, kv_caches=kv_caches)
            logits = logits[:, -1, :]
            
            # Apply temperature and top-k sampling
            logits = self._apply_sampling_temperature(logits, temperature)
            if topk is not None:
                logits = self._apply_topk_filtering(logits, topk)
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        self.train()
        return idx
    
    def _prepare_generation_input(self, idx, step, kv_caches):
        """Prepare input for generation step"""
        if step == 0:
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            return idx_cond
        else:
            return idx[:, -1:]
    
    def _apply_sampling_temperature(self, logits, temperature):
        """Apply temperature to logits for sampling"""
        if temperature > 0:
            return logits / temperature
        return logits
    
    def _apply_topk_filtering(self, logits, topk):
        """Apply top-k filtering to logits"""
        v, _ = torch.topk(logits, min(topk, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
        return logits



# Its purpose is to drastically reduce the memory footprint on non-Rank 0 GPUs (the "workers") 
# by ensuring they only store and compute the sharded expert weights and nothing else.
def create_worker_model(config: LLMconfig, device: str, moe_layer_mask: list[bool]):

    '''
    Purpose: Thin wrapper around the MoE class
    Why needed: Provides a consistent interface while stripping away all non-MoE components
    Memory saving: No LayerNorm, residual connections, or other transformer block components
    '''

    """Create a lightweight model for worker ranks (experts only)"""
    # This class serves as a minimalist placeholder for a single MoE block in the transformer stack.
    class WorkerMoEBlock(nn.Module):
        """Minimal block containing only MoE layers for worker ranks"""
        def __init__(self, config):
            super().__init__()

            # Only contains the MoE layer
            self.moe = MoE(config)
        
        def forward(self, x):
            # Direct pass-through to MoE
            return self.moe(x)
    

    # This is the main class that worker GPUs will use instead of the full LLM model.
    class WorkerLLM(nn.Module):
        '''
        moe_layer_mask is a boolean list from rank 0 indicating which layers are MoE
        Example: [False, True, False, True, False, True] for a 6-layer model
        Worker only creates blocks for True positions
        Massive memory savings: No parameters for attention layers, LayerNorms, etc.
        '''
        """Lightweight model for worker ranks containing only MoE layers"""
        # This is the main, lightweight model instance created on all worker GPUs (ranks !=0)

        def __init__(self, config, moe_layer_mask):
            super().__init__()
            self.config = config
            self.moe_layer_mask = moe_layer_mask
            
            # Only create MoE blocks for layers that are actually MoE in the full model
            '''
            The code iterates through the moe_layer_mask (e.g., [False, True, False, True, ...]):
            If a position is True (it's an MoE layer), it instantiates one WorkerMoEBlock (which contains the sharded MoE layer).
            If it's False (it's a regular MLP layer or simply a layer to be skipped), nothing is added.
        Result: The WorkerLLM only consists of an nn.ModuleList containing the exact MoE layers needed, ignoring all Attention, LayerNorm, and regular MLP parameters.

            '''
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



        # This function processes tokens through only the MoE layers that this worker GPU owns, 
        # completely skipping all other layers (attention, embeddings, etc.).
        def forward(self, x, targets=None, kv_caches=None):
            # Workers only participate in MoE computation via all_to_all
            # They don't compute loss or final outputs
            if kv_caches is None:
                kv_caches = [None] * self.config.n_layer
            
            total_aux_loss = 0.0
            moe_block_idx = 0
            for i in range(self.config.n_layer):
                if self.moe_layer_mask[i]:
                    # Ensure we don't exceed available MoE blocks
                    # Only process MoE layers that exist in this worker model
                    if moe_block_idx < len(self.moe_blocks):
                        x, aux_loss = self.moe_blocks[moe_block_idx](x)
                        total_aux_loss += aux_loss
                        moe_block_idx += 1
                # Skip non-MoE layers entirely on workers
            
            # Workers don't compute final output or loss
            return None, None, kv_caches
    
    return WorkerLLM(config, moe_layer_mask).to(device)

# This system handles checkpoint saving and resumption in a distributed environment where each GPU has different model components (experts).

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












# if parallel_flag==5 or parallel_flag == 4:
#     assert torch.cuda.device_count() > 1


### ----------- Training Script -----------


assert torch.cuda.is_available()
assert torch.cuda.device_count() > 1





# ______________DEVICE, DTYPE, DDP SETUP_________________

init_process_group(backend='nccl')

# Common device configuration logic
def setup_device_and_seeds(rank, local_rank):
    """Common setup for device configuration and seeding"""
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    torch.manual_seed(1729 + rank)
    torch.cuda.manual_seed(1729 + rank)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return device

# Common master process check and printing
def check_and_print_master(rank, world_size, flag_name=""):
    """Check if master process and print world size"""
    master_process = rank == 0
    if master_process:
        if flag_name:
            print(f"{flag_name}_WORLD_SIZE = {world_size}")
        else:
            print(f"Num GPUs = {world_size}")
    return master_process

# Handle parallel_flag configurations
if parallel_flag in [4, 8]:
    # Sequence Parallel or Context Parallel
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])

    world_size = int(os.environ['WORLD_SIZE'])
    ddp_world_size = world_size
    
    device = setup_device_and_seeds(rank, local_rank)
    master_process = check_and_print_master(rank, world_size)

elif parallel_flag == 5:
    # Tensor Parallel
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])

    world_size = int(os.environ['WORLD_SIZE'])
    ddp_world_size = world_size

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)
    master_process = check_and_print_master(rank, world_size)
    
    # Common tensor parallel seeding and settings
    torch.manual_seed(1729 + rank)
    torch.cuda.manual_seed(1729 + rank)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

elif parallel_flag in [1, 2, 3]:
    # DDP with different ZeRO configurations
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    world_size = ddp_world_size

    device = setup_device_and_seeds(ddp_rank, ddp_local_rank)
    master_process = check_and_print_master(ddp_rank, ddp_world_size, "DDP")
elif parallel_flag == 7:
    
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])


    world_size = int(os.environ['WORLD_SIZE'])
    ddp_world_size = world_size

    device = f"cuda:{local_rank}"
    master_process = rank == 0
    if master_process : print(f"Num GPUs = {world_size}")
    torch.cuda.set_device(device)
    torch.manual_seed(1729 + rank)
    torch.cuda.manual_seed(1729 + rank)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

else:
    # Standard DDP (fallback)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    world_size = ddp_world_size
    device = f"cuda:{ddp_local_rank}"
    master_process = check_and_print_master(ddp_rank, ddp_world_size, "DDP")

# Common dtype configuration
dtype = 'float16' if not torch.cuda.is_bf16_supported else 'bfloat16'
torch_dtype = getattr(torch, dtype)

# Common autocast and grad scaler setup
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
    chunks : int
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
    chunks = 8,
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
    attn = 'gqa', 
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

    parser.add_argument('--chunks', type=int, default=8)

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

    

if parallel_flag==5:
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


# Add WandB args to TrainingConfig
TrainingConfig.wandb_project = args.wandb_project
TrainingConfig.wandb_entity = args.wandb_entity
TrainingConfig.wandb_run_name = args.wandb_run_name
TrainingConfig.wandb_notes = args.wandb_notes
TrainingConfig.wandb_tags = args.wandb_tags
TrainingConfig.no_wandb = args.no_wandb






if parallel_flag==5:
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





if parallel_flag == 8:
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
        
        # Setup device
        self.device = torch.device(device)
        self.device_type = self.device.type
        
        # Setup context parallel if needed
        if parallel_flag == 8:
            self._setup_context_parallel(context_parallel_size, context_parallel_rank)
        
        # Load memory-mapped tokens
        self._load_tokens()
        
        # Validate dataset size
        self._validate_dataset_size()
    
    def _setup_context_parallel(self, context_parallel_size, context_parallel_rank):
        """Setup for context parallel configuration"""
        self.context_parallel_size = context_parallel_size
        self.context_parallel_rank = context_parallel_rank
        
        # Calculate local sequence length
        self.local_T = self.T // context_parallel_size
        assert self.T % context_parallel_size == 0, "Sequence length must be divisible by context parallel size"
    
    def _load_tokens(self):
        """Load memory-mapped tokens from file"""
        self.tokens = np.memmap(self.file_path, dtype=np.uint16, mode='r')
        self.N = len(self.tokens)
    
    def _validate_dataset_size(self):
        """Validate that batch size and sequence length fit in dataset"""
        if self.B * self.T + 1 > self.N:
            raise ValueError(f"Batch size {self.B} and block size {self.T} are too large for dataset of length {self.N}")
    
    def _sample_start_indices(self, batch_size, sequence_length):
        """Sample random starting positions for sequences"""
        return torch.randint(0, self.N - sequence_length - 1, (batch_size,))
    
    def _move_to_device(self, x, y):
        """Move tensors to the appropriate device with optimization"""
        if self.device_type == 'cuda':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y
    
    def _process_standard_batch(self):
        """Process a batch for standard (non-context parallel) configuration"""
        B, T = self.B, self.T
        start_indices = self._sample_start_indices(B, T)
        
        x_list = []
        y_list = []
        
        for start in start_indices:
            seq = self.tokens[start: start + T + 1].astype(np.int64)
            x_list.append(seq[:-1])
            y_list.append(seq[1:])
        
        # Stack into tensors
        x = torch.from_numpy(np.stack(x_list)).long()
        y = torch.from_numpy(np.stack(y_list)).long()
        
        return x, y
    
    def _process_context_parallel_batch(self):
        """Process a batch for context parallel configuration"""
        B, local_T = self.B, self.local_T
        start_indices = self._sample_start_indices(B, self.T)
        
        x_list = []
        y_list = []
        
        for start in start_indices:
            full_seq = self.tokens[start: start + self.T + 1].astype(np.int64)
            
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
        
        return x, y
    
    def next_batch(self):
        """
        Returns (x, y) where:
        - x is (B, T) input tokens
        - y is (B, T) target tokens (shifted by one)
        """
        if parallel_flag == 8:
            x, y = self._process_context_parallel_batch()
        else:
            x, y = self._process_standard_batch()
        
        # Move to device
        x, y = self._move_to_device(x, y)
        
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



def chunked_cross_entropy(lm_head, hidden_states, targets, chunk_size=128):
    logits_flat = hidden_states.view(-1, hidden_states.size(-1))
    targets_flat = targets.view(-1)
    num_tokens = targets_flat.size(0)
    total_loss = 0.0
    
    for i in range(0, num_tokens, chunk_size):
        x_chunk = logits_flat[i:i+chunk_size]
        y_chunk = targets_flat[i:i+chunk_size]
        logits_chunk = lm_head(x_chunk)
        loss_chunk = F.cross_entropy(logits_chunk, y_chunk, ignore_index=-1, reduction='sum')
        total_loss += loss_chunk
    
    valid_tokens = (targets_flat != -1).sum()
    return total_loss / valid_tokens if valid_tokens > 0 else total_loss

class PipelineStage(nn.Module):
    def __init__(self, full_model, config, start_layer, end_layer, 
                 is_first, is_last, rank):
        super().__init__()
        self.config = config
        self.rank = rank
        self.is_first = is_first
        self.is_last = is_last
        
        if is_first:
            self.tkn_emb = full_model.tkn_emb
            if config.pos_emb == 'rope':
                self.register_buffer('freqs_cis', full_model.freqs_cis.clone())
            self.drop = full_model.transformer.drop
        
        self.blocks = nn.ModuleList([
            full_model.transformer.h[i] for i in range(start_layer, end_layer)
        ])
        
        if is_last:
            self.ln_f = full_model.transformer.ln_f
            self.lm_head = full_model.lm_head
        
        self.to(f'cuda:{rank}')
    
    def forward(self, inputs, targets=None):
        if self.is_first:
            idx = inputs.clamp(0, self.config.vocab_size - 1)
            B, T = idx.size()
            x = self.tkn_emb(idx)
            freqs_cis = self.freqs_cis[:T] if self.config.pos_emb == 'rope' else None
            x = self.drop(x)
            total_aux = 0.0
        else:
            x, freqs_cis, total_aux = inputs
        
        for block in self.blocks:
            x, _, aux = block(x, freqs_cis, None, False)
            total_aux += aux
        
        if self.is_last:
            x = self.ln_f(x)
            if targets is not None:
                main_loss = chunked_cross_entropy(self.lm_head, x, targets)
                return main_loss + total_aux / self.config.n_layer
            return self.lm_head(x)
        
        return (x, freqs_cis, total_aux)
        
def run_pipeline_1f1b_with_profiler(stage, rank, world_size, train_loader, config, num_chunks, max_iters, learning_rate=3e-4, profile_iters=5, profile_start=10):
    from time import perf_counter
    
    optimizer = torch.optim.AdamW(stage.parameters(), lr=learning_rate)
    scaler = GradScaler()
    master_process = (rank == 0)

    if master_process:
        prof = create_profiler(output_dir=f"./profiler_logs/rank_{rank}")
        prof.start()
        print("🔍 Profiler initialized")
    
    # Get total parameters
    stage_params = sum(p.numel() for p in stage.parameters())
    total_params_tensor = torch.tensor([stage_params], dtype=torch.long, device=f'cuda:{rank}')
    dist.all_reduce(total_params_tensor, op=dist.ReduceOp.SUM)
    total_params = total_params_tensor.item()
    
    # === CRITICAL FIX: Initialize scaler for ALL non-last ranks ===
    if rank < world_size - 1:
        dummy = torch.zeros(1, device=f'cuda:{rank}', requires_grad=True)
        dummy_scaled = scaler.scale(dummy)
        dummy_scaled.backward()
        optimizer.zero_grad()
    
    for iteration in range(max_iters):
        with record_function(f"iteration_{iteration}"):
            t0 = perf_counter()
            
            optimizer.zero_grad()
            
            # Broadcast batch shape
            if rank == 0:
                x_full, y_full = train_loader.next_batch()
                x_full = x_full.to(f'cuda:{rank}')
                y_full = y_full.to(f'cuda:{rank}')
                B, T = x_full.shape
                shape_tensor = torch.tensor([B, T], dtype=torch.long, device=f'cuda:{rank}')
            else:
                shape_tensor = torch.zeros(2, dtype=torch.long, device=f'cuda:{rank}')
            
            dist.broadcast(shape_tensor, src=0)
            B, T = int(shape_tensor[0]), int(shape_tensor[1])
            
            if B < num_chunks:
                if rank == 0:
                    print(f"Warning: Adjusting chunks from {num_chunks} to {B}")
                num_chunks = B
            
            chunk_B = B // num_chunks
            activations = {}     
            recv_inputs = {}     
            losses = []
            
            def run_forward_step(micro_id):
                with autocast(device_type='cuda'):
                    if rank == 0:
                        x_chunk = x_full[micro_id*chunk_B:(micro_id+1)*chunk_B]
                        output = stage(x_chunk)
                        activations[micro_id] = output[0] if isinstance(output, tuple) else output
                        x_out, freqs, aux = output
                        dist.send(x_out.contiguous(), dst=rank+1)
                        if freqs is not None: 
                            dist.send(freqs.contiguous(), dst=rank+1)
                        dist.send(torch.tensor([aux], device=x_out.device), dst=rank+1)
                        y_chunk = y_full[micro_id*chunk_B:(micro_id+1)*chunk_B]
                        dist.send(y_chunk.contiguous(), dst=world_size-1)

                    elif rank < world_size - 1:
                        x_recv = torch.empty(chunk_B, T, config.n_embd, device=f'cuda:{rank}')
                        dist.recv(x_recv, src=rank-1)
                        x_recv.requires_grad_(True)
                        recv_inputs[micro_id] = x_recv 
                        
                        freqs_recv = None
                        if config.pos_emb == 'rope':
                            d = config.rope_head_dim if config.attn=='mla' else config.n_embd // config.n_head
                            freqs_recv = torch.empty(T, d // 2, dtype=torch.complex64, device=f'cuda:{rank}')
                            dist.recv(freqs_recv, src=rank-1)
                        
                        aux_recv = torch.empty(1, device=f'cuda:{rank}')
                        dist.recv(aux_recv, src=rank-1)
                        
                        output = stage((x_recv, freqs_recv, aux_recv.item()))
                        activations[micro_id] = output[0] if isinstance(output, tuple) else output
                        
                        x_out, freqs, aux = output
                        dist.send(x_out.contiguous(), dst=rank+1)
                        if freqs is not None: 
                            dist.send(freqs.contiguous(), dst=rank+1)
                        dist.send(torch.tensor([aux], device=x_out.device), dst=rank+1)

                    else:  # Last rank
                        x_recv = torch.empty(chunk_B, T, config.n_embd, device=f'cuda:{rank}')
                        dist.recv(x_recv, src=rank-1)
                        x_recv.requires_grad_(True)
                        recv_inputs[micro_id] = x_recv
                        
                        freqs_recv = None
                        if config.pos_emb == 'rope':
                            d = config.rope_head_dim if config.attn=='mla' else config.n_embd // config.n_head
                            freqs_recv = torch.empty(T, d // 2, dtype=torch.complex64, device=f'cuda:{rank}')
                            dist.recv(freqs_recv, src=rank-1)
                        
                        aux_recv = torch.empty(1, device=f'cuda:{rank}')
                        dist.recv(aux_recv, src=rank-1)
                        
                        y_chunk = torch.empty(chunk_B, T, dtype=torch.long, device=f'cuda:{rank}')
                        dist.recv(y_chunk, src=0)
                        
                        loss = stage((x_recv, freqs_recv, aux_recv.item()), y_chunk)
                        losses.append(loss)

            def run_backward_step(backward_id):
                if rank == world_size - 1:
                    scaler.scale(losses[backward_id] / num_chunks).backward()
                    grad_to_send = recv_inputs[backward_id].grad
                    dist.send(grad_to_send.contiguous(), dst=rank-1)
                    del recv_inputs[backward_id]

                elif rank > 0:
                    grad_recv = torch.empty_like(activations[backward_id])
                    dist.recv(grad_recv, src=rank+1)
                    activations[backward_id].backward(grad_recv)
                    grad_to_send = recv_inputs[backward_id].grad
                    dist.send(grad_to_send.contiguous(), dst=rank-1)
                    del activations[backward_id]
                    del recv_inputs[backward_id]

                else:  # First rank
                    grad_recv = torch.empty_like(activations[backward_id])
                    dist.recv(grad_recv, src=rank+1)
                    activations[backward_id].backward(grad_recv)
                    del activations[backward_id]

            # ============ SINGLE EXECUTION OF PIPELINE (REMOVED DUPLICATE) ============
            num_warmup = min(world_size, num_chunks)
            
            with record_function("warmup_phase"):
                for micro_id in range(num_warmup):
                    run_forward_step(micro_id)
            
            with record_function("1f1b_phase"):
                for micro_id in range(num_warmup, num_chunks):
                    run_forward_step(micro_id)
                    backward_id = micro_id - num_warmup
                    run_backward_step(backward_id)
            
            with record_function("cooldown_phase"):
                cooldown_start = max(0, num_chunks - num_warmup)
                for i in range(num_warmup):
                    backward_id = cooldown_start + i
                    if backward_id < num_chunks:
                        run_backward_step(backward_id)
            
            with record_function("optimizer_step"):
                dist.barrier()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(stage.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            
            # ============ GATHER LOSS FROM LAST RANK ============
            if rank == world_size - 1:
                avg_loss = sum(l.item() for l in losses) / len(losses) if losses else 0.0
                loss_tensor = torch.tensor([avg_loss], device=f'cuda:{rank}')
                dist.send(loss_tensor, dst=0)
            elif rank == 0:
                loss_tensor = torch.zeros(1, device=f'cuda:{rank}')
                dist.recv(loss_tensor, src=world_size - 1)
            else:
                loss_tensor = torch.zeros(1, device=f'cuda:{rank}')
            
            dist.broadcast(loss_tensor, src=0)
            avg_loss = loss_tensor.item()
            
            # Profiler step
            if master_process and profile_start <= iteration < profile_start + profile_iters:
                prof.step()
            elif master_process and iteration == profile_start + profile_iters:
                prof.stop()
                print("✅ Pipeline profiling complete! Check ./profiler_logs/ for traces")
                profiler_enabled = False  # Disable after profiling
            
            # ============ LOGGING ============
            if master_process:
                torch.cuda.synchronize()
                mem = torch.cuda.memory_reserved()
                dt = (perf_counter() - t0) * 1000
                
                tokens_per_iter = B * T * num_chunks
                tokens_per_sec = tokens_per_iter / (dt / 1000.0)
                
                mfu = compute_mfu_a40(
                    tokens_per_sec=tokens_per_sec,
                    n_params=total_params,
                    n_layers=config.n_layer,
                    n_heads=config.n_head,
                    head_dim=config.n_embd // config.n_head,
                    seq_len=T,
                    n_gpus=world_size,
                    include_attention=True,
                )
                
                print(
                    f"step: {iteration} | "
                    f"loss:{avg_loss:.4f} | "
                    f"dt:{dt:.2f}ms | "
                    f"tok/s:{tokens_per_sec:,.0f} | "
                    f"MFU:{mfu:.2f}% | "
                    f"GPU RAM:{mem/1024**3:.2f}GB"
                )





# Broadcast function to ensure all ranks have same data
def broadcast_batch(x, y, src=0):
    """Ensure all TP ranks have the same batch"""
    if dist.is_initialized():
        dist.broadcast(x, src=src)
        dist.broadcast(y, src=src)
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





def reduce_scatter_sequence(tensor, group=None):
    """Reduce-scatter sequence chunks to context parallel ranks"""
    world_size = torch.distributed.get_world_size(group=group)
    
    if world_size == 1:
        return tensor
    
    # Split tensor into chunks for reduce-scatter
    tensor_chunks = list(tensor.chunk(world_size, dim=1))
    output = torch.zeros_like(tensor_chunks[0])
    torch.distributed.reduce_scatter(output, tensor_chunks, group=group)
    
    return output





if parallel_flag == 8:

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
def compute_mfu_a40(
    *,
    tokens_per_sec: float,
    n_params: float,
    n_layers: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    n_gpus: int,
    include_attention: bool = True,
):
    """
    PaLM / Kaplan style MFU computation for NVIDIA A40 GPUs
    """

    # NVIDIA A40 peak FP16/BF16 Tensor Core throughput
    A40_PEAK_TFLOPS = 312.0
    peak_flops_per_sec = A40_PEAK_TFLOPS * 1e12 * n_gpus

    # 6N FLOPs per token (forward + backward, non-attention)
    non_attn_flops_per_token = 6.0 * n_params

    # 6 * L * H * (2 * Q * T) attention FLOPs per token
    attn_flops_per_token = (
        6.0 * n_layers * n_heads * (2.0 * head_dim * seq_len)
        if include_attention else 0.0
    )

    flops_per_token = non_attn_flops_per_token + attn_flops_per_token
    achieved_flops_per_sec = tokens_per_sec * flops_per_token

    mfu = achieved_flops_per_sec / peak_flops_per_sec

    return mfu * 100.0  # return percentage



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




def cleanup():
    """Cleanup function for distributed training"""
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    
    if master_process:
        print("✅ Training completed successfully")


#___________GRAD_ACCUM SETUP_____________

total_batch_size = TrainingConfig.total_batch_size
B = TrainingConfig.batch_size    # microbatch size
T = ModelConfig.block_size       # sequence length

if parallel_flag == 7:
    # Pipeline parallelism handles batching via chunks, no grad accumulation
    grad_accum_steps = 1
elif parallel_flag == 6 or parallel_flag == 5:
    assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T)
elif parallel_flag == 4 or parallel_flag == 8:
    assert total_batch_size % (B * T *world_size) == 0, "make sure total_batch_size is divisible by B * T * world_size"
    grad_accum_steps = total_batch_size // (B * T *world_size)
else:
    assert total_batch_size % (B * T *ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T *ddp_world_size)




#___________CREATE YOUR MODEL_____________
fsdp_wrap_policy = None
mp_policy = None
if parallel_flag == 4:
    fsdp_wrap_policy = ModuleWrapPolicy({Block})

    mp_policy = MixedPrecision(
        param_dtype=torch_dtype,
        reduce_dtype=torch_dtype,
        buffer_dtype=torch_dtype,
    )



if parallel_flag==5:
    model = LLM(ModelConfig , tp_group=tp_group).to(device)
else:
    model = LLM(ModelConfig).to(device)



if master_process : 
    total, active = model.get_num_params()
    print(f"total parameters = {total:,}, active parameters = {active:,}")
    if model.print_fused_adamw: print("Using Fused AdamW")
    if model.print_act_recomp: print("Using Activation Recomputation")


if parallel_flag == 1 or parallel_flag == 2 or parallel_flag==3:
    
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=ModelConfig.moe)

elif parallel_flag == 8:
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

elif parallel_flag == 4:
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


if parallel_flag == 6:
    os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
    os.environ.setdefault('NCCL_DEBUG', 'WARN') 
    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    if os.getenv('NCCL_P2P_DISABLE') is None:
        os.environ.setdefault('NCCL_P2P_DISABLE', '0')  # Enable by default for better perf on NVLink

    # Version check at program start
    assert version.parse(torch.__version__) >= version.parse("2.1.0"), \
        "EP MoE requires PyTorch >= 2.1.0 for autograd on all_to_all_single"




# Initialize WandB only on master process
use_wandb = not TrainingConfig.no_wandb and master_process
if use_wandb:
    if not TrainingConfig.wandb_run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        TrainingConfig.wandb_run_name = f"{TrainingConfig.dataset}_{ModelConfig.attn}_{timestamp}"
    
    # Determine if MoE is used
    moe_info = "MoE" if ModelConfig.moe else "Dense"
    
    wandb.init(
        project=TrainingConfig.wandb_project,
        entity=TrainingConfig.wandb_entity,
        name=TrainingConfig.wandb_run_name,
        notes=TrainingConfig.wandb_notes,
        tags=TrainingConfig.wandb_tags + [f"parallel_flag_{parallel_flag}", moe_info],
        config={
            # Training config
            "batch_size": TrainingConfig.batch_size,
            "total_batch_size": TrainingConfig.total_batch_size,
            "max_iters": TrainingConfig.max_iters,
            "learning_rate": TrainingConfig.learning_rate,
            "warmup_steps": TrainingConfig.warmup_steps,
            "grad_clip": TrainingConfig.grad_clip,
            "act_recomp": TrainingConfig.act_recomp,
            "chunks": TrainingConfig.chunks if hasattr(TrainingConfig, 'chunks') else 0,
            
            # Model config
            "vocab_size": ModelConfig.vocab_size,
            "block_size": ModelConfig.block_size,
            "n_embd": ModelConfig.n_embd,
            "pos_emb": ModelConfig.pos_emb,
            "n_layer": ModelConfig.n_layer,
            "dropout": ModelConfig.dropout,
            "attn": ModelConfig.attn,
            "n_head": ModelConfig.n_head,
            "n_kv_heads": ModelConfig.n_kv_heads,
            
            # MoE config (if applicable)
            "moe": ModelConfig.moe,
            "n_exp": ModelConfig.n_exp if ModelConfig.moe else 0,
            "n_shared": ModelConfig.n_shared if ModelConfig.moe else 0,
            "n_act": ModelConfig.n_act if ModelConfig.moe else 0,
            "aux_free": ModelConfig.aux_free if ModelConfig.moe else False,
            
            # Parallelism info
            "parallel_flag": parallel_flag,
            "world_size": world_size if 'world_size' in locals() else 1,
            "ddp_world_size": ddp_world_size if 'ddp_world_size' in locals() else 1,
            
            # Parameter counts
            "total_params": total,
            "active_params": active,
            "grad_accum_steps": grad_accum_steps,
            "dtype": dtype,
        }
    )
    
    # Watch the model
    wandb.watch(model, log="all", log_freq=100)
    
    if master_process:
        print(f"WandB initialized: project={TrainingConfig.wandb_project}, run={TrainingConfig.wandb_run_name}")






if parallel_flag == 7:
    # ============== PIPELINE PARALLELISM TRAINING ==============
    use_wandb = not TrainingConfig.no_wandb and master_process
    if use_wandb:
        if not TrainingConfig.wandb_run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            TrainingConfig.wandb_run_name = f"pipeline_{TrainingConfig.dataset}_{timestamp}"
        
        wandb.init(
            project=TrainingConfig.wandb_project,
            entity=TrainingConfig.wandb_entity,
            name=TrainingConfig.wandb_run_name,
            notes=TrainingConfig.wandb_notes,
            tags=TrainingConfig.wandb_tags + ["pipeline_parallel", f"chunks_{TrainingConfig.chunks}"],
            config=vars(TrainingConfig)
        )

    config = ModelConfig

    n_layers = config.n_layer
    layers_per_rank = n_layers // world_size
    remainder = n_layers % world_size
    
    # Distribute layers evenly
    if rank < remainder:
        start_layer = rank * (layers_per_rank + 1)
        end_layer = start_layer + layers_per_rank + 1
    else:
        start_layer = remainder * (layers_per_rank + 1) + (rank - remainder) * layers_per_rank
        end_layer = start_layer + layers_per_rank
    
    if master_process:
        print(f"\n{'='*60}")
        print(f"PIPELINE PARALLELISM: {world_size} stages, {TrainingConfig.chunks} microbatches")
        print(f"{'='*60}")
        print(f"Total layers: {n_layers}")
        for r in range(world_size):
            if r < remainder:
                s = r * (layers_per_rank + 1)
                e = s + layers_per_rank + 1
            else:
                s = remainder * (layers_per_rank + 1) + (r - remainder) * layers_per_rank
                e = s + layers_per_rank
            print(f"  Rank {r} (GPU {r}): Layers {s}-{e-1} ({e-s} layers)")
        print(f"{'='*60}\n")
    
    # Create full model and extract this rank's stage
    full_model = LLM(config)
    if master_process:
        total, active = full_model.get_num_params()
        print(f"Total parameters: {total:,}, Active parameters: {active:,}\n")
    
    stage = PipelineStage(
        full_model, config, start_layer, end_layer,
        is_first=(rank == 0), 
        is_last=(rank == world_size - 1), 
        rank=rank
    )
    
    # Run pipeline training
    run_pipeline_1f1b_with_profiler(
        stage, rank, world_size, train_loader, config, 
        num_chunks=TrainingConfig.chunks, 
        max_iters=TrainingConfig.max_iters,
        learning_rate=TrainingConfig.learning_rate
    )
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    
    if master_process:
        print("\n✅ Pipeline parallelism training complete!")

    if use_wandb and master_process:
        wandb.finish()

    exit()

else:




    if parallel_flag == 5 or parallel_flag == 6:
        raw_model:LLM = model
    else:
        raw_model:LLM = model.module


    optimizer = raw_model.configure_optimizers(weight_decay=0.1,learning_rate=TrainingConfig.learning_rate,device=device)



    # Get first batch
    if parallel_flag in [5, 6]:
        if master_process:
            x, y = train_loader.next_batch()
        else:
            x = torch.empty(B, T, dtype=torch.long, device=device)
            y = torch.empty(B, T, dtype=torch.long, device=device)

        x, y = broadcast_batch(x, y, src=0)
    else:
        x, y = train_loader.next_batch()

    loss_stats = []


    profiler_enabled = True  # Set to False to disable profiling
    profiler_start_iter = 10  # Start profiling after N iterations
    profiler_duration = 10    # Profile for N iterations

    if profiler_enabled and master_process:
        prof = create_profiler(output_dir="./profiler_logs")
        prof.start()
        print("🔍 Profiler initialized")

    for iter in range(TrainingConfig.max_iters + 1):
        t0 = perf_counter()

        lr = get_lr(iter, TrainingConfig) 
        for param_grp in optimizer.param_groups:
            param_grp['lr'] = lr
        
        optimizer.zero_grad(set_to_none=True)

        a, b = 0, 0
        if TrainingConfig.eval and (iter % TrainingConfig.eval_interval == 0 or iter == TrainingConfig.max_iters) and iter != 0:
            with record_function("validation"):  # Profile validation
                a = perf_counter()
                losses = estimate_loss(model, TrainingConfig, train_loader, val_loader)
                b = perf_counter()
                if master_process:
                    print(f"--------val run-------- train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | dt {1000*(b-a):.4f}ms")
                t0 = b

        if parallel_flag in [5, 6]:
            optimizer.zero_grad(set_to_none=True)

        # Gradient accumulation loop
        if parallel_flag == 6:
            for micro_step in range(grad_accum_steps):
                with record_function(f"microstep_{micro_step}"):  # Profile each microstep
                    
                    with record_function("data_loading"):
                        if master_process:
                            x, y = train_loader.next_batch()
                        else:
                            x = torch.empty(B, T, dtype=torch.long, device=device)
                            y = torch.empty(B, T, dtype=torch.long, device=device)
                        
                        x, y = broadcast_batch(x, y, src=0)
                    
                    with record_function("forward_pass"):
                        with torch.cuda.amp.autocast(dtype=torch_dtype):
                            _, loss, _ = model(x, y)
                            loss = loss / grad_accum_steps
                    
                    with record_function("backward_pass"):
                        scaler.scale(loss).backward()
        
        elif parallel_flag == 5:
            for micro_step in range(grad_accum_steps):
                with record_function(f"microstep_{micro_step}"):
                    
                    with record_function("data_loading"):
                        if master_process:
                            x, y = train_loader.next_batch()
                        else:
                            x = torch.empty(B, T, dtype=torch.long, device=device)
                            y = torch.empty(B, T, dtype=torch.long, device=device)
                        
                        x, y = broadcast_batch(x, y, src=0)
                    
                    with record_function("forward_pass"):
                        with torch.cuda.amp.autocast(dtype=torch_dtype):
                            _, loss, _ = model(x, y)
                            loss = loss / grad_accum_steps
                    
                    with record_function("backward_pass"):
                        scaler.scale(loss).backward()
        
        else:
            for micro_step in range(grad_accum_steps):
                with record_function(f"microstep_{micro_step}"):
                    model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

                    with record_function("forward_pass"):
                        with ctx:
                            _, loss, _ = model(x, y)
                            loss = loss / grad_accum_steps

                    with record_function("data_loading"):
                        x, y = train_loader.next_batch()
                    
                    loss_stats.append(loss.cpu())
                    
                    with record_function("backward_pass"):
                        scaler.scale(loss).backward()

        with record_function("optimizer_step"):
            if TrainingConfig.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.grad_clip)

            scaler.step(optimizer)
            scaler.update()

        if profiler_enabled and master_process:
            if profiler_start_iter <= iter < profiler_start_iter + profiler_duration:
                prof.step()
            elif iter == profiler_start_iter + profiler_duration:
                prof.stop()
                print("✅ Profiling complete! Check ./profiler_logs/ for traces")
                profiler_enabled = False  # Disable after profiling    

        # if master_process:
        #     torch.cuda.synchronize()
        #     mem = torch.cuda.memory_reserved()
        #     dt = (perf_counter() - t0) * 1000
        #     print(f"step: {iter} | train loss:{loss.item()*grad_accum_steps:.4f} | dt: {dt:.2f}ms | grad_accum_steps: {grad_accum_steps} | GPU RAM: {mem/1024**3:.2f}GB")
        if master_process:
            torch.cuda.synchronize()
            mem = torch.cuda.memory_reserved()
            dt = (perf_counter() - t0) * 1000
            
            tokens_per_iter = B * T * grad_accum_steps * ddp_world_size
            tokens_per_sec = tokens_per_iter / (dt / 1000.0)

            mfu = compute_mfu_a40(
                tokens_per_sec=tokens_per_sec,
                n_params=total,
                n_layers=ModelConfig.n_layer,
                n_heads=ModelConfig.n_head,
                head_dim=ModelConfig.n_embd // ModelConfig.n_head,
                seq_len=T,
                n_gpus=ddp_world_size,
                include_attention=True,
            )

            # print(
            #     f"step: {iter} | "
            #     f"loss:{loss.item()*grad_accum_steps:.4f} | "
            #     f"dt:{dt:.2f}ms | "
            #     f"tok/s:{tokens_per_sec:,.0f} | "
            #     f"MFU:{mfu:.2f}% | "
            #     f"GPU RAM:{mem/1024**3:.2f}GB"
            # )
            if use_wandb:
                log_data = {
                "train/loss": loss.detach() * grad_accum_steps,  # Tensor for graph
                "train/lr": torch.tensor(lr, device=device),
                "train/step": iter,
                "perf/iteration_time_ms": dt,
                "perf/throughput_tokens_per_sec": tokens_per_sec,
                "perf/throughput_tokens_per_sec_per_gpu": tokens_per_sec / ddp_world_size,
                "perf/mfu_percent": mfu,
                "memory/allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "memory/reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "memory/max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
                }
                # Add gradient norms for better debugging
                if TrainingConfig.grad_clip != 0.0:
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    log_data["train/grad_norm"] = total_norm
                
                wandb.log(log_data)
                
                # Add MFU if computed
                if 'mfu' in locals():
                    log_data["perf/mfu_percent"] = mfu
            
                wandb.log(log_data)
        
            # Print to console
            if 'mfu' in locals():
                print(
                    f"step: {iter} | "
                    f"loss:{loss.item()*grad_accum_steps:.4f} | "
                    f"dt:{dt:.2f}ms | "
                    f"tok/s:{tokens_per_sec:,.0f} | "
                    f"MFU:{mfu:.2f}% | "
                    f"GPU RAM:{mem/1024**3:.2f}GB"
                )
            else:
                print(
                    f"step: {iter} | "
                    f"loss:{loss.item()*grad_accum_steps:.4f} | "
                    f"dt:{dt:.2f}ms | "
                    f"tok/s:{tokens_per_sec:,.0f} | "
                    f"GPU RAM:{mem/1024**3:.2f}GB"
                )

    # Cleanup
    if parallel_flag == 6:
        cleanup()
    elif parallel_flag == 5:
        if dist.is_initialized():
            dist.destroy_process_group()

if TrainingConfig.save_model and master_process and False:  # For now lets not save the trash model
    checkpoint = {
        'config': ModelConfig,
        'model_state': raw_model.state_dict(),
        'iter_num': iter,
        'last_loss': losses,
        'train_losses': loss_stats
    }
    torch.save(checkpoint, 'llm_model.pt')
    print("checkpoint saved to llm_model.pt")

# Finish WandB run
if use_wandb and master_process:
    wandb.finish()
    print("WandB run completed")

def analyze_profiler_trace(trace_file):
    """
    Load and analyze a profiler trace file
    """
    import json
    
    with open(trace_file, 'r') as f:
        trace = json.load(f)
    
    print(f"Analyzing trace: {trace_file}")
    print(f"Total events: {len(trace.get('traceEvents', []))}")

# To view traces:
# 1. Open Chrome browser
# 2. Go to chrome://tracing
# 3. Load the generated JSON trace files
# 4. Use TensorBoard: tensorboard --logdir=./profiler_logs

print("""
📊 PROFILER USAGE GUIDE:

1. Traces are saved to ./profiler_logs/
2. View in Chrome: chrome://tracing (load JSON files)
3. View in TensorBoard: 
   tensorboard --logdir=./profiler_logs
   
4. Key metrics to look for:
   - CUDA kernel launch overhead
   - Memory allocation patterns
   - CPU/GPU utilization
   - Communication overhead (DDP/FSDP)
   - Kernel execution time
""")