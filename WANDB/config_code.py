


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



# 1 for plain DP
# 2 for zero1
# 3 for zero2
merging_code = 3
print('1 for plain DP')
print('2 for zero1')
print('3 for zero2')
print('------')
print('merging_code : ',merging_code)


    
# 1 for DP
# 2 for FSDP
ddp_flag = 1
print('1 for DP')
print('2 for FSDP')
print('------')


print('ddp_flag : ', ddp_flag)
print('------')


tp_code = 2
print('1 for TP')
print('2 for No TP')


ep_code = 1
print('1 for EP')
print('2 for No EP')


cp_code = 2
print('1 for CP')
print('2 for No CP')



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
        # freqs_cis has two parts : real and imaginary (cosθ, sinθ)
        # import code ; code.interact(local=locals())
        # Perform the rotation (vector * rotation matrix)
        x_re_out = x_re*freqs_cis.real - x_im*freqs_cis.imag    # (B, T, H, hs//2) * (1, T, 1, hs//2) - (B, T, H, hs//2) * (1, T, 1, hs//2) -> (B, T, H, hs//2)
        x_im_out = x_re*freqs_cis.imag + x_im*freqs_cis.real    # (B, T, H, hs//2) * (1, T, 1, hs//2) + (B, T, H, hs//2) * (1, T, 1, hs//2) -> (B, T, H, hs//2)
        
        # Stack the real and imaginary parts back together
        x_out = torch.stack([x_re_out, x_im_out], dim=-1).flatten(3) # (B, T, H, hs//2), (B, T, H, hs//2) -> (B, T, H, hs)

        return x_out.type_as(x)

    


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





class DataLoader:
    def __init__(self, B, T, file_path, device ,context_parallel_size=1, context_parallel_rank=0):
        self.B = B
        self.T = T
        self.file_path = file_path
        self.device = device
        # self.device_type = 'cuda'
        # FIX: Convert device string to torch.device object
        self.device = torch.device(device)
        self.device_type = self.device.type  # Store device type separately if needed

        if cp_code == 1:
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
        """
        Returns (x, y) where:
        - x is (B, T) input tokens
        - y is (B, T) target tokens (shifted by one)
        """
        B, T = self.B, self.T

        if cp_code == 1:
            local_T = self.local_T 

        # Sample B random starting positions independently
        start_indices = torch.randint(0, self.N - T - 1, (B,))

        # Gather sequences
        x_list = []
        y_list = []
        for start in start_indices:
            if cp_code == 1:
                full_seq = self.tokens[start : start + self.T + 1].astype(np.int64)
            
                # Extract local chunk for this context parallel rank
                local_start = self.context_parallel_rank * local_T
                local_end = local_start + local_T
                x_local = full_seq[local_start:local_end]
                y_local = full_seq[local_start + 1:local_end + 1]
                
                x_list.append(x_local)
                y_list.append(y_local)

            else:
                seq = self.tokens[start : start + T + 1].astype(np.int64)
                x_list.append(seq[:-1])
                y_list.append(seq[1:])

        # Stack into tensors
        x = torch.from_numpy(np.stack(x_list)).long()
        y = torch.from_numpy(np.stack(y_list)).long()

        if cp_code == 1:
            # Verify local sequence length
            assert x.shape[1] == self.local_T, f"Expected local_T={self.local_T}, got {x.shape[1]}"

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
    wandb_run_name : str | None = None
    wandb_project : str | None = None
    wandb_entity : str |None = None
    wandb_run_name : str | None = None





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