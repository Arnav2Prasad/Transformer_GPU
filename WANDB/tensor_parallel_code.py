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




from typing import Literal
from dataclasses import dataclass 
from torch.distributed.optim import ZeroRedundancyOptimizer

from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp.api  import ShardingStrategy, CPUOffload


from config_code import LLMconfig

from datetime import datetime
import wandb
from datetime import datetime
import glob  # <-- MISSING
import gc    # <-- MISSING
from llm_code import LLM


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



# we added the below code
def _get_group_and_ranks(tp_group = None):
    """Get TP group, world size, and rank - safer version""" 
    if not dist.is_initialized():
        print('inside if not dist.is_initialized() ')
        return None ,1, 0
    
    tp_group = tp_group or dist.group.WORLD

    return tp_group , dist.get_world_size(tp_group) , dist.get_rank(tp_group)


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


# Broadcast function to ensure all ranks have same data
def broadcast_batch(x, y, src=0):
    """Ensure all TP ranks have the same batch"""
    if dist.is_initialized():
        dist.broadcast(x, src=src)
        dist.broadcast(y, src=src)
    return x, y