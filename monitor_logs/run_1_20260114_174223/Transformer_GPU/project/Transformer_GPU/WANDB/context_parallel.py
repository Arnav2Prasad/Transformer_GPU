
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

