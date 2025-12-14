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


from config_code import LLMconfig, merging_code, ddp_flag , tp_code, ep_code, cp_code, DataLoader, _get_group_and_ranks, RowParallelLinear, ColumnParallelLinear

from datetime import datetime
import wandb
from datetime import datetime
import glob  # <-- MISSING
import gc    # <-- MISSING
from config_code import LLM






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