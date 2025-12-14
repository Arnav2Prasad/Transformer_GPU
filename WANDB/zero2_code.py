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
                        numel = p.numel()           # Number of elements in this parameter
                        p.grad.data.copy_(
                            grad_buffer[offset:offset + numel].view_as(p.data)
                        )
                        offset += numel
                else:
                    # Non-owning ranks can discard gradients
                    for p in params:
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

