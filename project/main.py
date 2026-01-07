

# 1 for plain DP
# 2 for zero1
# 3 for zero1 and 2
# 4 for fsdp 
# 5 for TP
# 6 for EP
# 7 for PP 
# 8 for cp

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
import time
import glob
import sys
import gc
import warnings
import numpy as np

from typing import Literal, Optional, Dict, List, Tuple
from dataclasses import dataclass
from contextlib import nullcontext
from packaging import version
from time import perf_counter

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.amp import autocast, GradScaler

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp.api import ShardingStrategy, CPUOffload

from torch.profiler import profile, record_function, ProfilerActivity, schedule



import torch._dynamo
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy



from config.cli import parse_args
from config.defaults import TrainingConfig, ModelConfig

# from config.model import LLMconfig
# from config.train import Trainconfig


from data.loader import DataLoader
from data.utils import tokenize_and_save


# from ar_logging.mfu import compute_mfu_a40
from ar_logging.mfu import compute_mfu_from_configs
from ar_logging.mfu import arnav_compute_mfu_from_configs


from ar_logging.profiler import create_profiler
from ar_logging.wandb import save_checkpoint_with_wandb


from models.attention.base import Attention
from models.attention.gqa import GQA

from models.layers.mlp import MLP

from models.moe.experts import Expert
from models.moe.moe import MoE

from models.transformer.block import Block
from models.transformer.model import LLM


from models.utils import get_lr , estimate_loss , cleanup


from parallel.ep import EPLayout , setup_ep_groups
from parallel.pipeline import PipelineStage
from parallel.pp import chunked_cross_entropy, run_pipeline_1f1b_with_profiler
from parallel.tp import ColumnParallelLinear , RowParallelLinear , _get_group_and_ranks , init_distributed


from parallel.utils import setup_device_and_seeds ,check_and_print_master , broadcast_batch , all_gather_sequence ,reduce_scatter_sequence
from parallel.zero_2 import ZeRO2GradientHandler , ZeRO2Optimizer


from train import parallel_flag
# from config.cli import parallel_flag




os.environ['WANDB_API_KEY']='c78410b3a816898642987ae3c3899430080b89d1'

warnings.filterwarnings("ignore")




















assert torch.cuda.is_available()
assert torch.cuda.device_count() > 1





# ______________DEVICE, DTYPE, DDP SETUP_________________

init_process_group(backend='nccl')



if parallel_flag == -1:
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])

    world_size = int(os.environ['WORLD_SIZE'])
    ddp_world_size = world_size

    master_process = rank==0
    torch.manual_seed(1729)
    torch.cuda.manual_seed(1729)
    torch.set_float32_matmul_precision('medium')   # Not sure if this has any effect when used with Auto Mixed Precision

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ctx = torch.amp.autocast(device_type=device_type, dtype=getattr(torch, dtype))
    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16')) if device == 'cuda' else nullcontext()

elif parallel_flag in [4, 8]:
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
    rank = ddp_rank
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

    rank = ddp_rank
    world_size = ddp_world_size
    device = f"cuda:{ddp_local_rank}"
    master_process = check_and_print_master(ddp_rank, ddp_world_size, "DDP")






# Common dtype configuration
dtype = 'float16' if not torch.cuda.is_bf16_supported else 'bfloat16'
torch_dtype = getattr(torch, dtype)

# Common autocast and grad scaler setup
ctx = torch.amp.autocast(device_type="cuda", dtype=torch_dtype)
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))










# ___________ CLI-OVERRIDE__________________



 

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



print('=============')
print('parallel_flag - ', parallel_flag)
print('parallel_flag - ', parallel_flag)
print('parallel_flag - ', parallel_flag)
print('=============')

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






tokenize_and_save() # Using The Tiny Shakespeare dataset for demo









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









#___________GRAD_ACCUM SETUP_____________

total_batch_size = TrainingConfig.total_batch_size
B = TrainingConfig.batch_size    # microbatch size
T = ModelConfig.block_size       # sequence length


grad_accum_steps = 0

if parallel_flag == 7:
    # Pipeline parallelism handles batching via chunks, no grad accumulation
    grad_accum_steps = 1
elif parallel_flag == 6 or parallel_flag == 5:
    assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T)
elif parallel_flag == 4 or parallel_flag == 8:
    assert total_batch_size % (B * T *world_size) == 0, "make sure total_batch_size is divisible by B * T * world_size"
    grad_accum_steps = total_batch_size // (B * T *world_size)
elif parallel_flag == -1:
    assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
    grad_accum_steps = total_batch_size // (B * T)
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
    # Wrap FSDP creation with dynamo disabled
    '''
    with torch._dynamo.disable():
        model = FSDP(
            model,
            auto_wrap_policy=fsdp_wrap_policy,
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=True,
            sync_module_states=True,
        )
    '''

  

    # Now wrap with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=fsdp_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        sync_module_states=True,
    )





if master_process : print("Using compiled model")
model = torch.compile(model)

running_mfu = -1.0

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
        if parallel_flag != -1:
            raw_model:LLM = model.module

    if parallel_flag == -1:
        optimizer = model.configure_optimizers(weight_decay=0.1,learning_rate=TrainingConfig.learning_rate,device=device)
    else:
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
    profiler_total_steps = 4   # total profiler steps (must match schedule)


    prof = None
    if profiler_enabled:
        prof = create_profiler(
            output_dir="./profiler_logs",
            rank=rank,   # IMPORTANT: pass rank
        )





    # if profiler_enabled and master_process:
    #     # prof = create_profiler(output_dir="./profiler_logs")
    #     prof = create_profiler(
    #         output_dir="./llama_profiler_logs",
    #         enable_memory=True,      # Enable for detailed memory analysis
    #         enable_stack_trace=True, # Enable for debugging
    #         enable_flops=True,       # Enable for performance analysis
    #         device=device
    #     )
    #     prof.start()
    #     print("🔍 Profiler initialized")







    
    with prof if profiler_enabled else nullcontext():

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



                

            # if profiler_enabled and master_process:
            #     if profiler_start_iter <= iter < profiler_start_iter + profiler_duration:
            #         prof.step()
            #     elif iter == profiler_start_iter + profiler_duration:
            #         prof.stop()
            #         print("✅ Profiling complete! Check ./profiler_logs/ for traces")
            #         profiler_enabled = False  # Disable after profiling  
            # 
            if profiler_enabled:
                if profiler_start_iter <= iter < profiler_start_iter + profiler_total_steps:
                    prof.step()
                elif iter == profiler_start_iter + profiler_total_steps:
                    profiler_enabled = False  

            
            
            if master_process:
                torch.cuda.synchronize()
                mem = torch.cuda.memory_reserved()
                dt = (perf_counter() - t0) * 1000
                
                tokens_per_iter = B * T * grad_accum_steps * ddp_world_size
                tokens_per_sec = tokens_per_iter / (dt / 1000.0)

            


                n_gpus = 2
                print("active->", active)

                mfu_pct = arnav_compute_mfu_from_configs(
                    dt_ms=dt,
                    n_params_active=active,
                    model_cfg=ModelConfig,
                    training_cfg=TrainingConfig,
                    n_gpus=n_gpus,
                    grad_accum_steps=grad_accum_steps,
                    peak_tflops_per_gpu=65.0,  # e.g. T4 ≈ 65 TFLOPS (fp16)
                    include_attention=True,
                    
                )

                print(f"MFU: {mfu_pct:.2f}%")
                mfu = mfu_pct
                

                

                running_mfu = (
                    mfu_pct if running_mfu < 0
                    else 0.9 * running_mfu + 0.1 * mfu_pct
                )

                mfu = mfu_pct

                print(
                    # f"iter {iter_num}: "
                    # f"loss {lossf:.4f}, "
                    # f"time {dt_ms:.2f}ms, "
                    f"mfu {running_mfu:.2f}%"
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