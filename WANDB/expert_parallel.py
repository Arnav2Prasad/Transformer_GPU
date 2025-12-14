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




def save_checkpoint(model, optimizer, iter, rank, checkpoint_dir="checkpoints"):
    """Save checkpoint with distributed expert handling"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'iteration': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rank': rank,
    }
    
    # Each rank saves its own checkpoint
    torch.save(checkpoint, f"{checkpoint_dir}/rank_{rank}_iter_{iter}.pt")
    
    # Rank 0 also saves a metadata file
    if rank == 0:
        metadata = {
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'iteration': iter,
            'timestamp': torch.tensor(torch.timestamp()),
        }
        torch.save(metadata, f"{checkpoint_dir}/metadata_iter_{iter}.pt")



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


'''
Rank 0 (Orchestrator):
├── Shared Experts: [Expert_A, Expert_B]     ← Process ALL tokens locally
├── Routed Experts: [Expert_0, Expert_1, Expert_2, Expert_3]  ← Local shard
└── Gate Network: Decides token routing

Rank 1 (Worker):
├── Shared Experts: ❌ NONE
└── Routed Experts: [Expert_4, Expert_5, Expert_6]  ← Local shard

Rank 2 (Worker): 
├── Shared Experts: ❌ NONE
└── Routed Experts: [Expert_7, Expert_8, Expert_9]  ← Local shard

Rank 3 (Worker):
├── Shared Experts: ❌ NONE  
└── Routed Experts: [Expert_10, Expert_11, Expert_12, Expert_13]  ← Local shard
'''
def main_worker(local_rank, world_size, TrainingConfig, ModelConfig):
    """Worker function with detailed step printing and proper termination"""
    train_loader = None
    val_loader = None
    
    try:
        # CRITICAL FIX: Use local_rank for device assignment (supports multi-node)
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(local_rank)
        
        # Set deterministic seeds (same across ranks for reproducibility)
        torch.manual_seed(42 + local_rank)  # Different expert params per rank
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + local_rank)
        
        # Initialize distributed
        # dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
        
        # Setup expert parallelism - use ALL GPUs for EP
        ep_group, ep_rank, ep_size = setup_ep_groups(world_size, local_rank, world_size)
        
        # Verify world size matches EP size
        if world_size != ep_size:
            raise ValueError(f"World size {world_size} must equal EP size {ep_size} for pure EP")
        
        # Create a copy of ModelConfig to avoid modifying the original
        model_config_copy = LLMconfig(
            vocab_size=ModelConfig.vocab_size,
            block_size=ModelConfig.block_size,
            n_embd=ModelConfig.n_embd,
            pos_emb=ModelConfig.pos_emb,
            up_dim=ModelConfig.up_dim,
            non_linearity=ModelConfig.non_linearity,
            dropout=ModelConfig.dropout,
            n_layer=ModelConfig.n_layer,
            moe=ModelConfig.moe,
            n_exp=ModelConfig.n_exp,
            n_shared=ModelConfig.n_shared,
            n_act=ModelConfig.n_act,
            coeff=ModelConfig.coeff,
            aux_free=ModelConfig.aux_free,
            alpha=ModelConfig.alpha,
            gamma=ModelConfig.gamma,
            attn=ModelConfig.attn,
            n_head=ModelConfig.n_head,
            n_kv_heads=ModelConfig.n_kv_heads,
            q_latent_dim=ModelConfig.q_latent_dim,
            kv_latent_dim=ModelConfig.kv_latent_dim,
            rope_head_dim=ModelConfig.rope_head_dim,
            act_recomp=ModelConfig.act_recomp,
            # Set EP attributes
            ep_size=ep_size,
            ep_rank=ep_rank,
            ep_group=ep_group
        )

        # Setup AMP
        device_type = 'cuda'
        amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast(device_type=device_type, dtype=amp_dtype)
        
        # Calculate gradient accumulation steps
        total_batch_size = TrainingConfig.total_batch_size
        B = TrainingConfig.batch_size
        T = model_config_copy.block_size
        assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
        grad_accum_steps = total_batch_size // (B * T)
        use_wandb = not TrainingConfig.no_wandb and local_rank == 0
        if use_wandb:
            if not TrainingConfig.wandb_run_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                TrainingConfig.wandb_run_name = f"EP_{TrainingConfig.dataset}_{timestamp}"
            
            wandb.init(
                project=TrainingConfig.wandb_project,
                entity=TrainingConfig.wandb_entity,
                name=TrainingConfig.wandb_run_name,
                config={
                    **vars(TrainingConfig),
                    **vars(ModelConfig),
                    "ep_size": world_size,
                    "local_rank": local_rank,
                }
            )
    
        
        if local_rank == 0:
            print(f"📈 Training with gradient accumulation: {grad_accum_steps} steps")
        
        # Different model creation for rank 0 vs workers
        if local_rank == 0:
            # Rank 0: full model
            model = LLM(model_config_copy).to(device)
            train_loader = DataLoader(B=TrainingConfig.batch_size, T=model_config_copy.block_size, 
                                    file_path="train.bin", device=device)
            total, active = model.get_num_params()
            print(f"total parameters = {total:,}, active parameters = {active:,}")
            
            # Full optimizer for rank 0
            optimizer = model.configure_optimizers(
                weight_decay=0.1, 
                learning_rate=TrainingConfig.learning_rate, 
                device=device
            )
            scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))
            
            # CRITICAL FIX: Create MoE layer mask to mirror full model structure
            moe_layer_mask = []
            '''
            Example Output:

            For a 6-layer model with MoE at layers 1, 3, 5:
                moe_layer_mask = [False, True, False, True, False, True]
                    # Layer:   0       1       2       3       4       5
                    # Type:    Attn    MoE     Attn    MoE     Attn    MoE
            '''
            for block in model.transformer.h:
                moe_layer_mask.append(hasattr(block, 'moe') and block.is_moe)
                
            # Get initial batch
            x, y = train_loader.next_batch()
        else:
            # Worker ranks need the MoE layer mask from rank 0
            moe_layer_mask = [False] * model_config_copy.n_layer  # Placeholder
            # Worker ranks get minimal loaders for cleanup consistency
            train_loader = None
            val_loader = None
            
        # Broadcast MoE layer mask from rank 0 to all workers
        '''
        Before Broadcast:
        Rank 0: [False, True, False, True, False, True]  ← Real mask
        Rank 1: [False, False, False, False, False, False] ← Placeholder  
        Rank 2: [False, False, False, False, False, False] ← Placeholder

        After Broadcast:
        Rank 0: [False, True, False, True, False, True]  ← Real mask
        Rank 1: [False, True, False, True, False, True]  ← Real mask
        Rank 2: [False, True, False, True, False, True]  ← Real mask
        '''
        if world_size > 1:
            moe_layer_mask_tensor = torch.tensor(moe_layer_mask, dtype=torch.bool, device=device)
            dist.broadcast(moe_layer_mask_tensor, src=0)
            moe_layer_mask = moe_layer_mask_tensor.cpu().tolist()
        
        if local_rank != 0:
            # Worker ranks: lightweight model with only experts, mirroring rank 0's structure
            model = create_worker_model(model_config_copy, device, moe_layer_mask)
            train_loader = None
            
            # Local optimizer for worker experts only
            local_params = []
            for module in model.modules():
                if hasattr(module, 'local_routed_experts'):
                    for expert in module.local_routed_experts:
                        local_params.extend([p for p in expert.parameters() if p.requires_grad])
            
            optimizer = torch.optim.AdamW(
                local_params, 
                lr=TrainingConfig.learning_rate,
                weight_decay=0.1
            )
            # Workers don't use GradScaler
            scaler = None
        
        # Get parameter dtype for dummy inputs
        param_dtype = next(model.parameters()).dtype
        
        # Set model to training mode
        model.train()
        
        # Training loop with corrected synchronization
        start_iter = 0
        
        # Optional: load checkpoint
        if TrainingConfig.save_model:
            start_iter = load_checkpoint(model, optimizer)
            # Ensure all ranks resume consistently and validate iteration
            if world_size > 1:
                start_iter_tensor = torch.tensor(start_iter, device=device)
                dist.broadcast(start_iter_tensor, src=0)
                start_iter = start_iter_tensor.item()
        
        
        # MAIN TRAINING LOOP
        for iter in range(start_iter, TrainingConfig.max_iters + 1):
            t0 = perf_counter()

            # Learning rate scheduling
            lr = get_lr(iter, TrainingConfig) 
            for param_grp in optimizer.param_groups:
                param_grp['lr'] = lr

            # Zero gradients on all ranks
            optimizer.zero_grad(set_to_none=True)
            
            # Track loss for printing (only on rank 0)
            current_loss = 0.0
            
            if local_rank == 0:
                # Get batch and forward/backward on rank 0
                x, y = train_loader.next_batch()
                
                # Forward pass (includes EP communication)
                with ctx:
                    _, loss, _ = model(x, y)
                    current_loss = loss.item()
                
                # Backward pass
                scaler.scale(loss).backward()
            else:
                # Worker ranks: dummy forward to trigger EP communication and gradients
                # Use a small dummy input to minimize memory usage
                dummy_x = torch.zeros(1, 1, ModelConfig.n_embd, device=device, dtype=param_dtype)
                with ctx:
                    _, _, _ = model(dummy_x)
                # No backward on workers - gradients flow via autograd through collectives
            
            # CRITICAL: Ensure backward is finished before stepping
            if world_size > 1:
                dist.barrier()
            
            # Optimization step
            if local_rank == 0:
                if TrainingConfig.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Worker optimization (no scaler)
                if TrainingConfig.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(local_params, TrainingConfig.grad_clip)
                
                optimizer.step()
            
            # Synchronize and measure time
            if "cuda" in device:
                torch.cuda.synchronize()
            dt = (perf_counter() - t0) * 1000  # Convert to milliseconds
            
            # Memory usage
            if torch.cuda.is_available():
                mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)  # Convert to GB
                torch.cuda.reset_peak_memory_stats(device)
            else:
                mem_gb = 0.0
            
            # Print step information (only on rank 0 to avoid duplicate output)
            if local_rank == 0:
                print(f"step: {iter} | train loss:{current_loss:.4f} | "
                      f"dt: {dt:.2f}ms | grad_accum_steps: {grad_accum_steps} | "
                      f"GPU RAM: {mem_gb:.2f}GB")

            
            
            # Save checkpoint periodically
            if TrainingConfig.save_model and iter % TrainingConfig.eval_interval == 0 and iter > 0 and local_rank == 0:
                save_checkpoint(model, optimizer, iter, local_rank)
            
            # FINAL ITERATION - CRITICAL: Break out and cleanup
            if iter == TrainingConfig.max_iters:
                if local_rank == 0:
                    print(f"🎉 Training completed all {TrainingConfig.max_iters} iterations!")
                
                # Save final checkpoint on rank 0
                if TrainingConfig.save_model and local_rank == 0:
                    save_checkpoint(model, optimizer, TrainingConfig.max_iters, local_rank)
                    print("💾 Final checkpoint saved")
                
                # BREAK OUT OF THE LOOP - THIS PREVENTS INFINITE LOOP
                break
            
            # Final synchronization for non-final iterations
            if world_size > 1:
                dist.barrier()
            if use_wandb and local_rank == 0:
                wandb.log({
                    "train/loss": current_loss,
                    "train/lr": lr,
                    "train/step": iter,
                    "perf/iteration_time_ms": dt,
                    "memory/allocated_gb": mem_gb,
                })
        if use_wandb:
            wandb.finish()
                
    except KeyboardInterrupt:
        if local_rank == 0:
            print("⏹️ Training interrupted by user")
        # Save partial checkpoint if desired
        if TrainingConfig.save_model and local_rank == 0:
            save_checkpoint(model, optimizer, iter, local_rank)
            print("💾 Partial checkpoint saved")
        
    except Exception as e:
        print(f"Rank {local_rank}: Training error: {e}")
        raise
        
    finally:
        # GUARANTEED cleanup - this will always run and ensure termination
        finalize_training(local_rank, train_loader, val_loader)
        
        if local_rank == 0:
            print("🏁 Worker process completed and cleaned up")




def main(TrainingConfig, ModelConfig):
    """Main function for torchrun launch method"""
    import os
    try:
        # Get distributed setup from environment variables (torchrun sets these)
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        print(f"🚀 Starting torchrun training - Rank {local_rank}/{world_size-1}")
        print(f"📊 Target: {TrainingConfig.max_iters} iterations")
        
        # Call the worker function
        main_worker(local_rank, world_size, TrainingConfig, ModelConfig)
        
        # If we reach here, training completed successfully
        if local_rank == 0:
            print("✅ All training iterations completed successfully")
            
    except KeyboardInterrupt:
        if 'local_rank' in locals() and local_rank == 0:
            print("⏹️ Training interrupted by user")
        # Re-raise to ensure torchrun sees the interruption
        raise
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        # Re-raise to ensure torchrun propagates the error
        raise
        
    finally:
        # Final cleanup in the main process
        if 'local_rank' in locals() and local_rank == 0:
            print("🧹 Final cleanup completed")
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()