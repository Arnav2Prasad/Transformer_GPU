

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from ar_logging.profiler import create_profiler
from torch.profiler import profile, record_function, ProfilerActivity, schedule



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



