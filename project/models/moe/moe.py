

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model import LLMconfig
from config.cli import parallel_flag

from models.moe.experts import Expert
from parallel.ep import EPLayout


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

