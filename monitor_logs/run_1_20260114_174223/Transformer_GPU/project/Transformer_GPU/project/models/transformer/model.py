

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model import LLMconfig
from train import parallel_flag

from models.transformer.block import Block


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

