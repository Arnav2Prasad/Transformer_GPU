


import torch
import torch.nn as nn


class PipelineStage(nn.Module):
    def __init__(self, full_model, config, start_layer, end_layer, 
                 is_first, is_last, rank):
        super().__init__()
        self.config = config
        self.rank = rank
        self.is_first = is_first
        self.is_last = is_last
        
        if is_first:
            self.tkn_emb = full_model.tkn_emb
            if config.pos_emb == 'rope':
                self.register_buffer('freqs_cis', full_model.freqs_cis.clone())
            self.drop = full_model.transformer.drop
        
        self.blocks = nn.ModuleList([
            full_model.transformer.h[i] for i in range(start_layer, end_layer)
        ])
        
        if is_last:
            self.ln_f = full_model.transformer.ln_f
            self.lm_head = full_model.lm_head
        
        self.to(f'cuda:{rank}')
    
    def forward(self, inputs, targets=None):
        if self.is_first:
            idx = inputs.clamp(0, self.config.vocab_size - 1)
            B, T = idx.size()
            x = self.tkn_emb(idx)
            freqs_cis = self.freqs_cis[:T] if self.config.pos_emb == 'rope' else None
            x = self.drop(x)
            total_aux = 0.0
        else:
            x, freqs_cis, total_aux = inputs
        
        for block in self.blocks:
            x, _, aux = block(x, freqs_cis, None, False)
            total_aux += aux
        
        if self.is_last:
            x = self.ln_f(x)
            if targets is not None:
                main_loss = chunked_cross_entropy(self.lm_head, x, targets)
                return main_loss + total_aux / self.config.n_layer
            return self.lm_head(x)
        
        return (x, freqs_cis, total_aux)

