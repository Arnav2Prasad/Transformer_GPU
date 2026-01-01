



class Attention(nn.Module):
    """ Routes the attention mechanism according to the config"""

    def __init__(self, config:LLMconfig, tp_group=None):  # ← ADDED tp_group parameter
        super().__init__()
        self.config = config
        if config.attn in ('mha','mqa','gqa'):
            self.attn = GQA(config, tp_group=tp_group)  # ← Pass tp_group
        else:
            raise NotImplementedError("Only GQA supported")
        
        
        # elif config.attn == 'mla':
        #     if config.pos_emb != 'rope':
        #         self.attn = NaiveMHLA(config)
        #     else:
        #         self.attn = FullMHLA(config)

                
    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor|None = None, kv_cache=None, VAL_RUN=False):
        return self.attn(x, freqs_cis, kv_cache, VAL_RUN)

