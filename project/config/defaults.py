


# ____________PARAMS-CONFIG_________________

from config.train import Trainconfig
from config.model import LLMconfig

import os


TrainingConfig = Trainconfig(
    dataset='shakespeare',
    total_batch_size = 2**12,
    batch_size = 2**1, # how many independent sequences will we process in parallel?
    max_iters = 2500,
    eval = False,
    eval_interval=100,
    eval_iters=100,
    learning_rate = 3e-4,
    warmup_steps = 100,
    grad_clip = 1.0,    
    compile = False if os.name != 'posix' else True,
    save_model = True,
    file_name='llm_model',
    chunks = 8,
    act_recomp=False)   # Default to False

ModelConfig = LLMconfig(
    # token params
    vocab_size = 50304, 
    block_size = 2**10,
    # n_embd = 256, 
    n_embd = 768, 
    pos_emb = 'rope',
    
    # MoE
    moe = True,

    up_dim = 1024, 
    non_linearity = 'swiglu',  
    dropout=0.0,
    n_layer = 12,

    n_exp = 8,
    n_shared = 1,
    n_act = 4,        ### INCLUDES THE SHARED EXPERTS

    coeff=0.01,
    aux_free=True,
    alpha = 0.0001,
    gamma = 0.001,

    # Attention
    attn = 'gqa', 
    n_head = 8,
    n_kv_heads=4,
    # MHLA
    q_latent_dim = 32, 
    kv_latent_dim = 32,
    rope_head_dim = 16,

    # ADD EP CONFIG
    ep_size=1,  # Will be updated in training script
    ep_rank=0,  # Will be updated in training script
    ep_group=None,  # Will be updated in training script
    
    act_recomp=TrainingConfig.act_recomp)


