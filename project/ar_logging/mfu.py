
# ____________ UTIL FUNCTIONS _________________
def compute_mfu_a40(
    *,
    tokens_per_sec: float,
    n_params: float,
    n_layers: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    n_gpus: int,
    include_attention: bool = True,
):
    """
    PaLM / Kaplan style MFU computation for NVIDIA A40 GPUs
    """

    # NVIDIA A40 peak FP16/BF16 Tensor Core throughput
    A40_PEAK_TFLOPS = 312.0
    peak_flops_per_sec = A40_PEAK_TFLOPS * 1e12 * n_gpus

    # 6N FLOPs per token (forward + backward, non-attention)
    non_attn_flops_per_token = 6.0 * n_params

    # 6 * L * H * (2 * Q * T) attention FLOPs per token
    attn_flops_per_token = (
        6.0 * n_layers * n_heads * (2.0 * head_dim * seq_len)
        if include_attention else 0.0
    )

    flops_per_token = non_attn_flops_per_token + attn_flops_per_token
    achieved_flops_per_sec = tokens_per_sec * flops_per_token

    mfu = achieved_flops_per_sec / peak_flops_per_sec

    return mfu * 100.0  # return percentage
