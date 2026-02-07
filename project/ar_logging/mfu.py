
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
    # 312e12
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





# ____________ UTIL FUNCTIONS _________________
def compute_mfu_t4_nanogpt(
    *,
    dt: float,
    n_params: float,
    n_layers: int,
    n_heads: int,
    seq_len: int,
    n_gpus: int = 2,
    batch_size_per_gpu: int,
    grad_accum_steps: int = 1,
    include_attention: bool = True,
    peak_tflops_per_gpu: float = 65.0,  # Tesla T4 tensor (FP16 mixed precision) peak
) -> float:
    """
    Estimate Model FLOPs Utilization (MFU) using the nanoGPT / PaLM Appendix-B style formula.

    Definitions (IMPORTANT):
    - seq_len (T): block size / context length.
    - dt: wall time (seconds) for *one optimizer iteration* (i.e., after grad accumulation).
    - batch_size_per_gpu: micro-batch size per GPU (number of sequences per GPU per micro-step).
    - grad_accum_steps: number of micro-steps accumulated before optimizer step.
    - Returned MFU is % of theoretical tensor-core peak across all GPUs (n_gpus).

    Notes:
    - This is a *rough* estimate. It assumes dense training FLOPs/token ≈ 6N plus optional attention term.
    """

    if dt <= 0:
        raise ValueError("dt must be > 0")
    if n_gpus <= 0:
        raise ValueError("n_gpus must be > 0")
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if batch_size_per_gpu <= 0 or grad_accum_steps <= 0:
        raise ValueError("batch_size_per_gpu and grad_accum_steps must be > 0")

    # ---- Model shape terms (nanoGPT convention) ----
    L = n_layers
    H = n_heads
    Q = (n_params * 0)  # placeholder to keep the structure readable (not used)
    # In nanoGPT, Q is "head_dim" = n_embd // n_head. We can derive head_dim if you pass it,
    # but if you don't, compute it externally and pass as head_dim in a variant.
    # Here we reconstruct head_dim from typical transformer definition is not possible from params alone.

    # So: require head_dim explicitly for attention FLOPs if include_attention=True
    # We'll implement attention with an explicit head_dim input to avoid wrong assumptions.
    # (See below: we compute attn term only if caller provides head_dim through kwargs.)
    # To keep this function strict & correct, we calculate attention term only if include_attention=False
    # OR you use the alternative function below that accepts head_dim.

    # ---- FLOPs per token ----
    # Base: 6N FLOPs per token (training: forward+backward approx)
    flops_per_token = 6.0 * float(n_params)

    # Optional attention term needs head_dim (Q). If you want it, use the function below.
    if include_attention:
        raise ValueError(
            "include_attention=True requires head_dim. "
            "Use compute_mfu_t4_nanogpt_with_attention(...) below."
        )

    # ---- Convert per-token to per-(fwd+bwd) over one sequence of length T ----
    flops_per_fwd_bwd = flops_per_token * float(seq_len)

    # ---- How many sequences are processed per optimizer step (global) ----
    # nanoGPT-style: fwd_bwd_per_iter = batch_size * grad_accum, and batch_size is global or per-GPU depending on setup.
    # Here we explicitly compute global sequences per optimizer step:
    sequences_per_iter_global = batch_size_per_gpu * n_gpus * grad_accum_steps

    flops_per_iter = flops_per_fwd_bwd * float(sequences_per_iter_global)

    # ---- Achieved FLOPs/s ----
    flops_achieved_per_sec = flops_per_iter / float(dt)

    # ---- Theoretical peak FLOPs/s (tensor core peak) ----
    flops_peak_per_sec = peak_tflops_per_gpu * 1e12 * n_gpus

    mfu = flops_achieved_per_sec / flops_peak_per_sec
    return mfu * 100.0




def compute_mfu_t4_nanogpt_with_attention(
    *,
    dt: float,
    n_params: float,
    n_layers: int,
    n_heads: int,
    head_dim: int,   # Q in nanoGPT
    seq_len: int,    # T in nanoGPT
    n_gpus: int = 2,
    batch_size_per_gpu: int,
    grad_accum_steps: int = 1,
    peak_tflops_per_gpu: float = 65.0,
) -> float:
    """
    nanoGPT / PaLM Appendix-B style MFU with attention term:
    flops_per_token = 6N + 12 L H Q T
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if n_gpus <= 0:
        raise ValueError("n_gpus must be > 0")
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if batch_size_per_gpu <= 0 or grad_accum_steps <= 0:
        raise ValueError("batch_size_per_gpu and grad_accum_steps must be > 0")
    if head_dim <= 0:
        raise ValueError("head_dim must be > 0")

    L = n_layers
    H = n_heads
    Q = head_dim
    T = seq_len

    flops_per_token = 6.0 * float(n_params) + 12.0 * float(L) * float(H) * float(Q) * float(T)
    flops_per_fwd_bwd = flops_per_token * float(T)

    sequences_per_iter_global = batch_size_per_gpu * n_gpus * grad_accum_steps
    flops_per_iter = flops_per_fwd_bwd * float(sequences_per_iter_global)

    flops_achieved_per_sec = flops_per_iter / float(dt)
    flops_peak_per_sec = peak_tflops_per_gpu * 1e12 * n_gpus

    return (flops_achieved_per_sec / flops_peak_per_sec) * 100.0






from dataclasses import dataclass
from typing import Any


def _require_attr(obj: Any, name: str):
    if not hasattr(obj, name):
        raise AttributeError(f"Missing required config attribute: {name}")
    return getattr(obj, name)


def compute_grad_accum_steps(training_cfg: Any, model_cfg: Any, n_gpus: int) -> int:
    """
    Derive gradient accumulation steps from:
      total_batch_size = tokens per optimizer step (global, across all GPUs)
      batch_size       = sequences per micro-step per GPU
      block_size       = tokens per sequence

    Assumes:
      tokens_per_optimizer_step = total_batch_size
      tokens_per_micro_step_global = batch_size * block_size * n_gpus
      grad_accum_steps = total_batch_size / tokens_per_micro_step_global

    Raises if not divisible (you should make it divisible for clean accounting).
    """
    total_batch_tokens = int(_require_attr(training_cfg, "total_batch_size"))
    micro_seq_per_gpu = int(_require_attr(training_cfg, "batch_size"))
    T = int(_require_attr(model_cfg, "block_size"))

    tokens_per_micro_step_global = micro_seq_per_gpu * T * int(n_gpus)
    if tokens_per_micro_step_global <= 0:
        raise ValueError("Computed tokens_per_micro_step_global must be > 0")

    if total_batch_tokens % tokens_per_micro_step_global != 0:
        raise ValueError(
            "total_batch_size must be divisible by (batch_size * block_size * n_gpus). "
            f"Got total_batch_size={total_batch_tokens}, "
            f"batch_size={micro_seq_per_gpu}, block_size={T}, n_gpus={n_gpus} "
            f"-> tokens_per_micro_step_global={tokens_per_micro_step_global}"
        )

    return total_batch_tokens // tokens_per_micro_step_global







def compute_mfu_from_configs(
    *,
    dt: float,
    n_params_active: float,
    model_cfg: Any,
    training_cfg: Any,
    n_gpus: int,
    peak_tflops_per_gpu: float,
    include_attention: bool = True,
) -> float:
    """
    nanoGPT / PaLM Appendix-B MFU estimate (percentage).

    Inputs:
      - dt: seconds per *optimizer step* (i.e., after grad accumulation)
      - n_params_active: "active" parameter count used per token (dense => total params)
                        For MoE, pass active params (e.g., shared + top-k experts).
      - model_cfg: ModelConfig (must have: block_size, n_layer, n_head, n_embd)
      - training_cfg: TrainingConfig (must have: total_batch_size, batch_size)
      - n_gpus: number of GPUs (Kaggle T4x2 => 2)
      - peak_tflops_per_gpu: e.g. 65 for T4 FP16 tensor peak (pass from your runtime/constants)
      - include_attention: whether to add attention FLOPs term (recommended)

    Returns:
      MFU in percent (% of theoretical peak across all GPUs).
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if n_gpus <= 0:
        raise ValueError("n_gpus must be > 0")
    if peak_tflops_per_gpu <= 0:
        raise ValueError("peak_tflops_per_gpu must be > 0")
    if n_params_active <= 0:
        raise ValueError("n_params_active must be > 0")

    dt_ms = dt
    dt_s = dt_ms / 1000.0

    # ---- Pull model dimensions from config ----
    T = int(_require_attr(model_cfg, "block_size"))   # seq length / block size
    L = int(_require_attr(model_cfg, "n_layer"))
    H = int(_require_attr(model_cfg, "n_head"))
    n_embd = int(_require_attr(model_cfg, "n_embd"))

    print('T->',T)
    print('L->',L)
    print('H->',H)
    print('n_embd->',n_embd)

    if n_embd % H != 0:
        raise ValueError(f"n_embd must be divisible by n_head. Got n_embd={n_embd}, n_head={H}")
    Q = n_embd // H  # head_dim

    print('Q->',Q)

    # ---- Derive grad accumulation from configs (global tokens per optimizer step) ----
    micro_seq_per_gpu = int(_require_attr(training_cfg, "batch_size"))
    print('micro_seq_per_gpu->',micro_seq_per_gpu)
    grad_accum_steps = compute_grad_accum_steps(training_cfg, model_cfg, n_gpus)
    print('grad_accum_steps->',grad_accum_steps)

    # How many sequences are processed per optimizer step (GLOBAL, across GPUs)
    sequences_per_iter_global = micro_seq_per_gpu * n_gpus * grad_accum_steps
    print('sequences_per_iter_global->',sequences_per_iter_global)

    # ---- FLOPs accounting (PaLM/nanoGPT style) ----
    # FLOPs per token (training) ≈ 6N + 12 L H Q T
    flops_per_token = 6.0 * float(n_params_active)
    if include_attention:
        flops_per_token += 12.0 * float(L) * float(H) * float(Q) * float(T)

    # Convert per-token -> per-(fwd+bwd) over a full sequence of length T
    flops_per_fwd_bwd = flops_per_token * float(T)
    print('flops_per_fwd_bwd->',flops_per_fwd_bwd)

    # Per optimizer iteration
    flops_per_iter = flops_per_fwd_bwd * float(sequences_per_iter_global)
    print('flops_per_iter->',flops_per_iter)

    # Achieved FLOPs/s
    flops_achieved_per_sec = flops_per_iter / float(dt)
    print('flops_achieved_per_sec->',flops_achieved_per_sec)

    # Peak FLOPs/s across all GPUs
    flops_peak_per_sec = float(peak_tflops_per_gpu) * 1e12 * int(n_gpus)
    print('flops_peak_per_sec->',flops_peak_per_sec)

    mfu = flops_achieved_per_sec / flops_peak_per_sec
    print('mfu->',mfu)
    return mfu * 100.0















def arnav_compute_mfu_from_configs(
    dt_ms: float,
    n_params_active: int,
    model_cfg,
    training_cfg,
    n_gpus: int,
    grad_accum_steps: int,
    peak_tflops_per_gpu: float,
    batch_size : int,
    include_attention: bool = True,
    
    
):
   

    print('calling arnav_compute_mfu_from_configs()')

    dt = dt_ms / 1000.0

    # Unpack model config
    L = model_cfg.n_layer
    H = model_cfg.n_head
    Q = model_cfg.n_embd // model_cfg.n_head
    T = model_cfg.block_size

    # PaLM Appendix B FLOPs estimate
    flops_per_token = 6 * n_params_active

    if include_attention:
        flops_per_token += 12 * L * H * Q * T

    flops_per_fwdbwd = flops_per_token * T

    # Account for gradient accumulation
    # flops_per_iter = flops_per_fwdbwd * grad_accum_steps
    flops_per_fwdbwd = grad_accum_steps * batch_size
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    print('flops_per_iter')

    # Achieved FLOPs/sec
    flops_achieved = flops_per_iter / dt

    # Theoretical peak FLOPs/sec
    flops_peak = peak_tflops_per_gpu * 1e12 

    # MFU
    mfu = flops_achieved / flops_peak

    return mfu * 100.0







