



# Common device configuration logic
def setup_device_and_seeds(rank, local_rank):
    """Common setup for device configuration and seeding"""
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    torch.manual_seed(1729 + rank)
    torch.cuda.manual_seed(1729 + rank)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return device

# Common master process check and printing
def check_and_print_master(rank, world_size, flag_name=""):
    """Check if master process and print world size"""
    master_process = rank == 0
    if master_process:
        if flag_name:
            print(f"{flag_name}_WORLD_SIZE = {world_size}")
        else:
            print(f"Num GPUs = {world_size}")
    return master_process





# Broadcast function to ensure all ranks have same data
def broadcast_batch(x, y, src=0):
    """Ensure all TP ranks have the same batch"""
    if dist.is_initialized():
        dist.broadcast(x, src=src)
        dist.broadcast(y, src=src)
    return x, y





def all_gather_sequence(tensor: torch.Tensor, dim: int, group=None) -> torch.Tensor:
    """Efficient all-gather along specified dimension using all_gather_into_tensor"""
    if not torch.distributed.is_initialized():
        return tensor
        
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return tensor

    # Move target dimension to front for all_gather_into_tensor
    perm = list(range(tensor.ndim))
    perm[0], perm[dim] = perm[dim], perm[0]
    t_perm = tensor.permute(perm).contiguous()

    T_local = t_perm.size(0)
    out_perm = torch.empty(
        (T_local * world_size, *t_perm.shape[1:]),
        dtype=t_perm.dtype, 
        device=t_perm.device
    )

    torch.distributed.all_gather_into_tensor(out_perm, t_perm, group=group)

    inv_perm = list(range(tensor.ndim))
    inv_perm[0], inv_perm[dim] = inv_perm[dim], inv_perm[0]
    out = out_perm.permute(inv_perm).contiguous()
    
    return out





def reduce_scatter_sequence(tensor, group=None):
    """Reduce-scatter sequence chunks to context parallel ranks"""
    world_size = torch.distributed.get_world_size(group=group)
    
    if world_size == 1:
        return tensor
    
    # Split tensor into chunks for reduce-scatter
    tensor_chunks = list(tensor.chunk(world_size, dim=1))
    output = torch.zeros_like(tensor_chunks[0])
    torch.distributed.reduce_scatter(output, tensor_chunks, group=group)
    
    return output


