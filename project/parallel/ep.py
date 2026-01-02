from math import ceil


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


