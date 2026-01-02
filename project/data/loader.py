

import torch

class DataLoader:
    def __init__(self, B, T, file_path, device, context_parallel_size=1, context_parallel_rank=0):
        self.B = B
        self.T = T
        self.file_path = file_path
        
        # Setup device
        self.device = torch.device(device)
        self.device_type = self.device.type
        
        # Setup context parallel if needed
        if parallel_flag == 8:
            self._setup_context_parallel(context_parallel_size, context_parallel_rank)
        
        # Load memory-mapped tokens
        self._load_tokens()
        
        # Validate dataset size
        self._validate_dataset_size()
    
    def _setup_context_parallel(self, context_parallel_size, context_parallel_rank):
        """Setup for context parallel configuration"""
        self.context_parallel_size = context_parallel_size
        self.context_parallel_rank = context_parallel_rank
        
        # Calculate local sequence length
        self.local_T = self.T // context_parallel_size
        assert self.T % context_parallel_size == 0, "Sequence length must be divisible by context parallel size"
    
    def _load_tokens(self):
        """Load memory-mapped tokens from file"""
        self.tokens = np.memmap(self.file_path, dtype=np.uint16, mode='r')
        self.N = len(self.tokens)
    
    def _validate_dataset_size(self):
        """Validate that batch size and sequence length fit in dataset"""
        if self.B * self.T + 1 > self.N:
            raise ValueError(f"Batch size {self.B} and block size {self.T} are too large for dataset of length {self.N}")
    
    def _sample_start_indices(self, batch_size, sequence_length):
        """Sample random starting positions for sequences"""
        return torch.randint(0, self.N - sequence_length - 1, (batch_size,))
    
    def _move_to_device(self, x, y):
        """Move tensors to the appropriate device with optimization"""
        if self.device_type == 'cuda':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y
    
    def _process_standard_batch(self):
        """Process a batch for standard (non-context parallel) configuration"""
        B, T = self.B, self.T
        start_indices = self._sample_start_indices(B, T)
        
        x_list = []
        y_list = []
        
        for start in start_indices:
            seq = self.tokens[start: start + T + 1].astype(np.int64)
            x_list.append(seq[:-1])
            y_list.append(seq[1:])
        
        # Stack into tensors
        x = torch.from_numpy(np.stack(x_list)).long()
        y = torch.from_numpy(np.stack(y_list)).long()
        
        return x, y
    
    def _process_context_parallel_batch(self):
        """Process a batch for context parallel configuration"""
        B, local_T = self.B, self.local_T
        start_indices = self._sample_start_indices(B, self.T)
        
        x_list = []
        y_list = []
        
        for start in start_indices:
            full_seq = self.tokens[start: start + self.T + 1].astype(np.int64)
            
            # Extract local chunk for this context parallel rank
            local_start = self.context_parallel_rank * local_T
            local_end = local_start + local_T
            x_local = full_seq[local_start:local_end]
            y_local = full_seq[local_start + 1:local_end + 1]
            
            x_list.append(x_local)
            y_list.append(y_local)
        
        x = torch.from_numpy(np.stack(x_list)).long()
        y = torch.from_numpy(np.stack(y_list)).long()
        
        # Verify local sequence length
        assert x.shape[1] == self.local_T, f"Expected local_T={self.local_T}, got {x.shape[1]}"
        
        return x, y
    
    def next_batch(self):
        """
        Returns (x, y) where:
        - x is (B, T) input tokens
        - y is (B, T) target tokens (shifted by one)
        """
        if parallel_flag == 8:
            x, y = self._process_context_parallel_batch()
        else:
            x, y = self._process_standard_batch()
        
        # Move to device
        x, y = self._move_to_device(x, y)
        
        return x, y
    
    def close(self):
        """Close memory-mapped file to release resources"""
        try:
            # Access the underlying mmap object and close it
            if hasattr(self.tokens, '_mmap'):
                self.tokens._mmap.close()
            # Also try to delete the reference
            del self.tokens
        except Exception as e:
            # Silently fail if closing doesn't work
            pass

