


def create_profiler(output_dir="./profiler_logs", trace_handler=None):
    """
    Create a PyTorch profiler with customizable settings
    
    Args:
        output_dir: Directory to save profiler traces
        trace_handler: Custom trace handler function (optional)
    """

    os.makedirs(output_dir, exist_ok=True)
    
    # Default trace handler if none provided
    if trace_handler is None:
        def default_trace_handler(prof):
            # Export Chrome trace for visualization
            prof.export_chrome_trace(f"{output_dir}/trace_{prof.step_num}.json")
            
            # Print table summary
            print(prof.key_averages().table(
                sort_by="cuda_time_total", 
                row_limit=10
            ))
        
        trace_handler = default_trace_handler
    
    # Create profiler with schedule
    return profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=schedule(
            wait=1,      # Skip first iteration (warmup)
            warmup=1,    # Warmup for 1 iteration
            active=3,    # Profile for 3 iterations
            repeat=2     # Repeat the cycle 2 times
        ),
        on_trace_ready=trace_handler,
        record_shapes=True,       # Record tensor shapes
        profile_memory=True,      # Track memory allocations
        with_stack=True,          # Record stack traces
        with_flops=True,          # Estimate FLOPs
    )

