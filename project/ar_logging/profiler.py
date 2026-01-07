

import os
from torch.profiler import profile, record_function, ProfilerActivity, schedule



'''

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



'''


import os
import json
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from datetime import datetime

def create_profiler(
    output_dir="./profiler_logs", 
    trace_handler=None,
    enable_memory=True,
    enable_stack_trace=True,
    enable_flops=True,
    enable_shape_record=True,
    device="cuda"
):
    """
    Create an enhanced PyTorch profiler for Transformer LLMs with maximum logging
    
    Args:
        output_dir: Directory to save profiler traces and logs
        trace_handler: Custom trace handler function (optional)
        enable_memory: Enable memory profiling (can be heavy for large models)
        enable_stack_trace: Enable stack trace recording
        enable_flops: Enable FLOPs estimation
        enable_shape_record: Enable tensor shape recording
        device: Target device ('cuda' or 'cpu')
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create metadata file
    metadata = {
        "creation_time": datetime.now().isoformat(),
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "settings": {
            "enable_memory": enable_memory,
            "enable_stack_trace": enable_stack_trace,
            "enable_flops": enable_flops,
            "enable_shape_record": enable_shape_record
        }
    }
    
    with open(f"{output_dir}/profiler_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Default enhanced trace handler with maximum logging
    if trace_handler is None:
        def default_trace_handler(prof):
            step_num = prof.step_num
            
            # Create timestamp for this profiling step
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Export Chrome trace for visualization (most detailed)
            chrome_trace_path = f"{output_dir}/trace_{timestamp}_step{step_num}.json"
            prof.export_chrome_trace(chrome_trace_path)
            print(f"[Profiler] Chrome trace saved: {chrome_trace_path}")
            
            # 2. Export raw data for custom analysis
            try:
                raw_events = prof.events()
                with open(f"{output_dir}/raw_events_{timestamp}_step{step_num}.json", "w") as f:
                    json.dump([event.__dict__ for event in raw_events], f, indent=2, default=str)
            except Exception as e:
                print(f"[Profiler] Could not export raw events: {e}")
            
            # 3. Comprehensive table summaries with different sortings
            print(f"\n{'='*80}")
            print(f"PROFILING REPORT - Step {step_num} - {timestamp}")
            print(f"{'='*80}")
            
            # GPU Time Analysis
            print(f"\n{'='*60}")
            print("TOP OPERATIONS BY GPU TIME")
            print(f"{'='*60}")
            table_gpu_time = prof.key_averages(group_by_input_shape=True, group_by_stack_n=5).table(
                sort_by="cuda_time_total",
                row_limit=20,
                header=["Name", "Self CPU %", "Self CPU", "CPU total %", "CPU total", "CPU time avg", 
                        "Self CUDA %", "Self CUDA", "CUDA total %", "CUDA total", "CUDA time avg", 
                        "Input Shapes"]
            )
            print(table_gpu_time)
            
            # Save GPU time table
            with open(f"{output_dir}/gpu_time_table_step{step_num}.txt", "w") as f:
                f.write(table_gpu_time)
            
            # CPU Time Analysis
            print(f"\n{'='*60}")
            print("TOP OPERATIONS BY CPU TIME")
            print(f"{'='*60}")
            table_cpu_time = prof.key_averages(group_by_input_shape=True).table(
                sort_by="cpu_time_total",
                row_limit=15
            )
            print(table_cpu_time)
            
            # Memory Analysis (if enabled)
            if enable_memory:
                print(f"\n{'='*60}")
                print("MEMORY USAGE ANALYSIS")
                print(f"{'='*60}")
                
                # Memory usage by operator
                table_memory = prof.key_averages().table(
                    sort_by="self_cuda_memory_usage",
                    row_limit=15
                )
                print("Top operations by self memory usage:")
                print(table_memory)
                
                # Save memory table
                with open(f"{output_dir}/memory_table_step{step_num}.txt", "w") as f:
                    f.write(table_memory)
                
                # Additional memory statistics
                print(f"\nMemory Allocation Statistics:")
                try:
                    memory_stats = torch.cuda.memory_stats() if torch.cuda.is_available() else {}
                    for key in ['allocated_bytes.all.current', 'reserved_bytes.all.current', 
                                'allocated_bytes.all.peak', 'reserved_bytes.all.peak']:
                        if key in memory_stats:
                            value = memory_stats[key]
                            if value > 1e9:
                                print(f"  {key}: {value/1e9:.2f} GB")
                            elif value > 1e6:
                                print(f"  {key}: {value/1e6:.2f} MB")
                            else:
                                print(f"  {key}: {value/1e3:.2f} KB")
                except Exception as e:
                    print(f"  Could not retrieve detailed memory stats: {e}")
            
            # FLOPs Analysis (if enabled)
            if enable_flops:
                print(f"\n{'='*60}")
                print("FLOPS ESTIMATION")
                print(f"{'='*60}")
                
                # Get operations with FLOPs estimates
                events = prof.key_averages()
                total_flops = 0
                flops_by_op = {}
                
                for event in events:
                    if hasattr(event, 'flops') and event.flops > 0:
                        total_flops += event.flops
                        op_name = event.key
                        if op_name in flops_by_op:
                            flops_by_op[op_name] += event.flops
                        else:
                            flops_by_op[op_name] = event.flops
                
                print(f"Total Estimated FLOPs: {total_flops:,}")
                if total_flops > 0:
                    print("\nFLOPs by operation type:")
                    for op_name, flops in sorted(flops_by_op.items(), key=lambda x: x[1], reverse=True)[:10]:
                        percentage = (flops / total_flops) * 100
                        print(f"  {op_name}: {flops:,} ({percentage:.1f}%)")
            
            # Shape Statistics
            if enable_shape_record:
                print(f"\n{'='*60}")
                print("TENSOR SHAPE STATISTICS")
                print(f"{'='*60}")
                
                events = prof.key_averages(group_by_input_shape=True)
                shape_count = {}
                
                for event in events:
                    if hasattr(event, 'input_shapes') and event.input_shapes:
                        shapes_str = str(event.input_shapes)
                        if shapes_str in shape_count:
                            shape_count[shapes_str] += 1
                        else:
                            shape_count[shapes_str] = 1
                
                print(f"Unique tensor shape patterns: {len(shape_count)}")
                print("\nMost frequent tensor shapes:")
                for shapes_str, count in sorted(shape_count.items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"  Count {count}: {shapes_str}")
            
            # Kernel Statistics (CUDA specific)
            if torch.cuda.is_available() and device == "cuda":
                print(f"\n{'='*60}")
                print("CUDA KERNEL STATISTICS")
                print(f"{'='*60}")
                
                table_kernels = prof.key_averages(group_by_stack_n=5).table(
                    sort_by="cuda_time_total",
                    row_limit=10
                )
                print(table_kernels)
            
            # Save summary to JSON
            summary = {
                "step": step_num,
                "timestamp": timestamp,
                "total_cuda_time": sum(event.cuda_time_total for event in prof.key_averages()),
                "total_cpu_time": sum(event.cpu_time_total for event in prof.key_averages()),
                "event_count": len(list(prof.events())) if hasattr(prof, 'events') else 0,
                "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                "gpu_memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
            }
            
            with open(f"{output_dir}/summary_step{step_num}.json", "w") as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n{'='*80}")
            print(f"END OF PROFILING REPORT - Step {step_num}")
            print(f"{'='*80}\n")
            
            # Log file locations
            log_files = [
                chrome_trace_path,
                f"{output_dir}/gpu_time_table_step{step_num}.txt",
                f"{output_dir}/summary_step{step_num}.json"
            ]
            
            if enable_memory:
                log_files.append(f"{output_dir}/memory_table_step{step_num}.txt")
            
            print(f"Profiling data saved to:")
            for log_file in log_files:
                print(f"  - {log_file}")
        
        trace_handler = default_trace_handler
    
    # Configure activities based on device
    activities = [ProfilerActivity.CPU]
    if device == "cuda" and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    elif device == "cuda" and not torch.cuda.is_available():
        print(f"[Warning] CUDA requested but not available. Profiling CPU only.")
    
    # Create profiler with schedule optimized for Transformer LLMs
    prof = profile(
        activities=activities,
        schedule=schedule(
            wait=1,       # Skip first iteration (warmup)
            warmup=2,     # Warmup for 2 iterations (important for transformer caching)
            active=4,     # Profile for 4 iterations
            repeat=1      # Repeat the cycle once
        ),
        on_trace_ready=trace_handler,
        record_shapes=enable_shape_record,
        profile_memory=enable_memory,
        with_stack=enable_stack_trace,
        with_flops=enable_flops,
        with_modules=True,      # Track module hierarchy
        with_threads=True,      # Track thread information
    )
    
    print(f"[Profiler] Created with output directory: {output_dir}")
    print(f"[Profiler] Memory profiling: {'ENABLED' if enable_memory else 'DISABLED'}")
    print(f"[Profiler] Stack tracing: {'ENABLED' if enable_stack_trace else 'DISABLED'}")
    print(f"[Profiler] FLOPs estimation: {'ENABLED' if enable_flops else 'DISABLED'}")
    print(f"[Profiler] Shape recording: {'ENABLED' if enable_shape_record else 'DISABLED'}")
    
    return prof