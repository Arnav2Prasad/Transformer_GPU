import os
import time

# Path to train.py
train_file = "train.py"

def overwrite_train(parallel_value):
    """Overwrite train.py with the given parallel_flag value."""
    with open(train_file, "w") as f:
        f.write(f"parallel_flag = {parallel_value}\n")
        f.write(f"print('parallel_flag : ', parallel_flag)\n")
    print(f"train.py updated with parallel_flag = {parallel_value}")

def run_torchrun():
    """Run the torchrun command."""
    cmd = (
        "torchrun --standalone --nproc_per_node=2 main.py "
        "--moe --aux_free --eval --max_iters=250 --eval_interval=50 --attn gqa"
    )
    print("Running torchrun command...")
    os.system(cmd)

def main():
    # First run
    overwrite_train(8)
    run_torchrun()

    # Print empty spaces for clarity in logs
    print("\n" * 10)

    # Second run
    overwrite_train(5)
    run_torchrun()

if __name__ == "__main__":
    main()
