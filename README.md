# Transformer_GPU
Improving training of LLM models following parallelism techniques

Command to run the code:
\n
!torchrun --standalone --nproc_per_node=2 my_code.py --moe --aux_free --eval --max_iters=250 --eval_interval=50 --attn gqa
