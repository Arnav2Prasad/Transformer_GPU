
'''
Grouped-Query Attention (GQA) is an architecture that balances the high quality of Multi-Head Attention (MHA) 
    with the inference speed and memory efficiency of Multi-Query Attention (MQA).

Its core idea is to group Query heads, where each group of Queries shares a single Key (K) and Value (V) head. 
This reduces the size of the K/V cache, which is a major bottleneck during the inference phase of large 
language models.

The Problem: The KV Cache Bottleneck
To understand GQA, you first need to understand the role of the Key-Value (KV) cache in transformer 
            inference.

Autoregressive Decoding: When a model generates text, it does so one token at a time. At each step, the model 
            attends to all the previously generated tokens.



The Cache: To avoid re-computing the Key and Value vectors for all previous tokens at every new step, they are 
        stored in memory. This is the KV cache.

The Bottleneck: For long sequences, this cache becomes very large. Loading the entire cache from slow 
        High-Bandwidth Memory (HBM) to the fast on-chip SRAM for every single new token generation is the 
        main performance bottleneck.


---- 
Attention Mechanisms Compared
Let's compare MHA, MQA, and GQA to see how they address this.

Imagine we have a model with 8 Query heads.


1. Multi-Head Attention (MHA)

= In standard MHA, every Query (Q) head has its own dedicated Key (K) and Value (V) head.
= Structure: 8 Q heads, 8 K heads, 8 V heads.


->Pros: Highest model quality and performance.

->Cons: Creates a very large KV cache, leading to slow inference due to high memory bandwidth requirements.



2. Multi-Query Attention (MQA)

MQA is one extreme optimization. All Query heads share a single Key and Value head.
= Structure: 8 Q heads, 1 K head, 1 V head.


->Pros: Drastically reduces the KV cache size, making inference much faster.
->Cons: Can sometimes lead to a noticeable degradation in model quality.



3. Grouped-Query Attention (GQA)

GQA is the middle ground, offering the best of both worlds. It groups the Query heads and assigns a 
        single K/V pair to each group.

Structure: Let's create 2 groups. We have 8 Q heads, so each group has 4 Q heads. 
Each group gets one K/V pair. This results in 8 Q heads, 2 K heads, and 2 V heads.


->Pros: Significantly reduces the KV cache size compared to MHA (though not as much as MQA), leading to faster 
        inference while maintaining quality very close to the original MHA model.
->Cons: Slightly more complex to implement than MHA or MQA.



What is GQA?
Think of MHA→MQA→GQA as a spectrum based on how many K/V head sets you keep:


---> MHA (Multi-Head Attention): n_kv_heads = n_head
        
    Each query head has its own K and V projection. Max flexibility, max memory.


---> MQA (Multi-Query Attention): n_kv_heads = 1
    All query heads share a single K and a single V. Minimal memory, least flexible.


---> GQA (Grouped-Query Attention): 1 < n_kv_heads < n_head
Query heads are partitioned into groups; each group shares one K and one V. You trade a bit of flexibility for a big win in KV-cache size and projection parameters.
Key idea (one sentence)
Let r = n_head / n_kv_heads = queries per KV group. Each group of r query heads uses the same K and V tensors.
Shapes & math (with a concrete example)
Let:
Batch B, sequence length T
Embedding C
Query heads n_head
KV heads n_kv_heads
Head size d = C / n_head
Group size r = n_head / n_kv_heads (must be an integer)
Projections
Q: x @ W_q -> (B, T, n_head, d)
K: x @ W_k -> (B, T, n_kv_heads, d)
V: x @ W_v -> (B, T, n_kv_heads, d)
Grouping / broadcast
Each KV head is shared by r query heads. Implementation-wise you can:
Either repeat/broadcast K,V across the r query heads in its group
Or index the correct K,V for each query head’s group during attention
Attention per query head
For query head h, let g = floor(h / r) be its KV group index (0..n_kv_heads-1). Then:
Use Q[..., h, :] with K[..., g, :] and V[..., g, :].
Example numbers
Say C=1024, n_head=8, n_kv_heads=2 ⇒ d=128, r=4:
Q: (B,T,8,128)
K,V: (B,T,2,128)
Heads 0–3 attend with KV-head 0; heads 4–7 with KV-head 1.
Why this helps (parameters & KV cache)
Projection parameters for K and V shrink from C × (n_head·d) = C×C to
C × (n_kv_heads·d) = C × (C·n_kv_heads/n_head) = C×C / r (a 1/r reduction).
KV-cache memory at inference also drops by 1/r (since you cache K,V for n_kv_heads not n_head).
Compute cost is similar for attention matmuls (you still compute scores for all query heads), but bandwidth/memory wins are large—this is why GQA is popular on long sequences.
How RoPE (or PE) fits
If you use RoPE, apply it to Q and K on a per-head basis. In GQA, that means apply RoPE to:
Q for every query head
K for each KV head (then shared across the group)






An embedding is a vector representation of something (like a word, token, or item) in a continuous numerical space.
Original data (text, IDs, categories) is discrete (e.g., "cat", "dog", "apple").
Neural networks can’t work directly with raw symbols—they need numbers.
An embedding maps each discrete symbol into a dense vector of floats that captures meaning and relationships.
So instead of treating "cat" and "dog" as unrelated IDs, embeddings let the model learn vectors like:
"cat" → [0.2, -0.1, 0.9, ...]
"dog" → [0.3, -0.2, 0.8, ...]
"apple" → [-0.5, 0.7, -0.1, ...]
2. In LLMs (like GPT)
Vocabulary size: maybe 50,000 tokens (words/pieces).
Embedding size (n_embd): e.g. 768, 1024, 4096.
The embedding layer is basically a lookup table:
A big matrix of shape (vocab_size, n_embd).
When a token ID comes in:
The model looks up its corresponding row in this matrix.
That row is a vector of length n_embd.
Example:
If n_embd=1024, then every token is represented as a 1024-dimensional float vector before going into attention layers.
3. Why embeddings are powerful
Because during training, embeddings are adjusted so that similar things end up near each other in the vector space.
"cat" and "dog" embeddings might be closer than "cat" and "apple".
"king" - "man" + "woman" ≈ "queen" (famous word2vec analogy).
This happens because the model learns to arrange embeddings to minimize prediction error on language modeling.
4. Difference from “one-hot”
One-hot encoding: "cat" = [0,0,0,1,0,0,...] (very sparse, no notion of similarity).
Embedding: "cat" = [0.2, -0.1, 0.9, ...] (dense, learned, meaningful).
So embeddings are like a compressed, semantic replacement for one-hot vectors.
5. Embedding size (n_embd)
This is exactly what your earlier asserts are checking.
It’s the width of these vectors (per token).
It must divide evenly across attention heads → so each head sees a slice of the embedding.
✅ Summary:
An embedding is a learned dense vector that represents discrete tokens (words, subwords, IDs) in a continuous space. In LLMs, n_embd is the size of that vector, and it’s the backbone dimension used everywhere in the Transformer (attention, MLPs, residuals).


'''



'''
ColumnParallelLinear (q_proj, kv_proj): 
    - Split input, independent computation
    - No cross-GPU communication during forward

RowParallelLinear (c_proj):
    - Independent computation on sharded input  
    - All-reduce to combine results
'''

class GQA(nn.Module):
    """ Grouped-Query Attention with or without RoPE """

    '''
    Input x: (B, T, n_embd)

    # After q_proj (TP sharded):
    q_local: (B, T, n_embd/tp_size) → reshaped to (B, T, n_head_per_rank, head_size)

    # After kv_proj:
    if partitioned: (B, T, kv_out_features/tp_size)  
    else: (B, T, kv_out_features)  # replicated

    # After attention + c_proj:
    y: (B, T, n_embd)  # gathered via all-reduce
    '''

    def __init__(self, config:LLMconfig, tp_group = None):
        super().__init__()

        # for MHA : nummber of kv heads = number of heads
        if config.attn == 'mha' : config.n_kv_heads = config.n_head

        # for MQA : nummber of kv heads = number of heads
        elif config.attn == 'mqa' : config.n_kv_heads = 1

        # for GQA : nummber of kv heads = number of heads
        #ie every KV matrix should applied for the same numner of groups
        else : assert config.n_head % config.n_kv_heads == 0, "n_head must be divisible by n_kv_heads"
        

        # n_embd -> width(columns) of the key matrix = K1 ,K2 should be divisible by the number of heads
        # input batch x has shape (B, T, n_embd)
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"



        # You’ll read hyperparams like n_embd, n_head, n_kv_heads, dropout, etc., 
        # from here throughout the module.
        self.config = config



        # for TP -> added the below line
        '''
        tp_group: which workers (GPUs/processes) participate in tensor parallelism together.
        tp_size: how many workers are in that group (e.g., 2, 4, 8).
        tp_rank: this worker’s index in the group (0 … tp_size-1).
        '''
        self.tp_group, self.tp_size, self.tp_rank = _get_group_and_ranks(tp_group)


        # Critical divisibility assertions
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"


        '''
        tp_size = number of GPUs
        What it enforces
            The number of query heads must split evenly across TP ranks.

        Why you need it
            In TP, you typically shard the head dimension so each rank computes attention for a subset of heads. 
            If n_head doesn’t divide tp_size, some rank would get a fractional number of heads—impossible to 
                index or compute.

        '''
        assert config.n_head % self.tp_size == 0, "n_head must be divisible by tp_size"


        assert config.n_embd % self.tp_size == 0, "n_embd must be divisible by tp_size"





        # n_emb acts as width 
        #  n_head acts as divisor:)
        #for each gpu, what would be the head size?
        self.head_size = config.n_embd // config.n_head  



        #divide heads across GPUs
        # Distribute heads across TP ranks
        self.n_head_per_rank = config.n_head // self.tp_size



        '''
        # k,q,v in a btach
        self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * config.n_kv_heads * self.head_size)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        '''


        # Critical: Handle KV head partitioning correctly
        # number of kv heads = number of groups
        # so this is a check if we can apply TP or not
        self.partition_kv = (config.n_kv_heads % self.tp_size == 0)


        '''
        n_head = total query heads.
        n_kv_heads = total key/value heads (aka groups).

        Each KV head (group) is shared by r = n_head / n_kv_heads query heads.
        Shapes after projection (before grouping/broadcast):
        Q: (B, T, n_head, d)
        K: (B, T, n_kv_heads, d)
        V: (B, T, n_kv_heads, d)
        where d = n_embd / n_head is per-head dimension.
        So n_kv_heads controls both:
        How many distinct K/V sets you learn/cache, and
        The group size (r) of query heads that share each K/V.

        '''


        if self.partition_kv:
            # here we are finding the -> number of groups per GPU
            # Each GPU becomes responsible for a unique, smaller subset of the total KV heads.
            self.n_kv_heads_per_rank = config.n_kv_heads // self.tp_size


            # Additional safety check for KV projection divisibility
            '''
            Your KV projection layer (kv_proj) takes x of shape (B, T, n_embd) and produces both K and V in 
                    one go by concatenating them along features.
            For each KV head, you need a vector of length head_size for K and another head_size for V → that’s 
                    2 * head_size features per KV head.

            '''
            kv_out_features = 2 * config.n_kv_heads * self.head_size


            assert kv_out_features % self.tp_size == 0, \
                "KV out features must be divisible by tp_size when partitioning KV"
        else:
            # every gpu has all the kv heads
            self.n_kv_heads_per_rank = config.n_kv_heads  # Replicated on all ranks


        


        # Q projection: Always TP-sharded
        '''
        gather_output=False: This is a crucial instruction for parallelism. 
        It means that after each GPU computes its part of the output, the results are not gathered back 
            together into a single, full-sized tensor. The output remains split, or "sharded," across the GPUs.

        group=self.tp_group: This specifies the group of GPUs that are working together and across which the 
            layer should be split. 

        '''


        '''
        Input: (B, T, n_embd) - Full embedding dimension
        Output: (B, T, n_embd/tp_size) - Sharded across TP group
        Weight Matrix: W_q of shape (n_embd, n_embd) is split column-wise

        Total query heads: n_head
        Heads per GPU: n_head_per_rank = n_head / tp_size
        Features per GPU: n_embd / tp_size

        Example: If n_embd=1024, tp_size=4, each GPU gets 256 features containing its portion of query heads.
        '''
        # split the vector into H slices, compute attention separately for each slice
        self.q_proj = ColumnParallelLinear(
            config.n_embd, config.n_embd, 
            bias=True, gather_output=False, group=self.tp_group
        )


        '''
        n_kv_heads: Number of KV heads (groups)
        head_size = n_embd / n_head
        2: For both Keys and Values concatenated
        Partitioning Logic:

        If divisible (partition_kv=True): Split KV heads across GPUs
        If not divisible: Replicate KV heads on all GPUs
        '''

        
        # KV projection: Sharded only if divisible, else replicated
        kv_out_features = 2 * config.n_kv_heads * self.head_size


        if self.partition_kv:
            self.kv_proj = ColumnParallelLinear(
                config.n_embd, kv_out_features,
                bias=True, gather_output=False, group=self.tp_group
            )
        else:
            self.kv_proj = nn.Linear(config.n_embd, kv_out_features, bias=True)



        '''
        Input: (B, T, n_embd/tp_size) - Sharded input from attention computation
        Output: (B, T, n_embd) - Full output after all-reduce
        Weight Matrix: W_out of shape (n_embd, n_embd) is split row-wise

        '''
        
        # Output projection: RowParallel with all-reduce
        self.c_proj = RowParallelLinear(
            config.n_embd, config.n_embd,
            bias=True, input_is_parallel=True, group=self.tp_group
        )
        
        self.resid_dropout = nn.Dropout(config.dropout)


        


    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor|None, kv_cache=None, VAL_RUN=False):
        '''
        B, T, C = x.size()
        nh, nkvh, hs = self.config.n_head , self.config.n_kv_heads, self.head_size

        q_proj_size = C # n_embd
        kv_proj_size = nkvh * hs
        q, k, v = self.c_attn(x).split([q_proj_size, kv_proj_size, kv_proj_size], dim=2)
        q:torch.Tensor = q.view(B, T, nh, hs) # (B, T, nh, hs)
        k:torch.Tensor = k.view(B, T, nkvh, hs) # (B, T, n_kvh, hs)
        v:torch.Tensor = v.view(B, T, nkvh, hs).transpose(1, 2) # (B, n_kvh, T, hs)

        if self.config.pos_emb == 'rope':
        # Apply RoPE
            q = LLMconfig.apply_rotary_emb(q, freqs_cis)
            k = LLMconfig.apply_rotary_emb(k, freqs_cis)

        q,k = q.transpose(1, 2), k.transpose(1, 2) # (B, nh, T, hs) # (B, n_kvh, T, hs)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        updated_kv_cache = (k, v)

        if nkvh != nh:
            num_repeats = nh // nkvh
            k = k.repeat_interleave(num_repeats, dim=1)
            v = v.repeat_interleave(num_repeats, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.config.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y, updated_kv_cache
        '''


        '''
        Input Tensor Structure:

            x: Input tensor of shape (B, T, C)

            B = Batch size (number of sequences)
            T = Sequence length (number of tokens)
            C = Embedding dimension (config.n_embd)
            hs = self.head_size = C / n_head

            This is the dimension of each attention head
        '''

        B, T, C = x.size()
        hs = self.head_size
        


        '''
        Before projection:

        
        x shape: (B, T, C)  # Full embedding dimension
        After ColumnParallelLinear projection:


        q_local shape: (B, T, C/tp_size)  # Sharded across TP group
        Mathematical Transformation:

        The ColumnParallelLinear layer splits the output features across the tensor parallelism group
        If tp_size = 2, C = 1024, then each GPU gets 1024/2 = 512 output features
        This is equivalent to splitting the query heads across GPUs
        '''
        # Query projection (always TP-sharded) - with contiguity guarantee
        q_local = self.q_proj(x)  # [B, T, C/tp_size]


        '''
        example: C=1024, n_head=8, hs=128, tp_size=2 → n_head_per_rank=4.
            q_local: [B,T,512] gets viewed as [B,T,4,128].
        '''
        # C/tp_size = (n_head/tp_size) * head_size = n_head_per_rank * hs


        q = q_local.contiguous().view(B, T, self.n_head_per_rank, hs)
        




        # Key-Value projection (sharded or replicated)
        # here we are shardig key and value column wise
        kv = self.kv_proj(x)


        
        # Shape safety check for KV projection output
        if self.partition_kv:
            expected_kv_dim = 2 * self.n_kv_heads_per_rank * hs
        else:
            expected_kv_dim = 2 * self.config.n_kv_heads * hs


        
        assert kv.shape[-1] == expected_kv_dim, \
            f"KV projection output dim {kv.shape[-1]} != expected {expected_kv_dim}"


        '''
        Instead of having two separate linear layers—one for Keys and one for Values—the model uses a single, wider layer called self.kv_proj. 
                This is a common optimization.

        This line performs one matrix multiplication to create a single, 
            combined tensor kv that contains all the data for both the Key and Value heads packed together side-by-side. 
            
        Think of the output kv tensor's last dimension as [all_key_data | all_value_data].

        '''
        
        # Calculate split sizes based on actual output
        kv_split_size = expected_kv_dim // 2

        k, v = kv.split([kv_split_size, kv_split_size], dim=2)


        

        # Ensure contiguity before view operations
        k = k.contiguous().view(B, T, self.n_kv_heads_per_rank, hs)
        v = v.contiguous().view(B, T, self.n_kv_heads_per_rank, hs)

        
        # Apply rotary embeddings if needed (expects [B, T, heads, hs])
        if self.config.pos_emb == 'rope' and freqs_cis is not None:
            # q = self.apply_rotary_emb(q, freqs_cis)
            # k = self.apply_rotary_emb(k, freqs_cis)
            q = LLMconfig.apply_rotary_emb(q, freqs_cis)  # ✅ Static method call
            k = LLMconfig.apply_rotary_emb(k, freqs_cis)  # ✅ Static method call


        
        # Transpose for attention: [B, heads, T, hs]
        '''
        This line reshapes the Query, Key, and Value tensors to get them into the standard format expected by most optimized attention functions.

        Before Transpose: After the previous reshaping step, the tensors have a shape of 
            (B, T, heads, hs), which stands for (Batch, Tokens, Heads, Head_Size).

        The Operation: The .transpose(1, 2) command swaps dimension 1 (the T dimension) with dimension 2 (the heads dimension).

        After Transpose: The new shape is (B, heads, T, hs).
        '''
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)



        '''
        What is KV Cache?

            KV Cache (Key-Value Cache) is an optimization technique used during autoregressive text 
                generation where tokens are generated one at a time.


        The Problem:

            When generating token #N, we need to attend to all previous tokens #0 through #(N-1)
            Without caching, we'd need to recompute Keys and Values for all previous tokens at each step
            This would be computationally expensive: O(N²) complexity.


        The Solution: KV Cache

            Store computed Keys and Values for all previous tokens
            When generating new token, only compute Keys/Values for the new token
            Concatenate with cached previous tokens.
        '''

        # Handle KV cache
        # re concatenate past keys with new keys -> on dimension = T
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        updated_kv_cache = (k, v)


        
        # Repeat KV heads to match Q heads if needed - WITH SAFETY ASSERT
        '''
        Attention kernels expect the same head count for Q, K, and V.
        But with GQA/MQA we have fewer KV heads than Q heads: H_kv_local < H_q_local.
        So we replicate each KV head across its group of query heads:

        Group size (per rank):
                r_local = H_q_local / H_kv_local
 
        The assert guarantees this ratio is an integer (no fractional heads).

        '''
        if self.n_kv_heads_per_rank != self.n_head_per_rank:
            assert self.n_head_per_rank % self.n_kv_heads_per_rank == 0, \
                "Local n_head must be a multiple of local n_kv_heads when repeating."
            num_repeats = self.n_head_per_rank // self.n_kv_heads_per_rank

            '''
            This is the "magic" of GQA.

            What it does: It expands the k tensor by repeating its heads. 
            The repetition happens along dim=1, which is the head dimension in our (Batch, Heads, Tokens, Head_Size) tensor.


            If we have 8 query heads and 2 key heads on this GPU

            Example: If k originally had 2 heads [k_head_0, k_head_1], after this line, 
                it will have 8 heads: [k_head_0, k_head_0, k_head_0, k_head_0, k_head_1, k_head_1, k_head_1, k_head_1].

            The exact same operation is performed on the v tensor.

            '''
            k = k.repeat_interleave(num_repeats, dim=1)
            v = v.repeat_interleave(num_repeats, dim=1)




        '''
        Inputs (head-major layout):
        q: [B, H_q_local, T_q, d]
        k: [B, H_q_local, T_k, d] (after your GQA repeat, K/V head count matches Q)
        v: [B, H_q_local, T_k, d]
        '''
        # Scaled dot-product attention
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.config.dropout if self.training else 0,
            is_causal=True    #this forces is to be causal attention -> present should depend on past and not on future
        )
        # now this 'y' -> after attention , the dimension is  (B , number_of_heads_per_rank , T , Hs)
        
        # Reshape back to [B, T, local_dim]
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        # y.transpose(1, 2) -> (B , T , number_of_heads_per_rank , Hs)
        # this falltens last 2 dimesnion : view(B, T, -1) -> number_of_heads_per_rank * Hs = Embedding_Dim_per_GPU
        # so finally we get (B , T , C_local)
        '''

        Flattens the last two dimensions (Heads and Head_Size) into a single dimension. 
        The -1 automatically calculates the correct size, which is Heads * Head_Size (the local_dim for this GPU). 
        The final shape is (Batch, Tokens, Embedding_Dim_per_GPU)


        This is the final linear transformation of the attention output.

        self.c_proj(y): The reshaped attention output y is passed through the final projection layer (c_proj). 
            If using Tensor Parallelism, this is a RowParallelLinear layer, which not only performs the linear transformation but also 
            combines the results from all GPUs (using an all-reduce operation) to produce the complete output tensor.

        self.resid_dropout(...): A final dropout is applied for regularization.

        '''
        
        # Output projection with all-reduce
        # so (B , T , C_local)
        y = self.resid_dropout(self.c_proj(y))
        
        return y, updated_kv_cache


'''
1. The reshape step
y = y.transpose(1, 2).contiguous().view(B, T, -1)


After attention, y has shape [B, n_head_per_rank, T, hs].

Transposing swaps the head and sequence axes: [B, T, n_head_per_rank, hs].

.view(B, T, -1) flattens the last two dimensions (head × head_size) into local embedding dimension per TP rank:

local embedding size per rank
local embedding size per rank=n_head_per_rank×head_size

Why it works:

n_head_per_rank = n_head / tp_size → each TP rank only has a slice of the full embedding.

head_size = n_embd / n_head → multiplying n_head_per_rank * head_size = C / tp_size.

So after view(B, T, -1), y’s last dimension = local embedding for this TP rank.

✅ This aligns perfectly with row-parallel linear, which expects input shape [B, T, C_local].

2️⃣ Row-parallel output projection
y = self.resid_dropout(self.c_proj(y))


self.c_proj is RowParallelLinear:

Shards the input dimension (C_local) across TP ranks.

Each rank computes a partial matrix multiplication: [B, T, C_local] @ [C_local, C_out].

To produce consistent output across ranks, it typically performs an all-reduce internally.

Why this works:

The previous reshape guarantees that each rank has exactly the local slice of the embedding dimension that RowParallelLinear expects.

The row-parallel projection is designed to combine results across TP ranks. So even though each rank only sees C_local, the all-reduce ensures the final result behaves as if the full embedding dimension was multiplied by a standard dense layer.

Dropout is applied after the projection as regularization.


'''


'''
out_features is short for output features, and it simply means the output size of a neural network layer.



The Factory Analogy 🏭

->Think of a neural network layer, like nn.Linear, as a factory.
-> Input (in_features): The raw materials you send into the factory. This is a vector of a certain size.
-> The Layer (nn.Linear): The factory itself, with all its machinery (the weights).
-> Output (out_features): The finished products that come out of the factory. This is a new vector, and its 
        size is the out_features.



So, out_features just defines how big the output vector will be after it passes through the layer.



How It Applies to Your Code

In your code, the specific variable is kv_out_features.

```
kv_out_features = 2 * config.n_kv_heads * self.head_size
```


This variable represents the total combined size of all the Key (K) and Value (V) vectors that the kv_proj layer 
needs to create.


Let's break down the formula:
1. config.n_kv_heads: The number of Key heads you have. Let's say you have 8. You'll also have 8 Value heads.
2. self.head_size: The size of the vector for each head. Let's say this is 128.
3. Size for all Keys: To get all the Key vectors, you need 8 heads * 128 size = 1024.
4. Size for all Values: To get all the Value vectors, you also need 8 heads * 128 size = 1024.

->The 2 *: The kv_proj layer is designed to be efficient and calculates both the K and V vectors at the same time. So, its total output needs space for both.

Total Output Size (kv_out_features) = (Size for all Keys) + (Size for all Values)
= 1024 + 1024
= 2048

This matches the formula: 2 * 8 * 128 = 2048.

So, kv_out_features represents the total size of the single large vector that comes out of the kv_proj layer, 
which contains all the individual Key and Value vectors packed together. The model then splits this large vector 
up to use the K and V parts separately in the attention calculation.


Question : but why its a scalar value? output would be a matrix right?? 
        so 2D or multi dimension values should be there right??

Answer
out_features is a scalar integer.
It doesn’t store a matrix itself — it just tells the layer how many numbers each input vector should be 
    mapped into.
PyTorch uses that integer to build the weight matrix of shape (in_features, out_features).

out_features is like the width of the output per token.

''' 