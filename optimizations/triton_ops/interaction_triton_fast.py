# Triton kernel for optimized pairwise interaction computation
# Supports dynamic shapes and efficient memory access patterns
import triton
import triton.language as tl
import torch

@triton.jit
def interaction_kernel(emb_ptr, out_ptr, B: tl.constexpr, F: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    """
    Optimized pairwise interaction kernel using Triton
    
    Args:
        emb_ptr: Pointer to embedding tensor [B, F, D]
        out_ptr: Pointer to output tensor [B, F*(F-1)/2]
        B: Batch size
        F: Number of features
        D: Embedding dimension
        BLOCK: Block size for vectorized operations
    """
    pid = tl.program_id(axis=0)
    total_pairs = F * (F - 1) // 2
    batch_idx = pid // total_pairs
    pair_idx = pid % total_pairs

    # Efficiently compute upper triangular indices (ai, aj) where ai < aj
    # Using mathematical formula to avoid loops
    # For pair_idx, find ai such that ai*(2F-ai-1)/2 <= pair_idx < (ai+1)*(2F-ai-2)/2
    ai = tl.int32(0)
    cumsum = tl.int32(0)
    
    # Optimized index computation with minimal branching
    for ii in range(F - 1):
        next_cumsum = cumsum + (F - ii - 1)
        cond = pair_idx >= next_cumsum
        ai = tl.where(cond, ii + 1, ai)
        cumsum = tl.where(cond, next_cumsum, cumsum)
    
    aj = ai + 1 + (pair_idx - cumsum)
    
    # Bounds checking
    valid = (batch_idx < B) & (ai < F) & (aj < F) & (ai != aj)
    
    if not valid:
        return

    # Compute memory addresses
    emb_base = emb_ptr + batch_idx * F * D
    a_ptr = emb_base + ai * D
    b_ptr = emb_base + aj * D

    # Vectorized dot product computation
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    
    # Process in blocks for better memory coalescing
    for k in range(0, tl.cdiv(D, BLOCK)):
        offs = k * BLOCK + tl.arange(0, BLOCK)
        mask = offs < D
        
        # Load with proper masking
        a_vals = tl.load(a_ptr + offs, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + offs, mask=mask, other=0.0)
        
        # Accumulate dot product
        acc += a_vals * b_vals
    
    # Reduce to scalar
    result = tl.sum(acc, axis=0)
    
    # Store result
    out_idx = batch_idx * total_pairs + pair_idx
    tl.store(out_ptr + out_idx, result)

@triton.jit
def interaction_kernel_optimized(emb_ptr, out_ptr, B: tl.constexpr, F: tl.constexpr, D: tl.constexpr, 
                                 BLOCK_F: tl.constexpr, BLOCK_D: tl.constexpr):
    """
    Further optimized version with 2D tiling
    
    This version processes multiple feature pairs simultaneously
    for better GPU utilization
    """
    pid_batch = tl.program_id(axis=0)
    pid_pair = tl.program_id(axis=1)
    
    if pid_batch >= B:
        return
    
    # Process multiple pairs per thread block
    pair_start = pid_pair * BLOCK_F
    total_pairs = F * (F - 1) // 2
    
    for local_pair in range(BLOCK_F):
        pair_idx = pair_start + local_pair
        if pair_idx >= total_pairs:
            break
            
        # Decode pair indices
        ai = tl.int32(0)
        cumsum = tl.int32(0)
        
        for ii in range(F - 1):
            next_cumsum = cumsum + (F - ii - 1)
            cond = pair_idx >= next_cumsum
            ai = tl.where(cond, ii + 1, ai)
            cumsum = tl.where(cond, next_cumsum, cumsum)
        
        aj = ai + 1 + (pair_idx - cumsum)
        
        # Compute interaction
        emb_base = emb_ptr + pid_batch * F * D
        a_ptr = emb_base + ai * D
        b_ptr = emb_base + aj * D
        
        # Vectorized computation with larger blocks
        acc = 0.0
        for k in range(0, tl.cdiv(D, BLOCK_D)):
            offs = k * BLOCK_D + tl.arange(0, BLOCK_D)
            mask = offs < D
            
            a_vals = tl.load(a_ptr + offs, mask=mask, other=0.0)
            b_vals = tl.load(b_ptr + offs, mask=mask, other=0.0)
            
            acc += tl.sum(a_vals * b_vals)
        
        # Store result
        out_idx = pid_batch * total_pairs + pair_idx
        tl.store(out_ptr + out_idx, acc)

@triton.jit  
def interaction_kernel_fused(emb_ptr, out_ptr, stats_ptr, B: tl.constexpr, F: tl.constexpr, 
                            D: tl.constexpr, BLOCK: tl.constexpr):
    """
    Fused kernel that computes pairwise interactions and statistics
    
    Outputs:
        out_ptr: Pairwise interaction results [B, F*(F-1)/2]
        stats_ptr: Statistics [B, 4] (mean, max, min, std)
    """
    pid = tl.program_id(axis=0)
    batch_idx = pid
    
    if batch_idx >= B:
        return
    
    total_pairs = F * (F - 1) // 2
    emb_base = emb_ptr + batch_idx * F * D
    out_base = out_ptr + batch_idx * total_pairs
    
    # Initialize statistics accumulators
    sum_val = 0.0
    max_val = -1e9
    min_val = 1e9
    sum_sq = 0.0
    count = 0
    
    # Compute all pairs for this batch
    for ai in range(F - 1):
        for aj in range(ai + 1, F):
            # Compute pair index
            pair_idx = ai * (2 * F - ai - 1) // 2 + (aj - ai - 1)
            
            # Load embeddings
            a_ptr = emb_base + ai * D
            b_ptr = emb_base + aj * D
            
            # Compute dot product
            acc = 0.0
            for k in range(0, tl.cdiv(D, BLOCK)):
                offs = k * BLOCK + tl.arange(0, BLOCK)
                mask = offs < D
                
                a_vals = tl.load(a_ptr + offs, mask=mask, other=0.0)
                b_vals = tl.load(b_ptr + offs, mask=mask, other=0.0)
                
                acc += tl.sum(a_vals * b_vals)
            
            # Store result
            tl.store(out_base + pair_idx, acc)
            
            # Update statistics
            sum_val += acc
            max_val = tl.maximum(max_val, acc)
            min_val = tl.minimum(min_val, acc)
            sum_sq += acc * acc
            count += 1
    
    # Compute and store statistics
    if count > 0:
        mean_val = sum_val / count
        variance = (sum_sq / count) - (mean_val * mean_val)
        std_val = tl.sqrt(tl.maximum(variance, 0.0))
        
        stats_base = stats_ptr + batch_idx * 4
        tl.store(stats_base + 0, mean_val)
        tl.store(stats_base + 1, max_val) 
        tl.store(stats_base + 2, min_val)
        tl.store(stats_base + 3, std_val)
