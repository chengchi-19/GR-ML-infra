# Triton kernel prototype for pairwise interaction (toy)
import triton
import triton.language as tl

@triton.jit
def interaction_kernel(emb_ptr, out_ptr, B: tl.constexpr, F: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    batch_idx = pid // (F*(F-1)//2)
    pair_idx = pid % (F*(F-1)//2)
    # naive triangular decode (for demo only)
    a = 0; s = 0
    for i in range(F):
        for j in range(i+1, F):
            if s == pair_idx:
                ai, aj = i, j; break
            s += 1
        if s > pair_idx:
            break
    emb_base = emb_ptr + batch_idx * F * D
    a_ptr = emb_base + ai * D
    b_ptr = emb_base + aj * D
    acc = tl.zeros([1], dtype=tl.float32)
    for k in range(0, D, BLOCK):
        a = tl.load(a_ptr + tl.arange(0, BLOCK))
        b = tl.load(b_ptr + tl.arange(0, BLOCK))
        acc += tl.sum(a * b)
    tl.store(out_ptr + batch_idx * (F*(F-1)//2) + pair_idx, acc)
