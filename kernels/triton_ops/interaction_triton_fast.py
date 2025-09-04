# Triton kernel prototype for pairwise interaction (toy)
import triton
import triton.language as tl

@triton.jit
def interaction_kernel(emb_ptr, out_ptr, B: tl.constexpr, F: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    total_pairs = F * (F - 1) // 2
    batch_idx = pid // total_pairs
    pair_idx = pid % total_pairs

    # 解析上三角索引 (ai, aj) 满足 0 <= ai < aj < F
    # 使用封闭形式而非 Python 循环
    # 计算所在的行 ai
    # 累积数列 S(ai) = ai*(2F-ai-1)/2 <= pair_idx < S(ai+1)
    ai = 0
    s = 0
    # 小规模F时用有限循环仍可接受，但避免嵌套与break
    for ii in range(0, F):
        next_s = s + (F - ii - 1)
        cond = pair_idx >= next_s
        ai = tl.where(cond, ii + 1, ai)
        s = tl.where(cond, next_s, s)
    aj = ai + 1 + (pair_idx - s)

    emb_base = emb_ptr + batch_idx * F * D
    a_ptr = emb_base + ai * D
    b_ptr = emb_base + aj * D

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    offs = tl.arange(0, BLOCK)
    for k in range(0, D, BLOCK):
        idx = k + offs
        mask = idx < D
        a = tl.load(a_ptr + idx, mask=mask, other=0.0)
        b = tl.load(b_ptr + idx, mask=mask, other=0.0)
        acc += a * b
    val = tl.sum(acc, axis=0)
    tl.store(out_ptr + batch_idx * total_pairs + pair_idx, val)
