import torch
from interaction_triton_fast import interaction_kernel

def interaction_op(emb: torch.Tensor, BLOCK: int = 64):
    assert emb.is_cuda, "emb must be CUDA tensor"
    B, F, D = emb.shape
    out_pairs = F*(F-1)//2
    out = torch.empty((B, out_pairs), device='cuda', dtype=torch.float32)
    grid = (B * out_pairs,)
    interaction_kernel[grid](emb.data_ptr(), out.data_ptr(), B, F, D, BLOCK, num_warps=4)
    return out
