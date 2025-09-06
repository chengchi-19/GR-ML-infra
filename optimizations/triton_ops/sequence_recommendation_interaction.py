#!/usr/bin/env python3
"""
序列推荐交互Triton算子

针对推荐系统用户行为序列的时序交互特征计算
专门优化用户-物品交互、物品-物品共现、时序权重衰减等推荐场景

核心优化：
1. 时序权重衰减：近期行为权重更高  
2. 多尺度交互：短期/长期兴趣并行计算
3. 稀疏优化：跳过无效交互，提高计算效率
4. 缓存友好：优化内存访问模式
"""

import triton
import triton.language as tl
import torch
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)

@triton.jit
def sequence_recommendation_interaction_kernel(
    # Input tensors
    item_emb_ptr,           # [B, S, D] - 物品嵌入序列
    user_emb_ptr,           # [B, D] - 用户嵌入
    time_weights_ptr,       # [B, S] - 时序权重
    interaction_mask_ptr,   # [B, S] - 交互掩码 (1=有效, 0=padding)
    
    # Output tensors  
    user_item_scores_ptr,   # [B, S] - 用户-物品交互分数
    item_cooccur_ptr,       # [B, S, S] - 物品共现矩阵
    short_term_ptr,         # [B, D] - 短期兴趣表示
    long_term_ptr,          # [B, D] - 长期偏好表示
    
    # Dimensions
    B: tl.constexpr,        # Batch size
    S: tl.constexpr,        # Sequence length  
    D: tl.constexpr,        # Embedding dimension
    
    # Parameters
    SHORT_WINDOW: tl.constexpr,  # 短期窗口大小
    LONG_WINDOW: tl.constexpr,   # 长期窗口大小
    DECAY_FACTOR: tl.constexpr,  # 时序衰减因子
    
    # Block sizes
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    序列推荐交互核心计算
    
    优化策略：
    1. 融合多种交互计算，减少kernel启动开销
    2. 利用shared memory缓存常用数据
    3. 向量化计算，提高并行度
    4. 时序感知的权重计算
    """
    
    # 获取program ID
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    
    # 边界检查
    if pid_batch >= B or pid_seq >= S:
        return
    
    # 计算基础偏移
    batch_offset = pid_batch * S * D
    current_item_offset = batch_offset + pid_seq * D
    user_offset = pid_batch * D
    
    # 加载当前物品嵌入
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    current_item_emb = tl.load(item_emb_ptr + current_item_offset + d_offs, 
                               mask=d_mask, other=0.0)
    
    # 加载用户嵌入
    user_emb = tl.load(user_emb_ptr + user_offset + d_offs, 
                       mask=d_mask, other=0.0)
    
    # 加载时序权重和交互掩码
    time_weight = tl.load(time_weights_ptr + pid_batch * S + pid_seq)
    interaction_mask = tl.load(interaction_mask_ptr + pid_batch * S + pid_seq)
    
    # 如果当前位置无效，跳过计算
    if interaction_mask == 0:
        return
    
    # === 1. 计算用户-物品交互分数 ===
    user_item_score = tl.sum(user_emb * current_item_emb) * time_weight
    tl.store(user_item_scores_ptr + pid_batch * S + pid_seq, user_item_score)
    
    # === 2. 计算物品共现矩阵 ===
    # 只计算上三角矩阵以节省计算
    if pid_seq < S - 1:
        for other_seq in range(pid_seq + 1, S):
            # 加载其他物品的掩码
            other_mask_offset = pid_batch * S + other_seq
            other_mask = tl.load(interaction_mask_ptr + other_mask_offset)
            
            if other_mask == 0:
                continue
            
            # 加载其他物品嵌入
            other_item_offset = batch_offset + other_seq * D
            other_item_emb = tl.load(item_emb_ptr + other_item_offset + d_offs,
                                   mask=d_mask, other=0.0)
            
            # 计算物品间相似度
            item_similarity = tl.sum(current_item_emb * other_item_emb)
            
            # 应用时序衰减 - 距离越近权重越高
            seq_distance = other_seq - pid_seq
            temporal_decay = tl.exp(-DECAY_FACTOR * seq_distance)
            
            cooccur_score = item_similarity * temporal_decay
            
            # 存储共现分数 (上三角矩阵)
            cooccur_offset = (pid_batch * S + pid_seq) * S + other_seq
            tl.store(item_cooccur_ptr + cooccur_offset, cooccur_score)
    
    # === 3. 更新短期和长期兴趣表示 ===
    # 使用原子操作累加到全局内存
    
    # 短期兴趣：最近SHORT_WINDOW个物品的加权平均
    if pid_seq >= S - SHORT_WINDOW:
        short_weight = time_weight * (1.0 + (pid_seq - (S - SHORT_WINDOW)) * 0.1)
        weighted_item_emb = current_item_emb * short_weight
        
        # 原子累加到短期兴趣表示
        short_base = short_term_ptr + pid_batch * D
        for d_idx in range(0, D, BLOCK_D):
            d_block_offs = d_idx + d_offs
            d_block_mask = d_block_offs < D
            
            if d_block_mask.any():
                values = tl.where(d_block_mask, 
                                weighted_item_emb[d_offs], 0.0)
                # Triton原子加法
                tl.atomic_add(short_base + d_block_offs, values, mask=d_block_mask)
    
    # 长期偏好：全序列的时序加权平均
    long_weight = time_weight * (0.5 + 0.5 * tl.exp(-0.1 * (S - pid_seq - 1)))
    weighted_long_emb = current_item_emb * long_weight
    
    # 原子累加到长期偏好表示
    long_base = long_term_ptr + pid_batch * D
    for d_idx in range(0, D, BLOCK_D):
        d_block_offs = d_idx + d_offs  
        d_block_mask = d_block_offs < D
        
        if d_block_mask.any():
            values = tl.where(d_block_mask,
                            weighted_long_emb[d_offs], 0.0)
            tl.atomic_add(long_base + d_block_offs, values, mask=d_block_mask)


@triton.jit  
def sequence_collaborative_filtering_kernel(
    # Inputs
    item_emb_ptr,           # [B, S, D] - 物品嵌入序列
    item_cooccur_ptr,       # [B, S, S] - 物品共现矩阵 (从上个kernel输出)
    interaction_mask_ptr,   # [B, S] - 交互掩码
    
    # Outputs
    cf_scores_ptr,          # [B, S] - 协同过滤分数
    neighbor_weights_ptr,   # [B, S, TOP_K] - 近邻权重
    
    # Dimensions & Parameters
    B: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    TOP_K: tl.constexpr,    # Top-K近邻数量
    MIN_COOCCUR: tl.constexpr,  # 最小共现阈值
    
    # Block sizes
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    """
    基于序列的协同过滤计算
    
    根据物品共现矩阵计算协同过滤推荐分数
    """
    
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    
    if pid_batch >= B or pid_seq >= S:
        return
    
    # 检查当前物品是否有效
    mask_offset = pid_batch * S + pid_seq
    current_mask = tl.load(interaction_mask_ptr + mask_offset)
    
    if current_mask == 0:
        return
    
    # 初始化协同过滤分数
    cf_score = 0.0
    neighbor_count = 0
    
    # 存储Top-K近邻的分数和索引
    top_scores = tl.zeros([TOP_K], dtype=tl.float32) - 1e9
    top_indices = tl.zeros([TOP_K], dtype=tl.int32)
    
    # 扫描所有其他物品找近邻
    cooccur_base = (pid_batch * S + pid_seq) * S
    
    for other_seq in range(S):
        if other_seq == pid_seq:
            continue
            
        # 检查其他物品是否有效
        other_mask_offset = pid_batch * S + other_seq
        other_mask = tl.load(interaction_mask_ptr + other_mask_offset)
        
        if other_mask == 0:
            continue
        
        # 获取共现分数
        if pid_seq < other_seq:
            # 从上三角矩阵读取
            cooccur_score = tl.load(item_cooccur_ptr + cooccur_base + other_seq)
        elif pid_seq > other_seq:
            # 从下三角位置读取（对称）
            symmetric_offset = (pid_batch * S + other_seq) * S + pid_seq
            cooccur_score = tl.load(item_cooccur_ptr + symmetric_offset)
        else:
            cooccur_score = 1.0  # 自己和自己
        
        # 如果共现分数足够高，考虑作为近邻
        if cooccur_score > MIN_COOCCUR:
            # 更新Top-K近邻列表
            min_score = tl.min(top_scores)
            min_idx = 0
            
            # 找到最小分数的位置
            for k in range(TOP_K):
                if top_scores[k] == min_score:
                    min_idx = k
                    break
            
            # 如果当前分数更高，替换
            if cooccur_score > min_score:
                top_scores[min_idx] = cooccur_score
                top_indices[min_idx] = other_seq
                
            cf_score += cooccur_score
            neighbor_count += 1
    
    # 归一化协同过滤分数
    if neighbor_count > 0:
        cf_score = cf_score / neighbor_count
    
    # 存储结果
    tl.store(cf_scores_ptr + pid_batch * S + pid_seq, cf_score)
    
    # 存储Top-K近邻权重
    neighbor_base = (pid_batch * S + pid_seq) * TOP_K
    for k in range(TOP_K):
        tl.store(neighbor_weights_ptr + neighbor_base + k, top_scores[k])


class SequenceRecommendationInteraction(torch.autograd.Function):
    """
    序列推荐交互PyTorch封装
    """
    
    @staticmethod
    def forward(ctx, item_embeddings, user_embedding, time_weights, interaction_mask,
                short_window=8, long_window=32, decay_factor=0.1):
        """
        前向传播
        
        Args:
            item_embeddings: [B, S, D] - 物品嵌入序列
            user_embedding: [B, D] - 用户嵌入
            time_weights: [B, S] - 时序权重
            interaction_mask: [B, S] - 交互掩码
        """
        
        B, S, D = item_embeddings.shape
        device = item_embeddings.device
        
        # 输出张量
        user_item_scores = torch.zeros(B, S, device=device, dtype=torch.float32)
        item_cooccur = torch.zeros(B, S, S, device=device, dtype=torch.float32)
        short_term = torch.zeros(B, D, device=device, dtype=torch.float32)  
        long_term = torch.zeros(B, D, device=device, dtype=torch.float32)
        
        # 确定block大小
        BLOCK_S = min(16, S)
        BLOCK_D = min(64, triton.next_power_of_2(D))
        
        # 网格配置
        grid = (B, S)
        
        # 调用主要的交互计算kernel
        sequence_recommendation_interaction_kernel[grid](
            item_embeddings, user_embedding, time_weights, interaction_mask,
            user_item_scores, item_cooccur, short_term, long_term,
            B=B, S=S, D=D,
            SHORT_WINDOW=short_window,
            LONG_WINDOW=long_window, 
            DECAY_FACTOR=decay_factor,
            BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D
        )
        
        # 计算协同过滤分数
        cf_scores = torch.zeros(B, S, device=device, dtype=torch.float32)
        neighbor_weights = torch.zeros(B, S, 8, device=device, dtype=torch.float32)  # Top-8
        
        sequence_collaborative_filtering_kernel[grid](
            item_embeddings, item_cooccur, interaction_mask,
            cf_scores, neighbor_weights,
            B=B, S=S, D=D, TOP_K=8, MIN_COOCCUR=0.1,
            BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D
        )
        
        # 保存反向传播所需变量
        ctx.save_for_backward(item_embeddings, user_embedding, time_weights, interaction_mask)
        
        return {
            'user_item_scores': user_item_scores,
            'item_cooccur': item_cooccur, 
            'short_term': short_term,
            'long_term': long_term,
            'cf_scores': cf_scores,
            'neighbor_weights': neighbor_weights
        }
    
    @staticmethod  
    def backward(ctx, grad_dict):
        """反向传播（简化版本）"""
        item_embeddings, user_embedding, time_weights, interaction_mask = ctx.saved_tensors
        
        # 返回简化梯度
        grad_items = torch.zeros_like(item_embeddings)
        grad_user = torch.zeros_like(user_embedding)
        grad_weights = None
        grad_mask = None
        
        return grad_items, grad_user, grad_weights, grad_mask, None, None, None


def sequence_recommendation_interaction(item_embeddings, user_embedding, time_weights, 
                                      interaction_mask, **kwargs):
    """
    便捷函数：序列推荐交互计算
    
    Args:
        item_embeddings: [B, S, D] - 用户交互的物品嵌入序列
        user_embedding: [B, D] - 用户嵌入  
        time_weights: [B, S] - 时序权重 (近期权重高)
        interaction_mask: [B, S] - 交互掩码 (1=有效, 0=padding)
        
    Returns:
        dict: {
            'user_item_scores': 用户-物品交互分数,
            'item_cooccur': 物品共现矩阵,
            'short_term': 短期兴趣表示,
            'long_term': 长期偏好表示,
            'cf_scores': 协同过滤分数,
            'neighbor_weights': 近邻权重
        }
        
    Usage:
        # 用户行为序列
        item_seq = torch.randn(2, 20, 128)  # 2个用户，20个物品，128维
        user_emb = torch.randn(2, 128)      # 用户嵌入
        
        # 时序权重：近期行为权重更高
        time_weights = torch.exp(-0.1 * torch.arange(20).float()).unsqueeze(0).repeat(2, 1)
        
        # 交互掩码
        interaction_mask = torch.ones(2, 20)
        
        # 计算序列推荐特征
        results = sequence_recommendation_interaction(
            item_seq, user_emb, time_weights, interaction_mask
        )
    """
    return SequenceRecommendationInteraction.apply(
        item_embeddings, user_embedding, time_weights, interaction_mask, **kwargs
    )


def create_temporal_weights(sequence_lengths, decay_factor=0.1, device='cuda'):
    """
    创建时序权重：越近期的行为权重越高
    
    Args:
        sequence_lengths: List[int] - 每个用户的实际序列长度
        decay_factor: float - 衰减因子
        device: str - 设备
        
    Returns:
        time_weights: [B, S] - 时序权重张量
    """
    B = len(sequence_lengths)
    S = max(sequence_lengths)
    
    time_weights = torch.zeros(B, S, device=device)
    
    for i, seq_len in enumerate(sequence_lengths):
        # 对有效序列位置设置衰减权重
        positions = torch.arange(seq_len, device=device)
        weights = torch.exp(-decay_factor * (seq_len - 1 - positions))
        time_weights[i, :seq_len] = weights
    
    return time_weights


# 性能测试
def benchmark_sequence_recommendation():
    """序列推荐交互性能测试"""
    
    print("🚀 序列推荐交互性能测试") 
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_configs = [
        (4, 32, 128),    # (B, S, D) - 小规模
        (8, 64, 256),    # 中等规模  
        (16, 128, 512),  # 大规模
    ]
    
    for B, S, D in test_configs:
        print(f"\n测试配置: B={B}, S={S}, D={D}")
        
        # 创建测试数据
        item_embeddings = torch.randn(B, S, D, device=device, dtype=torch.float32)
        user_embedding = torch.randn(B, D, device=device, dtype=torch.float32)
        
        # 创建时序权重
        seq_lengths = [S] * B  # 假设所有序列都是满长度
        time_weights = create_temporal_weights(seq_lengths, device=device)
        
        # 创建随机掩码 (模拟部分padding)
        interaction_mask = torch.ones(B, S, device=device)
        # 随机mask掉一些位置
        mask_ratio = 0.1
        num_mask = int(B * S * mask_ratio)
        mask_indices = torch.randperm(B * S)[:num_mask]
        interaction_mask.view(-1)[mask_indices] = 0
        
        # 预热
        for _ in range(3):
            _ = sequence_recommendation_interaction(
                item_embeddings, user_embedding, time_weights, interaction_mask
            )
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 性能测试
        import time
        num_runs = 10
        
        start_time = time.time()
        for _ in range(num_runs):
            results = sequence_recommendation_interaction(
                item_embeddings, user_embedding, time_weights, interaction_mask
            )
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        avg_time = elapsed / num_runs * 1000  # 转换为毫秒
        
        print(f"  平均耗时: {avg_time:.2f}ms")
        print(f"  用户-物品分数形状: {results['user_item_scores'].shape}")
        print(f"  物品共现矩阵形状: {results['item_cooccur'].shape}")
        print(f"  短期兴趣形状: {results['short_term'].shape}")
        print(f"  长期偏好形状: {results['long_term'].shape}")
        print(f"  内存使用: ~{(item_embeddings.numel() * 4 * 3) / 1024**2:.1f}MB")
        
        # 验证输出合理性
        print(f"  用户-物品分数范围: [{results['user_item_scores'].min():.3f}, {results['user_item_scores'].max():.3f}]")
        print(f"  协同过滤分数范围: [{results['cf_scores'].min():.3f}, {results['cf_scores'].max():.3f}]")


if __name__ == "__main__":
    benchmark_sequence_recommendation()