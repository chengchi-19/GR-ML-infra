import torch
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any, Union

logger = logging.getLogger(__name__)

try:
    from interaction_triton_fast import (
        interaction_kernel, 
        interaction_kernel_optimized, 
        interaction_kernel_fused
    )
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    logger.warning("Triton不可用，将使用CPU实现")

class InteractionOperator:
    """
    优化的pairwise交互算子
    支持GPU加速(Triton)和CPU回退
    """
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_triton = TRITON_AVAILABLE and self.device == 'cuda'
        self.block_size = 64
        self.stats = {'gpu_calls': 0, 'cpu_calls': 0, 'cache_hits': 0}
        
        # 结果缓存(用于相同输入)
        self._cache = {}
        self._max_cache_size = 100
        
        logger.info(f"InteractionOperator initialized: device={self.device}, triton={self.use_triton}")
    
    def __call__(self, emb: torch.Tensor, BLOCK: int = 64, 
                 mode: str = 'basic', return_stats: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        计算pairwise交互
        
        Args:
            emb: 输入嵌入 [B, F, D]
            BLOCK: 块大小
            mode: 计算模式 ('basic', 'optimized', 'fused')
            return_stats: 是否返回统计信息
            
        Returns:
            交互结果 [B, F*(F-1)/2] 或 (结果, 统计信息)
        """
        if not isinstance(emb, torch.Tensor):
            raise TypeError(f"输入必须是torch.Tensor，得到{type(emb)}")
        
        if emb.dim() != 3:
            raise ValueError(f"输入维度必须是3 [B,F,D]，得到{emb.dim()}维")
        
        B, F, D = emb.shape
        
        if F < 2:
            raise ValueError(f"特征数量必须至少为2，得到{F}")
        
        # 检查缓存
        cache_key = self._get_cache_key(emb, BLOCK, mode)
        if cache_key in self._cache:
            self.stats['cache_hits'] += 1
            result = self._cache[cache_key]
            if return_stats:
                return result, self._compute_stats(result)
            return result
        
        # 执行计算
        if self.use_triton:
            result = self._triton_compute(emb, BLOCK, mode)
            self.stats['gpu_calls'] += 1
        else:
            result = self._cpu_compute(emb)
            self.stats['cpu_calls'] += 1
        
        # 更新缓存
        self._update_cache(cache_key, result)
        
        if return_stats:
            stats = self._compute_stats(result)
            return result, stats
        
        return result
    
    def _triton_compute(self, emb: torch.Tensor, BLOCK: int, mode: str) -> torch.Tensor:
        """使用Triton进行GPU计算"""
        B, F, D = emb.shape
        out_pairs = F * (F - 1) // 2
        
        # 确保在GPU上
        if not emb.is_cuda:
            emb = emb.cuda()
        
        out = torch.empty((B, out_pairs), device='cuda', dtype=torch.float32)
        
        try:
            if mode == 'basic':
                grid = (B * out_pairs,)
                interaction_kernel[grid](
                    emb.data_ptr(), out.data_ptr(), 
                    B, F, D, BLOCK, 
                    num_warps=4
                )
            elif mode == 'optimized':
                BLOCK_F = min(32, out_pairs)
                BLOCK_D = min(128, D)
                grid = (B, (out_pairs + BLOCK_F - 1) // BLOCK_F)
                interaction_kernel_optimized[grid](
                    emb.data_ptr(), out.data_ptr(),
                    B, F, D, BLOCK_F, BLOCK_D,
                    num_warps=8
                )
            elif mode == 'fused':
                stats = torch.empty((B, 4), device='cuda', dtype=torch.float32)
                grid = (B,)
                interaction_kernel_fused[grid](
                    emb.data_ptr(), out.data_ptr(), stats.data_ptr(),
                    B, F, D, BLOCK,
                    num_warps=4
                )
            else:
                raise ValueError(f"未知模式: {mode}")
                
        except Exception as e:
            logger.warning(f"Triton计算失败: {e}，回退到CPU")
            return self._cpu_compute(emb)
        
        return out
    
    def _cpu_compute(self, emb: torch.Tensor) -> torch.Tensor:
        """CPU计算实现"""
        B, F, D = emb.shape
        out_pairs = F * (F - 1) // 2
        
        # 转到CPU
        emb_cpu = emb.cpu() if emb.is_cuda else emb
        
        result = torch.zeros((B, out_pairs), dtype=torch.float32)
        
        pair_idx = 0
        for i in range(F):
            for j in range(i + 1, F):
                # 计算点积
                dot_product = torch.sum(emb_cpu[:, i, :] * emb_cpu[:, j, :], dim=1)
                result[:, pair_idx] = dot_product
                pair_idx += 1
        
        # 如果原始输入在GPU上，将结果也放到GPU
        if emb.is_cuda:
            result = result.cuda()
        
        return result
    
    def _get_cache_key(self, emb: torch.Tensor, BLOCK: int, mode: str) -> str:
        """生成缓存键"""
        # 简化的缓存键生成
        shape_key = f"{emb.shape[0]}_{emb.shape[1]}_{emb.shape[2]}"
        data_key = hash(emb.data_ptr()) % 10000  # 简化的数据键
        return f"{shape_key}_{BLOCK}_{mode}_{data_key}"
    
    def _update_cache(self, key: str, result: torch.Tensor):
        """更新缓存"""
        if len(self._cache) >= self._max_cache_size:
            # 移除最旧的条目
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = result.clone()
    
    def _compute_stats(self, result: torch.Tensor) -> Dict[str, Any]:
        """计算统计信息"""
        return {
            'mean': float(result.mean().item()),
            'std': float(result.std().item()),
            'min': float(result.min().item()),
            'max': float(result.max().item()),
            'shape': list(result.shape)
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.stats.copy()
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("缓存已清空")
    
    def benchmark(self, shapes: list, num_runs: int = 10) -> Dict[str, list]:
        """性能基准测试"""
        results = {'shapes': [], 'triton_times': [], 'cpu_times': []}
        
        for B, F, D in shapes:
            emb = torch.randn(B, F, D, dtype=torch.float32)
            
            # GPU测试
            if self.use_triton:
                emb_gpu = emb.cuda()
                torch.cuda.synchronize()
                
                import time
                start = time.time()
                for _ in range(num_runs):
                    _ = self._triton_compute(emb_gpu, 64, 'basic')
                torch.cuda.synchronize()
                triton_time = (time.time() - start) / num_runs
            else:
                triton_time = float('inf')
            
            # CPU测试
            import time
            start = time.time()
            for _ in range(num_runs):
                _ = self._cpu_compute(emb)
            cpu_time = (time.time() - start) / num_runs
            
            results['shapes'].append(f"{B}x{F}x{D}")
            results['triton_times'].append(triton_time * 1000)  # 转换为毫秒
            results['cpu_times'].append(cpu_time * 1000)
        
        return results

# 全局实例
_global_operator = None

def interaction_op(emb: torch.Tensor, BLOCK: int = 64, **kwargs) -> torch.Tensor:
    """
    便捷函数：计算pairwise交互
    
    Args:
        emb: 输入嵌入 [B, F, D]
        BLOCK: 块大小
        **kwargs: 其他参数
        
    Returns:
        交互结果 [B, F*(F-1)/2]
    """
    global _global_operator
    if _global_operator is None:
        _global_operator = InteractionOperator()
    
    return _global_operator(emb, BLOCK, **kwargs)

def get_operator_stats() -> Dict[str, Any]:
    """获取全局算子的统计信息"""
    global _global_operator
    if _global_operator is None:
        return {}
    return _global_operator.get_performance_stats()

if __name__ == "__main__":
    # 测试代码
    print("测试Interaction算子...")
    
    # 创建测试数据
    B, F, D = 2, 8, 32
    emb = torch.randn(B, F, D, dtype=torch.float32)
    
    if torch.cuda.is_available():
        emb_cuda = emb.cuda()
    
    # 创建算子
    op = InteractionOperator()
    
    # 测试基本功能
    result = op(emb, mode='basic')
    print(f"基本模式结果形状: {result.shape}")
    
    # 测试统计信息
    result, stats = op(emb, mode='basic', return_stats=True)
    print(f"统计信息: {stats}")
    
    # 性能测试
    benchmark_results = op.benchmark([(1, 16, 64), (2, 32, 128)], num_runs=5)
    print(f"基准测试结果: {benchmark_results}")
    
    # 性能统计
    perf_stats = op.get_performance_stats()
    print(f"性能统计: {perf_stats}")
