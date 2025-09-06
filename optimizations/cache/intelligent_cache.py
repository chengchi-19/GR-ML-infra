#!/usr/bin/env python3
"""
智能缓存模块

基于embedding_service_v2.py的核心功能，
提供GPU热缓存和智能预测功能。
"""

import numpy as np
import torch
from collections import OrderedDict, defaultdict
from threading import Lock, RLock
import time
import logging
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from queue import Queue

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """缓存统计信息"""
    gpu_hits: int = 0
    host_hits: int = 0
    misses: int = 0
    total_requests: int = 0
    gpu_memory_usage: float = 0.0
    host_memory_usage: float = 0.0
    avg_access_time: float = 0.0
    hit_rate: float = 0.0
    
    def update_hit_rate(self):
        if self.total_requests > 0:
            self.hit_rate = (self.gpu_hits + self.host_hits) / self.total_requests
        
    def to_dict(self) -> Dict:
        return asdict(self)

class HotSpotPredictor:
    """热点预测器"""
    
    def __init__(self, window_size: int = 1000, decay_factor: float = 0.95):
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.access_history = Queue(maxsize=window_size)
        self.item_scores = defaultdict(float)
        self.category_scores = defaultdict(float)
        self.temporal_patterns = defaultdict(list)
        self.lock = RLock()
        
    def record_access(self, item_id: int, category: Optional[str] = None, timestamp: float = None):
        """记录访问"""
        if timestamp is None:
            timestamp = time.time()
            
        with self.lock:
            # 记录访问历史
            if self.access_history.full():
                old_item, old_time, old_cat = self.access_history.get()
                self.item_scores[old_item] *= self.decay_factor
            
            self.access_history.put((item_id, timestamp, category))
            
            # 更新分数
            self.item_scores[item_id] += 1.0
            if category:
                self.category_scores[category] += 1.0
                
            # 记录时间模式
            hour = int(timestamp // 3600) % 24
            self.temporal_patterns[hour].append(item_id)
            if len(self.temporal_patterns[hour]) > 100:
                self.temporal_patterns[hour] = self.temporal_patterns[hour][-100:]
    
    def predict_hot_items(self, n: int = 100) -> List[Tuple[int, float]]:
        """预测热点项目"""
        with self.lock:
            # 基于历史访问频率
            frequency_scores = dict(self.item_scores)
            
            # 时间衰减
            current_time = time.time()
            for item_id in frequency_scores:
                # 根据最近访问时间调整分数
                time_decay = np.exp(-(current_time % 86400) / 86400)  # 按天衰减
                frequency_scores[item_id] *= time_decay
            
            # 时间模式预测
            current_hour = int(time.time() // 3600) % 24
            temporal_items = self.temporal_patterns.get(current_hour, [])
            for item_id in temporal_items:
                frequency_scores[item_id] = frequency_scores.get(item_id, 0) + 0.5
            
            # 排序并返回top-n
            sorted_items = sorted(frequency_scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_items[:n]


class IntelligentEmbeddingCache:
    """智能嵌入缓存"""
    
    def __init__(self, cache_size: int, embedding_dim: int, dtype: torch.dtype = torch.float32, 
                 device: str = None, enable_prediction: bool = True):
        self.cache_size = cache_size
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        self.enable_prediction = enable_prediction
        
        # 设备自动选择
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # 缓存数据结构
        self.cache = OrderedDict()  # item_id -> slot_index
        self.metadata = {}  # item_id -> metadata
        
        # GPU缓存缓冲区
        self.gpu_buffer = torch.zeros((cache_size, embedding_dim), device=self.device, dtype=dtype)
        self.free_slots = list(range(cache_size))
        self.slot_to_item = {}  # slot_index -> item_id
        
        # 线程安全
        self.lock = RLock()
        
        # 预测器
        if enable_prediction:
            self.predictor = HotSpotPredictor()
        else:
            self.predictor = None
        
        # 统计信息
        self.stats = CacheStats()
        
        logger.info(f"IntelligentEmbeddingCache initialized: size={cache_size}, dim={embedding_dim}, device={device}")
    
    def get(self, idx: int, category: str = None) -> Optional[int]:
        """获取嵌入的slot索引"""
        with self.lock:
            current_time = time.time()
            
            # 记录访问用于预测
            if self.predictor:
                self.predictor.record_access(idx, category, current_time)
            
            # 检查缓存
            if idx in self.cache:
                slot = self.cache[idx]
                # 移到LRU末尾
                self.cache.move_to_end(idx)
                self.stats.gpu_hits += 1
                return slot
            
            return None
    
    def put(self, idx: int, embedding: torch.Tensor, category: str = None, 
           force_cache: bool = False) -> int:
        """放入嵌入到缓存"""
        with self.lock:
            current_time = time.time()
            
            # 如果已存在，更新并移到末尾
            if idx in self.cache:
                slot = self.cache[idx]
                self.cache.move_to_end(idx)
                self._update_buffer(slot, embedding)
                return slot
            
            # 如果没有空闲slot，需要驱逐
            if not self.free_slots:
                evicted_slot = self._evict_item(force_cache)
                if evicted_slot is None:
                    logger.warning("无法驱逐缓存项，缓存可能已满")
                    return -1
            else:
                evicted_slot = self.free_slots.pop(0)
            
            # 添加新项
            self._update_buffer(evicted_slot, embedding)
            self.cache[idx] = evicted_slot
            self.cache.move_to_end(idx)
            self.slot_to_item[evicted_slot] = idx
            
            return evicted_slot
    
    def _evict_item(self, force_cache: bool = False) -> Optional[int]:
        """智能驱逐策略"""
        if not self.cache:
            return None
        
        # 使用LRU策略
        evicted_item, slot = self.cache.popitem(last=False)
        
        # 清理元数据
        if evicted_item in self.metadata:
            del self.metadata[evicted_item]
        if slot in self.slot_to_item:
            del self.slot_to_item[slot]
        
        return slot
    
    def _update_buffer(self, slot: int, embedding: torch.Tensor):
        """更新GPU缓冲区"""
        target = embedding.to(self.device, dtype=self.dtype)
        if target.shape[0] != self.embedding_dim:
            # 如果维度不匹配，尝试reshape或截断
            if target.numel() == self.embedding_dim:
                target = target.view(self.embedding_dim)
            elif target.numel() > self.embedding_dim:
                target = target[:self.embedding_dim]
            else:
                # 填充零
                padded = torch.zeros(self.embedding_dim, device=self.device, dtype=self.dtype)
                padded[:target.numel()] = target.view(-1)
                target = padded
        
        self.gpu_buffer[slot].copy_(target)
    
    def get_embedding(self, slot: int) -> torch.Tensor:
        """获取slot对应的嵌入"""
        return self.gpu_buffer[slot]
    
    def preload_hot_items(self, hot_items: List[Tuple[int, torch.Tensor]], 
                         categories: List[str] = None) -> int:
        """预加载热点项目"""
        loaded_count = 0
        categories = categories or [None] * len(hot_items)
        
        with self.lock:
            for i, (item_id, embedding) in enumerate(hot_items):
                category = categories[i] if i < len(categories) else None
                
                if item_id not in self.cache and self.free_slots:
                    self.put(item_id, embedding, category, force_cache=True)
                    loaded_count += 1
        
        logger.info(f"预加载了 {loaded_count} 个热点项目")
        return loaded_count
    
    def get_cache_info(self) -> Dict:
        """获取缓存详细信息"""
        with self.lock:
            return {
                'cache_size': self.cache_size,
                'used_slots': len(self.cache),
                'free_slots': len(self.free_slots),
                'memory_usage_mb': self.gpu_buffer.numel() * self.gpu_buffer.element_size() / 1024 / 1024,
                'top_items': list(self.cache.keys())[-10:] if len(self.cache) >= 10 else list(self.cache.keys())
            }
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.metadata.clear()
            self.slot_to_item.clear()
            self.free_slots = list(range(self.cache_size))
            if self.predictor:
                self.predictor = HotSpotPredictor()


if __name__ == "__main__":
    # 测试智能缓存
    cache = IntelligentEmbeddingCache(
        cache_size=100,
        embedding_dim=512,
        enable_prediction=True
    )
    
    # 测试缓存操作
    for i in range(50):
        embedding = torch.randn(512)
        slot = cache.put(i, embedding)
        retrieved_slot = cache.get(i)
        
        if slot != retrieved_slot:
            print(f"缓存测试失败: {i}")
        
        if i % 10 == 0:
            print(f"已测试 {i+1} 个嵌入")
    
    # 获取缓存信息
    info = cache.get_cache_info()
    print(f"缓存信息: {info}")
    
    # 测试热点预测
    if cache.predictor:
        hot_items = cache.predictor.predict_hot_items(10)
        print(f"预测的热点项目: {hot_items[:5]}")
    
    print("智能缓存测试完成")