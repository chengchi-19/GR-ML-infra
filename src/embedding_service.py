import numpy as np
import torch
from collections import OrderedDict
from threading import Lock
import time
import logging
from typing import List, Dict, Optional, Tuple
import pickle
import os

class EmbeddingCache:
    """GPU and host memory cache for embeddings"""
    
    def __init__(self, cache_size: int, embedding_dim: int, dtype: torch.dtype = torch.float32):
        self.cache_size = cache_size
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        self.cache = OrderedDict()
        self.gpu_buffer = torch.zeros((cache_size, embedding_dim), device='cuda', dtype=dtype)
        self.free_slots = list(range(cache_size))
        self.lock = Lock()
        
    def get(self, idx: int) -> Optional[int]:
        """Get slot index for embedding, return None if not cached"""
        with self.lock:
            return self.cache.get(idx)
    
    def put(self, idx: int, embedding: torch.Tensor) -> int:
        """Put embedding in cache, return slot index"""
        with self.lock:
            if idx in self.cache:
                return self.cache[idx]
            
            if not self.free_slots:
                # Evict least recently used
                evicted_idx = next(iter(self.cache))
                slot = self.cache.pop(evicted_idx)
            else:
                slot = self.free_slots.pop(0)
            
            # Copy embedding to GPU buffer
            self.gpu_buffer[slot].copy_(embedding.to('cuda', dtype=self.dtype))
            self.cache[idx] = slot
            return slot
    
    def get_embedding(self, slot: int) -> torch.Tensor:
        """Get embedding tensor for slot"""
        return self.gpu_buffer[slot]
    
    def clear(self):
        """Clear all cached embeddings"""
        with self.lock:
            self.cache.clear()
            self.free_slots = list(range(self.cache_size))

class EmbeddingService:
    """
    High-performance embedding service with GPU hot-cache and host fallback.
    
    Features:
    - GPU memory cache for frequently accessed embeddings
    - Host memory cache for less frequent embeddings
    - Automatic cache eviction with LRU policy
    - Batch lookup optimization
    - Persistence support
    """
    
    def __init__(self, num_items: int = 50000, emb_dim: int = 128, 
                 gpu_cache_size: int = 4096, host_cache_size: int = 20000,
                 dtype: str = 'float32', enable_persistence: bool = False,
                 cache_dir: str = './embedding_cache'):
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.dtype_str = dtype
        self.dtype = getattr(torch, dtype)
        self.enable_persistence = enable_persistence
        self.cache_dir = cache_dir
        
        # Initialize caches
        self.gpu_cache = EmbeddingCache(gpu_cache_size, emb_dim, self.dtype)
        self.host_cache = OrderedDict()
        self.host_cache_size = host_cache_size
        
        # Statistics
        self.stats = {
            'gpu_hits': 0,
            'host_hits': 0,
            'misses': 0,
            'total_requests': 0
        }
        
        # Initialize embedding table
        self._initialize_embedding_table()
        
        # Load persistent cache if enabled
        if enable_persistence:
            self._load_persistent_cache()
        
        logging.info(f"EmbeddingService initialized: {num_items} items, {emb_dim} dims, "
                    f"GPU cache: {gpu_cache_size}, Host cache: {host_cache_size}")
    
    def _initialize_embedding_table(self):
        """Initialize the full embedding table"""
        # In practice, this would load from disk or generate embeddings
        # For demo purposes, we create random embeddings
        self.full_table = torch.randn(self.num_items, self.emb_dim, dtype=self.dtype)
        
        # Normalize embeddings
        self.full_table = torch.nn.functional.normalize(self.full_table, p=2, dim=1)
    
    def _load_persistent_cache(self):
        """Load persistent cache from disk"""
        try:
            cache_file = os.path.join(self.cache_dir, 'host_cache.pkl')
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.host_cache = OrderedDict(cached_data)
                logging.info(f"Loaded persistent cache with {len(self.host_cache)} items")
        except Exception as e:
            logging.warning(f"Failed to load persistent cache: {e}")
    
    def _save_persistent_cache(self):
        """Save persistent cache to disk"""
        if not self.enable_persistence:
            return
        
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.cache_dir, 'host_cache.pkl')
            with open(cache_file, 'wb') as f:
                pickle.dump(list(self.host_cache.items()), f)
            logging.info(f"Saved persistent cache with {len(self.host_cache)} items")
        except Exception as e:
            logging.warning(f"Failed to save persistent cache: {e}")
    
    def _promote_to_gpu(self, idx: int, embedding: torch.Tensor) -> int:
        """Promote embedding to GPU cache"""
        return self.gpu_cache.put(idx, embedding)
    
    def lookup_single(self, idx: int) -> torch.Tensor:
        """Lookup single embedding"""
        self.stats['total_requests'] += 1
        
        # Check GPU cache first
        gpu_slot = self.gpu_cache.get(idx)
        if gpu_slot is not None:
            self.stats['gpu_hits'] += 1
            return self.gpu_cache.get_embedding(gpu_slot)
        
        # Check host cache
        if idx in self.host_cache:
            self.stats['host_hits'] += 1
            embedding = self.host_cache[idx]
            # Promote to GPU cache
            gpu_slot = self._promote_to_gpu(idx, embedding)
            return self.gpu_cache.get_embedding(gpu_slot)
        
        # Cache miss - load from full table
        self.stats['misses'] += 1
        embedding = self.full_table[idx].clone()
        
        # Add to host cache
        self.host_cache[idx] = embedding
        if len(self.host_cache) > self.host_cache_size:
            self.host_cache.popitem(last=False)
        
        # Promote to GPU cache
        gpu_slot = self._promote_to_gpu(idx, embedding)
        return self.gpu_cache.get_embedding(gpu_slot)
    
    def lookup_batch(self, ids: List[int]) -> torch.Tensor:
        """Lookup batch of embeddings with optimization"""
        if not ids:
            return torch.empty((0, self.emb_dim), device='cuda', dtype=self.dtype)
        
        # Pre-allocate output tensor
        batch_size = len(ids)
        result = torch.zeros((batch_size, self.emb_dim), device='cuda', dtype=self.dtype)
        
        # Process each embedding
        for i, idx in enumerate(ids):
            result[i] = self.lookup_single(idx)
        
        return result
    
    def lookup_batch_optimized(self, ids: List[int]) -> torch.Tensor:
        """Optimized batch lookup with reduced GPU transfers"""
        if not ids:
            return torch.empty((0, self.emb_dim), device='cuda', dtype=self.dtype)
        
        batch_size = len(ids)
        result = torch.zeros((batch_size, self.emb_dim), device='cuda', dtype=self.dtype)
        
        # Group lookups by cache location
        gpu_indices = []
        host_indices = []
        miss_indices = []
        
        for i, idx in enumerate(ids):
            gpu_slot = self.gpu_cache.get(idx)
            if gpu_slot is not None:
                gpu_indices.append((i, gpu_slot))
            elif idx in self.host_cache:
                host_indices.append(i)
            else:
                miss_indices.append(i)
        
        # Process GPU cache hits
        for i, slot in gpu_indices:
            result[i] = self.gpu_cache.get_embedding(slot)
            self.stats['gpu_hits'] += 1
        
        # Process host cache hits
        for i in host_indices:
            idx = ids[i]
            embedding = self.host_cache[idx]
            gpu_slot = self._promote_to_gpu(idx, embedding)
            result[i] = self.gpu_cache.get_embedding(gpu_slot)
            self.stats['host_hits'] += 1
        
        # Process misses
        for i in miss_indices:
            idx = ids[i]
            embedding = self.full_table[idx].clone()
            
            # Add to host cache
            self.host_cache[idx] = embedding
            if len(self.host_cache) > self.host_cache_size:
                self.host_cache.popitem(last=False)
            
            # Promote to GPU cache
            gpu_slot = self._promote_to_gpu(idx, embedding)
            result[i] = self.gpu_cache.get_embedding(gpu_slot)
            self.stats['misses'] += 1
        
        self.stats['total_requests'] += batch_size
        return result
    
    def warmup_cache(self, indices: List[int]):
        """Warm up cache with frequently accessed embeddings"""
        logging.info(f"Warming up cache with {len(indices)} embeddings")
        for idx in indices:
            if idx < self.num_items:
                self.lookup_single(idx)
        logging.info("Cache warmup completed")
    
    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache statistics"""
        total = self.stats['total_requests']
        if total == 0:
            return {'gpu_hit_rate': 0.0, 'host_hit_rate': 0.0, 'miss_rate': 0.0}
        
        return {
            'gpu_hit_rate': self.stats['gpu_hits'] / total,
            'host_hit_rate': self.stats['host_hits'] / total,
            'miss_rate': self.stats['misses'] / total,
            'total_requests': total,
            'gpu_cache_size': len(self.gpu_cache.cache),
            'host_cache_size': len(self.host_cache)
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.gpu_cache.clear()
        self.host_cache.clear()
        logging.info("All caches cleared")
    
    def save_cache(self):
        """Save persistent cache"""
        self._save_persistent_cache()
    
    def __del__(self):
        """Cleanup on destruction"""
        if self.enable_persistence:
            self._save_persistent_cache()

# Convenience function for backward compatibility
def create_embedding_service(num_items: int = 50000, emb_dim: int = 32, 
                           gpu_cache_size: int = 4096) -> EmbeddingService:
    """Create embedding service with default parameters"""
    return EmbeddingService(
        num_items=num_items,
        emb_dim=emb_dim,
        gpu_cache_size=gpu_cache_size
    )

# Example usage and testing
if __name__ == "__main__":
    # Create embedding service
    service = EmbeddingService(
        num_items=10000,
        emb_dim=128,
        gpu_cache_size=1024,
        host_cache_size=5000,
        enable_persistence=True
    )
    
    # Test single lookup
    embedding = service.lookup_single(42)
    print(f"Single lookup result shape: {embedding.shape}")
    
    # Test batch lookup
    batch_ids = [1, 5, 10, 15, 20]
    batch_embeddings = service.lookup_batch(batch_ids)
    print(f"Batch lookup result shape: {batch_embeddings.shape}")
    
    # Test optimized batch lookup
    optimized_embeddings = service.lookup_batch_optimized(batch_ids)
    print(f"Optimized batch lookup result shape: {optimized_embeddings.shape}")
    
    # Print statistics
    stats = service.get_cache_stats()
    print("Cache statistics:", stats)
    
    # Warmup cache
    warmup_indices = list(range(100))
    service.warmup_cache(warmup_indices)
    
    # Print updated statistics
    stats = service.get_cache_stats()
    print("Updated cache statistics:", stats)
