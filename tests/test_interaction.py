import pytest
import torch
import numpy as np
import sys
import os

# Add src and kernels to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../kernels/triton_ops'))

try:
    from interaction_wrapper import interaction_op
    INTERACTION_AVAILABLE = True
except ImportError:
    INTERACTION_AVAILABLE = False
    print("Warning: interaction_wrapper not available, skipping GPU tests")

try:
    from embedding_service import EmbeddingService
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("Warning: embedding_service not available, skipping embedding tests")

class TestInteractionKernel:
    """Test suite for interaction kernel operations"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def test_interaction_small(self):
        """Test interaction kernel with small inputs"""
        if not INTERACTION_AVAILABLE or not torch.cuda.is_available():
            pytest.skip("Interaction kernel or CUDA not available")
        
        B, F, D = 2, 8, 32
        emb = torch.randn((B, F, D), device='cuda', dtype=torch.float16)
        
        # Test with different block sizes
        for block_size in [32, 64, 128]:
            try:
                out = interaction_op(emb, BLOCK=block_size)
                
                # Reference implementation
                ref = torch.zeros((B, F*(F-1)//2), device='cuda', dtype=torch.float32)
                k = 0
                for i in range(F):
                    for j in range(i+1, F):
                        ref[:,k] = (emb[:,i,:].float() * emb[:,j,:].float()).sum(dim=1)
                        k += 1
                
                # Check correctness
                assert torch.allclose(out.to(ref.dtype), ref, rtol=1e-2, atol=1e-2), \
                    f"Interaction kernel failed for block size {block_size}"
                
            except Exception as e:
                pytest.fail(f"Interaction kernel failed for block size {block_size}: {e}")
    
    def test_interaction_large(self):
        """Test interaction kernel with larger inputs"""
        if not INTERACTION_AVAILABLE or not torch.cuda.is_available():
            pytest.skip("Interaction kernel or CUDA not available")
        
        B, F, D = 4, 16, 64
        emb = torch.randn((B, F, D), device='cuda', dtype=torch.float16)
        
        out = interaction_op(emb, BLOCK=64)
        expected_shape = (B, F*(F-1)//2)
        
        assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"
        assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"
        assert out.device.type == 'cuda', f"Expected CUDA tensor, got {out.device}"
    
    def test_interaction_edge_cases(self):
        """Test interaction kernel with edge cases"""
        if not INTERACTION_AVAILABLE or not torch.cuda.is_available():
            pytest.skip("Interaction kernel or CUDA not available")
        
        # Test with minimum feature count
        B, F, D = 1, 2, 16
        emb = torch.randn((B, F, D), device='cuda', dtype=torch.float16)
        out = interaction_op(emb, BLOCK=32)
        assert out.shape == (B, 1), f"Expected shape ({B}, 1), got {out.shape}"
        
        # Test with large batch size
        B, F, D = 16, 8, 32
        emb = torch.randn((B, F, D), device='cuda', dtype=torch.float16)
        out = interaction_op(emb, BLOCK=64)
        assert out.shape == (B, F*(F-1)//2), f"Expected shape ({B}, {F*(F-1)//2}), got {out.shape}"
    
    def test_interaction_dtype_handling(self):
        """Test interaction kernel with different data types"""
        if not INTERACTION_AVAILABLE or not torch.cuda.is_available():
            pytest.skip("Interaction kernel or CUDA not available")
        
        B, F, D = 2, 8, 32
        
        # Test with float32 input
        emb_f32 = torch.randn((B, F, D), device='cuda', dtype=torch.float32)
        out_f32 = interaction_op(emb_f32, BLOCK=64)
        assert out_f32.dtype == torch.float32
        
        # Test with float16 input
        emb_f16 = torch.randn((B, F, D), device='cuda', dtype=torch.float16)
        out_f16 = interaction_op(emb_f16, BLOCK=64)
        assert out_f16.dtype == torch.float32  # Output should be float32
    
    def test_interaction_error_handling(self):
        """Test interaction kernel error handling"""
        if not INTERACTION_AVAILABLE or not torch.cuda.is_available():
            pytest.skip("Interaction kernel or CUDA not available")
        
        B, F, D = 2, 8, 32
        
        # Test with CPU tensor (should fail)
        emb_cpu = torch.randn((B, F, D), device='cpu', dtype=torch.float16)
        with pytest.raises(AssertionError):
            interaction_op(emb_cpu, BLOCK=64)
        
        # Test with invalid block size
        emb = torch.randn((B, F, D), device='cuda', dtype=torch.float16)
        with pytest.raises(Exception):
            interaction_op(emb, BLOCK=0)

class TestEmbeddingService:
    """Test suite for embedding service"""
    
    @pytest.fixture
    def embedding_service(self):
        """Create embedding service for testing"""
        if not EMBEDDING_AVAILABLE:
            pytest.skip("Embedding service not available")
        
        return EmbeddingService(
            num_items=1000,
            emb_dim=64,
            gpu_cache_size=256,
            host_cache_size=500,
            enable_persistence=False
        )
    
    def test_single_lookup(self, embedding_service):
        """Test single embedding lookup"""
        if not EMBEDDING_AVAILABLE:
            pytest.skip("Embedding service not available")
        
        # Test valid indices
        for idx in [0, 100, 500, 999]:
            embedding = embedding_service.lookup_single(idx)
            assert embedding.shape == (embedding_service.emb_dim,)
            assert embedding.device.type == 'cuda'
            assert embedding.dtype == embedding_service.dtype
        
        # Test invalid index
        with pytest.raises(IndexError):
            embedding_service.lookup_single(1000)
    
    def test_batch_lookup(self, embedding_service):
        """Test batch embedding lookup"""
        if not EMBEDDING_AVAILABLE:
            pytest.skip("Embedding service not available")
        
        # Test batch lookup
        indices = [1, 5, 10, 15, 20]
        embeddings = embedding_service.lookup_batch(indices)
        
        assert embeddings.shape == (len(indices), embedding_service.emb_dim)
        assert embeddings.device.type == 'cuda'
        assert embeddings.dtype == embedding_service.dtype
        
        # Test optimized batch lookup
        optimized_embeddings = embedding_service.lookup_batch_optimized(indices)
        assert torch.allclose(embeddings, optimized_embeddings, rtol=1e-5, atol=1e-5)
    
    def test_cache_behavior(self, embedding_service):
        """Test cache behavior and statistics"""
        if not EMBEDDING_AVAILABLE:
            pytest.skip("Embedding service not available")
        
        # Initial statistics
        initial_stats = embedding_service.get_cache_stats()
        assert initial_stats['total_requests'] == 0
        
        # Lookup same embedding multiple times
        for _ in range(10):
            embedding_service.lookup_single(42)
        
        # Check statistics
        stats = embedding_service.get_cache_stats()
        assert stats['total_requests'] == 10
        assert stats['gpu_hit_rate'] > 0.5  # Should have GPU cache hits
        
        # Test cache warmup
        warmup_indices = list(range(50))
        embedding_service.warmup_cache(warmup_indices)
        
        # Check updated statistics
        updated_stats = embedding_service.get_cache_stats()
        assert updated_stats['total_requests'] == 60  # 10 + 50
    
    def test_cache_eviction(self, embedding_service):
        """Test cache eviction behavior"""
        if not EMBEDDING_AVAILABLE:
            pytest.skip("Embedding service not available")
        
        # Fill GPU cache
        for i in range(embedding_service.gpu_cache.cache_size + 10):
            embedding_service.lookup_single(i)
        
        # Check that cache size is maintained
        stats = embedding_service.get_cache_stats()
        assert stats['gpu_cache_size'] <= embedding_service.gpu_cache.cache_size
    
    def test_persistence(self):
        """Test embedding service persistence"""
        if not EMBEDDING_AVAILABLE:
            pytest.skip("Embedding service not available")
        
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create service with persistence
            service = EmbeddingService(
                num_items=100,
                emb_dim=32,
                gpu_cache_size=50,
                host_cache_size=25,
                enable_persistence=True,
                cache_dir=temp_dir
            )
            
            # Add some embeddings to cache
            for i in range(10):
                service.lookup_single(i)
            
            # Save cache
            service.save_cache()
            
            # Check that cache file exists
            cache_file = os.path.join(temp_dir, 'host_cache.pkl')
            assert os.path.exists(cache_file)

class TestIntegration:
    """Integration tests for the complete pipeline"""
    
    def test_onnx_export(self):
        """Test ONNX export functionality"""
        try:
            from export_onnx import GenerativeRecommendationModel, create_dummy_data
            
            # Create model
            model = GenerativeRecommendationModel(
                vocab_size=1000,
                embedding_dim=64,
                num_features=8
            )
            model.eval()
            
            # Create dummy data
            input_ids, dense_features, attention_mask = create_dummy_data(
                batch_size=2, seq_len=16, num_features=8, vocab_size=1000
            )
            
            # Test forward pass
            with torch.no_grad():
                logits, feature_scores, hidden_states = model.forward_prefill(
                    input_ids, dense_features, attention_mask
                )
            
            assert logits.shape == (2, 16, 1000)
            assert feature_scores.shape == (2, 1)
            assert hidden_states.shape == (2, 16, 64)
            
        except ImportError:
            pytest.skip("ONNX export module not available")
    
    def test_tensorrt_build(self):
        """Test TensorRT engine building"""
        # This test would require TensorRT installation
        # For now, we just check if the module can be imported
        try:
            from build_engine import build_with_api, validate_engine
            # Module can be imported
            assert True
        except ImportError:
            pytest.skip("TensorRT build module not available")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
