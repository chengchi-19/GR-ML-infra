# Triton Python backend model for embedding service
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
import sys
import os
import logging
from typing import List, Dict, Any

# Add src directory to path for importing embedding service
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../src'))

try:
    from embedding_service import EmbeddingService
    EMBEDDING_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import embedding_service: {e}")
    EMBEDDING_AVAILABLE = False

class EmbeddingServiceModel:
    """Triton Python backend model for embedding service"""
    
    def __init__(self):
        self.model_config = None
        self.embedding_service = None
        self.embedding_dim = 32
        self.num_items = 50000
        self.gpu_cache_size = 4096
        self.host_cache_size = 20000
        
    def initialize(self, args: Dict[str, Any]) -> None:
        """Initialize the model with configuration"""
        self.model_config = args['model_config']
        
        # Parse model configuration
        try:
            if 'parameters' in self.model_config:
                embedding_dim_param = self.model_config['parameters'].get('embedding_dim', {'string_value': '32'})
                self.embedding_dim = int(embedding_dim_param['string_value'])
                
                num_items_param = self.model_config['parameters'].get('num_items', {'string_value': '50000'})
                self.num_items = int(num_items_param['string_value'])
                
                gpu_cache_size_param = self.model_config['parameters'].get('gpu_cache_size', {'string_value': '4096'})
                self.gpu_cache_size = int(gpu_cache_size_param['string_value'])
                
                host_cache_size_param = self.model_config['parameters'].get('host_cache_size', {'string_value': '20000'})
                self.host_cache_size = int(host_cache_size_param['string_value'])
                
            logging.info(f"Embedding service config: dim={self.embedding_dim}, items={self.num_items}, "
                        f"gpu_cache={self.gpu_cache_size}, host_cache={self.host_cache_size}")
        except Exception as e:
            logging.warning(f"Failed to parse embedding service config: {e}")
        
        # Initialize embedding service
        if EMBEDDING_AVAILABLE:
            try:
                self.embedding_service = EmbeddingService(
                    num_items=self.num_items,
                    emb_dim=self.embedding_dim,
                    gpu_cache_size=self.gpu_cache_size,
                    host_cache_size=self.host_cache_size,
                    enable_persistence=False  # Disable persistence for Triton backend
                )
                logging.info("Embedding service initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize embedding service: {e}")
                raise
        else:
            logging.error("Embedding service not available")
            raise RuntimeError("Embedding service module not available")
    
    def execute(self, requests: List[pb_utils.InferenceRequest]) -> List[pb_utils.InferenceResponse]:
        """Execute embedding lookup requests"""
        responses = []
        
        for request in requests:
            try:
                # Get input tensor
                input_tensor = pb_utils.get_input_tensor_by_name(request, 'ids')
                if input_tensor is None:
                    raise ValueError("Input tensor 'ids' not found")
                
                # Convert to numpy
                ids_np = input_tensor.as_numpy()
                if ids_np is None:
                    raise ValueError("Failed to convert input tensor to numpy")
                
                # Validate input
                if len(ids_np.shape) != 1:
                    raise ValueError(f"Expected 1D input, got shape {ids_np.shape}")
                
                # Convert to list of integers
                ids_list = ids_np.tolist()
                
                # Lookup embeddings
                embeddings_tensor = self.embedding_service.lookup_batch_optimized(ids_list)
                
                # Convert to numpy and ensure correct shape
                embeddings_np = embeddings_tensor.cpu().numpy()
                
                # Ensure output shape is correct (batch_size, embedding_dim)
                if len(embeddings_np.shape) == 1:
                    embeddings_np = embeddings_np.reshape(-1, self.embedding_dim)
                
                # Convert to FP16 for output
                embeddings_fp16 = embeddings_np.astype(np.float16)
                
                # Create output tensor
                output_tensor_pb = pb_utils.Tensor('embeddings', embeddings_fp16)
                
                # Create response
                response = pb_utils.InferenceResponse(output_tensors=[output_tensor_pb])
                responses.append(response)
                
                logging.debug(f"Embedding lookup completed for {len(ids_list)} items")
                
            except Exception as e:
                logging.error(f"Error processing embedding request: {e}")
                # Create error response
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"Embedding lookup failed: {str(e)}")
                )
                responses.append(error_response)
        
        return responses
    
    def finalize(self) -> None:
        """Clean up resources"""
        if self.embedding_service is not None:
            self.embedding_service.clear_cache()
        logging.info("Embedding service model finalized")

# Global model instance
_model = None

def initialize(args: Dict[str, Any]) -> None:
    """Initialize the model (called by Triton)"""
    global _model
    _model = EmbeddingServiceModel()
    _model.initialize(args)

def execute(requests: List[pb_utils.InferenceRequest]) -> List[pb_utils.InferenceResponse]:
    """Execute embedding lookup requests (called by Triton)"""
    global _model
    if _model is None:
        raise RuntimeError("Model not initialized")
    return _model.execute(requests)

def finalize() -> None:
    """Finalize the model (called by Triton)"""
    global _model
    if _model is not None:
        _model.finalize()
        _model = None
