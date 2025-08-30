# Triton Python backend model implementation for pairwise interaction
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
import sys
import os
import time
import logging
from typing import List, Dict, Any

# Add kernels directory to path for importing custom operations
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../kernels/triton_ops'))

try:
    from interaction_wrapper import interaction_op
except ImportError as e:
    logging.error(f"Failed to import interaction_wrapper: {e}")
    interaction_op = None

class InteractionModel:
    """Triton Python backend model for pairwise interaction computation"""
    
    def __init__(self):
        self.model_config = None
        self.block_size = 64  # Default block size for Triton kernel
        self.device = 'cuda'
        self.dtype = torch.float16
        
    def initialize(self, args: Dict[str, Any]) -> None:
        """Initialize the model with configuration"""
        self.model_config = args['model_config']
        
        # Parse model configuration
        try:
            # Get block size from model config if available
            if 'parameters' in self.model_config:
                block_param = self.model_config['parameters'].get('block_size', {'string_value': '64'})
                self.block_size = int(block_param['string_value'])
                logging.info(f"Using block size: {self.block_size}")
        except Exception as e:
            logging.warning(f"Failed to parse block size from config: {e}")
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logging.error("CUDA is not available")
            raise RuntimeError("CUDA is required for this model")
        
        # Verify interaction operation is available
        if interaction_op is None:
            logging.error("Interaction operation not available")
            raise RuntimeError("Failed to import interaction operation")
        
        logging.info("Interaction model initialized successfully")
    
    def execute(self, requests: List[pb_utils.InferenceRequest]) -> List[pb_utils.InferenceResponse]:
        """Execute inference requests"""
        responses = []
        
        for request in requests:
            try:
                # Get input tensor
                input_tensor = pb_utils.get_input_tensor_by_name(request, 'embeddings')
                if input_tensor is None:
                    raise ValueError("Input tensor 'embeddings' not found")
                
                # Convert to numpy and validate
                embeddings_np = input_tensor.as_numpy()
                if embeddings_np is None:
                    raise ValueError("Failed to convert input tensor to numpy")
                
                # Validate input shape
                if len(embeddings_np.shape) != 3:
                    raise ValueError(f"Expected 3D input, got shape {embeddings_np.shape}")
                
                batch_size, num_features, feature_dim = embeddings_np.shape
                
                # Convert to PyTorch tensor and move to GPU
                start_time = time.time()
                embeddings_tensor = torch.from_numpy(embeddings_np).to(
                    device=self.device, 
                    dtype=self.dtype,
                    non_blocking=True
                )
                
                # Perform interaction operation
                try:
                    output_tensor = interaction_op(embeddings_tensor, BLOCK=self.block_size)
                except Exception as e:
                    logging.error(f"Interaction operation failed: {e}")
                    # Fallback to CPU implementation if GPU fails
                    logging.info("Falling back to CPU implementation")
                    output_tensor = self._cpu_interaction(embeddings_tensor)
                
                # Convert back to numpy
                output_np = output_tensor.cpu().numpy()
                
                # Create output tensor
                output_tensor_pb = pb_utils.Tensor('pairwise', output_np)
                
                # Create response
                response = pb_utils.InferenceResponse(output_tensors=[output_tensor_pb])
                responses.append(response)
                
                inference_time = time.time() - start_time
                logging.debug(f"Inference completed in {inference_time*1000:.2f}ms for batch size {batch_size}")
                
            except Exception as e:
                logging.error(f"Error processing request: {e}")
                # Create error response
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"Model execution failed: {str(e)}")
                )
                responses.append(error_response)
        
        return responses
    
    def _cpu_interaction(self, embeddings: torch.Tensor) -> torch.Tensor:
        """CPU fallback implementation of pairwise interaction"""
        batch_size, num_features, feature_dim = embeddings.shape
        num_interactions = num_features * (num_features - 1) // 2
        
        # Move to CPU if not already there
        if embeddings.device.type != 'cpu':
            embeddings = embeddings.cpu().float()
        
        # Initialize output tensor
        output = torch.zeros(batch_size, num_interactions, dtype=torch.float32)
        
        # Compute pairwise interactions
        interaction_idx = 0
        for i in range(num_features):
            for j in range(i + 1, num_features):
                interaction = torch.sum(embeddings[:, i, :] * embeddings[:, j, :], dim=1)
                output[:, interaction_idx] = interaction
                interaction_idx += 1
        
        return output
    
    def finalize(self) -> None:
        """Clean up resources"""
        logging.info("Interaction model finalized")

# Global model instance
_model = None

def initialize(args: Dict[str, Any]) -> None:
    """Initialize the model (called by Triton)"""
    global _model
    _model = InteractionModel()
    _model.initialize(args)

def execute(requests: List[pb_utils.InferenceRequest]) -> List[pb_utils.InferenceResponse]:
    """Execute inference requests (called by Triton)"""
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
