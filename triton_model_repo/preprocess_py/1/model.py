# Triton Python backend model for preprocessing
import numpy as np
import triton_python_backend_utils as pb_utils
import logging
from typing import List, Dict, Any

class PreprocessModel:
    """Triton Python backend model for input preprocessing"""
    
    def __init__(self):
        self.model_config = None
        self.feature_dim = 16
        self.vocab_size = 10000
        
    def initialize(self, args: Dict[str, Any]) -> None:
        """Initialize the model with configuration"""
        self.model_config = args['model_config']
        
        # Parse model configuration
        try:
            if 'parameters' in self.model_config:
                feature_dim_param = self.model_config['parameters'].get('feature_dim', {'string_value': '16'})
                self.feature_dim = int(feature_dim_param['string_value'])
                
                vocab_size_param = self.model_config['parameters'].get('vocab_size', {'string_value': '10000'})
                self.vocab_size = int(vocab_size_param['string_value'])
                
            logging.info(f"Preprocess model initialized: feature_dim={self.feature_dim}, vocab_size={self.vocab_size}")
        except Exception as e:
            logging.warning(f"Failed to parse model config: {e}")
    
    def execute(self, requests: List[pb_utils.InferenceRequest]) -> List[pb_utils.InferenceResponse]:
        """Execute preprocessing requests"""
        responses = []
        
        for request in requests:
            try:
                # Get input tensor
                input_tensor = pb_utils.get_input_tensor_by_name(request, 'raw_input')
                if input_tensor is None:
                    raise ValueError("Input tensor 'raw_input' not found")
                
                # Convert to numpy
                raw_input_np = input_tensor.as_numpy()
                if raw_input_np is None:
                    raise ValueError("Failed to convert input tensor to numpy")
                
                # Validate input shape
                if len(raw_input_np.shape) != 2:
                    raise ValueError(f"Expected 2D input, got shape {raw_input_np.shape}")
                
                batch_size, seq_len = raw_input_np.shape
                
                # Preprocessing logic: extract features from raw input
                # For demo purposes, we'll create dummy features
                # In practice, this would involve tokenization, feature extraction, etc.
                
                # Create dummy dense features (batch_size, feature_dim)
                dense_features = np.random.randn(batch_size, self.feature_dim).astype(np.float32)
                
                # For ensemble, we need to output feature indices
                # Convert dense features to indices (simplified)
                feature_indices = np.arange(batch_size * self.feature_dim, dtype=np.int32)
                
                # Create output tensor
                output_tensor_pb = pb_utils.Tensor('out_feats', feature_indices)
                
                # Create response
                response = pb_utils.InferenceResponse(output_tensors=[output_tensor_pb])
                responses.append(response)
                
                logging.debug(f"Preprocessing completed for batch size {batch_size}")
                
            except Exception as e:
                logging.error(f"Error processing request: {e}")
                # Create error response
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"Preprocessing failed: {str(e)}")
                )
                responses.append(error_response)
        
        return responses
    
    def finalize(self) -> None:
        """Clean up resources"""
        logging.info("Preprocess model finalized")

# Global model instance
_model = None

def initialize(args: Dict[str, Any]) -> None:
    """Initialize the model (called by Triton)"""
    global _model
    _model = PreprocessModel()
    _model.initialize(args)

def execute(requests: List[pb_utils.InferenceRequest]) -> List[pb_utils.InferenceResponse]:
    """Execute preprocessing requests (called by Triton)"""
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
