# Triton Python backend model for preprocessing
import numpy as np
import triton_python_backend_utils as pb_utils
import logging
from typing import List, Dict, Any

class PreprocessModel:
    """Triton Python backend model for input preprocessing"""
    
    def __init__(self):
        self.model_config = None
        self.feature_dim = 32  # 扩展特征维度
        self.vocab_size = 10000
        self.max_sequence_length = 50
        self.enable_extended_features = True
        
    def initialize(self, args: Dict[str, Any]) -> None:
        """Initialize the model with configuration"""
        self.model_config = args['model_config']
        
        # Parse model configuration
        try:
            if 'parameters' in self.model_config:
                feature_dim_param = self.model_config['parameters'].get('feature_dim', {'string_value': '32'})
                self.feature_dim = int(feature_dim_param['string_value'])
                
                vocab_size_param = self.model_config['parameters'].get('vocab_size', {'string_value': '10000'})
                self.vocab_size = int(vocab_size_param['string_value'])
                
                max_seq_param = self.model_config['parameters'].get('max_sequence_length', {'string_value': '50'})
                self.max_sequence_length = int(max_seq_param['string_value'])
                
                extended_param = self.model_config['parameters'].get('enable_extended_features', {'string_value': 'true'})
                self.enable_extended_features = extended_param['string_value'].lower() == 'true'
                
            logging.info(f"Preprocess model initialized: feature_dim={self.feature_dim}, vocab_size={self.vocab_size}, max_seq_len={self.max_sequence_length}, extended_features={self.enable_extended_features}")
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
                
                # 使用用户行为序列处理
                if self.enable_extended_features:
                    # 解析用户行为序列数据
                    # 假设输入是JSON格式的用户行为序列
                    try:
                        # 这里应该解析实际的用户行为数据
                        # 为了演示，我们创建模拟的用户行为特征
                        
                        # 模拟用户行为特征提取
                        dense_features = np.zeros((batch_size, self.feature_dim), dtype=np.float32)
                        
                        for i in range(batch_size):
                            # 观看时长特征 (0-9)
                            dense_features[i, 0:10] = np.random.uniform(0, 1, 10)
                            
                            # 观看百分比特征 (10-14)
                            dense_features[i, 10:15] = np.random.uniform(0, 1, 5)
                            
                            # 交互标志特征 (15-19): like, favorite, share, comment, follow
                            dense_features[i, 15:20] = np.random.choice([0, 1], 5, p=[0.7, 0.3])
                            
                            # 时间特征 (20-24): time_of_day, day_of_week
                            dense_features[i, 20:22] = np.random.uniform(0, 1, 2)
                            
                            # 设备特征 (25-29): device_type, network_type
                            dense_features[i, 25:27] = np.random.uniform(0, 1, 2)
                            
                            # 推荐特征 (30-31): source, position
                            dense_features[i, 30:32] = np.random.uniform(0, 1, 2)
                        
                        # 转换为特征索引
                        feature_indices = np.arange(batch_size * self.feature_dim, dtype=np.int32)
                        
                        logging.debug(f"处理了 {batch_size} 个用户行为序列，特征维度: {self.feature_dim}")
                        
                    except Exception as e:
                        logging.warning(f"用户行为序列处理失败，使用默认特征: {e}")
                        # 回退到默认处理
                        dense_features = np.random.randn(batch_size, self.feature_dim).astype(np.float32)
                        feature_indices = np.arange(batch_size * self.feature_dim, dtype=np.int32)
                else:
                    # 原始处理逻辑
                    dense_features = np.random.randn(batch_size, self.feature_dim).astype(np.float32)
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
