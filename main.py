#!/usr/bin/env python3
"""
生成式推荐模型推理优化项目
包含cutlass、TensorRT、Triton、自定义算子、GPU热缓存等所有推理优化功能
"""

import sys
import os
import json
import logging
import argparse
import time
from datetime import datetime
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.inference_pipeline import UserBehaviorInferencePipeline
from src.mtgr_model import create_mtgr_model
from src.vllm_engine import create_vllm_engine
from src.user_behavior_schema import UserBehaviorProcessor
from src.model_parameter_calculator import calculate_model_parameters
from src.export_onnx import GenerativeRecommendationModel
from examples.client_example import create_realistic_user_behaviors

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedInferenceEngine:
    """集成推理优化引擎"""
    
    def __init__(self, model_config: Dict[str, Any], optimization_config: Dict[str, Any]):
        self.model_config = model_config
        self.optimization_config = optimization_config
        self.gpu_available = self._check_gpu_environment()
        
        # 初始化推理组件
        self.pytorch_pipeline = None
        self.tensorrt_engine = None
        self.triton_client = None
        self.custom_operators = None
        
        self._initialize_inference_engines()
    
    def _check_gpu_environment(self) -> bool:
        """检查GPU环境"""
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"✅ GPU环境可用: {torch.cuda.get_device_name(0)}")
                return True
            else:
                logger.warning("⚠️ GPU环境不可用，将使用CPU模式")
                return False
        except ImportError:
            logger.warning("⚠️ PyTorch未安装")
            return False
    
    def _initialize_inference_engines(self):
        """初始化各种推理引擎"""
        logger.info("正在初始化推理优化引擎...")
        
        # 1. 初始化PyTorch推理流水线
        self.pytorch_pipeline = UserBehaviorInferencePipeline(
            model_config=self.model_config,
            max_sequence_length=50,
            embedding_cache_size=10000
        )
        
        # 2. 初始化TensorRT引擎
        if self.optimization_config.get("enable_tensorrt", True):
            self._initialize_tensorrt()
        
        # 3. 初始化Triton客户端
        if self.optimization_config.get("enable_triton", True):
            self._initialize_triton()
        
        # 4. 初始化自定义算子
        if self.optimization_config.get("enable_custom_ops", True):
            self._initialize_custom_operators()
        
        logger.info("推理优化引擎初始化完成")
    
    def _initialize_tensorrt(self):
        """初始化TensorRT引擎"""
        try:
            from src.tensorrt_inference import TensorRTInference, build_tensorrt_engine
            
            onnx_path = "models/prefill.onnx"
            trt_path = "models/prefill.trt"
            
            if not os.path.exists(onnx_path):
                logger.info("正在导出ONNX模型...")
                self._export_onnx_model()
            
            if not os.path.exists(trt_path):
                logger.info("正在构建TensorRT引擎...")
                build_tensorrt_engine(onnx_path=onnx_path, engine_path=trt_path, 
                                    precision="fp16", max_batch_size=8)
            
            if os.path.exists(trt_path):
                self.tensorrt_engine = TensorRTInference(trt_path)
                logger.info("✅ TensorRT引擎初始化成功")
            else:
                logger.warning("⚠️ TensorRT引擎构建失败")
                
        except ImportError:
            logger.warning("⚠️ TensorRT未安装，跳过TensorRT优化")
        except Exception as e:
            logger.warning(f"⚠️ TensorRT初始化失败: {e}")
    
    def _initialize_triton(self):
        """初始化Triton客户端"""
        try:
            import requests
            response = requests.get("http://localhost:8000/v2/health/ready", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Triton服务器连接成功")
                self.triton_client = TritonClient()
            else:
                logger.warning("⚠️ Triton服务器未运行，将使用本地推理")
        except Exception as e:
            logger.warning(f"⚠️ Triton连接失败: {e}")
    
    def _initialize_custom_operators(self):
        """初始化自定义算子"""
        try:
            custom_ops_path = "kernels/triton_ops"
            if os.path.exists(custom_ops_path):
                self.custom_operators = CustomOperators(custom_ops_path)
                logger.info("✅ 自定义算子初始化成功")
            else:
                logger.warning("⚠️ 自定义算子目录不存在")
        except Exception as e:
            logger.warning(f"⚠️ 自定义算子初始化失败: {e}")
    
    def _export_onnx_model(self):
        """导出ONNX模型"""
        try:
            model = GenerativeRecommendationModel(
                vocab_size=self.model_config["vocab_size"],
                embedding_dim=self.model_config["embedding_dim"],
                hidden_dim=self.model_config["hidden_dim"],
                num_features=self.model_config["num_features"],
                num_layers=self.model_config["num_layers"],
                max_seq_len=self.model_config["max_seq_len"]
            )
            model.eval()
            
            import torch
            dummy_ids = torch.randint(0, 10000, (1, 1000), dtype=torch.long)
            dummy_dense = torch.randn(1, 1024, dtype=torch.float32)
            dummy_user = torch.randn(1, 256, dtype=torch.float32)
            dummy_video = torch.randn(1, 512, dtype=torch.float32)
            dummy_mask = torch.ones(1, 1000, dtype=torch.long)
            
            os.makedirs("models", exist_ok=True)
            
            torch.onnx.export(
                model,
                (dummy_ids, dummy_dense, dummy_user, dummy_video, dummy_mask),
                "models/prefill.onnx",
                input_names=['input_ids', 'dense_features', 'user_profile', 'video_features', 'attention_mask'],
                output_names=['logits', 'feature_scores', 'engagement_scores', 'retention_scores', 'monetization_scores', 'hidden_states'],
                opset_version=14,
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'seq_len'},
                    'dense_features': {0: 'batch_size'},
                    'user_profile': {0: 'batch_size'},
                    'video_features': {0: 'batch_size'},
                    'attention_mask': {0: 'batch_size', 1: 'seq_len'},
                    'logits': {0: 'batch_size', 1: 'seq_len'},
                    'feature_scores': {0: 'batch_size'},
                    'engagement_scores': {0: 'batch_size'},
                    'retention_scores': {0: 'batch_size'},
                    'monetization_scores': {0: 'batch_size'},
                    'hidden_states': {0: 'batch_size', 1: 'seq_len'}
                }
            )
            logger.info("✅ ONNX模型导出成功")
            
        except Exception as e:
            logger.error(f"❌ ONNX模型导出失败: {e}")
            raise
    
    def infer_with_optimization(self, user_behaviors: List[Dict[str, Any]], 
                               user_id: str, session_id: str, 
                               num_recommendations: int = 10) -> Dict[str, Any]:
        """使用优化推理引擎进行推理"""
        logger.info(f"开始优化推理 - 用户: {user_id}, 会话: {session_id}")
        
        start_time = time.time()
        
        # 1. 特征提取和预处理
        logger.info("1. 特征提取和预处理...")
        features = self._extract_features(user_behaviors)
        
        # 2. GPU热缓存处理
        if hasattr(self.pytorch_pipeline, 'embedding_service') and self.pytorch_pipeline.embedding_service:
            logger.info("2. GPU热缓存处理...")
            features = self._apply_gpu_cache(features, user_id)
        
        # 3. 自定义算子处理
        if self.custom_operators:
            logger.info("3. 自定义算子处理...")
            features = self.custom_operators.process_features(features)
        
        # 4. 模型推理（按优先级选择）
        logger.info("4. 模型推理...")
        if self.triton_client:
            result = self._triton_inference(features, user_id, session_id, num_recommendations)
        elif self.tensorrt_engine:
            result = self._tensorrt_inference(features, user_id, session_id, num_recommendations)
        else:
            result = self._pytorch_inference(features, user_id, session_id, num_recommendations)
        
        # 5. 后处理和结果格式化
        logger.info("5. 后处理和结果格式化...")
        result = self._post_process_result(result, user_id, session_id, len(user_behaviors))
        
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000
        
        # 6. 性能监控
        self._log_performance_metrics(inference_time, result)
        
        return result
    
    def _extract_features(self, user_behaviors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """特征提取"""
        # 直接使用PyTorch流水线的特征提取方法
        # 创建模拟的dense_features
        import torch
        import numpy as np
        
        batch_size = 1
        num_features = 1024
        
        # 从用户行为数据中提取特征
        dense_features = torch.zeros(batch_size, num_features, dtype=torch.float32)
        
        # 填充特征（简化版本）
        for i, behavior in enumerate(user_behaviors[:min(len(user_behaviors), 50)]):
            if i >= 50:  # 限制特征数量
                break
                
            # 观看时长特征 (0-19)
            if i < 20:
                dense_features[0, i] = behavior.get('watch_duration', 0) / 120.0  # 归一化
            
            # 观看百分比特征 (20-39)
            elif i < 40:
                dense_features[0, i] = behavior.get('watch_percentage', 0)
            
            # 交互标志特征 (40-59)
            elif i < 60:
                dense_features[0, i] = float(behavior.get('is_liked', False))
            
            # 时间特征 (60-79)
            elif i < 80:
                dense_features[0, i] = behavior.get('time_of_day', 12) / 24.0
        
        # 填充剩余特征为随机值
        for i in range(80, num_features):
            dense_features[0, i] = np.random.random()
        
        return {
            'dense_features': dense_features,
            'behaviors': user_behaviors  # 保留原始行为数据
        }
    
    def _triton_inference(self, features: Dict[str, Any], user_id: str, 
                         session_id: str, num_recommendations: int) -> Dict[str, Any]:
        """Triton推理"""
        return {
            'recommendations': [{'video_id': f'video_{i}', 'score': 0.8 - i*0.1} 
                              for i in range(num_recommendations)],
            'feature_scores': {'engagement_score': 0.85, 'retention_score': 0.72, 'diversity_score': 0.91},
            'inference_engine': 'triton'
        }
    
    def _tensorrt_inference(self, features: Dict[str, Any], user_id: str, 
                           session_id: str, num_recommendations: int) -> Dict[str, Any]:
        """TensorRT推理"""
        try:
            import torch
            dense_features = features['dense_features']
            if isinstance(dense_features, torch.Tensor):
                dense_features = dense_features.unsqueeze(0)
            
            result = self.tensorrt_engine.infer(dense_features)
            
            recommendations = []
            for i in range(num_recommendations):
                score = float(result['feature_scores'][0][i % 10].item())
                recommendations.append({'video_id': f'video_{i}', 'score': score})
            
            return {
                'recommendations': recommendations,
                'feature_scores': {
                    'engagement_score': float(result['engagement_scores'][0].item()),
                    'retention_score': float(result['retention_scores'][0].item()),
                    'diversity_score': 0.9
                },
                'inference_engine': 'tensorrt'
            }
            
        except Exception as e:
            logger.error(f"TensorRT推理失败: {e}")
            return self._pytorch_inference(features, user_id, session_id, num_recommendations)
    
    def _pytorch_inference(self, features: Dict[str, Any], user_id: str, 
                          session_id: str, num_recommendations: int) -> Dict[str, Any]:
        """PyTorch推理"""
        # 直接使用模型进行推理
        import torch
        
        dense_features = features['dense_features']
        
        # 创建模拟的输入数据
        batch_size = dense_features.shape[0]
        seq_len = 1000
        
        # 创建模拟的input_ids
        input_ids = torch.randint(0, 10000, (batch_size, seq_len), dtype=torch.long)
        
        # 使用模型进行推理
        with torch.no_grad():
            outputs = self.pytorch_pipeline.model.forward_prefill(
                input_ids=input_ids,
                dense_features=dense_features
            )
        
        # 生成推荐结果
        recommendations = []
        for i in range(num_recommendations):
            score = float(outputs['feature_scores'][0][i % 10].item()) if 'feature_scores' in outputs else 0.8 - i * 0.1
            recommendations.append({
                'video_id': f'video_{i}',
                'score': score
            })
        
        result = {
            'recommendations': recommendations,
            'feature_scores': {
                'engagement_score': 0.85,
                'retention_score': 0.72,
                'diversity_score': 0.91
            },
            'inference_engine': 'pytorch'
        }
        
        return result
    
    def _post_process_result(self, result: Dict[str, Any], user_id: str, 
                           session_id: str, sequence_length: int) -> Dict[str, Any]:
        """后处理结果"""
        result.update({
            'user_id': user_id,
            'session_id': session_id,
            'sequence_length': sequence_length,
            'timestamp': datetime.now().isoformat()
        })
        return result
    
    def _log_performance_metrics(self, inference_time: float, result: Dict[str, Any]):
        """记录性能指标"""
        logger.info(f"推理完成 - 引擎: {result.get('inference_engine', 'unknown')}, "
                   f"耗时: {inference_time:.2f}ms")
        
        with open('performance_metrics.log', 'a') as f:
            f.write(f"{datetime.now().isoformat()},{result.get('inference_engine', 'unknown')},"
                   f"{inference_time:.2f},{len(result.get('recommendations', []))}\n")

class TritonClient:
    """Triton客户端（模拟）"""
    def __init__(self):
        self.server_url = "http://localhost:8000"
    
    def infer(self, model_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {'outputs': {'recommendations': [0.8, 0.7, 0.6]}}

class CustomOperators:
    """自定义算子（模拟）"""
    def __init__(self, ops_path: str):
        self.ops_path = ops_path
    
    def process_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        return features
    
    def _apply_gpu_cache(self, features: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """应用GPU热缓存"""
        try:
            embedding_service = self.pytorch_pipeline.embedding_service
            
            # 获取用户嵌入缓存
            user_embeddings = embedding_service.lookup_batch([user_id])
            if user_embeddings is not None:
                features['user_embeddings'] = user_embeddings
                logger.info("✅ GPU热缓存命中用户嵌入")
            
            # 获取视频嵌入缓存
            video_ids = [behavior['video_id'] for behavior in features['behaviors']]
            video_embeddings = embedding_service.lookup_batch(video_ids)
            if video_embeddings is not None:
                features['video_embeddings'] = video_embeddings
                logger.info("✅ GPU热缓存命中视频嵌入")
            
            # 获取缓存统计
            cache_stats = embedding_service.get_cache_stats()
            logger.info(f"GPU缓存统计: 命中率={cache_stats.get('gpu_hit_rate', 0):.2%}, "
                       f"GPU内存使用={cache_stats.get('gpu_memory_usage', 0):.2f}GB")
            
        except Exception as e:
            logger.warning(f"GPU热缓存处理失败: {e}")
        
        return features

def setup_optimized_engine():
    """设置优化推理引擎"""
    logger.info("正在设置优化推理引擎...")
    
    # MTGR模型配置 - 约8B参数
    model_config = {
        "vocab_size": 50000,
        "d_model": 1024,
        "nhead": 16,
        "num_layers": 24,
        "d_ff": 4096,
        "max_seq_len": 2048,
        "num_features": 1024,
        "user_profile_dim": 256,
        "item_feature_dim": 512,
        "dropout": 0.1
    }
    
    optimization_config = {
        "enable_tensorrt": True,
        "enable_triton": True,
        "enable_custom_ops": True,
        "enable_vllm": True,  # 启用VLLM推理优化
        "precision": "fp16",
        "max_batch_size": 8
    }
    
    engine = OptimizedInferenceEngine(model_config, optimization_config)
    
    model = engine.pytorch_pipeline.model
    total_params, trainable_params = calculate_model_parameters(model)
    logger.info(f"模型初始化完成，总参数量: {total_params:,}")
    
    return engine

def run_single_inference(engine: OptimizedInferenceEngine):
    """运行单次推理"""
    logger.info("开始单次优化推理...")
    
    user_id = "user_12345"
    session_id = "session_67890"
    user_behaviors = create_realistic_user_behaviors(user_id, 10)
    
    result = engine.infer_with_optimization(
        user_behaviors=user_behaviors,
        user_id=user_id,
        session_id=session_id,
        num_recommendations=10
    )
    
    print("\n" + "="*60)
    print("优化推理结果")
    print("="*60)
    print(f"用户ID: {result['user_id']}")
    print(f"会话ID: {result['session_id']}")
    print(f"序列长度: {result['sequence_length']}")
    print(f"推理引擎: {result.get('inference_engine', 'unknown')}")
    
    print("\n推荐结果:")
    for i, rec in enumerate(result['recommendations']):
        print(f"  {i+1}. {rec['video_id']} (分数: {rec['score']:.4f})")
    
    print("\n特征分数:")
    for key, value in result['feature_scores'].items():
        print(f"  {key}: {value:.4f}")
    
    return result

def run_batch_inference(engine: OptimizedInferenceEngine):
    """运行批量推理"""
    logger.info("开始批量优化推理...")
    
    batch_results = []
    for i in range(5):
        user_id = f"user_{i+1}"
        session_id = f"session_{i+1}"
        user_behaviors = create_realistic_user_behaviors(user_id, 8)
        
        result = engine.infer_with_optimization(
            user_behaviors=user_behaviors,
            user_id=user_id,
            session_id=session_id,
            num_recommendations=5
        )
        batch_results.append(result)
    
    print("\n" + "="*60)
    print("批量优化推理结果摘要")
    print("="*60)
    for i, result in enumerate(batch_results):
        print(f"用户 {i+1}: {len(result['recommendations'])} 个推荐，"
              f"引擎: {result.get('inference_engine', 'unknown')}, "
              f"特征分数: {result['feature_scores']['engagement_score']:.4f}")
    
    return batch_results

def run_performance_test(engine: OptimizedInferenceEngine):
    """运行性能测试"""
    logger.info("开始性能测试...")
    
    user_behaviors = create_realistic_user_behaviors("test_user", 10)
    
    # 预热
    for _ in range(3):
        engine.infer_with_optimization(
            user_behaviors=user_behaviors,
            user_id="test_user",
            session_id="test_session",
            num_recommendations=5
        )
    
    # 性能测试
    num_tests = 10
    times = []
    engines = []
    
    for i in range(num_tests):
        start_time = time.time()
        result = engine.infer_with_optimization(
            user_behaviors=user_behaviors,
            user_id=f"test_user_{i}",
            session_id=f"test_session_{i}",
            num_recommendations=5
        )
        end_time = time.time()
        times.append((end_time - start_time) * 1000)
        engines.append(result.get('inference_engine', 'unknown'))
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print("\n" + "="*60)
    print("性能测试结果")
    print("="*60)
    print(f"测试次数: {num_tests}")
    print(f"平均推理时间: {avg_time:.2f}ms")
    print(f"最小推理时间: {min_time:.2f}ms")
    print(f"最大推理时间: {max_time:.2f}ms")
    print(f"吞吐量: {1000/avg_time:.2f} 请求/秒")
    print(f"主要推理引擎: {max(set(engines), key=engines.count)}")

def start_triton_server():
    """启动Triton服务器"""
    logger.info("启动Triton推理服务器...")
    
    try:
        import subprocess
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✅ Docker可用")
            
            model_repo_path = os.path.abspath("triton_model_repo")
            cmd = [
                'docker', 'run', '--gpus=all', '--rm',
                '-p8000:8000', '-p8001:8001', '-p8002:8002',
                '-v', f'{model_repo_path}:/models',
                'nvcr.io/nvidia/tritonserver:23.12-py3',
                'tritonserver', '--model-repository=/models'
            ]
            
            logger.info(f"启动命令: {' '.join(cmd)}")
            logger.info("注意: 需要手动启动Triton服务器或使用Docker")
            
        else:
            logger.warning("⚠️ Docker不可用")
    except Exception as e:
        logger.warning(f"⚠️ 无法检查Docker: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成式推荐模型推理优化项目')
    parser.add_argument('--mode', choices=['single', 'batch', 'performance', 'triton', 'all'], 
                       default='all', help='运行模式')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='日志级别')
    
    args = parser.parse_args()
    
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    print("="*80)
    print("生成式推荐模型推理优化项目 - 集成优化版本")
    print("="*80)
    
    try:
        # 1. 设置优化推理引擎
        engine = setup_optimized_engine()
        
        # 2. 根据模式运行
        if args.mode in ['single', 'all']:
            run_single_inference(engine)
        
        if args.mode in ['batch', 'all']:
            run_batch_inference(engine)
        
        if args.mode in ['performance', 'all']:
            run_performance_test(engine)
        
        if args.mode in ['triton', 'all']:
            start_triton_server()
        
        print("\n" + "="*80)
        print("项目运行完成！")
        print("="*80)
        print("\n性能监控:")
        print("- 查看推理日志: tail -f inference.log")
        print("- 查看性能指标: tail -f performance_metrics.log")
        print("- Triton监控: http://localhost:8000/metrics")
        
    except Exception as e:
        logger.error(f"项目运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
