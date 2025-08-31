#!/usr/bin/env python3
"""
生成式推荐模型推理优化项目 - 主入口文件
实现完整的推理流程，从用户行为数据到推荐结果
"""

import sys
import os
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.inference_pipeline import UserBehaviorInferencePipeline
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

def setup_model_and_pipeline():
    """设置模型和推理流水线"""
    logger.info("正在初始化模型和推理流水线...")
    
    # 模型配置
    model_config = {
        "vocab_size": 10000,
        "embedding_dim": 512,
        "hidden_dim": 1024,
        "num_features": 1024,  # 扩展的1024维特征
        "num_layers": 6,
        "max_seq_len": 2048
    }
    
    # 创建推理流水线
    pipeline = UserBehaviorInferencePipeline(
        model_config=model_config,
        cache_config={
            "enable_cache": True,
            "cache_size": 10000,
            "cache_ttl": 3600
        }
    )
    
    # 计算模型参数
    model = pipeline.model
    total_params, trainable_params = calculate_model_parameters(model)
    logger.info(f"模型初始化完成，总参数量: {total_params:,}")
    
    return pipeline

def create_sample_data():
    """创建示例用户行为数据"""
    logger.info("正在创建示例用户行为数据...")
    
    # 创建用户行为处理器
    processor = UserBehaviorProcessor()
    
    # 生成示例用户行为
    user_behaviors = create_realistic_user_behaviors("demo_user", 5)
    
    logger.info(f"创建了 {len(user_behaviors)} 个用户行为")
    return user_behaviors

def run_single_inference(pipeline: UserBehaviorInferencePipeline, user_behaviors: List[Dict[str, Any]]):
    """运行单次推理"""
    logger.info("开始单次推理...")
    
    # 用户信息
    user_id = "user_12345"
    session_id = "session_67890"
    
    # 执行推理
    start_time = datetime.now()
    result = pipeline.infer_recommendations(
        user_id=user_id,
        session_id=session_id,
        behaviors=user_behaviors,
        num_recommendations=10
    )
    end_time = datetime.now()
    
    # 计算推理时间
    inference_time = (end_time - start_time).total_seconds() * 1000
    
    logger.info(f"推理完成，耗时: {inference_time:.2f}ms")
    
    # 打印结果
    print("\n" + "="*60)
    print("推理结果")
    print("="*60)
    print(f"用户ID: {result['user_id']}")
    print(f"会话ID: {result['session_id']}")
    print(f"序列长度: {result['sequence_length']}")
    print(f"推理时间: {inference_time:.2f}ms")
    
    print("\n推荐结果:")
    for i, rec in enumerate(result['recommendations']):
        print(f"  {i+1}. {rec['video_id']} (分数: {rec['score']:.4f})")
    
    print("\n特征分数:")
    for key, value in result['feature_scores'].items():
        print(f"  {key}: {value:.4f}")
    
    return result

def run_batch_inference(pipeline: UserBehaviorInferencePipeline):
    """运行批量推理"""
    logger.info("开始批量推理...")
    
    # 创建多个用户的请求
    batch_requests = []
    for i in range(5):
        user_behaviors = create_realistic_user_behaviors(f"user_{i+1}", 5)
        batch_requests.append({
            "user_id": f"user_{i+1}",
            "session_id": f"session_{i+1}",
            "behaviors": user_behaviors
        })
    
    # 执行批量推理
    start_time = datetime.now()
    results = pipeline.batch_infer(
        user_requests=batch_requests,
        num_recommendations=5
    )
    end_time = datetime.now()
    
    # 计算总时间
    total_time = (end_time - start_time).total_seconds() * 1000
    avg_time = total_time / len(results)
    
    logger.info(f"批量推理完成，总耗时: {total_time:.2f}ms，平均: {avg_time:.2f}ms/用户")
    
    # 打印批量结果摘要
    print("\n" + "="*60)
    print("批量推理结果摘要")
    print("="*60)
    for i, result in enumerate(results):
        if 'error' not in result:
            print(f"用户 {i+1}: {len(result['recommendations'])} 个推荐，"
                  f"特征分数: {result['feature_scores']['engagement_score']:.4f}")
        else:
            print(f"用户 {i+1}: 推理失败 - {result['error']}")
    
    return results

def export_models():
    """导出ONNX模型"""
    logger.info("正在导出ONNX模型...")
    
    try:
        # 导入导出模块
        from src.export_onnx import GenerativeRecommendationModel
        
        # 创建模型
        model = GenerativeRecommendationModel(
            vocab_size=10000,
            embedding_dim=512,
            hidden_dim=1024,
            num_features=1024,
            num_layers=6,
            max_seq_len=2048
        )
        model.eval()
        
        # 导出模型
        import torch
        import torch.onnx
        
        # 创建示例数据
        dummy_ids = torch.randint(0, 10000, (1, 1000), dtype=torch.long)
        dummy_dense = torch.randn(1, 1024, dtype=torch.float32)
        dummy_user = torch.randn(1, 256, dtype=torch.float32)
        dummy_video = torch.randn(1, 512, dtype=torch.float32)
        dummy_mask = torch.ones(1, 1000, dtype=torch.long)
        
        # 导出prefill模型
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
        
        logger.info("ONNX模型导出完成")
        return True
        
    except Exception as e:
        logger.error(f"ONNX模型导出失败: {e}")
        return False

def run_performance_test(pipeline: UserBehaviorInferencePipeline):
    """运行性能测试"""
    logger.info("开始性能测试...")
    
    # 创建测试数据
    test_behaviors = create_realistic_user_behaviors("test_user", 5)
    
    # 预热
    for _ in range(3):
        pipeline.infer_recommendations(
            user_id="test_user",
            session_id="test_session",
            behaviors=test_behaviors,
            num_recommendations=5
        )
    
    # 性能测试
    import time
    num_tests = 10
    times = []
    
    for i in range(num_tests):
        start_time = time.time()
        result = pipeline.infer_recommendations(
            user_id=f"test_user_{i}",
            session_id=f"test_session_{i}",
            behaviors=test_behaviors,
            num_recommendations=5
        )
        end_time = time.time()
        times.append((end_time - start_time) * 1000)
    
    # 计算统计信息
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

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成式推荐模型推理优化项目')
    parser.add_argument('--mode', choices=['single', 'batch', 'export', 'performance', 'all'], 
                       default='all', help='运行模式')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    print("="*80)
    print("生成式推荐模型推理优化项目")
    print("="*80)
    
    try:
        # 1. 设置模型和推理流水线
        pipeline = setup_model_and_pipeline()
        
        # 2. 创建示例数据
        user_behaviors = create_sample_data()
        
        # 3. 根据模式运行
        if args.mode in ['single', 'all']:
            run_single_inference(pipeline, user_behaviors)
        
        if args.mode in ['batch', 'all']:
            run_batch_inference(pipeline)
        
        if args.mode in ['performance', 'all']:
            run_performance_test(pipeline)
        
        if args.mode in ['export', 'all']:
            # 创建models目录
            os.makedirs('models', exist_ok=True)
            export_models()
        
        print("\n" + "="*80)
        print("项目运行完成！")
        print("="*80)
        
    except Exception as e:
        logger.error(f"项目运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
