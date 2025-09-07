#!/usr/bin/env python3
"""
HSTU推理优化项目 - 配置和测试工具

主要功能:
- 生成推理优化配置
- 快速系统可用性检查
- 简单的推理测试

注意: 生产环境请使用API服务 (python api_server.py)
"""

import sys
import os
import json
import logging
import argparse
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_optimized_config() -> Dict[str, Any]:
    """创建优化配置"""
    
    config = {
        # Meta HSTU模型配置
        'hstu': {
            'vocab_size': 50000,
            'd_model': 1024,
            'num_layers': 12,
            'num_heads': 16,
            'd_ff': 4096,
            'max_seq_len': 2048,
            'dropout': 0.1,
            'hstu_expansion_factor': 4,
            'hstu_gate_type': 'sigmoid',
            'enable_hierarchical_attention': True,
            'similarity_dim': 256,
            'temperature': 0.1,
            'pretrained_path': None,
        },
        
        # VLLM推理优化配置
        'vllm': {
            'model_name': 'hstu-generative-recommender',
            'model_path': None,
            'tensor_parallel_size': 1,
            'pipeline_parallel_size': 1,
            'gpu_memory_utilization': 0.85,
            'max_model_len': 2048,
            'max_num_seqs': 256,
            'max_num_batched_tokens': None,
            'block_size': 16,
            'dtype': 'float16',
            'seed': 42,
            'quantization': None,
            'enable_chunked_prefill': True,
        },
        
        # TensorRT推理加速配置
        'tensorrt': {
            'model_name': 'hstu-tensorrt-optimized',
            'onnx_path': None,
            'engine_path': 'models/hstu_fp16.trt',
            'precision': 'fp16',
            'max_batch_size': 8,
            'max_workspace_size': 2 << 30,  # 2GB
            'optimization_level': 5,
            'enable_dynamic_shapes': True,
            'enable_strict_types': False,
            'enable_fp16_io': True,
            'min_shapes': {
                'input_ids': (1, 8),
                'attention_mask': (1, 8),
                'dense_features': (1, 1024),
            },
            'opt_shapes': {
                'input_ids': (4, 128),
                'attention_mask': (4, 128),
                'dense_features': (4, 1024),
            },
            'max_shapes': {
                'input_ids': (8, 2048),
                'attention_mask': (8, 2048),
                'dense_features': (8, 1024),
            },
        },
        
        # 自定义算子优化配置
        'optimizations': {
            'enable_triton_ops': True,
            'enable_cutlass_ops': True,
            'enable_intelligent_cache': True,
            'cache_size': 8192,
            'triton_operators': {
                'enable_fused_attention_layernorm': True,
                'enable_hierarchical_sequence_fusion': True,
                'enable_hstu_hierarchical_attention': True,
                'enable_sequence_recommendation_interaction': True,
                'enable_interaction_operator': True,
            }
        }
    }
    
    return config

def check_system_availability():
    """检查系统可用性"""
    logger.info("🔍 检查系统框架可用性...")
    
    try:
        from integrations.framework_controller import create_integrated_controller
        
        # 创建控制器并检查可用性
        controller = create_integrated_controller()
        availability = controller.framework_availability
        
        logger.info("📊 框架可用性状态:")
        logger.info(f"  Meta HSTU模型: {'✅' if availability.get('hstu', False) else '❌'}")
        logger.info(f"  VLLM推理引擎: {'✅' if availability.get('vllm', False) else '❌'}")
        logger.info(f"  TensorRT加速: {'✅' if availability.get('tensorrt', False) else '❌'}")
        logger.info(f"  Triton算子: {'✅' if availability.get('triton_ops', False) else '❌'}")
        logger.info(f"  智能缓存: {'✅' if availability.get('cache', False) else '❌'}")
        
        available_count = sum(availability.values())
        total_count = len(availability)
        success_rate = available_count / total_count if total_count > 0 else 0
        
        logger.info(f"📈 总体可用性: {available_count}/{total_count} ({success_rate:.1%})")
        
        if success_rate >= 0.8:
            logger.info("✅ 系统状态良好，建议启动API服务")
        elif success_rate >= 0.5:
            logger.info("⚠️ 系统部分可用，API服务可启动但性能可能受限")
        else:
            logger.info("❌ 系统可用性较低，建议检查环境配置")
            
        return controller
        
    except Exception as e:
        logger.error(f"❌ 系统检查失败: {e}")
        logger.info("💡 请检查项目依赖和环境配置")
        return None

def run_simple_test():
    """运行简单测试"""
    logger.info("🧪 运行简单推理测试...")
    
    try:
        controller = check_system_availability()
        if controller is None:
            return False
            
        # 创建测试数据
        test_behaviors = [
            {
                'video_id': 12345,
                'timestamp': 1700000000,
                'interaction_type': 'view',
                'duration': 120.5,
                'device_type': 'mobile'
            }
        ]
        
        # 执行测试推理
        result = controller.infer_with_optimal_strategy(
            user_id="test_user_001",
            session_id="test_session_001",
            user_behaviors=test_behaviors,
            num_recommendations=5,
            requested_strategy="unified"
        )
        
        if 'error' not in result:
            logger.info("✅ 简单推理测试通过")
            logger.info(f"📊 推理策略: {result.get('strategy_used', 'unknown')}")
            logger.info(f"⏱️ 推理耗时: {result.get('inference_time_ms', 0):.2f}ms")
            return True
        else:
            logger.error(f"❌ 推理测试失败: {result.get('error', 'unknown')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 测试执行失败: {e}")
        return False

def save_config_to_file(config: Dict[str, Any], filename: str = "config.json"):
    """保存配置到文件"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ 配置已保存到: {filename}")
        return True
    except Exception as e:
        logger.error(f"❌ 保存配置失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='HSTU推理优化项目 - 配置和测试工具')
    parser.add_argument('--action', 
                       choices=['config', 'check', 'test'], 
                       default='check',
                       help='操作模式: config(生成配置), check(检查可用性), test(运行测试)')
    parser.add_argument('--config-file', 
                       default='config.json',
                       help='配置文件路径')
    parser.add_argument('--log-level', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', 
                       help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    print("🌟" * 50)
    print("HSTU推理优化项目 - 配置和测试工具")
    print("🌟" * 50)
    print("集成技术栈:")
    print("  📚 Meta HSTU (Hierarchical Sequential Transduction Units)")
    print("  ⚡ VLLM (PagedAttention + Continuous Batching)")
    print("  🚀 TensorRT (GPU Inference Acceleration)")
    print("  🔧 Custom Triton + CUTLASS Operators")
    print("  🧠 Intelligent GPU Hot Cache")
    print("")
    print("💡 生产环境推荐使用API服务:")
    print("   python api_server.py")
    print("   访问 http://localhost:8000/docs 查看API文档")
    print("🌟" * 50)
    
    success = False
    
    try:
        if args.action == 'config':
            # 生成配置文件
            logger.info("⚙️ 生成推理优化配置...")
            config = create_optimized_config()
            success = save_config_to_file(config, args.config_file)
            
        elif args.action == 'check':
            # 检查系统可用性
            controller = check_system_availability()
            success = controller is not None
            
        elif args.action == 'test':
            # 运行简单测试
            success = run_simple_test()
        
        if success:
            print(f"\n🎉 {args.action} 操作完成！")
        else:
            print(f"\n❌ {args.action} 操作失败！")
            
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("用户中断，正在退出...")
        return 1
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())