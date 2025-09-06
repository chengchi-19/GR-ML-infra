#!/usr/bin/env python3
"""
基于开源框架的生成式推荐模型推理优化项目

集成了真正的开源框架：
- Meta HSTU (Hierarchical Sequential Transduction Units) 生成式推荐模型
- VLLM 推理优化框架 (PagedAttention + Continuous Batching)  
- TensorRT GPU推理加速
- 自定义Triton和CUTLASS算子优化
- 智能GPU热缓存系统

不再使用手写的模拟实现，而是基于真正的开源技术栈。
"""

import sys
import os
import json
import logging
import argparse
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入集成控制器
from integrations.framework_controller import OpenSourceFrameworkController, create_integrated_controller

# 导入示例数据生成
from examples.client_example import create_realistic_user_behaviors

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('opensoure_inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_optimized_config():
    """创建优化配置"""
    
    config = {
        # Meta HSTU模型配置
        'hstu': {
            'vocab_size': 50000,          # 词汇表大小
            'd_model': 1024,              # 模型隐藏维度
            'num_layers': 12,             # 层数 (约3.2B参数)
            'num_heads': 16,              # 注意力头数
            'd_ff': 4096,                 # FFN维度
            'max_seq_len': 2048,          # 最大序列长度
            'dropout': 0.1,               # Dropout率
            'hstu_expansion_factor': 4,   # HSTU扩展因子
            'hstu_gate_type': 'sigmoid',  # 门控类型
            'enable_hierarchical_attention': True,  # 启用分层注意力
            'similarity_dim': 256,        # 相似度计算维度
            'temperature': 0.1,           # 温度参数
            'pretrained_path': None,      # 预训练模型路径(如果有)
        },
        
        # VLLM推理优化配置
        'vllm': {
            'model_name': 'hstu-generative-recommender',
            'model_path': None,           # 如果有本地模型路径
            'tensor_parallel_size': 1,    # 张量并行大小
            'pipeline_parallel_size': 1,  # 流水线并行大小
            'gpu_memory_utilization': 0.85,  # GPU内存利用率
            'max_model_len': 2048,        # 最大模型长度
            'max_num_seqs': 256,          # 最大并发序列数
            'max_num_batched_tokens': None,  # 最大批处理token数
            'block_size': 16,             # PagedAttention块大小
            'dtype': 'float16',           # 数据类型
            'seed': 42,                   # 随机种子
            'quantization': None,         # 量化方式 (None, 'gptq', 'awq')
            'enable_chunked_prefill': True,  # 启用分块预填充
        },
        
        # TensorRT推理加速配置
        'tensorrt': {
            'model_name': 'hstu-tensorrt-optimized',
            'onnx_path': None,            # ONNX模型路径
            'engine_path': 'models/hstu_fp16.trt',  # TensorRT引擎路径
            'precision': 'fp16',          # 精度模式 ('fp32', 'fp16', 'int8')
            'max_batch_size': 8,          # 最大批处理大小
            'max_workspace_size': 2 << 30,  # 最大工作空间 (2GB)
            'optimization_level': 5,      # 优化等级 (0-5)
            'enable_dynamic_shapes': True,  # 启用动态形状
            'enable_strict_types': False, # 启用严格类型
            'enable_fp16_io': True,       # 启用FP16 I/O
            # 动态形状配置
            'min_shapes': {
                'input_ids': (1, 8),
                'attention_mask': (1, 8),
                'dense_features': (1, 1024),
            },
            'opt_shapes': {
                'input_ids': (4, 64),
                'attention_mask': (4, 64), 
                'dense_features': (4, 1024),
            },
            'max_shapes': {
                'input_ids': (8, 2048),
                'attention_mask': (8, 2048),
                'dense_features': (8, 1024),
            },
        },
        
        # 自定义算子优化配置
        'custom_operators': {
            'cache_size': 2000,           # 算子缓存大小
            'enable_benchmarking': True,  # 启用性能基准测试
            'triton_block_size': 64,      # Triton块大小
            'enable_cutlass': True,       # 启用CUTLASS算子
            'fusion_threshold': 32,       # 算子融合阈值
        },
        
        # 智能GPU热缓存配置
        'intelligent_cache': {
            'gpu_cache_size': 8192,       # GPU缓存大小
            'embedding_dim': 1024,        # 嵌入维度
            'enable_prediction': True,    # 启用热点预测
            'dtype': 'float32',           # 数据类型
            'prediction_window': 1000,    # 预测窗口大小
            'decay_factor': 0.95,         # 衰减因子
        },
        
        # 推理策略配置
        'inference_strategy': {
            'auto_selection': True,       # 自动策略选择
            'vllm_sequence_threshold': 100,    # VLLM序列长度阈值
            'tensorrt_sequence_threshold': 50, # TensorRT序列长度阈值
            'fallback_strategy': 'hstu',  # 回退策略
            'enable_batching': True,      # 启用批处理
            'batch_timeout_ms': 50,       # 批处理超时
        },
        
        # 性能监控配置
        'monitoring': {
            'enable_detailed_logging': True,  # 启用详细日志
            'log_inference_time': True,   # 记录推理时间
            'log_memory_usage': True,     # 记录内存使用
            'benchmark_interval': 100,    # 基准测试间隔
            'save_performance_metrics': True,  # 保存性能指标
        }
    }
    
    return config

class OpenSourceRecommenderSystem:
    """基于开源框架的推荐系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_metrics = defaultdict(list)
        
        # 创建集成控制器
        logger.info("初始化基于开源框架的推荐系统...")
        self.controller = create_integrated_controller(config)
        
        # 检查框架可用性
        self._check_framework_availability()
        
        logger.info("✅ 开源推荐系统初始化完成")
    
    def _check_framework_availability(self):
        """检查框架可用性"""
        availability = self.controller.framework_availability
        
        logger.info("开源框架可用性检查:")
        logger.info(f"  Meta HSTU模型: {'✅' if availability['hstu'] else '❌'}")
        logger.info(f"  VLLM推理引擎: {'✅' if availability['vllm'] else '❌'}")
        logger.info(f"  TensorRT加速: {'✅' if availability['tensorrt'] else '❌'}")
        logger.info(f"  Triton算子: {'✅' if availability['triton_ops'] else '❌'}")
        logger.info(f"  智能缓存: {'✅' if availability['cache'] else '❌'}")
        
        available_count = sum(availability.values())
        total_count = len(availability)
        logger.info(f"总体可用性: {available_count}/{total_count} ({available_count/total_count:.1%})")
    
    def generate_recommendations(
        self,
        user_id: str,
        session_id: str,
        user_behaviors: List[Dict[str, Any]],
        num_recommendations: int = 10,
        strategy: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """生成推荐结果"""
        
        logger.info(f"为用户 {user_id} 生成 {num_recommendations} 个推荐 (策略: {strategy})")
        
        # 使用集成控制器进行推理
        result = self.controller.infer_with_optimal_strategy(
            user_id=user_id,
            session_id=session_id,
            user_behaviors=user_behaviors,
            num_recommendations=num_recommendations,
            strategy=strategy,
            **kwargs
        )
        
        # 记录性能指标
        if 'inference_time_ms' in result:
            self.performance_metrics['inference_times'].append(result['inference_time_ms'])
            self.performance_metrics['strategies'].append(result.get('inference_strategy', 'unknown'))
        
        return result
    
    async def batch_generate_recommendations(
        self,
        requests: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """批量生成推荐"""
        
        logger.info(f"批量生成推荐，请求数量: {len(requests)}")
        
        start_time = time.time()
        results = await self.controller.batch_infer(requests, **kwargs)
        batch_time = time.time() - start_time
        
        logger.info(f"批量推理完成，总耗时: {batch_time:.2f}s，平均每请求: {batch_time/len(requests)*1000:.2f}ms")
        
        return results
    
    def benchmark_performance(
        self,
        test_cases: List[Dict[str, Any]],
        strategies: List[str] = ['auto', 'hstu', 'vllm', 'tensorrt'],
        iterations: int = 50
    ) -> Dict[str, Any]:
        """性能基准测试"""
        
        logger.info(f"开始性能基准测试，策略: {strategies}，迭代次数: {iterations}")
        
        benchmark_results = {}
        
        for strategy in strategies:
            logger.info(f"测试策略: {strategy}")
            times = []
            success_count = 0
            
            for test_case in test_cases:
                for _ in range(iterations):
                    start_time = time.time()
                    
                    result = self.generate_recommendations(
                        user_id=f"bench_user_{success_count}",
                        session_id=f"bench_session_{success_count}",
                        user_behaviors=test_case.get('user_behaviors', []),
                        num_recommendations=test_case.get('num_recommendations', 10),
                        strategy=strategy
                    )
                    
                    end_time = time.time()
                    
                    if 'error' not in result:
                        times.append((end_time - start_time) * 1000)  # 转换为毫秒
                        success_count += 1
            
            if times:
                benchmark_results[strategy] = {
                    'avg_latency_ms': np.mean(times),
                    'min_latency_ms': np.min(times),
                    'max_latency_ms': np.max(times),
                    'p50_latency_ms': np.percentile(times, 50),
                    'p95_latency_ms': np.percentile(times, 95),
                    'p99_latency_ms': np.percentile(times, 99),
                    'std_latency_ms': np.std(times),
                    'throughput_rps': 1000 / np.mean(times) if times else 0,
                    'success_rate': success_count / (len(test_cases) * iterations),
                    'total_tests': len(test_cases) * iterations,
                    'successful_tests': success_count,
                }
                
                logger.info(f"策略 {strategy} 结果:")
                logger.info(f"  平均延迟: {benchmark_results[strategy]['avg_latency_ms']:.2f}ms")
                logger.info(f"  P95延迟: {benchmark_results[strategy]['p95_latency_ms']:.2f}ms") 
                logger.info(f"  吞吐量: {benchmark_results[strategy]['throughput_rps']:.2f} RPS")
                logger.info(f"  成功率: {benchmark_results[strategy]['success_rate']:.2%}")
        
        return benchmark_results
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """获取综合统计信息"""
        
        # 获取控制器统计
        controller_stats = self.controller.get_comprehensive_stats()
        
        # 添加系统级统计
        system_stats = {
            'system_info': {
                'python_version': sys.version,
                'cuda_available': self._check_cuda_availability(),
                'gpu_count': self._get_gpu_count(),
                'memory_usage': self._get_memory_usage(),
            },
            'performance_metrics': dict(self.performance_metrics),
            'uptime': time.time(),  # 可以添加启动时间跟踪
        }
        
        return {
            'controller_stats': controller_stats,
            'system_stats': system_stats,
            'timestamp': datetime.now().isoformat(),
        }
    
    def _check_cuda_availability(self) -> bool:
        """检查CUDA可用性"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _get_gpu_count(self) -> int:
        """获取GPU数量"""
        try:
            import torch
            return torch.cuda.device_count() if torch.cuda.is_available() else 0
        except ImportError:
            return 0
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        try:
            import psutil
            import torch
            
            # 系统内存
            system_memory = psutil.virtual_memory()
            
            memory_info = {
                'system_memory_total_gb': system_memory.total / (1024**3),
                'system_memory_used_gb': system_memory.used / (1024**3),
                'system_memory_percent': system_memory.percent,
            }
            
            # GPU内存
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory
                    gpu_allocated = torch.cuda.memory_allocated(i)
                    gpu_reserved = torch.cuda.memory_reserved(i)
                    
                    memory_info[f'gpu_{i}_total_gb'] = gpu_memory / (1024**3)
                    memory_info[f'gpu_{i}_allocated_gb'] = gpu_allocated / (1024**3)
                    memory_info[f'gpu_{i}_reserved_gb'] = gpu_reserved / (1024**3)
                    memory_info[f'gpu_{i}_utilization'] = gpu_allocated / gpu_memory
            
            return memory_info
            
        except ImportError:
            return {'memory_info': 'psutil not available'}

def run_single_inference_demo():
    """运行单次推理演示"""
    logger.info("🚀 开始单次推理演示...")
    
    # 创建配置
    config = create_optimized_config()
    
    # 创建推荐系统
    recommender = OpenSourceRecommenderSystem(config)
    
    # 创建测试数据
    user_behaviors = create_realistic_user_behaviors("demo_user", 15)
    
    # 生成推荐
    result = recommender.generate_recommendations(
        user_id="demo_user_001",
        session_id="demo_session_001", 
        user_behaviors=user_behaviors,
        num_recommendations=10,
        strategy="unified"
    )
    
    print("\n" + "="*80)
    print("单次推理演示结果")
    print("="*80)
    print(f"用户ID: {result.get('user_id', 'unknown')}")
    print(f"推理策略: {result.get('inference_strategy', 'unknown')}")
    print(f"推理时间: {result.get('inference_time_ms', 0):.2f}ms")
    print(f"引擎类型: {result.get('engine_type', 'unknown')}")
    
    if 'recommendations' in result:
        print(f"\n📝 生成了 {len(result['recommendations'])} 个推荐:")
        for i, rec in enumerate(result['recommendations'][:5]):  # 只显示前5个
            print(f"  {i+1}. {rec.get('video_id', 'unknown')} "
                  f"(分数: {rec.get('score', 0):.4f}) - {rec.get('reason', '')}")
    
    if 'error' in result:
        print(f"\n❌ 推理出错: {result['error']}")
    
    return result

def run_batch_inference_demo():
    """运行批量推理演示"""
    logger.info("🚀 开始批量推理演示...")
    
    config = create_optimized_config()
    recommender = OpenSourceRecommenderSystem(config)
    
    # 创建批量请求
    batch_requests = []
    strategies = ['unified', 'hstu', 'vllm', 'tensorrt']  # 主要使用unified流程
    
    for i in range(8):  # 8个并发请求
        user_behaviors = create_realistic_user_behaviors(f"batch_user_{i}", 10)
        strategy = strategies[i % len(strategies)]
        
        batch_requests.append({
            'user_id': f'batch_user_{i}',
            'session_id': f'batch_session_{i}',
            'user_behaviors': user_behaviors,
            'num_recommendations': 5,
            'strategy': strategy
        })
    
    # 执行批量推理
    async def run_batch():
        return await recommender.batch_generate_recommendations(batch_requests)
    
    batch_results = asyncio.run(run_batch())
    
    print("\n" + "="*80)
    print("批量推理演示结果")
    print("="*80)
    
    strategy_stats = defaultdict(list)
    
    for i, result in enumerate(batch_results):
        strategy = result.get('inference_strategy', 'unknown')
        inference_time = result.get('inference_time_ms', 0)
        num_recs = len(result.get('recommendations', []))
        
        strategy_stats[strategy].append(inference_time)
        
        print(f"请求 {i+1}: 策略={strategy}, 时间={inference_time:.2f}ms, 推荐数={num_recs}")
    
    print("\n📊 策略性能统计:")
    for strategy, times in strategy_stats.items():
        if times:
            print(f"  {strategy}: 平均 {np.mean(times):.2f}ms, "
                  f"最小 {np.min(times):.2f}ms, 最大 {np.max(times):.2f}ms")
    
    return batch_results

def run_performance_benchmark():
    """运行性能基准测试"""
    logger.info("🚀 开始性能基准测试...")
    
    config = create_optimized_config()
    recommender = OpenSourceRecommenderSystem(config)
    
    # 创建测试用例
    test_cases = [
        {
            'user_behaviors': create_realistic_user_behaviors("bench_user_short", 5),
            'num_recommendations': 5,
            'name': 'short_sequence'
        },
        {
            'user_behaviors': create_realistic_user_behaviors("bench_user_medium", 25),
            'num_recommendations': 10,
            'name': 'medium_sequence'
        },
        {
            'user_behaviors': create_realistic_user_behaviors("bench_user_long", 100),
            'num_recommendations': 15,
            'name': 'long_sequence'
        },
    ]
    
    # 运行基准测试
    benchmark_results = recommender.benchmark_performance(
        test_cases=test_cases,
        strategies=['unified', 'hstu', 'vllm', 'tensorrt'],
        iterations=20
    )
    
    print("\n" + "="*80)
    print("性能基准测试结果")
    print("="*80)
    
    # 输出详细结果
    for strategy, metrics in benchmark_results.items():
        print(f"\n📈 策略: {strategy.upper()}")
        print(f"  平均延迟: {metrics['avg_latency_ms']:.2f}ms")
        print(f"  P95延迟: {metrics['p95_latency_ms']:.2f}ms")
        print(f"  P99延迟: {metrics['p99_latency_ms']:.2f}ms")
        print(f"  吞吐量: {metrics['throughput_rps']:.2f} RPS")
        print(f"  成功率: {metrics['success_rate']:.2%}")
        print(f"  标准差: {metrics['std_latency_ms']:.2f}ms")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"benchmark_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"基准测试结果已保存到: {results_file}")
    
    return benchmark_results

def run_comprehensive_demo():
    """运行综合演示"""
    logger.info("🚀 开始综合演示...")
    
    config = create_optimized_config()
    recommender = OpenSourceRecommenderSystem(config)
    
    print("\n" + "="*80)
    print("开源框架集成推荐系统 - 综合演示")
    print("="*80)
    
    # 1. 单次推理演示
    print("\n🔸 1. 单次推理演示")
    single_result = run_single_inference_demo()
    
    # 2. 批量推理演示
    print("\n🔸 2. 批量推理演示")
    batch_results = run_batch_inference_demo()
    
    # 3. 性能基准测试
    print("\n🔸 3. 性能基准测试")
    benchmark_results = run_performance_benchmark()
    
    # 4. 综合统计信息
    print("\n🔸 4. 系统统计信息")
    stats = recommender.get_comprehensive_stats()
    
    print("\n📊 框架可用性:")
    framework_availability = stats['controller_stats']['framework_availability']
    for framework, available in framework_availability.items():
        print(f"  {framework}: {'✅ 可用' if available else '❌ 不可用'}")
    
    print(f"\n💻 系统信息:")
    system_info = stats['system_stats']['system_info']
    print(f"  CUDA可用: {'✅' if system_info['cuda_available'] else '❌'}")
    print(f"  GPU数量: {system_info['gpu_count']}")
    
    if 'system_memory_total_gb' in system_info['memory_usage']:
        memory_info = system_info['memory_usage']
        print(f"  系统内存: {memory_info['system_memory_used_gb']:.1f}GB / {memory_info['system_memory_total_gb']:.1f}GB")
    
    print(f"\n🏆 总推理次数: {stats['controller_stats']['total_inferences']}")
    
    return {
        'single_result': single_result,
        'batch_results': batch_results,
        'benchmark_results': benchmark_results,
        'system_stats': stats,
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于开源框架的生成式推荐模型推理优化项目')
    parser.add_argument('--mode', 
                       choices=['single', 'batch', 'benchmark', 'comprehensive'], 
                       default='comprehensive',
                       help='运行模式')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--log-level', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', 
                       help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    print("🌟" * 40)
    print("基于开源框架的生成式推荐模型推理优化项目")
    print("🌟" * 40)
    print("集成技术栈:")
    print("  📚 Meta HSTU (Hierarchical Sequential Transduction Units)")
    print("  ⚡ VLLM (PagedAttention + Continuous Batching)")
    print("  🚀 TensorRT (GPU Inference Acceleration)")
    print("  🔧 Custom Triton + CUTLASS Operators")
    print("  🧠 Intelligent GPU Hot Cache")
    print("🌟" * 40)
    
    try:
        # 运行指定模式
        if args.mode == 'single':
            result = run_single_inference_demo()
        elif args.mode == 'batch':
            result = run_batch_inference_demo()
        elif args.mode == 'benchmark':
            result = run_performance_benchmark()
        else:  # comprehensive
            result = run_comprehensive_demo()
        
        print("\n🎉 演示完成！")
        print("\n📋 监控和日志:")
        print("  - 推理日志: tail -f opensoure_inference.log")
        print("  - 基准测试结果: benchmark_results_*.json")
        
        return 0
        
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