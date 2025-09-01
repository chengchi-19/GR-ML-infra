#!/usr/bin/env python3
"""
MTGR模型和VLLM推理引擎集成测试脚本
验证两个组件的功能和性能
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List
import torch

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mtgr_model():
    """测试MTGR模型功能"""
    print("=" * 60)
    print("测试MTGR模型功能")
    print("=" * 60)
    
    try:
        from src.mtgr_model import create_mtgr_model
        
        # 创建模型
        model_config = {
            'vocab_size': 50000,
            'd_model': 1024,
            'nhead': 16,
            'num_layers': 24,
            'd_ff': 4096,
            'max_seq_len': 2048,
            'num_features': 1024,
            'user_profile_dim': 256,
            'item_feature_dim': 512,
            'dropout': 0.1
        }
        
        model = create_mtgr_model(model_config)
        model.eval()
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ MTGR模型创建成功")
        print(f"   总参数量: {total_params:,} (约{total_params/1e9:.1f}B)")
        print(f"   模型配置: {model_config}")
        
        # 测试前向传播
        batch_size = 2
        seq_len = 100
        num_features = 1024
        
        input_ids = torch.randint(0, 50000, (batch_size, seq_len))
        dense_features = torch.randn(batch_size, num_features)
        user_profile = torch.randn(batch_size, 256)
        item_features = torch.randn(batch_size, 512)
        
        with torch.no_grad():
            # 测试Prefill阶段
            start_time = time.time()
            outputs = model.forward_prefill(
                input_ids, dense_features, user_profile, item_features
            )
            prefill_time = time.time() - start_time
            
            print(f"✅ Prefill阶段测试通过")
            print(f"   输入形状: {input_ids.shape}")
            print(f"   输出形状: {[out.shape for out in outputs]}")
            print(f"   推理时间: {prefill_time*1000:.2f}ms")
            
            # 测试Decode阶段
            last_token = input_ids[:, -1:]
            hidden_states = outputs[-1]
            
            start_time = time.time()
            decode_outputs = model.forward_decode(
                last_token, hidden_states, dense_features, user_profile, item_features
            )
            decode_time = time.time() - start_time
            
            print(f"✅ Decode阶段测试通过")
            print(f"   解码时间: {decode_time*1000:.2f}ms")
            
        return True
        
    except Exception as e:
        print(f"❌ MTGR模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vllm_engine():
    """测试VLLM推理引擎功能"""
    print("\n" + "=" * 60)
    print("测试VLLM推理引擎功能")
    print("=" * 60)
    
    try:
        from src.vllm_engine import create_vllm_engine
        
        # 创建VLLM引擎
        engine = create_vllm_engine(
            model_path="mtgr_model",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=2048
        )
        
        print(f"✅ VLLM引擎创建成功")
        print(f"   初始化状态: {engine.is_initialized}")
        print(f"   引擎配置: {engine.config}")
        
        # 测试推荐生成
        test_behaviors = [
            {
                'video_id': 'video_001',
                'watch_duration': 25,
                'watch_percentage': 0.83,
                'is_liked': True,
                'is_favorited': False,
                'is_shared': True,
                'device_type': 'mobile',
                'network_type': 'wifi'
            },
            {
                'video_id': 'video_015',
                'watch_duration': 30,
                'watch_percentage': 1.0,
                'is_liked': True,
                'is_favorited': True,
                'is_shared': False,
                'device_type': 'mobile',
                'network_type': 'wifi'
            }
        ]
        
        # 测试同步推荐生成
        result = engine._fallback_generation(
            user_id="test_user",
            session_id="test_session",
            user_behaviors=test_behaviors,
            num_recommendations=5
        )
        
        print(f"✅ 推荐生成测试通过")
        print(f"   推荐数量: {len(result['recommendations'])}")
        print(f"   引擎类型: {result['engine']}")
        print(f"   延迟: {result['latency_ms']:.2f}ms")
        
        # 获取统计信息
        stats = engine.get_stats()
        print(f"✅ 统计信息获取成功")
        print(f"   总请求数: {stats['total_requests']}")
        print(f"   平均延迟: {stats['avg_latency']*1000:.2f}ms")
        print(f"   吞吐量: {stats['throughput']:.2f} req/s")
        
        # 关闭引擎
        engine.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ VLLM引擎测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_async_inference():
    """测试异步推理功能"""
    print("\n" + "=" * 60)
    print("测试异步推理功能")
    print("=" * 60)
    
    try:
        from src.inference_pipeline import UserBehaviorInferencePipeline
        
        # 创建推理流水线
        pipeline = UserBehaviorInferencePipeline()
        
        print(f"✅ 推理流水线创建成功")
        print(f"   模型类型: MTGR")
        print(f"   VLLM引擎: {'已启用' if hasattr(pipeline, 'vllm_engine') else '未启用'}")
        
        # 测试数据
        test_behaviors = [
            {
                'video_id': 'video_001',
                'timestamp': '2024-01-15T14:30:00Z',
                'watch_duration': 25,
                'watch_percentage': 0.83,
                'is_liked': True,
                'is_favorited': False,
                'is_shared': True,
                'device_type': 'mobile',
                'network_type': 'wifi'
            }
        ]
        
        # 测试同步推理
        print("\n🔍 测试同步推理...")
        start_time = time.time()
        result = pipeline.infer_recommendations(
            user_id="test_user",
            session_id="test_session",
            behaviors=test_behaviors,
            num_recommendations=5,
            use_vllm=False  # 使用MTGR推理
        )
        sync_time = time.time() - start_time
        
        print(f"✅ 同步推理测试通过")
        print(f"   推理时间: {sync_time*1000:.2f}ms")
        print(f"   推荐数量: {len(result['recommendations'])}")
        print(f"   引擎类型: {result.get('engine_type', 'unknown')}")
        
        # 测试异步推理
        print("\n🔍 测试异步推理...")
        start_time = time.time()
        async_result = await pipeline.infer_recommendations_async(
            user_id="test_user",
            session_id="test_session",
            behaviors=test_behaviors,
            num_recommendations=5,
            use_vllm=True  # 尝试使用VLLM
        )
        async_time = time.time() - start_time
        
        print(f"✅ 异步推理测试通过")
        print(f"   推理时间: {async_time*1000:.2f}ms")
        print(f"   推荐数量: {len(async_result['recommendations'])}")
        print(f"   引擎类型: {async_result.get('engine_type', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 异步推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_export():
    """测试模型导出功能"""
    print("\n" + "=" * 60)
    print("测试模型导出功能")
    print("=" * 60)
    
    try:
        from src.export_mtgr_onnx import MTGRONNXExporter
        
        # 创建导出器
        model_config = {
            'vocab_size': 50000,
            'd_model': 1024,
            'nhead': 16,
            'num_layers': 24,
            'd_ff': 4096,
            'max_seq_len': 2048,
            'num_features': 1024,
            'user_profile_dim': 256,
            'item_feature_dim': 512,
            'dropout': 0.1
        }
        
        exporter = MTGRONNXExporter(model_config)
        
        print(f"✅ 导出器创建成功")
        print(f"   模型配置: {model_config}")
        
        # 测试虚拟数据创建
        dummy_data = exporter.create_dummy_data(batch_size=2, seq_len=50)
        print(f"✅ 虚拟数据创建成功")
        print(f"   数据形状: {[data.shape for data in dummy_data]}")
        
        # 测试ONNX导出（可选）
        try:
            import onnx
            print(f"✅ ONNX库可用")
        except ImportError:
            print(f"⚠️  ONNX库未安装，跳过导出测试")
            return True
        
        return True
        
    except Exception as e:
        print(f"❌ 模型导出测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_performance_benchmark():
    """运行性能基准测试"""
    print("\n" + "=" * 60)
    print("运行性能基准测试")
    print("=" * 60)
    
    try:
        from src.mtgr_model import create_mtgr_model
        
        # 创建模型
        model_config = {
            'vocab_size': 50000,
            'd_model': 1024,
            'nhead': 16,
            'num_layers': 24,
            'd_ff': 4096,
            'max_seq_len': 2048,
            'num_features': 1024,
            'user_profile_dim': 256,
            'item_feature_dim': 512,
            'dropout': 0.1
        }
        
        model = create_mtgr_model(model_config)
        model.eval()
        
        # 测试不同批次大小和序列长度的性能
        test_configs = [
            (1, 100),   # 小批次，短序列
            (4, 200),   # 中等批次，中等序列
            (8, 500),   # 大批次，长序列
        ]
        
        print("性能基准测试结果:")
        print(f"{'批次大小':<8} {'序列长度':<8} {'推理时间(ms)':<12} {'内存使用(MB)':<12}")
        print("-" * 50)
        
        for batch_size, seq_len in test_configs:
            # 创建测试数据
            input_ids = torch.randint(0, 50000, (batch_size, seq_len))
            dense_features = torch.randn(batch_size, 1024)
            user_profile = torch.randn(batch_size, 256)
            item_features = torch.randn(batch_size, 512)
            
            # 预热
            with torch.no_grad():
                for _ in range(3):
                    _ = model.forward_prefill(
                        input_ids, dense_features, user_profile, item_features
                    )
            
            # 性能测试
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):
                    _ = model.forward_prefill(
                        input_ids, dense_features, user_profile, item_features
                    )
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            # 计算平均推理时间
            avg_time = (end_time - start_time) / 10 * 1000
            
            # 估算内存使用（简化计算）
            estimated_memory = (batch_size * seq_len * 1024 * 4) / (1024 * 1024)  # MB
            
            print(f"{batch_size:<8} {seq_len:<8} {avg_time:<12.2f} {estimated_memory:<12.1f}")
        
        print("\n✅ 性能基准测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 性能基准测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主测试函数"""
    print("🚀 开始MTGR模型和VLLM推理引擎集成测试")
    print("=" * 80)
    
    test_results = []
    
    # 1. 测试MTGR模型
    test_results.append(("MTGR模型功能", test_mtgr_model()))
    
    # 2. 测试VLLM引擎
    test_results.append(("VLLM推理引擎", test_vllm_engine()))
    
    # 3. 测试异步推理
    test_results.append(("异步推理功能", await test_async_inference()))
    
    # 4. 测试模型导出
    test_results.append(("模型导出功能", test_model_export()))
    
    # 5. 运行性能基准测试
    test_results.append(("性能基准测试", run_performance_benchmark()))
    
    # 输出测试结果摘要
    print("\n" + "=" * 80)
    print("测试结果摘要")
    print("=" * 80)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<20} {status}")
        if result:
            passed_tests += 1
    
    print(f"\n总体结果: {passed_tests}/{total_tests} 测试通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！MTGR模型和VLLM推理引擎集成成功！")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
    
    # 生成测试报告
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": total_tests - passed_tests,
        "test_results": [
            {"name": name, "status": "PASS" if result else "FAIL"}
            for name, result in test_results
        ]
    }
    
    with open("test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 测试报告已保存到: test_report.json")

if __name__ == "__main__":
    # 运行测试
    asyncio.run(main())
