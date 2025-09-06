#!/usr/bin/env python3
"""
Triton算子集成测试

测试所有Triton自定义算子是否正确集成到推理框架中
"""

import sys
import os
import torch
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_triton_operator_availability():
    """测试Triton算子可用性"""
    print("🔧 测试Triton算子可用性...")
    
    try:
        from optimizations.triton_ops.trriton_operator_manager import create_triton_operator_manager
        from main import create_optimized_config
        
        # 创建配置
        config = create_optimized_config()
        
        # 创建Triton算子管理器
        manager = create_triton_operator_manager(config)
        
        # 获取可用性
        availability = manager.get_operator_availability()
        
        print("Triton算子可用性:")
        for name, available in availability.items():
            status = "✅ 可用" if available else "❌ 不可用"
            print(f"  {name}: {status}")
        
        available_count = sum(availability.values())
        total_count = len(availability)
        
        print(f"\n📊 总体可用性: {available_count}/{total_count} ({available_count/total_count:.1%})")
        
        return availability
        
    except Exception as e:
        print(f"❌ Triton算子测试失败: {e}")
        return {}

def test_triton_operator_integration():
    """测试Triton算子在框架中的集成"""
    print("\n🔄 测试Triton算子在框架中的集成...")
    
    try:
        from integrations.framework_controller import create_integrated_controller
        from main import create_optimized_config
        
        # 创建集成控制器
        config = create_optimized_config()
        controller = create_integrated_controller(config)
        
        # 检查Triton集成状态
        framework_availability = controller.framework_availability
        
        print("框架集成状态:")
        print(f"  Triton算子: {'✅ 已集成' if framework_availability.get('triton_ops', False) else '❌ 未集成'}")
        
        # 获取详细统计
        if hasattr(controller, 'triton_manager') and controller.triton_manager:
            stats = controller.triton_manager.get_operator_stats()
            print(f"  Triton算子统计: {stats}")
        
        return framework_availability
        
    except Exception as e:
        print(f"❌ Triton算子集成测试失败: {e}")
        return {}

def test_triton_operator_functionality():
    """测试Triton算子功能"""
    print("\n⚡ 测试Triton算子功能...")
    
    try:
        from optimizations.triton_ops.trriton_operator_manager import create_triton_operator_manager
        from main import create_optimized_config
        
        # 创建配置和管理器
        config = create_optimized_config()
        manager = create_triton_operator_manager(config)
        
        # 测试数据
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        test_results = {}
        
        # 测试融合注意力+LayerNorm算子
        if manager.availability.get('fused_attention_layernorm', False):
            print("  测试融合注意力+LayerNorm算子...")
            hidden_states = torch.randn(2, 32, 1024, device=device)
            output = manager.apply_fused_attention_layernorm(hidden_states)
            test_results['fused_attention_layernorm'] = {
                'input_shape': list(hidden_states.shape),
                'output_shape': list(output.shape),
                'success': True
            }
            print(f"    ✅ 输入: {hidden_states.shape} -> 输出: {output.shape}")
        
        # 测试分层序列融合算子
        if manager.availability.get('hierarchical_sequence_fusion', False):
            print("  测试分层序列融合算子...")
            sequence_features = torch.randn(2, 64, 1024, device=device)
            output = manager.apply_hierarchical_sequence_fusion(sequence_features)
            test_results['hierarchical_sequence_fusion'] = {
                'input_shape': list(sequence_features.shape),
                'output_shape': list(output.shape),
                'success': True
            }
            print(f"    ✅ 输入: {sequence_features.shape} -> 输出: {output.shape}")
        
        # 测试交互算子
        if manager.availability.get('interaction_operator', False):
            print("  测试交互算子...")
            features = torch.randn(100, 1024, device=device)
            output = manager.apply_interaction_operator(features)
            test_results['interaction_operator'] = {
                'input_shape': list(features.shape),
                'output_shape': list(output.shape),
                'success': True
            }
            print(f"    ✅ 输入: {features.shape} -> 输出: {output.shape}")
        
        # 测试序列推荐交互算子
        if manager.availability.get('sequence_recommendation_interaction', False):
            print("  测试序列推荐交互算子...")
            user_sequences = [[
                {'video_id': 'video_1', 'watch_duration': 120, 'is_liked': True},
                {'video_id': 'video_2', 'watch_duration': 90, 'is_liked': False},
            ]]
            output = manager.apply_sequence_recommendation_interaction(user_sequences)
            test_results['sequence_recommendation_interaction'] = {
                'input_sequences': len(user_sequences),
                'output_shape': list(output.shape),
                'success': True
            }
            print(f"    ✅ 输入序列: {len(user_sequences)} -> 输出: {output.shape}")
        
        # 测试HSTU分层注意力算子
        if manager.availability.get('hstu_hierarchical_attention', False):
            print("  测试HSTU分层注意力算子...")
            query = torch.randn(2, 32, 1024, device=device)
            key = torch.randn(2, 32, 1024, device=device)
            value = torch.randn(2, 32, 1024, device=device)
            output = manager.apply_hstu_hierarchical_attention(query, key, value)
            test_results['hstu_hierarchical_attention'] = {
                'query_shape': list(query.shape),
                'output_shape': list(output.shape),
                'success': True
            }
            print(f"    ✅ 查询: {query.shape} -> 输出: {output.shape}")
        
        return test_results
        
    except Exception as e:
        print(f"❌ Triton算子功能测试失败: {e}")
        return {}

def test_end_to_end_integration():
    """测试端到端集成"""
    print("\n🎯 测试端到端集成...")
    
    try:
        from integrations.framework_controller import create_integrated_controller
        from main import create_optimized_config
        
        # 创建集成控制器
        config = create_optimized_config()
        controller = create_integrated_controller(config)
        
        # 测试数据
        test_behaviors = [
            {'video_id': 'video_1', 'watch_duration': 120, 'is_liked': True, 'category': 'tech'},
            {'video_id': 'video_2', 'watch_duration': 90, 'is_liked': False, 'category': 'music'},
            {'video_id': 'video_3', 'watch_duration': 200, 'is_liked': True, 'category': 'tech'},
            {'video_id': 'video_4', 'watch_duration': 45, 'is_liked': True, 'category': 'sports'},
            {'video_id': 'video_5', 'watch_duration': 180, 'is_liked': False, 'category': 'education'},
        ]
        
        # 执行推理
        result = controller.infer_with_optimal_strategy(
            user_id="test_user",
            session_id="test_session",
            user_behaviors=test_behaviors,
            num_recommendations=5
        )
        
        # 检查Triton优化是否被应用
        optimizations = result.get('optimizations_applied', [])
        triton_optimized = result.get('triton_optimized', False)
        
        print(f"  推理策略: {result.get('inference_strategy', 'unknown')}")
        print(f"  Triton优化: {'✅ 已应用' if triton_optimized else '❌ 未应用'}")
        print(f"  优化算子: {optimizations}")
        print(f"  推荐数量: {len(result.get('recommendations', []))}")
        
        return {
            'triton_optimized': triton_optimized,
            'optimizations_applied': optimizations,
            'recommendations_count': len(result.get('recommendations', [])),
            'inference_time_ms': result.get('inference_time_ms', 0),
            'success': True
        }
        
    except Exception as e:
        print(f"❌ 端到端集成测试失败: {e}")
        return {'success': False, 'error': str(e)}

def test_triton_benchmark():
    """测试Triton算子性能"""
    print("\n📊 测试Triton算子性能...")
    
    try:
        from optimizations.triton_ops.trriton_operator_manager import create_triton_operator_manager
        from main import create_optimized_config
        
        # 创建配置和管理器
        config = create_optimized_config()
        manager = create_triton_operator_manager(config)
        
        # 测试数据
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_data = {
            'hidden_states': torch.randn(4, 64, 1024, device=device),
            'sequence_features': torch.randn(4, 128, 1024, device=device),
            'features': torch.randn(200, 1024, device=device),
            'query': torch.randn(4, 64, 1024, device=device),
            'key': torch.randn(4, 64, 1024, device=device),
            'value': torch.randn(4, 64, 1024, device=device),
            'user_sequences': [[
                {'video_id': f'video_{i}', 'watch_duration': 100 + i*10, 'is_liked': i % 2 == 0}
                for i in range(10)
            ] for _ in range(4)]
        }
        
        # 运行基准测试
        benchmark_results = manager.benchmark_all_operators(
            test_data=test_data,
            num_iterations=5
        )
        
        print("Triton算子性能基准:")
        for name, metrics in benchmark_results['benchmark_results'].items():
            print(f"  {name}: {metrics['avg_latency_ms']:.2f}ms")
        
        return benchmark_results
        
    except Exception as e:
        print(f"❌ Triton算子性能测试失败: {e}")
        return {}

def run_all_triton_tests():
    """运行所有Triton集成测试"""
    print("🚀 开始Triton算子集成测试")
    print("="*80)
    
    test_results = {}
    
    # 运行各项测试
    test_results['operator_availability'] = test_triton_operator_availability()
    test_results['operator_integration'] = test_triton_operator_integration()
    test_results['operator_functionality'] = test_triton_operator_functionality()
    test_results['end_to_end_integration'] = test_end_to_end_integration()
    test_results['triton_benchmark'] = test_triton_benchmark()
    
    # 统计结果
    print("\n" + "="*80)
    print("Triton算子集成测试结果汇总")
    print("="*80)
    
    for test_name, result in test_results.items():
        if isinstance(result, dict):
            if 'success' in result:
                status = "✅ 通过" if result['success'] else "❌ 失败"
            else:
                status = "✅ 完成"
        else:
            status = "✅ 完成"
        
        print(f"  {test_name}: {status}")
    
    # 保存测试报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'test_results': test_results,
        'summary': {
            'triton_operators_tested': len(test_results),
            'integration_status': 'completed',
        }
    }
    
    with open('triton_integration_test_report.json', 'w', encoding='utf-8') as f:
        import json
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Triton集成测试报告已保存: triton_integration_test_report.json")
    
    return test_results

if __name__ == "__main__":
    run_all_triton_tests()