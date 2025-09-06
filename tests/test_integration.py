#!/usr/bin/env python3
"""
简化测试脚本

测试开源框架集成的基本功能，不依赖外部包的安装。
"""

import sys
import os
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_config_generation():
    """测试配置生成"""
    print("🔧 测试配置生成...")
    
    try:
        from main_opensource import create_optimized_config
        config = create_optimized_config()
        
        # 检查配置结构
        required_sections = ['hstu', 'vllm', 'tensorrt', 'custom_operators', 'intelligent_cache']
        
        for section in required_sections:
            if section not in config:
                print(f"❌ 缺少配置节: {section}")
                return False
            else:
                print(f"✅ 配置节存在: {section}")
        
        print(f"✅ 配置生成成功，包含 {len(config)} 个节")
        return True
        
    except Exception as e:
        print(f"❌ 配置生成失败: {e}")
        return False

def test_framework_imports():
    """测试框架导入（无外部依赖）"""
    print("\n📦 测试框架导入...")
    
    import_results = {}
    
    # 测试HSTU模型
    try:
        from integrations.hstu.hstu_model import HSTUModelConfig
        config = HSTUModelConfig(vocab_size=1000, d_model=128)
        import_results['hstu_config'] = True
        print("✅ HSTU配置类导入成功")
    except Exception as e:
        import_results['hstu_config'] = False
        print(f"❌ HSTU配置导入失败: {e}")
    
    # 测试VLLM配置
    try:
        from integrations.vllm.vllm_engine import VLLMConfig
        config = VLLMConfig(model_name="test", tensor_parallel_size=1)
        import_results['vllm_config'] = True
        print("✅ VLLM配置类导入成功")
    except Exception as e:
        import_results['vllm_config'] = False
        print(f"❌ VLLM配置导入失败: {e}")
    
    # 测试TensorRT配置
    try:
        from integrations.tensorrt.tensorrt_engine import TensorRTConfig
        config = TensorRTConfig(model_name="test", precision="fp16")
        import_results['tensorrt_config'] = True
        print("✅ TensorRT配置类导入成功")
    except Exception as e:
        import_results['tensorrt_config'] = False
        print(f"❌ TensorRT配置导入失败: {e}")
    
    # 测试智能缓存
    try:
        from optimizations.cache.intelligent_cache import IntelligentEmbeddingCache
        # 测试CPU模式
        cache = IntelligentEmbeddingCache(100, 64, device='cpu')
        import_results['intelligent_cache'] = True
        print("✅ 智能缓存导入成功")
    except Exception as e:
        import_results['intelligent_cache'] = False
        print(f"❌ 智能缓存导入失败: {e}")
    
    success_count = sum(import_results.values())
    total_count = len(import_results)
    print(f"\n📊 导入测试结果: {success_count}/{total_count} 成功")
    
    return import_results

def test_intelligent_cache():
    """测试智能缓存功能"""
    print("\n🧠 测试智能缓存功能...")
    
    try:
        from optimizations.cache.intelligent_cache import IntelligentEmbeddingCache
        import torch
        
        # 创建缓存
        cache = IntelligentEmbeddingCache(10, 64, device='cpu')
        
        # 测试缓存操作
        for i in range(5):
            embedding = torch.randn(64)
            slot = cache.put(i, embedding)
            retrieved_slot = cache.get(i)
            
            if slot != retrieved_slot:
                print(f"❌ 缓存测试失败: 项目 {i}")
                return False
        
        # 测试缓存信息
        cache_info = cache.get_cache_info()
        print(f"✅ 缓存测试成功，缓存信息: {cache_info}")
        
        # 测试热点预测
        if cache.predictor:
            hot_items = cache.predictor.predict_hot_items(3)
            print(f"✅ 热点预测测试成功: {len(hot_items)} 个热点项目")
        
        return True
        
    except Exception as e:
        print(f"❌ 智能缓存测试失败: {e}")
        return False

def test_framework_controller_creation():
    """测试框架控制器创建"""
    print("\n🎛️ 测试框架控制器创建...")
    
    try:
        from integrations.framework_controller import OpenSourceFrameworkController
        from main_opensource import create_optimized_config
        
        # 创建简化配置
        config = create_optimized_config()
        
        # 创建控制器（在没有外部依赖的情况下会使用回退模式）
        controller = OpenSourceFrameworkController(config)
        
        # 检查框架可用性
        availability = controller.framework_availability
        print(f"✅ 框架控制器创建成功")
        print(f"   框架可用性: {availability}")
        
        # 测试获取统计信息
        stats = controller.get_comprehensive_stats()
        print(f"✅ 统计信息获取成功: {len(stats)} 个统计项")
        
        return True
        
    except Exception as e:
        print(f"❌ 框架控制器创建失败: {e}")
        return False

def test_integration_flow():
    """测试集成流程"""
    print("\n🔄 测试集成流程...")
    
    try:
        from integrations.framework_controller import OpenSourceFrameworkController
        from examples.client_example import create_realistic_user_behaviors
        from main_opensource import create_optimized_config
        
        # 创建配置和控制器
        config = create_optimized_config()
        controller = OpenSourceFrameworkController(config)
        
        # 创建测试数据
        user_behaviors = create_realistic_user_behaviors("test_user", 5)
        
        # 测试推理（会使用回退模式）
        result = controller.infer_with_optimal_strategy(
            user_id="test_user",
            session_id="test_session",
            user_behaviors=user_behaviors,
            num_recommendations=5,
            strategy="auto"
        )
        
        # 检查结果
        if 'recommendations' in result:
            print(f"✅ 推理测试成功，生成了 {len(result['recommendations'])} 个推荐")
            print(f"   使用策略: {result.get('inference_strategy', 'unknown')}")
            print(f"   推理时间: {result.get('inference_time_ms', 0):.2f}ms")
            return True
        else:
            print(f"❌ 推理测试失败: {result}")
            return False
        
    except Exception as e:
        print(f"❌ 集成流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_project_structure():
    """测试项目结构"""
    print("\n📁 测试项目结构...")
    
    required_dirs = [
        'integrations',
        'integrations/hstu',
        'integrations/vllm', 
        'integrations/tensorrt',
        'optimizations',
        'optimizations/triton_ops',
        'optimizations/cutlass_ops',
        'optimizations/cache',
        'external',
        'external/meta-hstu',
        'external/vllm',
    ]
    
    required_files = [
        'integrations/hstu/hstu_model.py',
        'integrations/vllm/vllm_engine.py',
        'integrations/tensorrt/tensorrt_engine.py',
        'integrations/framework_controller.py',
        'optimizations/cache/intelligent_cache.py',
        'main_opensource.py',
        'requirements.txt',
    ]
    
    missing_dirs = []
    missing_files = []
    
    # 检查目录
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    # 检查文件
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_dirs:
        print(f"❌ 缺少目录: {missing_dirs}")
        return False
    
    if missing_files:
        print(f"❌ 缺少文件: {missing_files}")
        return False
    
    print(f"✅ 项目结构完整，检查了 {len(required_dirs)} 个目录和 {len(required_files)} 个文件")
    return True

def run_all_tests():
    """运行所有测试"""
    print("🚀 开始开源框架集成测试")
    print("="*80)
    
    test_results = {}
    
    # 运行各项测试
    test_results['project_structure'] = test_project_structure()
    test_results['config_generation'] = test_config_generation()
    test_results['framework_imports'] = test_framework_imports()
    test_results['intelligent_cache'] = test_intelligent_cache()
    test_results['framework_controller'] = test_framework_controller_creation()
    test_results['integration_flow'] = test_integration_flow()
    
    # 统计结果
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    print(f"\n📊 总体结果: {passed_tests}/{total_tests} 测试通过 ({passed_tests/total_tests:.1%})")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！开源框架集成准备就绪。")
    else:
        print("⚠️ 部分测试失败，请检查相关组件。")
    
    # 保存测试报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'test_results': test_results,
        'summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests,
        },
        'framework_integration_ready': passed_tests >= 4,  # 至少4个核心测试通过
    }
    
    with open('integration_test_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 测试报告已保存: integration_test_report.json")
    
    return test_results

if __name__ == "__main__":
    run_all_tests()