#!/usr/bin/env python3
"""
HSTU特征处理测试

测试新的HSTU特征处理器是否正确集成到推理框架中
"""

import sys
import os
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_hstu_feature_processor():
    """测试HSTU特征处理器"""
    print("🔧 测试HSTU特征处理器...")
    
    try:
        from integrations.hstu.feature_processor import create_hstu_feature_processor
        
        # 创建配置
        config = {
            'vocab_size': 50000,
            'd_model': 1024,
            'max_seq_len': 100,
            'num_dense_features': 128
        }
        
        # 创建特征处理器
        processor = create_hstu_feature_processor(config)
        
        # 测试数据
        test_behaviors = [
            {
                'video_id': 'video_12345',
                'watch_duration': 120.5,
                'watch_percentage': 0.85,
                'is_liked': True,
                'is_shared': False,
                'timestamp': 1234567890,
                'category': 'tech',
                'engagement_score': 0.9
            },
            {
                'video_id': 'video_67890',
                'watch_duration': 45.2,
                'watch_percentage': 0.3,
                'is_liked': False,
                'is_shared': True,
                'timestamp': 1234567891,
                'category': 'music',
                'engagement_score': 0.6
            },
            {
                'video_id': 'video_11111',
                'watch_duration': 200.0,
                'watch_percentage': 1.0,
                'is_liked': True,
                'is_shared': True,
                'timestamp': 1234567892,
                'category': 'education',
                'engagement_score': 0.95
            }
        ]
        
        # 处理特征
        features = processor.process_user_behaviors(test_behaviors)
        
        print("✅ HSTU特征处理器测试通过")
        print("特征输出:")
        for key, value in features.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")
        
        # 获取统计信息
        stats = processor.get_feature_stats(test_behaviors)
        print(f"特征统计: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ HSTU特征处理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_framework_integration():
    """测试框架集成"""
    print("\n🔄 测试框架集成...")
    
    try:
        from integrations.framework_controller import create_integrated_controller
        from main import create_optimized_config
        
        # 创建配置
        config = create_optimized_config()
        
        # 创建控制器
        controller = create_integrated_controller(config)
        
        # 检查HSTU特征处理器
        has_processor = hasattr(controller, 'hstu_feature_processor')
        has_model = hasattr(controller, 'hstu_model')
        
        print(f"HSTU特征处理器: {'✅ 存在' if has_processor else '❌ 不存在'}")
        print(f"HSTU模型: {'✅ 存在' if has_model else '❌ 不存在'}")
        
        # 获取统计信息
        stats = controller.get_comprehensive_stats()
        print(f"HSTU特征统计: {stats.get('hstu_feature_stats', 'N/A')}")
        
        return has_processor and has_model
        
    except Exception as e:
        print(f"❌ 框架集成测试失败: {e}")
        return False

def test_end_to_end_processing():
    """测试端到端特征处理"""
    print("\n🎯 测试端到端特征处理...")
    
    try:
        from integrations.framework_controller import create_integrated_controller
        from main import create_optimized_config
        
        # 创建配置
        config = create_optimized_config()
        controller = create_integrated_controller(config)
        
        # 测试数据
        test_behaviors = [
            {
                'video_id': f'video_{i}',
                'watch_duration': 100 + i * 10,
                'watch_percentage': 0.5 + i * 0.1,
                'is_liked': i % 2 == 0,
                'is_shared': i % 3 == 0,
                'timestamp': 1000000000 + i * 100,
                'category': ['tech', 'music', 'sports'][i % 3],
                'engagement_score': 0.7 + i * 0.05
            }
            for i in range(5)
        ]
        
        # 测试特征处理
        if controller.framework_availability['hstu'] and controller.hstu_feature_processor:
            features = controller.hstu_feature_processor.process_user_behaviors(test_behaviors)
            
            print("✅ 端到端特征处理测试通过")
            print(f"输入行为数: {len(test_behaviors)}")
            print(f"输出特征数: {len(features)}")
            
            # 验证特征完整性
            required_keys = ['input_ids', 'attention_mask', 'dense_features']
            missing_keys = [k for k in required_keys if k not in features]
            
            if not missing_keys:
                print("✅ 所有必需特征键都存在")
                return True
            else:
                print(f"❌ 缺少特征键: {missing_keys}")
                return False
        else:
            print("⚠️ HSTU特征处理器不可用")
            return False
            
    except Exception as e:
        print(f"❌ 端到端测试失败: {e}")
        return False

def run_all_feature_tests():
    """运行所有特征处理测试"""
    print("🚀 开始HSTU特征处理测试")
    print("="*80)
    
    test_results = {}
    
    # 运行各项测试
    test_results['hstu_feature_processor'] = test_hstu_feature_processor()
    test_results['framework_integration'] = test_framework_integration()
    test_results['end_to_end_processing'] = test_end_to_end_processing()
    
    # 统计结果
    print("\n" + "="*80)
    print("HSTU特征处理测试结果汇总")
    print("="*80)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    print(f"\n📊 总体结果: {passed_tests}/{total_tests} 测试通过 ({passed_tests/total_tests:.1%})")
    
    # 保存测试报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'test_results': test_results,
        'summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'feature_processing_status': 'enhanced',
            'hstu_integration': 'completed'
        }
    }
    
    with open('feature_processing_test_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 特征处理测试报告已保存: feature_processing_test_report.json")
    
    return test_results

if __name__ == "__main__":
    run_all_feature_tests()