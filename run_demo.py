#!/usr/bin/env python3
"""
生成式推荐模型推理优化项目 - 快速演示脚本
简化版本，用于快速验证项目功能
"""

import sys
import os
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_quick_demo():
    """运行快速演示"""
    print("="*60)
    print("生成式推荐模型推理优化项目 - 快速演示")
    print("="*60)
    
    try:
        # 1. 导入必要模块
        print("1. 导入模块...")
        from src.inference_pipeline import UserBehaviorInferencePipeline
        from examples.client_example import create_realistic_user_behaviors
        
        # 2. 创建推理流水线
        print("2. 创建推理流水线...")
        pipeline = UserBehaviorInferencePipeline(
            model_config={
                "vocab_size": 10000,
                "embedding_dim": 512,
                "hidden_dim": 1024,
                "num_features": 1024,
                "num_layers": 6,
                "max_seq_len": 2048
            }
        )
        
        # 3. 创建示例数据
        print("3. 创建示例数据...")
        user_behaviors = create_realistic_user_behaviors("demo_user", 5)
        
        # 4. 执行推理
        print("4. 执行推理...")
        start_time = datetime.now()
        result = pipeline.infer_recommendations(
            user_id="demo_user",
            session_id="demo_session",
            behaviors=user_behaviors,
            num_recommendations=5
        )
        end_time = datetime.now()
        
        # 5. 显示结果
        inference_time = (end_time - start_time).total_seconds() * 1000
        
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
        
        print("\n" + "="*60)
        print("演示完成！项目运行成功！")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_model_info():
    """显示模型信息"""
    print("\n" + "="*60)
    print("模型信息")
    print("="*60)
    
    try:
        from src.model_parameter_calculator import main as calc_main
        calc_main()
    except Exception as e:
        print(f"无法显示模型信息: {e}")

if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.WARNING)
    
    # 运行演示
    success = run_quick_demo()
    
    if success:
        # 显示模型信息
        run_model_info()
        
        print("\n" + "="*60)
        print("下一步:")
        print("1. 运行完整版本: python main.py --mode all")
        print("2. 查看详细文档: docs/project_runtime_guide.md")
        print("3. 运行性能测试: python main.py --mode performance")
        print("="*60)
    else:
        print("\n演示失败，请检查:")
        print("1. 依赖是否安装: pip install -r requirements.txt")
        print("2. 是否在项目根目录运行")
        print("3. Python版本是否兼容 (推荐3.8+)")
