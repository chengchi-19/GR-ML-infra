#!/usr/bin/env python3
"""
生成式推荐模型推理优化项目 - 推理优化演示
展示TensorRT、Triton、GPU加速等核心优化功能
"""

import sys
import os
import logging
import subprocess
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_gpu_availability():
    """检查GPU可用性"""
    print("="*60)
    print("GPU环境检查")
    print("="*60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用，GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            return True
        else:
            print("❌ CUDA不可用，将使用CPU模式")
            return False
    except ImportError:
        print("❌ PyTorch未安装")
        return False

def run_tensorrt_optimization():
    """运行TensorRT优化"""
    print("\n" + "="*60)
    print("TensorRT优化演示")
    print("="*60)
    
    try:
        # 检查TensorRT
        import tensorrt as trt
        print(f"✅ TensorRT版本: {trt.__version__}")
        
        # 检查ONNX模型是否存在
        if os.path.exists("models/prefill.onnx"):
            print("✅ 找到ONNX模型，可以构建TensorRT引擎")
            
            # 构建TensorRT引擎
            print("正在构建TensorRT引擎...")
            from src.build_engine import build_tensorrt_engine
            
            engine_path = build_tensorrt_engine(
                onnx_path="models/prefill.onnx",
                engine_path="models/prefill.trt",
                precision="fp16",
                max_batch_size=8
            )
            
            if os.path.exists(engine_path):
                print(f"✅ TensorRT引擎构建成功: {engine_path}")
                
                # 测试TensorRT推理
                print("测试TensorRT推理...")
                from src.tensorrt_inference import TensorRTInference
                
                trt_inference = TensorRTInference(engine_path)
                
                # 创建测试数据
                import torch
                dummy_input = torch.randn(1, 1024, dtype=torch.float32)
                
                # 预热
                for _ in range(3):
                    trt_inference.infer(dummy_input)
                
                # 性能测试
                num_tests = 10
                times = []
                
                for _ in range(num_tests):
                    start_time = time.time()
                    result = trt_inference.infer(dummy_input)
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)
                
                avg_time = sum(times) / len(times)
                print(f"TensorRT推理性能: {avg_time:.2f}ms (平均)")
                
                return True
            else:
                print("❌ TensorRT引擎构建失败")
                return False
        else:
            print("❌ 未找到ONNX模型，请先运行模型导出")
            return False
            
    except ImportError:
        print("❌ TensorRT未安装")
        return False
    except Exception as e:
        print(f"❌ TensorRT优化失败: {e}")
        return False

def run_triton_deployment():
    """运行Triton部署演示"""
    print("\n" + "="*60)
    print("Triton推理服务器部署演示")
    print("="*60)
    
    try:
        # 检查Triton模型仓库
        model_repo_path = "triton_model_repo"
        if os.path.exists(model_repo_path):
            print(f"✅ 找到Triton模型仓库: {model_repo_path}")
            
            # 列出模型
            models = []
            for item in os.listdir(model_repo_path):
                item_path = os.path.join(model_repo_path, item)
                if os.path.isdir(item_path):
                    config_path = os.path.join(item_path, "config.pbtxt")
                    if os.path.exists(config_path):
                        models.append(item)
            
            print(f"发现 {len(models)} 个Triton模型:")
            for model in models:
                print(f"  - {model}")
            
            # 检查ensemble模型
            ensemble_path = os.path.join(model_repo_path, "ensemble_model")
            if os.path.exists(ensemble_path):
                print("✅ 找到ensemble模型配置")
                
                # 启动Triton服务器（模拟）
                print("启动Triton推理服务器...")
                print("注意: 实际部署需要安装Triton Inference Server")
                
                # 显示启动命令
                print("\n启动命令示例:")
                print("docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 \\")
                print(f"  -v {os.path.abspath(model_repo_path)}:/models \\")
                print("  nvcr.io/nvidia/tritonserver:23.12-py3 \\")
                print("  tritonserver --model-repository=/models")
                
                return True
            else:
                print("❌ 未找到ensemble模型配置")
                return False
        else:
            print("❌ 未找到Triton模型仓库")
            return False
            
    except Exception as e:
        print(f"❌ Triton部署失败: {e}")
        return False

def run_custom_operators():
    """运行自定义算子演示"""
    print("\n" + "="*60)
    print("自定义算子演示")
    print("="*60)
    
    try:
        # 检查自定义算子
        kernels_path = "kernels"
        if os.path.exists(kernels_path):
            print(f"✅ 找到自定义算子目录: {kernels_path}")
            
            # 列出算子类型
            operator_types = []
            for item in os.listdir(kernels_path):
                item_path = os.path.join(kernels_path, item)
                if os.path.isdir(item_path):
                    operator_types.append(item)
            
            print(f"发现 {len(operator_types)} 种自定义算子:")
            for op_type in operator_types:
                print(f"  - {op_type}")
                
                # 检查具体实现
                op_path = os.path.join(kernels_path, op_type)
                if os.path.exists(op_path):
                    files = os.listdir(op_path)
                    for file in files:
                        if file.endswith(('.cpp', '.cu', '.py')):
                            print(f"    └── {file}")
            
            # 测试Triton DSL算子
            triton_ops_path = os.path.join(kernels_path, "triton_ops")
            if os.path.exists(triton_ops_path):
                print("\n测试Triton DSL算子...")
                
                # 检查是否有编译好的算子
                compiled_ops = []
                for file in os.listdir(triton_ops_path):
                    if file.endswith('.so') or file.endswith('.pt'):
                        compiled_ops.append(file)
                
                if compiled_ops:
                    print(f"✅ 找到 {len(compiled_ops)} 个编译好的算子:")
                    for op in compiled_ops:
                        print(f"  - {op}")
                else:
                    print("⚠️  未找到编译好的算子，需要先编译")
                    print("编译命令示例:")
                    print("cd kernels/triton_ops")
                    print("python setup.py build_ext --inplace")
            
            # 测试TensorRT插件
            trt_plugin_path = os.path.join(kernels_path, "trt_plugin_skeleton")
            if os.path.exists(trt_plugin_path):
                print("\n测试TensorRT插件...")
                
                plugin_files = []
                for file in os.listdir(trt_plugin_path):
                    if file.endswith(('.cpp', '.h', '.cu')):
                        plugin_files.append(file)
                
                if plugin_files:
                    print(f"✅ 找到 {len(plugin_files)} 个TensorRT插件文件:")
                    for file in plugin_files:
                        print(f"  - {file}")
                    
                    print("\n编译TensorRT插件命令:")
                    print("cd kernels/trt_plugin_skeleton")
                    print("mkdir build && cd build")
                    print("cmake .. && make")
                else:
                    print("❌ 未找到TensorRT插件文件")
            
            return True
        else:
            print("❌ 未找到自定义算子目录")
            return False
            
    except Exception as e:
        print(f"❌ 自定义算子演示失败: {e}")
        return False

def run_performance_comparison():
    """运行性能对比"""
    print("\n" + "="*60)
    print("性能对比演示")
    print("="*60)
    
    try:
        # 创建测试数据
        import torch
        batch_size = 4
        seq_len = 1000
        num_features = 1024
        
        dummy_input = torch.randn(batch_size, num_features, dtype=torch.float32)
        dummy_ids = torch.randint(0, 10000, (batch_size, seq_len), dtype=torch.long)
        
        print(f"测试配置: batch_size={batch_size}, seq_len={seq_len}, features={num_features}")
        
        # PyTorch CPU推理
        print("\n1. PyTorch CPU推理:")
        from src.inference_pipeline import UserBehaviorInferencePipeline
        
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
        
        # 预热
        for _ in range(3):
            pipeline.model.forward_prefill(dummy_ids, dummy_input)
        
        # 性能测试
        num_tests = 10
        times = []
        
        for _ in range(num_tests):
            start_time = time.time()
            result = pipeline.model.forward_prefill(dummy_ids, dummy_input)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
        
        avg_time = sum(times) / len(times)
        print(f"  PyTorch CPU: {avg_time:.2f}ms (平均)")
        
        # PyTorch GPU推理（如果可用）
        if torch.cuda.is_available():
            print("\n2. PyTorch GPU推理:")
            
            # 移动模型到GPU
            pipeline.model.cuda()
            dummy_input_gpu = dummy_input.cuda()
            dummy_ids_gpu = dummy_ids.cuda()
            
            # 预热
            for _ in range(3):
                pipeline.model.forward_prefill(dummy_ids_gpu, dummy_input_gpu)
            
            # 性能测试
            times = []
            
            for _ in range(num_tests):
                start_time = time.time()
                result = pipeline.model.forward_prefill(dummy_ids_gpu, dummy_input_gpu)
                torch.cuda.synchronize()  # 确保GPU计算完成
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
            
            avg_time = sum(times) / len(times)
            print(f"  PyTorch GPU: {avg_time:.2f}ms (平均)")
            
            # 计算加速比
            cpu_time = sum(times) / len(times)
            gpu_time = avg_time
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            print(f"  GPU加速比: {speedup:.2f}x")
        
        # TensorRT推理（如果可用）
        trt_engine_path = "models/prefill.trt"
        if os.path.exists(trt_engine_path):
            print("\n3. TensorRT推理:")
            try:
                from src.tensorrt_inference import TensorRTInference
                
                trt_inference = TensorRTInference(trt_engine_path)
                
                # 预热
                for _ in range(3):
                    trt_inference.infer(dummy_input)
                
                # 性能测试
                times = []
                
                for _ in range(num_tests):
                    start_time = time.time()
                    result = trt_inference.infer(dummy_input)
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)
                
                avg_time = sum(times) / len(times)
                print(f"  TensorRT: {avg_time:.2f}ms (平均)")
                
                # 计算加速比
                if torch.cuda.is_available():
                    trt_speedup = gpu_time / avg_time if avg_time > 0 else 0
                    print(f"  TensorRT加速比: {trt_speedup:.2f}x (相对于PyTorch GPU)")
                
            except Exception as e:
                print(f"  TensorRT推理失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能对比失败: {e}")
        return False

def run_optimization_pipeline():
    """运行完整优化流水线"""
    print("\n" + "="*60)
    print("完整优化流水线演示")
    print("="*60)
    
    print("优化流水线步骤:")
    print("1. 模型导出 (PyTorch → ONNX)")
    print("2. TensorRT优化 (ONNX → TRT)")
    print("3. 自定义算子集成")
    print("4. Triton部署配置")
    print("5. 性能测试和监控")
    
    # 步骤1: 模型导出
    print("\n步骤1: 导出ONNX模型...")
    try:
        from src.export_onnx import GenerativeRecommendationModel
        import torch
        
        model = GenerativeRecommendationModel(
            vocab_size=10000,
            embedding_dim=512,
            hidden_dim=1024,
            num_features=1024,
            num_layers=6,
            max_seq_len=2048
        )
        model.eval()
        
        # 创建示例数据
        dummy_ids = torch.randint(0, 10000, (1, 1000), dtype=torch.long)
        dummy_dense = torch.randn(1, 1024, dtype=torch.float32)
        dummy_user = torch.randn(1, 256, dtype=torch.float32)
        dummy_video = torch.randn(1, 512, dtype=torch.float32)
        dummy_mask = torch.ones(1, 1000, dtype=torch.long)
        
        # 确保models目录存在
        os.makedirs("models", exist_ok=True)
        
        # 导出ONNX
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
        
        if os.path.exists("models/prefill.onnx"):
            print("✅ ONNX模型导出成功")
        else:
            print("❌ ONNX模型导出失败")
            return False
            
    except Exception as e:
        print(f"❌ ONNX导出失败: {e}")
        return False
    
    # 步骤2: TensorRT优化
    print("\n步骤2: TensorRT优化...")
    trt_success = run_tensorrt_optimization()
    
    # 步骤3: 自定义算子
    print("\n步骤3: 自定义算子集成...")
    custom_op_success = run_custom_operators()
    
    # 步骤4: Triton部署
    print("\n步骤4: Triton部署配置...")
    triton_success = run_triton_deployment()
    
    # 步骤5: 性能测试
    print("\n步骤5: 性能测试...")
    perf_success = run_performance_comparison()
    
    return trt_success and custom_op_success and triton_success and perf_success

def main():
    """主函数"""
    print("="*80)
    print("生成式推荐模型推理优化项目 - 推理优化演示")
    print("="*80)
    
    # 设置日志级别
    logging.basicConfig(level=logging.WARNING)
    
    try:
        # 1. 检查GPU环境
        gpu_available = check_gpu_availability()
        
        # 2. 运行完整优化流水线
        success = run_optimization_pipeline()
        
        if success:
            print("\n" + "="*80)
            print("推理优化演示完成！")
            print("="*80)
            print("\n下一步:")
            print("1. 安装TensorRT: pip install tensorrt")
            print("2. 安装Triton: 参考官方文档")
            print("3. 编译自定义算子: cd kernels && make")
            print("4. 启动Triton服务器: scripts/run_server.sh")
            print("5. 运行性能测试: python bench/benchmark.py")
        else:
            print("\n" + "="*80)
            print("推理优化演示部分失败")
            print("="*80)
            print("\n请检查:")
            print("1. 依赖是否完整安装")
            print("2. GPU环境是否正确配置")
            print("3. 模型文件是否存在")
        
    except Exception as e:
        print(f"\n演示失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
