#!/usr/bin/env python3
"""
TensorRT推理模块
提供TensorRT引擎的推理接口
"""

import os
import numpy as np
import torch
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT未安装，将使用模拟模式")

class TensorRTInference:
    """TensorRT推理类"""
    
    def __init__(self, engine_path: str):
        """
        初始化TensorRT推理
        
        Args:
            engine_path: TensorRT引擎文件路径
        """
        if not TENSORRT_AVAILABLE:
            logger.warning("TensorRT不可用，使用模拟模式")
            self.simulation_mode = True
            return
            
        self.simulation_mode = False
        self.engine_path = engine_path
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT引擎文件不存在: {engine_path}")
        
        # 初始化TensorRT
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # 加载引擎
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        # 创建执行上下文
        self.context = self.engine.create_execution_context()
        
        # 绑定信息与缓冲区
        self.input_names = []
        self.output_names = []
        self.binding_indices = {}
        self.binding_dtypes = {}
        self.host_inputs = {}
        self.host_outputs = {}
        self.device_buffers = {}
        self.bindings = [None] * self.engine.num_bindings

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            self.binding_indices[name] = i
            self.binding_dtypes[name] = trt.nptype(self.engine.get_binding_dtype(i))
            if self.engine.binding_is_input(i):
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        # CUDA stream
        self.stream = cuda.Stream()
        
        logger.info(f"TensorRT推理器初始化完成，输入: {self.input_names}, 输出: {self.output_names}")
    
    def infer(self, input_data: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        执行TensorRT推理
        
        Args:
            input_data: 输入张量
            **kwargs: 其他输入参数
            
        Returns:
            推理结果字典
        """
        if self.simulation_mode:
            # 模拟模式，返回随机结果
            return self._simulate_inference(input_data)
        
        try:
            # 准备输入数据
            if isinstance(input_data, torch.Tensor):
                input_data = input_data.detach().cpu().numpy()

            # 设置动态形状（假设单输入）
            input_name = self.input_names[0]
            input_index = self.binding_indices[input_name]
            input_dtype = self.binding_dtypes[input_name]
            self.context.set_binding_shape(input_index, input_data.shape)

            # 分配/重用 host/device 缓冲
            in_bytes = input_data.astype(input_dtype).nbytes
            if input_name not in self.host_inputs or self.host_inputs[input_name].nbytes != in_bytes:
                self.host_inputs[input_name] = cuda.pagelocked_empty(shape=input_data.size, dtype=input_dtype)
                self.device_buffers[input_name] = cuda.mem_alloc(in_bytes)
                self.bindings[input_index] = int(self.device_buffers[input_name])
            np.copyto(self.host_inputs[input_name], input_data.ravel())

            # 为每个输出准备缓冲
            for name in self.output_names:
                idx = self.binding_indices[name]
                out_shape = tuple(self.context.get_binding_shape(idx))
                out_dtype = self.binding_dtypes[name]
                out_size = int(np.prod(out_shape)) if len(out_shape) > 0 else 1
                nbytes = out_size * np.dtype(out_dtype).itemsize
                if name not in self.host_outputs or self.host_outputs[name].nbytes != nbytes:
                    self.host_outputs[name] = cuda.pagelocked_empty(shape=out_size, dtype=out_dtype)
                    self.device_buffers[name] = cuda.mem_alloc(nbytes)
                    self.bindings[idx] = int(self.device_buffers[name])

            # H2D
            cuda.memcpy_htod_async(self.device_buffers[input_name], self.host_inputs[input_name], self.stream)

            # 执行推理（异步）
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

            # D2H
            for name in self.output_names:
                cuda.memcpy_dtoh_async(self.host_outputs[name], self.device_buffers[name], self.stream)

            # 同步
            self.stream.synchronize()

            # 整理输出
            results = {}
            for name in self.output_names:
                idx = self.binding_indices[name]
                out_shape = tuple(self.context.get_binding_shape(idx))
                arr = np.array(self.host_outputs[name]).reshape(out_shape)
                results[name] = torch.from_numpy(arr.copy())

            return results
            
        except Exception as e:
            logger.error(f"TensorRT推理失败: {e}")
            raise
    
    def _simulate_inference(self, input_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """模拟推理（当TensorRT不可用时）"""
        batch_size = input_data.shape[0]
        
        # 生成模拟输出
        results = {
            'logits': torch.randn(batch_size, 1000, 10000),
            'feature_scores': torch.randn(batch_size, 1024),
            'engagement_scores': torch.randn(batch_size, 1),
            'retention_scores': torch.randn(batch_size, 1),
            'monetization_scores': torch.randn(batch_size, 1),
            'hidden_states': torch.randn(batch_size, 1000, 512)
        }
        
        return results
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'context') and self.context:
            del self.context
        if hasattr(self, 'engine') and self.engine:
            del self.engine
        if hasattr(self, 'runtime') and self.runtime:
            del self.runtime

def build_tensorrt_engine(onnx_path: str, engine_path: str, 
                         precision: str = "fp16", max_batch_size: int = 8) -> str:
    """
    构建TensorRT引擎
    
    Args:
        onnx_path: ONNX模型路径
        engine_path: 输出引擎路径
        precision: 精度模式 ("fp32", "fp16", "int8")
        max_batch_size: 最大批次大小
        
    Returns:
        引擎文件路径
    """
    if not TENSORRT_AVAILABLE:
        logger.warning("TensorRT不可用，跳过引擎构建")
        return engine_path
    
    try:
        # 创建TensorRT构建器
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        
        # 设置精度
        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("启用FP16精度")
        elif precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            logger.info("启用INT8精度")
        
        # 设置最大工作空间
        config.max_workspace_size = 1 << 30  # 1GB
        
        # 解析ONNX模型
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    logger.error(f"ONNX解析错误: {parser.get_error(error)}")
                raise RuntimeError("ONNX模型解析失败")
        
        # 构建引擎
        logger.info("开始构建TensorRT引擎...")
        engine = builder.build_engine(network, config)
        
        if engine is None:
            raise RuntimeError("TensorRT引擎构建失败")
        
        # 保存引擎
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        
        logger.info(f"TensorRT引擎构建成功: {engine_path}")
        return engine_path
        
    except Exception as e:
        logger.error(f"TensorRT引擎构建失败: {e}")
        raise

def benchmark_tensorrt_performance(engine_path: str, num_iterations: int = 100) -> Dict[str, float]:
    """
    基准测试TensorRT性能
    
    Args:
        engine_path: 引擎文件路径
        num_iterations: 测试迭代次数
        
    Returns:
        性能指标字典
    """
    if not TENSORRT_AVAILABLE:
        logger.warning("TensorRT不可用，跳过性能测试")
        return {}
    
    try:
        # 创建推理器
        trt_inference = TensorRTInference(engine_path)
        
        # 创建测试数据
        batch_size = 4
        input_data = torch.randn(batch_size, 1024, dtype=torch.float32)
        
        # 预热
        for _ in range(10):
            trt_inference.infer(input_data)
        
        # 性能测试
        import time
        times = []
        
        for _ in range(num_iterations):
            start_time = time.time()
            result = trt_inference.infer(input_data)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        # 计算统计信息
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        # 计算吞吐量
        throughput = batch_size / (avg_time / 1000)  # 样本/秒
        
        results = {
            'avg_latency_ms': avg_time,
            'min_latency_ms': min_time,
            'max_latency_ms': max_time,
            'std_latency_ms': std_time,
            'throughput_samples_per_sec': throughput,
            'batch_size': batch_size
        }
        
        logger.info(f"TensorRT性能测试结果: {results}")
        return results
        
    except Exception as e:
        logger.error(f"TensorRT性能测试失败: {e}")
        return {}
