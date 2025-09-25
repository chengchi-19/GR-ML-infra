# 🌟 基于生成式推荐模型的推理优化项目

## 📋 项目概述

这是一个基于**开源框架**的生成式推荐模型推理优化部署项目，集成了Meta开源的生成式HSTU模型、自定义Triton和CUTLASS算子、TensorRT加速引擎、VLLM推理引擎：
- **Meta HSTU** (Hierarchical Sequential Transduction Units) 生成式推荐模型
- **自定义Triton和CUTLASS算子**优化
- **TensorRT** GPU推理加速引擎
- **VLLM** (PagedAttention + Continuous Batching) 推理优化框架  

## 🎯 核心特性

### 🚀 开源框架技术栈
- ✅ **Meta HSTU模型**: Hierarchical Sequential Transduction Units架构
- ✅ **VLLM推理引擎**: PagedAttention内存优化  
- ✅ **TensorRT加速**: GPU推理优化，支持FP16/INT8量化
- ✅ **智能缓存系统**: 预测式GPU热缓存
- ✅ **自定义算子**: Triton DSL和CUTLASS高性能算子
- 🌐 **RESTful API服务**: FastAPI提供生产级API接口

### 🧠 统一推理流程
- 🔄 **HSTU模型**: 负责特征提取和序列建模
- 📤 **ONNX导出**: 将HSTU模型导出为标准ONNX格式
- ⚡ **TensorRT编译期优化**: 专注算子融合与Kernel特化，生成优化的引擎文件.engine
- 🎯 **VLLM端到端推理**: 负责从Prefill到Decode的完整流程，实现高性能生成

## 🔧 核心功能模块

### 1. 统一框架控制器 (`integrations/framework_controller.py`)

**统一推理流程**:
```python
def infer_with_unified_pipeline(self, user_behaviors, ...):
    # Step 1: HSTU模型特征提取
    hstu_inputs = self._prepare_hstu_inputs(user_behaviors)
    
    # Step 2: TensorRT GPU优化推理
    trt_outputs = self.tensorrt_engine.infer(hstu_inputs)
    
    # Step 3: VLLM推理服务优化
    final_result = self._vllm_service_optimization(trt_outputs, ...)
    
    return final_result
```

### 2. Meta HSTU模型集成 (`integrations/hstu/hstu_model.py`)

**核心特性**:
- 集成Meta开源的HSTU架构
- 支持Hierarchical Sequential Transduction Units
- 多任务学习(engagement, retention, monetization)

### 3. VLLM推理引擎 (`integrations/vllm/vllm_engine.py`)

**性能优化**:
- **端到端推理**: 负责从Prefill到Decode的完整流程
- **PagedAttention**: KV缓存内存管理，极大提升吞吐
- **Continuous Batching**: 动态批处理，最大化GPU利用率
- **TensorRT集成**: 可加载经TensorRT编译优化的模型，获得极致性能
- 异步推理支持

### 4. TensorRT优化引擎 (`integrations/tensorrt/tensorrt_engine.py`)

**编译期加速特性**:
- **离线深度优化**: 专注在模型编译阶段进行算子融合、层消除与精度校准
- **Kernel特化**: 为特定GPU架构生成高度优化的CUDA Kernel
- **插件支持**: 集成自定义的高性能CUDA算子
- **最终产物**: 生成一个优化的 `.engine` 文件，供VLLM等下游引擎加载

### 5. 智能缓存系统 (`optimizations/cache/intelligent_cache.py`)

**智能特性**:
- GPU热缓存优化
- 热点预测算法
- 智能驱逐策略
- 与开源框架无缝集成

### 6. FastAPI服务层 (`api_server.py`)

**服务特性**:
- RESTful API接口设计
- 异步请求处理
- 实时性能监控
- 自动API文档生成
- 完善的错误处理和日志记录

## 📊 系统架构

### 核心组件
- **统一框架控制器**: 智能策略选择和资源调度
- **多引擎支持**: HSTU、VLLM、TensorRT协同工作
- **自定义优化**: Triton和CUTLASS算子加速
- **智能缓存**: 预测式GPU内存管理

### 技术优势
- **开源框架集成**: 基于成熟的开源技术栈
- **统一推理管道**: HSTU→ONNX→TensorRT→VLLM的端到端优化
- **自动优化链**: 从模型到服务的完整优化链路
- **完整监控体系**: 实时性能监控和调优


## 🧪 测试和验证

```bash

# 运行Triton算子测试
python tests/test_triton_integration.py

# 启动API服务
python api_server.py
```

# 🙏 致谢

感谢以下开源项目和团队：
- **Meta AI** - HSTU生成式推荐模型
- **VLLM团队** - 高性能推理优化框架
- **NVIDIA** - TensorRT GPU加速引擎
- **OpenAI** - Triton DSL自定义算子框架

---

**🎯 项目重点**: 这是一个基于开源框架的推荐系统推理优化项目，通过集成Meta HSTU、VLLM、TensorRT等顶级开源技术，实现了生成式推荐模型的推理优化部署流程。项目针对生成式推荐模型HSTU自定义了多个Triton和CUTLASS算子，通过TensorRT推理引擎进行加速，最后通过VLLM推理引擎进行推理。通过集成多种开源框架，本项目实现了**1+1+1+1>4的协同效应**，为推荐系统推理优化提供了完整的解决方案。