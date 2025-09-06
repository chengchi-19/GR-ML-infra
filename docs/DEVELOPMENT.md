# 开发者指南

## 🛠️ 开发环境搭建

### 环境要求
- Python 3.8+
- CUDA 11.8+ (GPU开发)
- Git 2.20+
- Docker (可选，用于容器化开发)

### 开发环境安装
```bash
# 1. 克隆项目
git clone <repository-url>
cd GR-ML-infra

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装开发依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 开发工具

# 4. 验证安装
python tests/test_integration.py
```

### IDE配置推荐
```json
// .vscode/settings.json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

## 🏗️ 项目架构理解

### 核心设计模式
1. **适配器模式**: 统一不同开源框架的接口
2. **管道模式**: HSTU→ONNX→TensorRT→VLLM的统一处理链
3. **观察者模式**: 性能监控和日志记录
4. **工厂模式**: 创建和管理推理引擎

### 关键抽象层级
```
应用层 (main.py)
    ↓
控制层 (framework_controller.py)
    ↓
适配层 (integrations/*.py)
    ↓  
引擎层 (HSTU/VLLM/TensorRT)
```

### 数据流图
```
用户请求 → 特征预处理 → HSTU特征提取 → ONNX导出 → TensorRT优化 → VLLM推理 → 结果后处理 → 响应输出
    ↑                                                                 ↓
 监控统计 ←——————————————————— 性能记录 ←——————————————————— 缓存管理
```

## 🔧 核心组件开发

### 1. 新增推理框架

#### Step 1: 创建框架适配器
```python
# integrations/new_framework/new_engine.py
class NewFrameworkEngine:
    def __init__(self, config):
        self.config = config
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """检查框架是否可用"""
        try:
            import new_framework
            return True
        except ImportError:
            return False
    
    def infer(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行推理"""
        if not self.available:
            raise RuntimeError("New framework not available")
        
        # 实现推理逻辑
        return results
    
    async def batch_infer(self, requests: List[Dict]) -> List[Dict]:
        """批量推理"""
        # 实现批量推理逻辑
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'available': self.available,
            'model_info': {},
            'performance_metrics': {}
        }
```

#### Step 2: 集成到控制器
```python
# integrations/framework_controller.py
from integrations.new_framework.new_engine import NewFrameworkEngine

class OpenSourceFrameworkController:
    def __init__(self, config):
        # 现有代码...
        self.new_engine = NewFrameworkEngine(config.get('new_framework', {}))
        self.framework_availability['new_framework'] = self.new_engine.available
    
    def _select_optimal_strategy(self, user_behaviors):
        # 添加新的策略选择逻辑
        if some_condition and self.new_engine.available:
            return "new_framework"
        # 现有逻辑...
```

#### Step 3: 添加配置支持
```python
# main.py - create_optimized_config()
def create_optimized_config():
    config = {
        # 现有配置...
        'new_framework': {
            'model_name': 'new-model',
            'precision': 'fp16',
            'max_batch_size': 8,
            # 框架特定配置...
        }
    }
    return config
```

### 2. 自定义算子开发

#### Triton算子示例
```python
# optimizations/triton_ops/custom_attention.py
import triton
import triton.language as tl

@triton.jit
def custom_attention_kernel(
    Q, K, V, O,
    seq_len, head_dim,
    BLOCK_SIZE: tl.constexpr
):
    # Triton kernel实现
    pass

def custom_attention(q, k, v):
    """自定义注意力算子"""
    # 调用Triton kernel
    return output
```

#### 集成到推理引擎
```python
# integrations/hstu/hstu_model.py
from optimizations.triton_ops.custom_attention import custom_attention

class HSTUGenerativeRecommender:
    def forward(self, input_ids, attention_mask, **kwargs):
        # 使用自定义算子
        if self.use_custom_ops:
            attention_output = custom_attention(q, k, v)
        else:
            attention_output = self.attention(q, k, v)
        
        return outputs
```

### 3. 缓存策略扩展

#### 新缓存策略
```python
# optimizations/cache/new_cache_strategy.py
class NewCacheStrategy:
    def __init__(self, config):
        self.config = config
    
    def should_cache(self, key: str, value: Any) -> bool:
        """判断是否需要缓存"""
        pass
    
    def get_priority(self, key: str) -> float:
        """获取缓存优先级"""  
        pass
    
    def eviction_candidate(self, cache_items) -> str:
        """选择驱逐候选项"""
        pass
```

#### 集成到智能缓存
```python
# optimizations/cache/intelligent_cache.py
from .new_cache_strategy import NewCacheStrategy

class IntelligentEmbeddingCache:
    def __init__(self, config):
        # 现有代码...
        if config.get('strategy') == 'new_strategy':
            self.cache_strategy = NewCacheStrategy(config)
```

## 🧪 测试开发

### 单元测试
```python
# tests/test_new_framework.py
import unittest
from integrations.new_framework.new_engine import NewFrameworkEngine

class TestNewFramework(unittest.TestCase):
    def setUp(self):
        self.config = {'model_name': 'test-model'}
        self.engine = NewFrameworkEngine(self.config)
    
    def test_availability_check(self):
        """测试框架可用性检查"""
        availability = self.engine._check_availability()
        self.assertIsInstance(availability, bool)
    
    def test_inference(self):
        """测试推理功能"""
        inputs = {'input_ids': [1, 2, 3]}
        result = self.engine.infer(inputs)
        self.assertIn('predictions', result)
    
    def test_batch_inference(self):
        """测试批量推理"""
        requests = [{'input_ids': [1, 2, 3]}] * 5
        results = self.engine.batch_infer(requests)
        self.assertEqual(len(results), 5)
```

### 集成测试
```python
# tests/test_integration_extended.py
def test_new_framework_integration():
    """测试新框架集成"""
    config = create_optimized_config()
    controller = create_integrated_controller(config)
    
    # 验证框架注册
    assert 'new_framework' in controller.framework_availability
    
    # 测试策略选择
    strategy = controller._select_optimal_strategy(mock_behaviors)
    assert strategy in ['hstu', 'vllm', 'tensorrt', 'new_framework', 'fallback']
```

### 性能测试
```python
# tests/test_performance_extended.py
import time
import pytest

@pytest.mark.performance
def test_new_framework_performance():
    """测试新框架性能"""
    engine = NewFrameworkEngine(test_config)
    
    if not engine.available:
        pytest.skip("New framework not available")
    
    # 性能基准测试
    start_time = time.time()
    for _ in range(100):
        result = engine.infer(test_input)
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / 100 * 1000  # ms
    assert avg_latency < 50.0, f"Performance regression: {avg_latency}ms"
```

## 📊 监控与调试

### 添加自定义指标
```python
# integrations/framework_controller.py
class OpenSourceFrameworkController:
    def __init__(self, config):
        # 现有代码...
        self.custom_metrics = defaultdict(list)
    
    def _record_custom_metric(self, metric_name: str, value: float):
        """记录自定义指标"""
        self.custom_metrics[metric_name].append({
            'timestamp': time.time(),
            'value': value
        })
```

### 日志配置
```python
# utils/logging_config.py
import logging
import sys

def setup_logging(level='INFO'):
    """配置日志系统"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('development.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 设置特定模块的日志级别
    logging.getLogger('integrations').setLevel(logging.DEBUG)
    logging.getLogger('optimizations').setLevel(logging.INFO)
```

### 调试工具
```python
# utils/debug_tools.py
import functools
import time

def performance_timer(func):
    """性能计时装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {(end-start)*1000:.2f}ms")
        return result
    return wrapper

def memory_tracker(func):
    """内存使用追踪装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        result = func(*args, **kwargs)
        mem_after = process.memory_info().rss / 1024 / 1024
        print(f"{func.__name__} memory usage: {mem_after - mem_before:.2f}MB")
        return result
    return wrapper
```

## 🚀 部署与发布

### 开发环境部署
```bash
# 本地开发服务器
python main.py --mode=single --log-level=DEBUG

# 热重载开发
pip install watchdog
watchmedo auto-restart --patterns="*.py" --recursive -- python main.py
```

### 容器化开发
```dockerfile
# Dockerfile.dev
FROM nvidia/cuda:11.8-devel-ubuntu22.04

WORKDIR /app
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

COPY . .
CMD ["python", "main.py", "--mode=comprehensive"]
```

### CI/CD配置
```yaml
# .github/workflows/test.yml
name: CI Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: pip install -r requirements-dev.txt
    - name: Run tests
      run: python -m pytest tests/ -v
    - name: Run integration tests
      run: python tests/test_integration.py
```

## 📋 代码规范

### 代码风格
```python
# 使用Black格式化
black --line-length 88 .

# 使用flake8检查
flake8 --max-line-length 88 --ignore E203,W503 .

# 使用isort排序导入
isort --profile black .
```

### 文档规范
```python
def infer_recommendations(
    self,
    user_id: str,
    session_id: str, 
    user_behaviors: List[Dict[str, Any]],
    num_recommendations: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    执行推荐推理
    
    Args:
        user_id: 用户唯一标识
        session_id: 会话标识  
        user_behaviors: 用户行为序列
        num_recommendations: 推荐数量
        **kwargs: 额外参数
        
    Returns:
        Dict containing:
            - recommendations: 推荐结果列表
            - inference_strategy: 使用的推理策略
            - inference_time_ms: 推理时间(毫秒)
            
    Raises:
        ValueError: 参数无效
        RuntimeError: 推理失败
    """
```

### Git提交规范
```bash
# 提交消息格式
git commit -m "feat: 添加新的推理框架支持"
git commit -m "fix: 修复VLLM内存泄漏问题" 
git commit -m "docs: 更新API文档"
git commit -m "test: 添加性能回归测试"
git commit -m "refactor: 重构缓存管理模块"
```

---

**🎯 开发原则**: 
1. **模块化设计**: 新功能独立开发，接口统一
2. **测试驱动**: 先写测试，再实现功能
3. **性能优先**: 每个改动都要考虑性能影响
4. **向后兼容**: 保持API兼容性
5. **文档完善**: 代码即文档，注释清晰