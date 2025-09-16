#!/usr/bin/env python3
"""
GR-ML-infra TensorRT插件Python绑定

提供Python接口来加载和使用TensorRT插件
"""

import os
import sys
import ctypes
import logging
from typing import Optional, List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT Python API不可用")

class GRMLInfraTensorRTPlugins:
    """GR-ML-infra TensorRT插件管理器"""

    def __init__(self, plugin_lib_path: Optional[str] = None):
        self.plugin_lib = None
        self.plugins_initialized = False

        if not TENSORRT_AVAILABLE:
            logger.warning("TensorRT不可用，插件功能将被禁用")
            return

        # 查找插件库
        if plugin_lib_path is None:
            plugin_lib_path = self._find_plugin_library()

        if plugin_lib_path and os.path.exists(plugin_lib_path):
            self._load_plugin_library(plugin_lib_path)
            self._initialize_plugins()
        else:
            logger.warning(f"TensorRT插件库未找到: {plugin_lib_path}")

    def _find_plugin_library(self) -> Optional[str]:
        """查找插件库文件"""
        possible_paths = [
            # 相对于当前文件的路径
            os.path.join(os.path.dirname(__file__), "libgr_ml_infra_tensorrt_plugins.so"),
            os.path.join(os.path.dirname(__file__), "cpp", "libgr_ml_infra_tensorrt_plugins.so"),

            # 系统路径
            "/usr/local/lib/libgr_ml_infra_tensorrt_plugins.so",
            "/usr/lib/libgr_ml_infra_tensorrt_plugins.so",

            # 项目根目录
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib", "libgr_ml_infra_tensorrt_plugins.so"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"找到TensorRT插件库: {path}")
                return path

        return None

    def _load_plugin_library(self, lib_path: str):
        """加载插件库"""
        try:
            self.plugin_lib = ctypes.CDLL(lib_path)

            # 定义C函数签名
            self.plugin_lib.initialize_gr_ml_infra_plugins.argtypes = []
            self.plugin_lib.initialize_gr_ml_infra_plugins.restype = None

            self.plugin_lib.get_num_registered_plugins.argtypes = []
            self.plugin_lib.get_num_registered_plugins.restype = ctypes.c_int

            logger.info(f"✅ TensorRT插件库加载成功: {lib_path}")

        except Exception as e:
            logger.error(f"❌ 加载TensorRT插件库失败: {e}")
            self.plugin_lib = None

    def _initialize_plugins(self):
        """初始化插件"""
        if self.plugin_lib is None:
            return

        try:
            # 初始化插件
            self.plugin_lib.initialize_gr_ml_infra_plugins()

            # 获取插件数量
            num_plugins = self.plugin_lib.get_num_registered_plugins()

            self.plugins_initialized = True
            logger.info(f"✅ TensorRT插件初始化成功，注册了 {num_plugins} 个插件")

        except Exception as e:
            logger.error(f"❌ TensorRT插件初始化失败: {e}")

    def is_available(self) -> bool:
        """检查插件是否可用"""
        return TENSORRT_AVAILABLE and self.plugins_initialized

    def get_num_registered_plugins(self) -> int:
        """获取已注册插件数量"""
        if not self.is_available():
            return 0

        try:
            return self.plugin_lib.get_num_registered_plugins()
        except Exception as e:
            logger.error(f"获取插件数量失败: {e}")
            return 0

    def create_onnx_custom_node(
        self,
        plugin_name: str,
        inputs: List[str],
        outputs: List[str],
        attributes: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """创建ONNX自定义节点配置"""
        if attributes is None:
            attributes = {}

        return {
            'op_type': plugin_name,
            'inputs': inputs,
            'outputs': outputs,
            'attributes': attributes,
            'domain': 'gr.ml.infra'
        }

    def get_supported_plugins(self) -> List[str]:
        """获取支持的插件列表"""
        return [
            'FusedAttentionLayerNorm',
            'HierarchicalSequenceFusion',
            'HSTUHierarchicalAttention',
            'InteractionTritonFast',
            'SequenceRecommendationInteraction'
        ]

# 全局插件管理器实例
_plugin_manager = None

def get_plugin_manager() -> GRMLInfraTensorRTPlugins:
    """获取全局插件管理器实例"""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = GRMLInfraTensorRTPlugins()
    return _plugin_manager

def initialize_plugins(plugin_lib_path: Optional[str] = None) -> bool:
    """初始化TensorRT插件"""
    global _plugin_manager
    _plugin_manager = GRMLInfraTensorRTPlugins(plugin_lib_path)
    return _plugin_manager.is_available()

def is_plugins_available() -> bool:
    """检查插件是否可用"""
    return get_plugin_manager().is_available()

def get_num_registered_plugins() -> int:
    """获取已注册插件数量"""
    return get_plugin_manager().get_num_registered_plugins()

def create_fused_attention_layernorm_node(
    input_name: str,
    output_name: str,
    hidden_dim: int,
    num_heads: int,
    dropout_rate: float = 0.1,
    layer_norm_eps: float = 1e-5
) -> Dict[str, Any]:
    """创建融合注意力LayerNorm节点"""
    return get_plugin_manager().create_onnx_custom_node(
        plugin_name='FusedAttentionLayerNorm',
        inputs=[input_name],
        outputs=[output_name],
        attributes={
            'hidden_dim': hidden_dim,
            'num_heads': num_heads,
            'dropout_rate': dropout_rate,
            'layer_norm_eps': layer_norm_eps
        }
    )

def create_hierarchical_sequence_fusion_node(
    input_name: str,
    output_name: str,
    hidden_dim: int,
    num_levels: int = 3
) -> Dict[str, Any]:
    """创建层次化序列融合节点"""
    return get_plugin_manager().create_onnx_custom_node(
        plugin_name='HierarchicalSequenceFusion',
        inputs=[input_name],
        outputs=[output_name],
        attributes={
            'hidden_dim': hidden_dim,
            'num_levels': num_levels
        }
    )

def create_interaction_triton_fast_node(
    input_name: str,
    output_name: str,
    num_features: int,
    embedding_dim: int
) -> Dict[str, Any]:
    """创建快速交互算子节点"""
    return get_plugin_manager().create_onnx_custom_node(
        plugin_name='InteractionTritonFast',
        inputs=[input_name],
        outputs=[output_name],
        attributes={
            'num_features': num_features,
            'embedding_dim': embedding_dim
        }
    )

if __name__ == "__main__":
    # 测试插件加载
    print("🔧 测试GR-ML-infra TensorRT插件加载...")

    if initialize_plugins():
        print(f"✅ 插件加载成功，注册了 {get_num_registered_plugins()} 个插件")

        # 测试创建自定义节点
        node = create_fused_attention_layernorm_node(
            input_name="input",
            output_name="output",
            hidden_dim=1024,
            num_heads=16
        )
        print(f"🎯 融合注意力节点配置: {node}")

    else:
        print("❌ 插件加载失败")