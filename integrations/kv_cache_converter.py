#!/usr/bin/env python3
"""
KV Cache格式转换工具

实现TensorRT/HSTU格式与vLLM格式之间的KV Cache转换
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class KVCacheConverter:
    """KV Cache格式转换器"""

    def __init__(self):
        self.supported_formats = ['hstu_tensorrt', 'vllm_paged', 'standard']
        logger.info("✅ KV Cache转换器初始化完成")

    def convert_tensorrt_to_vllm(
        self,
        tensorrt_kv_cache: Dict[str, torch.Tensor],
        block_size: int = 16,
        max_seq_len: Optional[int] = None
    ) -> Dict[str, Any]:
        """将TensorRT/HSTU格式的KV Cache转换为vLLM格式

        Args:
            tensorrt_kv_cache: TensorRT格式的KV Cache
            block_size: vLLM的block大小
            max_seq_len: 最大序列长度

        Returns:
            vLLM格式的KV Cache字典
        """

        logger.info("🔄 开始转换KV Cache: TensorRT -> vLLM")

        if not tensorrt_kv_cache:
            logger.warning("输入KV Cache为空")
            return {}

        try:
            vllm_kv_cache = {}
            conversion_info = {
                'source_format': 'hstu_tensorrt',
                'target_format': 'vllm_paged',
                'block_size': block_size,
                'layers_converted': 0,
                'total_blocks': 0
            }

            # 遍历每一层
            for layer_name, layer_cache in tensorrt_kv_cache.items():
                if not isinstance(layer_cache, dict) or 'key' not in layer_cache or 'value' not in layer_cache:
                    logger.warning(f"跳过无效层: {layer_name}")
                    continue

                key_tensor = layer_cache['key']
                value_tensor = layer_cache['value']

                # 验证输入格式 [batch_size, seq_len, num_heads, head_dim]
                if len(key_tensor.shape) != 4:
                    logger.warning(f"层 {layer_name} Key张量形状不正确: {key_tensor.shape}")
                    continue

                batch_size, seq_len, num_heads, head_dim = key_tensor.shape

                # 转换为vLLM的block格式
                vllm_layer_cache = self._convert_layer_to_vllm_blocks(
                    key_tensor, value_tensor, block_size, layer_name
                )

                if vllm_layer_cache:
                    vllm_kv_cache[layer_name] = vllm_layer_cache
                    conversion_info['layers_converted'] += 1
                    conversion_info['total_blocks'] += vllm_layer_cache.get('num_blocks', 0)

            conversion_info['success'] = len(vllm_kv_cache) > 0
            vllm_kv_cache['_conversion_info'] = conversion_info

            logger.info(f"✅ KV Cache转换完成: {conversion_info['layers_converted']}层, {conversion_info['total_blocks']}个blocks")

            return vllm_kv_cache

        except Exception as e:
            logger.error(f"❌ KV Cache转换失败: {e}")
            return {}

    def _convert_layer_to_vllm_blocks(
        self,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
        block_size: int,
        layer_name: str
    ) -> Dict[str, Any]:
        """将单层的K, V张量转换为vLLM block格式"""

        try:
            batch_size, seq_len, num_heads, head_dim = key_tensor.shape

            # 计算需要的block数量
            num_blocks = (seq_len + block_size - 1) // block_size
            padded_seq_len = num_blocks * block_size

            # Padding到block边界
            if seq_len < padded_seq_len:
                padding_size = padded_seq_len - seq_len

                # 创建padding张量
                key_padding = torch.zeros(
                    batch_size, padding_size, num_heads, head_dim,
                    dtype=key_tensor.dtype, device=key_tensor.device
                )
                value_padding = torch.zeros(
                    batch_size, padding_size, num_heads, head_dim,
                    dtype=value_tensor.dtype, device=value_tensor.device
                )

                # 拼接padding
                key_padded = torch.cat([key_tensor, key_padding], dim=1)
                value_padded = torch.cat([value_tensor, value_padding], dim=1)
            else:
                key_padded = key_tensor
                value_padded = value_tensor

            # 重塑为block格式: [batch_size, num_blocks, block_size, num_heads, head_dim]
            key_blocks = key_padded.view(batch_size, num_blocks, block_size, num_heads, head_dim)
            value_blocks = value_padded.view(batch_size, num_blocks, block_size, num_heads, head_dim)

            # vLLM通常使用第一个batch（简化处理）
            # 实际生产环境可能需要处理整个batch
            key_vllm = key_blocks[0]  # [num_blocks, block_size, num_heads, head_dim]
            value_vllm = value_blocks[0]

            # 转换为vLLM期望的维度顺序: [num_blocks, num_heads, block_size, head_dim]
            key_vllm = key_vllm.transpose(1, 2)  # [num_blocks, num_heads, block_size, head_dim]
            value_vllm = value_vllm.transpose(1, 2)

            return {
                'key_blocks': key_vllm,
                'value_blocks': value_vllm,
                'block_info': {
                    'num_blocks': num_blocks,
                    'block_size': block_size,
                    'num_heads': num_heads,
                    'head_dim': head_dim,
                    'original_seq_len': seq_len,
                    'padded_seq_len': padded_seq_len,
                    'padding_size': padded_seq_len - seq_len
                },
                'metadata': {
                    'layer_name': layer_name,
                    'original_shape': list(key_tensor.shape),
                    'block_shape': list(key_vllm.shape),
                    'dtype': str(key_tensor.dtype),
                    'device': str(key_tensor.device)
                }
            }

        except Exception as e:
            logger.error(f"❌ 层 {layer_name} 转换失败: {e}")
            return {}

    def convert_vllm_to_tensorrt(
        self,
        vllm_kv_cache: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """将vLLM格式的KV Cache转换回TensorRT格式"""

        logger.info("🔄 开始转换KV Cache: vLLM -> TensorRT")

        if not vllm_kv_cache:
            logger.warning("输入vLLM KV Cache为空")
            return {}

        try:
            tensorrt_kv_cache = {}

            for layer_name, layer_cache in vllm_kv_cache.items():
                if layer_name.startswith('_'):  # 跳过元数据字段
                    continue

                if not isinstance(layer_cache, dict):
                    continue

                # 提取block数据
                key_blocks = layer_cache.get('key_blocks')
                value_blocks = layer_cache.get('value_blocks')
                block_info = layer_cache.get('block_info', {})

                if key_blocks is None or value_blocks is None:
                    logger.warning(f"跳过无效层: {layer_name}")
                    continue

                # 转换回原始格式
                original_key, original_value = self._convert_vllm_blocks_to_tensor(
                    key_blocks, value_blocks, block_info, layer_name
                )

                if original_key is not None and original_value is not None:
                    tensorrt_kv_cache[layer_name] = {
                        'key': original_key,
                        'value': original_value
                    }

            logger.info(f"✅ vLLM -> TensorRT转换完成: {len(tensorrt_kv_cache)}层")
            return tensorrt_kv_cache

        except Exception as e:
            logger.error(f"❌ vLLM -> TensorRT转换失败: {e}")
            return {}

    def _convert_vllm_blocks_to_tensor(
        self,
        key_blocks: torch.Tensor,
        value_blocks: torch.Tensor,
        block_info: Dict[str, int],
        layer_name: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """将vLLM blocks转换回原始张量格式"""

        try:
            # 获取block信息
            num_blocks = block_info.get('num_blocks', key_blocks.shape[0])
            block_size = block_info.get('block_size', key_blocks.shape[2])
            num_heads = block_info.get('num_heads', key_blocks.shape[1])
            head_dim = block_info.get('head_dim', key_blocks.shape[3])
            original_seq_len = block_info.get('original_seq_len')

            # vLLM格式: [num_blocks, num_heads, block_size, head_dim]
            # 转换为: [num_blocks, block_size, num_heads, head_dim]
            key_reshaped = key_blocks.transpose(1, 2)  # [num_blocks, block_size, num_heads, head_dim]
            value_reshaped = value_blocks.transpose(1, 2)

            # 重塑为序列格式: [1, padded_seq_len, num_heads, head_dim]
            padded_seq_len = num_blocks * block_size
            key_sequence = key_reshaped.view(1, padded_seq_len, num_heads, head_dim)
            value_sequence = value_reshaped.view(1, padded_seq_len, num_heads, head_dim)

            # 如果有原始序列长度信息，去除padding
            if original_seq_len and original_seq_len < padded_seq_len:
                key_sequence = key_sequence[:, :original_seq_len, :, :]
                value_sequence = value_sequence[:, :original_seq_len, :, :]

            return key_sequence, value_sequence

        except Exception as e:
            logger.error(f"❌ 层 {layer_name} blocks转换失败: {e}")
            return None, None

    def validate_kv_cache_format(
        self,
        kv_cache: Dict[str, Any],
        expected_format: str
    ) -> bool:
        """验证KV Cache格式是否正确"""

        if expected_format not in self.supported_formats:
            logger.error(f"不支持的格式: {expected_format}")
            return False

        if not kv_cache:
            logger.warning("KV Cache为空")
            return False

        try:
            if expected_format == 'hstu_tensorrt':
                return self._validate_tensorrt_format(kv_cache)
            elif expected_format == 'vllm_paged':
                return self._validate_vllm_format(kv_cache)
            else:
                logger.warning(f"格式验证未实现: {expected_format}")
                return True

        except Exception as e:
            logger.error(f"❌ KV Cache格式验证失败: {e}")
            return False

    def _validate_tensorrt_format(self, kv_cache: Dict[str, Any]) -> bool:
        """验证TensorRT格式的KV Cache"""

        for layer_name, layer_cache in kv_cache.items():
            if not isinstance(layer_cache, dict):
                logger.error(f"层 {layer_name} 不是字典格式")
                return False

            if 'key' not in layer_cache or 'value' not in layer_cache:
                logger.error(f"层 {layer_name} 缺少key或value")
                return False

            key_tensor = layer_cache['key']
            value_tensor = layer_cache['value']

            if not isinstance(key_tensor, torch.Tensor) or not isinstance(value_tensor, torch.Tensor):
                logger.error(f"层 {layer_name} key/value不是torch.Tensor")
                return False

            if len(key_tensor.shape) != 4 or len(value_tensor.shape) != 4:
                logger.error(f"层 {layer_name} 张量维度不正确")
                return False

            if key_tensor.shape != value_tensor.shape:
                logger.error(f"层 {layer_name} key和value形状不匹配")
                return False

        return True

    def _validate_vllm_format(self, kv_cache: Dict[str, Any]) -> bool:
        """验证vLLM格式的KV Cache"""

        for layer_name, layer_cache in kv_cache.items():
            if layer_name.startswith('_'):  # 跳过元数据
                continue

            if not isinstance(layer_cache, dict):
                logger.error(f"vLLM层 {layer_name} 不是字典格式")
                return False

            required_fields = ['key_blocks', 'value_blocks', 'block_info']
            for field in required_fields:
                if field not in layer_cache:
                    logger.error(f"vLLM层 {layer_name} 缺少字段: {field}")
                    return False

            key_blocks = layer_cache['key_blocks']
            value_blocks = layer_cache['value_blocks']

            if not isinstance(key_blocks, torch.Tensor) or not isinstance(value_blocks, torch.Tensor):
                logger.error(f"vLLM层 {layer_name} blocks不是torch.Tensor")
                return False

            if len(key_blocks.shape) != 4 or len(value_blocks.shape) != 4:
                logger.error(f"vLLM层 {layer_name} blocks维度不正确")
                return False

        return True

    def get_kv_cache_stats(self, kv_cache: Dict[str, Any]) -> Dict[str, Any]:
        """获取KV Cache统计信息"""

        if not kv_cache:
            return {'empty': True}

        stats = {
            'total_layers': 0,
            'total_parameters': 0,
            'memory_usage_mb': 0,
            'format_type': 'unknown',
            'layers': {}
        }

        try:
            # 检测格式类型
            if self.validate_kv_cache_format(kv_cache, 'hstu_tensorrt'):
                stats['format_type'] = 'hstu_tensorrt'
            elif self.validate_kv_cache_format(kv_cache, 'vllm_paged'):
                stats['format_type'] = 'vllm_paged'

            # 统计每层信息
            for layer_name, layer_cache in kv_cache.items():
                if layer_name.startswith('_'):  # 跳过元数据
                    continue

                layer_stats = self._get_layer_stats(layer_cache, stats['format_type'])
                if layer_stats:
                    stats['layers'][layer_name] = layer_stats
                    stats['total_layers'] += 1
                    stats['total_parameters'] += layer_stats.get('parameters', 0)
                    stats['memory_usage_mb'] += layer_stats.get('memory_mb', 0)

            return stats

        except Exception as e:
            logger.error(f"❌ 获取KV Cache统计失败: {e}")
            return {'error': str(e)}

    def _get_layer_stats(self, layer_cache: Dict[str, Any], format_type: str) -> Dict[str, Any]:
        """获取单层统计信息"""

        try:
            if format_type == 'hstu_tensorrt':
                key_tensor = layer_cache.get('key')
                value_tensor = layer_cache.get('value')

                if key_tensor is not None and value_tensor is not None:
                    key_params = key_tensor.numel()
                    value_params = value_tensor.numel()
                    total_params = key_params + value_params

                    # 估算内存使用（字节）
                    element_size = key_tensor.element_size()
                    memory_bytes = total_params * element_size
                    memory_mb = memory_bytes / (1024 * 1024)

                    return {
                        'shape': list(key_tensor.shape),
                        'parameters': total_params,
                        'memory_mb': memory_mb,
                        'dtype': str(key_tensor.dtype),
                        'device': str(key_tensor.device)
                    }

            elif format_type == 'vllm_paged':
                key_blocks = layer_cache.get('key_blocks')
                value_blocks = layer_cache.get('value_blocks')

                if key_blocks is not None and value_blocks is not None:
                    key_params = key_blocks.numel()
                    value_params = value_blocks.numel()
                    total_params = key_params + value_params

                    element_size = key_blocks.element_size()
                    memory_bytes = total_params * element_size
                    memory_mb = memory_bytes / (1024 * 1024)

                    block_info = layer_cache.get('block_info', {})

                    return {
                        'block_shape': list(key_blocks.shape),
                        'parameters': total_params,
                        'memory_mb': memory_mb,
                        'dtype': str(key_blocks.dtype),
                        'device': str(key_blocks.device),
                        'num_blocks': block_info.get('num_blocks', 0),
                        'block_size': block_info.get('block_size', 0)
                    }

            return {}

        except Exception as e:
            logger.error(f"❌ 获取层统计失败: {e}")
            return {}


# 全局转换器实例
_kv_cache_converter = None


def get_kv_cache_converter() -> KVCacheConverter:
    """获取全局KV Cache转换器实例"""
    global _kv_cache_converter
    if _kv_cache_converter is None:
        _kv_cache_converter = KVCacheConverter()
    return _kv_cache_converter


def convert_tensorrt_to_vllm(
    tensorrt_kv_cache: Dict[str, torch.Tensor],
    block_size: int = 16
) -> Dict[str, Any]:
    """便捷函数：TensorRT -> vLLM转换"""
    converter = get_kv_cache_converter()
    return converter.convert_tensorrt_to_vllm(tensorrt_kv_cache, block_size)


def convert_vllm_to_tensorrt(vllm_kv_cache: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """便捷函数：vLLM -> TensorRT转换"""
    converter = get_kv_cache_converter()
    return converter.convert_vllm_to_tensorrt(vllm_kv_cache)


if __name__ == "__main__":
    # 测试KV Cache转换器
    logger.info("🧪 测试KV Cache转换器...")

    # 创建测试用的TensorRT格式KV Cache
    batch_size, seq_len, num_heads, head_dim = 1, 64, 12, 64
    num_layers = 3

    test_tensorrt_cache = {}
    for i in range(num_layers):
        test_tensorrt_cache[f'layer_{i}'] = {
            'key': torch.randn(batch_size, seq_len, num_heads, head_dim),
            'value': torch.randn(batch_size, seq_len, num_heads, head_dim)
        }

    converter = KVCacheConverter()

    # 测试TensorRT -> vLLM转换
    vllm_cache = converter.convert_tensorrt_to_vllm(test_tensorrt_cache, block_size=16)
    print(f"TensorRT -> vLLM转换结果: {len(vllm_cache)}层")

    # 测试vLLM -> TensorRT转换
    converted_back = converter.convert_vllm_to_tensorrt(vllm_cache)
    print(f"vLLM -> TensorRT转换结果: {len(converted_back)}层")

    # 获取统计信息
    stats = converter.get_kv_cache_stats(test_tensorrt_cache)
    print(f"TensorRT Cache统计: {stats}")

    vllm_stats = converter.get_kv_cache_stats(vllm_cache)
    print(f"vLLM Cache统计: {vllm_stats}")

    print("✅ KV Cache转换器测试完成")