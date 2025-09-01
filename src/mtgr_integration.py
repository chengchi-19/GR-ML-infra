#!/usr/bin/env python3
"""
MTGR模型集成方案
提供多种导入方式：开源模型优先，自实现作为备选
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import os

logger = logging.getLogger(__name__)

class MTGRModelLoader:
    """
    MTGR模型加载器
    优先使用开源实现，回退到自实现
    """
    
    def __init__(self):
        self.model = None
        self.model_source = None
        self.load_success = False
    
    def load_model(self, 
                  model_path: str = None,
                  model_config: Dict[str, Any] = None,
                  use_open_source: bool = True) -> bool:
        """
        加载MTGR模型
        
        Args:
            model_path: 模型路径
            model_config: 模型配置
            use_open_source: 是否优先使用开源实现
            
        Returns:
            是否加载成功
        """
        
        if use_open_source:
            # 尝试加载开源MTGR模型
            if self._try_load_open_source_mtgr(model_path):
                return True
        
        # 回退到自实现
        if self._try_load_custom_mtgr(model_config):
            return True
        
        logger.error("所有MTGR模型加载方式都失败了")
        return False
    
    def _try_load_open_source_mtgr(self, model_path: str = None) -> bool:
        """尝试加载开源MTGR模型"""
        
        # 方式1: 从Hugging Face加载
        try:
            from transformers import AutoModel, AutoTokenizer
            
            if model_path:
                model_name = model_path
            else:
                # 尝试常见的MTGR模型名称
                possible_names = [
                    "meituan/mtgr-large",
                    "meituan/mtgr-base", 
                    "mtgr-large",
                    "mtgr-base"
                ]
                
                for name in possible_names:
                    try:
                        logger.info(f"尝试加载开源MTGR模型: {name}")
                        self.model = AutoModel.from_pretrained(name)
                        self.model_source = f"huggingface:{name}"
                        self.load_success = True
                        logger.info(f"成功加载开源MTGR模型: {name}")
                        return True
                    except Exception as e:
                        logger.debug(f"加载 {name} 失败: {e}")
                        continue
            
            return False
            
        except ImportError:
            logger.debug("transformers库未安装")
            return False
        except Exception as e:
            logger.debug(f"开源MTGR加载失败: {e}")
            return False
    
    def _try_load_custom_mtgr(self, model_config: Dict[str, Any] = None) -> bool:
        """尝试加载自实现MTGR模型"""
        try:
            from src.mtgr_model import create_mtgr_model
            
            if model_config is None:
                model_config = {
                    'vocab_size': 50000,
                    'd_model': 1024,
                    'nhead': 16,
                    'num_layers': 24,
                    'd_ff': 4096,
                    'max_seq_len': 2048,
                    'num_features': 1024,
                    'user_profile_dim': 256,
                    'item_feature_dim': 512,
                    'dropout': 0.1
                }
            
            self.model = create_mtgr_model(model_config)
            self.model_source = "custom_implementation"
            self.load_success = True
            logger.info("成功加载自实现MTGR模型")
            return True
            
        except Exception as e:
            logger.debug(f"自实现MTGR加载失败: {e}")
            return False
    
    def get_model(self) -> Optional[nn.Module]:
        """获取加载的模型"""
        return self.model
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.load_success or self.model is None:
            return {"status": "not_loaded"}
        
        info = {
            "status": "loaded",
            "source": self.model_source,
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "model_type": type(self.model).__name__
        }
        
        # 添加模型特定信息
        if hasattr(self.model, 'config'):
            info["config"] = self.model.config
        
        return info

def create_mtgr_model_loader(**kwargs) -> MTGRModelLoader:
    """创建MTGR模型加载器的便捷函数"""
    loader = MTGRModelLoader()
    loader.load_model(**kwargs)
    return loader

# 兼容性接口
def create_mtgr_model(config: Dict[str, Any] = None, **kwargs) -> nn.Module:
    """
    创建MTGR模型的统一接口
    
    Args:
        config: 模型配置
        **kwargs: 其他参数
        
    Returns:
        MTGR模型实例
    """
    
    # 优先尝试开源模型
    loader = MTGRModelLoader()
    
    if loader.load_model(use_open_source=True):
        logger.info("使用开源MTGR模型")
        return loader.get_model()
    
    # 回退到自实现
    if loader.load_model(use_open_source=False, model_config=config):
        logger.info("使用自实现MTGR模型")
        return loader.get_model()
    
    # 如果都失败了，抛出异常
    raise RuntimeError("无法加载任何MTGR模型")

# 测试函数
if __name__ == "__main__":
    print("测试MTGR模型加载器...")
    
    # 创建加载器
    loader = create_mtgr_model_loader(use_open_source=True)
    
    # 获取模型信息
    info = loader.get_model_info()
    print(f"模型信息: {info}")
    
    if loader.load_success:
        model = loader.get_model()
        print(f"模型类型: {type(model)}")
        print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    else:
        print("模型加载失败")
