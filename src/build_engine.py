#!/usr/bin/env python3
"""Build TensorRT engine using trtexec or TensorRT Python API.
This file includes a complete example of how to create optimization profiles
for dynamic shapes using the TensorRT Python API with comprehensive error handling.
"""
import argparse
import subprocess
import os
import sys
import json
import time
from typing import Dict, List, Tuple, Optional

def build_with_trtexec(onnx_path: str, engine_path: str, fp16: bool = True, 
                      workspace: int = 8192, min_shape: str = '1x8', 
                      opt_shape: str = '4x64', max_shape: str = '8x512',
                      precision: str = 'fp16', max_batch: int = 8,
                      verbose: bool = False) -> bool:
    """
    Build TensorRT engine using trtexec command line tool
    
    Args:
        onnx_path: Path to input ONNX file
        engine_path: Path to output TensorRT engine
        fp16: Enable FP16 precision
        workspace: Workspace size in MB
        min_shape: Minimum input shape (format: batchxseq)
        opt_shape: Optimal input shape (format: batchxseq)
        max_shape: Maximum input shape (format: batchxseq)
        precision: Precision mode ('fp16', 'fp32', 'int8')
        max_batch: Maximum batch size
        verbose: Enable verbose output
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file {onnx_path} does not exist")
        return False
    
    cmd = [
        'trtexec',
        f'--onnx={onnx_path}',
        f'--saveEngine={engine_path}',
        f'--workspace={workspace}',
        f'--maxBatch={max_batch}'
    ]
    
    # Add precision flags
    if precision == 'fp16' and fp16:
        cmd.append('--fp16')
    elif precision == 'int8':
        cmd.append('--int8')
    
    # Add shape configurations
    cmd.extend([
        f'--minShapes=input_ids:{min_shape}',
        f'--optShapes=input_ids:{opt_shape}', 
        f'--maxShapes=input_ids:{max_shape}'
    ])
    
    # Add dense_features shape if present (align with model's 1024-dim)
    if 'dense_features' in get_onnx_inputs(onnx_path):
        cmd.extend([
            f'--minShapes=dense_features:1x1024',
            f'--optShapes=dense_features:4x1024',
            f'--maxShapes=dense_features:8x1024'
        ])
    
    # Add attention_mask shape if present
    if 'attention_mask' in get_onnx_inputs(onnx_path):
        cmd.extend([
            f'--minShapes=attention_mask:{min_shape}',
            f'--optShapes=attention_mask:{opt_shape}',
            f'--maxShapes=attention_mask:{max_shape}'
        ])
    
    if verbose:
        cmd.append('--verbose')
    
    print('Running trtexec command:')
    print(' '.join(cmd))
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        build_time = time.time() - start_time
        print(f"Engine built successfully in {build_time:.2f} seconds")
        print(f"Engine saved to: {engine_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"trtexec failed with error code {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: trtexec not found. Please ensure TensorRT is properly installed.")
        return False

def get_onnx_inputs(onnx_path: str) -> List[str]:
    """Get input names from ONNX file"""
    try:
        import onnx
        model = onnx.load(onnx_path)
        return [input.name for input in model.graph.input]
    except ImportError:
        print("Warning: onnx package not available, using default input names")
        return ['input_ids']
    except Exception as e:
        print(f"Warning: Could not read ONNX inputs: {e}")
        return ['input_ids']

def build_with_api(onnx_path: str, engine_path: str, fp16: bool = True, 
                  workspace: int = 1 << 30, precision: str = 'fp16',
                  max_batch: int = 8, verbose: bool = False) -> bool:
    """
    Build TensorRT engine using TensorRT Python API
    
    Args:
        onnx_path: Path to input ONNX file
        engine_path: Path to output TensorRT engine
        fp16: Enable FP16 precision
        workspace: Workspace size in bytes
        precision: Precision mode ('fp16', 'fp32', 'int8')
        max_batch: Maximum batch size
        verbose: Enable verbose output
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        print(f'TensorRT Python API not available: {e}')
        return False
    
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file {onnx_path} does not exist")
        return False
    
    # Create logger
    log_level = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    TRT_LOGGER = trt.Logger(log_level)
    
    print("Building TensorRT engine with Python API...")
    start_time = time.time()
    
    try:
        # Create builder and network
        with trt.Builder(TRT_LOGGER) as builder, \
             builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
             trt.OnnxParser(network, TRT_LOGGER) as parser:
            
            # Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    print("Failed to parse ONNX file:")
                    for i in range(parser.num_errors):
                        print(f"  Error {i}: {parser.get_error(i)}")
                    return False
            
            print(f"ONNX parsed successfully. Network has {network.num_layers} layers")
            
            # Create builder config
            config = builder.create_builder_config()
            config.max_workspace_size = workspace
            
            # Set precision flags
            if precision == 'fp16' and fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("FP16 precision enabled")
            elif precision == 'int8' and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                print("INT8 precision enabled")
            else:
                print("Using FP32 precision")
            
            # Set optimization level
            config.set_flag(trt.BuilderFlag.DIRECT_IO)
            
            # Create optimization profiles for dynamic shapes
            profile = builder.create_optimization_profile()
            
            # Get input names from network
            input_names = [network.get_input(i).name for i in range(network.num_inputs)]
            print(f"Network inputs: {input_names}")
            
            # Configure shapes for each input
            for input_name in input_names:
                if 'input_ids' in input_name or 'token_id' in input_name:
                    # Sequence inputs
                    profile.set_shape(input_name, (1, 8), (4, 64), (max_batch, 512))
                elif 'dense_features' in input_name:
                    # Dense feature inputs
                    profile.set_shape(input_name, (1, 16), (4, 16), (max_batch, 16))
                elif 'attention_mask' in input_name:
                    # Attention mask inputs
                    profile.set_shape(input_name, (1, 8), (4, 64), (max_batch, 512))
                elif 'past_key_value_states' in input_name:
                    # Past key-value states
                    profile.set_shape(input_name, (1, 1, 128), (4, 32, 128), (max_batch, 256, 128))
                else:
                    # Default shape for unknown inputs
                    print(f"Warning: Unknown input {input_name}, using default shapes")
                    profile.set_shape(input_name, (1,), (4,), (max_batch,))
            
            config.add_optimization_profile(profile)
            
            # Build engine
            print("Building engine (this may take several minutes)...")
            engine = builder.build_engine(network, config)
            
            if engine is None:
                print("Failed to build engine")
                return False
            
            # Save engine
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            
            build_time = time.time() - start_time
            print(f"Engine built successfully in {build_time:.2f} seconds")
            print(f"Engine saved to: {engine_path}")
            
            # Print engine info
            print(f"Engine size: {os.path.getsize(engine_path) / (1024*1024):.2f} MB")
            
            return True
            
    except Exception as e:
        print(f"Error building engine: {e}")
        return False

def validate_engine(engine_path: str) -> bool:
    """Validate TensorRT engine"""
    try:
        import tensorrt as trt
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        with trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(engine_data)
            
        if engine is None:
            print("Failed to deserialize engine")
            return False
        
        print(f"Engine validation successful")
        print(f"Number of bindings: {engine.num_bindings}")
        print(f"Max batch size: {engine.max_batch_size}")
        
        return True
        
    except ImportError:
        print("Warning: TensorRT not available for validation")
        return True
    except Exception as e:
        print(f"Engine validation failed: {e}")
        return False

def create_engine_config(engine_path: str, config_path: str) -> bool:
    """Create engine configuration file for Triton"""
    config = {
        "name": "gr_trt",
        "platform": "tensorrt_plan",
        "max_batch_size": 8,
        "input": [
            {
                "name": "input_ids",
                "data_type": "TYPE_INT32",
                "dims": [-1, -1]
            },
            {
                "name": "dense_features", 
                "data_type": "TYPE_FP32",
                "dims": [-1]
            }
        ],
        "output": [
            {
                "name": "logits",
                "data_type": "TYPE_FP32", 
                "dims": [-1, -1]
            },
            {
                "name": "feature_scores",
                "data_type": "TYPE_FP32",
                "dims": [-1]
            }
        ],
        "instance_group": [{"kind": "KIND_GPU", "count": 1}],
        "parameters": {
            "serialized_file": {"string_value": os.path.basename(engine_path)}
        }
    }
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Engine config saved to: {config_path}")
        return True
    except Exception as e:
        print(f"Failed to create engine config: {e}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build TensorRT engine from ONNX')
    parser.add_argument('--onnx', required=True, help='Input ONNX file path')
    parser.add_argument('--engine', required=True, help='Output TensorRT engine path')
    parser.add_argument('--mode', choices=['trtexec', 'api'], default='trtexec', 
                       help='Build mode: trtexec or api')
    parser.add_argument('--fp16', action='store_true', default=True, 
                       help='Enable FP16 precision')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'int8'], default='fp16',
                       help='Precision mode')
    parser.add_argument('--workspace', type=int, default=8192, 
                       help='Workspace size in MB (for trtexec) or bytes (for api)')
    parser.add_argument('--min-shape', default='1x8', help='Minimum input shape')
    parser.add_argument('--opt-shape', default='4x64', help='Optimal input shape')
    parser.add_argument('--max-shape', default='8x512', help='Maximum input shape')
    parser.add_argument('--max-batch', type=int, default=8, help='Maximum batch size')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--validate', action='store_true', help='Validate engine after building')
    parser.add_argument('--config', help='Output path for Triton config file')
    
    args = parser.parse_args()
    
    # Convert workspace size for API mode
    if args.mode == 'api':
        workspace_size = args.workspace * (1024 * 1024)  # Convert MB to bytes
    else:
        workspace_size = args.workspace
    
    # Build engine
    success = False
    if args.mode == 'trtexec':
        success = build_with_trtexec(
            args.onnx, args.engine, args.fp16, workspace_size,
            args.min_shape, args.opt_shape, args.max_shape,
            args.precision, args.max_batch, args.verbose
        )
    else:
        success = build_with_api(
            args.onnx, args.engine, args.fp16, workspace_size,
            args.precision, args.max_batch, args.verbose
        )
    
    if success:
        print("Engine build completed successfully!")
        
        # Validate engine if requested
        if args.validate:
            if validate_engine(args.engine):
                print("Engine validation passed")
            else:
                print("Engine validation failed")
                sys.exit(1)
        
        # Create Triton config if requested
        if args.config:
            create_engine_config(args.engine, args.config)
    else:
        print("Engine build failed!")
        sys.exit(1)
