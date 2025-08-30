#!/bin/bash

# GR Inference Optimization Framework - 依赖安装脚本
# 此脚本安装项目所需的所有依赖

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查系统要求
check_system_requirements() {
    log_info "检查系统要求..."
    
    # 检查Python版本
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    log_info "Python版本: $python_version"
    
    # 检查CUDA
    if command -v nvidia-smi &> /dev/null; then
        cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
        log_info "CUDA驱动版本: $cuda_version"
    else
        log_warning "未检测到NVIDIA GPU或CUDA驱动"
    fi
    
    # 检查Docker
    if command -v docker &> /dev/null; then
        docker_version=$(docker --version)
        log_info "Docker版本: $docker_version"
    else
        log_warning "未安装Docker"
    fi
}

# 安装Python依赖
install_python_dependencies() {
    log_info "安装Python依赖..."
    
    # 升级pip
    python3 -m pip install --upgrade pip
    
    # 安装基础依赖
    log_info "安装基础依赖..."
    pip install -r requirements.txt
    
    # 检查是否安装开发依赖
    if [ "$1" = "--dev" ]; then
        log_info "安装开发依赖..."
        pip install -r requirements-dev.txt
    fi
    
    log_success "Python依赖安装完成"
}

# 安装TensorRT（可选）
install_tensorrt() {
    log_info "检查TensorRT安装..."
    
    if python3 -c "import tensorrt" 2>/dev/null; then
        log_success "TensorRT已安装"
    else
        log_warning "TensorRT未安装，请手动安装："
        echo "1. 访问 https://developer.nvidia.com/tensorrt"
        echo "2. 下载适合您CUDA版本的TensorRT"
        echo "3. 按照官方文档安装"
        echo "4. 将TensorRT路径添加到PYTHONPATH"
    fi
}

# 安装Triton Inference Server
install_triton() {
    log_info "检查Triton Inference Server..."
    
    if command -v tritonserver &> /dev/null; then
        log_success "Triton Inference Server已安装"
    else
        log_warning "Triton Inference Server未安装，建议使用Docker运行："
        echo "docker pull nvcr.io/nvidia/tritonserver:23.12-py3"
    fi
}

# 安装CUTLASS（可选）
install_cutlass() {
    log_info "检查CUTLASS..."
    
    if [ -d "/usr/local/cuda/include/cutlass" ]; then
        log_success "CUTLASS已安装"
    else
        log_warning "CUTLASS未安装，如需使用请手动安装："
        echo "git clone https://github.com/NVIDIA/cutlass.git"
        echo "cd cutlass"
        echo "mkdir build && cd build"
        echo "cmake .. -DCUTLASS_NVCC_ARCHS=80"
        echo "make -j$(nproc)"
        echo "sudo make install"
    fi
}

# 验证安装
verify_installation() {
    log_info "验证安装..."
    
    # 测试PyTorch
    if python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"; then
        log_success "PyTorch安装成功"
    else
        log_error "PyTorch安装失败"
        return 1
    fi
    
    # 测试ONNX
    if python3 -c "import onnx; print(f'ONNX版本: {onnx.__version__}')"; then
        log_success "ONNX安装成功"
    else
        log_error "ONNX安装失败"
        return 1
    fi
    
    # 测试Triton客户端
    if python3 -c "import tritonclient.http; print('Triton客户端安装成功')"; then
        log_success "Triton客户端安装成功"
    else
        log_error "Triton客户端安装失败"
        return 1
    fi
    
    log_success "所有核心依赖验证通过"
}

# 创建虚拟环境（可选）
create_virtual_env() {
    if [ "$1" = "--venv" ]; then
        log_info "创建虚拟环境..."
        
        if command -v conda &> /dev/null; then
            conda create -n gr-inference python=3.9 -y
            conda activate gr-inference
            log_success "Conda虚拟环境创建成功"
        else
            python3 -m venv gr-inference-env
            source gr-inference-env/bin/activate
            log_success "Python虚拟环境创建成功"
        fi
    fi
}

# 主函数
main() {
    log_info "开始安装GR推理优化框架依赖..."
    
    # 检查系统要求
    check_system_requirements
    
    # 创建虚拟环境（如果指定）
    if [ "$1" = "--venv" ]; then
        create_virtual_env "$1"
    fi
    
    # 安装Python依赖
    install_python_dependencies "$1"
    
    # 安装可选依赖
    install_tensorrt
    install_triton
    install_cutlass
    
    # 验证安装
    verify_installation
    
    log_success "依赖安装完成！"
    log_info "下一步："
    echo "1. 运行测试: python -m pytest tests/"
    echo "2. 导出模型: python src/export_onnx.py"
    echo "3. 构建引擎: python src/build_engine.py"
    echo "4. 启动服务: ./scripts/run_server.sh"
}

# 显示帮助
show_help() {
    echo "GR推理优化框架 - 依赖安装脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --dev     安装开发依赖"
    echo "  --venv    创建虚拟环境"
    echo "  --help    显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                    # 安装基础依赖"
    echo "  $0 --dev              # 安装所有依赖（包括开发依赖）"
    echo "  $0 --venv --dev       # 创建虚拟环境并安装所有依赖"
}

# 解析参数
case "$1" in
    --help)
        show_help
        exit 0
        ;;
    --dev|--venv|"")
        main "$1"
        ;;
    *)
        log_error "未知参数: $1"
        show_help
        exit 1
        ;;
esac
