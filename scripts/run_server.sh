#!/bin/bash

# GR推理优化框架 - Triton服务器启动脚本

set -e

# 默认配置
TRITON_IMAGE=${TRITON_IMAGE:-"nvcr.io/nvidia/tritonserver:23.11-py3"}
MODEL_REPO=${MODEL_REPO:-"./triton_model_repo"}
HTTP_PORT=${HTTP_PORT:-8000}
GRPC_PORT=${GRPC_PORT:-8001}
METRICS_PORT=${METRICS_PORT:-8002}
GPU_DEVICES=${GPU_DEVICES:-"all"}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}
CONTAINER_NAME=${CONTAINER_NAME:-"gr-inference-server"}

# 颜色输出
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

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help              显示此帮助信息"
    echo "  -i, --image IMAGE       指定Triton镜像 (默认: $TRITON_IMAGE)"
    echo "  -m, --model-repo PATH   指定模型仓库路径 (默认: $MODEL_REPO)"
    echo "  -p, --http-port PORT    HTTP端口 (默认: $HTTP_PORT)"
    echo "  -g, --grpc-port PORT    gRPC端口 (默认: $GRPC_PORT)"
    echo "  -t, --metrics-port PORT 指标端口 (默认: $METRICS_PORT)"
    echo "  -d, --gpu-devices DEV   指定GPU设备 (默认: $GPU_DEVICES)"
    echo "  -l, --log-level LEVEL   日志级别 (默认: $LOG_LEVEL)"
    echo "  -n, --name NAME         容器名称 (默认: $CONTAINER_NAME)"
    echo "  -s, --stop              停止服务器"
    echo "  -r, --restart           重启服务器"
    echo "  -c, --clean             清理容器和镜像"
    echo ""
    echo "环境变量:"
    echo "  TRITON_IMAGE            Triton镜像"
    echo "  MODEL_REPO              模型仓库路径"
    echo "  HTTP_PORT               HTTP端口"
    echo "  GRPC_PORT               gRPC端口"
    echo "  METRICS_PORT            指标端口"
    echo "  GPU_DEVICES             GPU设备"
    echo "  LOG_LEVEL               日志级别"
    echo "  CONTAINER_NAME          容器名称"
    echo ""
    echo "示例:"
    echo "  $0"
    echo "  $0 -m ./models -p 8080 -g 8081"
    echo "  $0 -d 0,1 -l DEBUG"
    echo "  $0 -s"
}

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装或未在PATH中"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker守护进程未运行"
        exit 1
    fi
    
    if ! command -v nvidia-docker &> /dev/null && ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        log_warning "NVIDIA Docker支持不可用，将使用CPU模式"
        GPU_DEVICES=""
    fi
    
    log_success "依赖检查通过"
}

# 检查模型仓库
check_model_repo() {
    log_info "检查模型仓库: $MODEL_REPO"
    
    if [ ! -d "$MODEL_REPO" ]; then
        log_error "模型仓库不存在: $MODEL_REPO"
        exit 1
    fi
    
    # 检查必要的模型
    local required_models=("ensemble_model" "gr_trt" "interaction_python" "preprocess_py" "embedding_service")
    local missing_models=()
    
    for model in "${required_models[@]}"; do
        if [ ! -d "$MODEL_REPO/$model" ]; then
            missing_models+=("$model")
        fi
    done
    
    if [ ${#missing_models[@]} -gt 0 ]; then
        log_warning "缺少模型: ${missing_models[*]}"
        log_info "请确保所有模型都已正确配置"
    else
        log_success "模型仓库检查通过"
    fi
}

# 拉取Triton镜像
pull_triton_image() {
    log_info "拉取Triton镜像: $TRITON_IMAGE"
    
    if ! docker pull "$TRITON_IMAGE" > /dev/null 2>&1; then
        log_error "无法拉取Triton镜像: $TRITON_IMAGE"
        exit 1
    fi
    
    log_success "Triton镜像拉取完成"
}

# 启动服务器
start_server() {
    log_info "启动GR推理优化服务器..."
    
    # 构建Docker运行命令
    local docker_cmd="docker run"
    
    # 添加GPU支持
    if [ -n "$GPU_DEVICES" ] && [ "$GPU_DEVICES" != "none" ]; then
        docker_cmd="$docker_cmd --gpus $GPU_DEVICES"
    fi
    
    # 添加端口映射
    docker_cmd="$docker_cmd -p $HTTP_PORT:8000 -p $GRPC_PORT:8001 -p $METRICS_PORT:8002"
    
    # 添加卷挂载
    docker_cmd="$docker_cmd -v $(realpath $MODEL_REPO):/models"
    
    # 添加环境变量
    docker_cmd="$docker_cmd -e CUDA_VISIBLE_DEVICES=$GPU_DEVICES"
    docker_cmd="$docker_cmd -e TRITON_LOG_VERBOSE=1"
    
    # 添加容器名称
    docker_cmd="$docker_cmd --name $CONTAINER_NAME"
    
    # 添加镜像和命令
    docker_cmd="$docker_cmd $TRITON_IMAGE"
    docker_cmd="$docker_cmd tritonserver"
    docker_cmd="$docker_cmd --model-repository=/models"
    docker_cmd="$docker_cmd --strict-model-config=false"
    docker_cmd="$docker_cmd --log-verbose=1"
    docker_cmd="$docker_cmd --log-info=1"
    docker_cmd="$docker_cmd --log-warning=1"
    docker_cmd="$docker_cmd --log-error=1"
    docker_cmd="$docker_cmd --http-thread-count=8"
    docker_cmd="$docker_cmd --grpc-infer-allocation-pool-size=16"
    docker_cmd="$docker_cmd --cuda-memory-pool-size=67108864"
    docker_cmd="$docker_cmd --min-supported-compute-capability=7.0"
    
    log_info "执行命令: $docker_cmd"
    
    # 运行容器
    eval "$docker_cmd"
}

# 停止服务器
stop_server() {
    log_info "停止GR推理优化服务器..."
    
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        docker stop "$CONTAINER_NAME"
        docker rm "$CONTAINER_NAME"
        log_success "服务器已停止"
    else
        log_warning "服务器未运行"
    fi
}

# 重启服务器
restart_server() {
    log_info "重启GR推理优化服务器..."
    stop_server
    sleep 2
    start_server
}

# 清理资源
clean_resources() {
    log_info "清理Docker资源..."
    
    # 停止并删除容器
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
    fi
    
    # 删除相关镜像
    if docker images -q "$TRITON_IMAGE" | grep -q .; then
        docker rmi "$TRITON_IMAGE" 2>/dev/null || true
    fi
    
    log_success "资源清理完成"
}

# 检查服务器状态
check_server_status() {
    log_info "检查服务器状态..."
    
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        local container_id=$(docker ps -q -f name="$CONTAINER_NAME")
        local status=$(docker inspect --format='{{.State.Status}}' "$container_id")
        local health=$(docker inspect --format='{{.State.Health.Status}}' "$container_id" 2>/dev/null || echo "N/A")
        
        log_success "服务器状态: $status (健康状态: $health)"
        
        # 显示端口信息
        echo ""
        log_info "服务端点:"
        echo "  HTTP API:     http://localhost:$HTTP_PORT"
        echo "  gRPC API:     localhost:$GRPC_PORT"
        echo "  Metrics API:  http://localhost:$METRICS_PORT"
        echo "  Health Check: http://localhost:$HTTP_PORT/v2/health/ready"
        
        # 显示日志
        echo ""
        log_info "最近日志:"
        docker logs --tail 20 "$CONTAINER_NAME"
        
    else
        log_warning "服务器未运行"
    fi
}

# 显示服务器信息
show_server_info() {
    log_info "GR推理优化服务器信息"
    echo "=================================="
    echo "容器名称: $CONTAINER_NAME"
    echo "Triton镜像: $TRITON_IMAGE"
    echo "模型仓库: $MODEL_REPO"
    echo "HTTP端口: $HTTP_PORT"
    echo "gRPC端口: $GRPC_PORT"
    echo "指标端口: $METRICS_PORT"
    echo "GPU设备: $GPU_DEVICES"
    echo "日志级别: $LOG_LEVEL"
    echo ""
    
    # 显示系统信息
    log_info "系统信息:"
    echo "  Docker版本: $(docker --version)"
    echo "  CUDA版本: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null || echo "不可用")"
    echo "  GPU数量: $(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")"
    
    # 显示模型信息
    echo ""
    log_info "模型信息:"
    if [ -d "$MODEL_REPO" ]; then
        for model_dir in "$MODEL_REPO"/*/; do
            if [ -d "$model_dir" ]; then
                local model_name=$(basename "$model_dir")
                local config_file="$model_dir/config.pbtxt"
                if [ -f "$config_file" ]; then
                    echo "  ✓ $model_name"
                else
                    echo "  ✗ $model_name (缺少配置)"
                fi
            fi
        done
    else
        echo "  模型仓库不存在"
    fi
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -i|--image)
                TRITON_IMAGE="$2"
                shift 2
                ;;
            -m|--model-repo)
                MODEL_REPO="$2"
                shift 2
                ;;
            -p|--http-port)
                HTTP_PORT="$2"
                shift 2
                ;;
            -g|--grpc-port)
                GRPC_PORT="$2"
                shift 2
                ;;
            -t|--metrics-port)
                METRICS_PORT="$2"
                shift 2
                ;;
            -d|--gpu-devices)
                GPU_DEVICES="$2"
                shift 2
                ;;
            -l|--log-level)
                LOG_LEVEL="$2"
                shift 2
                ;;
            -n|--name)
                CONTAINER_NAME="$2"
                shift 2
                ;;
            -s|--stop)
                stop_server
                exit 0
                ;;
            -r|--restart)
                restart_server
                exit 0
                ;;
            -c|--clean)
                clean_resources
                exit 0
                ;;
            --status)
                check_server_status
                exit 0
                ;;
            --info)
                show_server_info
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 主函数
main() {
    log_info "GR推理优化框架 - Triton服务器管理"
    echo "=================================="
    
    # 解析参数
    parse_args "$@"
    
    # 显示服务器信息
    show_server_info
    
    # 检查依赖
    check_dependencies
    
    # 检查模型仓库
    check_model_repo
    
    # 拉取镜像
    pull_triton_image
    
    # 启动服务器
    start_server
}

# 运行主函数
main "$@"
