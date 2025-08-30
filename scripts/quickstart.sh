#!/bin/bash

# GR推理优化框架 - 快速开始脚本
# 自动化整个流程：从模型导出到性能测试

set -e

# 默认配置
WORKSPACE_DIR=$(pwd)
OUTPUT_DIR="./output"
LOG_FILE="./quickstart.log"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help              显示此帮助信息"
    echo "  -w, --workspace DIR     工作目录 (默认: 当前目录)"
    echo "  -o, --output DIR        输出目录 (默认: ./output)"
    echo "  -l, --log FILE          日志文件 (默认: ./quickstart.log)"
    echo "  --skip-export           跳过ONNX导出"
    echo "  --skip-build            跳过TensorRT构建"
    echo "  --skip-server           跳过服务器启动"
    echo "  --skip-test             跳过性能测试"
    echo "  --clean                 清理输出目录"
    echo ""
    echo "示例:"
    echo "  $0"
    echo "  $0 -o ./my_output"
    echo "  $0 --skip-export --skip-build"
}

# 检查系统要求
check_requirements() {
    log_info "检查系统要求..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3未安装"
        exit 1
    fi
    
    # 检查CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        log_warning "NVIDIA GPU不可用，某些功能可能受限"
    else
        log_success "CUDA环境检测到"
    fi
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装"
        exit 1
    fi
    
    # 检查必要的Python包
    local required_packages=("torch" "numpy" "onnx")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            log_error "Python包未安装: $package"
            exit 1
        fi
    done
    
    log_success "系统要求检查通过"
}

# 创建输出目录
create_output_dirs() {
    log_info "创建输出目录..."
    
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/models"
    mkdir -p "$OUTPUT_DIR/engines"
    mkdir -p "$OUTPUT_DIR/results"
    
    log_success "输出目录创建完成"
}

# 清理输出目录
clean_output() {
    log_info "清理输出目录..."
    
    if [ -d "$OUTPUT_DIR" ]; then
        rm -rf "$OUTPUT_DIR"
        log_success "输出目录已清理"
    fi
}

# 导出ONNX模型
export_onnx_models() {
    log_info "开始导出ONNX模型..."
    
    cd "$WORKSPACE_DIR"
    
    # 导出prefill模型
    log_info "导出prefill模型..."
    python3 src/export_onnx.py \
        --vocab_size 10000 \
        --embedding_dim 128 \
        --num_features 16 \
        --prefill "$OUTPUT_DIR/models/prefill.onnx" \
        --decode "$OUTPUT_DIR/models/decode.onnx"
    
    if [ $? -eq 0 ]; then
        log_success "ONNX模型导出完成"
    else
        log_error "ONNX模型导出失败"
        exit 1
    fi
}

# 构建TensorRT引擎
build_tensorrt_engines() {
    log_info "开始构建TensorRT引擎..."
    
    cd "$WORKSPACE_DIR"
    
    # 构建prefill引擎
    log_info "构建prefill引擎..."
    python3 src/build_engine.py \
        --onnx "$OUTPUT_DIR/models/prefill.onnx" \
        --engine "$OUTPUT_DIR/engines/prefill.engine" \
        --mode api \
        --precision fp16 \
        --validate
    
    if [ $? -eq 0 ]; then
        log_success "TensorRT引擎构建完成"
    else
        log_error "TensorRT引擎构建失败"
        exit 1
    fi
}

# 准备Triton模型仓库
prepare_triton_repo() {
    log_info "准备Triton模型仓库..."
    
    cd "$WORKSPACE_DIR"
    
    # 复制引擎到模型仓库
    if [ -f "$OUTPUT_DIR/engines/prefill.engine" ]; then
        cp "$OUTPUT_DIR/engines/prefill.engine" "triton_model_repo/gr_trt/model.engine"
        log_success "TensorRT引擎已复制到模型仓库"
    fi
    
    # 验证模型仓库
    local required_models=("ensemble_model" "gr_trt" "interaction_python" "preprocess_py" "embedding_service")
    for model in "${required_models[@]}"; do
        if [ ! -d "triton_model_repo/$model" ]; then
            log_error "缺少模型: $model"
            exit 1
        fi
    done
    
    log_success "Triton模型仓库准备完成"
}

# 启动Triton服务器
start_triton_server() {
    log_info "启动Triton服务器..."
    
    cd "$WORKSPACE_DIR"
    
    # 使用后台模式启动服务器
    bash scripts/run_server.sh --name gr-inference-server &
    local server_pid=$!
    
    # 等待服务器启动
    log_info "等待服务器启动..."
    local max_wait=60
    local wait_count=0
    
    while [ $wait_count -lt $max_wait ]; do
        if curl -s http://localhost:8000/v2/health/ready > /dev/null 2>&1; then
            log_success "Triton服务器已启动"
            echo $server_pid > "$OUTPUT_DIR/server.pid"
            return 0
        fi
        
        sleep 2
        wait_count=$((wait_count + 2))
    done
    
    log_error "Triton服务器启动超时"
    exit 1
}

# 运行性能测试
run_performance_tests() {
    log_info "运行性能测试..."
    
    cd "$WORKSPACE_DIR"
    
    # 等待服务器完全就绪
    sleep 10
    
    # 运行性能测试
    bash bench/run_triton_perf.sh \
        gr_pipeline \
        localhost:8000 \
        bench/perf_input.json \
        1:16:2 \
        4 \
        1000 \
        10 \
        3
    
    if [ $? -eq 0 ]; then
        log_success "性能测试完成"
        
        # 复制结果到输出目录
        if [ -d "bench/results" ]; then
            cp -r bench/results/* "$OUTPUT_DIR/results/"
            log_success "性能测试结果已保存到 $OUTPUT_DIR/results/"
        fi
    else
        log_error "性能测试失败"
        exit 1
    fi
}

# 生成报告
generate_report() {
    log_info "生成快速开始报告..."
    
    local report_file="$OUTPUT_DIR/quickstart_report.md"
    
    cat > "$report_file" << EOF
# GR推理优化框架 - 快速开始报告

## 执行时间
- 开始时间: $(date)
- 完成时间: $(date)

## 系统信息
\`\`\`
$(uname -a)
$(python3 --version 2>/dev/null || echo "Python版本: 未知")
$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "GPU信息: 不可用")
\`\`\`

## 生成的文件
EOF
    
    # 列出生成的文件
    echo "" >> "$report_file"
    echo "### ONNX模型" >> "$report_file"
    if [ -f "$OUTPUT_DIR/models/prefill.onnx" ]; then
        echo "- prefill.onnx" >> "$report_file"
    fi
    if [ -f "$OUTPUT_DIR/models/decode.onnx" ]; then
        echo "- decode.onnx" >> "$report_file"
    fi
    
    echo "" >> "$report_file"
    echo "### TensorRT引擎" >> "$report_file"
    if [ -f "$OUTPUT_DIR/engines/prefill.engine" ]; then
        echo "- prefill.engine" >> "$report_file"
    fi
    
    echo "" >> "$report_file"
    echo "### 性能测试结果" >> "$report_file"
    if [ -d "$OUTPUT_DIR/results" ]; then
        for result_file in "$OUTPUT_DIR/results"/*; do
            if [ -f "$result_file" ]; then
                echo "- $(basename "$result_file")" >> "$report_file"
            fi
        done
    fi
    
    echo "" >> "$report_file"
    echo "## 服务端点" >> "$report_file"
    echo "- HTTP API: http://localhost:8000" >> "$report_file"
    echo "- gRPC API: localhost:8001" >> "$report_file"
    echo "- Metrics API: http://localhost:8002" >> "$report_file"
    echo "- Health Check: http://localhost:8000/v2/health/ready" >> "$report_file"
    
    echo "" >> "$report_file"
    echo "## 下一步" >> "$report_file"
    echo "1. 访问 http://localhost:8000/v2/models 查看可用模型" >> "$report_file"
    echo "2. 使用性能测试脚本进行详细基准测试" >> "$report_file"
    echo "3. 查看 $OUTPUT_DIR/results/ 中的性能报告" >> "$report_file"
    
    log_success "快速开始报告已生成: $report_file"
}

# 清理资源
cleanup() {
    log_info "清理资源..."
    
    # 停止服务器
    if [ -f "$OUTPUT_DIR/server.pid" ]; then
        local server_pid=$(cat "$OUTPUT_DIR/server.pid")
        if kill -0 "$server_pid" 2>/dev/null; then
            kill "$server_pid"
            log_info "Triton服务器已停止"
        fi
        rm -f "$OUTPUT_DIR/server.pid"
    fi
    
    # 清理Docker容器
    if docker ps -q -f name=gr-inference-server | grep -q .; then
        docker stop gr-inference-server 2>/dev/null || true
        docker rm gr-inference-server 2>/dev/null || true
    fi
}

# 解析命令行参数
parse_args() {
    SKIP_EXPORT=false
    SKIP_BUILD=false
    SKIP_SERVER=false
    SKIP_TEST=false
    CLEAN_OUTPUT=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -w|--workspace)
                WORKSPACE_DIR="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -l|--log)
                LOG_FILE="$2"
                shift 2
                ;;
            --skip-export)
                SKIP_EXPORT=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-server)
                SKIP_SERVER=true
                shift
                ;;
            --skip-test)
                SKIP_TEST=true
                shift
                ;;
            --clean)
                CLEAN_OUTPUT=true
                shift
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
    log_info "GR推理优化框架 - 快速开始"
    echo "=================================="
    
    # 解析参数
    parse_args "$@"
    
    # 设置信号处理
    trap cleanup EXIT
    
    # 清理输出目录（如果请求）
    if [ "$CLEAN_OUTPUT" = true ]; then
        clean_output
    fi
    
    # 检查系统要求
    check_requirements
    
    # 创建输出目录
    create_output_dirs
    
    # 导出ONNX模型
    if [ "$SKIP_EXPORT" = false ]; then
        export_onnx_models
    else
        log_info "跳过ONNX导出"
    fi
    
    # 构建TensorRT引擎
    if [ "$SKIP_BUILD" = false ]; then
        build_tensorrt_engines
    else
        log_info "跳过TensorRT构建"
    fi
    
    # 准备Triton模型仓库
    prepare_triton_repo
    
    # 启动Triton服务器
    if [ "$SKIP_SERVER" = false ]; then
        start_triton_server
    else
        log_info "跳过服务器启动"
    fi
    
    # 运行性能测试
    if [ "$SKIP_TEST" = false ]; then
        run_performance_tests
    else
        log_info "跳过性能测试"
    fi
    
    # 生成报告
    generate_report
    
    log_success "快速开始流程完成！"
    log_info "输出目录: $OUTPUT_DIR"
    log_info "日志文件: $LOG_FILE"
    
    if [ "$SKIP_SERVER" = false ]; then
        echo ""
        log_info "Triton服务器正在运行:"
        echo "  HTTP API: http://localhost:8000"
        echo "  gRPC API: localhost:8001"
        echo "  Metrics API: http://localhost:8002"
        echo ""
        echo "按 Ctrl+C 停止服务器"
        
        # 等待用户中断
        wait
    fi
}

# 运行主函数
main "$@"
