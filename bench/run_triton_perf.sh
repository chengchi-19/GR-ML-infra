#!/bin/bash

# Triton性能基准测试脚本
# 用于测试GR推理优化框架的性能

set -e

# 默认参数
MODEL=${1:-gr_pipeline}
URL=${2:-localhost:8000}
INPUT=${3:-bench/perf_input.json}
CONCURRENCY_RANGE=${4:-1:16:2}
BATCH_SIZE=${5:-4}
MEASUREMENT_INTERVAL=${6:-1000}
STABILITY_PERCENTAGE=${7:-10}
MAX_TRIALS=${8:-3}

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

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."
    
    if ! command -v perf_analyzer &> /dev/null; then
        log_error "perf_analyzer 未找到，请安装 Triton Client"
        exit 1
    fi
    
    if ! command -v curl &> /dev/null; then
        log_error "curl 未找到"
        exit 1
    fi
    
    log_success "依赖检查通过"
}

# 检查Triton服务器状态
check_triton_server() {
    log_info "检查Triton服务器状态..."
    
    local max_retries=30
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        if curl -s http://$URL/v2/health/ready > /dev/null 2>&1; then
            log_success "Triton服务器已就绪"
            return 0
        fi
        
        log_warning "等待Triton服务器启动... (${retry_count}/${max_retries})"
        sleep 2
        retry_count=$((retry_count + 1))
    done
    
    log_error "Triton服务器启动超时"
    return 1
}

# 检查模型状态
check_model_status() {
    log_info "检查模型状态: $MODEL"
    
    local model_status=$(curl -s http://$URL/v2/models/$MODEL/status 2>/dev/null || echo "{}")
    
    if echo "$model_status" | grep -q "ready"; then
        log_success "模型 $MODEL 已就绪"
        return 0
    else
        log_error "模型 $MODEL 未就绪"
        return 1
    fi
}

# 运行性能测试
run_performance_test() {
    log_info "开始性能测试..."
    log_info "模型: $MODEL"
    log_info "URL: $URL"
    log_info "输入文件: $INPUT"
    log_info "并发范围: $CONCURRENCY_RANGE"
    log_info "批次大小: $BATCH_SIZE"
    log_info "测量间隔: ${MEASUREMENT_INTERVAL}ms"
    
    # 创建输出目录
    local output_dir="bench/results/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$output_dir"
    
    # 运行perf_analyzer
    local perf_output="$output_dir/perf_results.csv"
    local perf_summary="$output_dir/perf_summary.txt"
    
    log_info "运行perf_analyzer..."
    
    perf_analyzer \
        -m "$MODEL" \
        -u "$URL" \
        --input-data="$INPUT" \
        --concurrency-range="$CONCURRENCY_RANGE" \
        --batch-size="$BATCH_SIZE" \
        --measurement-interval-ms="$MEASUREMENT_INTERVAL" \
        --stability-percentage="$STABILITY_PERCENTAGE" \
        --max-trials="$MAX_TRIALS" \
        --percentile=95 \
        --output-shared-memory-size=268435456 \
        --csv="$perf_output" \
        --verbose \
        2>&1 | tee "$perf_summary"
    
    if [ $? -eq 0 ]; then
        log_success "性能测试完成"
        log_info "结果保存在: $output_dir"
        
        # 显示关键指标
        echo ""
        log_info "关键性能指标:"
        echo "=================================="
        
        if [ -f "$perf_output" ]; then
            # 提取最佳吞吐量和延迟
            local best_throughput=$(tail -n +2 "$perf_output" | cut -d',' -f2 | sort -nr | head -1)
            local best_latency=$(tail -n +2 "$perf_output" | cut -d',' -f3 | sort -n | head -1)
            
            echo "最佳吞吐量: ${best_throughput} infer/sec"
            echo "最佳延迟: ${best_latency} ms"
        fi
        
        # 显示GPU使用情况
        if command -v nvidia-smi &> /dev/null; then
            echo ""
            log_info "GPU使用情况:"
            echo "=================================="
            nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
        fi
        
    else
        log_error "性能测试失败"
        return 1
    fi
}

# 运行交互内核自动调优
run_interaction_autotune() {
    log_info "运行交互内核自动调优..."
    
    local autotune_output="bench/results/interaction_autotune_$(date +%Y%m%d_%H%M%S).json"
    
    python3 kernels/triton_ops/autotune_interaction.py \
        --B 8 \
        --F 16 \
        --D 64 \
        --blocks 16,32,64,128,256 \
        --iters 100 \
        --out "$autotune_output"
    
    if [ $? -eq 0 ]; then
        log_success "交互内核自动调优完成"
        log_info "结果保存在: $autotune_output"
        
        # 显示最佳配置
        if [ -f "$autotune_output" ]; then
            echo ""
            log_info "最佳交互内核配置:"
            echo "=================================="
            python3 -c "
import json
with open('$autotune_output', 'r') as f:
    results = json.load(f)
best_block = min(results.items(), key=lambda x: x[1] if x[1] is not None else float('inf'))
print(f'最佳块大小: {best_block[0]}')
print(f'延迟: {best_block[1]:.4f} ms')
"
        fi
    else
        log_error "交互内核自动调优失败"
        return 1
    fi
}

# 运行内存基准测试
run_memory_benchmark() {
    log_info "运行内存基准测试..."
    
    local memory_output="bench/results/memory_benchmark_$(date +%Y%m%d_%H%M%S).txt"
    
    # 测试嵌入服务内存使用
    python3 -c "
import torch
import sys
sys.path.append('src')
from embedding_service import EmbeddingService

print('=== 嵌入服务内存基准测试 ===')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'当前GPU: {torch.cuda.current_device()}')
    print(f'GPU名称: {torch.cuda.get_device_name()}')

# 测试不同配置的内存使用
configs = [
    (1000, 32, 256, 1000),
    (5000, 64, 512, 2000),
    (10000, 128, 1024, 5000),
]

for num_items, emb_dim, gpu_cache, host_cache in configs:
    print(f'\\n配置: items={num_items}, dim={emb_dim}, gpu_cache={gpu_cache}, host_cache={host_cache}')
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        service = EmbeddingService(
            num_items=num_items,
            emb_dim=emb_dim,
            gpu_cache_size=gpu_cache,
            host_cache_size=host_cache
        )
        
        # 预热缓存
        for i in range(min(100, num_items)):
            service.lookup_single(i)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        current_memory = torch.cuda.memory_allocated() / 1024**2
        
        print(f'  峰值内存: {peak_memory:.2f} MB')
        print(f'  当前内存: {current_memory:.2f} MB')
        
        service.clear_cache()
        del service
        
    except Exception as e:
        print(f'  错误: {e}')
    
    torch.cuda.empty_cache()
" > "$memory_output" 2>&1
    
    if [ $? -eq 0 ]; then
        log_success "内存基准测试完成"
        log_info "结果保存在: $memory_output"
        
        # 显示关键结果
        echo ""
        log_info "内存使用摘要:"
        echo "=================================="
        tail -20 "$memory_output"
    else
        log_error "内存基准测试失败"
        return 1
    fi
}

# 生成性能报告
generate_report() {
    log_info "生成性能报告..."
    
    local report_file="bench/results/performance_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$report_file" << EOF
# GR推理优化框架性能报告

## 测试环境
- 测试时间: $(date)
- 模型: $MODEL
- URL: $URL
- 并发范围: $CONCURRENCY_RANGE
- 批次大小: $BATCH_SIZE

## 系统信息
\`\`\`
$(uname -a)
$(python3 --version 2>/dev/null || echo "Python版本: 未知")
$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "GPU信息: 不可用")
\`\`\`

## 性能指标
EOF
    
    # 添加性能数据
    if [ -f "bench/results/perf_results.csv" ]; then
        echo "" >> "$report_file"
        echo "### 吞吐量和延迟" >> "$report_file"
        echo "\`\`\`csv" >> "$report_file"
        cat bench/results/perf_results.csv >> "$report_file"
        echo "\`\`\`" >> "$report_file"
    fi
    
    log_success "性能报告生成完成: $report_file"
}

# 主函数
main() {
    log_info "开始GR推理优化框架性能测试"
    echo "=================================="
    
    # 检查依赖
    check_dependencies
    
    # 检查Triton服务器
    check_triton_server
    
    # 检查模型状态
    check_model_status
    
    # 运行各种基准测试
    run_performance_test
    run_interaction_autotune
    run_memory_benchmark
    
    # 生成报告
    generate_report
    
    log_success "所有性能测试完成！"
    log_info "结果保存在: bench/results/"
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [模型名] [URL] [输入文件] [并发范围] [批次大小] [测量间隔] [稳定性百分比] [最大试验次数]"
    echo ""
    echo "参数:"
    echo "  模型名               Triton模型名称 (默认: gr_pipeline)"
    echo "  URL                  Triton服务器URL (默认: localhost:8000)"
    echo "  输入文件             性能测试输入文件 (默认: bench/perf_input.json)"
    echo "  并发范围             并发请求范围 (默认: 1:16:2)"
    echo "  批次大小             批次大小 (默认: 4)"
    echo "  测量间隔             测量间隔(ms) (默认: 1000)"
    echo "  稳定性百分比         稳定性百分比 (默认: 10)"
    echo "  最大试验次数         最大试验次数 (默认: 3)"
    echo ""
    echo "示例:"
    echo "  $0 gr_pipeline localhost:8000"
    echo "  $0 gr_pipeline localhost:8000 bench/perf_input.json 1:32:4 8"
}

# 检查参数
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

# 运行主函数
main "$@"
