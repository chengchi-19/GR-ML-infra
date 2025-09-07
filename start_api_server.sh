#!/bin/bash

# HSTU推理优化API服务启动脚本

echo "🌟================================================🌟"
echo "    HSTU推理优化API服务启动脚本"
echo "🌟================================================🌟"

# 设置默认参数
HOST=${1:-"0.0.0.0"}
PORT=${2:-8000}
WORKERS=${3:-1}
LOG_LEVEL=${4:-"info"}

echo "📋 启动参数："
echo "   - 主机地址: $HOST"
echo "   - 端口号: $PORT"
echo "   - 工作进程: $WORKERS"
echo "   - 日志级别: $LOG_LEVEL"
echo ""

# 检查Python环境
echo "🔍 检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装，请先安装Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python版本: $PYTHON_VERSION"

# 检查依赖
echo "🔍 检查FastAPI依赖..."
if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "⚠️  FastAPI依赖未安装，正在安装..."
    pip install fastapi uvicorn[standard] pydantic
else
    echo "✅ FastAPI依赖已安装"
fi

# 检查项目核心模块
echo "🔍 检查项目核心模块..."
if ! python3 -c "from integrations.framework_controller import create_integrated_controller" 2>/dev/null; then
    echo "❌ 项目核心模块导入失败，请检查项目环境"
    exit 1
fi
echo "✅ 项目核心模块检查通过"

# 创建日志目录
mkdir -p logs

echo ""
echo "🚀 启动HSTU推理优化API服务..."
echo "📚 API文档将在以下地址可用："
echo "   - Swagger UI: http://$HOST:$PORT/docs"
echo "   - ReDoc: http://$HOST:$PORT/redoc"
echo ""
echo "🔧 健康检查: http://$HOST:$PORT/health"
echo "📊 统计信息: http://$HOST:$PORT/stats" 
echo ""
echo "按 Ctrl+C 停止服务"
echo "🌟================================================🌟"

# 启动服务
exec python3 api_server.py \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL"