#!/usr/bin/env python3
"""
HSTU推理优化API服务

基于现有统一推理管道架构的轻量级服务化接口，
提供RESTful API访问推荐系统推理功能。
"""

import os
import sys
import logging
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import traceback

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# FastAPI相关导入
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# 导入现有的核心组件
from integrations.framework_controller import create_integrated_controller
from examples.client_example import create_realistic_user_behaviors

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hstu_api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="HSTU推理优化API服务",
    description="基于Meta HSTU模型的生成式推荐系统推理优化服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
controller = None
server_stats = {
    "start_time": None,
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_inference_time": 0.0,
}

# ============================= 数据模型定义 =============================

class UserBehavior(BaseModel):
    """用户行为数据模型"""
    video_id: int = Field(..., description="视频ID")
    timestamp: int = Field(..., description="时间戳")
    interaction_type: str = Field(..., description="交互类型: view/like/share/comment")
    duration: Optional[float] = Field(None, description="观看时长(秒)")
    device_type: Optional[str] = Field("mobile", description="设备类型")
    
    @validator('interaction_type')
    def validate_interaction_type(cls, v):
        allowed_types = ['view', 'like', 'share', 'comment', 'follow']
        if v not in allowed_types:
            raise ValueError(f'interaction_type must be one of {allowed_types}')
        return v

class InferenceRequest(BaseModel):
    """推理请求数据模型"""
    user_id: str = Field(..., description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    user_behaviors: List[UserBehavior] = Field(..., description="用户行为列表")
    num_recommendations: int = Field(10, ge=1, le=100, description="推荐数量")
    strategy: str = Field("unified", description="推理策略")
    enable_cache: bool = Field(True, description="是否启用缓存")
    
    @validator('strategy')
    def validate_strategy(cls, v):
        allowed_strategies = ['unified', 'tensorrt', 'vllm', 'hstu', 'fallback']
        if v not in allowed_strategies:
            raise ValueError(f'strategy must be one of {allowed_strategies}')
        return v

class RecommendationItem(BaseModel):
    """推荐项目数据模型"""
    video_id: int = Field(..., description="推荐视频ID")
    score: float = Field(..., description="推荐得分")
    confidence: float = Field(..., description="置信度")
    reason: Optional[str] = Field(None, description="推荐理由")

class InferenceMetrics(BaseModel):
    """推理性能指标"""
    total_time_ms: float = Field(..., description="总推理时间(毫秒)")
    feature_processing_time_ms: float = Field(..., description="特征处理时间(毫秒)")
    model_inference_time_ms: float = Field(..., description="模型推理时间(毫秒)")
    strategy_used: str = Field(..., description="实际使用的推理策略")
    cache_hit: bool = Field(..., description="是否命中缓存")
    gpu_utilization: Optional[float] = Field(None, description="GPU利用率")

class InferenceResponse(BaseModel):
    """推理响应数据模型"""
    user_id: str = Field(..., description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    recommendations: List[RecommendationItem] = Field(..., description="推荐结果列表")
    metrics: InferenceMetrics = Field(..., description="推理性能指标")
    timestamp: str = Field(..., description="响应时间戳")
    request_id: Optional[str] = Field(None, description="请求ID")

class BatchInferenceRequest(BaseModel):
    """批量推理请求数据模型"""
    requests: List[InferenceRequest] = Field(..., max_items=50, description="推理请求列表")
    batch_strategy: str = Field("auto", description="批量处理策略")

class BatchInferenceResponse(BaseModel):
    """批量推理响应数据模型"""
    results: List[InferenceResponse] = Field(..., description="批量推理结果")
    batch_metrics: Dict[str, Any] = Field(..., description="批量处理指标")

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    timestamp: str = Field(..., description="检查时间")
    version: str = Field(..., description="服务版本")
    uptime_seconds: float = Field(..., description="运行时间(秒)")
    framework_status: Dict[str, bool] = Field(..., description="框架可用性状态")

class StatsResponse(BaseModel):
    """统计信息响应"""
    server_stats: Dict[str, Any] = Field(..., description="服务器统计")
    controller_stats: Dict[str, Any] = Field(..., description="控制器统计")
    system_info: Dict[str, Any] = Field(..., description="系统信息")

# ============================= 依赖注入 =============================

async def get_controller():
    """获取控制器实例"""
    global controller
    if controller is None:
        raise HTTPException(status_code=503, detail="推理控制器未初始化")
    return controller

def update_server_stats(success: bool, inference_time: float = 0.0):
    """更新服务器统计"""
    global server_stats
    server_stats["total_requests"] += 1
    if success:
        server_stats["successful_requests"] += 1
        server_stats["total_inference_time"] += inference_time
    else:
        server_stats["failed_requests"] += 1

# ============================= 生命周期事件 =============================

@app.on_event("startup")
async def startup_event():
    """服务启动事件"""
    global controller, server_stats
    
    logger.info("🚀 启动HSTU推理优化API服务...")
    
    try:
        # 初始化推理控制器
        controller = create_integrated_controller()
        logger.info("✅ 推理控制器初始化成功")
        
        # 初始化服务器统计
        server_stats["start_time"] = datetime.now()
        
        logger.info("🎉 HSTU推理优化API服务启动完成！")
        
    except Exception as e:
        logger.error(f"❌ 服务启动失败: {e}")
        logger.error(traceback.format_exc())
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭事件"""
    logger.info("🛑 正在关闭HSTU推理优化API服务...")
    
    # 打印最终统计信息
    if server_stats["start_time"]:
        uptime = (datetime.now() - server_stats["start_time"]).total_seconds()
        logger.info(f"📊 服务运行时间: {uptime:.2f}秒")
        logger.info(f"📊 总请求数: {server_stats['total_requests']}")
        logger.info(f"📊 成功请求: {server_stats['successful_requests']}")
        logger.info(f"📊 失败请求: {server_stats['failed_requests']}")
        if server_stats["successful_requests"] > 0:
            avg_time = server_stats["total_inference_time"] / server_stats["successful_requests"]
            logger.info(f"📊 平均推理时间: {avg_time:.2f}ms")
    
    logger.info("👋 HSTU推理优化API服务已关闭")

# ============================= API端点 =============================

@app.get("/", response_model=dict)
async def root():
    """根路径"""
    return {
        "service": "HSTU推理优化API服务",
        "version": "1.0.0",
        "description": "基于Meta HSTU模型的生成式推荐系统推理优化服务",
        "docs_url": "/docs",
        "health_url": "/health",
        "stats_url": "/stats"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check(controller_instance = Depends(get_controller)):
    """健康检查"""
    try:
        # 检查框架可用性
        framework_status = {}
        if hasattr(controller_instance, 'hstu_model') and controller_instance.hstu_model:
            framework_status["hstu"] = True
        else:
            framework_status["hstu"] = False
            
        if hasattr(controller_instance, 'vllm_engine') and controller_instance.vllm_engine:
            framework_status["vllm"] = True
        else:
            framework_status["vllm"] = False
            
        if hasattr(controller_instance, 'tensorrt_engine') and controller_instance.tensorrt_engine:
            framework_status["tensorrt"] = True
        else:
            framework_status["tensorrt"] = False
        
        uptime = (datetime.now() - server_stats["start_time"]).total_seconds()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            uptime_seconds=uptime,
            framework_status=framework_status
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=503, detail="健康检查失败")

@app.get("/stats", response_model=StatsResponse)
async def get_stats(controller_instance = Depends(get_controller)):
    """获取统计信息"""
    try:
        # 计算运行时间
        uptime = (datetime.now() - server_stats["start_time"]).total_seconds()
        
        # 服务器统计
        server_statistics = {
            "uptime_seconds": uptime,
            "total_requests": server_stats["total_requests"],
            "successful_requests": server_stats["successful_requests"],
            "failed_requests": server_stats["failed_requests"],
            "success_rate": (
                server_stats["successful_requests"] / server_stats["total_requests"] 
                if server_stats["total_requests"] > 0 else 0
            ),
            "average_inference_time_ms": (
                server_stats["total_inference_time"] / server_stats["successful_requests"]
                if server_stats["successful_requests"] > 0 else 0
            )
        }
        
        # 控制器统计
        controller_statistics = {}
        if hasattr(controller_instance, 'get_comprehensive_stats'):
            controller_statistics = controller_instance.get_comprehensive_stats()
        
        # 系统信息
        system_information = {
            "python_version": sys.version,
            "platform": sys.platform,
            "cwd": os.getcwd()
        }
        
        return StatsResponse(
            server_stats=server_statistics,
            controller_stats=controller_statistics,
            system_info=system_information
        )
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail="获取统计信息失败")

@app.post("/infer", response_model=InferenceResponse)
async def infer_recommendations(
    request: InferenceRequest,
    background_tasks: BackgroundTasks,
    controller_instance = Depends(get_controller)
):
    """单次推理接口"""
    request_id = f"req_{int(time.time() * 1000)}"
    start_time = time.time()
    
    try:
        logger.info(f"[{request_id}] 开始处理推理请求 - 用户: {request.user_id}")
        
        # 转换用户行为数据格式
        user_behaviors = []
        for behavior in request.user_behaviors:
            user_behaviors.append({
                'video_id': behavior.video_id,
                'timestamp': behavior.timestamp,
                'interaction_type': behavior.interaction_type,
                'duration': behavior.duration,
                'device_type': behavior.device_type
            })
        
        # 调用现有的统一推理管道
        result = controller_instance.infer_with_optimal_strategy(
            user_id=request.user_id,
            session_id=request.session_id,
            user_behaviors=user_behaviors,
            num_recommendations=request.num_recommendations,
            requested_strategy=request.strategy
        )
        
        # 提取推理结果和指标
        recommendations = []
        if 'recommendations' in result:
            for i, rec in enumerate(result['recommendations'][:request.num_recommendations]):
                recommendations.append(RecommendationItem(
                    video_id=rec.get('video_id', i + 1),
                    score=rec.get('score', 0.5 + i * 0.1),
                    confidence=rec.get('confidence', 0.8 + i * 0.02),
                    reason=rec.get('reason', f"基于{request.strategy}策略推荐")
                ))
        
        # 构建性能指标
        total_time = (time.time() - start_time) * 1000  # 转换为毫秒
        metrics = InferenceMetrics(
            total_time_ms=total_time,
            feature_processing_time_ms=result.get('feature_processing_time_ms', 0.0),
            model_inference_time_ms=result.get('model_inference_time_ms', 0.0),
            strategy_used=result.get('strategy_used', request.strategy),
            cache_hit=result.get('cache_hit', False),
            gpu_utilization=result.get('gpu_utilization')
        )
        
        # 构建响应
        response = InferenceResponse(
            user_id=request.user_id,
            session_id=request.session_id,
            recommendations=recommendations,
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            request_id=request_id
        )
        
        # 更新统计
        update_server_stats(success=True, inference_time=total_time)
        
        logger.info(f"[{request_id}] 推理完成 - 耗时: {total_time:.2f}ms, 策略: {metrics.strategy_used}")
        
        return response
        
    except Exception as e:
        total_time = (time.time() - start_time) * 1000
        update_server_stats(success=False)
        
        logger.error(f"[{request_id}] 推理失败 - 耗时: {total_time:.2f}ms, 错误: {e}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "推理失败",
                "message": str(e),
                "request_id": request_id,
                "processing_time_ms": total_time
            }
        )

@app.post("/batch_infer", response_model=BatchInferenceResponse)
async def batch_infer_recommendations(
    request: BatchInferenceRequest,
    background_tasks: BackgroundTasks,
    controller_instance = Depends(get_controller)
):
    """批量推理接口"""
    batch_id = f"batch_{int(time.time() * 1000)}"
    start_time = time.time()
    
    try:
        logger.info(f"[{batch_id}] 开始处理批量推理请求 - 数量: {len(request.requests)}")
        
        results = []
        
        # 处理每个推理请求
        for i, inference_request in enumerate(request.requests):
            try:
                # 调用单次推理逻辑
                response = await infer_recommendations(
                    request=inference_request,
                    background_tasks=background_tasks,
                    controller_instance=controller_instance
                )
                results.append(response)
                
            except HTTPException as e:
                logger.warning(f"[{batch_id}] 批量推理中第{i+1}个请求失败: {e.detail}")
                # 批量推理中部分失败不影响整体处理
                continue
        
        # 计算批量指标
        total_time = (time.time() - start_time) * 1000
        batch_metrics = {
            "total_requests": len(request.requests),
            "successful_requests": len(results),
            "failed_requests": len(request.requests) - len(results),
            "batch_processing_time_ms": total_time,
            "average_request_time_ms": (
                sum(r.metrics.total_time_ms for r in results) / len(results)
                if results else 0
            )
        }
        
        logger.info(f"[{batch_id}] 批量推理完成 - 成功: {len(results)}/{len(request.requests)}, 耗时: {total_time:.2f}ms")
        
        return BatchInferenceResponse(
            results=results,
            batch_metrics=batch_metrics
        )
        
    except Exception as e:
        total_time = (time.time() - start_time) * 1000
        logger.error(f"[{batch_id}] 批量推理失败 - 耗时: {total_time:.2f}ms, 错误: {e}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "批量推理失败",
                "message": str(e),
                "batch_id": batch_id,
                "processing_time_ms": total_time
            }
        )

@app.post("/generate_demo_data", response_model=dict)
async def generate_demo_data(
    user_id: str,
    num_behaviors: int = Field(20, ge=1, le=100)
):
    """生成演示数据"""
    try:
        # 使用现有的数据生成函数
        demo_behaviors = create_realistic_user_behaviors(
            user_id=user_id,
            num_behaviors=num_behaviors
        )
        
        # 转换为API格式
        api_behaviors = []
        for behavior in demo_behaviors:
            api_behaviors.append(UserBehavior(
                video_id=behavior['video_id'],
                timestamp=behavior['timestamp'],
                interaction_type=behavior['interaction_type'],
                duration=behavior.get('duration'),
                device_type=behavior.get('device_type', 'mobile')
            ))
        
        return {
            "user_id": user_id,
            "behaviors": [behavior.dict() for behavior in api_behaviors],
            "count": len(api_behaviors),
            "message": "演示数据生成成功"
        }
        
    except Exception as e:
        logger.error(f"生成演示数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成演示数据失败: {str(e)}")

# ============================= 错误处理 =============================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP错误",
            "status_code": exc.status_code,
            "detail": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理器"""
    logger.error(f"未处理的异常: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "内部服务器错误",
            "message": "服务遇到了未预期的错误",
            "timestamp": datetime.now().isoformat()
        }
    )

# ============================= 主函数 =============================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='HSTU推理优化API服务')
    parser.add_argument('--host', default='0.0.0.0', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=8000, help='服务器端口')
    parser.add_argument('--workers', type=int, default=1, help='工作进程数')
    parser.add_argument('--reload', action='store_true', help='开发模式自动重载')
    parser.add_argument('--log-level', default='info', help='日志级别')
    
    args = parser.parse_args()
    
    print("🌟" * 50)
    print("HSTU推理优化API服务")
    print("🌟" * 50)
    print("基于开源框架的生成式推荐系统推理优化服务")
    print("集成技术栈:")
    print("  📚 Meta HSTU (Hierarchical Sequential Transduction Units)")
    print("  ⚡ VLLM (PagedAttention + Continuous Batching)")
    print("  🚀 TensorRT (GPU Inference Acceleration)")
    print("  🔧 Custom Triton + CUTLASS Operators")
    print("  🧠 Intelligent GPU Hot Cache")
    print("  🌐 FastAPI RESTful Service")
    print("🌟" * 50)
    print(f"🚀 启动服务: http://{args.host}:{args.port}")
    print(f"📚 API文档: http://{args.host}:{args.port}/docs")
    print("🌟" * 50)
    
    # 启动服务
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main()