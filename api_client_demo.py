# API客户端使用示例

from api_server import UserBehavior, InferenceRequest
import requests
import json
from datetime import datetime
import asyncio
import aiohttp

class HSTUAPIClient:
    """HSTU推理优化API客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self):
        """健康检查"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_stats(self):
        """获取统计信息"""
        try:
            response = self.session.get(f"{self.base_url}/stats")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def generate_demo_data(self, user_id: str, num_behaviors: int = 20):
        """生成演示数据"""
        try:
            response = self.session.post(
                f"{self.base_url}/generate_demo_data",
                params={"user_id": user_id, "num_behaviors": num_behaviors}
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def infer(self, request: dict):
        """单次推理"""
        try:
            response = self.session.post(
                f"{self.base_url}/infer",
                json=request,
                headers={"Content-Type": "application/json"}
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def batch_infer(self, requests: list):
        """批量推理"""
        try:
            batch_request = {"requests": requests}
            response = self.session.post(
                f"{self.base_url}/batch_infer",
                json=batch_request,
                headers={"Content-Type": "application/json"}
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def create_sample_request():
    """创建示例请求"""
    # 模拟用户行为数据
    behaviors = [
        {
            "video_id": 12345,
            "timestamp": 1700000000,
            "interaction_type": "view",
            "duration": 120.5,
            "device_type": "mobile"
        },
        {
            "video_id": 12346,
            "timestamp": 1700001000,
            "interaction_type": "like",
            "duration": None,
            "device_type": "mobile"
        },
        {
            "video_id": 12347,
            "timestamp": 1700002000,
            "interaction_type": "share",
            "duration": None,
            "device_type": "desktop"
        }
    ]
    
    return {
        "user_id": "user_demo_001",
        "session_id": "session_" + str(int(datetime.now().timestamp())),
        "user_behaviors": behaviors,
        "num_recommendations": 10,
        "strategy": "unified",
        "enable_cache": True
    }

def demo_single_inference():
    """演示单次推理"""
    print("🔍 演示单次推理...")
    
    client = HSTUAPIClient()
    
    # 健康检查
    print("1. 健康检查:")
    health = client.health_check()
    print(f"   状态: {health.get('status', 'unknown')}")
    
    # 创建推理请求
    request = create_sample_request()
    print(f"\n2. 推理请求:")
    print(f"   用户ID: {request['user_id']}")
    print(f"   行为数量: {len(request['user_behaviors'])}")
    print(f"   推荐策略: {request['strategy']}")
    
    # 执行推理
    print("\n3. 执行推理...")
    result = client.infer(request)
    
    if 'error' in result:
        print(f"❌ 推理失败: {result['error']}")
        return
    
    print("✅ 推理成功!")
    print(f"   推荐数量: {len(result.get('recommendations', []))}")
    print(f"   总耗时: {result.get('metrics', {}).get('total_time_ms', 0):.2f}ms")
    print(f"   使用策略: {result.get('metrics', {}).get('strategy_used', 'unknown')}")
    
    # 显示前3个推荐
    recommendations = result.get('recommendations', [])[:3]
    print(f"\n4. 推荐结果预览:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. 视频ID: {rec.get('video_id')}, 得分: {rec.get('score', 0):.3f}")

def demo_batch_inference():
    """演示批量推理"""
    print("\n🔍 演示批量推理...")
    
    client = HSTUAPIClient()
    
    # 创建多个推理请求
    requests = []
    for i in range(3):
        request = create_sample_request()
        request['user_id'] = f"user_batch_{i+1:03d}"
        requests.append(request)
    
    print(f"1. 批量请求数量: {len(requests)}")
    
    # 执行批量推理
    print("2. 执行批量推理...")
    result = client.batch_infer(requests)
    
    if 'error' in result:
        print(f"❌ 批量推理失败: {result['error']}")
        return
    
    print("✅ 批量推理成功!")
    batch_metrics = result.get('batch_metrics', {})
    print(f"   处理请求: {batch_metrics.get('successful_requests', 0)}/{batch_metrics.get('total_requests', 0)}")
    print(f"   总耗时: {batch_metrics.get('batch_processing_time_ms', 0):.2f}ms")
    print(f"   平均耗时: {batch_metrics.get('average_request_time_ms', 0):.2f}ms")

def demo_stats_monitoring():
    """演示统计监控"""
    print("\n🔍 演示统计监控...")
    
    client = HSTUAPIClient()
    
    # 获取统计信息
    stats = client.get_stats()
    
    if 'error' in stats:
        print(f"❌ 获取统计失败: {stats['error']}")
        return
    
    server_stats = stats.get('server_stats', {})
    print("📊 服务器统计:")
    print(f"   运行时间: {server_stats.get('uptime_seconds', 0):.1f}秒")
    print(f"   总请求数: {server_stats.get('total_requests', 0)}")
    print(f"   成功率: {server_stats.get('success_rate', 0)*100:.1f}%")
    print(f"   平均延迟: {server_stats.get('average_inference_time_ms', 0):.2f}ms")

def demo_auto_data_generation():
    """演示自动数据生成"""
    print("\n🔍 演示自动数据生成...")
    
    client = HSTUAPIClient()
    
    # 生成演示数据
    demo_data = client.generate_demo_data("user_auto_demo", num_behaviors=15)
    
    if 'error' in demo_data:
        print(f"❌ 数据生成失败: {demo_data['error']}")
        return
    
    print("✅ 演示数据生成成功!")
    print(f"   用户ID: {demo_data.get('user_id')}")
    print(f"   行为数量: {demo_data.get('count')}")
    
    # 使用生成的数据进行推理
    behaviors = demo_data.get('behaviors', [])
    if behaviors:
        inference_request = {
            "user_id": demo_data.get('user_id'),
            "user_behaviors": behaviors,
            "num_recommendations": 10,
            "strategy": "unified"
        }
        
        print("🔄 使用生成的数据进行推理...")
        result = client.infer(inference_request)
        
        if 'error' not in result:
            print("✅ 推理成功!")
            print(f"   推荐数量: {len(result.get('recommendations', []))}")
            print(f"   推理耗时: {result.get('metrics', {}).get('total_time_ms', 0):.2f}ms")

def main():
    """主函数"""
    print("🌟" * 50)
    print("HSTU推理优化API - 客户端使用演示")
    print("🌟" * 50)
    
    try:
        # 1. 单次推理演示
        demo_single_inference()
        
        # 2. 批量推理演示
        demo_batch_inference()
        
        # 3. 统计监控演示
        demo_stats_monitoring()
        
        # 4. 自动数据生成演示
        demo_auto_data_generation()
        
        print("\n🎉 所有演示完成！")
        print("\n📚 更多信息:")
        print("   - API文档: http://localhost:8000/docs")
        print("   - 健康检查: http://localhost:8000/health")
        print("   - 统计信息: http://localhost:8000/stats")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出错: {e}")
        print("💡 请确保API服务已启动: python api_server.py")

if __name__ == "__main__":
    main()