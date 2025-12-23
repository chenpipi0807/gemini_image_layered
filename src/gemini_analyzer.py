"""
Gemini 图像分析模块
负责分析图片，识别元素并输出切图位置
"""
import json
import base64
import requests
from pathlib import Path
from typing import List, Dict, Any
from .config import LLM_API_KEY, LLM_BASE_URL, REASONING_MODEL


ANALYSIS_PROMPT = """你是一个专业的图像分层分析师。请分析这张图片，识别出所有可以独立分层的视觉元素。

**重要：不要输出背景层，只输出前景元素（人物、物体、文字、装饰等）**

对于每个前景元素，请提供：
1. name: 元素名称（简短，如 "滑板女孩"、"粉色Girls文字"）
2. type: 元素类型
   - person: 人物
   - object: 物体（产品、道具等）
   - text: 文字
   - decoration: 装饰元素
3. bbox: 边界框坐标 [x, y, width, height]，使用像素值，必须精确包围元素
4. priority: 图层优先级（数字越大越靠前，从1开始）
5. needs_cutout: 是否需要抠图
   - true: 人物(person)、物体(object)、装饰(decoration) 需要抠图
   - false: 文字(text) 不需要抠图

请以JSON格式输出：
{
    "image_size": {"width": 图片宽度, "height": 图片高度},
    "layers": [
        {
            "name": "元素名称",
            "type": "person/object/text/decoration",
            "bbox": [x, y, width, height],
            "priority": 1,
            "needs_cutout": true/false,
            "description": "简短描述"
        }
    ]
}

**关键要求：**
1. bbox 必须精确！x是左边距，y是上边距，width是宽度，height是高度
2. bbox 要紧密包围元素，不要太大也不要太小
3. 人物和物体设置 needs_cutout: true
4. 文字设置 needs_cutout: false（文字层保留原始背景）
5. 不要遗漏任何明显的前景元素
6. 只输出JSON，不要其他内容
"""


class GeminiAnalyzer:
    def __init__(self):
        self.api_key = LLM_API_KEY
        self.base_url = LLM_BASE_URL
        self.model = REASONING_MODEL
    
    def _encode_image(self, image_path: str) -> str:
        """将图片编码为base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _get_mime_type(self, image_path: str) -> str:
        """获取图片MIME类型"""
        suffix = Path(image_path).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_map.get(suffix, "image/jpeg")
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        分析图片，返回分层信息
        
        Args:
            image_path: 图片路径
            
        Returns:
            包含图片尺寸和图层信息的字典
        """
        from PIL import Image as PILImage
        
        # 获取图片实际尺寸
        with PILImage.open(image_path) as img:
            img_width, img_height = img.size
        
        image_data = self._encode_image(image_path)
        mime_type = self._get_mime_type(image_path)
        
        # 构建请求 - 适配 openproxy.zuoyebang.cc 代理格式
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        
        # 使用完整的 API Key
        api_key = self.api_key
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
            "Accept": "*/*"
        }
        
        # 在prompt中加入图片尺寸信息
        prompt_with_size = ANALYSIS_PROMPT + f"\n\n图片实际尺寸: 宽度={img_width}px, 高度={img_height}px"
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt_with_size},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_data
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 8192,
                "thinkingConfig": {
                    "thinkingBudget": 1024
                }
            }
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        
        # 调试：打印错误详情
        if response.status_code != 200:
            print(f"[API错误] 状态码: {response.status_code}")
            print(f"[API错误] URL: {url}")
            print(f"[API错误] 响应: {response.text[:500]}")
        
        response.raise_for_status()
        
        result = response.json()
        
        # 调试：打印响应结构
        # print(f"[DEBUG] 响应: {json.dumps(result, ensure_ascii=False)[:1000]}")
        
        # 解析响应 - 处理不同的响应格式
        try:
            candidates = result.get("candidates", [])
            if not candidates:
                raise Exception("没有返回候选结果")
            
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            
            if not parts:
                raise Exception(f"没有返回内容部分: {content}")
            
            text_content = parts[0].get("text", "")
            if not text_content:
                raise Exception(f"没有返回文本: {parts[0]}")
        except Exception as e:
            print(f"[解析错误] {e}")
            print(f"[响应内容] {json.dumps(result, ensure_ascii=False)[:500]}")
            raise
        
        # 提取JSON
        text_content = text_content.strip()
        if text_content.startswith("```json"):
            text_content = text_content[7:]
        if text_content.startswith("```"):
            text_content = text_content[3:]
        if text_content.endswith("```"):
            text_content = text_content[:-3]
        
        layer_info = json.loads(text_content.strip())
        
        # 确保返回正确的图片尺寸
        layer_info["image_size"] = {"width": img_width, "height": img_height}
        
        return layer_info


def analyze(image_path: str) -> Dict[str, Any]:
    """便捷函数：分析图片"""
    analyzer = GeminiAnalyzer()
    return analyzer.analyze_image(image_path)
