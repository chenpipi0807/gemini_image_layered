"""
生成式图层分解器 V7
核心思路：模仿 Qwen-Image-Layered 的生成式方法

原理：
1. Gemini VL 理解图像语义，分析图层结构
2. Gemini 图像生成模型**逐层重新绘制**每个元素
3. 不是"切图"，而是"生成"完整的 RGBA 图层

关键创新：
- 被遮挡的部分也能生成（因为是重新画的）
- 每个图层都是完整的、干净的
- 透明背景自然生成
"""
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List, Tuple
import json
import base64
import requests
from pathlib import Path
from io import BytesIO

from .config import LLM_API_KEY, LLM_BASE_URL, REASONING_MODEL, IMAGE_MODEL
from .background_remover import BackgroundRemover


def remove_solid_background(image: Image.Image, tolerance: int = 25) -> Image.Image:
    """
    移除纯色背景（中性灰、绿幕或白色），转换为真正的透明通道
    
    策略：
    1. 检测中性灰 (#808080)
    2. 检测绿幕 (#00FF00)
    3. 检测白色背景
    4. 使用连通区域分析，只移除与边缘相连的背景
    
    Args:
        image: 带纯色背景的图像
        tolerance: 颜色容差
    
    Returns:
        带透明背景的 RGBA 图像
    """
    img_array = np.array(image.convert("RGBA"))
    
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    
    # 检测中性灰：RGB 都接近 128，但要求非常严格避免误伤
    gray_mask = (
        (np.abs(r.astype(int) - 128) < 10) &
        (np.abs(g.astype(int) - 128) < 10) &
        (np.abs(b.astype(int) - 128) < 10) &
        (np.abs(r.astype(int) - g.astype(int)) < 5) &
        (np.abs(g.astype(int) - b.astype(int)) < 5)
    )
    
    # 检测绿幕：纯绿色 #00FF00
    green_mask = (
        (g > 200) &
        (r < 60) &
        (b < 60) &
        (g > r + 100) &
        (g > b + 100)
    )
    
    # 检测白色/浅色背景：RGB 都很高且接近
    white_mask = (
        (r > 235) &
        (g > 235) &
        (b > 235) &
        (np.abs(r.astype(int) - g.astype(int)) < 20) &
        (np.abs(g.astype(int) - b.astype(int)) < 20)
    )
    
    # 合并所有背景掩码（不使用gray_mask，因为Gemini实际生成白色背景，灰色检测会误伤阴影）
    background_mask = green_mask | white_mask
    
    # 使用连通区域分析，只移除与边缘相连的背景区域
    from scipy import ndimage
    
    # 创建边缘种子
    h, w = background_mask.shape
    edge_mask = np.zeros_like(background_mask)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    
    # 只保留与边缘连通的背景
    labeled, num_features = ndimage.label(background_mask)
    edge_labels = set(labeled[edge_mask & background_mask])
    
    connected_bg_mask = np.isin(labeled, list(edge_labels))
    
    # 将背景区域设为透明
    img_array[:, :, 3] = np.where(connected_bg_mask, 0, 255)
    
    return Image.fromarray(img_array, mode="RGBA")


# 语义分析 Prompt
SEMANTIC_ANALYSIS_PROMPT = """你是一个专业的图像分层分析师。请详细分析这张图片的语义结构。

**任务**：识别所有可独立分层的元素，为后续的图层生成做准备。

对于每个元素，请提供：
1. **name**: 元素名称（简短中文，如"滑板女孩"、"粉色Girls文字"）
2. **type**: 类型 (person/object/text/decoration/background)
3. **depth**: 深度层级（背景=0，越靠前数字越大）
4. **description**: 详细描述（用于指导图像生成）
   - 包括：颜色、形状、风格、位置、细节特征
   - 如果是文字：包括文字内容、字体风格、颜色
   - 如果是人物：包括姿势、服装、表情、配饰
5. **style**: 图像风格（如"像素风格"、"卡通风格"）
6. **position**: 在画面中的位置描述（如"画面中央"、"左下角"）
7. **occluded_parts**: 被遮挡的部分描述（如果有）

**关键要求**：
- description 要足够详细，能让图像生成模型准确重现该元素
- 注意描述被遮挡部分应该是什么样子
- 保持风格一致性描述

输出 JSON：
{
    "image_description": "整体图像描述",
    "style": "整体风格",
    "layers": [
        {
            "name": "元素名",
            "type": "类型",
            "depth": 0,
            "description": "详细描述",
            "style": "风格",
            "position": "位置",
            "occluded_parts": "被遮挡部分描述或null"
        }
    ]
}

只输出 JSON。
"""


class GenerativeDecomposer:
    """
    生成式图层分解器
    
    核心思路：用 Gemini 图像生成模型逐层重新绘制每个元素
    """
    
    def __init__(self, bg_removal_method: str = "color"):
        """
        Args:
            bg_removal_method: 背景移除方法 ("color", "rembg", "inspyrenet")
        """
        self.api_key = LLM_API_KEY
        self.base_url = LLM_BASE_URL
        self.reasoning_model = REASONING_MODEL
        self.image_model = IMAGE_MODEL
        
        self.image_path: Optional[str] = None
        self.original_image: Optional[Image.Image] = None
        self.canvas_size: Optional[Tuple[int, int]] = None
        self.semantic_analysis: Optional[Dict] = None
        
        # 初始化背景移除器
        self.bg_remover = BackgroundRemover(method=bg_removal_method)
        print(f"      背景移除方法: {bg_removal_method}")
    
    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _encode_pil_image(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def _get_mime_type(self, image_path: str) -> str:
        ext = Path(image_path).suffix.lower()
        return {"jpg": "image/jpeg", "jpeg": "image/jpeg", 
                "png": "image/png"}.get(ext, "image/png")
    
    def analyze_semantics(self, image_path: str) -> Dict[str, Any]:
        """
        使用 Gemini VL 分析图像语义
        """
        image_data = self._encode_image(image_path)
        mime_type = self._get_mime_type(image_path)
        
        prompt = f"图片尺寸: {self.canvas_size[0]}x{self.canvas_size[1]} 像素\n\n{SEMANTIC_ANALYSIS_PROMPT}"
        
        url = f"{self.base_url}/v1beta/models/{self.reasoning_model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        
        payload = {
            "contents": [{"parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": image_data}}
            ]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 8192,
                "thinkingConfig": {"thinkingBudget": 2048}
            }
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        
        result = response.json()
        text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
        # 提取 JSON
        text = text.strip()
        if text.startswith("```json"): text = text[7:]
        if text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
        
        self.semantic_analysis = json.loads(text.strip())
        return self.semantic_analysis
    
    def generate_layer(self, layer_info: Dict, reference_image: Image.Image) -> Optional[Image.Image]:
        """
        使用 Gemini 图像生成模型生成单个图层
        
        关键：让模型重新绘制该元素，而不是从原图切割
        """
        name = layer_info.get("name", "元素")
        description = layer_info.get("description", "")
        style = layer_info.get("style", "")
        position = layer_info.get("position", "")
        occluded = layer_info.get("occluded_parts", "")
        layer_type = layer_info.get("type", "object")
        
        # 构建生成 prompt
        if layer_type == "background":
            prompt = f"""请根据参考图片，生成一个干净的背景图层。

要求：
1. 只保留背景，移除所有前景元素（人物、文字、装饰等）
2. 被前景遮挡的背景区域需要合理补全（inpainting）
3. 保持原图的风格和色调
4. 输出尺寸：{self.canvas_size[0]}x{self.canvas_size[1]}

背景描述：{description}
风格：{style}"""
        else:
            prompt = f"""请根据参考图片，**只**绘制"{name}"这个元素，背景使用纯白色。

**严格要求**：
1. **只绘制"{name}"这一个元素**，不要绘制任何其他元素
2. **背景必须是纯白色 #FFFFFF**（RGB: 255,255,255）
3. 不要绘制图片中的其他任何内容（人物、其他文字、装饰等）
4. 保持与原图完全一致的风格、颜色、细节
5. 如果该元素有被遮挡的部分，需要补全完整
6. 元素位置和大小与原图一致
7. 输出尺寸：{self.canvas_size[0]}x{self.canvas_size[1]}

元素描述：{description}
风格：{style}
位置：{position}
{"被遮挡部分需要补全：" + occluded if occluded else ""}

**再次强调：只绘制"{name}"，背景必须是纯白色#FFFFFF！**"""

        # 调用 Gemini 图像生成
        image_data = self._encode_pil_image(reference_image)
        
        url = f"{self.base_url}/v1beta/models/{self.image_model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        
        payload = {
            "contents": [{"parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": image_data}}
            ]}],
            "generationConfig": {
                "temperature": 0.3,
                "responseModalities": ["TEXT", "IMAGE"]
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            
            result = response.json()
            candidates = result.get("candidates", [])
            
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    if "inlineData" in part:
                        img_data = part["inlineData"]["data"]
                        img_bytes = base64.b64decode(img_data)
                        generated_image = Image.open(BytesIO(img_bytes)).convert("RGBA")
                        
                        # 调整尺寸
                        if generated_image.size != self.canvas_size:
                            generated_image = generated_image.resize(self.canvas_size, Image.LANCZOS)
                        
                        # 抠图：移除背景，转为真正的透明通道
                        if layer_type != "background":
                            generated_image = self.bg_remover.remove_background(generated_image)
                        
                        return generated_image
            
            return None
            
        except Exception as e:
            print(f"      [生成失败] {e}")
            return None
    
    def decompose(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """
        执行完整的生成式图层分解
        """
        print(f"[1/4] 加载图片: {image_path}")
        self.image_path = image_path
        self.original_image = Image.open(image_path).convert("RGBA")
        self.canvas_size = (self.original_image.width, self.original_image.height)
        print(f"      尺寸: {self.canvas_size[0]}x{self.canvas_size[1]}")
        
        # Step 1: 语义分析
        print("[2/4] Gemini 语义分析...")
        self.analyze_semantics(image_path)
        
        layers = self.semantic_analysis.get("layers", [])
        style = self.semantic_analysis.get("style", "")
        print(f"      识别到 {len(layers)} 个图层")
        print(f"      整体风格: {style}")
        
        for layer in layers:
            print(f"      - {layer['name']} ({layer['type']})")
        
        # Step 2: 并行生成所有图层
        print("[3/4] Gemini 并行生成...")
        
        # 按深度排序（从后到前）
        layers.sort(key=lambda x: x.get("depth", 0))
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        output_layers = []
        
        def generate_single_layer(layer_info):
            """单个图层生成任务"""
            name = layer_info.get("name", "未命名")
            layer_type = layer_info.get("type", "object")
            
            generated = self.generate_layer(layer_info, self.original_image)
            
            if generated:
                return {
                    "name": name,
                    "type": layer_type,
                    "image": generated,
                    "depth": layer_info.get("depth", 0),
                    "description": layer_info.get("description", "")
                }
            return None
        
        # 并行生成，最多同时 5 个请求
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_layer = {executor.submit(generate_single_layer, layer): layer for layer in layers}
            
            for future in as_completed(future_to_layer):
                layer_info = future_to_layer[future]
                name = layer_info.get("name", "未命名")
                
                try:
                    result = future.result()
                    if result:
                        output_layers.append(result)
                        print(f"      ✓ {name}")
                    else:
                        print(f"      ✗ {name}")
                except Exception as e:
                    print(f"      ✗ {name} - {e}")
        
        # Step 3: 导出
        print("[4/4] 导出图层...")
        output_path = Path(output_dir) / Path(image_path).stem
        output_path.mkdir(parents=True, exist_ok=True)
        
        export_info = {"canvas_size": list(self.canvas_size), "layers": []}
        
        for i, layer in enumerate(output_layers):
            safe_name = layer['name'].replace("'", "").replace('"', '').replace('/', '_').replace(' ', '_')
            filename = f"layer_{i:02d}_{safe_name}.png"
            filepath = output_path / filename
            
            layer["image"].save(filepath, "PNG")
            
            export_info["layers"].append({
                "index": i,
                "name": layer["name"],
                "type": layer["type"],
                "depth": layer["depth"],
                "file": str(filepath)
            })
            print(f"      ✓ {filename}")
        
        # 保存元数据
        with open(output_path / "project.json", "w", encoding="utf-8") as f:
            json.dump(export_info, f, ensure_ascii=False, indent=2)
        
        print(f"      导出完成: {output_path}")
        
        return {
            "status": "success",
            "input": image_path,
            "output_dir": str(output_path),
            "layer_count": len(output_layers)
        }


def decompose_image(image_path: str, output_dir: str) -> Dict[str, Any]:
    decomposer = GenerativeDecomposer()
    return decomposer.decompose(image_path, output_dir)
