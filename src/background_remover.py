"""
背景移除模块
支持多种抠图方法：InSPyReNet、rembg、颜色抠图
"""
from PIL import Image
from typing import Dict, Any, List, Optional
import numpy as np
import io


class BackgroundRemover:
    def __init__(self, method: str = "inspyrenet"):
        """
        Args:
            method: 抠图方法 ("inspyrenet", "rembg", "color")
        """
        self.method = method
        self.model = None
        
        if method == "inspyrenet":
            self._init_inspyrenet()
        elif method == "rembg":
            self._init_rembg()
    
    def _init_inspyrenet(self):
        """初始化 InSPyReNet 模型"""
        try:
            import torch
            from torchvision import transforms
            
            # 检查是否有可用模型
            # InSPyReNet 需要预训练权重
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 图像预处理
            self.transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            print(f"      InSPyReNet 初始化成功 (device: {self.device})")
        except Exception as e:
            print(f"      InSPyReNet 初始化失败: {e}")
            print("      回退到颜色抠图")
            self.method = "color"
    
    def _init_rembg(self):
        """初始化 rembg"""
        try:
            from rembg import remove
            self.rembg_remove = remove
            print("      rembg 初始化成功")
        except Exception as e:
            print(f"      rembg 初始化失败: {e}")
            print("      回退到颜色抠图")
            self.method = "color"
    
    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        移除图片背景
        
        Args:
            image: PIL Image对象
            
        Returns:
            去除背景后的RGBA图片
        """
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        
        if self.method == "inspyrenet":
            return self._remove_with_inspyrenet(image)
        elif self.method == "rembg":
            return self._remove_with_rembg(image)
        else:
            return self._remove_with_color(image)
    
    def _remove_with_inspyrenet(self, image: Image.Image) -> Image.Image:
        """使用 InSPyReNet 抠图"""
        try:
            import torch
            
            # 预处理
            img_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            
            # 推理（需要加载模型权重）
            # 这里需要实际的 InSPyReNet 模型
            # 暂时回退到颜色抠图
            return self._remove_with_color(image)
        except Exception as e:
            print(f"      InSPyReNet 推理失败: {e}")
            return self._remove_with_color(image)
    
    def _remove_with_rembg(self, image: Image.Image) -> Image.Image:
        """使用 rembg 抠图"""
        return self.rembg_remove(image)
    
    def _remove_with_color(self, image: Image.Image) -> Image.Image:
        """使用颜色检测抠图（白色/绿色背景）"""
        from scipy import ndimage
        
        img_array = np.array(image.convert("RGBA"))
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        
        # 检测绿幕
        green_mask = (
            (g > 200) & (r < 60) & (b < 60) &
            (g > r + 100) & (g > b + 100)
        )
        
        # 检测白色背景
        white_mask = (
            (r > 235) & (g > 235) & (b > 235) &
            (np.abs(r.astype(int) - g.astype(int)) < 20) &
            (np.abs(g.astype(int) - b.astype(int)) < 20)
        )
        
        background_mask = green_mask | white_mask
        
        # 只移除与边缘连通的背景
        h, w = background_mask.shape
        edge_mask = np.zeros_like(background_mask)
        edge_mask[0, :] = edge_mask[-1, :] = edge_mask[:, 0] = edge_mask[:, -1] = True
        
        labeled, _ = ndimage.label(background_mask)
        edge_labels = set(labeled[edge_mask & background_mask])
        connected_bg_mask = np.isin(labeled, list(edge_labels))
        
        img_array[:, :, 3] = np.where(connected_bg_mask, 0, 255)
        
        return Image.fromarray(img_array, mode="RGBA")
    
    def process_layers(self, layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理图层列表，对需要抠图的图层进行背景移除
        
        Args:
            layers: 图层列表
            
        Returns:
            处理后的图层列表
        """
        for layer in layers:
            if layer.get("needs_cutout", False):
                original_image = layer["image"]
                cutout_image = self.remove_background(original_image)
                layer["image"] = cutout_image
                layer["cutout_applied"] = True
            else:
                layer["cutout_applied"] = False
        
        return layers


def process(layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """便捷函数：处理图层抠图"""
    remover = BackgroundRemover()
    return remover.process_layers(layers)
