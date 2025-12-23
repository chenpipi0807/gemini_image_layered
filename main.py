"""
PIP-Image-Layered V7 主入口
生成式图层分解 - 模仿 Qwen-Image-Layered 的思路

核心：用 Gemini 图像生成模型逐层重新绘制每个元素

使用方法:
    python main_v7.py <图片路径> [输出目录]
"""
import sys
import argparse
from pathlib import Path

from src.generative_decomposer import GenerativeDecomposer


def main():
    parser = argparse.ArgumentParser(
        description="图像分层处理工具 V7 - 生成式分解（并行生成）"
    )
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument(
        "-o", "--output", 
        default="./output_v7",
        help="输出目录 (默认: ./output_v7)"
    )
    parser.add_argument(
        "-m", "--method",
        default="color",
        choices=["color", "rembg", "inspyrenet"],
        help="背景移除方法 (默认: color)"
    )
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"错误: 图片不存在 - {image_path}")
        sys.exit(1)
    
    decomposer = GenerativeDecomposer(bg_removal_method=args.method)
    
    print(f"处理图片: {image_path}")
    print(f"输出目录: {args.output}")
    print("=" * 50)
    
    result = decomposer.decompose(str(image_path), args.output)
    
    print("=" * 50)
    if result["status"] == "success":
        print(f"处理完成!")
        print(f"输出目录: {result['output_dir']}")
        print(f"图层数量: {result['layer_count']}")
    else:
        print(f"处理失败: {result.get('message', '未知错误')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
