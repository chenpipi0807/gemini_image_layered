# gemini-Image-Layered

基于 Gemini 的智能图像分层工具 - 生成式图层分解

## 📖 简介

PIP-Image-Layered 是一个创新的图像分层工具，使用 Google Gemini 多模态 AI 模型实现智能图像分解。与传统的分割方法不同，本工具采用**生成式**方法，让 AI 重新绘制每个图层元素，从而获得更干净、更完整的分层效果。

### 核心特性

- 🎨 **生成式分层**：不是简单切图，而是让 AI 重新绘制每个元素
- 🧠 **智能语义理解**：使用 Gemini VL 模型深度理解图像结构
- 🔄 **遮挡补全**：自动补全被遮挡的部分，生成完整图层
- ⚡ **高并发处理**：支持并行生成多个图层，速度快
- 🎯 **多种抠图方法**：支持颜色抠图、rembg、InSPyReNet 等多种背景移除方案
- 📦 **即开即用**：自动生成透明背景的 RGBA 图层

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置 API

创建 `.env` 文件，配置 Gemini API：

```env
LLM_API_KEY=your_gemini_api_key
LLM_BASE_URL=https://your-proxy-url/
REASONING_MODEL=gemini-3-pro-preview
IMAGE_MODEL=gemini-3-pro-image-preview
```

### 运行

```bash
# 基础用法
python main.py ./test.png

# 指定输出目录
python main.py ./test.png -o ./output

# 选择抠图方法
python main.py ./test.png -m color    # 颜色抠图（默认）
python main.py ./test.png -m rembg    # 使用 rembg
python main.py ./test.png -m inspyrenet  # 使用 InSPyReNet
```

## 📊 工作原理

### 生成式分层流程

```
输入图片
    ↓
[1] Gemini VL 语义分析
    - 识别所有可分层元素
    - 分析深度关系和遮挡情况
    - 生成详细的元素描述
    ↓
[2] 并行生成图层（5个并发）
    - 根据描述重新绘制每个元素
    - 生成纯色背景（白色）
    - 自动补全被遮挡部分
    ↓
[3] 背景移除
    - 智能抠图，转为透明背景
    - 保留元素细节和边缘
    ↓
[4] 导出 RGBA 图层
    - 按深度排序
    - 生成元数据 JSON
```

### 核心优势

与传统分割方法（SAM、rembg）相比：

| 特性 | 传统分割 | 生成式分层 |
|------|---------|-----------|
| 遮挡处理 | ❌ 无法补全 | ✅ 自动补全 |
| 边缘质量 | ⚠️ 可能有锯齿 | ✅ 干净平滑 |
| 语义理解 | ❌ 基于像素 | ✅ 深度理解 |
| 风格一致性 | ⚠️ 可能失真 | ✅ 保持原风格 |
| 处理速度 | ✅ 快 | ⚠️ 需要 API 调用 |

## 🛠️ 抠图方法对比

### 1. Color（颜色抠图）- 默认

- **优点**：快速、无需额外依赖、适合 Gemini 生成的图像
- **缺点**：只能处理纯色背景
- **适用场景**：Gemini 生成的白色/绿色背景图层

### 2. Rembg

- **优点**：通用性强、效果稳定
- **缺点**：需要下载模型、速度较慢
- **适用场景**：复杂背景、真实照片

### 3. InSPyReNet

- **优点**：高精度、边缘细腻
- **缺点**：需要预训练权重、GPU 加速
- **适用场景**：专业级抠图需求

## 📁 输出结构

```
output_v7/
└── test/
    ├── layer_00_青色背景.png
    ├── layer_01_粉色Girls文字.png
    ├── layer_02_黑色girl文字.png
    ├── layer_03_滑板女孩.png
    ├── layer_04_绿色滑板.png
    └── project.json  # 元数据
```

### project.json 示例

```json
{
  "canvas_size": [1328, 1328],
  "layers": [
    {
      "name": "青色背景",
      "type": "background",
      "depth": 0,
      "file": "layer_00_青色背景.png",
      "description": "复古像素风格的青色背景"
    },
    {
      "name": "滑板女孩",
      "type": "person",
      "depth": 7,
      "file": "layer_03_滑板女孩.png",
      "description": "像素风格的滑板女孩，穿着彩色T恤和蓝色短裤"
    }
  ]
}
```

## ⚙️ 高级配置

### 并发数调整

编辑 `src/generative_decomposer.py`：

```python
# 默认 5 个并发
with ThreadPoolExecutor(max_workers=5) as executor:
```

### 自定义抠图参数

编辑 `src/background_remover.py` 中的阈值：

```python
# 白色背景检测阈值
white_mask = (
    (r > 235) &  # 调整这个值
    (g > 235) &
    (b > 235) &
    ...
)
```

## 🔧 故障排除

### API 超时

如果遇到 `Read timed out` 错误：
- 检查网络连接
- 减少并发数（`max_workers`）
- 增加 timeout 时间

### 抠图效果不佳

1. 尝试不同的抠图方法：`-m rembg`
2. 调整颜色检测阈值
3. 检查 Gemini 生成的背景颜色是否符合预期

### 图层不完整

- 检查 Gemini 的语义分析结果
- 调整 prompt 描述的详细程度
- 确保 API 配置正确

## 📝 开发路线

- [x] 基于 Gemini 的语义分析
- [x] 生成式图层重绘
- [x] 并行生成优化
- [x] 多种抠图方法支持
- [ ] InSPyReNet 完整集成
- [ ] GUI 界面
- [ ] 批量处理
- [ ] 图层编辑功能

## 📄 许可证

MIT License

## 🙏 致谢

- Google Gemini API
- Segment Anything Model (SAM)
- rembg
- InSPyReNet

---

**注意**：本工具需要 Gemini API 访问权限。生成质量取决于 API 的响应和网络状况。

