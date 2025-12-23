# PIP-Image-Layered

AI-powered image layer decomposition using Google Gemini - Generative approach

[中文文档](README_CN.md)

## Overview

PIP-Image-Layered is an innovative image layering tool that uses Google Gemini's multimodal AI to intelligently decompose images. Unlike traditional segmentation methods, this tool uses a **generative** approach where AI redraws each layer element for cleaner, more complete results.

## Features

- 🎨 **Generative Layering**: AI redraws each element instead of simple cropping
- 🧠 **Semantic Understanding**: Deep image structure analysis using Gemini VL
- 🔄 **Occlusion Completion**: Automatically completes hidden parts
- ⚡ **Parallel Processing**: Concurrent layer generation for speed
- 🎯 **Multiple Matting Methods**: Color-based, rembg, InSPyReNet support
- 📦 **Ready to Use**: Automatic RGBA layer generation with transparency

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file with your Gemini API credentials:

```env
LLM_API_KEY=your_gemini_api_key
LLM_BASE_URL=https://your-proxy-url/
REASONING_MODEL=gemini-3-pro-preview
IMAGE_MODEL=gemini-3-pro-image-preview
```

### Usage

```bash
# Basic usage
python main.py ./image.png

# Specify output directory
python main.py ./image.png -o ./output

# Choose matting method
python main.py ./image.png -m color      # Color-based (default)
python main.py ./image.png -m rembg      # Use rembg
python main.py ./image.png -m inspyrenet # Use InSPyReNet
```

## How It Works

```
Input Image
    ↓
[1] Gemini VL Semantic Analysis
    - Identify all layerable elements
    - Analyze depth and occlusion
    - Generate detailed descriptions
    ↓
[2] Parallel Layer Generation (5 concurrent)
    - Redraw each element from description
    - Generate with solid background
    - Auto-complete occluded parts
    ↓
[3] Background Removal
    - Smart matting to transparency
    - Preserve edge details
    ↓
[4] Export RGBA Layers
    - Sorted by depth
    - With metadata JSON
```

## Output Structure

```
output/
└── image/
    ├── layer_00_background.png
    ├── layer_01_text.png
    ├── layer_02_character.png
    └── project.json
```

## Matting Methods

| Method | Pros | Cons | Use Case |
|--------|------|------|----------|
| **color** | Fast, no extra deps | Solid backgrounds only | Gemini-generated images |
| **rembg** | Universal, stable | Requires model download | Complex backgrounds |
| **inspyrenet** | High precision | Needs pretrained weights | Professional quality |

## Requirements

- Python 3.8+
- Gemini API access
- See `requirements.txt` for dependencies

## License

MIT License

## Acknowledgments

- Google Gemini API
- rembg
- InSPyReNet
