<div align="center">

# ComfyUI Depth Anything TensorRT

[![python](https://img.shields.io/badge/python-3.10.12-green)](https://www.python.org/downloads/release/python-31012/)
[![cuda](https://img.shields.io/badge/cuda-12.3-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-10.0-green)](https://developer.nvidia.com/tensorrt)
[![mit](https://img.shields.io/badge/license-MIT-blue)](https://github.com/spacewalk01/depth-anything-tensorrt/blob/main/LICENSE)

</div>

This repo provides a ComfyUI Custom Node implementation of the [Depth-Anything-Tensorrt](https://github.com/spacewalk01/depth-anything-tensorrt) in Python for ultra fast depth map generation (up to 14x faster than [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux))

<p align="center">
  <img src="assets/demo.gif" />
</p>



## ⭐ Support
If you like my projects and wish to see updates and new features, please consider supporting me. It helps a lot! 

[![ComfyUI-Depth-Anything-Tensorrt](https://img.shields.io/badge/ComfyUI--Depth--Anything--Tensorrt-blue?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Depth-Anything-Tensorrt)
[![ComfyUI-Upscaler-Tensorrt](https://img.shields.io/badge/ComfyUI--Upscaler--Tensorrt-blue?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt)
[![ComfyUI-Dwpose-Tensorrt](https://img.shields.io/badge/ComfyUI--Dwpose--Tensorrt-blue?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Dwpose-Tensorrt)
[![ComfyUI-Rife-Tensorrt](https://img.shields.io/badge/ComfyUI--Rife--Tensorrt-blue?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Rife-Tensorrt)

[![ComfyUI-Whisper](https://img.shields.io/badge/ComfyUI--Whisper-gray?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Whisper)
[![ComfyUI_InvSR](https://img.shields.io/badge/ComfyUI__InvSR-gray?style=flat-square)](https://github.com/yuvraj108c/ComfyUI_InvSR)
[![ComfyUI-FLOAT](https://img.shields.io/badge/ComfyUI--FLOAT-gray?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-FLOAT)
[![ComfyUI-Thera](https://img.shields.io/badge/ComfyUI--Thera-gray?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Thera)
[![ComfyUI-Video-Depth-Anything](https://img.shields.io/badge/ComfyUI--Video--Depth--Anything-gray?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything)
[![ComfyUI-PiperTTS](https://img.shields.io/badge/ComfyUI--PiperTTS-gray?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-PiperTTS)

Special thanks to [livepeer.org](https://www.livepeer.org/) for supporting the project! 

[![buy-me-coffees](https://i.imgur.com/3MDbAtw.png)](https://www.buymeacoffee.com/yuvraj108cZ)
[![paypal-donation](https://i.imgur.com/w5jjubk.png)](https://paypal.me/yuvraj108c)
---

## ⏱️ Performance (Depth Anything V1)

_Note: The following results were benchmarked on FP16 engines inside ComfyUI_

| Device  |      Model       | Model Input (WxH) | Image Resolution (WxH) | FPS  |
| :-----: | :--------------: | :---------------: | :--------------------: | :--: |
| RTX4090 | Depth-Anything-S |      518x518      |        1280x720        |  35  |
| RTX4090 | Depth-Anything-B |      518x518      |        1280x720        |  33  |
| RTX4090 | Depth-Anything-L |      518x518      |        1280x720        |  24  |
|  H100   | Depth-Anything-L |      518x518      |        1280x720        | 125+ |

## ⏱️ Performance (Depth Anything V2)

_Note: The following results were benchmarked on FP16 engines inside ComfyUI_

| Device |      Model       | Model Input (WxH) | Image Resolution (WxH) | FPS |
| :----: | :--------------: | :---------------: | :--------------------: | :-: |
|  H100  | Depth-Anything-S |      518x518      |        1280x720        | 213 |
|  H100  | Depth-Anything-B |      518x518      |        1280x720        | 180 |
|  H100  | Depth-Anything-L |      518x518      |        1280x720        | 109 |

## ⏱️ Performance (Distill Any Depth)

_Note: The following results were benchmarked on FP16 engines inside ComfyUI_

| Device |      Model       | Model Input (WxH) | Image Resolution (WxH) | FPS |
| :----: | :--------------: | :---------------: | :--------------------: | :-: |
|  H100  | Distill-Any-Depth-Multi-Teacher-Small |      518x518      |        1280x720        | 76 |
|  H100  | Distill-Any-Depth-Multi-Teacher-Base |      518x518      |        1280x720        | 68 |
|  H100  | Distill-Any-Depth-Multi-Teacher-Large |      518x518      |        1280x720        | 59 |
|  H100  | Distill-Any-Depth-Dav2-Teacher-Large-2w-iter |      518x518      |        1280x720        | 57 |

## 🚀 Installation

Navigate to the ComfyUI `/custom_nodes` directory

```bash
git clone https://github.com/yuvraj108c/ComfyUI-Depth-Anything-Tensorrt.git
cd ./ComfyUI-Depth-Anything-Tensorrt
pip install -r requirements.txt
```

## 🛠️ Building TensorRT Engine

There are two ways to build TensorRT engines:

### Method 1: Using the EngineBuilder Node
1. Insert node by `Right Click -> tensorrt -> Depth Anything Engine Builder`
2. Select the model version (v1 or v2 or DAD) and size (small, base, or large)
3. Optionally customize the engine name, FP16 settings, and onnx path
4. Run the workflow to build the engine

The engine will be automatically downloaded and built in the specified location. Refresh the webpage or strike 'r' on your keyboard, and the new engine will appear in the Depth Anything Tensorrt node. 

### Method 2: Manual Building
1. Download one of the available onnx models:
   - [Depth Anything v1](https://huggingface.co/yuvraj108c/Depth-Anything-Onnx/tree/main)
   - [Depth Anything v2](https://huggingface.co/yuvraj108c/Depth-Anything-2-Onnx/tree/main)
   - [Distill Any Depth](https://huggingface.co/yuvraj108c/distill-any-depth-onnx/tree/main)
2. Run the export script, e.g
 ```bash
python export_trt.py --onnx-path ./depth_anything_vitl14-fp16.onnx --trt-path ./depth_anything_vitl14-fp16.engine
 ```
3. Place the exported engine inside ComfyUI `/models/tensorrt/depth-anything` directory

## ☀️ Usage

- Insert node by `Right Click -> tensorrt -> Depth Anything Tensorrt`
- Choose the appropriate engine from the dropdown

## 🤖 Environment tested

- Ubuntu 22.04 LTS, Cuda 12.3, Tensorrt 10.0.1, Python 3.10, RTX 4090 GPU
- Windows (Not tested)

## 📝 Changelog

- 16/09/2025

  - Add support for v2 metric models (depth_anything_v2_metric_hypersim_vitl, depth_anything_v2_metric_vkitti_vitl) https://github.com/yuvraj108c/ComfyUI-Depth-Anything-Tensorrt/pull/21
    
- 08/07/2025

  - Add support for [Distill-Any-Depth](https://github.com/Westlake-AGI-Lab/Distill-Any-Depth)
  - Add benchmark for Distill-Any-Depth models

- 20/05/2025

  - Merge [PR#15](https://github.com/yuvraj108c/ComfyUI-Depth-Anything-Tensorrt/pull/15) for auto engine building inside comfyui by [ryanontheinside](https://github.com/ryanontheinside)
  - Merge [PR#14](https://github.com/yuvraj108c/ComfyUI-Depth-Anything-Tensorrt/pull/14) for configurable params in export_trt.py by [rickstaa](https://github.com/rickstaa)
    
- 02/07/2024

  - Add Depth Anything V2 onnx models + benchmarks
  - Merge [PR](https://github.com/yuvraj108c/ComfyUI-Depth-Anything-Tensorrt/pull/9) for engine caching in memory by [BuffMcBigHuge](https://github.com/BuffMcBigHuge)

- 26/04/2024

  - Update to tensorrt 10.0.1
  - Massive code refactor, remove trtexec, remove pycuda, show engine building progress
  - Update and standardise engine directory and node category for upcoming tensorrt custom nodes suite

- 7/04/2024

  - Fix image resize bug during depth map post processing

- 30/03/2024

  - Fix CUDNN_STATUS_MAPPING_ERROR

- 27/03/2024

  - Major refactor and optimisation (remove subprocess)

## 👏 Credits

- [NVIDIA/Stable-Diffusion-WebUI-TensorRT](https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT)
- [spacewalk01/depth-anything-tensorrt](https://github.com/spacewalk01/depth-anything-tensorrt)
- [martenwikman/depth-anything-tensorrt-docker](https://github.com/martenwikman/depth-anything-tensorrt-docker)
- [Westlake-AGI-Lab/Distill-Any-Depth](https://github.com/Westlake-AGI-Lab/Distill-Any-Depth)
- [kijai/ComfyUI-DepthAnythingV2](https://github.com/kijai/ComfyUI-DepthAnythingV2)
