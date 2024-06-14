<div align="center">

# ComfyUI Depth Anything TensorRT

[![python](https://img.shields.io/badge/python-3.10.12-green)](https://www.python.org/downloads/release/python-31012/)
[![cuda](https://img.shields.io/badge/cuda-12.3-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-10.0-green)](https://developer.nvidia.com/tensorrt)
[![by-nc-sa/4.0](https://img.shields.io/badge/license-CC--BY--NC--SA--4.0-lightgrey)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)

</div>

<p align="center">
  <img src="assets/demo.gif" />
</p>

This project is licensed under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/), everyone is FREE to access, use, modify and redistribute with the same license.  

For commercial purposes, please contact me directly at yuvraj108c@gmail.com

If you like the project, please give me a star! ⭐

****

This repo provides a ComfyUI Custom Node implementation of the [Depth-Anything-Tensorrt](https://github.com/spacewalk01/depth-anything-tensorrt) in Python for ultra fast depth map generation (up to 14x faster compared to [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux))

## ⏱️ Performance
_Note: The following results were benchmarked on FP16 engines inside ComfyUI_

| Device | Model | Model Input (WxH) | Image Resolution (WxH)|FPS
|:---------------:|:------------:|:------------:|:------------:|:------------:|
| RTX4090 | Depth-Anything-S |518x518 | 1280x720 | 35
| RTX4090 | Depth-Anything-B |518x518 | 1280x720 | 33
| RTX4090 | Depth-Anything-L |518x518 | 1280x720 | 24
| H100 | Depth-Anything-L |518x518 | 1280x720 | 125+

## 🚀 Installation

Navigate to the ComfyUI `/custom_nodes` directory

```bash
git clone https://github.com/yuvraj108c/ComfyUI-Depth-Anything-Tensorrt.git
cd ./ComfyUI-Depth-Anything-Tensorrt
pip install -r requirements.txt
```

## 🛠️ Building Tensorrt Engine

1. Download one of the available [onnx models](https://huggingface.co/yuvraj108c/Depth-Anything-Onnx/tree/main) _(e.g depth_anything_vitl14.onnx)_
2. Edit model paths inside [export_trt.py](export_trt.py) accordingly and run `python export_trt.py`
3. Place the exported engine inside ComfyUI `/models/tensorrt/depth-anything` directory

## ☀️ Usage
- Insert node by `Right Click -> tensorrt -> Depth Anything Tensorrt`
- Choose the appropriate engine from the dropdown

## 🤖 Environment tested

- Ubuntu 22.04 LTS, Cuda 12.3, Tensorrt 10.0.1, Python 3.10, RTX 4090 GPU
- Windows (Not tested, but should work)

## 📝 Changelog
- 08/05/2024

  - Clean utilities.py
  - Fix engine path in custom node
  - Add citation in readme

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

```bibtex
@inproceedings{depthanything,
      title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
      author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
      booktitle={CVPR},
      year={2024}
}
```
- [NVIDIA/Stable-Diffusion-WebUI-TensorRT](https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT)
- [spacewalk01/depth-anything-tensorrt](https://github.com/spacewalk01/depth-anything-tensorrt)
- [martenwikman/depth-anything-tensorrt-docker](https://github.com/martenwikman/depth-anything-tensorrt-docker)

## License
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
