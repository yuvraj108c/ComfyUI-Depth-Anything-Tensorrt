# Place this file inside Depth-Anything-3 root directory (https://github.com/ByteDance-Seed/Depth-Anything-3)
# Unsqueeze x in dim 0 inside Depth-Anything-3/src/depth_anything_3/model/da3.py forward function (for compatibility)
# Dynamic shapes not working!

import torch
from depth_anything_3.api import DepthAnything3

device = torch.device("cuda")

api_model = DepthAnything3.from_pretrained("depth-anything/DA3Mono-Large").to(device)
api_model.eval()

dummy_input = torch.randn(1, 3, 518, 518, device=device)

torch.onnx.export(
    api_model.model,
    dummy_input,
    "/workspace/DA3Mono-Large.onnx",
    export_params=True,
    opset_version=20,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    verbose=True,
    dynamo=False, # else fails
)