import os
import folder_paths
import numpy as np
import torch.nn.functional as F
import torch
import comfy
import cv2
from .utilities import Engine

ENGINE_DIR = os.path.join(folder_paths.models_dir,"depth_trt_engines")

class DepthAnythingTensorrtNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "images": ("IMAGE",),
                "engine": (os.listdir(ENGINE_DIR),),
            }
        }
    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "Depth Anything Tensorrt"

    def main(self, images, engine):

        # setup tensorrt engine
        engine = Engine(os.path.join(ENGINE_DIR,engine))
        engine.load()
        engine.activate()
        engine.allocate_buffers()
        cudaStream = torch.cuda.current_stream().cuda_stream

        pbar = comfy.utils.ProgressBar(images.shape[0])
        images = images.permute(0, 3, 1, 2)
        images_resized = F.interpolate(images, size=(518,518), mode='bilinear', align_corners=False)
        images_list = list(torch.split(images_resized, split_size_or_sections=1))

        depth_frames = []

        for img in images_list:
            result = engine.infer({"input": img},cudaStream)
            depth = result['output']

            # Process the depth output
            depth = np.reshape(depth.cpu().numpy(), (518,518))
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            depth = cv2.resize(depth, (images.shape[3], images.shape[2]))
            depth = cv2.cvtColor(depth, cv2.COLOR_RGB2BGR)

            depth_frames.append(depth)
            pbar.update(1)
        
        depth_frames_np = np.array(depth_frames).astype(np.float32) / 255.0
        return (torch.from_numpy(depth_frames_np),)


NODE_CLASS_MAPPINGS = { 
    "DepthAnythingTensorrtNode" : DepthAnythingTensorrtNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
     "DepthAnythingTensorrtNode" : "Depth Anything Tensorrt",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']