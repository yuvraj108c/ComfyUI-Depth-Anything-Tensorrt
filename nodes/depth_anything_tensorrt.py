import os
import folder_paths
import numpy as np
import torch.nn.functional as F
import torch
from comfy.utils import ProgressBar
import cv2
from ..trt_utilities import Engine
from ..utilities import ColoredLogger
from tqdm import tqdm

logger = ColoredLogger('ComfyUI-DepthAnythingTensorrt')
logger.info("Node loaded successfully ⚡")

ENGINE_DIR = os.path.join(folder_paths.models_dir,"tensorrt", "depth-anything")
os.makedirs(ENGINE_DIR, exist_ok=True)

class DepthAnythingTensorrt:
    def __init__(self):
        self.engine = None
        self.engine_label = None

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
    CATEGORY = "tensorrt"
    OUTPUT_NODE = True

    def main(self, images, engine):

        # setup tensorrt engine
        if (not hasattr(self, 'engine') or self.engine_label != engine):
            logger.info(f"Loading TensorRT engine: {engine}")
            self.engine = Engine(os.path.join(ENGINE_DIR,engine))
            self.engine.load()
            self.engine.activate()
            self.engine.allocate_buffers()
            self.engine_label = engine
        
        cudaStream = torch.cuda.current_stream().cuda_stream
        pbar = ProgressBar(images.shape[0])
        images = images.permute(0, 3, 1, 2)
        images_resized = F.interpolate(images, size=(518,518), mode='bilinear', align_corners=False)
        images_list = list(torch.split(images_resized, split_size_or_sections=1))

        depth_frames = []
        num_frames = len(images_list)

        # Custom tqdm format matching logger style
        bar_format = "\033[32m[ComfyUI-DepthAnythingTensorrt|INFO]\033[0m - {desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"

        progress_bar = tqdm(
            total=num_frames,
            desc="Processing",
            bar_format=bar_format,
            disable=(num_frames == 1)
        )

        for img in images_list:
            result = self.engine.infer({"input": img},cudaStream)
            depth = result['output']

            if "DA3" in engine:
                depth = self.postprocess_da3(depth)
            else:
                depth = self.postprocess_da2(depth)

            depth = cv2.resize(depth, (images.shape[3], images.shape[2]))
            depth_frames.append(depth)
            pbar.update(1)
            progress_bar.update(1)

        progress_bar.close()

        depth_frames_np = np.array(depth_frames).astype(np.float32) / 255.0
        if "metric" in engine:
            result = 1 - torch.from_numpy(depth_frames_np)
        else:
            result = torch.from_numpy(depth_frames_np)

        logger.info("Depth processing completed")
        return (result,)

    def postprocess_da3(self, depth_t):
        depth = depth_t.squeeze().cpu().numpy()
        depth = depth.copy()
        
        valid_mask = depth > 0
        depth[valid_mask] = 1 / depth[valid_mask]
        
        percentile = 2
        depth_min = None
        depth_max = None
        
        if depth_min is None:
            if valid_mask.sum() <= 10:
                depth_min = 0
            else:
                depth_min = np.percentile(depth[valid_mask], percentile)
        if depth_max is None:
            if valid_mask.sum() <= 10:
                depth_max = 0
            else:
                depth_max = np.percentile(depth[valid_mask], 100 - percentile)
        if depth_min == depth_max:
            depth_min = depth_min - 1e-6
            depth_max = depth_max + 1e-6
        
        depth = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)
        depth = (depth * 255.0).astype(np.uint8)
        
        return depth

    def postprocess_da2(self, depth_t):
        depth = np.reshape(depth_t.cpu().numpy(), (518,518))
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        return depth
        


