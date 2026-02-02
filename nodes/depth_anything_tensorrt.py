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

class _DepthAnythingBase:
    """Shared base for DepthAnythingTensorrt nodes."""

    CATEGORY = "tensorrt"

    def __init__(self):
        self.engine = None
        self.engine_label = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input image or batch of images to estimate depth for."}),
                "engine": (os.listdir(ENGINE_DIR), {"tooltip": "TensorRT engine file to use for inference. Engine names containing 'DA3' use Depth Anything V3, others use V2/V1 postprocessing."}),
            }
        }

    def _ensure_engine(self, engine):
        if not hasattr(self, 'engine') or self.engine_label != engine:
            logger.info(f"Loading TensorRT engine: {engine}")
            self.engine = Engine(os.path.join(ENGINE_DIR, engine))
            self.engine.load()
            self.engine.activate()
            self.engine.allocate_buffers()
            self.engine_label = engine

    def _run_inference(self, images, engine, desc, process_frame):
        self._ensure_engine(engine)

        cudaStream = torch.cuda.current_stream().cuda_stream
        pbar = ProgressBar(images.shape[0])
        images = images.permute(0, 3, 1, 2)
        images_resized = F.interpolate(images, size=(518, 518), mode='bilinear', align_corners=False)
        images_list = list(torch.split(images_resized, split_size_or_sections=1))

        depth_frames = []
        num_frames = len(images_list)

        bar_format = "\033[32m[ComfyUI-DepthAnythingTensorrt|INFO]\033[0m - {desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"

        progress_bar = tqdm(
            total=num_frames,
            desc=desc,
            bar_format=bar_format,
            disable=(num_frames == 1)
        )

        for img in images_list:
            result = self.engine.infer({"input": img}, cudaStream)
            depth = result['output']
            depth = process_frame(depth, engine, images)
            depth_frames.append(depth)
            pbar.update(1)
            progress_bar.update(1)

        progress_bar.close()
        logger.info("Depth processing completed")
        return depth_frames, images

    @staticmethod
    def _postprocess_da3(depth_t):
        depth = depth_t.squeeze().cpu().numpy().copy()

        valid_mask = depth > 0
        depth[valid_mask] = 1 / depth[valid_mask]

        percentile = 2
        if valid_mask.sum() <= 10:
            depth_min, depth_max = 0, 0
        else:
            depth_min = np.percentile(depth[valid_mask], percentile)
            depth_max = np.percentile(depth[valid_mask], 100 - percentile)
        if depth_min == depth_max:
            depth_min -= 1e-6
            depth_max += 1e-6

        depth = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)
        depth = (depth * 255.0).astype(np.uint8)
        return depth

    @staticmethod
    def _postprocess_da2(depth_t):
        depth = np.reshape(depth_t.cpu().numpy(), (518, 518))
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        return depth

class DepthAnythingTensorrt(_DepthAnythingBase):
    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("Normalized depth map as a grayscale image (0-1 range). For metric engines, depth is inverted so closer objects are brighter.",)
    FUNCTION = "main"
    CATEGORY = "tensorrt"
    OUTPUT_NODE = True
    DESCRIPTION = "Estimates monocular depth from images using a Depth Anything TensorRT engine. Returns a postprocessed grayscale depth image suitable for direct use in workflows."

    def main(self, images, engine):
        def process_frame(depth, engine, orig_images):
            if "DA3" in engine:
                depth = self._postprocess_da3(depth)
            else:
                depth = self._postprocess_da2(depth)
            return cv2.resize(depth, (orig_images.shape[3], orig_images.shape[2]))

        depth_frames, _ = self._run_inference(images, engine, "Processing", process_frame)

        depth_frames_np = np.array(depth_frames).astype(np.float32) / 255.0
        if "metric" in engine:
            result = 1 - torch.from_numpy(depth_frames_np)
        else:
            result = torch.from_numpy(depth_frames_np)
        return (result,)

class DepthAnythingTensorrtAdvanced(_DepthAnythingBase):
    RETURN_NAMES = ("depths",)
    RETURN_TYPES = ("DEPTHS",)
    OUTPUT_TOOLTIPS = ("Raw linear depth values as a float32 numpy array. Use with the Depth Map Display node for visualization with colormaps and adjustments.",)
    FUNCTION = "process"
    CATEGORY = "tensorrt"
    DESCRIPTION = "Estimates monocular depth from images using a Depth Anything TensorRT engine. Returns raw linear depth values for downstream processing such as colormap visualization, 3D reconstruction, or custom depth manipulation."

    def process(self, images, engine):
        def process_frame(depth, engine, orig_images):
            depth_np = depth.squeeze().cpu().numpy()
            # DA3 outputs inverse depth; convert to linear depth
            if "DA3" in engine:
                valid = depth_np > 0
                depth_np[valid] = 1.0 / depth_np[valid]
            return cv2.resize(depth_np, (orig_images.shape[3], orig_images.shape[2]))

        depth_frames, _ = self._run_inference(images, engine, "Processing", process_frame)

        depth_frames_np = np.array(depth_frames).astype(np.float32)
        return (depth_frames_np,)

