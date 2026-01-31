import numpy as np
import torch
import cv2
from ..utilities import ColoredLogger

logger = ColoredLogger('ComfyUI-DepthAnythingTensorrt')


class DepthMapDisplay:
    """Display depth map with various color maps"""

    COLORMAPS = [
        "grayscale",
        "inferno",
        "viridis",
        "plasma",
        "magma",
        "turbo",
        "jet",
        "hot",
        "cool",
        "spring",
        "summer",
        "autumn",
        "winter",
        "bone",
        "rainbow",
        "ocean",
        "hsv",
        "parula",
        "pink"
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "depths": ("DEPTHS", {"tooltip": "Raw depth array from the Depth Anything Advanced node."}),
                "colormap": (s.COLORMAPS, {"tooltip": "Color scheme for visualizing the depth map. 'grayscale' outputs a single-channel gray image, others apply OpenCV colormaps."}),
                "invert": ("BOOLEAN", {"default": False, "tooltip": "Flip the depth values so that near becomes far and vice versa."}),
            },
            "optional": {
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1, "tooltip": "Adjusts the spread of depth values around the midpoint. Values above 1.0 increase separation, below 1.0 compress the range."}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05, "tooltip": "Shifts all depth values up or down. Positive values brighten the output, negative values darken it."}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1, "tooltip": "Non-linear tone curve. Values below 1.0 reveal more detail in dark/distant regions, above 1.0 reveals more detail in bright/near regions."}),
                "percentile_clip": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.5, "tooltip": "Clips outlier depth values at this percentile from both ends of the distribution before normalizing. Prevents extreme depth values from compressing the useful range. Set to 0 for no clipping."}),
            }
        }

    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("Colormapped depth visualization as an RGB image.",)
    FUNCTION = "display"
    CATEGORY = "tensorrt"
    OUTPUT_NODE = True
    DESCRIPTION = "Visualizes raw depth arrays using colormaps with adjustable contrast, brightness, gamma, and percentile clipping. Connect to the Depth Anything Advanced node's DEPTHS output."

    def display(self, depths, colormap, invert, contrast=1.0, brightness=0.0,
                gamma=1.0, percentile_clip=2.0):
        logger.info(f"Applying colormap: {colormap}")

        depth = depths.copy()

        # Percentile-based normalization: clips outlier depth values
        # at the given percentile from both ends of the distribution
        if percentile_clip > 0.0:
            depth_min = np.percentile(depth, percentile_clip)
            depth_max = np.percentile(depth, 100.0 - percentile_clip)
        else:
            depth_min = depth.min()
            depth_max = depth.max()

        if depth_min == depth_max:
            depth_min -= 1e-6
            depth_max += 1e-6

        depth = (depth - depth_min) / (depth_max - depth_min)
        depth = np.clip(depth, 0.0, 1.0)

        # Apply inversion
        if invert:
            depth = 1.0 - depth

        # Apply gamma correction
        if gamma != 1.0:
            depth = np.power(depth, gamma)

        # Apply contrast adjustment
        if contrast != 1.0:
            depth = (depth - 0.5) * contrast + 0.5

        # Apply brightness offset
        if brightness != 0.0:
            depth = depth + brightness

        depth = np.clip(depth, 0.0, 1.0)

        # Apply colormap
        if colormap == "grayscale":
            # Return as grayscale (replicate to 3 channels for IMAGE format)
            if len(depth.shape) == 3:  # [B, H, W]
                result = np.stack([depth] * 3, axis=-1)
            else:  # [B, H, W, C]
                result = np.stack([depth.mean(axis=-1)] * 3, axis=-1)
        else:
            colormap_dict = {
                "inferno": cv2.COLORMAP_INFERNO,
                "viridis": cv2.COLORMAP_VIRIDIS,
                "plasma": cv2.COLORMAP_PLASMA,
                "magma": cv2.COLORMAP_MAGMA,
                "turbo": cv2.COLORMAP_TURBO,
                "jet": cv2.COLORMAP_JET,
                "hot": cv2.COLORMAP_HOT,
                "cool": cv2.COLORMAP_COOL,
                "spring": cv2.COLORMAP_SPRING,
                "summer": cv2.COLORMAP_SUMMER,
                "autumn": cv2.COLORMAP_AUTUMN,
                "winter": cv2.COLORMAP_WINTER,
                "bone": cv2.COLORMAP_BONE,
                "rainbow": cv2.COLORMAP_RAINBOW,
                "ocean": cv2.COLORMAP_OCEAN,
                "hsv": cv2.COLORMAP_HSV,
                "parula": cv2.COLORMAP_PARULA,
                "pink": cv2.COLORMAP_PINK,
            }

            cv_colormap = colormap_dict.get(colormap, cv2.COLORMAP_VIRIDIS)

            # Apply colormap to each frame
            colored_frames = []
            for i in range(depth.shape[0]):
                # Get single frame and convert to uint8
                frame = depth[i]
                if len(frame.shape) == 3 and frame.shape[2] > 1:
                    # If multi-channel, take mean
                    frame = frame.mean(axis=2)

                frame_uint8 = (frame * 255).astype(np.uint8)
                colored = cv2.applyColorMap(frame_uint8, cv_colormap)
                colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

                # Normalize back to 0-1
                colored = colored.astype(np.float32) / 255.0
                colored_frames.append(colored)

            result = np.array(colored_frames)

        result = torch.from_numpy(result)
        return (result,)
