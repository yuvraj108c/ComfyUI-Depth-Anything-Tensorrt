# made by gpt 5.5

import numpy as np
import torch

from comfy.utils import ProgressBar
from ..utilities import ColoredLogger


logger = ColoredLogger("ComfyUI-DepthAnythingTensorrt")


class DepthTemporalStabilizerGPU:
    """
    Fast CUDA temporal stabilizer for raw depth-map video sequences.

    Workflow:
        Depth Anything Tensorrt Advanced
                ↓ DEPTHS
        Depth Temporal Stabilizer Fast GPU
                ↓ DEPTHS
        Depth Map Display

    Best for:
        - Mild frame-to-frame depth flicker.
        - Global depth-range pulsing.
        - Small surface instability.
        - Fast video-depth workflows.

    Limitation:
        - Because there is no motion compensation, high temporal strength can
          cause trailing on fast-moving edges or strong camera motion.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depths": (
                    "DEPTHS",
                    {
                        "tooltip": (
                            "Raw DEPTHS output from Depth Anything Tensorrt Advanced. "
                            "Do not use visualised grayscale depth images."
                        )
                    },
                ),
                "temporal_strength": (
                    "FLOAT",
                    {
                        "default": 0.45,
                        "min": 0.0,
                        "max": 0.95,
                        "step": 0.01,
                        "tooltip": (
                            "How much stable pixels inherit depth from the previous "
                            "stabilised frame. Higher reduces more flicker but can cause trails."
                        ),
                    },
                ),
                "depth_consistency": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.001,
                        "max": 0.50,
                        "step": 0.001,
                        "tooltip": (
                            "Only pixels with similar depth values are smoothed. "
                            "Lower values better protect movement and edges."
                        ),
                    },
                ),
                "align_depth_range": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Correct global frame-to-frame relative depth scale and "
                            "offset pulsing before local temporal smoothing."
                        ),
                    },
                ),
                "alignment_strength": (
                    "FLOAT",
                    {
                        "default": 0.70,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "How strongly global depth-range alignment is applied. "
                            "This usually has less ghosting risk than increasing temporal strength."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("DEPTHS",)
    RETURN_NAMES = ("depths",)
    FUNCTION = "stabilize"
    CATEGORY = "tensorrt"
    DESCRIPTION = (
        "Fast CUDA temporal smoothing for raw video depth maps. "
        "Reduces mild flicker and depth-range pulsing without optical flow."
    )

    @staticmethod
    def _to_numpy_depths(depths) -> np.ndarray:
        """
        Convert incoming DEPTHS into contiguous float32 numpy format.

        The current Advanced TensorRT node outputs numpy depth maps, but this
        also safely accepts a torch tensor if the interface changes later.
        """
        if isinstance(depths, torch.Tensor):
            depths = depths.detach().float().cpu().numpy()

        depths_np = np.asarray(depths, dtype=np.float32)

        if depths_np.ndim != 3:
            raise ValueError(
                "Depth Temporal Stabilizer Fast GPU expected DEPTHS shaped "
                f"[frames, height, width], received {depths_np.shape}."
            )

        return np.ascontiguousarray(depths_np)

    @staticmethod
    def _align_current_to_previous(
        current: torch.Tensor,
        previous: torch.Tensor,
        alignment_strength: float,
        alignment_quantiles: torch.Tensor,
    ) -> torch.Tensor:
        """
        Match the current frame's robust relative-depth range to the previous
        stabilised frame.

        Fits a simple scale and shift using:
            q10, q50, q90

        This targets whole-frame relative-depth pulsing without directly
        averaging unrelated moving pixels.
        """
        current_q = torch.quantile(current.flatten(), alignment_quantiles)
        previous_q = torch.quantile(previous.flatten(), alignment_quantiles)

        current_range = current_q[2] - current_q[0]
        previous_range = previous_q[2] - previous_q[0]

        valid_range = (
            (current_range.abs() >= 1e-6)
            & (previous_range.abs() >= 1e-6)
        )

        safe_current_range = torch.clamp(current_range, min=1e-6)

        scale = previous_range / safe_current_range
        scale = torch.clamp(scale, min=0.50, max=2.00)

        shift = previous_q[1] - current_q[1] * scale

        aligned = current * scale + shift
        aligned = torch.where(valid_range, aligned, current)

        return torch.lerp(
            current,
            aligned,
            float(alignment_strength),
        )

    @staticmethod
    def _depth_range(
        depth: torch.Tensor,
        range_quantiles: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate a robust depth range used to express temporal differences
        independently of the current video's absolute relative-depth values.
        """
        values = torch.quantile(depth.flatten(), range_quantiles)

        return torch.clamp(
            values[1] - values[0],
            min=1e-6,
        )

    def stabilize(
        self,
        depths,
        temporal_strength=0.45,
        depth_consistency=0.05,
        align_depth_range=True,
        alignment_strength=0.70,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Depth Temporal Stabilizer Fast GPU requires an NVIDIA CUDA GPU."
            )

        depths_np = self._to_numpy_depths(depths)
        frame_count = depths_np.shape[0]

        if frame_count <= 1 or temporal_strength <= 0.0:
            return (depths_np.copy(),)

        device = torch.device("cuda")

        logger.info(
            "GPU temporal stabilisation: "
            f"{frame_count} frames, "
            f"strength={temporal_strength:.2f}, "
            f"consistency={depth_consistency:.3f}, "
            f"range_alignment={'enabled' if align_depth_range else 'disabled'}"
        )

        # Upload the complete raw depth sequence once.
        depth_gpu = torch.from_numpy(depths_np).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        output_gpu = torch.empty_like(depth_gpu)
        output_gpu[0] = depth_gpu[0]

        # Create quantile tensors once instead of allocating them every frame.
        alignment_quantiles = torch.tensor(
            [0.10, 0.50, 0.90],
            device=device,
            dtype=torch.float32,
        )

        range_quantiles = torch.tensor(
            [0.02, 0.98],
            device=device,
            dtype=torch.float32,
        )

        pbar = ProgressBar(frame_count - 1)

        with torch.inference_mode():
            for index in range(1, frame_count):
                current = depth_gpu[index]
                previous = output_gpu[index - 1]

                if align_depth_range and alignment_strength > 0.0:
                    current = self._align_current_to_previous(
                        current=current,
                        previous=previous,
                        alignment_strength=alignment_strength,
                        alignment_quantiles=alignment_quantiles,
                    )

                robust_range = self._depth_range(
                    depth=previous,
                    range_quantiles=range_quantiles,
                )

                residual = torch.abs(current - previous) / robust_range

                # Pixels with small depth disagreement are likely stable surfaces
                # affected by flicker and receive stronger temporal smoothing.
                #
                # Pixels with larger disagreement are likely moving boundaries,
                # occlusions or genuine geometry changes and mostly keep the
                # current prediction.
                confidence = torch.clamp(
                    1.0 - residual / max(float(depth_consistency), 1e-6),
                    min=0.0,
                    max=1.0,
                )

                previous_weight = confidence * float(temporal_strength)

                output_gpu[index] = (
                    current * (1.0 - previous_weight)
                    + previous * previous_weight
                )

                pbar.update(1)

        # Return CPU float32 numpy DEPTHS for compatibility with Depth Map Display.
        result = (
            output_gpu
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

        logger.info("GPU temporal stabilisation completed")

        return (result,)