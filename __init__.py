from .nodes.depth_anything_tensorrt import DepthAnythingTensorrt, DepthAnythingTensorrtAdvanced
from .nodes.depth_map_display import DepthMapDisplay
from .nodes.engine_builder import DepthAnythingEngineBuilder
from .nodes.depth_temporal_stabilizer_gpu import DepthTemporalStabilizerGPU

NODE_CLASS_MAPPINGS = {
    "DepthAnythingTensorrt" : DepthAnythingTensorrt,
    "DepthAnythingEngineBuilder" : DepthAnythingEngineBuilder,
    "DepthAnythingTensorrtAdvanced" : DepthAnythingTensorrtAdvanced,
    "DepthAnythingMapDisplay" : DepthMapDisplay,
    "DepthTemporalStabilizerGPU" : DepthTemporalStabilizerGPU,
}

NODE_DISPLAY_NAME_MAPPINGS = {
     "DepthAnythingTensorrt" : "Depth Anything Tensorrt ⚡",
     "DepthAnythingEngineBuilder" : "Depth Anything Tensorrt Engine Builder ⚡",
     "DepthAnythingTensorrtAdvanced" : "Depth Anything Tensorrt Advanced ⚡",
     "DepthAnythingMapDisplay" : "Depth Map Display ⚡",
     "DepthTemporalStabilizerGPU" : "Depth Temporal Stabilizer GPU ⚡",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']