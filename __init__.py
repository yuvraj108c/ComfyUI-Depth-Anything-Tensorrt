from .nodes.depth_anything_tensorrt import DepthAnythingTensorrt, DepthAnythingTensorrtAdvanced
from .nodes.depth_map_display import DepthMapDisplay
from .nodes.engine_builder import DepthAnythingEngineBuilder

NODE_CLASS_MAPPINGS = {
    "DepthAnythingTensorrt" : DepthAnythingTensorrt,
    "DepthAnythingEngineBuilder" : DepthAnythingEngineBuilder,
    "DepthAnythingTensorrtAdvanced" : DepthAnythingTensorrtAdvanced,
    "DepthAnythingMapDisplay" : DepthMapDisplay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
     "DepthAnythingTensorrt" : "Depth Anything Tensorrt ⚡",
     "DepthAnythingEngineBuilder" : "Depth Anything Tensorrt Engine Builder ⚡",
     "DepthAnythingTensorrtAdvanced" : "Depth Anything Tensorrt Advanced ⚡",
     "DepthAnythingMapDisplay" : "Depth Map Display ⚡",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']