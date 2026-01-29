from .nodes.depth_anything_tensorrt import DepthAnythingTensorrt
from .nodes.engine_builder import DepthAnythingEngineBuilder

NODE_CLASS_MAPPINGS = { 
    "DepthAnythingTensorrt" : DepthAnythingTensorrt,
    "DepthAnythingEngineBuilder" : DepthAnythingEngineBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
     "DepthAnythingTensorrt" : "Depth Anything Tensorrt ⚡",
     "DepthAnythingEngineBuilder" : "Depth Anything Engine Builder ⚡",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']