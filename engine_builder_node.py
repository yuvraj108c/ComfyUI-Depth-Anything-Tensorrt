import os
import folder_paths
from .export_trt import export_trt
import urllib.request
import shutil

ENGINE_DIR = os.path.join(folder_paths.models_dir,"tensorrt", "depth-anything")
ONNX_DIR = os.path.join(folder_paths.models_dir,"onnx", "depth-anything")

# Model URLs and configurations
DEPTH_ANYTHING_MODELS = {
    "v1_small": {
        "url": "https://huggingface.co/yuvraj108c/Depth-Anything-Onnx/resolve/main/depth_anything_vits14.onnx",
        "filename": "depth_anything_vits14.onnx",
        "engine_name": "v1_depth_anything_vits14-fp16.engine"
    },
    "v1_base": {
        "url": "https://huggingface.co/yuvraj108c/Depth-Anything-Onnx/resolve/main/depth_anything_vitb14.onnx",
        "filename": "depth_anything_vitb14.onnx",
        "engine_name": "v1_depth_anything_vitb14-fp16.engine"
    },
    "v1_large": {
        "url": "https://huggingface.co/yuvraj108c/Depth-Anything-Onnx/resolve/main/depth_anything_vitl14.onnx",
        "filename": "depth_anything_vitl14.onnx",
        "engine_name": "v1_depth_anything_vitl14-fp16.engine"
    },
    "v2_small": {
        "url": "https://huggingface.co/yuvraj108c/Depth-Anything-2-Onnx/resolve/main/depth_anything_v2_vits.onnx",
        "filename": "depth_anything_v2_vits.onnx",
        "engine_name": "v2_depth_anything_vits-fp16.engine"
    },
    "v2_base": {
        "url": "https://huggingface.co/yuvraj108c/Depth-Anything-2-Onnx/resolve/main/depth_anything_v2_vitb.onnx",
        "filename": "depth_anything_v2_vitb.onnx",
        "engine_name": "v2_depth_anything_vitb-fp16.engine"
    },
    "v2_large": {
        "url": "https://huggingface.co/yuvraj108c/Depth-Anything-2-Onnx/resolve/main/depth_anything_v2_vitl.onnx",
        "filename": "depth_anything_v2_vitl.onnx",
        "engine_name": "v2_depth_anything_vitl-fp16.engine"
    },
    "v2_metric_hypersim_large": {
        "url": "https://huggingface.co/yuvraj108c/Depth-Anything-2-Onnx/resolve/main/depth_anything_v2_metric_hypersim_vitl.onnx",
        "filename": "depth_anything_v2_metric_hypersim_vitl.onnx",
        "engine_name": "v2_depth_anything_metric_hypersim_vitl-fp16.engine"
    },
    "v2_metric_vkitti_large": {
        "url": "https://huggingface.co/yuvraj108c/Depth-Anything-2-Onnx/resolve/main/depth_anything_v2_metric_vkitti_vitl.onnx",
        "filename": "depth_anything_v2_metric_vkitti_vitl.onnx",
        "engine_name": "v2_depth_anything_metric_vkitti_vitl-fp16.engine"
    },
    "DAD_small": {
        "url": "https://huggingface.co/yuvraj108c/distill-any-depth-onnx/resolve/main/Distill-Any-Depth-Multi-Teacher-Small.onnx",
        "filename": "Distill-Any-Depth-Multi-Teacher-Small.onnx",
        "engine_name": "Distill-Any-Depth-Multi-Teacher-Small-fp16.engine"
    },
    "DAD_base": {
        "url": "https://huggingface.co/yuvraj108c/distill-any-depth-onnx/resolve/main/Distill-Any-Depth-Multi-Teacher-Base.onnx",
        "filename": "Distill-Any-Depth-Multi-Teacher-Base.onnx",
        "engine_name": "Distill-Any-Depth-Multi-Teacher-Base-fp16.engine"
    },
    "DAD_large": {
        "url": "https://huggingface.co/yuvraj108c/distill-any-depth-onnx/resolve/main/Distill-Any-Depth-Multi-Teacher-Large.onnx",
        "filename": "Distill-Any-Depth-Multi-Teacher-Large.onnx",
        "engine_name": "Distill-Any-Depth-Multi-Teacher-Large-fp16.engine"
    },
    "DAD_large_2w_iter": {
        "url": "https://huggingface.co/yuvraj108c/distill-any-depth-onnx/resolve/main/Distill-Any-Depth-Dav2-Teacher-Large-2w-iter.onnx",
        "filename": "Distill-Any-Depth-Dav2-Teacher-Large-2w-iter.onnx",
        "engine_name": "Distill-Any-Depth-Dav2-Teacher-Large-2w-iter-fp16.engine"
    },
    "DA3Mono-Large": {
        "url": "https://huggingface.co/yuvraj108c/Depth-Anything-3-Onnx/resolve/main/DA3Mono-Large.onnx",
        "filename": "DA3Mono-Large.onnx",
        "engine_name": "DA3Mono-Large-fp16.engine"
    },
    "DA3Metric-Large": {
        "url": "https://huggingface.co/yuvraj108c/Depth-Anything-3-Onnx/resolve/main/DA3Metric-Large.onnx",
        "filename": "DA3Metric-Large.onnx",
        "engine_name": "DA3Metric-Large-fp16.engine"
    }
}

def download_onnx_model(model_key):
    """Download ONNX model if it doesn't exist"""
    if model_key not in DEPTH_ANYTHING_MODELS:
        return f"Invalid model key: {model_key}", None
    
    model_info = DEPTH_ANYTHING_MODELS[model_key]
    url = model_info["url"]
    filename = model_info["filename"]
    local_path = os.path.join(ONNX_DIR, filename)
    
    if os.path.exists(local_path):
        return f"Model already exists at: {local_path}", local_path
    
    try:
        print(f"Downloading model from {url} to {local_path}...")
        
        # Create a temporary file for downloading
        temp_file = local_path + ".tmp"
        
        # Download the file
        with urllib.request.urlopen(url) as response, open(temp_file, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        
        # Rename the temporary file to the target filename
        shutil.move(temp_file, local_path)
        
        return f"Successfully downloaded model to: {local_path}", local_path
    except Exception as e:
        return f"Error downloading model: {str(e)}", None

class DepthAnythingEngineBuilder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model_version": (list(DEPTH_ANYTHING_MODELS.keys()), {
                    "default": "v1_large",
                    "tooltip": "Select the Depth Anything model version and size. Larger models provide better quality but require more VRAM."
                }),
                "custom_engine_name": ("STRING", {
                    "default": "",
                    "tooltip": "Optional custom name for the TensorRT engine file. If empty, will use the default name based on the model."
                }),
                "use_fp16": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable FP16 precision for faster inference and lower VRAM usage. Disable if you experience stability issues."
                }),
                "custom_onnx_path": ("STRING", {
                    "default": "",
                    "optional": True,
                    "tooltip": "Optional path to a custom ONNX model file. If provided, will use this instead of downloading the predefined model."
                }),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("message",)
    FUNCTION = "build_engine"
    CATEGORY = "tensorrt"
    OUTPUT_NODE = True
    
    def build_engine(self, model_version, custom_engine_name, use_fp16, custom_onnx_path=""):
        # Ensure directories exist
        os.makedirs(ENGINE_DIR, exist_ok=True)
        os.makedirs(ONNX_DIR, exist_ok=True)
        
        # Determine ONNX path and engine name
        if custom_onnx_path and os.path.exists(custom_onnx_path):
            onnx_path = custom_onnx_path
            version = "v1" if "v1" in os.path.basename(custom_onnx_path) else "v2"
            engine_name = custom_engine_name if custom_engine_name else f"{version}_{os.path.basename(custom_onnx_path).replace('.onnx', '-fp16.engine' if use_fp16 else '.engine')}"
        else:
            # Download and use predefined model
            model_info = DEPTH_ANYTHING_MODELS[model_version]
            message, onnx_path = download_onnx_model(model_version)
            
            if not onnx_path:
                return (message,)
                
            engine_name = custom_engine_name if custom_engine_name else model_info["engine_name"]
            if use_fp16 and not engine_name.endswith("-fp16.engine"):
                engine_name = engine_name.replace(".engine", "-fp16.engine")
            elif not use_fp16 and "-fp16.engine" in engine_name:
                engine_name = engine_name.replace("-fp16.engine", ".engine")
        
        engine_path = os.path.join(ENGINE_DIR, engine_name)
        
        # Check if engine already exists
        if os.path.exists(engine_path):
            return (f"Engine already exists at: {engine_path}. Delete it manually if you want to rebuild.",)
        
        try:
            print(f"Building TensorRT engine from {onnx_path} to {engine_path}...")
            result = export_trt(trt_path=engine_path, onnx_path=onnx_path, use_fp16=use_fp16)
            if result == 0:
                return (f"Successfully built engine: {engine_path}",)
            else:
                return (f"Failed to build engine. Check console for details.",)
        except Exception as e:
            return (f"Error building engine: {str(e)}",)
