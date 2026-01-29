import os
import folder_paths
from ..scripts.export_trt import export_trt
import json
from ..utilities import ColoredLogger
from huggingface_hub import hf_hub_download

logger = ColoredLogger('ComfyUI-DepthAnythingTensorrt')

ENGINE_DIR = os.path.join(folder_paths.models_dir,"tensorrt", "depth-anything")
ONNX_DIR = os.path.join(folder_paths.models_dir,"onnx", "depth-anything")
os.makedirs(ENGINE_DIR, exist_ok=True)
os.makedirs(ONNX_DIR, exist_ok=True)

# Default fallback models (minimal set)
DEFAULT_MODELS = {
    "v1_large": {
        "url": "https://huggingface.co/yuvraj108c/Depth-Anything-Onnx/resolve/main/depth_anything_vitl14.onnx",
        "filename": "depth_anything_vitl14.onnx",
        "engine_name": "v1_depth_anything_vitl14-fp16.engine"
    }
}

def load_models_config():
    """Load model configurations from JSON file with fallback to defaults"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "models.json")

    try:
        with open(config_path, 'r') as f:
            models = json.load(f)
            # logger.info(f"Loaded {len(models)} models from config/models.json")
            return models
    except FileNotFoundError:
        logger.warning("models.json not found, using default models")
        return DEFAULT_MODELS
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in models.json: {e}")
        logger.warning("Using default models")
        return DEFAULT_MODELS
    except Exception as e:
        logger.error(f"Error loading models.json: {e}")
        logger.warning("Using default models")
        return DEFAULT_MODELS

# Load models from JSON or use defaults
DEPTH_ANYTHING_MODELS = load_models_config()

def download_onnx_model(model_key):
    """Download ONNX model using Hugging Face Hub"""
    if model_key not in DEPTH_ANYTHING_MODELS:
        logger.error(f"Invalid model key: {model_key}")
        return f"Invalid model key: {model_key}", None

    model_info = DEPTH_ANYTHING_MODELS[model_key]
    url = model_info["url"]
    filename = model_info["filename"]
    local_path = os.path.join(ONNX_DIR, filename)

    if os.path.exists(local_path):
        logger.info(f"Onnx model already exists at: {local_path}")
        return f"Onnx model already exists at: {local_path}", local_path

    try:
        logger.info(f"Downloading {filename}")

        # Parse repo_id from URL (format: https://huggingface.co/{repo_id}/resolve/main/{filename})
        repo_id = url.split("huggingface.co/")[1].split("/resolve/")[0]

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=ONNX_DIR,
            local_dir=ONNX_DIR,
        )

        logger.info(f"Successfully downloaded onnx model to: {downloaded_path}")
        return f"Successfully downloaded onnx model to: {downloaded_path}", downloaded_path

    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
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
        # Determine ONNX path and engine name
        if custom_onnx_path and os.path.exists(custom_onnx_path):
            onnx_path = custom_onnx_path
            basename = os.path.basename(custom_onnx_path)
            # Improved version detection
            if "DA3" in basename or "da3" in basename:
                version_prefix = "DA3"
            elif "DAD" in basename or "distill" in basename.lower():
                version_prefix = "DAD"
            elif "v2" in basename:
                version_prefix = "v2"
            else:
                version_prefix = "v1"

            engine_name = custom_engine_name if custom_engine_name else f"{version_prefix}_{basename.replace('.onnx', '-fp16.engine' if use_fp16 else '.engine')}"
            logger.info(f"Using custom onnx model: {onnx_path}")
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
            logger.warning(f"TensorRT engine already exists at: {engine_path}")
            return (f"TensorRT engine already exists at: {engine_path}. Delete it manually if you want to rebuild.",)

        try:
            logger.info(f"Building TensorRT engine: {engine_name}")
            logger.info(f"Source onnx: {onnx_path}")
            logger.info(f"Precision: {'FP16' if use_fp16 else 'FP32'}")

            result = export_trt(trt_path=engine_path, onnx_path=onnx_path, use_fp16=use_fp16)

            if result == 0:
                logger.info(f"Successfully built engine: {engine_path}")
                return (f"Successfully built tensorrt engine: {engine_path}",)
            else:
                logger.error("Failed to build engine. Check console for details.")
                return (f"Failed to build engine. Check console for details.",)
        except Exception as e:
            logger.error(f"Error building engine: {str(e)}")
            return (f"Error building engine: {str(e)}",)
