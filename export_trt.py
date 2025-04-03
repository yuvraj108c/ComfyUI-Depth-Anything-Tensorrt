import torch
import time
import argparse
from utilities import Engine


def export_trt(trt_path: str, onnx_path: str, use_fp16: bool):
    engine = Engine(trt_path)

    torch.cuda.empty_cache()

    s = time.time()
    ret = engine.build(
        onnx_path,
        use_fp16,
        enable_preview=True,
    )
    e = time.time()
    print(f"Time taken to build: {(e-s)} seconds")

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export TensorRT engine from ONNX model."
    )
    parser.add_argument(
        "--trt-path",
        type=str,
        default="./depth_anything_vitl14-fp16.engine",
        help="Path to save the TensorRT engine file.",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="./depth_anything_vitl14.onnx",
        help="Path to the ONNX model file.",
    )
    parser.add_argument(
        "--use-fp32",
        action="store_true",
        help="Use FP32 precision (default is FP16).",
    )
    args = parser.parse_args()

    export_trt(
        trt_path=args.trt_path, onnx_path=args.onnx_path, use_fp16=not args.use_fp32
    )
