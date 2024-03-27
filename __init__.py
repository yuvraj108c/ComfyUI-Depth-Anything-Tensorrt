import os
import folder_paths

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

import torch.nn.functional as F
import torch
import comfy
import cv2
import torchvision.transforms as transforms

ENGINE_DIR = os.path.join(folder_paths.models_dir,"depth_trt_engines")

class DepthAnythingTensorrtNode:
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
    CATEGORY = "Depth Anything Tensorrt"

    def main(self, images, engine):
        import pycuda.autoinit

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        with open(os.path.join(ENGINE_DIR,engine), "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        with engine.create_execution_context() as context:
            input_shape = context.get_tensor_shape('input')
            output_shape = context.get_tensor_shape('output')
            h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
            h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
            d_input = cuda.mem_alloc(h_input.nbytes)
            d_output = cuda.mem_alloc(h_output.nbytes)
            stream = cuda.Stream()

            pbar = comfy.utils.ProgressBar(images.shape[0])
            images = images.permute(0, 3, 1, 2)

            images_resized = F.interpolate(images, size=input_shape[2:], mode='bilinear', align_corners=False)

            images_list = list(torch.split(images_resized, split_size_or_sections=1))

            depth_frames = []

            for img in images_list:
                np.copyto(h_input, img.ravel())
            
                cuda.memcpy_htod_async(d_input, h_input, stream)
                context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
                cuda.memcpy_dtoh_async(h_output, d_output, stream)
                stream.synchronize()
                depth = h_output
            
                # Process the depth output
                depth = np.reshape(depth, output_shape[2:])
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.astype(np.uint8)
                depth = cv2.resize(depth, (images.shape[2], images.shape[3]))
                depth = cv2.cvtColor(depth, cv2.COLOR_RGB2BGR)
                transform = transforms.Compose([transforms.ToTensor()])
                depth = transform(depth).unsqueeze(0).permute(0, 2, 3, 1)
                depth_frames.append(depth)
                pbar.update(1)

        result = torch.cat(depth_frames,dim=0)
        return (result,)


NODE_CLASS_MAPPINGS = { 
    "DepthAnythingTensorrtNode" : DepthAnythingTensorrtNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
     "DepthAnythingTensorrtNode" : "Depth Anything Tensorrt",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']