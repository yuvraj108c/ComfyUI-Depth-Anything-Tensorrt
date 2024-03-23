import time
import os
import folder_paths
import torch
import torchvision
import subprocess

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

        # TODO: Use a better approach instead of:  
        # -> converting tensor images to mp4 video
        # -> load video back in opencv for inference 
        # -> convert images back to video
        # -> load final video and convert back to tensor format

        # convert images to video for processing
        os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
        temp_video_name = f"depth_{int(round(time.time() * 1000))}.mp4"
        temp_video_path = os.path.join(folder_paths.get_temp_directory(), temp_video_name)
        images = torch.clamp(images *  255,  0,  255).byte() 
        torchvision.io.write_video(temp_video_path, images, 24)

        video_save_path = os.path.join(folder_paths.get_temp_directory(),f"result_{temp_video_name}")

        # run inference
        subprocess.run(f"python {folder_paths.get_folder_paths('custom_nodes')[0]}/ComfyUI-Depth-Anything-Tensorrt/inference.py --engine {os.path.join(ENGINE_DIR,engine)} --video {temp_video_path} --save_path {video_save_path}", shell=True)
       
        # convert video_frames as tensor
        video_frames, _, _ =  torchvision.io.read_video(video_save_path,pts_unit="sec")
        video_frames = video_frames.float() /  255.0

        return (video_frames,)



NODE_CLASS_MAPPINGS = { 
    "DepthAnythingTensorrtNode" : DepthAnythingTensorrtNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
     "DepthAnythingTensorrtNode" : "Depth Anything Tensorrt",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']