import argparse
import os
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from utils import *
from tqdm import tqdm

transform = Compose(
    [
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=False,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
)

def run(args):
    # Create logger and load the TensorRT engine
    logger = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    with engine.create_execution_context() as context:
        input_shape = context.get_tensor_shape('input')
        output_shape = context.get_tensor_shape('output')
        h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
        h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        stream = cuda.Stream()

        cap = cv2.VideoCapture(args.video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=frame_count)
        depth_frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            image = transform({"image": image})["image"]  # C, H, W
            image = image[None]  # B, C, H, W
        
            # Copy the input image to the pagelocked memory
            np.copyto(h_input, image.ravel())
            
            # Copy the input to the GPU, execute the inference, and copy the output back to the CPU
            cuda.memcpy_htod_async(d_input, h_input, stream)
            context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()
            depth = h_output
        
            # Process the depth output
            depth = np.reshape(depth, output_shape[2:])
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
            depth = cv2.cvtColor(depth, cv2.COLOR_RGB2BGR)

            depth_frames.append(depth)
            pbar.update(1)
            
        pbar.close()
        cap.release()

        # save video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
        frame_size = (depth_frames[0].shape[1], depth_frames[0].shape[0]) # Frame size (width, height)
        video_writer = cv2.VideoWriter(args.save_path, fourcc, 24, frame_size)
        for frame in depth_frames:
            video_writer.write(frame)
        video_writer.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run depth estimation with a TensorRT engine.')
    parser.add_argument('--video', type=str, required=True, help='Path to the video')
    parser.add_argument('--save_path', type=str, required=True ,help='Output path for depth video')
    parser.add_argument('--engine', type=str, required=True, help='Path to the TensorRT engine')
    
    args = parser.parse_args()
    run(args)
