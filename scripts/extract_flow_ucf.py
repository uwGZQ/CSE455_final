import cv2
import numpy as np
import os

def extract_and_save_optical_flow(frames_dir_base, output_dir_base):
    """
    Traverses given directories of video frames, extracts optical flow from grayscale images, 
    and saves the original RGB frames and optical flow horizontal and vertical components 
    into separate directories for each video.

    :param frames_dir_base: Base directory path containing folders of frames for each video.
    :param output_dir_base: Base directory path where the output directories should be saved.
    """
    for class_name in os.listdir(frames_dir_base):
        class_path = os.path.join(frames_dir_base, class_name)
        if not os.path.isdir(class_path):
            continue
        
        for video_name in os.listdir(class_path):
            video_path = os.path.join(class_path, video_name)
            if not os.path.isdir(video_path):
                continue
            
            frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
            prev_frame = None
            
            for frame_file in frame_files:
                frame_path = os.path.join(video_path, frame_file)
                
                # Load the frame in original RGB color
                frame_rgb = cv2.imread(frame_path)  # cv2.IMREAD_COLOR is default and can be omitted
                
                # Convert the frame to grayscale for optical flow calculation
                frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
                
                if frame_gray is None:
                    print(f"Failed to read frame: {frame_path}")
                    continue
                
                # Save original RGB frames to img folder
                img_dir = os.path.join(output_dir_base, class_name, video_name, "img")
                os.makedirs(img_dir, exist_ok=True)
                cv2.imwrite(os.path.join(img_dir, frame_file), frame_rgb)
                
                if prev_frame is None:
                    prev_frame = frame_gray
                    continue
                
                # Compute optical flow
                flow = cv2.calcOpticalFlowFarneback(prev_frame, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_x, flow_y = flow[..., 0], flow[..., 1]
                
                flow_x_dir = os.path.join(output_dir_base, class_name, video_name, "flow_x")
                flow_y_dir = os.path.join(output_dir_base, class_name, video_name, "flow_y")
                os.makedirs(flow_x_dir, exist_ok=True)
                os.makedirs(flow_y_dir, exist_ok=True)
                
                base_name = os.path.splitext(frame_file)[0]
                flow_x_file = os.path.join(flow_x_dir, f"{base_name}_flow_x.npy")
                flow_y_file = os.path.join(flow_y_dir, f"{base_name}_flow_y.npy")
                
                np.save(flow_x_file, flow_x)
                np.save(flow_y_file, flow_y)
                
                prev_frame = frame_gray

# Example paths, replace with your actual paths
frames_dir_base = '/data3/cse455/ucf_256x256q5_l8'
output_dir_base = '/data3/cse455/ucf_256x256q5_rgb_flow'
extract_and_save_optical_flow(frames_dir_base, output_dir_base)
