import cv2
import os

from .vis import image2video


# Convert images to video
def make_video(root_path, ini_frame_id, max_frame_id, step_frames):
    # animate the simulation
    img_0_path = os.path.join(root_path, f'frame_{ini_frame_id+step_frames+1}.jpg')
    img_0 = cv2.imread(img_0_path)
    h, w, _ = img_0.shape
    # Initialize the video writer
    video_path = os.path.join(root_path, 'simulation.mp4')
    I2V = image2video(w, h)
    I2V.start(video_path, fps=int(25/step_frames+1.e-6))
    
    # Write the rest frames
    for frame_id in range(ini_frame_id + step_frames + 1, max_frame_id + 1, step_frames):
        img_path = os.path.join(root_path, f'frame_{frame_id}.jpg')
        img = cv2.imread(img_path)
        if img is None:
            continue
        I2V.record(img)
    
    # Release the video writer
    I2V.end()
    
    print('='*50)
    print(f'Video saved to {video_path}')
    print('='*50)
