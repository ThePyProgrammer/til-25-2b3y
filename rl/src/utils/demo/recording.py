import os
import numpy as np
import cv2
from moviepy.editor import ImageSequenceClip


def create_output_dirs():
    """Create output directories for recording demo.
    
    Returns:
        Dictionary with paths to various output directories
    """
    dirs = {
        'base': '_output',
        'frames': '_output/frames',
        'frames_oracle': '_output/frames/oracle',
        'frames_map': '_output/frames/map',
        'frames_proba': '_output/frames/proba',
        'frames_combined': '_output/frames/combined',
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def create_video(frames, output_path, fps=5):
    """Create a video from a list of frames.
    
    Args:
        frames: List of frames (numpy arrays)
        output_path: Path where the video will be saved
        fps: Frames per second for the output video
        
    Returns:
        None
    """
    if not frames:
        print(f"No frames to create video at {output_path}")
        return

    height, width = frames[0].shape[:2]

    # Ensure all frames have the same size
    processed_frames = []
    for frame in frames:
        if frame is not None:
            # If frame size doesn't match the expected size, resize it
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            processed_frames.append(frame)

    if not processed_frames:
        print(f"No valid frames to create video at {output_path}")
        return

    # Make sure output path has .mp4 extension
    if not output_path.lower().endswith('.mp4'):
        output_path = f"{output_path}.mp4"

    # Convert BGR (OpenCV) to RGB (MoviePy) if needed
    rgb_frames = []
    for frame in processed_frames:
        if frame is not None:
            # Assume BGR format from OpenCV, convert to RGB for MoviePy
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frames.append(rgb_frame)

    clip = ImageSequenceClip(rgb_frames, fps=fps)

    # Write video file
    clip.write_videofile(output_path, codec="libx264", fps=fps)
    print(f"Video saved to {output_path} using MoviePy")


def save_frame(frame, directory, step, name='frame'):
    """Save a single frame to a file.
    
    Args:
        frame: The frame to save (numpy array)
        directory: Directory where the frame will be saved
        step: Current step number (for filename)
        name: Base name for the frame file
        
    Returns:
        None
    """
    if frame is None:
        return

    # Make sure the frame is in BGR format for OpenCV
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        # This is a simplistic heuristic - may need refinement
        frame_to_save = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame_to_save = frame

    filename = f"{name}_{step:04d}.png"
    filepath = os.path.join(directory, filename)
    cv2.imwrite(filepath, frame_to_save)
    
    return filepath