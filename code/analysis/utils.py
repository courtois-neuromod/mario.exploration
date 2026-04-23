import os
import numpy as np

# Excel serial date map for scene IDs
EXCEL_SERIAL_MAP = {
    '39173': '4-1-7',
    '39173.0': '4-1-7',
    39173: '4-1-7',
    39173.0: '4-1-7'
}

def get_clip_duration(clip_path, cleared):
    """
    Attempts to get clip duration from file.
    If fails, returns None.
    """
    # Import locally to avoid hard dependency at module level if not needed immediately
    from moviepy import VideoFileClip
    import cv2 
    try:
        cap = cv2.VideoCapture(clip_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = total_frames/fps
    except Exception as e:
        print('problem with cv2:', e)
        pass
    try:
        # Check if file exists to avoid moviepy error spam
        if not os.path.exists(clip_path):
            print('Pas le bon path')
            return None
            
        with VideoFileClip(clip_path) as clip:
            if cleared:
                return duration
            else:
                return duration-2
    except Exception:
        return None

def calculate_dice(v1, v2):
    """Calculates Dice coefficient between two boolean vectors."""
    intersection = (v1 & v2).sum()
    denom = v1.sum() + v2.sum()
    if denom == 0:
        return 1.0 if len(v1) > 0 else 0.0
    return 2 * intersection / denom
