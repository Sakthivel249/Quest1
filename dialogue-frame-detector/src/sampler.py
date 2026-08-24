"""
sampler.py
----------
Responsible for selecting which frames to process. 
Instead of checking every single frame (which is too slow), we use a two-stage approach:

1. Coarse Sampling: Check 1 frame every second to quickly find roughly where the dialogue happens.
2. Fine Sampling: Once we find a match, check every frame (or every 0.1s) in a tight window around the match to find the exact start frame.
"""

from typing import List
from video import VideoMeta

def get_coarse_sample_indices(meta: VideoMeta, interval_seconds: float = 1.0) -> List[int]:
    """
    Generate a list of frame indices spaced by `interval_seconds`.
    Used to quickly scan the entire video.
    """
    if meta.duration_seconds <= 0 or meta.fps <= 0:
        return []
        
    indices = []
    current_ts = 0.0
    
    while current_ts <= meta.duration_seconds:
        frame_idx = meta.ts_to_frame(current_ts)
        # Ensure we don't go out of bounds
        if frame_idx >= meta.frame_count:
            break
        indices.append(frame_idx)
        current_ts += interval_seconds
        
    return indices

def get_fine_sample_indices(meta: VideoMeta, center_ts: float, window_seconds: float = 1.5, interval_seconds: float = 0.1) -> List[int]:
    """
    Generate a dense list of frame indices around a specific timestamp.
    If `center_ts` is where we found the text roughly, we scan `window_seconds` 
    before and after it at a much higher frequency (e.g. 10 frames a second).
    """
    if meta.duration_seconds <= 0 or meta.fps <= 0:
        return []
        
    start_ts = max(0.0, center_ts - window_seconds)
    end_ts = min(meta.duration_seconds, center_ts + window_seconds)
    
    indices = []
    current_ts = start_ts
    
    # We use a set to avoid adding the same frame index multiple times 
    # if interval_seconds is very small compared to FPS
    seen = set()
    
    while current_ts <= end_ts:
        frame_idx = meta.ts_to_frame(current_ts)
        
        if frame_idx < meta.frame_count and frame_idx not in seen:
            indices.append(frame_idx)
            seen.add(frame_idx)
            
        current_ts += interval_seconds
        
    return sorted(indices)

def get_exact_frame_range(meta: VideoMeta, start_ts: float, end_ts: float) -> List[int]:
    """
    Generate every single frame index between two timestamps.
    Used for maximum precision when we narrow down the window.
    """
    start_frame = meta.ts_to_frame(max(0.0, start_ts))
    end_frame = meta.ts_to_frame(min(meta.duration_seconds, end_ts))
    
    # Ensure bounds
    start_frame = max(0, min(start_frame, meta.frame_count - 1))
    end_frame = max(0, min(end_frame, meta.frame_count - 1))
    
    if start_frame > end_frame:
        return []
        
    return list(range(start_frame, end_frame + 1))
