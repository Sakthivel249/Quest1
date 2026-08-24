"""
ocr.py
------
Performs Visual Text Search using EasyOCR.
Scans frames (first coarsely, then finely) to find the exact frame 
where the target dialogue appears as text on screen.
"""

import logging
import easyocr
import numpy as np
from typing import Optional

from matcher import is_match
from sampler import get_coarse_sample_indices, get_fine_sample_indices
from video import VideoReader

logger = logging.getLogger(__name__)

class VisualSearcher:
    def __init__(self):
        """
        Initialize the EasyOCR reader. 
        Using English ('en') by default. GPU is used automatically if available.
        """
        logger.info("Initializing EasyOCR (this may take a moment on first run)...")
        # Initialize Reader (downloads model weights on first run)
        self.reader = easyocr.Reader(['en'], gpu=True)
        logger.info("EasyOCR initialized successfully.")

    def _extract_text_from_frame(self, frame: np.ndarray) -> str:
        """
        Run OCR on a single frame and return the combined text.
        """
        if frame is None:
            return ""
            
        # EasyOCR readtext returns a list of (bbox, text, confidence) tuples
        results = self.reader.readtext(frame)
        
        # Join all the detected text segments with a space
        extracted_text = " ".join([result[1] for result in results])
        return extracted_text

    def find_dialogue_frame(self, url: str, target_dialogue: str, stop_event=None) -> Optional[int]:
        """
        Find the exact frame index where the target dialogue appears.
        Uses the Coarse-to-Fine strategy.
        Returns the frame index, or None if not found.
        """
        logger.info("Starting Visual Search (OCR) on: %s", url)
        logger.info("Searching for dialogue: '%s'", target_dialogue)
        
        with VideoReader(url) as v:
            logger.info("Video loaded. Starting Phase 1: Coarse Search...")
            
            # --- PHASE 1: COARSE SEARCH ---
            # Check 1 frame every second
            coarse_indices = get_coarse_sample_indices(v.meta, interval_seconds=1.0)
            
            match_found = False
            coarse_match_ts = 0.0
            
            for idx in coarse_indices:
                if stop_event and stop_event.is_set():
                    logger.info("Visual search aborted by early exit signal.")
                    return None
                    
                frame = v.get_frame(idx)
                text = self._extract_text_from_frame(frame)
                
                if is_match(text, target_dialogue):
                    coarse_match_ts = v.meta.frame_to_ts(idx)
                    logger.info("✅ Coarse match found at %s (Frame %d). Text: '%s'", 
                                v.meta.format_ts(coarse_match_ts), idx, text)
                    match_found = True
                    break
                    
            if not match_found:
                logger.info("❌ Visual Search failed. Dialogue not found in coarse scan.")
                return None
                
            # --- PHASE 2: FINE SEARCH ---
            logger.info("Starting Phase 2: Fine Search around %s...", v.meta.format_ts(coarse_match_ts))
            # Check frames in a tight 2-second window around the coarse match (1s before, 1s after)
            fine_indices = get_fine_sample_indices(
                v.meta, 
                center_ts=coarse_match_ts, 
                window_seconds=1.0, 
                interval_seconds=0.1
            )
            
            # We want the *very first* frame where the text appears, 
            # so we check sequentially from earliest to latest.
            for idx in fine_indices:
                if stop_event and stop_event.is_set():
                    logger.info("Visual search aborted by early exit signal.")
                    return None
                    
                frame = v.get_frame(idx)
                text = self._extract_text_from_frame(frame)
                
                if is_match(text, target_dialogue):
                    logger.info("🎯 Exact Frame Found: %d (Timestamp: %s)", 
                                idx, v.meta.format_ts(v.meta.frame_to_ts(idx)))
                    return idx
                    
            # If fine search fails for some weird reason, fallback to coarse match frame
            logger.warning("Fine search didn't re-trigger match. Falling back to coarse frame.")
            return v.meta.ts_to_frame(coarse_match_ts)
