"""
detector.py
-----------
Orchestrates the search process. 
Determines whether to use Audio Search, Visual Search, or both based on user input.
"""

import logging
from typing import Optional, Tuple

from video import VideoReader
from audio_search import AudioSearcher
from ocr import VisualSearcher

logger = logging.getLogger(__name__)

class DialogueDetector:
    def __init__(self, mode: str = "auto"):
        """
        mode: 'audio', 'visual', or 'auto'
        """
        self.mode = mode.lower()
        if self.mode not in ("audio", "visual", "auto"):
            logger.warning("Unknown mode '%s'. Defaulting to 'auto'.", self.mode)
            self.mode = "auto"
            
        self.audio_searcher = None
        self.visual_searcher = None

    def _get_audio_searcher(self) -> AudioSearcher:
        if self.audio_searcher is None:
            self.audio_searcher = AudioSearcher()
        return self.audio_searcher

    def _get_visual_searcher(self) -> VisualSearcher:
        if self.visual_searcher is None:
            self.visual_searcher = VisualSearcher()
        return self.visual_searcher

    def find_dialogue(self, v: VideoReader, target_dialogue: str) -> Optional[int]:
        """
        Finds the dialogue based on the selected mode.
        Returns the frame_index.
        """
        logger.info("=== Starting Dialogue Detection ===")
        logger.info("Target: '%s'", target_dialogue)
        logger.info("Mode: %s", self.mode)
        
        local_path = v._temp_path if v._temp_path else v._url
        meta = v.meta
        
        # --- AUDIO SEARCH ---
        if self.mode in ("audio", "auto"):
            audio_ts = self._get_audio_searcher().find_dialogue_timestamp(local_path, target_dialogue)
            if audio_ts is not None:
                # Found it via audio! Convert timestamp to frame index
                frame_idx = meta.ts_to_frame(audio_ts)
                logger.info("🎉 Audio Search Succeeded. Frame index: %d", frame_idx)
                return frame_idx
            else:
                if self.mode == "auto":
                    logger.info("Audio search didn't find it. Falling back to Visual search...")
                else:
                    logger.error("Audio search failed.")
                    return None

        # --- VISUAL SEARCH ---
        if self.mode in ("visual", "auto"):
            # We can use the already downloaded local_path to save time
            frame_idx = self._get_visual_searcher().find_dialogue_frame(local_path, target_dialogue)
            if frame_idx is not None:
                logger.info("🎉 Visual Search Succeeded. Frame index: %d", frame_idx)
                return frame_idx
            else:
                logger.error("Visual search failed.")
                return None
                
        return None
