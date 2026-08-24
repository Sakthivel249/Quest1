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
        import threading
        import concurrent.futures

        logger.info("=== Starting Dialogue Detection ===")
        logger.info("Target: '%s'", target_dialogue)
        logger.info("Mode: %s", self.mode)
        
        local_path = v._temp_path if v._temp_path else v._url
        meta = v.meta
        
        stop_event = threading.Event()
        result_frame_idx = None
        
        def run_audio():
            audio_ts = self._get_audio_searcher().find_dialogue_timestamp(local_path, target_dialogue, stop_event)
            if audio_ts is not None:
                frame_idx = meta.ts_to_frame(audio_ts)
                logger.info("🎉 Audio Search Succeeded. Frame index: %d", frame_idx)
                return frame_idx
            return None

        def run_visual():
            frame_idx = self._get_visual_searcher().find_dialogue_frame(local_path, target_dialogue, stop_event)
            if frame_idx is not None:
                logger.info("🎉 Visual Search Succeeded. Frame index: %d", frame_idx)
                return frame_idx
            return None

        if self.mode == "auto":
            # Run both in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both tasks
                future_audio = executor.submit(run_audio)
                future_visual = executor.submit(run_visual)
                
                # Wait for whichever finishes first with a non-None result
                for future in concurrent.futures.as_completed([future_audio, future_visual]):
                    res = future.result()
                    if res is not None:
                        result_frame_idx = res
                        stop_event.set() # Stop the other thread
                        break
                        
                # If neither succeeded
                if result_frame_idx is None:
                    logger.error("❌ Both Audio and Visual searches failed.")
                    
        elif self.mode == "audio":
            result_frame_idx = run_audio()
            if result_frame_idx is None:
                logger.error("❌ Audio search failed.")
                
        elif self.mode == "visual":
            result_frame_idx = run_visual()
            if result_frame_idx is None:
                logger.error("❌ Visual search failed.")

        return result_frame_idx
