"""
audio_search.py
---------------
Performs Audio Search using OpenAI's Whisper model.
Extracts speech from the video and finds the exact timestamp where the dialogue occurs.
"""

import logging
import whisper
import warnings
from typing import Optional

from matcher import is_match

logger = logging.getLogger(__name__)

# Suppress some noisy whisper warnings
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

class AudioSearcher:
    def __init__(self, model_name: str = "tiny"):
        """
        """
        import os
        # Ensure ffmpeg is in PATH for Whisper (handles winget installations before terminal restart)
        winget_ffmpeg = r"C:\Users\sakth\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0-full_build\bin"
        if os.path.isdir(winget_ffmpeg) and winget_ffmpeg not in os.environ.get("PATH", ""):
            os.environ["PATH"] += os.pathsep + winget_ffmpeg
            
        logger.info("Loading Whisper ASR model '%s'...", model_name)
        # Whisper automatically downloads the model weights if not cached
        self.model = whisper.load_model(model_name)
        logger.info("Whisper model loaded successfully.")

    def find_dialogue_timestamp(self, video_path: str, target_dialogue: str, stop_event=None) -> Optional[float]:
        """
        Transcribe the audio from the video file and search for the target dialogue.
        Returns the start timestamp (in seconds) of the segment containing the dialogue,
        or None if not found.
        """
        logger.info("Starting Audio Search (Whisper) on: %s", video_path)
        logger.info("Searching for dialogue: '%s'", target_dialogue)
        
        if stop_event and stop_event.is_set():
            return None

        logger.info("Transcribing audio (this may take a moment)...")
        # Whisper automatically extracts the audio from the video file
        try:
            result = self.model.transcribe(video_path, word_timestamps=True)
        except Exception as e:
            logger.error("Whisper transcription failed: %s", e)
            return None
            
        if stop_event and stop_event.is_set():
            return None

        segments = result.get("segments", [])
        logger.info("Transcription complete. Searching through %d speech segments.", len(segments))

        for segment in segments:
            if stop_event and stop_event.is_set():
                logger.info("Audio search aborted: match found in another thread.")
                return None
                
            text = segment.get("text", "")
            
            if is_match(text, target_dialogue):
                logger.info("✅ Audio Match Found in chunk: '%s'", text.strip())
                
                # Attempt to find the exact word timestamp using a sliding window
                exact_start = segment['start']
                if 'words' in segment and target_dialogue:
                    target_words = target_dialogue.split()
                    target_len = len(target_words)
                    words_list = segment['words']
                    
                    best_score = 0
                    best_start = exact_start
                    
                    # We check every possible starting word in the segment
                    for i in range(len(words_list)):
                        # Grab a window of words roughly the same length as the target
                        window = words_list[i:i+target_len+1] # +1 for safety against split errors
                        window_text = " ".join([w.get('word', '').strip().lower() for w in window])
                        
                        # Use standard ratio (not partial) since lengths are similar
                        import rapidfuzz.fuzz as fuzz
                        score = fuzz.ratio(window_text, target_dialogue.lower())
                        
                        if score > best_score:
                            best_score = score
                            best_start = words_list[i]['start']
                            
                    exact_start = best_start
                            
                logger.info("   Exact Timestamp: %.2fs", exact_start)
                return exact_start

        logger.info("❌ Audio Search failed to find the dialogue.")
        return None
