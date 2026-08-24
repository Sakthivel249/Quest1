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
    def __init__(self, model_name: str = "base"):
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

    def find_dialogue_timestamp(self, video_path: str, target_dialogue: str) -> Optional[float]:
        """
        Transcribe the audio from the video file and search for the target dialogue.
        Returns the start timestamp (in seconds) of the segment containing the dialogue,
        or None if not found.
        """
        logger.info("Starting Audio Search (Whisper) on: %s", video_path)
        logger.info("Searching for dialogue: '%s'", target_dialogue)
        
        try:
            # Whisper can read audio directly from the video file via ffmpeg internally
            logger.info("Transcribing audio (this may take a moment)...")
            result = self.model.transcribe(video_path)
        except Exception as e:
            logger.error("Whisper transcription failed: %s", e)
            return None

        segments = result.get("segments", [])
        logger.info("Transcription complete. Searching through %d speech segments.", len(segments))

        for segment in segments:
            text = segment.get("text", "")
            start_ts = segment.get("start", 0.0)
            end_ts = segment.get("end", 0.0)
            
            # Check if this spoken segment matches our target dialogue
            if is_match(text, target_dialogue):
                logger.info("✅ Audio Match Found!")
                logger.info("   Spoken Text: '%s'", text.strip())
                logger.info("   Timestamp: %.2fs -> %.2fs", start_ts, end_ts)
                return start_ts
                
        logger.info("❌ Audio Search failed to find the dialogue.")
        return None
