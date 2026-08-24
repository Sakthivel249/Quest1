"""
main.py
-------
Command-line interface for the Dialogue Frame Detector.
Usage:
    python src/main.py <URL> <"Target Dialogue"> [--mode auto|audio|visual]
"""

import sys
import argparse
import logging
import os
import cv2

from detector import DialogueDetector
from video import VideoReader

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Find the exact frame where dialogue occurs.")
    parser.add_argument("url", type=str, help="The URL or local path of the video.")
    parser.add_argument("dialogue", type=str, help="The target text/dialogue to find.")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["auto", "audio", "visual"], 
        default="auto",
        help="Search mode. 'auto' tries audio first, then falls back to visual OCR."
    )
    
    args = parser.parse_args()
    
    detector = DialogueDetector(mode=args.mode)
    
    # Run the detector while holding the VideoReader open
    # This prevents the temp file from being deleted before we can grab the final frame
    with VideoReader(args.url) as v:
        frame_idx = detector.find_dialogue(v, args.dialogue)
        
        if frame_idx is not None:
            # We found it! Let's extract and save the final image.
            out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
            os.makedirs(out_dir, exist_ok=True)
            
            frame = v.get_frame(frame_idx)
            if frame is not None:
                ts_str = v.meta.format_ts(v.meta.frame_to_ts(frame_idx))
                # Clean up the dialogue string for the filename
                safe_dialogue = "".join([c if c.isalnum() else "_" for c in args.dialogue])[:20]
                out_path = os.path.join(out_dir, f"match_{safe_dialogue}_{frame_idx}.jpg")
                
                cv2.imwrite(out_path, frame)
                logger.info("==================================================")
                logger.info("🎯 SUCCESS!")
                logger.info("Timestamp: %s", ts_str)
                logger.info("Frame saved to: %s", out_path)
                logger.info("==================================================")
            else:
                logger.error("Match found, but failed to extract the final frame image.")
        else:
            logger.warning("==================================================")
            logger.warning("❌ Dialogue not found in the video.")
            logger.warning("==================================================")

if __name__ == "__main__":
    main()
