"""
video.py
--------
Universal video loader. Given ANY URL or local path:
  - Local file  -> open directly with OpenCV
  - Any URL     -> download with yt-dlp -> read locally -> auto-delete after

Downloading is always more reliable than streaming:
  - No HLS multi-host CDN issues
  - No IP-locked token expiry
  - No OpenCV URL parsing bugs
  - Consistent frame seeking accuracy

Usage:
    with VideoReader("https://youtube.com/watch?v=...") as v:
        print(v.meta)
        frame = v.get_frame(500)       # numpy array (H, W, 3)
        frame = v.get_frame_at_ts(10)  # frame at 10 seconds
"""

import glob
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import yt_dlp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Video metadata container
# ---------------------------------------------------------------------------

@dataclass
class VideoMeta:
    """All essential properties of a video stream."""
    fps: float
    frame_count: int
    duration_seconds: float
    width: int
    height: int

    def frame_to_ts(self, frame_index: int) -> float:
        """Frame index -> seconds."""
        return frame_index / self.fps if self.fps > 0 else 0.0

    def ts_to_frame(self, seconds: float) -> int:
        """Seconds -> nearest frame index."""
        return max(0, int(seconds * self.fps))

    def format_ts(self, seconds: float) -> str:
        """Seconds -> HH:MM:SS.mmm string."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    def __str__(self) -> str:
        return (
            f"FPS={self.fps:.2f} | Frames={self.frame_count} | "
            f"Duration={self.format_ts(self.duration_seconds)} | "
            f"Resolution={self.width}x{self.height}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _download_video(url: str) -> str:
    """
    Download the video at url to a temporary directory using yt-dlp.
    Returns the local file path of the downloaded video.
    Raises RuntimeError if download fails.

    yt-dlp handles 1800+ platforms automatically:
    YouTube, Vimeo, ok.ru, Twitter, Reddit, Dailymotion, etc.
    """
    tmp_dir = tempfile.mkdtemp(prefix="vreader_")
    outtmpl = os.path.join(tmp_dir, "video.%(ext)s")

    opts = {
        # bestvideo+bestaudio: ffmpeg merges separate video+audio tracks (best quality)
        # Cap height at 720p to dramatically speed up download and cv2 decoding times
        "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]/best",
        "outtmpl": outtmpl,
        "quiet": False,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
        "socket_timeout": 30,
        "nocheckcertificate": True,
    }

    # Point yt-dlp to ffmpeg explicitly — needed when PATH hasn't been refreshed
    # (e.g. ffmpeg just installed via winget in the same terminal session)
    import shutil
    _winget_ffmpeg = (
        r"C:\Users\sakth\AppData\Local\Microsoft\WinGet\Packages"
        r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
        r"\ffmpeg-9.0-full_build\bin"
    )
    if os.path.isdir(_winget_ffmpeg):
        opts["ffmpeg_location"] = _winget_ffmpeg
        logger.info("Using ffmpeg at: %s", _winget_ffmpeg)
    elif shutil.which("ffmpeg"):
        logger.info("Using ffmpeg from system PATH.")
    else:
        logger.warning("ffmpeg not found — audio/video merging may fail.")

    logger.info("Downloading: %s -> %s", url, tmp_dir)
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    # Find the downloaded file (yt-dlp picks the extension)
    files = glob.glob(os.path.join(tmp_dir, "video.*"))
    if not files:
        raise RuntimeError(f"yt-dlp produced no file in {tmp_dir}")

    path = files[0]
    size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.info("Download complete: %.1f MB -> %s", size_mb, path)
    return path


def _read_meta(cap: cv2.VideoCapture) -> VideoMeta:
    """Read metadata from an open VideoCapture."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n / fps if fps > 0 and n > 0 else 0.0
    return VideoMeta(fps=fps, frame_count=n, duration_seconds=duration, width=w, height=h)


# ---------------------------------------------------------------------------
# Public VideoReader class
# ---------------------------------------------------------------------------

class VideoReader:
    """
    Universal video reader. Works with any URL or local file.

    Strategy:
      1. Local file -> open directly (no download needed)
      2. Any URL    -> download via yt-dlp -> open locally -> delete on close

    Usage:
        with VideoReader("https://youtube.com/watch?v=...") as v:
            print(v.meta)
            frame = v.get_frame(300)         # frame #300 as numpy array
            frame = v.get_frame_at_ts(5.0)  # frame at 5 seconds
    """

    def __init__(self, url: str):
        self._url = url
        self._temp_path: Optional[str] = None
        self._cap = self._open(url)
        self.meta = _read_meta(self._cap)
        logger.info("Ready: %s", self.meta)

    def _open(self, url: str) -> cv2.VideoCapture:
        """Open video from local file or by downloading from URL."""

        # Local file — open directly
        if os.path.isfile(url):
            logger.info("[Local] Opening file: %s", url)
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open local file: {url}")
            return cap

        # URL — download first, then open locally
        self._temp_path = _download_video(url)
        cap = cv2.VideoCapture(self._temp_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open downloaded file: {self._temp_path}")
        logger.info("[Download] Opened successfully.")
        return cap

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Return frame at zero-based index as a BGR numpy array (H, W, 3).
        Returns None if index is out of range or read fails.

        Seeking note: cv2 seeks to the nearest keyframe (I-frame), which
        may land before the target. We step forward to the exact index.
        """
        if not (0 <= frame_index < self.meta.frame_count):
            logger.warning("Frame %d out of range [0, %d).", frame_index, self.meta.frame_count)
            return None

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        actual = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Step forward to compensate for keyframe-aligned seeking
        while actual < frame_index:
            if not self._cap.grab():
                break
            actual += 1

        ok, frame = self._cap.read()
        return frame if ok and frame is not None else None

    def get_frame_at_ts(self, seconds: float) -> Optional[np.ndarray]:
        """Return the frame closest to the given timestamp (in seconds)."""
        return self.get_frame(self.meta.ts_to_frame(seconds))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Release VideoCapture and delete any downloaded temp file."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
        if self._temp_path and os.path.exists(self._temp_path):
            os.remove(self._temp_path)
            logger.info("Temp file deleted: %s", self._temp_path)

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_) -> None:
        self.release()
