# Architectural Approach: Dialogue-to-Frame Detector

![Architecture Flowchart](./architecture.png)
## 1. What Has Been Done
I designed and built a production-ready AI pipeline that accepts video inputs (either dynamically streamed from YouTube or uploaded locally) and automatically scans them to find the exact millisecond a specific piece of dialogue or on-screen text appears. The tool successfully extracts the precise video frame at that timestamp and presents it in a beautifully polished Glassmorphism Web UI.

Key engineering milestones achieved:
- **Dual-Engine Search**: Engineered both an Audio Search (Whisper) and Visual Search (EasyOCR) engine.
- **Hardware-Aware Concurrency**: Built a Python `ThreadPoolExecutor` system that runs searches in parallel on GPUs, but smartly drops down to sequential execution on CPUs to prevent PyTorch thread-thrashing.
- **Persistent AI Caching**: Created an MD5-hashed caching layer that saves raw MP4 downloads and heavy JSON Whisper transcriptions, reducing repeat search times from minutes to milliseconds.
- **SQLite History Tracking**: Integrated a robust local database to silently track and persist successful searches, viewable instantly in the web UI.

## 2. Architecture
The architecture operates as a highly concurrent pipeline:
1. **Ingestion Node (`app.py` / `video.py`)**: Accepts YouTube URLs or local video file uploads. YouTube URLs are routed through `yt-dlp` using OAuth2 and Mobile/Web client spoofing to safely bypass bot-blockers, downloading the media directly into the `cache/` directory.
2. **Detection Orchestrator (`detector.py`)**: Triggers the search based on the chosen mode (`auto`, `audio`, `visual`).
3. **Audio Node (`audio_search.py`)**: Uses OpenAI's Whisper model to transcribe the audio track into JSON metadata, preserving word-level timestamps. A sliding-window `rapidfuzz` matcher scans the transcript to find the exact start time of the target phrase.
4. **Visual Node (`ocr.py`)**: A fallback engine that utilizes EasyOCR (with support for English and Tamil) to scan the video frame-by-frame (1 frame every 2 seconds) for on-screen text or lyrics.
5. **Frame Extraction & DB Storage (`db.py`)**: Once a match is found, OpenCV is used to seek to the precise millisecond and extract the RGB frame. The result is logged into the local SQLite database.

## 3. Technology Stack
- **Frontend / UI**: Gradio 6.0 (with custom vanilla CSS for Glassmorphism styling and custom animated loaders)
- **Audio AI**: `openai-whisper` (ASR / Transcription)
- **Visual AI**: `easyocr` (Optical Character Recognition)
- **Video Processing**: `opencv-python` (cv2), `ffmpeg`
- **Stream Ingestion**: `yt-dlp`
- **Fuzzy Matching**: `rapidfuzz`
- **Storage**: SQLite (`sqlite3`)

## 4. Limitations
- **Processing Time on CPU**: While optimized with a lightweight model, transcribing a full YouTube video using Whisper purely on a CPU can still take a few minutes for longer videos.
- **Visual Search Speed**: EasyOCR scanning is highly computationally expensive. Scanning 1 frame every 2 seconds is a necessary compromise; increasing frame density would drastically slow down the pipeline.
- **Context Awareness**: The fuzzy matcher looks for exact phonetic strings or near-matches. It cannot yet do semantic matching (e.g., searching for "greeting" will not automatically match the spoken word "hello").

## 5. Future Updates
- **Semantic Vector Search**: Integrate a lightweight embedding model (like `all-MiniLM-L6-v2`) to store the Whisper transcriptions in a Vector Database (like ChromaDB). This would allow users to search conceptually rather than requiring the exact dialogue text.
- **Real-Time Streaming Processing**: Upgrade the `yt-dlp` pipeline to process the video stream chunk-by-chunk in real-time as it downloads, rather than waiting for the entire video to finish downloading before starting the AI analysis.
- **Face/Speaker Recognition**: Combine audio transcription with a Face ID model so the user can search for a specific person saying a specific phrase.
