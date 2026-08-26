# 🎬 Dialogue-to-Frame Detector

A powerful AI-driven tool that finds the exact video frame where a specific dialogue is spoken or shown on screen. It automatically downloads videos from YouTube (or uses local files) and seamlessly falls back between **Audio Search** (Whisper) and **Visual OCR Search** (EasyOCR).

## Features
- **Audio Search (Whisper)**: Transcribes speech and uses a sliding-window fuzzy matcher to find the exact millisecond a phrase begins.
- **Visual Search (EasyOCR)**: If audio fails (e.g. background text, different languages), it scans the video frames to find the text on-screen.
- **Hardware-Aware Multithreading**: Automatically detects if you have a GPU. If so, it runs both Audio and Visual searches concurrently. If on CPU, it runs sequentially to prevent thread-thrashing.
- **Bot-Block Bypass**: Employs Mobile/Web spoofing and OAuth2 to bypass YouTube download blocks.
- **Web UI**: Includes a beautiful Gradio frontend!

## Installation

1. Clone the repository and activate your virtual environment.
2. Install the Python dependencies:
   ```bash
   pip install -r dialogue-frame-detector/requirements.txt
   ```
3. Install `ffmpeg` (required by `yt-dlp` to merge video and audio):
   - **Windows**: `winget install --id Gyan.FFmpeg -e`
   - **Mac**: `brew install ffmpeg`

## Usage

### 1. Web UI (Recommended)
Launch the Gradio Web App to search visually in your browser:
```bash
python dialogue-frame-detector/app.py
```
Then navigate to `http://127.0.0.1:7860` in your browser.

### 2. Command Line (CLI)
You can also run the tool directly from your terminal. It will save the resulting frame image into the `output/` folder.
```bash
python dialogue-frame-detector/src/main.py "https://youtu.be/syFZfO_wfMQ" "thinking about it lately" --mode auto
```

#### Modes:
- `--mode auto` (Default): Tries Audio first, falls back to Visual (or runs concurrently on GPU).
- `--mode audio`: Forces Whisper transcription search only.
- `--mode visual`: Forces EasyOCR frame-scanning search only.
