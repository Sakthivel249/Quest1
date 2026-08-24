import sys
import os
import gradio as gr
import logging
import cv2

# Add src to Python path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from video import VideoReader
from detector import DialogueDetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

def process_video(url, dialogue, mode):
    if not url or not dialogue:
        return None, "Please provide both a YouTube URL and target dialogue."
        
    try:
        # We need to run the pipeline.
        detector = DialogueDetector(mode=mode)
        
        with VideoReader(url) as v:
            frame_idx = detector.find_dialogue(v, dialogue)
            
            if frame_idx is not None:
                # Extract the image frame as a numpy array
                frame_bgr = v.get_frame(frame_idx)
                
                # Gradio expects RGB images, OpenCV provides BGR
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                timestamp_str = v.meta.format_ts(v.meta.frame_to_ts(frame_idx))
                return frame_rgb, f"✅ Match found at {timestamp_str} (Frame {frame_idx})"
            else:
                return None, "❌ Dialogue not found in the video."
    except Exception as e:
        return None, f"Error: {str(e)}"

# Define the Gradio Interface
with gr.Blocks(title="Dialogue to Frame Extractor") as demo:
    gr.Markdown("# 🎬 Dialogue-to-Frame Extractor")
    gr.Markdown("Find the exact video frame where a specific dialogue is spoken or appears on screen.")
    
    with gr.Row():
        with gr.Column():
            url_input = gr.Textbox(label="YouTube URL", placeholder="https://youtu.be/...")
            dialogue_input = gr.Textbox(label="Target Dialogue", placeholder="Type the phrase to search for...")
            mode_input = gr.Radio(choices=["auto", "audio", "visual"], value="auto", label="Search Mode")
            search_button = gr.Button("🔍 Search Frame", variant="primary")
            
        with gr.Column():
            output_image = gr.Image(label="Extracted Frame")
            output_text = gr.Markdown()

    search_button.click(
        fn=process_video,
        inputs=[url_input, dialogue_input, mode_input],
        outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    # Launch the web app on local port 7860
    demo.launch(server_name="0.0.0.0", server_port=7860)
