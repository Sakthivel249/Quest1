import sys
import os
import gradio as gr
import logging
import cv2
import time

# Add src to Python path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from video import VideoReader
from detector import DialogueDetector
from db import save_search, get_history

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

def get_history_list():
    rows = get_history()
    if not rows:
        return []
    # format: url, dialogue, mode, timestamp, image_path, created_at
    return [[r[0], r[1], r[2], r[3], r[5]] for r in rows]

def process_video(url, local_video, dialogue, mode):
    target_video = local_video if local_video else url
    
    if not target_video or not dialogue:
        return None, "Please provide either a YouTube URL or upload a local video, along with the target dialogue.", url, local_video, dialogue, get_history_list()
        
    try:
        # We need to run the pipeline.
        detector = DialogueDetector(mode=mode)
        
        with VideoReader(target_video) as v:
            frame_idx = detector.find_dialogue(v, dialogue)
            
            if frame_idx is not None:
                # Extract the image frame as a numpy array
                frame_bgr = v.get_frame(frame_idx)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                timestamp_str = v.meta.format_ts(v.meta.frame_to_ts(frame_idx))
                
                # Save the image to cache for DB history
                cache_dir = os.path.join(os.path.dirname(__file__), "cache")
                os.makedirs(cache_dir, exist_ok=True)
                img_name = f"history_{int(time.time())}.jpg"
                img_path = os.path.join(cache_dir, img_name)
                cv2.imwrite(img_path, frame_bgr)
                
                # Save to Database
                display_url = "Local Upload" if local_video else url
                save_search(display_url, dialogue, mode, frame_idx, timestamp_str, img_path)
                
                success_msg = f"✅ Match found at {timestamp_str} (Frame {frame_idx})"
                # Return empty strings/None to clear inputs
                return frame_rgb, success_msg, "", None, "", get_history_list()
            else:
                return None, "❌ Dialogue not found in the video.", url, local_video, dialogue, get_history_list()
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return None, "❌ Dialogue not found in the video.", url, local_video, dialogue, get_history_list()

# --- PREMIUM UI DESIGN ---
custom_css = """
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364) !important;
    color: white !important;
}
.gradio-container {
    max-width: 1000px !important;
    margin: 50px auto !important;
}
.glass-panel {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(12px) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    padding: 25px !important;
    margin-bottom: 20px !important;
}
button.primary {
    background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%) !important;
    border: none !important;
    font-weight: bold !important;
    color: white !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 20px rgba(0, 114, 255, 0.4) !important;
}
input[type="text"], textarea {
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: white !important;
}
/* Aggressively hide all Gradio 6 footers and toolbars */
footer, .footer, [class*="footer"], .gradio-container + div, .gradio-container ~ div {
    display: none !important;
    opacity: 0 !important;
    visibility: hidden !important;
    height: 0 !important;
}
/* Custom Animated Loader */
.custom-loader {
    border: 4px solid rgba(255, 255, 255, 0.1);
    border-left-color: #00c6ff;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    animation: spin 1s linear infinite;
    display: inline-block;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
"""

loader_html = """
<div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-top: 20px;">
    <div class="custom-loader"></div>
    <h3 style="margin: 0; color: #00c6ff;">AI is processing the video. This may take a few minutes...</h3>
</div>
"""

with gr.Blocks(title="Dialogue to Frame Extractor") as demo:
    with gr.Tabs():
        with gr.Tab("🔍 Search"):
            with gr.Column(elem_classes="glass-panel"):
                gr.Markdown(
                    """
                    <div style="text-align: center;">
                        <h1 style="color: white; font-size: 2.5em; margin-bottom: 5px;">🎬 Dialogue-to-Frame Extractor</h1>
                        <p style="color: #a0aec0; font-size: 1.1em;">Find the exact video frame where a specific dialogue is spoken or appears on screen.</p>
                    </div>
                    """
                )
                
                with gr.Row():
                    with gr.Column():
                        url_input = gr.Textbox(label="YouTube URL", placeholder="https://youtu.be/...", elem_classes="gr-box")
                        gr.Markdown("<p style='text-align: center; color: #a0aec0; margin: 5px 0;'>— OR —</p>")
                        local_video_input = gr.Video(label="Upload Local Video", elem_classes="gr-box")
                        dialogue_input = gr.Textbox(label="Target Dialogue", placeholder="Type the phrase to search for...", elem_classes="gr-box")
                        mode_input = gr.Radio(choices=["auto", "audio", "visual"], value="auto", label="Search Mode", elem_classes="gr-box")
                        search_button = gr.Button("Extract Frame", variant="primary")
                        
                    with gr.Column():
                        output_image = gr.Image(label="Extracted Frame", elem_classes="gr-box")
                        output_text = gr.Markdown("<h3 style='text-align: center; color: #a0aec0; margin-top: 20px;'>Ready to search! 🚀</h3>")

        with gr.Tab("🕒 History"):
            with gr.Column(elem_classes="glass-panel"):
                gr.Markdown("## Past Searches")
                history_table = gr.Dataframe(
                    headers=["URL", "Dialogue", "Mode", "Timestamp", "Date"],
                    datatype=["str", "str", "str", "str", "str"],
                    value=get_history_list(),
                    interactive=False
                )

    # Custom Loading State: Instantly clear the image and show our custom animated CSS loader
    search_button.click(
        fn=lambda u, v, d, m: (None, loader_html, u, v, d, get_history_list()),
        inputs=[url_input, local_video_input, dialogue_input, mode_input],
        outputs=[output_image, output_text, url_input, local_video_input, dialogue_input, history_table],
        queue=False,
        show_progress="hidden"
    ).then(
        fn=process_video,
        inputs=[url_input, local_video_input, dialogue_input, mode_input],
        outputs=[output_image, output_text, url_input, local_video_input, dialogue_input, history_table],
        show_progress="hidden"
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, css=custom_css, theme=gr.themes.Base())
