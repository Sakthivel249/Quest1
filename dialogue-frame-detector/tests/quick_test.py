"""
quick_test.py  -  test VideoReader with any URL
Usage: python tests/quick_test.py "URL_HERE"
"""
import sys, os, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

import cv2
from video import VideoReader

if len(sys.argv) < 2:
    print("Usage: python tests/quick_test.py <VIDEO_URL>")
    sys.exit(1)

url = sys.argv[1]
print(f"\nTesting: {url}\n")

with VideoReader(url) as v:
    print("=" * 50)
    print("VIDEO INFO")
    print("=" * 50)
    print(f"  {v.meta}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(out_dir, exist_ok=True)

    checkpoints = {
        "first" : 0,
        "middle": v.meta.frame_count // 2,
        "last"  : max(0, v.meta.frame_count - 2),
    }

    print("\n" + "=" * 50)
    print("FRAME SAMPLES")
    print("=" * 50)
    for label, idx in checkpoints.items():
        frame = v.get_frame(idx)
        if frame is not None:
            ts = v.meta.format_ts(v.meta.frame_to_ts(idx))
            path = os.path.join(out_dir, f"quick_{label}_frame.jpg")
            cv2.imwrite(path, frame)
            kb = os.path.getsize(path) // 1024
            print(f"  [{label:6s}] Frame {idx:6d} | {ts} | {kb} KB -> {os.path.basename(path)}")
        else:
            print(f"  [{label:6s}] Could not read frame {idx}")

print("\nDone. Check output/ folder.\n")
