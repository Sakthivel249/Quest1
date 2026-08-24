"""
test_video.py
-------------
Quick manual test for video.py
Tests: URL resolution, metadata reading, frame extraction, timestamp conversion
"""

import sys
import os
import logging

# So Python can find src/video.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s"
)

from video import VideoReader, resolve_stream_url

# A small, reliable public mp4 (Big Buck Bunny sample - ~1MB, always available)
TEST_URL = "https://www.w3schools.com/html/mov_bbb.mp4"

def test_metadata():
    print("\n" + "="*50)
    print("TEST 1: Metadata Reading")
    print("="*50)
    with VideoReader(TEST_URL) as reader:
        m = reader.meta
        print(f"  FPS           : {m.fps}")
        print(f"  Frame Count   : {m.frame_count}")
        print(f"  Duration      : {m.duration_seconds:.2f} seconds")
        print(f"  Resolution    : {m.width} x {m.height}")
        assert m.fps > 0,         "FAIL: FPS should be > 0"
        assert m.frame_count > 0, "FAIL: Frame count should be > 0"
        assert m.width > 0,       "FAIL: Width should be > 0"
        print("  PASSED")

def test_get_frame():
    print("\n" + "="*50)
    print("TEST 2: Frame Extraction")
    print("="*50)
    with VideoReader(TEST_URL) as reader:
        frame = reader.get_frame(0)   # first frame
        print(f"  Frame 0 shape : {frame.shape}")   # (height, width, 3)
        assert frame is not None,     "FAIL: Frame should not be None"
        assert len(frame.shape) == 3, "FAIL: Frame should be H x W x 3"
        assert frame.shape[2] == 3,   "FAIL: Should have 3 color channels (BGR)"

        frame10 = reader.get_frame(10)
        print(f"  Frame 10 shape: {frame10.shape}")
        assert frame10 is not None,   "FAIL: Frame 10 should not be None"
        print("  PASSED")

def test_timestamp_conversion():
    print("\n" + "="*50)
    print("TEST 3: Timestamp Conversion")
    print("="*50)
    with VideoReader(TEST_URL) as reader:
        m = reader.meta
        ts = m.frame_to_timestamp(30)
        fi = m.timestamp_to_frame(ts)
        fmt = m.format_timestamp(3661.5)  # 1 hour, 1 min, 1.5 sec
        print(f"  frame 30 -> {ts:.3f}s -> frame {fi}")
        print(f"  format_timestamp(3661.5) -> {fmt}")
        assert fmt == "01:01:01.500", f"FAIL: Expected 01:01:01.500, got {fmt}"
        print("  PASSED")

def test_save_frame():
    print("\n" + "="*50)
    print("TEST 4: Save Frame as Image")
    print("="*50)
    import cv2
    output_path = os.path.join(os.path.dirname(__file__), "..", "output", "test_frame.jpg")
    with VideoReader(TEST_URL) as reader:
        frame = reader.get_frame(5)
        cv2.imwrite(output_path, frame)
        print(f"  Saved frame to: {os.path.abspath(output_path)}")
        assert os.path.exists(output_path), "FAIL: File was not saved"
        size = os.path.getsize(output_path)
        print(f"  File size     : {size} bytes")
        assert size > 0, "FAIL: Saved file is empty"
        print("  PASSED")

def test_out_of_range_frame():
    print("\n" + "="*50)
    print("TEST 5: Out-of-Range Frame (should return None gracefully)")
    print("="*50)
    with VideoReader(TEST_URL) as reader:
        frame = reader.get_frame(9999999)
        print(f"  Result: {frame}")
        assert frame is None, "FAIL: Should return None for out-of-range frame"
        print("  PASSED")

if __name__ == "__main__":
    print("\nRunning video.py tests...")
    print(f"Test URL: {TEST_URL}\n")
    try:
        test_metadata()
        test_get_frame()
        test_timestamp_conversion()
        test_save_frame()
        test_out_of_range_frame()
        print("\n" + "="*50)
        print("ALL TESTS PASSED")
        print("="*50)
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
