import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from video import VideoMeta
from sampler import get_coarse_sample_indices, get_fine_sample_indices, get_exact_frame_range

class TestSampler(unittest.TestCase):
    def setUp(self):
        # 10 fps, 100 frames, 10 seconds total duration
        self.meta = VideoMeta(fps=10.0, frame_count=100, duration_seconds=10.0, width=1920, height=1080)

    def test_coarse_sampling(self):
        # 1 frame every second
        indices = get_coarse_sample_indices(self.meta, interval_seconds=1.0)
        # Should be [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        # (It stops before 100 because frame indices are 0 to 99)
        self.assertEqual(indices, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

    def test_fine_sampling(self):
        # Center at 5 seconds, window of 1 second (so 4.0 to 6.0), interval of 0.5s
        # Expect frames at 4.0, 4.5, 5.0, 5.5, 6.0 -> 40, 45, 50, 55, 60
        indices = get_fine_sample_indices(self.meta, center_ts=5.0, window_seconds=1.0, interval_seconds=0.5)
        self.assertEqual(indices, [40, 45, 50, 55, 60])
        
    def test_fine_sampling_bounds(self):
        # Center at 0 seconds, window of 2 seconds -> should crop at 0
        indices = get_fine_sample_indices(self.meta, center_ts=0.0, window_seconds=2.0, interval_seconds=1.0)
        self.assertEqual(indices, [0, 10, 20])
        
    def test_exact_frame_range(self):
        # Between 1.0 and 1.5 seconds -> Frames 10 to 15
        indices = get_exact_frame_range(self.meta, start_ts=1.0, end_ts=1.5)
        self.assertEqual(indices, [10, 11, 12, 13, 14, 15])

if __name__ == "__main__":
    unittest.main()
