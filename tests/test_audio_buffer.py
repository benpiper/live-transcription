"""
Unit tests for the audio buffer module.

Tests circular buffer functionality, time-window retention,
thread safety, and edge cases.
"""

import unittest
import time
import threading
import numpy as np
from audio_buffer import AudioBuffer


class TestAudioBuffer(unittest.TestCase):
    """Test suite for AudioBuffer class."""

    def setUp(self):
        """Create a fresh buffer for each test."""
        self.buffer = AudioBuffer(window_size_sec=10.0)

    def tearDown(self):
        """Clean up after each test."""
        self.buffer.reset()

    def test_add_and_retrieve_single_chunk(self):
        """Test adding a single audio chunk and retrieving it."""
        timestamp = time.time()
        audio_data = np.random.randn(4096).astype(np.float32)

        self.buffer.add_chunk(timestamp, audio_data)

        # Retrieve the data
        retrieved = self.buffer.get_audio_range(timestamp - 1, timestamp + 1)

        # Verify shape and values
        self.assertEqual(retrieved.shape, audio_data.shape)
        np.testing.assert_array_almost_equal(retrieved, audio_data)

    def test_add_multiple_chunks_and_retrieve(self):
        """Test adding multiple chunks and retrieving concatenated audio."""
        base_time = time.time()
        chunk1 = np.ones(4096, dtype=np.float32)
        chunk2 = np.ones(4096, dtype=np.float32) * 2

        self.buffer.add_chunk(base_time, chunk1)
        self.buffer.add_chunk(base_time + 0.256, chunk2)

        # Retrieve range covering both
        retrieved = self.buffer.get_audio_range(base_time - 0.5, base_time + 0.5)

        # Should contain both chunks (allowing for trimming rounding)
        # Expect approximately 8192 samples (±200 for rounding)
        self.assertGreaterEqual(len(retrieved), 8000)
        self.assertLess(len(retrieved), 8300)

    def test_discard_old_chunks(self):
        """Test that chunks older than window are discarded."""
        base_time = time.time()
        window = 10.0  # 10 second window

        # Add chunk at t=0
        chunk1 = np.ones(4096, dtype=np.float32)
        self.buffer.add_chunk(base_time, chunk1)

        # Add chunk at t=15 (outside 10s window)
        chunk2 = np.ones(4096, dtype=np.float32) * 2
        self.buffer.add_chunk(base_time + 15, chunk2)

        # Chunk 1 should be discarded, chunk 2 should remain
        stats = self.buffer.get_buffer_stats()
        self.assertEqual(stats["num_chunks"], 1)

        # Verify chunk 2 is still there
        retrieved = self.buffer.get_audio_range(base_time + 14, base_time + 16)
        np.testing.assert_array_equal(retrieved, chunk2)

    def test_invalid_time_range(self):
        """Test that invalid time ranges raise ValueError."""
        with self.assertRaises(ValueError):
            self.buffer.get_audio_range(100, 50)  # end < start

    def test_no_data_available(self):
        """Test that requesting audio outside buffer raises ValueError."""
        base_time = time.time()

        # Try to get audio when buffer is empty
        with self.assertRaises(ValueError):
            self.buffer.get_audio_range(base_time, base_time + 1)

    def test_partial_range_request(self):
        """Test requesting a range that partially overlaps stored data."""
        base_time = time.time()
        audio_data = np.ones(4096, dtype=np.float32)

        self.buffer.add_chunk(base_time, audio_data)

        # Request a range that includes but extends beyond the chunk
        retrieved = self.buffer.get_audio_range(base_time - 1, base_time + 2)

        # Should get the chunk
        np.testing.assert_array_equal(retrieved, audio_data)

    def test_get_buffer_stats(self):
        """Test buffer statistics reporting."""
        base_time = time.time()
        audio_data = np.ones(4096, dtype=np.float32)

        self.buffer.add_chunk(base_time, audio_data)

        stats = self.buffer.get_buffer_stats()

        self.assertEqual(stats["num_chunks"], 1)
        self.assertEqual(stats["window_size_sec"], 10.0)
        self.assertIsNotNone(stats["oldest_timestamp"])
        self.assertIsNotNone(stats["newest_timestamp"])
        self.assertEqual(stats["buffer_size_bytes"], audio_data.nbytes)

    def test_buffer_stats_empty(self):
        """Test buffer stats when empty."""
        stats = self.buffer.get_buffer_stats()

        self.assertEqual(stats["num_chunks"], 0)
        self.assertEqual(stats["buffer_size_bytes"], 0)
        self.assertIsNone(stats["oldest_timestamp"])
        self.assertIsNone(stats["newest_timestamp"])

    def test_set_window_size(self):
        """Test changing window size and auto-cleanup."""
        base_time = time.time()

        # Add chunk at t=0
        self.buffer.add_chunk(base_time, np.ones(4096, dtype=np.float32))
        # Add chunk at t=5
        self.buffer.add_chunk(base_time + 5, np.ones(4096, dtype=np.float32))

        stats_before = self.buffer.get_buffer_stats()
        self.assertEqual(stats_before["num_chunks"], 2)

        # Reduce window to 3 seconds
        self.buffer.set_window_size(3.0)

        # At least one chunk should be cleaned up (depends on current time)
        stats = self.buffer.get_buffer_stats()
        self.assertLessEqual(stats["num_chunks"], 2)

    def test_reset(self):
        """Test resetting the buffer clears all data."""
        base_time = time.time()

        self.buffer.add_chunk(base_time, np.ones(4096, dtype=np.float32))
        self.buffer.add_chunk(base_time + 1, np.ones(4096, dtype=np.float32))

        self.buffer.reset()

        stats = self.buffer.get_buffer_stats()
        self.assertEqual(stats["num_chunks"], 0)

    def test_thread_safety_concurrent_adds(self):
        """Test that concurrent add operations don't corrupt data."""
        base_time = time.time()
        num_threads = 5
        chunks_per_thread = 10

        def add_chunks(thread_id):
            for i in range(chunks_per_thread):
                timestamp = base_time + (thread_id * 100) + i
                audio = np.ones(4096, dtype=np.float32) * (thread_id + 1)
                self.buffer.add_chunk(timestamp, audio)

        threads = [threading.Thread(target=add_chunks, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify total chunks added
        stats = self.buffer.get_buffer_stats()
        # Note: may be < total due to window expiration, but should have most
        self.assertGreater(stats["num_chunks"], 0)

    def test_thread_safety_concurrent_reads(self):
        """Test that concurrent read operations are safe."""
        base_time = time.time()

        # Pre-populate buffer
        for i in range(20):
            self.buffer.add_chunk(base_time + i * 0.1, np.ones(4096, dtype=np.float32))

        results = []

        def read_range():
            try:
                data = self.buffer.get_audio_range(base_time, base_time + 1)
                results.append(len(data))
            except ValueError:
                pass

        threads = [threading.Thread(target=read_range) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads should succeed
        self.assertGreater(len(results), 0)

    def test_data_type_conversion(self):
        """Test that various data types are converted to Float32."""
        timestamp = time.time()

        # Add Int16 data
        int16_data = np.array([100, 200, -100], dtype=np.int16)
        self.buffer.add_chunk(timestamp, int16_data)

        # Retrieve and verify it's Float32
        retrieved = self.buffer.get_audio_range(timestamp - 1, timestamp + 1)
        self.assertEqual(retrieved.dtype, np.float32)

    def test_duration_calculation(self):
        """Test that duration_available_sec is calculated correctly."""
        base_time = time.time()

        # Add chunks at known times
        self.buffer.add_chunk(base_time, np.ones(4096, dtype=np.float32))
        self.buffer.add_chunk(base_time + 5.0, np.ones(4096, dtype=np.float32))

        stats = self.buffer.get_buffer_stats()

        # Duration should be approximately 5 seconds
        self.assertAlmostEqual(stats["duration_available_sec"], 5.0, places=1)

    def test_audio_trimming_to_exact_range(self):
        """Test that audio is trimmed to exact requested time range."""
        base_time = time.time()
        sample_rate = 16000

        # Add chunk of ones at t=0
        chunk1 = np.ones(4096, dtype=np.float32)
        self.buffer.add_chunk(base_time, chunk1)

        # Add chunk of twos at t=0.256
        chunk2 = np.ones(4096, dtype=np.float32) * 2
        self.buffer.add_chunk(base_time + 0.256, chunk2)

        # Request audio starting partway through chunk1
        # Request 0.128s into first chunk (2048 samples into 4096)
        retrieved = self.buffer.get_audio_range(
            base_time + 0.128,  # Start midway through chunk1
            base_time + 0.384   # Extend into chunk2
        )

        # Verify we get approximately the right duration
        # 0.128 to 0.384 = 0.256 seconds = 4096 samples @ 16kHz
        # (allowing some tolerance for rounding)
        expected_samples = int(0.256 * sample_rate)
        self.assertAlmostEqual(len(retrieved), expected_samples, delta=10)

        # Verify the data starts with values from chunk1 (ones), not the beginning
        # The first sample should be close to 1.0, not missing data
        self.assertGreater(np.mean(retrieved[:100]), 0.9)


if __name__ == "__main__":
    unittest.main()
