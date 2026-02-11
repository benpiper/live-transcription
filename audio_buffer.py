"""
Thread-safe circular audio buffer for backend audio storage.

Stores audio chunks with timestamps in a time-windowed buffer.
Automatically discards chunks older than the configured window size.
"""

import time
import threading
import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Global audio buffer instance
_audio_buffer_instance = None
_buffer_lock = threading.Lock()


class AudioBuffer:
    """Thread-safe circular buffer for audio storage with time-window based retention."""

    def __init__(self, window_size_sec: float = 7200.0, sample_rate: int = 16000):
        """
        Initialize the audio buffer.

        Args:
            window_size_sec: Time window in seconds. Chunks older than this are discarded.
            sample_rate: Sample rate of audio (for metadata, not used in processing).
        """
        self.window_size_sec = window_size_sec
        self.sample_rate = sample_rate

        # Storage: dict mapping timestamp -> np.ndarray(Float32)
        self.chunks = {}
        self.lock = threading.Lock()

        logger.info(f"AudioBuffer initialized with window size {window_size_sec}s")

    def add_chunk(self, timestamp: float, audio_data: np.ndarray) -> None:
        """
        Add an audio chunk to the buffer.

        Automatically discards chunks older than the configured window.

        Args:
            timestamp: Unix timestamp (float, seconds.microseconds)
            audio_data: Audio samples as numpy array (any dtype, will be converted to Float32)
        """
        with self.lock:
            # Convert to Float32 if not already
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Add the chunk
            self.chunks[timestamp] = audio_data
            logger.debug(f"Added audio chunk at {timestamp:.3f}, size={len(audio_data)} samples ({audio_data.nbytes} bytes), rms={np.sqrt(np.mean(audio_data**2)):.4f}")

            # Discard old chunks outside the window
            cutoff_time = timestamp - self.window_size_sec
            old_keys = [t for t in self.chunks if t < cutoff_time]

            for old_key in old_keys:
                del self.chunks[old_key]
                logger.debug(f"Discarded audio chunk at {old_key:.2f}")

    def get_audio_range(
        self, start_time: float, end_time: float
    ) -> np.ndarray:
        """
        Retrieve concatenated audio for a time range, trimmed to exact boundaries.

        Args:
            start_time: Start timestamp (Unix float)
            end_time: End timestamp (Unix float)

        Returns:
            Concatenated Float32 audio data as numpy array, trimmed to requested range.

        Raises:
            ValueError: If time range is invalid or no data available.
        """
        if end_time <= start_time:
            raise ValueError(
                f"Invalid time range: end_time ({end_time}) must be > start_time ({start_time})"
            )

        with self.lock:
            # Find chunks within or overlapping the time range
            # Include chunks that could overlap: chunk_start <= end_time
            relevant_chunks = [
                (t, data)
                for t, data in sorted(self.chunks.items())
                if t <= end_time and t >= start_time - 1.0  # Within ~1 second before start
            ]

            if not relevant_chunks:
                raise ValueError(
                    f"No audio available for range {start_time:.2f}-{end_time:.2f}"
                )

            # Concatenate audio chunks
            audio_arrays = [data for _, data in relevant_chunks]
            concatenated = np.concatenate(audio_arrays)

            # Trim to exact requested time range
            # Calculate sample offsets based on start_time of first chunk
            first_chunk_time = relevant_chunks[0][0]
            sample_rate = self.sample_rate

            # Offset from first chunk time to requested start time
            time_offset = start_time - first_chunk_time
            start_sample = max(0, int(time_offset * sample_rate))

            # Offset to end time
            time_offset_end = end_time - first_chunk_time
            end_sample = min(len(concatenated), int(time_offset_end * sample_rate))

            # Ensure we have valid range
            if start_sample >= len(concatenated):
                raise ValueError(
                    f"Requested start_time {start_time:.2f} is beyond available audio"
                )

            trimmed = concatenated[start_sample:end_sample]

            if len(trimmed) == 0:
                raise ValueError(
                    f"No audio available for range {start_time:.2f}-{end_time:.2f}"
                )

            rms = np.sqrt(np.mean(trimmed**2))
            logger.debug(f"Retrieved audio range [{start_time:.3f}, {end_time:.3f}]: {len(trimmed)} samples, rms={rms:.4f}")

            return trimmed.astype(np.float32)

    def get_buffer_stats(self) -> dict:
        """
        Get current buffer statistics.

        Returns:
            Dictionary with:
            - window_size_sec: Configured window size
            - buffer_size_bytes: Current total size
            - buffer_size_mb: Current total size in MB
            - oldest_timestamp: Earliest chunk timestamp (or None if empty)
            - newest_timestamp: Latest chunk timestamp (or None if empty)
            - num_chunks: Number of chunks in buffer
            - duration_available_sec: Duration of audio available (or 0 if empty)
        """
        with self.lock:
            if not self.chunks:
                return {
                    "window_size_sec": self.window_size_sec,
                    "buffer_size_bytes": 0,
                    "buffer_size_mb": 0.0,
                    "oldest_timestamp": None,
                    "newest_timestamp": None,
                    "num_chunks": 0,
                    "duration_available_sec": 0.0,
                }

            # Calculate total size
            total_bytes = sum(
                data.nbytes for data in self.chunks.values()
            )
            total_mb = total_bytes / (1024 * 1024)

            # Get time range
            timestamps = sorted(self.chunks.keys())
            oldest = timestamps[0]
            newest = timestamps[-1]
            duration = newest - oldest

            return {
                "window_size_sec": self.window_size_sec,
                "buffer_size_bytes": total_bytes,
                "buffer_size_mb": round(total_mb, 2),
                "oldest_timestamp": round(oldest, 3),
                "newest_timestamp": round(newest, 3),
                "num_chunks": len(self.chunks),
                "duration_available_sec": round(duration, 2),
            }

    def set_window_size(self, window_size_sec: float) -> None:
        """
        Update the window size and perform cleanup.

        Args:
            window_size_sec: New window size in seconds.
        """
        with self.lock:
            self.window_size_sec = window_size_sec

            # Immediately clean up old data
            now = time.time()
            cutoff_time = now - window_size_sec
            old_keys = [t for t in self.chunks if t < cutoff_time]

            for old_key in old_keys:
                del self.chunks[old_key]

            logger.info(f"AudioBuffer window size changed to {window_size_sec}s, cleaned {len(old_keys)} old chunks")

    def reset(self) -> None:
        """Clear all audio data from the buffer (for testing/cleanup)."""
        with self.lock:
            num_chunks = len(self.chunks)
            self.chunks.clear()
            logger.info(f"AudioBuffer reset, cleared {num_chunks} chunks")


def get_audio_buffer() -> AudioBuffer:
    """
    Get the global audio buffer instance (singleton).

    Returns:
        The global AudioBuffer instance.
    """
    global _audio_buffer_instance

    if _audio_buffer_instance is None:
        with _buffer_lock:
            if _audio_buffer_instance is None:
                _audio_buffer_instance = AudioBuffer()

    return _audio_buffer_instance


# Convenience alias for direct import
audio_buffer = get_audio_buffer()
