"""
Integration tests for audio API endpoints.

Tests the /api/audio/range and /api/audio/buffer-status endpoints
with real FastAPI application instance.
"""

import pytest
import numpy as np
import time
import json
from unittest.mock import patch, MagicMock
from audio_buffer import AudioBuffer
from config import load_config, get_setting

# Mock the modules that require GPU/CUDA before importing web_server
with patch('transcription_engine.setup_cuda_paths'):
    with patch('web_server.set_queues'):
        from web_server import create_app


@pytest.fixture
def audio_buffer_test():
    """Create a test audio buffer with sample data."""
    buffer = AudioBuffer(window_size_sec=100.0)

    # Add some test audio chunks
    base_time = time.time()
    for i in range(10):
        timestamp = base_time + (i * 0.256)  # ~256ms chunks
        audio_data = np.sin(2 * np.pi * 440 * np.arange(4096) / 16000).astype(np.float32)
        buffer.add_chunk(timestamp, audio_data)

    yield buffer

    buffer.reset()


@pytest.fixture
def client(audio_buffer_test):
    """Create a FastAPI test client."""
    # Patch audio_buffer in web_server before creating app
    with patch('web_server.audio_buffer', audio_buffer_test):
        app = create_app()
        from fastapi.testclient import TestClient
        return TestClient(app)


def test_ping_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/ping")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_buffer_status_endpoint(client):
    """Test audio buffer status endpoint."""
    response = client.get("/api/audio/buffer-status")
    assert response.status_code == 200
    data = response.json()

    assert "window_size_sec" in data
    assert "buffer_size_bytes" in data
    assert "buffer_size_mb" in data
    assert "num_chunks" in data
    assert data["num_chunks"] == 10


def test_buffer_status_empty_buffer(client):
    """Test buffer status with empty buffer."""
    empty_buffer = AudioBuffer(window_size_sec=100.0)
    with patch('web_server.audio_buffer', empty_buffer):
        app = create_app()
        from fastapi.testclient import TestClient
        test_client = TestClient(app)

        response = test_client.get("/api/audio/buffer-status")
        assert response.status_code == 200
        data = response.json()
        assert data["num_chunks"] == 0
        assert data["buffer_size_bytes"] == 0


def test_audio_range_wav_format(client, audio_buffer_test):
    """Test retrieving audio in WAV format."""
    # Get time range from buffer
    stats = audio_buffer_test.get_buffer_stats()
    start_time = stats["oldest_timestamp"]
    end_time = stats["newest_timestamp"]

    response = client.get(
        f"/api/audio/range?start_time={start_time}&end_time={end_time}&format=wav"
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert b"RIFF" in response.content[:4]  # WAV header
    assert b"WAVE" in response.content[8:12]


def test_audio_range_raw_format(client, audio_buffer_test):
    """Test retrieving audio in raw Float32 format."""
    stats = audio_buffer_test.get_buffer_stats()
    start_time = stats["oldest_timestamp"]
    end_time = stats["newest_timestamp"]

    response = client.get(
        f"/api/audio/range?start_time={start_time}&end_time={end_time}&format=raw"
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"

    # Verify it's valid Float32 data
    float_data = np.frombuffer(response.content, dtype=np.float32)
    assert len(float_data) > 0


def test_audio_range_invalid_format(client, audio_buffer_test):
    """Test that invalid format raises error."""
    stats = audio_buffer_test.get_buffer_stats()
    start_time = stats["oldest_timestamp"]
    end_time = stats["newest_timestamp"]

    response = client.get(
        f"/api/audio/range?start_time={start_time}&end_time={end_time}&format=invalid"
    )

    assert response.status_code == 400
    data = response.json()
    assert "Invalid format" in data["detail"]


def test_audio_range_invalid_time_range(client):
    """Test that invalid time range (end <= start) raises error."""
    response = client.get(
        "/api/audio/range?start_time=100.0&end_time=50.0&format=wav"
    )

    assert response.status_code == 400
    data = response.json()
    assert "Invalid time range" in data["detail"]


def test_audio_range_outside_buffer(client):
    """Test requesting audio outside buffer window."""
    # Request far future time
    future_time = time.time() + 10000
    response = client.get(
        f"/api/audio/range?start_time={future_time}&end_time={future_time + 1}&format=wav"
    )

    assert response.status_code == 400
    data = response.json()
    assert "No audio available" in data["detail"]


def test_audio_range_partial_overlap(client, audio_buffer_test):
    """Test requesting range that partially overlaps stored audio."""
    stats = audio_buffer_test.get_buffer_stats()
    oldest = stats["oldest_timestamp"]

    # Request before stored data but extending into it
    response = client.get(
        f"/api/audio/range?start_time={oldest - 10}&end_time={oldest + 1}&format=raw"
    )

    # Should succeed with available portion
    assert response.status_code == 200
    float_data = np.frombuffer(response.content, dtype=np.float32)
    assert len(float_data) > 0


def test_audio_range_default_format(client, audio_buffer_test):
    """Test that default format is WAV."""
    stats = audio_buffer_test.get_buffer_stats()
    start_time = stats["oldest_timestamp"]
    end_time = stats["newest_timestamp"]

    # Omit format parameter (should default to wav)
    response = client.get(
        f"/api/audio/range?start_time={start_time}&end_time={end_time}"
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"


def test_audio_range_missing_parameters(client):
    """Test that missing required parameters cause 422 error."""
    response = client.get("/api/audio/range")
    assert response.status_code == 422  # Unprocessable Entity


def test_audio_data_integrity(client, audio_buffer_test):
    """Test that retrieved audio data matches what was stored."""
    # Get all data
    stats = audio_buffer_test.get_buffer_stats()
    start_time = stats["oldest_timestamp"]
    end_time = stats["newest_timestamp"]

    response = client.get(
        f"/api/audio/range?start_time={start_time}&end_time={end_time}&format=raw"
    )

    assert response.status_code == 200
    retrieved_data = np.frombuffer(response.content, dtype=np.float32)

    # Compare with direct buffer retrieval
    direct_data = audio_buffer_test.get_audio_range(start_time, end_time)

    np.testing.assert_array_almost_equal(retrieved_data, direct_data)


def test_wav_format_validity(client, audio_buffer_test):
    """Test that WAV format is valid and can be parsed."""
    import wave
    import io

    stats = audio_buffer_test.get_buffer_stats()
    start_time = stats["oldest_timestamp"]
    end_time = stats["newest_timestamp"]

    response = client.get(
        f"/api/audio/range?start_time={start_time}&end_time={end_time}&format=wav"
    )

    assert response.status_code == 200

    # Try to parse as WAV
    wav_buffer = io.BytesIO(response.content)
    with wave.open(wav_buffer, "rb") as wav_file:
        assert wav_file.getnchannels() == 1  # Mono
        assert wav_file.getsampwidth() == 4  # 32-bit (4 bytes per sample)
        assert wav_file.getframerate() == 16000
        frames = wav_file.readframes(-1)
        assert len(frames) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
