"""
Web server and WebSocket handling for the live transcription dashboard.

This module provides the FastAPI application, WebSocket endpoints,
and the broadcaster worker for real-time updates.
"""

import asyncio
import logging
import queue
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
import io
import numpy as np

from models import (
    HealthCheckResponse,
    CurrentSessionResponse,
    SessionListResponse,
    SessionMetadata,
    SessionFull,
    SessionCreateResponse,
    SaveSessionResponse,
    DeleteSessionResponse,
    SessionComparisonRequest,
    SessionComparisonResponse,
    ArchiveSessionResponse,
    RestoreSessionResponse,
    SessionRolloverStatus,
    EngineStatusResponse,
)

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections for broadcasting."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Active: {len(self.active_connections)}")
        print(f"Client connected. Active: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection from the active list."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("Client disconnected.")
            print("Client disconnected.")
    
    async def broadcast_json(self, data: dict):
        """Send JSON data to all connected clients."""
        for connection in list(self.active_connections):
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.debug(f"Failed to send to client: {e}")
                self.disconnect(connection)
    
    async def broadcast_bytes(self, data: bytes):
        """Send binary data to all connected clients."""
        for connection in list(self.active_connections):
            try:
                await connection.send_bytes(data)
            except Exception as e:
                logger.debug(f"Failed to send bytes to client: {e}")
                self.disconnect(connection)

    async def close_all(self):
        """Close all active WebSocket connections gracefully."""
        if not self.active_connections:
            return

        logger.info(f"Closing {len(self.active_connections)} WebSocket connection(s)...")

        # Send shutdown message
        shutdown_msg = {"type": "shutdown", "message": "Server is shutting down"}
        await self.broadcast_json(shutdown_msg)

        # Give clients 500ms to process
        await asyncio.sleep(0.5)

        # Close all connections
        for connection in list(self.active_connections):
            try:
                await connection.close()
            except Exception as e:
                logger.debug(f"Error closing connection: {e}")

        self.active_connections.clear()
        logger.info("All WebSocket connections closed")


# Global connection manager instance
ws_manager = ConnectionManager()

# Queues for inter-thread communication (set by main module)
broadcast_queue: queue.Queue = None
audio_broadcast_queue: queue.Queue = None
stop_event: threading.Event = None


def set_queues(json_queue: queue.Queue, audio_queue: queue.Queue, stop_evt: threading.Event = None):
    """Set the broadcast queues and stop event from the main module."""
    global broadcast_queue, audio_broadcast_queue, stop_event
    broadcast_queue = json_queue
    audio_broadcast_queue = audio_queue
    stop_event = stop_evt


def create_wav_from_float32(audio_data: np.ndarray, sample_rate: int = 16000) -> bytes:
    """
    Convert Float32 numpy array to WAV format bytes (16-bit PCM).

    Args:
        audio_data: Audio samples as Float32 numpy array (should be in range [-1.0, 1.0])
        sample_rate: Sample rate in Hz (default 16000)

    Returns:
        WAV file bytes
    """
    import wave

    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # Create in-memory WAV file
    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit PCM (2 bytes per sample)
        wav_file.setframerate(sample_rate)

        # Convert Float32 to 16-bit PCM
        # Clip to [-1.0, 1.0] range and scale to int16 range
        audio_clipped = np.clip(audio_data, -1.0, 1.0)
        pcm_int16 = (audio_clipped * 32767).astype(np.int16)
        wav_file.writeframes(pcm_int16.tobytes())

    wav_buffer.seek(0)
    return wav_buffer.getvalue()


async def websocket_broadcaster():
    """
    Background worker to broadcast data from queues to WebSocket clients.

    Continuously drains the broadcast queues and sends updates to all
    connected clients.
    """
    logger.info("WebSocket Broadcaster started.")
    print("WebSocket Broadcaster started.")

    while not (stop_event and stop_event.is_set()):
        try:
            # Broadcast JSON transcripts
            if broadcast_queue is not None:
                while not broadcast_queue.empty():
                    try:
                        data = broadcast_queue.get_nowait()
                        await ws_manager.broadcast_json(data)
                    except queue.Empty:
                        break

            # Broadcast raw audio
            if audio_broadcast_queue is not None:
                while not audio_broadcast_queue.empty():
                    try:
                        audio_data = audio_broadcast_queue.get_nowait()
                        await ws_manager.broadcast_bytes(audio_data)
                    except queue.Empty:
                        break

            await asyncio.sleep(0.02)  # 50 updates/sec max

        except Exception as e:
            logger.error(f"Broadcaster error: {e}")
            await asyncio.sleep(1)

    # Shutdown sequence
    logger.info("WebSocket broadcaster shutting down...")
    await ws_manager.close_all()


def create_app(boot_callback=None, input_callback=None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        boot_callback: Optional function to call on startup (loads models, etc.)
        input_callback: Optional function to start audio input

    Returns:
        Configured FastAPI application
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage application lifecycle."""
        # Startup
        broadcast_task = asyncio.create_task(websocket_broadcaster())

        if boot_callback:
            boot_callback()
        if input_callback:
            import threading
            threading.Thread(target=input_callback, daemon=True).start()

        yield

        # Shutdown
        if stop_event:
            stop_event.set()
        await ws_manager.close_all()
        broadcast_task.cancel()
        try:
            await broadcast_task
        except asyncio.CancelledError:
            pass

    tags_metadata = [
        {
            "name": "health",
            "description": "Health and status checks"
        },
        {
            "name": "sessions",
            "description": "Session management endpoints"
        },
        {
            "name": "realtime",
            "description": "Real-time WebSocket streaming"
        }
    ]

    app = FastAPI(
        title="Live Transcription API",
        version="1.0.0",
        description="Real-time audio transcription system with speaker diarization, multi-session support, and comparison features.",
        lifespan=lifespan,
        openapi_tags=tags_metadata
    )
    
    @app.get("/ping", response_model=HealthCheckResponse, tags=["health"])
    async def ping():
        """
        Health check endpoint.

        Returns the server status and number of active WebSocket connections.
        """
        return {
            "status": "ok",
            "connections": len(ws_manager.active_connections)
        }

    @app.get("/api/engine/status", response_model=EngineStatusResponse, tags=["health"])
    async def get_engine_status_endpoint():
        """
        Get the current transcription engine status.
        
        Returns details about the active processing device (CPU/GPU), 
        fallback status, and model configuration.
        """
        from transcription_engine import get_engine_status
        return get_engine_status()

    @app.websocket("/ws", name="websocket")
    async def websocket_endpoint(websocket: WebSocket):
        """
        WebSocket endpoint for real-time transcription updates.

        Streams two types of messages:
        - **JSON**: Transcript data with speaker, confidence, and timestamps
        - **Binary**: Raw audio chunks (Float32Array) for visualization

        The connection stays open until the client disconnects or the server shuts down.
        """
        logger.info("New WebSocket client connecting...")
        print("New WebSocket client connecting...")
        await ws_manager.connect(websocket)
        try:
            while True:
                # Keep connection alive (client doesn't send data)
                await websocket.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)

    # Session API endpoints
    @app.get("/api/session/current", response_model=CurrentSessionResponse, tags=["sessions"])
    async def get_current_session():
        """
        Get the current active session's data.

        Returns the currently active transcription session or an empty response if none is active.
        """
        from session import get_session
        session = get_session()
        if session:
            return {
                "active": True,
                "name": session.name,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "transcripts": session.transcripts
            }
        return {"active": False}

    @app.get("/api/sessions", response_model=SessionListResponse, tags=["sessions"])
    async def list_sessions_endpoint():
        """
        List all saved sessions.

        Returns metadata for all saved sessions including name, creation time, and transcript count.
        Sessions are sorted by most recently updated first.
        """
        from session import list_sessions
        return {"sessions": list_sessions()}

    @app.post("/api/sessions", response_model=SessionCreateResponse, tags=["sessions"])
    async def create_session_endpoint(name: str = None):
        """
        Create a new session.

        Creates and saves a new transcription session with an optional custom name.
        If no name is provided, a timestamp-based name is auto-generated.

        **Parameters:**
        - `name`: Optional session name (defaults to Session_YYYYMMDD_HHMMSS)
        """
        from session import init_session, save_session
        session = init_session(session_name=name)
        save_session(session)
        return {"name": session.name, "created_at": session.created_at}

    @app.get("/api/sessions/{name}", response_model=SessionFull, tags=["sessions"])
    async def get_session_endpoint(name: str):
        """
        Get a session's full data including all transcripts.

        Retrieves a saved session by name and returns all transcripts with metadata.

        **Parameters:**
        - `name`: Session name to retrieve

        **Raises:**
        - 404: Session not found
        """
        from session import load_session_from_file
        try:
            session = load_session_from_file(name)
            return session.to_dict()
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Session not found: {name}")

    @app.post("/api/sessions/{name}/save", response_model=SaveSessionResponse, tags=["sessions"])
    async def save_session_endpoint(name: str):
        """
        Save the current active session with a new name.

        Persists the currently active session to disk with the specified name.

        **Parameters:**
        - `name`: New session name

        **Raises:**
        - 400: No active session
        """
        from session import get_session, save_session
        session = get_session()
        if not session:
            raise HTTPException(status_code=400, detail="No active session")
        path = save_session(session, name)
        return {"saved": True, "path": path}

    @app.delete("/api/sessions/{name}", response_model=DeleteSessionResponse, tags=["sessions"])
    async def delete_session_endpoint(name: str):
        """
        Delete a saved session.

        Permanently removes a session file from disk.

        **Parameters:**
        - `name`: Session name to delete

        **Raises:**
        - 404: Session not found
        """
        from session import delete_session
        deleted = delete_session(name)
        if deleted:
            return {"deleted": True}
        else:
            raise HTTPException(status_code=404, detail=f"Session not found: {name}")

    @app.get("/api/sessions/archived", response_model=SessionListResponse, tags=["sessions"])
    async def list_archived_sessions_endpoint():
        """
        List all archived sessions.

        Returns metadata for all archived sessions including name, creation time, and transcript count.
        Sessions are sorted by most recently updated first.
        """
        from session import list_archived_sessions
        return {"sessions": list_archived_sessions()}

    @app.post("/api/sessions/{name}/archive", response_model=ArchiveSessionResponse, tags=["sessions"])
    async def archive_session_endpoint(name: str):
        """
        Archive a saved session.

        Moves a session from the active sessions folder to the archive folder.
        Archived sessions are preserved but separated from active sessions.

        **Parameters:**
        - `name`: Session name to archive

        **Raises:**
        - 404: Session not found
        """
        from session import archive_session, SESSIONS_DIR
        from pathlib import Path

        archived = archive_session(name)
        if archived:
            safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
            archive_path = SESSIONS_DIR / "archive" / f"{safe_name}.json"
            return {
                "name": name,
                "archived": True,
                "path": str(archive_path)
            }
        else:
            raise HTTPException(status_code=404, detail=f"Session not found: {name}")

    @app.post("/api/sessions/{name}/restore", response_model=RestoreSessionResponse, tags=["sessions"])
    async def restore_session_endpoint(name: str):
        """
        Restore an archived session.

        Moves a session from the archive folder back to the active sessions folder.

        **Parameters:**
        - `name`: Session name to restore

        **Raises:**
        - 404: Archived session not found
        """
        from session import restore_session, SESSIONS_DIR

        restored = restore_session(name)
        if restored:
            safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
            file_path = SESSIONS_DIR / f"{safe_name}.json"
            return {
                "name": name,
                "restored": True,
                "path": str(file_path)
            }
        else:
            raise HTTPException(status_code=404, detail=f"Archived session not found: {name}")

    @app.get("/api/session/rollover-status", response_model=SessionRolloverStatus, tags=["sessions"])
    async def get_rollover_status():
        """
        Get the current session's rollover timer status.

        Returns information about rollover thresholds and time remaining before automatic session rollover.
        Useful for monitoring deployment health and predicting when sessions will rotate.

        If no session is active, returns default zero values.
        """
        from session import get_session_rollover_status

        return get_session_rollover_status()

    @app.post("/api/sessions/compare", response_model=SessionComparisonResponse, tags=["sessions"])
    async def compare_sessions_endpoint(request: SessionComparisonRequest):
        """
        Compare multiple sessions side-by-side or merged.

        Compares 2-3 sessions and returns either:
        - **merged**: Chronological timeline of all transcripts sorted by timestamp
        - **side-by-side**: Transcripts grouped by session

        Optional speaker filtering limits results to specific speakers.

        **Parameters:**
        - `session_names`: List of 2-3 session names to compare
        - `mode`: Comparison mode ('merged' or 'side-by-side', default: 'merged')
        - `speaker_filter`: Optional list of speaker names to include

        **Raises:**
        - 400: Invalid number of sessions or sessions not found
        - 422: Invalid request format
        """
        from comparison import compare_sessions

        try:
            result = compare_sessions(
                request.session_names,
                request.mode,
                request.speaker_filter
            )
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Audio API endpoints
    @app.get("/api/audio/buffer-status", tags=["audio"])
    async def get_audio_buffer_status():
        """
        Get current audio buffer state and statistics.

        Returns information about the audio buffer including window size,
        current data available, and duration of audio stored.

        **Returns:**
        - `window_size_sec`: Configured time window in seconds
        - `buffer_size_bytes`: Current total buffer size in bytes
        - `buffer_size_mb`: Current total buffer size in MB
        - `oldest_timestamp`: Earliest audio timestamp (Unix float)
        - `newest_timestamp`: Latest audio timestamp (Unix float)
        - `num_chunks`: Number of audio chunks in buffer
        - `duration_available_sec`: Total duration of audio available
        """
        try:
            from audio_buffer import audio_buffer
            stats = audio_buffer.get_buffer_stats()
            return stats
        except Exception as e:
            logger.error(f"Error getting buffer status: {e}")
            raise HTTPException(status_code=500, detail="Failed to get buffer status")

    @app.get("/api/audio/range", tags=["audio"])
    async def get_audio_range(start_time: float, end_time: float, format: str = "wav"):
        """
        Retrieve audio for a specific time range.

        Fetches audio data from the backend buffer for the specified time range.
        Audio outside the configured buffer window is not available.

        **Parameters:**
        - `start_time`: Start timestamp (Unix float, seconds)
        - `end_time`: End timestamp (Unix float, seconds)
        - `format`: Output format - "wav" (default) or "raw" (Float32 bytes)

        **Returns:**
        - `format=wav`: WAV audio file (audio/wav)
        - `format=raw`: Raw Float32 audio bytes (application/octet-stream)

        **Raises:**
        - 400: Invalid time range or no audio available
        - 500: Internal server error
        """
        try:
            from audio_buffer import audio_buffer

            logger.info(f"Audio range request: [{start_time:.3f}, {end_time:.3f}] format={format}")

            # Validate time range
            if end_time <= start_time:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid time range: end_time must be > start_time"
                )

            # Get audio from buffer
            try:
                audio_data = audio_buffer.get_audio_range(start_time, end_time)
                rms = np.sqrt(np.mean(audio_data**2))
                max_val = np.max(np.abs(audio_data))
                logger.info(f"Audio retrieved: {len(audio_data)} samples ({len(audio_data)/16000:.3f}s), rms={rms:.4f}, max={max_val:.4f}")

                # Check if audio is actually silent
                if rms < 0.0001:
                    logger.warning(f"WARNING: Retrieved audio is essentially silent (rms={rms:.6f})")

                # Check buffer status
                stats = audio_buffer.get_buffer_stats()
                logger.info(f"Buffer stats: {stats['num_chunks']} chunks, duration={stats['duration_available_sec']:.1f}s, oldest={stats.get('oldest_timestamp', '?')}")

            except ValueError as e:
                logger.warning(f"Audio retrieval failed: {e}")
                raise HTTPException(status_code=400, detail=str(e))

            # Return in requested format
            if format.lower() == "wav":
                # Convert to WAV format
                sample_rate = 16000
                wav_bytes = create_wav_from_float32(audio_data, sample_rate)
                return Response(
                    content=wav_bytes,
                    media_type="audio/wav",
                    headers={
                        "Content-Disposition": f"attachment; filename=audio_{start_time:.0f}.wav"
                    }
                )
            elif format.lower() == "raw":
                # Return raw Float32 bytes
                raw_bytes = audio_data.astype(np.float32).tobytes()
                return Response(
                    content=raw_bytes,
                    media_type="application/octet-stream",
                    headers={
                        "Content-Disposition": f"attachment; filename=audio_{start_time:.0f}.raw"
                    }
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid format: {format}. Use 'wav' or 'raw'."
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving audio range: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve audio")

    # Mount static files last (catch-all)
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
    
    return app
