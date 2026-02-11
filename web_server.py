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
from fastapi.staticfiles import StaticFiles

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
    
    # Mount static files last (catch-all)
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
    
    return app
