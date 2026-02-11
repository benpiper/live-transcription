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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

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
    
    app = FastAPI(lifespan=lifespan)
    
    @app.get("/ping")
    async def ping():
        """Health check endpoint."""
        return {
            "status": "ok", 
            "connections": len(ws_manager.active_connections)
        }
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates."""
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
    @app.get("/api/session/current")
    async def get_current_session():
        """Get the current active session's data."""
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
    
    @app.get("/api/sessions")
    async def list_sessions_endpoint():
        """List all saved sessions."""
        from session import list_sessions
        return {"sessions": list_sessions()}
    
    @app.post("/api/sessions")
    async def create_session_endpoint(name: str = None):
        """Create a new session."""
        from session import init_session, save_session
        session = init_session(session_name=name)
        save_session(session)
        return {"name": session.name, "created_at": session.created_at}
    
    @app.get("/api/sessions/{name}")
    async def get_session_endpoint(name: str):
        """Get a session's transcripts."""
        from session import load_session_from_file
        try:
            session = load_session_from_file(name)
            return session.to_dict()
        except FileNotFoundError:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Session not found: {name}")
    
    @app.post("/api/sessions/{name}/save")
    async def save_session_endpoint(name: str):
        """Save the current session."""
        from session import get_session, save_session
        session = get_session()
        if not session:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="No active session")
        path = save_session(session, name)
        return {"saved": True, "path": path}
    
    @app.delete("/api/sessions/{name}")
    async def delete_session_endpoint(name: str):
        """Delete a saved session."""
        from session import delete_session
        deleted = delete_session(name)
        if deleted:
            return {"deleted": True}
        else:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Session not found: {name}")
    
    # Mount static files last (catch-all)
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
    
    return app
