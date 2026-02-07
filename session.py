"""
Session management for the live transcription application.

This module provides the TranscriptionSession class which encapsulates
all runtime state for a transcription session, replacing scattered globals.
"""

import os
import json
import queue
import threading
import logging
from datetime import datetime
from pathlib import Path

from config import load_config, validate_config, TRANSCRIPTION_CONFIG

logger = logging.getLogger(__name__)

# Session storage directory
SESSIONS_DIR = Path(__file__).parent / "sessions"


class TranscriptionSession:
    """
    Encapsulates all runtime state for a transcription session.
    
    This class replaces the various global variables and function attributes
    that were previously scattered across modules, making the code easier
    to test and enabling multiple concurrent sessions if needed.
    """
    
    def __init__(self, config_path: str = None, session_name: str = None):
        """
        Initialize a new transcription session.
        
        Args:
            config_path: Optional path to config.json file
            session_name: Optional name for this session
        """
        # Session identity
        self.name = session_name or f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        
        # Thread-safe queues for audio processing pipeline
        self.audio_queue = queue.Queue()
        self.broadcast_queue = queue.Queue()
        self.audio_broadcast_queue = queue.Queue()
        
        # Control event for graceful shutdown
        self.stop_event = threading.Event()
        
        # Audio state
        self.current_volume = 0.0
        
        # Transcription state
        self.last_transcript = None
        self.chunk_count = 0
        self.transcripts = []  # List of transcript records
        
        # Speaker identification (loaded on demand)
        self.speaker_model = None
        self.speaker_manager = None
        
        # Output file handle
        self.output_file = None
        
        # Configuration
        self.config_path = config_path
        self.config_errors = []
        
        if config_path:
            self._load_and_validate_config()
    
    def _load_and_validate_config(self):
        """Load and validate configuration from file."""
        load_config(self.config_path)
        self.config_errors = validate_config()
        
        if self.config_errors:
            logger.warning(f"Config has {len(self.config_errors)} validation issue(s)")
    
    def is_valid(self) -> bool:
        """Check if the session configuration is valid."""
        return len(self.config_errors) == 0
    
    def get_config_errors(self) -> list:
        """Get list of configuration validation errors."""
        return self.config_errors
    
    def add_transcript(self, transcript: dict):
        """Add a transcript record to the session."""
        self.transcripts.append(transcript)
        
        # Cap in-memory transcripts to prevent unbounded growth
        # 2000 is a generous limit that stays well within safe memory bounds
        if len(self.transcripts) > 2000:
            self.transcripts.pop(0)
            
        self.updated_at = datetime.now().isoformat()
    
    def load_speaker_model(self):
        """
        Load the speaker embedding model for diarization.
        
        This is done on-demand to avoid loading heavy models if not needed.
        """
        try:
            from speechbrain.pretrained import EncoderClassifier
            from speaker_manager import SpeakerManager
            
            logger.info("Loading speaker embedding model...")
            print("Loading speaker embedding model...")
            
            self.speaker_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="/tmp/speaker_model"
            )
            self.speaker_manager = SpeakerManager()
            
            logger.info("Speaker model loaded successfully")
            print("Speaker model ready. Diarization enabled.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load speaker model: {e}")
            print(f"⚠️  Speaker model failed to load: {e}")
            print("Diarization will be disabled.")
            return False
    
    def stop(self):
        """Signal all threads to stop processing."""
        self.stop_event.set()
    
    def is_stopped(self) -> bool:
        """Check if stop has been signaled."""
        return self.stop_event.is_set()
    
    def set_output_file(self, file_path: str):
        """Set the output file for transcription text."""
        self.output_file = file_path
    
    def update_volume(self, volume: float):
        """Update the current audio volume level."""
        self.current_volume = volume
    
    def get_volume(self) -> float:
        """Get the current audio volume level."""
        return self.current_volume
    
    def to_dict(self) -> dict:
        """Convert session to serializable dictionary."""
        return {
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "transcript_count": len(self.transcripts),
            "transcripts": self.transcripts
        }
    
    @classmethod
    def from_dict(cls, data: dict, config_path: str = None):
        """Create session from dictionary data."""
        session = cls(config_path=config_path, session_name=data.get("name"))
        session.created_at = data.get("created_at", session.created_at)
        session.updated_at = data.get("updated_at", session.updated_at)
        session.transcripts = data.get("transcripts", [])
        return session


# Module-level session instance for backwards compatibility
_session = None


def get_session() -> TranscriptionSession:
    """Get the current session instance."""
    return _session


def init_session(config_path: str = None, session_name: str = None) -> TranscriptionSession:
    """
    Initialize and return a new session.
    
    Args:
        config_path: Optional path to config.json
        session_name: Optional name for the session
        
    Returns:
        The newly created TranscriptionSession
    """
    global _session
    _session = TranscriptionSession(config_path, session_name)
    return _session


def save_session(session: TranscriptionSession = None, name: str = None) -> str:
    """
    Save a session to disk.
    
    Args:
        session: Session to save (defaults to current session)
        name: Optional name override
        
    Returns:
        Path to saved session file
    """
    session = session or _session
    if not session:
        raise ValueError("No session to save")
    
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    save_name = name or session.name
    # Sanitize filename
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in save_name)
    file_path = SESSIONS_DIR / f"{safe_name}.json"
    
    session.name = save_name
    session.updated_at = datetime.now().isoformat()
    
    with open(file_path, "w") as f:
        json.dump(session.to_dict(), f, indent=2)
    
    logger.info(f"Session saved to {file_path}")
    return str(file_path)


def load_session_from_file(name: str, config_path: str = None) -> TranscriptionSession:
    """
    Load a session from disk.
    
    Args:
        name: Session name to load
        config_path: Optional config path
        
    Returns:
        Loaded TranscriptionSession
    """
    global _session
    
    # Sanitize filename
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
    file_path = SESSIONS_DIR / f"{safe_name}.json"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Session not found: {name}")
    
    with open(file_path, "r") as f:
        data = json.load(f)
    
    _session = TranscriptionSession.from_dict(data, config_path)
    logger.info(f"Session loaded: {name} ({len(_session.transcripts)} transcripts)")
    return _session


def list_sessions() -> list:
    """
    List all saved sessions.
    
    Returns:
        List of session metadata dicts
    """
    if not SESSIONS_DIR.exists():
        return []
    
    sessions = []
    for file_path in SESSIONS_DIR.glob("*.json"):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            sessions.append({
                "name": data.get("name", file_path.stem),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "transcript_count": data.get("transcript_count", 0),
                "file": file_path.name
            })
        except Exception as e:
            logger.warning(f"Failed to read session {file_path}: {e}")
    
    # Sort by updated_at descending
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return sessions


def delete_session(name: str) -> bool:
    """
    Delete a saved session.
    
    Args:
        name: Session name to delete
        
    Returns:
        True if deleted, False if not found
    """
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
    file_path = SESSIONS_DIR / f"{safe_name}.json"
    
    if file_path.exists():
        file_path.unlink()
        logger.info(f"Session deleted: {name}")
        return True
    return False
