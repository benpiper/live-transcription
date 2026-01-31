"""
Session management for the live transcription application.

This module provides the TranscriptionSession class which encapsulates
all runtime state for a transcription session, replacing scattered globals.
"""

import queue
import threading
import logging

from config import load_config, validate_config, TRANSCRIPTION_CONFIG

logger = logging.getLogger(__name__)


class TranscriptionSession:
    """
    Encapsulates all runtime state for a transcription session.
    
    This class replaces the various global variables and function attributes
    that were previously scattered across modules, making the code easier
    to test and enabling multiple concurrent sessions if needed.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize a new transcription session.
        
        Args:
            config_path: Optional path to config.json file
        """
        # Thread-safe queues for audio processing pipeline
        self.audio_queue = queue.Queue()
        self.broadcast_queue = queue.Queue()
        self.audio_broadcast_queue = queue.Queue()
        
        # Control event for graceful shutdown
        self.stop_event = threading.Event()
        
        # Audio state
        self.current_volume = 0.0
        
        # Transcription state (previously function attributes)
        self.last_transcript = None
        self.chunk_count = 0
        
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


# Module-level session instance for backwards compatibility
# This allows existing code to import and use a shared session
_session = None


def get_session() -> TranscriptionSession:
    """
    Get the current session instance.
    
    Returns:
        The active TranscriptionSession, or None if not initialized
    """
    return _session


def init_session(config_path: str = None) -> TranscriptionSession:
    """
    Initialize and return a new session.
    
    Args:
        config_path: Optional path to config.json
        
    Returns:
        The newly created TranscriptionSession
    """
    global _session
    _session = TranscriptionSession(config_path)
    return _session
