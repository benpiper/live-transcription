"""
Speaker identification and diarization management.

This module provides the SpeakerManager class for identifying and tracking
different speakers in audio using voice embeddings and optional pre-registered
voice profiles.
"""

import threading
import logging
import numpy as np
import torch
from typing import Optional

from config import TRANSCRIPTION_CONFIG

logger = logging.getLogger(__name__)

# Default minimum samples for speaker identification (1s at 16kHz)
# This can be overridden via config.json settings.min_speaker_samples
MIN_SPEAKER_ID_SAMPLES = 16000


def get_min_speaker_samples() -> int:
    """Get minimum samples for speaker ID from config."""
    return TRANSCRIPTION_CONFIG["settings"].get("min_speaker_samples", MIN_SPEAKER_ID_SAMPLES)


class SpeakerManager:
    """
    Manages speaker identification using voice embeddings.
    
    Compares new audio embeddings against:
    1. Pre-registered voice profiles (for known speakers)
    2. Dynamically discovered speakers (for unknown voices)
    """
    
    def __init__(self, threshold: float = 0.35):
        """
        Initialize the speaker manager.
        
        Args:
            threshold: Cosine similarity threshold for matching speakers.
                      Lower values = more inclusive (fewer distinct speakers).
        """
        self.threshold = threshold
        self.speakers = []  # List of (embedding, label)
        self._lock = threading.Lock()
        self._profile_manager = None
    
    @property
    def profile_manager(self):
        """Lazy-load the voice profile manager."""
        if self._profile_manager is None:
            from voice_profiles import voice_profile_manager
            self._profile_manager = voice_profile_manager
        return self._profile_manager
    
    def identify_speaker(
        self, 
        embedding: torch.Tensor, 
        audio_chunk: np.ndarray = None
    ) -> str:
        """
        Identify or create a speaker label for the given embedding.
        
        First checks against registered voice profiles, then falls back
        to dynamic speaker clustering.
        
        Args:
            embedding: Voice embedding tensor from SpeechBrain
            audio_chunk: Raw audio (unused, kept for API compatibility)
            
        Returns:
            Speaker label string (e.g., "Speaker 1", "Dispatcher A")
        """
        settings = TRANSCRIPTION_CONFIG["settings"]
        current_threshold = settings.get("diarization_threshold", self.threshold)
        
        # Step 1: Check against registered voice profiles
        profile_match = self.profile_manager.match(embedding)
        if profile_match:
            logger.debug(f"Matched voice profile: {profile_match}")
            return profile_match
        
        # Step 2: Fall back to dynamic speaker clustering
        with self._lock:
            if not self.speakers:
                label = "Speaker 1"
                self.speakers.append((embedding, label))
                return label
            
            # Compare with known speakers
            from torch.nn.functional import cosine_similarity
            max_sim = -1
            best_label = None
            
            for spk_emb, label in self.speakers:
                sim = cosine_similarity(
                    embedding.unsqueeze(0), 
                    spk_emb.unsqueeze(0)
                ).item()
                if sim > max_sim:
                    max_sim = sim
                    best_label = label
            
            if max_sim > current_threshold:
                return best_label
            else:
                # New speaker
                new_index = len(self.speakers) + 1
                new_label = f"Speaker {new_index}"
                self.speakers.append((embedding, new_label))
                return new_label
    
    def reset(self):
        """Clear all dynamically discovered speakers."""
        with self._lock:
            self.speakers.clear()
            logger.info("Speaker manager reset")
    
    def reload_profiles(self):
        """Reload voice profiles from disk."""
        count = self.profile_manager.load_profiles()
        logger.info(f"Reloaded {count} voice profiles")
        return count


# Global singleton instance
speaker_manager = SpeakerManager()
