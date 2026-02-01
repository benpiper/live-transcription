"""
Voice profile management for speaker identification.

This module provides the VoiceProfileManager class for managing pre-registered
voice profiles (embeddings) that can be matched against incoming audio.
"""

import os
import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, List
from torch.nn.functional import cosine_similarity

from config import TRANSCRIPTION_CONFIG

logger = logging.getLogger(__name__)


class VoiceProfileManager:
    """
    Manages voice profiles for speaker identification.
    
    Profiles are stored as JSON files containing voice embeddings extracted
    from sample audio. During transcription, incoming audio is compared against
    these profiles to identify known speakers.
    """
    
    def __init__(self, profiles_dir: str = None):
        """
        Initialize the voice profile manager.
        
        Args:
            profiles_dir: Directory containing voice profile JSON files.
                         Defaults to config setting or "voice_profiles".
        """
        settings = TRANSCRIPTION_CONFIG.get("settings", {})
        self.profiles_dir = Path(profiles_dir or settings.get("voice_profiles_dir", "voice_profiles"))
        self.threshold = settings.get("voice_match_threshold", 0.7)
        self.profiles: Dict[str, torch.Tensor] = {}  # name -> embedding
        
        # Create directory if it doesn't exist
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing profiles
        self.load_profiles()
    
    def load_profiles(self) -> int:
        """
        Load all voice profiles from the profiles directory.
        
        Returns:
            Number of profiles loaded.
        """
        self.profiles.clear()
        count = 0
        
        for profile_path in self.profiles_dir.glob("*.json"):
            try:
                with open(profile_path, 'r') as f:
                    data = json.load(f)
                
                name = data.get("name", profile_path.stem)
                embedding_list = data.get("embedding", [])
                
                if embedding_list:
                    embedding = torch.tensor(embedding_list, dtype=torch.float32)
                    self.profiles[name] = embedding
                    count += 1
                    logger.info(f"Loaded voice profile: {name}")
                    
            except Exception as e:
                logger.warning(f"Failed to load profile {profile_path}: {e}")
        
        logger.info(f"Loaded {count} voice profiles from {self.profiles_dir}")
        return count
    
    def match(self, embedding: torch.Tensor) -> Optional[str]:
        """
        Match an embedding against registered profiles.
        
        Args:
            embedding: Voice embedding tensor to match.
            
        Returns:
            Profile name if match found above threshold, else None.
        """
        if not self.profiles:
            return None
        
        # Get current threshold from config (allows runtime tuning)
        threshold = TRANSCRIPTION_CONFIG.get("settings", {}).get(
            "voice_match_threshold", self.threshold
        )
        
        best_match = None
        best_similarity = -1
        
        for name, profile_emb in self.profiles.items():
            similarity = cosine_similarity(
                embedding.unsqueeze(0),
                profile_emb.unsqueeze(0)
            ).item()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        if best_similarity >= threshold:
            logger.debug(f"Profile match: {best_match} (similarity: {best_similarity:.3f})")
            return best_match
        
        return None
    
    def enroll(
        self, 
        name: str, 
        embeddings: List[torch.Tensor],
        overwrite: bool = False
    ) -> bool:
        """
        Create a new voice profile from embeddings.
        
        Args:
            name: Profile name (will be used as speaker label).
            embeddings: List of embedding tensors from sample audio.
            overwrite: If True, overwrite existing profile with same name.
            
        Returns:
            True if profile was created successfully.
        """
        profile_path = self.profiles_dir / f"{name}.json"
        
        if profile_path.exists() and not overwrite:
            logger.warning(f"Profile '{name}' already exists. Use overwrite=True to replace.")
            return False
        
        if not embeddings:
            logger.error("No embeddings provided for enrollment")
            return False
        
        # Average multiple embeddings for robustness
        stacked = torch.stack(embeddings)
        averaged = torch.mean(stacked, dim=0)
        
        # Normalize the averaged embedding
        averaged = averaged / averaged.norm()
        
        profile_data = {
            "name": name,
            "embedding": averaged.tolist(),
            "sample_count": len(embeddings),
            "created": __import__("datetime").datetime.now().isoformat()
        }
        
        try:
            with open(profile_path, 'w') as f:
                json.dump(profile_data, f, indent=2)
            
            # Add to in-memory cache
            self.profiles[name] = averaged
            logger.info(f"Enrolled voice profile: {name} (from {len(embeddings)} samples)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save profile '{name}': {e}")
            return False
    
    def delete(self, name: str) -> bool:
        """
        Delete a voice profile.
        
        Args:
            name: Profile name to delete.
            
        Returns:
            True if profile was deleted successfully.
        """
        profile_path = self.profiles_dir / f"{name}.json"
        
        if not profile_path.exists():
            logger.warning(f"Profile '{name}' not found")
            return False
        
        try:
            profile_path.unlink()
            self.profiles.pop(name, None)
            logger.info(f"Deleted voice profile: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete profile '{name}': {e}")
            return False
    
    def list_profiles(self) -> List[str]:
        """Return list of registered profile names."""
        return list(self.profiles.keys())


# Global singleton instance
voice_profile_manager = VoiceProfileManager()
