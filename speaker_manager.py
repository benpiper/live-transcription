"""
Speaker identification and diarization management.

This module provides the SpeakerManager class for identifying and tracking
different speakers in audio, as well as detecting robotic/synthetic voices.
"""

import threading
import logging
import numpy as np
import torch
import librosa

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
    
    Compares new audio embeddings against known speakers and assigns
    labels. Optionally detects robotic/synthetic voices.
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
    
    def is_robotic_voice(self, audio_chunk: np.ndarray) -> tuple:
        """
        Analyze an audio chunk to detect robotic/synthetic voice patterns.
        
        Uses pitch variability and spectral flatness to identify TTS voices.
        
        Args:
            audio_chunk: Raw audio samples (float32, 16kHz)
            
        Returns:
            Tuple of (is_robotic, f0_mean, f0_std, spectral_flatness)
        """
        if audio_chunk is None or len(audio_chunk) < 4000:
            return False, 0, 0, 0
        
        try:
            # Extract fundamental frequency (pitch)
            f0, voiced_flag, _ = librosa.pyin(
                audio_chunk, 
                fmin=60, 
                fmax=600, 
                sr=16000
            )
            voiced_f0 = f0[voiced_flag]
            
            if len(voiced_f0) < 5:
                return False, 0, 0, 0
            
            f0_mean = np.nanmean(voiced_f0)
            f0_std = np.nanstd(voiced_f0)
            
            # Spectral flatness (how "noisy" vs "tonal" the signal is)
            flatness = librosa.feature.spectral_flatness(y=audio_chunk, n_fft=512)
            avg_flatness = float(np.mean(flatness))
            
            # Get configurable thresholds from config
            settings = TRANSCRIPTION_CONFIG["settings"]
            pitch_std_threshold = settings.get("robot_pitch_std_threshold", 8.0)  # Lower = stricter
            flatness_threshold = settings.get("robot_flatness_threshold", 0.12)   # Higher = stricter
            debug_robo = settings.get("debug_robo", False)
            
            # Robotic detection heuristics:
            # High-quality TTS often has:
            # 1. Unnaturally CLEAN audio (low flatness - no breath/lip noise)
            # 2. OR very low pitch variance (monotone)
            # Human speech has natural noise from breathing, lip sounds, etc.
            too_clean = avg_flatness < flatness_threshold  # Flipped: LOW flatness = synthetic
            low_pitch_variance = f0_std < pitch_std_threshold
            is_robotic = too_clean or low_pitch_variance
            
            # Verbose debug output
            if debug_robo:
                import sys
                pitch_marker = "✗" if low_pitch_variance else "✓"
                clean_marker = "✗" if too_clean else "✓"
                result = "🤖 BOT" if is_robotic else "👤 HUMAN"
                sys.stdout.write(f"\r" + " " * 80 + "\r")
                print(f"[RoboDebug] pitch_std={f0_std:.1f}Hz {pitch_marker} | flatness={avg_flatness:.3f} (too_clean:{clean_marker}) → {result}")
            
            logger.debug(
                f"RoboCheck: pitch_std={f0_std:.2f} (thresh:{pitch_std_threshold}), "
                f"flatness={avg_flatness:.4f} (thresh:{flatness_threshold}), "
                f"result={'BOT' if is_robotic else 'HUMAN'}"
            )
            
            return is_robotic, float(f0_mean), float(f0_std), avg_flatness
            
        except Exception as e:
            logger.debug(f"Robotic voice detection failed: {e}")
            return False, 0, 0, 0

    
    def identify_speaker(
        self, 
        embedding: torch.Tensor, 
        audio_chunk: np.ndarray = None
    ) -> str:
        """
        Identify or create a speaker label for the given embedding.
        
        Args:
            embedding: Voice embedding tensor from SpeechBrain
            audio_chunk: Optional raw audio for robotic detection
            
        Returns:
            Speaker label string (e.g., "Speaker 1", "Dispatcher (Bot)")
        """
        settings = TRANSCRIPTION_CONFIG["settings"]
        do_bot_detection = settings.get("detect_bots", False)
        current_threshold = settings.get("diarization_threshold", self.threshold)
        
        with self._lock:
            if not self.speakers:
                label = "Speaker 1"
                
                # Bot detection for first speaker
                if do_bot_detection and audio_chunk is not None:
                    is_robo, *_ = self.is_robotic_voice(audio_chunk)
                    if is_robo:
                        label = "Dispatcher (Bot)"
                
                self.speakers.append((embedding, label))
                return label
            
            # Compare with known speakers
            from torch.nn.functional import cosine_similarity
            max_sim = -1
            best_label = None
            best_idx = -1
            
            for i, (spk_emb, label) in enumerate(self.speakers):
                sim = cosine_similarity(
                    embedding.unsqueeze(0), 
                    spk_emb.unsqueeze(0)
                ).item()
                if sim > max_sim:
                    max_sim = sim
                    best_label = label
                    best_idx = i
            
            if max_sim > current_threshold:
                # Re-evaluate: upgrade generic speaker to bot if detected
                if do_bot_detection and best_label.startswith("Speaker"):
                    if audio_chunk is not None:
                        is_robo, *_ = self.is_robotic_voice(audio_chunk)
                        if is_robo:
                            self.speakers[best_idx] = (
                                self.speakers[best_idx][0], 
                                "Dispatcher (Bot)"
                            )
                            return "Dispatcher (Bot)"
                return best_label
            else:
                # New speaker
                new_index = len(self.speakers) + 1
                new_label = f"Speaker {new_index}"
                
                # Check if new speaker is robotic
                if do_bot_detection and audio_chunk is not None:
                    is_robo, *_ = self.is_robotic_voice(audio_chunk)
                    if is_robo:
                        new_label = "Dispatcher (Bot)"
                
                self.speakers.append((embedding, new_label))
                return new_label
    
    def reset(self):
        """Clear all known speakers."""
        with self._lock:
            self.speakers.clear()
            logger.info("Speaker manager reset")


# Global singleton instance
speaker_manager = SpeakerManager()
