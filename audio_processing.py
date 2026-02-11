"""
Audio quality improvements module.

Provides advanced audio processing techniques including:
- Silero VAD (Voice Activity Detection)
- Spectral noise reduction
- RMS-based normalization
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Global VAD model instance
_silero_vad_model = None
_silero_vad_utils = None


def initialize_vad():
    """
    Initialize Silero VAD model.

    Returns:
        tuple: (model, get_speech_timestamps) or (None, None) if initialization fails
    """
    global _silero_vad_model, _silero_vad_utils

    if _silero_vad_model is not None:
        return _silero_vad_model, _silero_vad_utils

    try:
        import torch
        _silero_vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        _silero_vad_utils = utils
        logger.info("Silero VAD model loaded successfully")
        return _silero_vad_model, utils
    except Exception as e:
        logger.warning(f"Failed to load Silero VAD: {e}")
        return None, None


def is_speech(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    confidence_threshold: float = 0.5
) -> tuple:
    """
    Detect if audio contains speech using Silero VAD.

    Args:
        audio_data: Audio samples (float32, mono)
        sample_rate: Sample rate in Hz (default: 16000)
        confidence_threshold: Minimum confidence to classify as speech (default: 0.5)

    Returns:
        tuple: (has_speech: bool, confidence: float)
              - has_speech: True if speech detected with confidence >= threshold
              - confidence: Confidence score (0.0 to 1.0)
    """
    model, utils = initialize_vad()
    if model is None or utils is None:
        # Fallback: return None to indicate VAD unavailable
        return None, None

    try:
        import torch

        get_speech_timestamps = utils[0]

        # Convert numpy to torch tensor
        if isinstance(audio_data, np.ndarray):
            audio_tensor = torch.from_numpy(audio_data).float()
        else:
            audio_tensor = audio_data

        # Ensure proper shape and dtype
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Get speech timestamps
        speech_dict = get_speech_timestamps(
            audio_tensor,
            model,
            sampling_rate=sample_rate,
            threshold=confidence_threshold
        )

        # Calculate confidence based on speech presence
        if isinstance(speech_dict, dict):
            # New format returns dict with 'positive_probability'
            confidence = float(speech_dict.get('positive_probability', 0.0))
        elif isinstance(speech_dict, list):
            # Old format returns list of speech segments
            # Confidence based on number of segments and duration
            if len(speech_dict) > 0:
                total_speech_duration = sum(seg['end'] - seg['start'] for seg in speech_dict)
                total_duration = len(audio_data) / sample_rate
                confidence = min(1.0, total_speech_duration / max(total_duration, 0.001))
            else:
                confidence = 0.0
        else:
            confidence = 0.0

        has_speech = confidence >= confidence_threshold

        return has_speech, confidence

    except Exception as e:
        logger.debug(f"VAD detection failed: {e}")
        return None, None


def reduce_noise(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    stationary: bool = True,
    prop_decrease: float = 1.0
) -> np.ndarray:
    """
    Apply spectral subtraction noise reduction.

    Uses noisereduce library for production-grade noise reduction
    suitable for radio/dispatch audio.

    Args:
        audio_data: Audio samples (float32, mono)
        sample_rate: Sample rate in Hz (default: 16000)
        stationary: Use stationary noise reduction (default: True)
        prop_decrease: Proportion of noise to reduce (default: 1.0)
                      Range: 0.0 to 2.0 (higher = more aggressive)

    Returns:
        np.ndarray: Noise-reduced audio
    """
    try:
        import noisereduce as nr

        # Apply noise reduction
        reduced = nr.reduce_noise(
            y=audio_data,
            sr=sample_rate,
            stationary=stationary,
            prop_decrease=prop_decrease
        )

        return reduced

    except ImportError:
        logger.warning("noisereduce library not available, skipping noise reduction")
        return audio_data
    except Exception as e:
        logger.debug(f"Noise reduction failed: {e}")
        return audio_data


def normalize_rms(
    audio_data: np.ndarray,
    target_rms: float = 0.1
) -> np.ndarray:
    """
    Apply RMS (Root Mean Square) normalization.

    Provides more consistent loudness normalization than peak normalization,
    especially useful for variable audio sources like radio broadcasts.

    Args:
        audio_data: Audio samples (float32)
        target_rms: Target RMS level (default: 0.1)
                   Range: 0.01 to 0.5 (higher = louder)

    Returns:
        np.ndarray: RMS-normalized audio
    """
    try:
        # Calculate current RMS
        current_rms = np.sqrt(np.mean(audio_data ** 2))

        # Avoid division by zero
        if current_rms > 1e-6:
            # Scale to target RMS
            normalized = audio_data * (target_rms / current_rms)

            # Soft clipping to prevent distortion (very rare)
            # Using a simple tanh-based soft clipping
            normalized = np.tanh(normalized)

            return normalized

        return audio_data

    except Exception as e:
        logger.debug(f"RMS normalization failed: {e}")
        return audio_data


def apply_audio_enhancements(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    enable_vad: bool = False,
    vad_confidence_threshold: float = 0.5,
    enable_noise_reduction: bool = False,
    noise_reduction_stationary: bool = True,
    normalization_method: str = "peak",
    rms_target_level: float = 0.1
) -> np.ndarray:
    """
    Apply all audio enhancements in sequence.

    Convenience function that applies configured enhancements.

    Args:
        audio_data: Input audio samples
        sample_rate: Sample rate in Hz
        enable_vad: Enable VAD pre-filtering (removes non-speech)
        vad_confidence_threshold: VAD confidence threshold
        enable_noise_reduction: Enable spectral noise reduction
        noise_reduction_stationary: Use stationary noise reduction
        normalization_method: "peak" or "rms"
        rms_target_level: Target RMS level if using RMS normalization

    Returns:
        np.ndarray: Enhanced audio
    """
    result = audio_data.copy()

    # 1. VAD filtering (removes non-speech segments)
    if enable_vad:
        has_speech, confidence = is_speech(
            result,
            sample_rate=sample_rate,
            confidence_threshold=vad_confidence_threshold
        )

        if has_speech is False:
            # No speech detected, return silence
            return np.zeros_like(result)

    # 2. Noise reduction
    if enable_noise_reduction:
        result = reduce_noise(
            result,
            sample_rate=sample_rate,
            stationary=noise_reduction_stationary,
            prop_decrease=1.0
        )

    # 3. Normalization
    if normalization_method == "rms":
        result = normalize_rms(result, target_rms=rms_target_level)
    else:
        # Default: peak normalization (existing behavior)
        max_val = np.max(np.abs(result))
        if max_val > 0.01:
            result = result / max_val

    return result


def cleanup_vad():
    """Clean up VAD model resources."""
    global _silero_vad_model, _silero_vad_utils

    try:
        if _silero_vad_model is not None:
            del _silero_vad_model
            _silero_vad_model = None
        if _silero_vad_utils is not None:
            del _silero_vad_utils
            _silero_vad_utils = None
        logger.info("VAD model cleaned up")
    except Exception as e:
        logger.debug(f"VAD cleanup error: {e}")
