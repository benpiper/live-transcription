"""
Configuration management for the live transcription application.

This module handles loading, validating, and providing access to
transcription settings from config.json files.
"""

import json
import logging

logger = logging.getLogger(__name__)

# Default configuration with all supported settings
DEFAULT_CONFIG = {
    "vocabulary": [],
    "corrections": {},
    "settings": {
        "device": "auto",
        "model_size": "medium.en",
        "compute_type": "auto",
        "no_speech_threshold": 0.8,
        "log_prob_threshold": -0.8,
        "compression_ratio_threshold": 2.4,
        "beam_size": 5,
        "min_silence_duration_ms": 500,
        "avg_logprob_cutoff": -0.8,
        "no_speech_prob_cutoff": 0.2,
        "extreme_confidence_cutoff": -0.4,
        "min_window_sec": 1.0,
        "max_window_sec": 5.0,
        "detect_bots": False,
        "cpu_threads": 4,
        "noise_floor": 0.001,
        "diarization_threshold": 0.35,
        "min_speaker_samples": 16000
    }
}

# Global config instance (will be populated by load_config)
TRANSCRIPTION_CONFIG = dict(DEFAULT_CONFIG)


def load_config(config_path: str) -> dict:
    """
    Load configuration from a JSON file and merge with defaults.
    
    Args:
        config_path: Path to the config.json file
        
    Returns:
        The merged configuration dictionary
    """
    global TRANSCRIPTION_CONFIG
    
    try:
        with open(config_path, "r") as f:
            user_config = json.load(f)
        
        # Merge vocabulary and corrections
        if "vocabulary" in user_config:
            TRANSCRIPTION_CONFIG["vocabulary"] = user_config["vocabulary"]
        if "corrections" in user_config:
            TRANSCRIPTION_CONFIG["corrections"] = user_config["corrections"]
        
        # Merge settings (preserve defaults for missing keys)
        if "settings" in user_config:
            TRANSCRIPTION_CONFIG["settings"].update(user_config["settings"])
        
        logger.info(f"Loaded config from {config_path}")
        print(f"Loaded config from {config_path}")
        
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        print(f"Config file not found: {config_path}. Using defaults.")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        print(f"Invalid JSON in config file: {e}. Using defaults.")
    
    return TRANSCRIPTION_CONFIG


def get_setting(key: str, default=None):
    """
    Get a specific setting value with optional default.
    
    Args:
        key: The setting key to retrieve
        default: Default value if key not found
        
    Returns:
        The setting value or default
    """
    return TRANSCRIPTION_CONFIG["settings"].get(key, default)


def get_vocabulary() -> list:
    """Get the vocabulary list for prompting."""
    return TRANSCRIPTION_CONFIG.get("vocabulary", [])


def get_corrections() -> dict:
    """Get the corrections dictionary for post-processing."""
    return TRANSCRIPTION_CONFIG.get("corrections", {})


def validate_config() -> list:
    """
    Validate the current configuration and return any errors/warnings.
    
    Checks types, ranges, and logical constraints.
    
    Returns:
        List of error/warning messages. Empty list means valid config.
    """
    errors = []
    settings = TRANSCRIPTION_CONFIG.get("settings", {})
    
    # Type validations
    type_checks = {
        "beam_size": int,
        "min_silence_duration_ms": int,
        "cpu_threads": int,
        "min_speaker_samples": int,
        "no_speech_threshold": (int, float),
        "log_prob_threshold": (int, float),
        "compression_ratio_threshold": (int, float),
        "avg_logprob_cutoff": (int, float),
        "no_speech_prob_cutoff": (int, float),
        "extreme_confidence_cutoff": (int, float),
        "min_window_sec": (int, float),
        "max_window_sec": (int, float),
        "noise_floor": (int, float),
        "diarization_threshold": (int, float),
        "detect_bots": bool,
        "debug_robo": bool,
        "device": str,
        "model_size": str,
        "compute_type": str,
    }
    
    for key, expected_type in type_checks.items():
        if key in settings:
            value = settings[key]
            if not isinstance(value, expected_type):
                errors.append(f"'{key}' should be {expected_type.__name__ if isinstance(expected_type, type) else 'number'}, got {type(value).__name__}")
    
    # Range validations (wrapped in try/except in case types are wrong)
    try:
        if settings.get("beam_size", 5) < 1 or settings.get("beam_size", 5) > 20:
            errors.append("beam_size should be between 1 and 20")
        
        if settings.get("cpu_threads", 4) < 1:
            errors.append("cpu_threads must be at least 1")
        
        if settings.get("noise_floor", 0.001) < 0 or settings.get("noise_floor", 0.001) > 1:
            errors.append("noise_floor should be between 0.0 and 1.0")
        
        if settings.get("diarization_threshold", 0.35) < 0 or settings.get("diarization_threshold", 0.35) > 1:
            errors.append("diarization_threshold should be between 0.0 and 1.0")
        
        # Logical constraints
        min_win = settings.get("min_window_sec", 1.0)
        max_win = settings.get("max_window_sec", 5.0)
        if max_win <= min_win:
            errors.append(f"max_window_sec ({max_win}) must be greater than min_window_sec ({min_win})")
        
        if settings.get("min_speaker_samples", 16000) < 4000:
            errors.append("min_speaker_samples should be at least 4000 (0.25 seconds)")
    except TypeError:
        # Skip range checks if types are invalid - type errors already logged above
        pass
    
    # Valid model sizes
    valid_models = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", 
                    "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3"]
    model_size = settings.get("model_size", "medium.en")
    if model_size not in valid_models:
        errors.append(f"model_size '{model_size}' is not valid. Choose from: {', '.join(valid_models)}")
    
    # Valid devices
    valid_devices = ["auto", "cuda", "cpu"]
    device = settings.get("device", "auto")
    if device not in valid_devices:
        errors.append(f"device '{device}' is not valid. Choose from: {', '.join(valid_devices)}")
    
    # Log results
    if errors:
        for err in errors:
            logger.warning(f"Config validation: {err}")
            print(f"⚠️  Config warning: {err}")
    else:
        logger.info("Config validation passed")
    
    return errors
