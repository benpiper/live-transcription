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
