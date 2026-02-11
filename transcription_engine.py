"""
Transcription engine for Whisper model management and audio processing.

This module handles loading the Whisper model, processing audio chunks,
and applying filters and corrections to transcription output.
"""

import os
import sys
import re
import logging
import numpy as np
import torch
from datetime import datetime
from faster_whisper import WhisperModel

from config import TRANSCRIPTION_CONFIG, get_setting, get_vocabulary, get_corrections

logger = logging.getLogger(__name__)

# Global model instance
model = None

# Audio parameters
SAMPLE_RATE = 16000


def setup_cuda_paths():
    """
    Adds all NVIDIA library paths to LD_LIBRARY_PATH and pre-loads them 
    to ensure they are available for faster-whisper/ctranslate2.
    """
    import ctypes
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    base_search_paths = [
        os.path.expanduser(f"~/.local/lib/{python_version}/site-packages/nvidia"),
        os.path.expanduser(f"/usr/local/lib/{python_version}/dist-packages/nvidia"),
        "/usr/local/cuda/lib64",
    ]
    
    extra_paths = []
    for base in base_search_paths:
        if not os.path.exists(base):
            continue
        
        for root, dirs, files in os.walk(base):
            if 'lib' in dirs:
                lib_path = os.path.join(root, 'lib')
                if lib_path not in extra_paths:
                    extra_paths.append(lib_path)
    
    if extra_paths:
        # Update LD_LIBRARY_PATH
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        ld_parts = extra_paths + ([current_ld] if current_ld else [])
        os.environ["LD_LIBRARY_PATH"] = ":".join(ld_parts)
        
        # Pre-load critical libraries to ensure they are in the process space
        # This is often necessary because changing LD_LIBRARY_PATH mid-process is unreliable
        critical_libs = [
            "libcublas.so.12", "libcudnn.so.9", "libcudnn_ops.so.9", 
            "libcudnn_cnn.so.9", "libcudnn_adv.so.9", "libcublasLt.so.12"
        ]
        
        loaded_count = 0
        for lib_name in critical_libs:
            for p in extra_paths:
                full_path = os.path.join(p, lib_name)
                if os.path.exists(full_path):
                    try:
                        ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)
                        loaded_count += 1
                        break  # Move to next critical lib
                    except Exception:
                        pass
        
        if loaded_count > 0:
            print(f"CUDA hardware acceleration: Pre-loaded {loaded_count} core libraries.")
        else:
            # Fallback check for v8/v11
            fallback_libs = ["libcublas.so.11", "libcudnn.so.8"]
            for lib_name in fallback_libs:
                for p in extra_paths:
                    full_path = os.path.join(p, lib_name)
                    if os.path.exists(full_path):
                        try:
                            ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)
                            loaded_count += 1
                        except Exception:
                            pass

    return extra_paths


def load_model():
    """
    Initialize the Faster-Whisper model using configuration settings.
    
    Automatically detects CUDA availability and selects appropriate
    compute type for optimal performance.
    """
    global model
    
    settings = TRANSCRIPTION_CONFIG["settings"]
    model_size = settings.get("model_size", "medium.en")
    device_setting = settings.get("device", "auto")
    compute_type = settings.get("compute_type", "auto")
    cpu_threads = settings.get("cpu_threads", 4)
    
    # Determine device
    if device_setting == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_setting
    
    # Determine compute type
    if compute_type == "auto":
        computed_type = "float16" if device == "cuda" else "int8"
    else:
        computed_type = compute_type
    
    if device == "cuda":
        print("CUDA detected. Using GPU acceleration (float16).")
        logger.info("Using CUDA GPU acceleration")
    else:
        print("CUDA not available. Using CPU.")
        logger.info("Using CPU for transcription")
    
    print(f"Loading faster-whisper model ({model_size}) on {device}...")
    
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=computed_type,
        cpu_threads=cpu_threads
    )
    
    print(f"Model loaded (Threads: {cpu_threads}, Device: {device}, Type: {computed_type}).")
    logger.info(f"Model loaded: {model_size} on {device}")


def transcribe_chunk(
    audio_data: np.ndarray,
    prev_context: np.ndarray = None,
    start_time: datetime = None,
    skip_diarization: bool = False,
    speaker_model=None,
    speaker_manager=None,
    output_callback=None
) -> np.ndarray:
    """
    Process a buffer of audio data and generate transcription.
    
    Args:
        audio_data: Concatenated audio samples (float32)
        prev_context: Previous audio context for continuity
        start_time: Timestamp when audio capture started
        skip_diarization: Skip speaker identification if True
        speaker_model: SpeechBrain speaker encoder
        speaker_manager: SpeakerManager instance
        output_callback: Callback for transcription output
        
    Returns:
        Audio context for the next chunk (last 0.5s)
    """
    if model is None:
        logger.error("Model not loaded!")
        return None
    
    audio_flat = audio_data.flatten().astype(np.float32)
    
    # Prepend previous context (0.5s) to help with word fragments
    if prev_context is not None:
        audio_flat = np.concatenate([prev_context, audio_flat])
    
    # Calculate volume
    volume_norm = np.linalg.norm(audio_flat) / np.sqrt(len(audio_flat))
    
    if volume_norm < 0.002:
        return _get_context_tail(audio_flat)
    
    # Normalize audio
    max_val = np.max(np.abs(audio_flat))
    if max_val > 0.01:
        audio_flat = audio_flat / max_val
    
    # Get settings
    settings = TRANSCRIPTION_CONFIG["settings"]
    
    # Build initial prompt
    prompt_parts = []
    
    # Add base prompt from config (domain context)
    base_prompt = settings.get("initial_prompt", "")
    if base_prompt:
        prompt_parts.append(base_prompt)
    
    vocabulary = get_vocabulary()
    if vocabulary:
        # Use a context preamble to reduce biasing weight
        prompt_parts.append(f"Context and technical terms that might be mentioned: {', '.join(vocabulary)}")
    
    # Add previous transcript for continuity (keep it short)
    last_text = getattr(transcribe_chunk, "last_transcript", None)
    if last_text:
        prompt_parts.append(f"Previously said: {last_text[-150:]}")
    
    initial_prompt = ". ".join(prompt_parts) if prompt_parts else None

    # Debug: Log the prompt being sent to Whisper
    if initial_prompt:
        logger.debug(f"Whisper prompt ({len(initial_prompt)} chars): {initial_prompt}")
    
    # Transcribe
    try:
        segments, info = model.transcribe(
            audio_flat,
            language="en",
            beam_size=settings.get("beam_size", 5),
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
            no_speech_threshold=settings.get("no_speech_threshold", 0.6),
            log_prob_threshold=settings.get("log_prob_threshold", -1.0),
            compression_ratio_threshold=settings.get("compression_ratio_threshold", 2.4),
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": settings.get("min_silence_duration_ms", 500)
            }
        )
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return _get_context_tail(audio_flat)
    
    # Process segments
    processed_segments = []
    avg_logprob_cutoff = settings.get("avg_logprob_cutoff", -0.8)
    no_speech_prob_cutoff = settings.get("no_speech_prob_cutoff", 0.2)
    extreme_conf_cutoff = settings.get("extreme_confidence_cutoff", -0.4)
    
    hallucination_patterns = [
        r"^(Thank you|Thanks)[\.\!\s]*$",
        r"^(Bye|Goodbye)[\.\!\s]*$",
        r"^(Hmm+|Uh+|Um+)[\.\!\s]*$",
        r"^[\.\!\?\s\-]+$",
        r"^\s*$",
        # Subtitle/caption credits
        r"(?i)subtitles?\s*(by|from|created)",
        r"(?i)amara\.org",
        r"(?i)transcribed\s*by",
        r"(?i)captioned\s*by",
        r"(?i)translated\s*by",
        # Music/sound notation
        r"^\[.*\]$",
        r"^♪.*♪$",
        r"^\(.*\)$",
        # Sponsor/promo spam
        r"(?i)subscribe\s*(to|and|now)",
        r"(?i)like\s*(and|this)\s*video",
        r"(?i)don'?t\s*forget\s*to",
    ]
    
    for segment in segments:
        text_segment = segment.text.strip()
        
        # Filter hallucinations
        if any(re.match(p, text_segment, re.IGNORECASE) for p in hallucination_patterns):
            continue
        
        # Confidence filtering
        if segment.avg_logprob < extreme_conf_cutoff:
            pass  # High confidence, skip other checks
        elif segment.avg_logprob < avg_logprob_cutoff:
            continue
        elif segment.no_speech_prob > no_speech_prob_cutoff:
            continue
            
        # Enhanced Confidence Filtering: Reject vocab-heavy segments with marginal confidence
        if vocabulary:
            vocab_lower = [v.lower() for v in vocabulary]
            words = text_segment.lower().split()
            vocab_hits = [w for w in words if any(v in w for v in vocab_lower)]
            
            # If segment is >50% vocabulary words, require even higher confidence
            if len(words) > 0 and (len(vocab_hits) / len(words)) > 0.5:
                strict_cutoff = avg_logprob_cutoff + 0.2
                if segment.avg_logprob < strict_cutoff:
                    logger.debug(f"Rejecting vocab-heavy segment (conf {segment.avg_logprob:.2f} < {strict_cutoff:.2f}): {text_segment}")
                    continue
        
        # Speaker identification (via profiles or dynamic clustering)
        seg_speaker_label = "Speaker"  # Default label
        start_sample = int(segment.start * SAMPLE_RATE)
        end_sample = int(segment.end * SAMPLE_RATE)
        audio_slice = audio_flat[start_sample:end_sample]
        
        # Full diarization path (with speaker embeddings)
        if speaker_model and speaker_manager and not skip_diarization:
            if len(audio_slice) >= settings.get("min_speaker_samples", 16000):
                try:
                    signal = torch.from_numpy(audio_slice).unsqueeze(0)
                    embeddings = speaker_model.encode_batch(signal)
                    emb = embeddings.squeeze()
                    seg_speaker_label = speaker_manager.identify_speaker(emb, audio_slice)
                except Exception as e:
                    logger.debug(f"Speaker identification failed: {e}")
        
        # Apply corrections
        text = text_segment
        corrections = get_corrections()
        for wrong, right in corrections.items():
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            text = pattern.sub(right, text)
        
        processed_segments.append({
            "speaker": seg_speaker_label,
            "text": text,
            "start": segment.start,
            "end": segment.end,
            "confidence": segment.avg_logprob
        })
    
    # Merge consecutive segments from same speaker
    if processed_segments:
        merged = _merge_segments(processed_segments)
        
        # Output results
        if output_callback and start_time:
            output_callback(merged, start_time, audio_flat)
        
        # Store for context
        transcribe_chunk.last_transcript = " ".join([m["text"] for m in merged])
    
    return _get_context_tail(audio_flat)


def _merge_segments(segments: list) -> list:
    """Merge consecutive segments from the same speaker."""
    merged = []
    for seg in segments:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            prev_count = merged[-1].get("_seg_count", 1)
            curr_conf = merged[-1].get("confidence", 0)
            new_conf = (curr_conf * prev_count + seg["confidence"]) / (prev_count + 1)
            
            merged[-1]["text"] += " " + seg["text"]
            merged[-1]["end"] = seg["end"]
            merged[-1]["confidence"] = new_conf
            merged[-1]["_seg_count"] = prev_count + 1
        else:
            seg["_seg_count"] = 1
            merged.append(seg)
    return merged


def _get_context_tail(audio: np.ndarray) -> np.ndarray:
    """Get the last 0.5s of audio for context."""
    context_samples = int(0.5 * SAMPLE_RATE)
    if len(audio) > context_samples:
        return audio[-context_samples:]
    return audio


def cleanup_model():
    """
    Release GPU resources held by the Whisper model.

    Called during graceful shutdown to ensure GPU memory is returned to the system.
    """
    global model

    if model is None:
        return

    logger.info("Cleaning up Whisper model...")

    try:
        del model
        model = None

        # Release GPU memory if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU memory released")

        # Force garbage collection
        import gc
        gc.collect()

        logger.info("Model cleanup complete")
    except Exception as e:
        logger.error(f"Error during model cleanup: {e}")
