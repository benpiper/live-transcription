#!/usr/bin/env python3
"""
Live Transcription with Faster-Whisper

A high-performance, real-time audio transcription tool with speaker
identification, web dashboard, and advanced hallucination filtering.

This is the main entry point that orchestrates the audio capture,
transcription engine, and web server modules.
"""

import os
import sys
import warnings
import logging
import argparse
import threading
import queue
import time
from datetime import datetime

import numpy as np
import sounddevice as sd
import torch

# Suppress library-level warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*autocast.*")
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Import our modules
from config import TRANSCRIPTION_CONFIG, load_config, get_setting, get_session_management_setting
from transcription_engine import setup_cuda_paths, load_model, transcribe_chunk, SAMPLE_RATE, set_status_callback
from speaker_manager import speaker_manager, MIN_SPEAKER_ID_SAMPLES
from audio_processing import is_speech
from audio_buffer import audio_buffer

# Setup CUDA before importing web server (which imports ctranslate2)
setup_cuda_paths()

from web_server import create_app, set_queues, ws_manager
from shutdown_handler import register_shutdown_handlers

# Audio parameters
BUFFER_SIZE = 4096

# Thread-safe queues
audio_queue = queue.Queue()
broadcast_queue = queue.Queue()
audio_broadcast_queue = queue.Queue()
stop_event = threading.Event()

# Global volume for visualization
current_volume = 0.0

# Speaker model (loaded optionally)
speaker_model = None

# Silence monitoring
last_transcript_time = None
silence_warning_shown = False
SILENCE_WARNING_THRESHOLD = 30 * 60  # 30 minutes in seconds


def audio_callback(indata, frames, time_info, status):
    """Callback function to capture audio data from the microphone."""
    global current_volume

    if status:
        logger.warning(f"Audio status: {status}")

    audio_data = indata[:, 0].copy()
    timestamp = datetime.now()

    # Calculate volume for visualization
    current_volume = np.linalg.norm(audio_data) / np.sqrt(len(audio_data))

    # Add to transcription queue
    audio_queue.put((timestamp, audio_data))

    # Use Unix timestamp (float) consistently for audio buffer alignment with transcripts
    now = timestamp.timestamp()

    # Add audio to backend buffer only if above noise floor
    # Avoids wasting buffer space on continuous silence during inactive periods
    try:
        noise_floor = get_setting("noise_floor", 0.001)
        if current_volume >= noise_floor:
            audio_buffer.add_chunk(now, audio_data)
    except Exception as e:
        logger.debug(f"Audio buffer error: {e}")

    # Noise-gated audio broadcast to web clients
    try:
        noise_floor = get_setting("noise_floor", 0.001)
        last_broadcast = getattr(audio_callback, "last_vol_broadcast", 0)

        if current_volume >= noise_floor:
            # Broadcast raw audio when above noise floor (for visualization)
            audio_broadcast_queue.put_nowait(audio_data.astype(np.float32).tobytes())
            audio_callback.last_vol_broadcast = now
        elif now - last_broadcast > 2.0:
            # Send a small JSON volume update during silence (every 2s)
            broadcast_queue.put({
                "type": "volume",
                "peak": float(current_volume)
            })
            audio_callback.last_vol_broadcast = now
    except queue.Full:
        pass
    except Exception as e:
        logger.debug(f"Audio broadcast error: {e}")



def visualizer():
    """Thread to display a simple volume meter in the terminal."""
    global current_volume
    
    while not stop_event.is_set():
        # Create a simple ASCII bar
        bars = int(current_volume * 300)  # multiplier for visibility
        bars = min(bars, 50)
        meter = "[" + "#" * bars + "-" * (50 - bars) + "]"
        
        # Determine status
        status = "Active" if current_volume > 0.002 else "Quiet "
        
        sys.stdout.write(f"\r{meter} {status} (vol: {current_volume:.4f})")
        sys.stdout.flush()
        time.sleep(0.05)



def transcribe_audio_loop():
    """Main transcription loop thread."""
    logger.info("Transcription thread started.")
    print("Transcription thread started.")
    
    active_buffer = []
    prev_context = None
    start_time = None
    chunk_count = 0
    
    # Initialize silence monitor start time
    global last_transcript_time, silence_warning_shown
    last_transcript_time = time.time()
    silence_warning_shown = False
    
    while not stop_event.is_set():
        try:
            # Check for backlog - configurable behavior
            q_size = audio_queue.qsize()
            max_queue_size = get_setting("max_queue_size", 0)  # 0 = no dropping
            
            if max_queue_size > 0 and q_size > max_queue_size:
                print(f"\r[BACKLOG] Queue is {q_size} chunks deep. Dropping oldest...")
                audio_queue.get_nowait()
                continue
            elif q_size > 50:  # Warn but don't drop
                print(f"\r[WARNING] Queue is {q_size} chunks deep - transcription is behind")
            
            # Check for extended silence
            if last_transcript_time is not None:
                silence_duration = time.time() - last_transcript_time
                if silence_duration > SILENCE_WARNING_THRESHOLD and not silence_warning_shown:
                    mins = int(silence_duration // 60)
                    print(f"\n[WARNING] No transcripts for {mins} minutes - check audio source!")
                    silence_warning_shown = True
            
            timestamp, new_chunk = audio_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        
        # Sentinel for flushing (end of file)
        if timestamp is None:
            if active_buffer:
                audio_data = np.concatenate(active_buffer)
                transcribe_chunk(
                    audio_data, prev_context, start_time,
                    speaker_model=speaker_model,
                    speaker_manager=speaker_manager,
                    output_callback=output_transcription
                )
            break
        
        if not active_buffer:
            start_time = timestamp
        
        active_buffer.append(new_chunk)

        # Calculate recent volume for silence detection / VAD fallback
        recent_vol = np.linalg.norm(new_chunk) / np.sqrt(len(new_chunk))

        # Determine if we should transcribe now
        buffer_duration = len(active_buffer) * (BUFFER_SIZE / SAMPLE_RATE)
        min_win = get_setting("min_window_sec", 1.0)
        max_win = get_setting("max_window_sec", 5.0)

        # Voice Activity Detection (optional ML-based speech detection)
        vad_enabled = get_setting("vad_enabled", False)
        vad_confidence_threshold = get_setting("vad_confidence_threshold", 0.5)
        is_speech_result = None
        vad_confidence = None

        if vad_enabled:
            is_speech_result, vad_confidence = is_speech(
                new_chunk,
                sample_rate=SAMPLE_RATE,
                confidence_threshold=vad_confidence_threshold
            )

        should_transcribe = False
        if buffer_duration >= min_win:
            # Check for silence using VAD if available, fallback to volume
            if is_speech_result is not None:
                # VAD is available
                is_silent = is_speech_result is False
            else:
                # Fallback to volume-based detection
                is_silent = recent_vol < 0.002

            if is_silent:
                should_transcribe = True
        elif buffer_duration >= max_win:
            should_transcribe = True

        if not should_transcribe:
            continue
        
        # Aggregate and volume check
        total_audio = np.concatenate(active_buffer)
        total_vol = np.linalg.norm(total_audio) / np.sqrt(len(total_audio))
        
        noise_floor = get_setting("noise_floor", 0.001)
        if total_vol < noise_floor:
            active_buffer = []
            start_time = None
            continue
        
        # Telemetry
        qsize = audio_queue.qsize()
        skip_diarization = qsize > 10
        
        t0 = time.time()
        try:
            prev_context = transcribe_chunk(
                total_audio, prev_context, start_time,
                skip_diarization=skip_diarization,
                speaker_model=speaker_model,
                speaker_manager=speaker_manager,
                output_callback=lambda segments, start, audio: output_transcription(
                    segments, start, audio,
                    vad_confidence=vad_confidence,
                    processing_time=time.time() - t0
                )
            )
        except Exception as e:
            logger.error(f"Critical error in transcription loop: {e}", exc_info=True)
            print(f"\n[CRITICAL ERROR] Transcription loop encountered an error: {e}")
            # Ensure we don't spin too fast if it's a persistent error
            time.sleep(0.5)
            # Clear active buffer to try and recover
            active_buffer = []
            start_time = None
            continue

        t_proc = time.time() - t0
        
        rtf = t_proc / buffer_duration
        
        # Telemetry output
        status_line = f"\n [Telemetry] Backlog: {qsize} | Proc: {t_proc:.2f}s | Audio: {buffer_duration:.2f}s | RTF: {rtf:.2f}"
        if skip_diarization:
            status_line += " | [!] DIARIZATION SKIPPED"
        
        if rtf > 1.0 or qsize > 0 or chunk_count % 10 == 0:
            sys.stdout.write(status_line + "\n")
            sys.stdout.flush()
        
        chunk_count += 1
        active_buffer = []


def output_transcription(merged_segments, start_time, audio_flat, vad_confidence=None, processing_time=None):
    """Callback to handle transcription output."""
    global last_transcript_time, silence_warning_shown

    # Reset silence tracking
    last_transcript_time = time.time()
    silence_warning_shown = False

    display_time = start_time.strftime("%Y-%m-%d %H:%M:%S")

    for m in merged_segments:
        speaker_tag = f" [{m['speaker']}]"
        conf_val = m.get("confidence", 0)
        conf_tag = f" (conf: {conf_val:.2f})"
        output_line = f"[{display_time}]{speaker_tag}{conf_tag} {m['text']}"

        # Broadcast to web clients
        try:
            broadcast_queue.put({
                "type": "transcript",
                "timestamp": display_time,
                "origin_time": start_time.timestamp() + m["start"],
                "duration": m["end"] - m["start"],
                "speaker": m["speaker"],
                "text": m["text"],
                "confidence": conf_val,
                "vad_confidence": vad_confidence,
                "processing_time": processing_time
            })
        except Exception as e:
            logger.debug(f"Broadcast failed: {e}")
        
        # Console output
        sys.stdout.write("\r" + " " * 80 + "\r")
        print(output_line)
        
        # File output
        output_file = getattr(transcribe_audio_loop, "output_file", None)
        if output_file:
            try:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(output_line + "\n")
                    f.flush()
            except Exception as e:
                logger.error(f"Error writing to output file: {e}")
        
        # Add to session for persistence
        try:
            from session import get_session
            session = get_session()
            if session:
                session.add_transcript({
                    "timestamp": display_time,
                    "origin_time": start_time.timestamp() + m["start"],
                    "duration": m["end"] - m["start"],
                    "speaker": m["speaker"],
                    "text": m["text"],
                    "confidence": conf_val
                })
        except Exception as e:
            logger.debug(f"Session tracking failed: {e}")


def cleanup_resources():
    """Clean up all resources before shutdown."""
    global speaker_model

    logger.info("Starting resource cleanup...")

    # 1. Set stop event
    stop_event.set()

    # 2. Drain critical queues
    remaining = []
    try:
        while not broadcast_queue.empty():
            remaining.append(broadcast_queue.get_nowait())
    except queue.Empty:
        pass

    if remaining:
        logger.info(f"Draining {len(remaining)} queued transcript(s)")

    # 3. Save session
    try:
        from session import get_session, save_session
        session = get_session()
        if session and len(session.transcripts) > 0:
            save_session(session)
            logger.info(f"Session saved: {session.name} ({len(session.transcripts)} transcript(s))")
    except Exception as e:
        logger.error(f"Session save failed: {e}")

    # 4. Close WebSocket connections
    try:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if not loop.is_closed():
            loop.run_until_complete(ws_manager.close_all())
    except Exception as e:
        logger.debug(f"WebSocket cleanup error: {e}")

    # 5. Clean up models
    try:
        from transcription_engine import cleanup_model
        cleanup_model()
    except Exception as e:
        logger.error(f"Model cleanup failed: {e}")

    # 6. Clean up speaker model
    if speaker_model:
        del speaker_model
        speaker_model = None

    logger.info("Resource cleanup complete")


def feed_file_to_queue(file_path: str):
    """Feed an audio file into the transcription queue at real-time speed."""
    import soundfile as sf
    import librosa
    
    print(f"\nProcessing audio file: {file_path}")
    
    try:
        audio, sr = sf.read(file_path, dtype="float32")
    except Exception as e:
        logger.error(f"Failed to read audio file: {e}")
        print(f"Error reading audio file: {e}")
        return
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    
    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    
    # Feed in chunks
    chunk_duration = BUFFER_SIZE / SAMPLE_RATE
    chunk_samples = BUFFER_SIZE
    
    for i in range(0, len(audio), chunk_samples):
        if stop_event.is_set():
            break
        
        chunk = audio[i:i + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        
        timestamp = datetime.now()
        audio_queue.put((timestamp, chunk))
        
        # Broadcast for web visualization
        try:
            audio_broadcast_queue.put_nowait(chunk.astype(np.float32).tobytes())
        except queue.Full:
            pass
        
        time.sleep(chunk_duration)
    
    print("\nEnd of file reached.")
    audio_queue.put((None, None))


def session_rollover_loop():
    """
    Monitor session state and trigger rollover based on time or transcript count.

    Runs as a daemon thread and periodically checks if rollover conditions are met.
    When triggered:
    1. Saves the current session
    2. Archives it (if enabled)
    3. Creates a new session
    """
    from config import get_session_management_setting
    from session import get_session, save_session, archive_session, init_session, cleanup_old_sessions

    logger.info("Session rollover monitor started")

    while not stop_event.is_set():
        try:
            # Check if rollover is enabled
            rollover_enabled = get_session_management_setting("enable_rollover", False)
            if not rollover_enabled:
                time.sleep(60)  # Check every minute
                continue

            session = get_session()
            if not session:
                time.sleep(60)
                continue

            # Get rollover thresholds
            rollover_hours = get_session_management_setting("rollover_time_hours", 24)
            rollover_count = get_session_management_setting("rollover_transcript_count", 10000)
            archive_old = get_session_management_setting("archive_old_sessions", False)
            archive_age_days = get_session_management_setting("archive_age_days", 30)

            # Check time-based rollover
            created_at = datetime.fromisoformat(session.created_at)
            elapsed_seconds = (datetime.now() - created_at).total_seconds()
            time_exceeded = (rollover_hours and elapsed_seconds > rollover_hours * 3600)

            # Check count-based rollover
            transcript_count = len(session.transcripts)
            count_exceeded = (rollover_count and transcript_count >= rollover_count)

            if time_exceeded or count_exceeded:
                trigger = "time" if time_exceeded else "count"
                logger.info(f"Session rollover triggered by {trigger}: {session.name}")
                print(f"\n📋 Session rollover triggered ({trigger}): {session.name}")

                # Save current session
                try:
                    save_session(session)
                    logger.info(f"Session saved before rollover: {session.name}")
                except Exception as e:
                    logger.error(f"Failed to save session before rollover: {e}")

                # Archive if enabled
                if archive_old:
                    try:
                        if archive_session(session.name):
                            logger.info(f"Session archived: {session.name}")
                            print(f"📦 Archived: {session.name}")
                    except Exception as e:
                        logger.error(f"Failed to archive session: {e}")

                # Create new session with timestamp name
                new_session_name = f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                try:
                    new_session = init_session(session_name=new_session_name)
                    save_session(new_session)
                    logger.info(f"New session created after rollover: {new_session_name}")
                    print(f"📁 New session: {new_session_name}")
                except Exception as e:
                    logger.error(f"Failed to create new session after rollover: {e}")

            # Periodically clean up old sessions (check hourly)
            if archive_old and hasattr(session_rollover_loop, "last_cleanup"):
                if time.time() - session_rollover_loop.last_cleanup > 3600:
                    try:
                        cleanup_old_sessions(archive_age_days)
                        session_rollover_loop.last_cleanup = time.time()
                    except Exception as e:
                        logger.warning(f"Scheduled cleanup failed: {e}")
            elif archive_old:
                session_rollover_loop.last_cleanup = time.time()

            time.sleep(60)  # Check every minute

        except Exception as e:
            logger.error(f"Session rollover monitor error: {e}")
            time.sleep(60)

    logger.info("Session rollover monitor stopped")


def boot_app():
    """Initialize the application (model loading, speaker model, etc.)."""
    global speaker_model
    
    # Initialize or load session
    from session import init_session, load_session_from_file
    
    session_name = getattr(args, 'session', None)
    if session_name:
        try:
            session = load_session_from_file(session_name, args.config)
            print(f"📂 Loaded session: {session_name} ({len(session.transcripts)} transcripts)")
        except FileNotFoundError:
            session = init_session(args.config, session_name)
            print(f"📁 Created new session: {session_name}")
    else:
        session = init_session(args.config)
    
    # Log any config validation warnings
    if session.get_config_errors():
        print(f"⚠️  {len(session.get_config_errors())} config warning(s) - see above")

    # Initialize audio buffer with configured window size
    window_sec = get_setting("audio_buffer_window_sec", 7200)
    audio_buffer.set_window_size(window_sec)
    logger.info(f"Audio buffer initialized with {window_sec}s window ({window_sec/3600:.1f} hours)")

    if args.diarize:
        print("Loading speaker identification model...")
        from speechbrain.inference.speaker import EncoderClassifier
        speaker_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"}
        )
        # Also set on session for future use
        session.speaker_model = speaker_model
    
    if args.output:
        transcribe_audio_loop.output_file = args.output
        session.set_output_file(args.output)
        print(f"Transcription will be saved to {args.output}")
    
    # Register status callback to broadcast engine events (like fallback)
    def engine_status_callback(status_type, data):
        logger.info(f"Engine status change: {status_type} - {data.get('message')}")
        try:
            broadcast_queue.put({
                "type": "status",
                "status": status_type,
                "data": data
            })
        except Exception as e:
            logger.error(f"Failed to broadcast status: {e}")
            
    set_status_callback(engine_status_callback)

    # Load Whisper model
    load_model()
    
    # Start processing threads
    threading.Thread(target=visualizer, daemon=True).start()
    transcription_thread = threading.Thread(target=transcribe_audio_loop, daemon=True)
    transcription_thread.start()
    print("Core processing threads started.")

    # Register shutdown handlers
    register_shutdown_handlers(stop_event, cleanup_resources)

    # Auto-save session periodically if --session flag is used
    if session_name:
        def autosave_loop():
            from session import save_session, get_session
            while not stop_event.is_set():
                time.sleep(120)
                try:
                    s = get_session()
                    if s and len(s.transcripts) > 0:
                        save_session(s)
                        logger.debug(f"Session auto-saved: {s.name} ({len(s.transcripts)} transcripts)")
                except Exception as e:
                    logger.warning(f"Auto-save failed: {e}")
        
        threading.Thread(target=autosave_loop, daemon=True).start()
        print(f"📁 Session will auto-save.")

    # Session rollover monitor
    rollover_enabled = get_session_management_setting("enable_rollover", False)
    if rollover_enabled:
        threading.Thread(target=session_rollover_loop, daemon=True).start()
        print(f"⏱️  Session rollover monitoring enabled.")


def run_input_source():
    """Run the selected input source (file or microphone)."""
    if args.input:
        try:
            feed_file_to_queue(args.input)
            time.sleep(5)
        except KeyboardInterrupt:
            pass
    else:
        device_id = args.device if args.device else None
        print(f"Using input device: {device_id or 'default'}")
        print("Listening... Press Ctrl+C to stop.")
        
        try:
            with sd.InputStream(
                callback=audio_callback,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=BUFFER_SIZE,
                device=device_id
            ):
                while not stop_event.is_set():
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error(f"Audio input error: {e}")
            print(f"\nAudio input error: {e}")


# CLI argument parsing
parser = argparse.ArgumentParser(description="Live Transcription with Faster-Whisper")
parser.add_argument("--input", type=str, help="Path to an audio file to test instead of microphone")
parser.add_argument("--config", "-c", type=str, default="config.json", help="Path to config.json for vocabulary, corrections, and settings")
parser.add_argument("--output", "-o", type=str, help="Path to save the transcribed text file")
parser.add_argument("--diarize", "-d", action="store_true", help="Enable real-time speaker identification")
parser.add_argument("--web", "-w", action="store_true", help="Enable the web dashboard")
parser.add_argument("--port", type=int, default=8000, help="Web dashboard port")
parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
parser.add_argument("--device", type=int, help="Input device ID")
parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (Web mode only)")
parser.add_argument("--ssl", action="store_true", help="Enable HTTPS/SSL for the web dashboard")
parser.add_argument("--session", "-s", type=str, help="Load or create a named session")
parser.add_argument("--list-sessions", action="store_true", help="List all saved sessions and exit")
args = parser.parse_args()

# Set broadcast queues for web server (needed at module level for reload)
set_queues(broadcast_queue, audio_broadcast_queue, stop_event)

# Create app at module level so uvicorn --reload can find it
app = create_app(boot_callback=boot_app, input_callback=run_input_source)


if __name__ == "__main__":
    # Handle device listing
    if args.list_devices:
        print("\nAvailable Audio Devices:")
        print(sd.query_devices())
        sys.exit(0)
    
    # Handle session listing
    if args.list_sessions:
        from session import list_sessions
        sessions = list_sessions()
        if sessions:
            print("\n📁 Saved Sessions:")
            print("-" * 60)
            for s in sessions:
                print(f"  {s['name']}")
                print(f"      Created: {s['created_at'][:19] if s.get('created_at') else 'Unknown'}")
                print(f"      Transcripts: {s['transcript_count']}")
                print()
        else:
            print("\nNo saved sessions found.")
            print("Use --session 'Name' to create a new session.")
        sys.exit(0)
    
    if args.web:
        import uvicorn
        
        ssl_enabled = args.ssl or get_setting("ssl_enabled", False)
        protocol = "https" if ssl_enabled else "http"
        print(f"\nWeb Dashboard available at {protocol}://localhost:{args.port}")
        
        ssl_keyfile = get_setting("ssl_keyfile", "key.pem") if ssl_enabled else None
        ssl_certfile = get_setting("ssl_certfile", "cert.pem") if ssl_enabled else None
        
        try:
            if args.reload:
                module_name = os.path.splitext(os.path.basename(__file__))[0]
                uvicorn.run(
                    f"{module_name}:app",
                    host="0.0.0.0",
                    port=args.port,
                    reload=True,
                    reload_includes=["*.py", "*.html", "*.js", "*.css", "config.json"],
                    reload_excludes=["sessions/*", "*.log", "transcript.txt"],
                    ssl_keyfile=ssl_keyfile,
                    ssl_certfile=ssl_certfile
                )
            else:
                uvicorn.run(
                    app,
                    host="0.0.0.0",
                    port=args.port,
                    log_level="error",
                    ssl_keyfile=ssl_keyfile,
                    ssl_certfile=ssl_certfile
                )
        except KeyboardInterrupt:
            cleanup_resources()
            sys.exit(0)
    else:
        # CLI-only mode
        boot_app()
        run_input_source()
        cleanup_resources()
        print("\nStopped.")
        sys.exit(0)
