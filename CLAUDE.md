# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time audio transcription system using faster-whisper with speaker diarization, web dashboard, and advanced hallucination filtering. Primary use case: emergency dispatch radio monitoring.

## Development Commands

### Running the Application

```bash
# Basic transcription with diarization and web UI
python3 real_time_transcription.py --diarize --web --config config.json

# With SSL enabled
python3 real_time_transcription.py --diarize --web --config config.json --ssl

# Test with audio file instead of microphone
python3 real_time_transcription.py --input test.mp3 --diarize --web --config config.json

# With session management
python3 real_time_transcription.py --session "Morning Shift" --web --config config.json

# List available audio devices
python3 real_time_transcription.py --list-devices

# List saved sessions
python3 real_time_transcription.py --list-sessions
```

### Voice Profile Management

```bash
# Enroll a new speaker profile
python3 enroll_voice.py enroll --name "Dispatcher A" --samples audio1.wav audio2.wav

# List enrolled profiles
python3 enroll_voice.py list

# Delete a profile
python3 enroll_voice.py delete --name "Dispatcher A"
```

### Email Alerts

```bash
# Run email alert monitoring
python3 email_alert.py

# Test email configuration
python3 email_alert.py --test
```

### Testing

```bash
# Test speaker identification
python3 test_speaker_id.py
```

## Architecture Overview

### Threading Model

The application uses a **multi-threaded pipeline architecture**:

1. **Audio Capture Thread**: Reads from microphone/file via `sounddevice` → pushes to `audio_queue`
2. **Transcription Thread**: Consumes `audio_queue` → processes via Whisper + diarization → pushes to `broadcast_queue`
3. **WebSocket Broadcaster** (asyncio task): Drains `broadcast_queue` and `audio_broadcast_queue` → sends to connected clients

All threads coordinate via `threading.Event` (`stop_event`) for graceful shutdown.

### Module Responsibilities

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `real_time_transcription.py` | Main entry point, orchestrates all components | CLI interface, audio capture thread, transcription thread |
| `transcription_engine.py` | Whisper model management and audio processing | `load_model()`, `transcribe_chunk()`, `setup_cuda_paths()` |
| `speaker_manager.py` | Speaker diarization and identification | `SpeakerManager` class, `speaker_manager` singleton |
| `voice_profiles.py` | Pre-registered speaker profiles | `VoiceProfileManager` class, `voice_profile_manager` singleton |
| `web_server.py` | FastAPI application and WebSocket handling | `create_app()`, `ConnectionManager`, `ws_manager` |
| `session.py` | Session state encapsulation | `TranscriptionSession` class |
| `config.py` | Configuration loading and validation | `load_config()`, `get_setting()`, `TRANSCRIPTION_CONFIG` |
| `email_alert.py` | Standalone keyword alert system | CLI interface for monitoring |
| `enroll_voice.py` | Voice profile enrollment CLI | CLI interface for profile management |

### Data Flow

```
Microphone/File
    ↓
audio_callback() [sounddevice callback]
    ↓
audio_queue (queue.Queue)
    ↓
transcription_worker() [thread]
    ├→ load_model() [faster-whisper]
    ├→ transcribe_chunk() [Whisper inference]
    ├→ speaker_manager.identify_speaker() [ECAPA-TDNN embeddings]
    └→ voice_profile_manager.match_profile() [cosine similarity]
    ↓
broadcast_queue (queue.Queue)
    ↓
websocket_broadcaster() [asyncio task]
    ↓
WebSocket clients (browser)
```

### Critical Configuration

`config.json` contains domain-specific settings:

- **vocabulary**: Terms to bias Whisper model (e.g., "Charleston", "Summerville", "10-4")
- **corrections**: Phonetic error mappings (e.g., "EMSOX" → "EMS ops")
- **settings**: Model parameters, hallucination filters, diarization thresholds

Key tunable settings:
- `model_size`: Whisper model (`tiny`, `base`, `small`, `medium`, `large-v3`)
- `no_speech_threshold`: VAD sensitivity (higher = stricter, fewer false positives)
- `avg_logprob_cutoff`: Confidence threshold for accepting transcripts
- `diarization_threshold`: Speaker clustering sensitivity (lower = fewer speakers)
- `min_window_sec`/`max_window_sec`: Timing for silence-aware chunking
- `initial_prompt`: Domain context for Whisper (e.g., "Emergency dispatch radio")

### CUDA Setup

The application **manually pre-loads NVIDIA libraries** before importing `faster-whisper`:

1. `setup_cuda_paths()` (in `transcription_engine.py`) walks `~/.local/lib/pythonX.Y/site-packages/nvidia/`
2. Updates `LD_LIBRARY_PATH` environment variable
3. Uses `ctypes.CDLL()` to force-load critical libs (`libcublas.so.12`, `libcudnn.so.9`, etc.)

**Important**: Must be called before importing `faster_whisper.WhisperModel` to avoid runtime linker errors.

### WebSocket Protocol

The `/ws` endpoint streams two types of messages:

1. **JSON** (transcript data):
   ```json
   {
     "type": "transcript",
     "timestamp": "2026-02-08 14:30:45",
     "origin_time": "2026-02-08 14:30:43",
     "duration": 2.3,
     "speaker": "Speaker 1",
     "text": "Engine 5 responding code 3",
     "confidence": -0.35
   }
   ```

2. **Binary** (raw audio for visualization):
   - `Float32Array` audio chunks (16kHz sample rate)
   - Sent only when volume exceeds `noise_floor` threshold

### Frontend Architecture

- **Vanilla JavaScript** (no framework)
- **IndexedDB** for audio blob storage (on-demand loading to reduce memory usage)
- **Canvas API** for real-time waveform visualization
- **localStorage** for persistent settings (theme, watchwords, scroll lock)
- **Efficient DOM management**: Uses `DocumentFragment` and batching to handle thousands of transcript entries

### Session Persistence

Sessions are stored as JSON files in `sessions/` directory:

```json
{
  "name": "Morning Shift",
  "created_at": "2026-02-08T08:00:00",
  "updated_at": "2026-02-08T12:00:00",
  "transcripts": [
    {
      "timestamp": "2026-02-08 08:15:30",
      "speaker": "Dispatcher A",
      "text": "All units clear for shift change",
      "confidence": -0.25
    }
  ]
}
```

REST API endpoints: `/api/sessions`, `/api/sessions/{name}`, `/api/sessions/{name}/save`, etc.

## Common Patterns

### Adding New Config Settings

1. Add default value in `config.py` → `DEFAULT_CONFIG["settings"]`
2. Use `get_setting("key_name", default)` to access
3. Update validation in `validate_config()` if needed
4. Document in README.md settings table

### Modifying Transcription Filters

Hallucination filters are applied in `transcription_engine.py` → `transcribe_chunk()`:

- `no_speech_prob` check: Segment likely silence
- `avg_logprob` check: Overall confidence too low
- `compression_ratio` check: Repetitive text (hallucination indicator)

Bypass filters for high-confidence segments using `extreme_confidence_cutoff`.

### Adding WebSocket Message Types

1. Define message structure in `real_time_transcription.py` → `transcription_worker()`
2. Push to `broadcast_queue`
3. Handle in `static/app.js` → `ws.onmessage` switch statement

### Voice Profile Storage

Profiles are stored as JSON in `voice_profiles/` directory:

```json
{
  "name": "Dispatcher A",
  "embedding": [0.123, -0.456, ...],  // 192-dimensional vector
  "created_at": "2026-02-08T10:00:00"
}
```

Generated by `enroll_voice.py` using SpeechBrain ECAPA-TDNN model.

## Important Notes

- **GPU Memory**: Large Whisper models (`large-v3`) require ~4GB VRAM. Use `int8_float16` compute type to reduce usage.
- **Latency vs Accuracy**: `max_window_sec` controls update frequency. Lower = faster UI updates, but may cut off mid-sentence.
- **Diarization Overhead**: Speaker identification adds ~200ms per chunk. Disable with `--diarize` flag omitted.
- **SSL Certificates**: For production, replace `cert.pem`/`key.pem` with proper certificates and update `config.json` paths.
- **Hot Reload**: Use `--reload` flag (web mode only) for development, but expect model reload delays.
