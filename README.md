# Live Transcription with Faster-Whisper

A high-performance, real-time audio transcription tool using the `faster-whisper` engine. Designed for minimal latency, high accuracy, and smart audio handling.

## ✨ Features

- **Real-time Transcription**: Transcribes audio as you speak with minimal latency.
- **Optimized Web Dashboard**: A modern interface designed for long-running sessions, using efficient DOM management to remain responsive even with thousands of transcription entries.
- **Speaker Identification (Diarization)**: Automatically identifies and labels different speakers (e.g., `[Speaker 1]`) using ECAPA-TDNN voice embeddings.
- **Voice Profiles**: Pre-register known speakers (dispatchers, IVR systems, key personnel) for instant identification by name.
- **Advanced Timestamps**: All outputs include full date/time stamps (e.g., `[2026-01-25 23:46:16]`) for accurate logging.
- **Silence-Aware Chunking**: Intelligently waits for natural pauses in speech to transcribe, preventing mid-word cut-offs.
- **Hallucination Suppression**: Advanced filtering using `no_speech_prob`, `avg_logprob`, and `compression_ratio` to eliminate "YouTube-style" ghosts.
- **Custom Vocabulary & Corrections**: Bias the AI toward specific terms and apply automated phonetic corrections via a JSON config file.
- **HTTPS & Secure WebSockets**: Optional SSL support for secure web dashboard access and data streaming.
- **GPU/CPU Presets & Fallback**: Device-optimized presets with automatic GPU→CPU fallback on errors and periodic GPU recovery.
- **File Output**: Automatically save all transcriptions to a text file with real-time flushing.

## 🛠 Prerequisites

Before running the script, ensure you have the necessary system-level dependencies for audio input.

### Linux
You may need to install `libportaudio2` and `libsndfile1`:
```bash
sudo apt-get install libportaudio2 libsndfile1
```

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd live-transcription
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Real-time Microphone Transcription (CLI + Web)
```bash
python3 real_time_transcription.py --diarize --web --config config.json
```

### Full Feature Set
```bash
python3 real_time_transcription.py --diarize --web --config config.json --output transcript.txt
```

### Testing with an Audio File
```bash
python3 real_time_transcription.py --input test.mp3 --diarize --web --config config.json
```

## 🔧 Command Line Arguments

| Argument | Shorthand | Description |
| :------- | :-------- | :---------- |
| `--input` | | Path to an audio file to test instead of microphone |
| `--config` | `-c` | Path to `config.json` for vocabulary, corrections, and settings |
| `--output` | `-o` | Path to save the transcribed text file |
| `--diarize` | `-d` | Enable real-time speaker identification |
| `--web` | `-w` | Enable the web dashboard |
| `--port` | | Web dashboard port (default: 8000) |
| `--session` | `-s` | Load or create a named session for transcript storage |
| `--list-sessions` | | List all saved sessions and exit |
| `--list-devices` | | List available audio devices and exit |
| `--device` | | Selected input device ID |
| `--ssl` | | Enable HTTPS/SSL for the web dashboard |
| `--reload` | | Auto-reload backend on code changes (Web mode only) |

## 📁 Sessions

Sessions allow you to save and resume transcription work by name.

### Create or Load a Session
```bash
python3 real_time_transcription.py --session "Morning Shift" --web --config config.json
```

### List Saved Sessions
```bash
python3 real_time_transcription.py --list-sessions
```

### REST API Endpoints
| Method | Endpoint | Description |
| :---: | :--- | :--- |
| GET | `/api/sessions` | List all saved sessions |
| POST | `/api/sessions?name=Name` | Create new session |
| GET | `/api/sessions/{name}` | Get session transcripts |
| POST | `/api/sessions/{name}/save` | Save current session |
| DELETE | `/api/sessions/{name}` | Delete session |

Sessions are stored as JSON files in the `sessions/` directory.


## 🏗️ System Architecture

The application follows a **client-server architecture** with a Python backend and JavaScript web frontend, designed for real-time audio processing with minimal latency.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              BACKEND (Python)                           │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌────────────────────┐    ┌───────────────────┐   │
│  │ Audio Input  │───▶│ Transcription      │───▶│ FastAPI Server    │   │
│  │ (sounddevice)│    │ Engine (Whisper)   │    │ (uvicorn)         │   │
│  └──────────────┘    │ + Diarization      │    │ ├─ REST API       │   │
│                      │ + Voice Profiles   │    │ └─ WebSocket /ws  │   │
│                      └────────────────────┘    └─────────┬─────────┘   │
│                                                          │             │
│  Session Storage: sessions/*.json (transcripts + metadata)             │
└──────────────────────────────────────────────────────────┼─────────────┘
                                                           │ WebSocket
                                                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          FRONTEND (JavaScript)                          │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │ Real-time Feed   │  │ Audio Visualizer │  │ Session Management   │  │
│  │ (transcript DOM) │  │ (Canvas API)     │  │ (load/save/switch)   │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘  │
│                                                                         │
│  Audio Storage: IndexedDB (on-demand loading for memory efficiency)        │
│  Settings: localStorage (theme, watchwords, display preferences)            │
└─────────────────────────────────────────────────────────────────────────┘
```

### Backend Components

| Component | Technology | Purpose |
|:----------|:-----------|:--------|
| **Audio Capture** | `sounddevice` | Captures realtime PCM audio from microphone or file |
| **Transcription Engine** | `faster-whisper` | Converts audio to text with VAD-aware chunking |
| **Speaker Diarization** | SpeechBrain ECAPA-TDNN | Identifies and labels different speakers |
| **Voice Profiles** | JSON embeddings | Matches known speakers by pre-registered voice signatures |
| **Web Server** | FastAPI + Uvicorn | Serves static files, REST API, and WebSocket |
| **Session Storage** | JSON files | Persists transcript text and metadata on server |

### Frontend Components

| Component | Technology | Purpose |
|:----------|:-----------|:--------|
| **Transcript Feed** | Vanilla JS + DOM | Displays live transcripts with speaker labels |
| **Audio Visualizer** | Canvas API | Real-time waveform visualization |
| **Audio Playback** | Web Audio API | Replay individual transcript segments |
| **Audio Storage** | IndexedDB | Browser-side audio blob storage with on-demand loading to minimize memory footprint |
| **Settings** | localStorage | Persistent preferences (theme, watchwords, scroll lock, etc.) |

### Data Flow

1. **Audio Capture Thread** → Captures PCM audio, pushes to `audio_queue`
2. **Transcription Thread** → Processes audio, generates text, pushes to `broadcast_queue`
3. **WebSocket Broadcaster** → Relays transcripts + raw audio to connected clients
4. **Frontend** → Renders transcripts, stores audio for playback, syncs with server session

### WebSocket Protocol (`/ws`)

| Direction | Type | Content |
|:----------|:-----|:--------|
| Server → Client | JSON | Transcript: `{type, timestamp, origin_time, duration, speaker, text, confidence}` |
| Server → Client | Binary | Raw `Float32Array` audio chunks (16kHz) for visualization |

## ⚙️ Custom Configuration (`config.json`)

You can provide a JSON file to help the AI with specific terminology and behavior:

```json
{
    "vocabulary": ["Charleston", "Summerville", "EMS ops", "10-4"],
    "corrections": {
        "git hub": "GitHub",
        "EMSOX": "EMS ops"
    },
    "settings": {
        "no_speech_threshold": 0.5,
        "avg_logprob_cutoff": -0.8,
        "min_window_sec": 1.0,
        "max_window_sec": 10.0,
        "min_silence_duration_ms": 500
    }
}
```

- **Vocabulary**: Specific terms used to bias the model toward correct spellings.
- **Corrections**: Automated search-and-replace for common phonetic errors.
- **Settings**: Tunable AI parameters for fine-tuning the transcription engine:

| Parameter | Default | Range | Description |
| :--- | :--- | :--- | :--- |
| `model_size` | `medium.en` | `tiny`, `base`, `small`, `medium`, `large-v3` | The Whisper model size. Larger models are more accurate but slower and require more VRAM. |
| `device` | `auto` | `auto`, `cuda`, `cpu` | Hardware device for execution. `cuda` requires an NVIDIA GPU and proper drivers. |
| `compute_type` | `auto` | `float16`, `int8_float16`, `int8`, `auto` | Precision for computation. `auto` selects the optimal type per device (GPU: `int8_float16`, CPU: `int8`). |
| `cpu_threads` | preset | `1`+ | Number of threads for CPU processing. Preset default: 4 (GPU), 8 (CPU). |
| `no_speech_threshold` | `0.6` | `0.0` - `1.0` | **(Higher = More Strict)**. Internal VAD sensitivity. If the probability that a segment is silence is *higher* than this, it's ignored. Increase if you see "phantom" text; decrease if the model skips soft speech. |
| `log_prob_threshold` | `-1.0` | `-inf` - `0.0` | **(Higher = More Strict)**. Whisper internal retry threshold. If tokens fall *below* this, the model retries with higher randomness. **Warning**: High values can increase latency by 5x due to re-processing attempts. |
| `compression_ratio_threshold` | `2.4` | `1.0` - `inf` | **(Lower = More Strict)**. Detects repetitive "loops." If the ratio is *higher* than this, the segment is rejected. Lower values (e.g., `1.8`) are aggressive at killing hallucinations. |
| `avg_logprob_cutoff` | `-0.8` | `-inf` - `0.0` | **(Higher = More Strict)**. The final confidence cut-off for the transcribed sentence. Closer to `0` is better. Lower to `-1.5` if you want "best guess" output for noisy audio. |
| `no_speech_prob_cutoff` | `0.2` | `0.0` - `1.0` | **(Lower = More Strict)**. Maximum probability that a segment is silence. If the model's "silence score" is *higher* than this, it's rejected. |
| `extreme_confidence_cutoff` | `-0.4` | `-inf` - `0.0` | **(Higher = More Strict)**. If a segment's confidence is *better* (higher) than this, it bypasses the `no_speech_prob` check completely. (Prevents short commands from being filtered). |
| `min_window_sec` | `1.0` | `0.0`+ | How long to wait for a natural pause before attempting transcription. |
| `max_window_sec` | `10.0` | `min_window_sec`+ | **(Lower = Faster Updates)**. Force transcription if no pause is found. Lowering to `5.0` provides faster UI updates but may cut off speakers mid-sentence. |
| `beam_size` | preset | `1` - `20` | **(Higher = More Accurate)**. Number of parallel search paths. Preset default: 6 (GPU), 3 (CPU). |
| `min_silence_duration_ms` | `500` | `0`+ | How long a silence gap must be to trigger the end of a sentence. |
| `initial_prompt` | `""` | string | Domain context to guide Whisper. Example: `"Emergency dispatch radio. 10-codes, unit numbers, street addresses."` Improves accuracy for specialized audio. |
| `max_queue_size` | `0` | `0`+ | Maximum audio chunks before dropping old ones. `0` = never drop (recommended). Set to `20+` if you need real-time priority over completeness. |
| `diarization_threshold` | `0.35` | `0.0` - `1.0` | **(Lower = More Inclusive)**. Sensitivity for merging voice embeddings. Lowering this value (e.g., `0.28`) will merge more speakers together and reduce over-detection. |
| `min_speaker_samples` | `16000` | `8000`+ | Minimum audio samples (at 16kHz) required for speaker identification. `16000` = 1 second, `32000` = 2 seconds. Higher values increase accuracy but may miss short utterances. |
| `noise_floor` | `0.001` | `0.0` - `1.0` | **(Higher = More Strict)**. The minimum audio volume required to trigger a transcription. Blocks below this level skip all heavy processing to save CPU. |
| `voice_profiles_dir` | `voice_profiles` | string | Directory containing voice profile JSON files. |
| `voice_match_threshold` | `0.7` | `0.0` - `1.0` | **(Higher = More Strict)**. Cosine similarity threshold for matching voice profiles. Higher values require closer matches. |
| `ssl_enabled` | `false` | `true`, `false` | Enable SSL for the web server by default. |
| `ssl_certfile` | `cert.pem` | string | Path to the SSL certificate file. |
| `ssl_keyfile` | `key.pem` | string | Path to the SSL private key file. |
| `gpu_recovery_interval_min` | `10` | `1`+ | Minutes between GPU recovery probes when running in CPU fallback mode. Set higher to reduce probe overhead. |

## 🎤 Voice Profiles

Pre-register known speakers for instant identification by name instead of generic "Speaker N" labels.

### Enroll a New Profile
```bash
python3 enroll_voice.py enroll --name "Dispatcher A" --samples dispatch1.wav dispatch2.wav
```

### List Profiles
```bash
python3 enroll_voice.py list
```

### Delete a Profile
```bash
python3 enroll_voice.py delete --name "Dispatcher A"
```

Profiles are stored as JSON files in the `voice_profiles/` directory. Each profile contains an averaged voice embedding extracted from the sample audio files.

## 📧 Email Alert Tool

A separate utility is provided to monitor transcripts and send email alerts when specific keywords are found.

### Setup

1. **Configure SMTP**: Copy `email_config.json.example` to `email_config.json` and fill in your settings.
   ```bash
   cp email_config.json.example email_config.json
   ```
2. **Run the Alert Tool**:
   ```bash
   python3 email_alert.py
   ```

3. **Test your configuration**:
   ```bash
   python3 email_alert.py --test
   ```

The tool connects to the live feed via WebSocket and applies a configurable rate limit (default 5 minutes) per keyword to avoid alert fatigue.

## 🔒 Security & HTTPS

To enable secure access to the web dashboard, use the `--ssl` flag:

```bash
python3 real_time_transcription.py --web --ssl
```

### Self-Signed Certificates
For local development, you can generate a self-signed certificate:
```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"
```
When using a self-signed certificate, your browser will display a warning. You must manually "Proceed" to access the UI.

### Production Certificates
For production use, place your `fullchain.pem` and `privkey.pem` (from Let's Encrypt or similar) in the project directory and update `config.json` accordingly.

## 📜 License

[MIT License](LICENSE)
