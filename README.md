# Live Transcription with Faster-Whisper

A high-performance, real-time audio transcription tool using the `faster-whisper` engine. Designed for minimal latency, high accuracy, and smart audio handling.

## ✨ Features

- **Real-time Transcription**: Transcribes audio as you speak with minimal latency.
- **Web Dashboard**: A modern, glassmorphic web interface to monitor transcription feeds, audio volume, and performance telemetry (Latency, Buffer).
- **Speaker Identification (Diarization)**: Automatically identifies and labels different speakers (e.g., `[Speaker 1]`) using ECAPA-TDNN voice embeddings.
- **Robotic Voice Detection**: Automatically identifies synthetic/computer-generated voices (e.g., dispatchers) and labels them as `[Dispatcher (Bot)]`.
- **Advanced Timestamps**: All outputs include full date/time stamps (e.g., `[2026-01-25 23:46:16]`) for accurate logging.
- **Silence-Aware Chunking**: Intelligently waits for natural pauses in speech to transcribe, preventing mid-word cut-offs.
- **Hallucination Suppression**: Advanced filtering using `no_speech_prob`, `avg_logprob`, and `compression_ratio` to eliminate "YouTube-style" ghosts.
- **Custom Vocabulary & Corrections**: Bias the AI toward specific terms and apply automated phonetic corrections via a JSON config file.
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
| `--reload` | | Auto-reload backend on code changes (Web mode only) |
| `--debug-robo` | | Print robotic voice detection stats for debugging |

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

The application is built with a multi-threaded, asynchronous design to ensure real-time performance and low-latency audio handling.

### 1. Data Flow & Threading

The system operates using three primary threads/processes coordinating via thread-safe queues:

- **Audio Capture Thread**: Captures raw PCM audio from the microphone (using `sounddevice`) or reads from a file and pushes it into the `audio_queue`.
- **Transcription Engine Thread**: 
  - Monitors the `audio_queue` for incoming data.
  - Applies a **Noise Floor Gate** to skip silence.
  - Performs **VAD-aware chunking** to find natural speech boundaries.
  - Generates transcriptions using `faster-whisper`.
  - Performs **Speaker Diarization** and **Robotic Voice Detection** on-the-fly.
  - Pushes processed results into the `broadcast_queue`.
- **FastAPI/Uvicorn Backend (Main Thread)**:
  - Serves the static web dashboard.
  - Manages WebSocket connections for real-time clients.
  - **Broadcaster Worker**: Continuously drains the `broadcast_queue` and relays results to all connected clients.

### 2. WebSocket Communication

The communication between the Web UI and Backend uses a single WebSocket connection on `/ws`:

| Data Type | format | Content |
| :--- | :--- | :--- |
| **Transcription** | JSON | The text, speaker label, timestamp, and performance telemetry. |
| **Audio Stream** | Binary | Raw `Float32Array` chunks sent at 16,000Hz for live visualization and history playback. |

### 3. Frontend Implementation

- **Web Audio API**: Used to manage the live audio stream, synchronize playback of history segments, and handle volume normalization.
- **Canvas API**: Provides the high-frequency volume visualizer reacting to the binary audio stream.
- **LocalStorage**: Persists your theme settings, watchwords, and recent transcription history (text-only) across page refreshes.

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
| `compute_type` | `auto` | `float16`, `int8`, `auto` | Precision for computation. `float16` is recommended for GPU; `int8` for CPU. |
| `cpu_threads` | `4` | `1`+ | Number of threads used for CPU processing. |
| `no_speech_threshold` | `0.6` | `0.0` - `1.0` | **(Higher = More Strict)**. Internal VAD sensitivity. If the probability that a segment is silence is *higher* than this, it's ignored. Increase if you see "phantom" text; decrease if the model skips soft speech. |
| `log_prob_threshold` | `-1.0` | `-inf` - `0.0` | **(Higher = More Strict)**. Whisper internal retry threshold. If tokens fall *below* this, the model retries with higher randomness. **Warning**: High values can increase latency by 5x due to re-processing attempts. |
| `compression_ratio_threshold` | `2.4` | `1.0` - `inf` | **(Lower = More Strict)**. Detects repetitive "loops." If the ratio is *higher* than this, the segment is rejected. Lower values (e.g., `1.8`) are aggressive at killing hallucinations. |
| `avg_logprob_cutoff` | `-0.8` | `-inf` - `0.0` | **(Higher = More Strict)**. The final confidence cut-off for the transcribed sentence. Closer to `0` is better. Lower to `-1.5` if you want "best guess" output for noisy audio. |
| `no_speech_prob_cutoff` | `0.2` | `0.0` - `1.0` | **(Lower = More Strict)**. Maximum probability that a segment is silence. If the model's "silence score" is *higher* than this, it's rejected. |
| `extreme_confidence_cutoff` | `-0.4` | `-inf` - `0.0` | **(Higher = More Strict)**. If a segment's confidence is *better* (higher) than this, it bypasses the `no_speech_prob` check completely. (Prevents short commands from being filtered). |
| `min_window_sec` | `1.0` | `0.0`+ | How long to wait for a natural pause before attempting transcription. |
| `max_window_sec` | `10.0` | `min_window_sec`+ | **(Lower = Faster Updates)**. Force transcription if no pause is found. Lowering to `5.0` provides faster UI updates but may cut off speakers mid-sentence. |
| `beam_size` | `5` | `1` - `20` | **(Higher = More Accurate)**. Number of parallel search paths. `5` is balanced; `10` is very accurate but doubles CPU/GPU load. |
| `min_silence_duration_ms` | `500` | `0`+ | How long a silence gap must be to trigger the end of a sentence. |
| `initial_prompt` | `""` | string | Domain context to guide Whisper. Example: `"Emergency dispatch radio. 10-codes, unit numbers, street addresses."` Improves accuracy for specialized audio. |
| `max_queue_size` | `0` | `0`+ | Maximum audio chunks before dropping old ones. `0` = never drop (recommended). Set to `20+` if you need real-time priority over completeness. |
| `detect_bots` | `false` | `true` / `false` | When enabled, analyzes pitch and spectral features to identify synthetic/TTS voices and labels them as `[Dispatcher (Bot)]`. Works with or without `--diarize`. |
| `debug_robo` | `false` | `true` / `false` | Prints real-time robot detection debug output showing pitch_std and flatness values for tuning. |
| `robot_pitch_std_threshold` | `8.0` | `0.0`+ | **(Lower = More Strict)**. Flags voices with pitch standard deviation below this as monotone/robotic. Human speech typically varies 20-100+ Hz. |
| `robot_flatness_threshold` | `0.012` | `0.0` - `1.0` | **(Higher = More Strict)**. Flags unnaturally "clean" audio (low spectral flatness) as synthetic. TTS voices often have flatness < 0.01 due to lack of breath noise. |
| `diarization_threshold` | `0.35` | `0.0` - `1.0` | **(Lower = More Inclusive)**. Sensitivity for merging voice embeddings. Lowering this value (e.g., `0.28`) will merge more speakers together and reduce over-detection. |
| `min_speaker_samples` | `16000` | `8000`+ | Minimum audio samples (at 16kHz) required for speaker identification. `16000` = 1 second, `32000` = 2 seconds. Higher values increase accuracy but may miss short utterances. |
| `noise_floor` | `0.001` | `0.0` - `1.0` | **(Higher = More Strict)**. The minimum audio volume required to trigger a transcription. Blocks below this level skip all heavy processing to save CPU. |

## 📜 License

[MIT License](LICENSE)
