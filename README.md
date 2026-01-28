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
| `--list-devices` | | List available audio devices and exit |
| `--device` | | Selected input device ID |
| `--debug-robo` | | Print robotic voice detection stats for debugging |

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
| `detect_bots` | `false` | `true` / `false` | When enabled, analyzes acoustic profiles to identify synthetic/robotic voices (AI dispatchers) and labels them as `[Dispatcher (Bot)]`. |

## 📜 License

[MIT License](LICENSE)
