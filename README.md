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
        "no_speech_threshold": 0.8,
        "avg_logprob_cutoff": -0.8,
        "min_window_sec": 1.0,
        "max_window_sec": 10.0
    }
}
```

- **Vocabulary**: Specific terms used to bias the model toward correct spellings.
- **Corrections**: Automated search-and-replace for common phonetic errors.
- **Settings**: Tunable AI parameters for silence detection, confidence floors, and windowing.

## 📜 License

[MIT License](LICENSE)
