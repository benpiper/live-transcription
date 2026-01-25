# Live Transcription with Faster-Whisper

A high-performance, real-time audio transcription tool using the `faster-whisper` engine. This tool is designed for minimal latency, high accuracy, and smart audio handling.

## ✨ Features

- **Real-time Transcription**: Transcribes audio as you speak with minimal latency.
- **Visual Volume Meter**: Real-time ASCII volume bar in the console to monitor mic input.
- **Silence-Aware Chunking**: Intelligently waits for natural pauses in speech to transcribe, preventing mid-word cut-offs.
- **Context Overlap**: Prepends a small "tail" of previous audio to current chunks to improve recognition of word fragments.
- **Hallucination Suppression**: Advanced filtering using `no_speech_prob`, `avg_logprob`, and `compression_ratio` to eliminate "YouTube-style" ghosts (e.g., "Thanks for watching") during silence.
- **File Testing Mode**: Stream and transcribe local audio files (`.mp3`, `.wav`, etc.) at real-time speed to test settings.
- **High-Quality Resampling**: Built-in resampling using `librosa` to ensure all audio sources match the 16kHz requirement.

## 🛠 Prerequisites

Before running the script, ensure you have the necessary system-level dependencies for audio input.

### Linux
You may need to install `libportaudio2` and libsndfile:
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

### Real-time Microphone Transcription
Simply run the main script to start listening:
```bash
python3 real_time_transcription.py
```

### Testing with an Audio File
To simulate real-time transcription using a local file:
```bash
python3 real_time_transcription.py --input test.mp3
```

- **Timestamps**: Transcribed text appears with a header like `[14:20:05] Hello world.`
- **Volume**: The `[#####-----]` meter at the bottom shows real-time intensity and detection status.
- **Stop**: Press `Ctrl+C` to terminate the session safely.

## ⚙️ Configuration

You can modify these parameters at the top of `real_time_transcription.py`:

- **Model**: Default is `small.en`. You can use `tiny.en` for speed or `large-v3` for maximum accuracy.
- **Inference device**: Currently set to `cpu`. Change to `cuda` if you have a compatible NVIDIA GPU and the necessary drivers.
- **Thresholds**: Adjust `0.002` (volume threshold) or `no_speech_threshold` to fine-tune sensitivity for your specific environment.

## 📜 License

[MIT License](LICENSE)
