# Live Transcription with Faster-Whisper

A high-performance, real-time audio transcription tool using the `faster-whisper` engine. Designed for minimal latency, high accuracy, and smart audio handling.

## ✨ Features

- **Real-time Transcription**: Transcribes audio as you speak with minimal latency.
- **Visual Volume Meter**: Real-time ASCII volume bar in the console to monitor mic input.
- **Silence-Aware Chunking**: Intelligently waits for natural pauses in speech to transcribe, preventing mid-word cut-offs.
- **Context Overlap**: Prepends a small "tail" of previous audio to current chunks to improve recognition of word fragments.
- **Hallucination Suppression**: Advanced filtering using `no_speech_prob`, `avg_logprob`, and `compression_ratio` to eliminate "YouTube-style" ghosts (e.g., "Thanks for watching") during silence.
- **Speaker Identification (Diarization)**: Automatically identifies and labels different speakers (e.g., `[Speaker 1]`, `[Speaker 2]`) using ECAPA-TDNN voice embeddings.
- **Custom Vocabulary & Corrections**: Bias the AI towards specific terms and apply automated phonetic corrections via a JSON config file.
- **File Output**: Automatically save all transcriptions to a timestamped text file.
- **File Testing Mode**: Stream and transcribe local audio files (`.mp3`, `.wav`, etc.) at real-time speed to test settings.
- **High-Quality Resampling**: Built-in resampling using `librosa` to ensure all audio sources match the 16kHz requirement.

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

### Real-time Microphone Transcription
```bash
python3 real_time_transcription.py --diarize --config config.json
```

### Full Feature Set (with Output File)
```bash
python3 real_time_transcription.py --diarize --config config.json --output transcript.txt
```

### Testing with an Audio File
```bash
python3 real_time_transcription.py --input test.mp3 --diarize --config config.json
```

## 🔧 Command Line Arguments

| Argument | Shorthand | Description |
| :------- | :-------- | :---------- |
| `--input` | | Path to an audio file to test instead of microphone |
| `--config` | `-c` | Path to `config.json` for vocabulary and corrections |
| `--output` | `-o` | Path to save the transcribed text file |
| `--diarize` | `-d` | Enable real-time speaker identification |

## ⚙️ Custom Configuration (`config.json`)

You can provide a JSON file to help the AI with specific terminology:

```json
{
    "vocabulary": ["Faster-Whisper", "GitHub", "Summerville"],
    "corrections": {
        "git hub": "GitHub",
        "fast whisper": "Faster-Whisper"
    }
}
```

- **Vocabulary**: Words added to the `initial_prompt` to "bias" the model toward these spellings.
- **Corrections**: A mapping for automated search-and-replace after transcription (useful for common phonetic errors).
- **Settings**: Tunable AI parameters including:
    - `no_speech_threshold`: Sensitivity for silence detection (higher = more skeptical).
    - `compression_ratio_threshold`: Strictness against repetitive text (higher = more tolerant).
    - `avg_logprob_cutoff`: Confidence floor for a phrase (closer to 0 is higher confidence).

## 📜 License

[MIT License](LICENSE)
