# System Agents

This document defines the specialized "agents" (components) that form the live-transcription system. Each agent has specific responsibilities and coordinates with others through a shared multi-threaded pipeline.

---

## 🎙️ Transcription Agent

**Primary Goal**: Convert raw audio chunks into high-accuracy text transcripts.

- **Responsibilities**:
  - Managing Whisper model lifecycle (loading, offloading).
  - Executing inference using `faster-whisper`.
  - Implementing domain-specific vocabulary biasing and phonetic corrections.
  - Applying hallucination filters (avg_logprob, no_speech_prob, compression_ratio).
  - Handling GPU-to-CPU fallback and recovery.
- **Key Modules**: [transcription_engine.py](file:///home/user/live-transcription/transcription_engine.py)

---

## 🎧 Audio Processing Agent

**Primary Goal**: Enhance audio quality and filter non-speech signals to improve transcription accuracy.

- **Responsibilities**:
  - Voice Activity Detection (VAD) using Silero VAD to filter background noise and silence.
  - Spectral Noise Reduction to clean noisy radio channels.
  - RMS Normalization to maintain consistent audio levels.
- **Key Modules**: [audio_processing.py](file:///home/user/live-transcription/audio_processing.py)

---

## 👤 Speaker Management Agent

**Primary Goal**: Identify and distinguish between different speakers in the audio stream.

- **Responsibilities**:
  - Generating 192-dimensional voice embeddings using ECAPA-TDNN.
  - Clustering audio segments into distinct speakers (diarization).
  - Matching real-time audio against pre-enrolled voice profiles using cosine similarity.
  - Managing speaker session state and metadata.
- **Key Modules**: [speaker_manager.py](file:///home/user/live-transcription/speaker_manager.py), [voice_profiles.py](file:///home/user/live-transcription/voice_profiles.py)

---

## 📬 Alerting & Notification Agent

**Primary Goal**: Monitor live transcripts for security-critical keywords or patterns and notify stakeholders.

- **Responsibilities**:
  - Regex and word-boundary pattern matching via a rules engine.
  - Intelligent deduplication and cooldown management to prevent alert storms.
  - Formatting and sending real-time email alerts.
  - Contextual filtering based on speaker identity or transcription confidence.
- **Key Modules**: [alert_rules.py](file:///home/user/live-transcription/alert_rules.py), [email_alert.py](file:///home/user/live-transcription/email_alert.py)

---

## 📁 Session Management Agent

**Primary Goal**: Maintain historical records of transcriptions and manage system state across restarts.

- **Responsibilities**:
  - Persisting daily or shift-based transcription logs to JSON.
  - Managing session rollover based on time or transcript volume.
  - Automatic archiving and cleanup of old session data.
- **Key Modules**: [session.py](file:///home/user/live-transcription/session.py)

---

## 🌐 Broadcasting Agent

**Primary Goal**: Deliver real-time data to end-users and provide a control interface.

- **Responsibilities**:
  - Orchestrating the multi-threaded pipeline.
  - Managing WebSocket connections for low-latency data streaming.
  - Serving the web dashboard and REST API.
  - Providing real-time waveform visualization and playback.
- **Key Modules**: [web_server.py](file:///home/user/live-transcription/web_server.py), [real_time_transcription.py](file:///home/user/live-transcription/real_time_transcription.py)
