# Test Suites TODO

## 1. Session Persistence And Lifecycle

- [x] `TranscriptionSession.add_transcript()` caps history at 2000 and updates `updated_at`.
- [x] `save_session()` sanitizes filenames and preserves session name.
- [x] `load_session_from_file()` restores transcript data correctly.
- [x] `list_sessions()` sorts by newest `updated_at`.
- [x] `archive_session()`, `restore_session()`, `list_archived_sessions()`, and `delete_session()`.
- [x] `cleanup_old_sessions()` archives only stale sessions.
- [x] `get_session_rollover_status()` for disabled rollover, time trigger, and count trigger.

## 2. Config Loading And Validation

- [x] Missing config file uses defaults.
- [x] Invalid JSON does not crash.
- [x] User settings merge without dropping defaults.
- [x] Validation catches invalid types, invalid ranges, bad `device`, and bad `model_size`.
- [x] `session_management` validation catches invalid rollover/archive values.
- [x] Config isolation between tests, since `TRANSCRIPTION_CONFIG` is global.

## 3. Transcription Engine Unit Tests With Mocked Whisper

- [ ] Silence returns only context tail.
- [ ] Peak normalization and RMS/noise-reduction branches.
- [ ] Prompt construction from `initial_prompt`, vocabulary, and previous transcript.
- [ ] Hallucination filters reject filler, subtitle credits, music notation, and empty punctuation.
- [ ] Confidence filters reject low-confidence and high `no_speech_prob` segments.
- [ ] Corrections apply case-insensitively.
- [ ] `_merge_segments()` merge and timeout behavior.
- [ ] CUDA failure path falls back to CPU and retries once without actual GPU.

## 4. Speaker And Voice Profile Tests

- [ ] Unknown embeddings create stable `Speaker N` labels.
- [ ] Similar embeddings return existing speakers.
- [ ] Dissimilar embeddings create new speakers.
- [ ] Configured `diarization_threshold` changes clustering behavior.
- [ ] Profile match takes priority over dynamic clustering.
- [ ] `VoiceProfileManager.enroll()` averages and normalizes embeddings.
- [ ] `load_profiles()` skips malformed profile files.
- [ ] `delete()` removes file and in-memory profile.
- [ ] `match()` respects `voice_match_threshold`.

## 5. Web API Session Tests

- [ ] `GET /api/session/current` with and without active session.
- [ ] `GET /api/sessions`.
- [ ] `POST /api/sessions?name=...`.
- [ ] `GET /api/sessions/{name}` success and 404.
- [ ] `POST /api/sessions/{name}/save` with no active session gives 400.
- [ ] `DELETE /api/sessions/{name}` success and 404.
- [ ] Archive and restore endpoints.
- [ ] `POST /api/sessions/compare` success, invalid mode, and missing session.
- [ ] `GET /api/engine/status` with mocked engine status.

## 6. Session Comparison Tests

- [ ] Reject fewer than 2 or more than 3 sessions.
- [ ] Missing session raises clear `ValueError`.
- [ ] `merged` mode sorts transcripts chronologically and adds `session_name`.
- [ ] `side-by-side` mode groups by session.
- [ ] `speaker_filter` applies in both modes.
- [ ] Stats are correct for empty and non-empty comparisons.

## 7. Real-Time Loop Behavior

- [ ] `audio_callback()` puts chunks on queues.
- [ ] Noise floor gates backend buffer writes.
- [ ] Silence sends periodic volume updates.
- [ ] Backlog handling drops oldest only when `max_queue_size > 0`.
- [ ] Sentinel `timestamp is None` flushes active buffer.
- [ ] VAD `False` triggers transcription after minimum window.
- [ ] Low-volume aggregate buffer is discarded.

## 8. Email Alert And Retry Tests

- [ ] `AudioBufferManager` timestamp offset initialization and drift resync.
- [ ] `extract_clip()` returns valid WAV bytes for overlapping chunks.
- [ ] Empty/missing clip returns `None`.
- [ ] `EmailRetryQueue` loads bad/missing queue files safely.
- [ ] Queue max size drops oldest.
- [ ] Backoff calculation clamps to max.
- [ ] `mark_success()` removes queued email and deletes audio attachment.
- [ ] `mark_failure()` increments retry count and removes after max retries.

## 9. Frontend Browser Tests

- [x] Dashboard loads and initial session fetch renders transcripts.
- [x] WebSocket transcript message appends one transcript row.
- [ ] Binary audio message updates visualizer state without crashing.
- [ ] Speaker filtering hides/shows expected rows.
- [ ] Watchwords highlight matching transcript text.
- [ ] Theme and audio-processing settings persist in `localStorage`.
- [ ] Backend audio fetch failure shows a playback error state.
- [ ] Reconnect behavior after WebSocket close.

## 10. Smoke/Integration Suite

- [ ] FastAPI app starts.
- [ ] `/ping`, `/api/audio/buffer-status`, and `/api/session/current` respond.
- [ ] WebSocket connects and receives queued JSON/audio broadcasts.
- [ ] No real microphone, Whisper, CUDA, or SpeechBrain required.
