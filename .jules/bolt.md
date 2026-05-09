## 2025-03-05 - Audio Buffer and RMS Optimizations
**Learning:** In audio processing loops, allocating intermediate arrays like `audio_data**2` for RMS calculations creates significant memory overhead and slowness. Also, relying on `sorted(dict.items())` or full list comprehensions on chronological data structures like `AudioBuffer` scales poorly as the buffer grows to handle 24-hour retention.
**Action:** Use `np.linalg.norm(audio_data) / np.sqrt(len(audio_data))` to leverage in-place fast C/BLAS routines without intermediate allocations. For dictionaries holding chronologically inserted data (in Python 3.7+), utilize their guaranteed insertion order with early `break` loops to turn O(N log N) and O(N) operations into amortized O(1).

## 2025-05-09 - PyTorch Vectorized Cosine Similarity
**Learning:** PyTorch comparison operations (e.g. `cosine_similarity`) in Python loops suffer from C++ transition overhead that degrades real-time backend throughput.
**Action:** Use batched tensor functions like `torch.stack` and broadcasting. When replacing loop structures with vectors, verify that the condition inherently handles empty sequences natively if not actively checked, or add `if not sequence` guards explicitly to avoid throwing `RuntimeError: stack expects a non-empty TensorList`.
