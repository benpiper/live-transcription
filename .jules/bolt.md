## 2025-03-05 - Audio Buffer and RMS Optimizations
**Learning:** In audio processing loops, allocating intermediate arrays like `audio_data**2` for RMS calculations creates significant memory overhead and slowness. Also, relying on `sorted(dict.items())` or full list comprehensions on chronological data structures like `AudioBuffer` scales poorly as the buffer grows to handle 24-hour retention.
**Action:** Use `np.linalg.norm(audio_data) / np.sqrt(len(audio_data))` to leverage in-place fast C/BLAS routines without intermediate allocations. For dictionaries holding chronologically inserted data (in Python 3.7+), utilize their guaranteed insertion order with early `break` loops to turn O(N log N) and O(N) operations into amortized O(1).

## 2026-05-11 - PyTorch Tensor Operations Optimization
**Learning:** For PyTorch operations involving comparisons across multiple items (like `cosine_similarity` for speaker identification), using Python loops introduces significant overhead due to repetitive Python-to-C++ transitions.
**Action:** Use batched tensor operations (e.g., `torch.stack`) to perform comparisons in a single vectorized call instead of looping. Always check that the sequence is not empty before calling `torch.stack()` to prevent runtime errors.
