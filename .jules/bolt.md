## 2025-03-05 - Audio Buffer and RMS Optimizations
**Learning:** In audio processing loops, allocating intermediate arrays like `audio_data**2` for RMS calculations creates significant memory overhead and slowness. Also, relying on `sorted(dict.items())` or full list comprehensions on chronological data structures like `AudioBuffer` scales poorly as the buffer grows to handle 24-hour retention.
**Action:** Use `np.linalg.norm(audio_data) / np.sqrt(len(audio_data))` to leverage in-place fast C/BLAS routines without intermediate allocations. For dictionaries holding chronologically inserted data (in Python 3.7+), utilize their guaranteed insertion order with early `break` loops to turn O(N log N) and O(N) operations into amortized O(1).

## 2026-05-13 - [PyTorch Tensor Batching for Cosine Similarity]
**Learning:** Found O(N) Python loops calculating `cosine_similarity` for each item individually against known speakers/profiles. This causes slow execution due to PyTorch dispatches for every loop iteration.
**Action:** Replace item-by-item loops with batched similarity calculations via `torch.stack()` and `torch.argmax(similarities)`. Ensure the sequence is not empty before stacking. For dictionary maps like `self.profiles.items()`, rely on Python's insertion-order guarantee to safely create parallel lists of keys and values.
