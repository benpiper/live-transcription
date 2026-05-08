import torch
import time
from torch.nn.functional import cosine_similarity

def benchmark():
    num_speakers = 50
    embedding = torch.randn(192)
    speakers = [(torch.randn(192), f"Speaker {i}") for i in range(num_speakers)]
    profile_embs = {f"Speaker {i}": torch.randn(192) for i in range(num_speakers)}

    # Looped
    start = time.perf_counter()
    for _ in range(100):
        max_sim = -1
        best_label = None
        for spk_emb, label in speakers:
            sim = cosine_similarity(embedding.unsqueeze(0), spk_emb.unsqueeze(0)).item()
            if sim > max_sim:
                max_sim = sim
                best_label = label
    t1 = time.perf_counter() - start

    # Batched
    start = time.perf_counter()
    for _ in range(100):
        spk_embs = torch.stack([spk[0] for spk in speakers])
        sims = cosine_similarity(embedding.unsqueeze(0), spk_embs)
        max_sim_val, max_idx = torch.max(sims, dim=0)
        max_sim = max_sim_val.item()
        best_label = speakers[max_idx.item()][1]
    t2 = time.perf_counter() - start

    print(f"Looped: {t1*1000:.2f} ms")
    print(f"Batched: {t2*1000:.2f} ms")
    print(f"Speedup: {t1/t2:.2f}x")

benchmark()
