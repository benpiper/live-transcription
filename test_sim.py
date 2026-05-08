import torch
from torch.nn.functional import cosine_similarity

# Mock embeddings
embedding = torch.randn(192)
speakers = [
    (torch.randn(192), "Speaker 1"),
    (torch.randn(192), "Speaker 2"),
    (torch.randn(192), "Speaker 3"),
]

# Old way
max_sim1 = -1
best_label1 = None
for spk_emb, label in speakers:
    sim = cosine_similarity(embedding.unsqueeze(0), spk_emb.unsqueeze(0)).item()
    if sim > max_sim1:
        max_sim1 = sim
        best_label1 = label

# New way
spk_embs = torch.stack([spk[0] for spk in speakers])
sims = cosine_similarity(embedding.unsqueeze(0), spk_embs)
max_sim_val, max_idx = torch.max(sims, dim=0)
max_sim2 = max_sim_val.item()
best_label2 = speakers[max_idx.item()][1]

print(f"Old: {max_sim1:.4f} {best_label1}")
print(f"New: {max_sim2:.4f} {best_label2}")
assert abs(max_sim1 - max_sim2) < 1e-6
assert best_label1 == best_label2
print("Success!")
