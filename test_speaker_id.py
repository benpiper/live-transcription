import torch
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier

def test_embedding():
    print("Loading speaker encoder...")
    # Use CPU since we want it to be compatible and fast enough for single segments
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cpu"})
    print("Encoder loaded.")

    # Create dummy audio (16kHz, 2 seconds)
    signal = torch.randn(1, 16000 * 2)
    
    # Get embedding
    embeddings = classifier.encode_batch(signal)
    print(f"Embedding shape: {embeddings.shape}")
    
    # Cosine similarity test
    emb1 = embeddings[0]
    emb2 = torch.randn_like(emb1)
    
    cos = torch.nn.CosineSimilarity(dim=-1)
    sim = cos(emb1, emb2)
    print(f"Similarity with noise: {sim.item()}")

if __name__ == "__main__":
    test_embedding()
