#!/usr/bin/env python3
"""
Voice Profile Enrollment Utility

Enroll new voice profiles from audio samples for speaker identification.
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from voice_profiles import voice_profile_manager


def load_audio(audio_path: str) -> np.ndarray:
    """Load audio file and resample to 16kHz mono."""
    import librosa
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    return audio


def extract_embedding(audio: np.ndarray, speaker_model) -> torch.Tensor:
    """Extract voice embedding from audio using SpeechBrain."""
    signal = torch.from_numpy(audio).unsqueeze(0)
    embeddings = speaker_model.encode_batch(signal)
    return embeddings.squeeze()


def main():
    parser = argparse.ArgumentParser(
        description="Manage voice profiles for speaker identification"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Enroll command
    enroll_parser = subparsers.add_parser("enroll", help="Enroll a new voice profile")
    enroll_parser.add_argument("--name", "-n", required=True, help="Profile name (speaker label)")
    enroll_parser.add_argument("--samples", "-s", nargs="+", required=True, 
                               help="Audio file paths (WAV, MP3, etc.)")
    enroll_parser.add_argument("--overwrite", "-o", action="store_true",
                               help="Overwrite existing profile")
    
    # List command
    subparsers.add_parser("list", help="List all voice profiles")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a voice profile")
    delete_parser.add_argument("--name", "-n", required=True, help="Profile name to delete")
    
    # Reload command
    subparsers.add_parser("reload", help="Reload profiles from disk")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "list":
        profiles = voice_profile_manager.list_profiles()
        if profiles:
            print(f"Voice Profiles ({len(profiles)}):")
            for name in sorted(profiles):
                print(f"  - {name}")
        else:
            print("No voice profiles registered.")
        return
    
    if args.command == "reload":
        count = voice_profile_manager.load_profiles()
        print(f"Reloaded {count} voice profiles")
        return
    
    if args.command == "delete":
        if voice_profile_manager.delete(args.name):
            print(f"Deleted profile: {args.name}")
        else:
            print(f"Failed to delete profile: {args.name}")
        return
    
    if args.command == "enroll":
        print(f"Enrolling voice profile: {args.name}")
        print(f"Loading SpeechBrain speaker model...")
        
        # Load SpeechBrain model
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            speaker_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )
        except Exception as e:
            print(f"Error loading speaker model: {e}")
            sys.exit(1)
        
        # Extract embeddings from each sample
        embeddings = []
        for sample_path in args.samples:
            try:
                print(f"  Processing: {sample_path}")
                audio = load_audio(sample_path)
                
                if len(audio) < 16000:  # Less than 1 second
                    print(f"    Warning: Audio too short ({len(audio)/16000:.1f}s), skipping")
                    continue
                
                emb = extract_embedding(audio, speaker_model)
                embeddings.append(emb)
                print(f"    Extracted embedding (duration: {len(audio)/16000:.1f}s)")
                
            except Exception as e:
                print(f"    Error processing {sample_path}: {e}")
        
        if not embeddings:
            print("No valid embeddings extracted. Enrollment failed.")
            sys.exit(1)
        
        # Enroll the profile
        if voice_profile_manager.enroll(args.name, embeddings, overwrite=args.overwrite):
            print(f"\n✓ Successfully enrolled '{args.name}' from {len(embeddings)} samples")
        else:
            print(f"\n✗ Failed to enroll '{args.name}'")
            sys.exit(1)


if __name__ == "__main__":
    main()
