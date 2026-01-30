import os
import sys
import subprocess
import warnings
import logging

# Suppress library-level warnings (especially from torch/speechbrain)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*autocast.*")
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Add CUDA/cuDNN paths for faster-whisper (ctranslate2) if they exist
def setup_cuda_paths():
    """Adds all NVIDIA library paths to LD_LIBRARY_PATH and pre-loads them to ensure they are available."""
    import ctypes
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    base_search_paths = [
        os.path.expanduser(f"~/.local/lib/{python_version}/site-packages/nvidia"),
        os.path.expanduser(f"/usr/local/lib/{python_version}/dist-packages/nvidia"),
        "/usr/local/cuda/lib64",
    ]
    
    extra_paths = []
    for base in base_search_paths:
        if not os.path.exists(base):
            continue
        
        for root, dirs, files in os.walk(base):
            if 'lib' in dirs:
                lib_path = os.path.join(root, 'lib')
                if lib_path not in extra_paths:
                    extra_paths.append(lib_path)
    
    if extra_paths:
        # Update LD_LIBRARY_PATH
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        ld_parts = extra_paths + ([current_ld] if current_ld else [])
        os.environ["LD_LIBRARY_PATH"] = ":".join(ld_parts)
        
        # Pre-load critical libraries to ensure they are in the process space
        # This is often necessary because changing LD_LIBRARY_PATH mid-process is unreliable
        critical_libs = [
            "libcublas.so.12", "libcudnn.so.9", "libcudnn_ops.so.9", 
            "libcudnn_cnn.so.9", "libcudnn_adv.so.9", "libcublasLt.so.12"
        ]
        
        loaded_count = 0
        for lib_name in critical_libs:
            for p in extra_paths:
                full_path = os.path.join(p, lib_name)
                if os.path.exists(full_path):
                    try:
                        ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)
                        loaded_count += 1
                        break # Move to next critical lib
                    except Exception:
                        pass
        
        if loaded_count > 0:
            print(f"CUDA hardware acceleration: Pre-loaded {loaded_count} core libraries.")
        else:
            # Fallback check for v8/v11
            fallback_libs = ["libcublas.so.11", "libcudnn.so.8"]
            for lib_name in fallback_libs:
                 for p in extra_paths:
                    full_path = os.path.join(p, lib_name)
                    if os.path.exists(full_path):
                        try:
                            ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)
                            loaded_count += 1
                        except Exception: pass

    return extra_paths

setup_cuda_paths()

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import torch
import queue
import threading
from datetime import datetime
import argparse
import soundfile as sf
import librosa
import json
import re
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import base64

# Application configuration with defaults
TRANSCRIPTION_CONFIG = {
    "vocabulary": [],
    "corrections": {},
    "settings": {
        "device": "auto",
        "model_size": "medium.en",
        "compute_type": "auto",
        "no_speech_threshold": 0.8,
        "log_prob_threshold": -0.8,
        "compression_ratio_threshold": 2.4,
        "beam_size": 5,
        "min_silence_duration_ms": 500,
        "avg_logprob_cutoff": -0.8,
        "no_speech_prob_cutoff": 0.2,
        "extreme_confidence_cutoff": -0.4,
        "min_window_sec": 1.0,
        "max_window_sec": 5.0,
        "detect_bots": False,
        "cpu_threads": 4,
        "noise_floor": 0.001
    }
}

# Placeholder for the Whisper model
model = None

def load_model():
    """Initializes the Faster-Whisper model using config settings."""
    global model
    settings = TRANSCRIPTION_CONFIG["settings"]
    
    model_size = settings.get("model_size", "medium.en")
    device = settings.get("device", "auto")
    compute_type = settings.get("compute_type", "auto")
    num_threads = settings.get("cpu_threads", 4)

    # Auto-adjust compute type for CUDA if requested
    if device == "auto" or device == "cuda":
        # Check if CUDA is available via torch
        try:
            import torch
            if torch.cuda.is_available():
                if device == "auto": device = "cuda"
                if compute_type == "auto": compute_type = "float16"
                print(f"CUDA detected. Using GPU acceleration ({compute_type}).")
            else:
                if device == "auto": device = "cpu"
                if compute_type == "auto": compute_type = "int8"
                print("CUDA not available. Falling back to CPU.")
        except ImportError:
            if device == "auto": device = "cpu"
            if compute_type == "auto": compute_type = "int8"
            print("Torch not found. Defaulting to CPU.")
    elif device == "cpu" and compute_type == "auto":
        compute_type = "int8"

    print(f"Loading faster-whisper model ({model_size}) on {device}...")
    model = WhisperModel(
        model_size, 
        device=device, 
        compute_type=compute_type, 
        cpu_threads=num_threads
    )
    print(f"Model loaded (Threads: {num_threads}, Device: {device}, Type: {compute_type}).")

# Speaker Identification (Optional)
speaker_model = None

class SpeakerManager:
    def __init__(self, threshold=0.35):
        self.threshold = threshold
        self.speakers = [] # List of (embedding, label)
    
    def is_robotic_voice(self, audio_chunk):
        """Analyzes an audio chunk to see if it matches a robotic/synthetic profile."""
        try:
            # Flatten and ensure float32
            y = audio_chunk.flatten().astype(np.float32)
            
            # Need more audio for a reliable robotic profile
            if len(y) < SAMPLE_RATE * 0.6: return False, 0, 0, 0
            
            try:
                # Expanded range for female voices
                f0 = librosa.yin(y, fmin=60, fmax=600, frame_length=1024)
                f0 = f0[~np.isnan(f0)]
            except:
                return False, 0, 0, 0
                
            if len(f0) < 10: return False, 0, 0, 0
            
            f0_std = np.std(f0)
            f0_mean = np.mean(f0)
            flatness = np.mean(librosa.feature.spectral_flatness(y=y))
            
            # Refined Heuristics (Less Aggressive):
            # 1. Monotone: std < 8.0 (Human speech is rarely this stable)
            # 2. Noisy/Vocoded: flatness > 0.28 (Common in older digital dispatcher synths)
            # 3. 'Clean' High-Pitch AI: flatness < 0.015 and mean > 240 (Speaker 8/5 profile)
            
            is_robotic = False
            if f0_std < 8.0: 
                is_robotic = True
            elif flatness > 0.28: 
                is_robotic = True
            elif f0_mean > 240 and flatness < 0.015: 
                is_robotic = True
                
            return is_robotic, f0_mean, f0_std, flatness
                
            return is_robotic, f0_mean, f0_std, flatness
        except Exception:
            return False, 0, 0, 0

    def identify_speaker(self, embedding, audio_chunk=None):
        if not self.speakers:
            label = "Speaker 1"
            
            # Bot detection (if enabled)
            do_bot_detection = TRANSCRIPTION_CONFIG["settings"].get("detect_bots", False)
            if do_bot_detection:
                is_robo, *_ = self.is_robotic_voice(audio_chunk) if audio_chunk is not None else (False,)
                if is_robo:
                    label = "Dispatcher (Bot)"
                    
            self.speakers.append((embedding, label))
            return label
        
        # Compare with known speakers
        from torch.nn.functional import cosine_similarity
        max_sim = -1
        best_label = None
        best_idx = -1
        
        for i, (spk_emb, label) in enumerate(self.speakers):
            sim = cosine_similarity(embedding.unsqueeze(0), spk_emb.unsqueeze(0)).item()
            if sim > max_sim:
                max_sim = sim
                best_label = label
                best_idx = i
        
        if max_sim > self.threshold:
            # Re-evaluation logic:
            # If the current label is generic ("Speaker X") AND this chunk looks robotic,
            # upgrade the label for all future instances of this speaker.
            do_bot_detection = TRANSCRIPTION_CONFIG["settings"].get("detect_bots", False)
            if do_bot_detection and best_label.startswith("Speaker"):
                is_robo, *_ = self.is_robotic_voice(audio_chunk) if audio_chunk is not None else (False,)
                if is_robo:
                    self.speakers[best_idx] = (self.speakers[best_idx][0], "Dispatcher (Bot)")
                    return "Dispatcher (Bot)"
            return best_label
        else:
            new_index = len(self.speakers) + 1
            new_label = f"Speaker {new_index}"
            
            # Check if this new unique speaker is robotic
            do_bot_detection = TRANSCRIPTION_CONFIG["settings"].get("detect_bots", False)
            if do_bot_detection:
                is_robo, *_ = self.is_robotic_voice(audio_chunk) if audio_chunk is not None else (False,)
                if is_robo:
                    new_label = f"Dispatcher (Bot)"
                
            self.speakers.append((embedding, new_label))
            return new_label

speaker_manager = SpeakerManager()

# Audio parameters
SAMPLE_RATE = 16000
BUFFER_SIZE = 4096
audio_queue = queue.Queue()
broadcast_queue = queue.Queue() # For web JSON data
audio_broadcast_queue = queue.Queue() # For web raw audio
stop_event = threading.Event()


import sys
import time

# WebSocket Management
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_json(self, data: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(data)
            except:
                self.disconnect(connection)

    async def broadcast_bytes(self, data: bytes):
        for connection in list(self.active_connections):
            try:
                await connection.send_bytes(data)
            except:
                self.disconnect(connection)

ws_manager = ConnectionManager()

# Background worker for WebSockets
async def websocket_broadcaster():
    """Worker to broadcast data from queues to all connected WebSocket clients."""
    print("WebSocket Broadcaster started.")
    while not stop_event.is_set():
        try:
            # Safely poll queues using a thread-safe helper if needed, 
            # but for now we'll just use non-blocking gets
            
            # Broadcast potential JSON data
            while not broadcast_queue.empty():
                try:
                    data = broadcast_queue.get_nowait()
                    await ws_manager.broadcast_json(data)
                except queue.Empty:
                    break

            # Broadcast potential audio data
            while not audio_broadcast_queue.empty():
                try:
                    audio_bytes = audio_broadcast_queue.get_nowait()
                    await ws_manager.broadcast_bytes(audio_bytes)
                except queue.Empty:
                    break
                    
            await asyncio.sleep(0.05) # Increased from 0.01 to reduce idle CPU
        except Exception as e:
            print(f"Broadcaster Error: {e}")
            await asyncio.sleep(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the WebSocket broadcaster task
    broadcast_task = asyncio.create_task(websocket_broadcaster())
    
    # Start core threads and audio input
    if args.web:
        boot_app()
        threading.Thread(target=run_input_source, daemon=True).start()
    
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/ping")
async def ping():
    return {"status": "ok", "connections": len(ws_manager.active_connections)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("New WebSocket client connecting...")
    await ws_manager.connect(websocket)
    print(f"Client connected. Active: {len(ws_manager.active_connections)}")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        print("Client disconnected.")
        ws_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket Error: {e}")
        ws_manager.disconnect(websocket)

app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Global for visualization
current_volume = 0.0

def audio_callback(indata, frames, time_info, status):
    """Callback function to capture audio data."""
    global current_volume
    if status:
        print(f"\nAudio Status Error: {status}")
    
    # Calculate current peak volume for the visualizer
    current_volume = np.linalg.norm(indata) / np.sqrt(len(indata))
    audio_queue.put((datetime.now(), indata.copy()))
    
    # Diagnostic print (once per ~1s)
    if not hasattr(audio_callback, "last_debug"):
        audio_callback.last_debug = 0
    if time.time() - audio_callback.last_debug > 2:
        # print(f"\nDebug: Audio callback firing (vol: {current_volume:.4f})") # Hidden but helpful for local debugging if needed
        audio_callback.last_debug = time.time()

    # Broadcast to web clients via queue ONLY if above noise floor or periodically
    try:
        settings = TRANSCRIPTION_CONFIG["settings"]
        noise_floor = settings.get("noise_floor", 0.001)
        
        # We also need to send a periodic ping even if quiet to let the UI know we're alive
        now = time.time()
        last_broadcast = getattr(audio_callback, "last_vol_broadcast", 0)
        
        if current_volume >= noise_floor:
            audio_broadcast_queue.put(indata.flatten().astype(np.float32).tobytes())
            audio_callback.last_vol_broadcast = now
        elif now - last_broadcast > 2.0:
            # Send a small JSON volume update instead of raw bytes
            broadcast_queue.put({
                "type": "volume",
                "peak": float(current_volume)
            })
            audio_callback.last_vol_broadcast = now
    except:
        pass


def visualizer():
    """Thread to display a simple volume meter."""
    while not stop_event.is_set():
        # Create a simple ASCII bar
        bars = int(current_volume * 300) # multiplier for visibility
        bars = min(bars, 50)
        meter = "[" + "#" * bars + "-" * (50 - bars) + "]"
        
        # Determine color/status
        status = "Active" if current_volume > 0.002 else "Quiet "
        
        # Print with carriage return to stay on one line
        sys.stdout.write(f"\r{meter} {status} (vol: {current_volume:.4f})")
        sys.stdout.flush()
        time.sleep(0.05)


def transcribe_audio():
    """Thread to transcribe audio with silence-aware chunking."""
    print("Transcription thread started.")
    
    active_buffer = []
    prev_context = None # To store a bit of the previous audio for context
    start_time = None
    
    while not stop_event.is_set():
        # Get new data from the queue
        try:
            # Check for backlog and catch up if necessary
            q_size = audio_queue.qsize()
            if q_size > 20: # Over 10 seconds of delay
                print(f"\r[BACKLOG] Queue is {q_size} chunks deep. Dropping oldest chunk to catch up...", end="")
                audio_queue.get_nowait()
                continue
                
            timestamp, new_chunk = audio_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        
        # Sentinel for flushing
        if timestamp is None:
            if active_buffer:
                # Force one last transcription
                transcribe_chunk(active_buffer, prev_context, start_time)
            break

        if not active_buffer:
            start_time = timestamp
            
        active_buffer.append(new_chunk)
        
        # Calculate recent volume for silence detection
        recent_vol = np.linalg.norm(new_chunk) / np.sqrt(len(new_chunk))
        
        # Determine if we should transcribe now
        buffer_duration = len(active_buffer) * (BUFFER_SIZE / SAMPLE_RATE)
        settings = TRANSCRIPTION_CONFIG["settings"]
        min_win = settings.get("min_window_sec", 1.0)
        max_win = settings.get("max_window_sec", 5.0)

        should_transcribe = False
        if buffer_duration >= min_win and recent_vol < 0.002:
            should_transcribe = True
        elif buffer_duration >= max_win:
            should_transcribe = True
            
        if not should_transcribe:
            continue

        # Aggressive silence gate for the entire block
        total_audio = np.concatenate(active_buffer)
        total_vol = np.linalg.norm(total_audio) / np.sqrt(len(total_audio))
        
        # Noise floor gating: if the entire block is extremely quiet, skip all heavy processing
        noise_floor = settings.get("noise_floor", 0.001)
        if total_vol < noise_floor: 
            active_buffer = []
            start_time = None
            continue

        # Telemetry: Check queue backlog
        qsize = audio_queue.qsize()
        skip_diarization = qsize > 10
        
        t0 = time.time()
        prev_context = transcribe_chunk(active_buffer, prev_context, start_time, skip_diarization=skip_diarization)
        t_proc = time.time() - t0
        
        # Real-time factor: Processing time / Audio duration
        rtf = t_proc / buffer_duration
        
        # Console reporting for the user
        status_line = f"\n [Telemetry] Backlog: {qsize} chunks | Proc: {t_proc:.2f}s | Audio: {buffer_duration:.2f}s | RTF: {rtf:.2f}"
        if skip_diarization:
            status_line += " | [!] DIARIZATION SKIPPED TO CATCH UP"
        
        # Only print telemetry if lagging or periodically
        if rtf > 1.0 or qsize > 0 or not hasattr(transcribe_audio, "chunk_count") or transcribe_audio.chunk_count % 10 == 0:
            sys.stdout.write(status_line + "\n")
            sys.stdout.flush()
            
        if not hasattr(transcribe_audio, "chunk_count"):
            transcribe_audio.chunk_count = 0
        transcribe_audio.chunk_count += 1
        
        active_buffer = []

def transcribe_chunk(active_buffer, prev_context, start_time, skip_diarization=False):
    """Helper to process a buffer of audio data."""
    # Prepare audio data
    audio_data = np.concatenate(active_buffer)
    audio_flat = audio_data.flatten().astype(np.float32)

    # Prepend a bit of previous context (0.5s) to help with word fragments
    if prev_context is not None:
        audio_flat = np.concatenate([prev_context, audio_flat])

    # Calculate total volume for the whole chunk
    volume_norm = np.linalg.norm(audio_flat) / np.sqrt(len(audio_flat))
    
    if volume_norm > 0.002:
        # Better normalization
        max_val = np.max(np.abs(audio_flat))
        if max_val > 0.01:
            audio_flat = audio_flat / max_val

        # Prepare initial prompt
        prompt_parts = []
        if TRANSCRIPTION_CONFIG["vocabulary"]:
            prompt_parts.append(", ".join(TRANSCRIPTION_CONFIG["vocabulary"]))
        
        last_text = getattr(transcribe_audio, "last_transcript", None)
        if last_text:
            prompt_parts.append(last_text)
        
        combined_prompt = ". ".join(prompt_parts) if prompt_parts else None

        settings = TRANSCRIPTION_CONFIG["settings"]

        # Transcribe with improved settings to suppress hallucinations
        # Safety guard for transcription model execution
        try:
            segments, info = model.transcribe(
                audio_flat, 
                beam_size=settings.get("beam_size", 5),
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=settings.get("min_silence_duration_ms", 500)),
                initial_prompt=combined_prompt,
                # Suppress hallucinations
                no_speech_threshold=settings.get("no_speech_threshold", 0.8),
                log_prob_threshold=settings.get("log_prob_threshold", -0.8),
                compression_ratio_threshold=settings.get("compression_ratio_threshold", 2.4),
                condition_on_previous_text=False # Prevent hallucinations from bleeding into context
            )
        except Exception as e:
            print(f"\n[CRITICAL] Transcription model error: {e}")
            return prev_context

        processed_segments = []
        for segment in segments:
            text_segment = segment.text.strip()
            if not text_segment:
                continue
                
            # Hallucination filter
            hallucination_patterns = ["thank you.", "thanks for watching.", "subscribe", "please subscribe", 
                                    "welcome back", "my channel", "social media", "I'll see you", 
                                    "hey everyone", "thanks for having me", "thanks for having us"]
            if any(p in text_segment.lower() for p in hallucination_patterns):
                continue

            # Stricter combined confidence check
            avg_cutoff = settings.get("avg_logprob_cutoff", -0.8)
            no_speech_cutoff = settings.get("no_speech_prob_cutoff", 0.2)
            extreme_cutoff = settings.get("extreme_confidence_cutoff", -0.4)
            
            if not (segment.avg_logprob > avg_cutoff and segment.no_speech_prob < no_speech_cutoff) and not (segment.avg_logprob > extreme_cutoff):
                continue

            # -- SEGMENT ANALYSIS --
            # Slice the audio for this specific sentence
            start_sample = int(segment.start * SAMPLE_RATE)
            end_sample = int(segment.end * SAMPLE_RATE)
            audio_slice = audio_flat[start_sample:end_sample]
            
            seg_speaker_label = "Unknown"
            if speaker_model and not skip_diarization and len(audio_slice) >= 8000: # At least 0.5s for speaker ID
                try:
                    signal = torch.from_numpy(audio_slice).unsqueeze(0)
                    embeddings = speaker_model.encode_batch(signal)
                    emb = embeddings.squeeze()
                    seg_speaker_label = speaker_manager.identify_speaker(emb, audio_slice)
                except:
                    pass
            elif skip_diarization:
                seg_speaker_label = "Unknown Speaker"
            
            # Apply corrections to text
            text = text_segment
            if TRANSCRIPTION_CONFIG["corrections"]:
                for wrong, right in TRANSCRIPTION_CONFIG["corrections"].items():
                    pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                    text = pattern.sub(right, text)
            
            processed_segments.append({
                "speaker": seg_speaker_label,
                "text": text,
                "start": segment.start,
                "end": segment.end,
                "confidence": segment.avg_logprob # Capture Whisper's log probability
            })

        if processed_segments:
            # Group identical consecutive speakers to keep output clean
            merged = []
            for seg in processed_segments:
                if merged and merged[-1]["speaker"] == seg["speaker"]:
                    # Average the confidence scores
                    prev_count = merged[-1].get("_seg_count", 1)
                    curr_conf = merged[-1].get("confidence", 0)
                    new_conf = (curr_conf * prev_count + seg["confidence"]) / (prev_count + 1)
                    
                    merged[-1]["text"] += " " + seg["text"]
                    merged[-1]["end"] = seg["end"]
                    merged[-1]["confidence"] = new_conf
                    merged[-1]["_seg_count"] = prev_count + 1
                else:
                    seg["_seg_count"] = 1
                    merged.append(seg)

            display_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
            
            for m in merged:
                speaker_tag = f" [{m['speaker']}]"
                conf_val = m.get("confidence", 0)
                conf_tag = f" (conf: {conf_val:.2f})"
                output_line = f"[{display_time}]{speaker_tag}{conf_tag} {m['text']}"
                
                # Diagnostic robotic print if enabled (only if speaker is robotic)
                if getattr(args, "debug_robo", False):
                    # We run it on the merged chunk
                    start_s = int(m['start'] * SAMPLE_RATE)
                    end_s = int(m['end'] * SAMPLE_RATE)
                    chunk_slice = audio_flat[start_s:end_s]
                    is_robo, f0_m, f0_s, flat = speaker_manager.is_robotic_voice(chunk_slice)
                    if is_robo:
                        debug_info = f"\n[RoboDebug] Segment Mean: {f0_m:.1f}, Std: {f0_s:.2f}, Flat: {flat:.4f} | Label: {m['speaker']}"
                        sys.stdout.write("\r" + " " * 80 + "\r")
                        print(debug_info)

                # Broadcast to web clients
                try:
                    broadcast_queue.put({
                        "type": "transcript",
                        "timestamp": display_time,
                        "origin_time": start_time.timestamp() + m['start'],
                        "speaker": m['speaker'],
                        "text": m['text'],
                        "confidence": m.get("confidence", 0)
                    })
                except:
                    pass

                # Print and log
                sys.stdout.write("\r" + " " * 80 + "\r")
                print(output_line)
                
                output_file = getattr(transcribe_audio, "output_file", None)
                if output_file:
                    try:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(output_line + "\n")
                            f.flush()
                    except Exception as e:
                        print(f"\nError writing to output file: {e}")
            
            # Store last text for prompt context
            transcribe_audio.last_transcript = " ".join([m['text'] for m in merged])

    # Store the last 0.5s as context for the next chunk
    context_samples = int(0.5 * SAMPLE_RATE)
    prev_context = audio_flat[-context_samples:]
    
    # Reset the active buffer
    active_buffer = []
    return prev_context


# Thread objects will be created in main

def feed_file_to_queue(file_path):
    """Feeds an audio file into the transcription queue at real-time speed."""
    print(f"Reading from file: {file_path}")
    data, samplerate = sf.read(file_path)
    
    # Proper Resampling
    if samplerate != SAMPLE_RATE:
        print(f"Resampling file from {samplerate}Hz to {SAMPLE_RATE}Hz...")
        data = librosa.resample(data, orig_sr=samplerate, target_sr=SAMPLE_RATE)
        samplerate = SAMPLE_RATE
    
    # Ensure it's mono
    if len(data.shape) > 1:
        data = data.mean(axis=1) # Better than just taking first channel
    
    # Feed in BUFFER_SIZE chunks
    for i in range(0, len(data), BUFFER_SIZE):
        chunk = data[i:i + BUFFER_SIZE].reshape(-1, 1).astype(np.float32)
        audio_queue.put((datetime.now(), chunk))
        
        # Real-time simulation: wait for the duration of the chunk
        # Calculate how long this chunk would take in seconds
        chunk_duration = len(chunk) / samplerate
        
        # We also need to update the visualizer's global volume
        global current_volume
        current_volume = np.linalg.norm(chunk) / np.sqrt(len(chunk))
        
        # Broadcast to web clients via queue
        try:
            audio_broadcast_queue.put(chunk.flatten().astype(np.float32).tobytes())
        except:
            pass
        
        time.sleep(chunk_duration)
    
    print("\nEnd of file reached.")
    audio_queue.put((None, None))

parser = argparse.ArgumentParser(description="Live Transcription with Faster-Whisper")
parser.add_argument("--input", type=str, help="Path to an audio file to test instead of microphone")
parser.add_argument("--config", "-c", type=str, help="Path to config.json for vocabulary and corrections")
parser.add_argument("--output", "-o", type=str, help="Path to output text file")
parser.add_argument("--diarize", "-d", action="store_true", help="Enable speaker identification")
parser.add_argument("--web", "-w", action="store_true", help="Enable web dashboard")
parser.add_argument("--port", type=int, default=8000, help="Web dashboard port")
parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
parser.add_argument("--device", type=int, help="Input device ID (from --list-devices)")
parser.add_argument("--debug-robo", action="store_true", help="Print robotic voice detection stats for debugging")
parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (Web mode only)")
args = parser.parse_args()

def boot_app():
    """Centralized boot sequence for both CLI and Web modes."""
    global transcription_thread, speaker_model
    
    if args.diarize and speaker_model is None:
        print("Loading speaker identification model...")
        from speechbrain.inference.speaker import EncoderClassifier
        speaker_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device":"cpu"}
        )

    if args.output:
        transcribe_audio.output_file = args.output
        print(f"Transcription will be saved to {args.output}")

    if args.config:
        try:
            with open(args.config, 'r') as f:
                loaded_config = json.load(f)
                if "settings" in loaded_config:
                    TRANSCRIPTION_CONFIG["settings"].update(loaded_config["settings"])
                for key in ["vocabulary", "corrections"]:
                    if key in loaded_config:
                        TRANSCRIPTION_CONFIG[key] = loaded_config[key]
                print(f"Loaded config from {args.config}")
        except Exception as e:
            print(f"Error loading config: {e}")

    # Load Whisper model AFTER config is parsed
    load_model()

    # Start core processing threads
    threading.Thread(target=visualizer, daemon=True).start()
    transcription_thread = threading.Thread(target=transcribe_audio, daemon=True)
    transcription_thread.start()
    print("Core processing threads started.")

def run_input_source():
    """Helper to run the selected input source (file or mic)."""
    if args.input:
        try:
            feed_file_to_queue(args.input)
            # Wait for transcription to finish flushing
            # Note: transcription_thread will be global by the time this runs
            if 'transcription_thread' in globals():
                transcription_thread.join(timeout=30)
            stop_event.set()
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        # Start capturing audio from the microphone
        try:
            device_info = sd.query_devices(args.device, 'input') if args.device is not None else sd.query_devices(kind='input')
            print(f"Using input device: {device_info['name']}")
            
            with sd.InputStream(
                callback=audio_callback, 
                channels=1, 
                samplerate=SAMPLE_RATE, 
                blocksize=BUFFER_SIZE,
                device=args.device
            ):
                print("Listening... Press Ctrl+C to stop.")
                while not stop_event.is_set():
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
            stop_event.set()
        except Exception as e:
            print(f"Microphone Error: {e}")
            stop_event.set()

if __name__ == "__main__":

    if args.list_devices:
        print("\nAvailable Audio Devices:")
        print(sd.query_devices())
        sys.exit(0)

    if not args.web:
        boot_app()
        # Run input source in main thread for CLI mode
        run_input_source()
    else:
        # In Web mode, boot_app and run_input_source are handled by lifespan
        print(f"\nWeb Dashboard available at http://localhost:{args.port}")
        try:
            if args.reload:
                module_name = os.path.splitext(os.path.basename(__file__))[0]
                uvicorn.run(f"{module_name}:app", host="0.0.0.0", port=args.port, reload=True)
            else:
                uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="error")
        except KeyboardInterrupt:
            print("\nStopping Web Server...")
            stop_event.set()
