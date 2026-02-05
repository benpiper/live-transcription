import asyncio
import json
import logging
import smtplib
import time
import wave
import io
import os
from collections import deque
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

import websockets
import numpy as np
import urllib.request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EmailAlert")

# Constants
SAMPLE_RATE = 16000
AUDIO_BUFFER_SECONDS = 120  # Keep 2 minutes of audio history

class EmailAlertTool:
    def __init__(self, config_path="email_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.last_alerts = {}  # keyword: timestamp
        self.running = True
        
        # Buffers for context and audio
        context_count = self.config["alerts"].get("context_count", 3)
        self.transcript_history = deque(maxlen=context_count + 1)
        self.audio_history = deque() # List of (timestamp, chunk_bytes)
        
    def load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            logger.info("Please copy email_config.json.example to email_config.json and fill in your settings.")
            exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing configuration: {e}")
            exit(1)

    def get_audio_clip(self, start_time, duration):
        """Extract a WAV audio clip for a specific time window."""
        end_time = start_time + duration
        
        # We add a small buffer (0.5s) to ensure we get the whole phrase
        start_time -= 0.5
        end_time += 0.5
        
        matching_chunks = []
        for ts, chunk in self.audio_history:
            # Each chunk is 4096 samples = 0.256s
            chunk_duration = 4096 / SAMPLE_RATE
            chunk_end = ts + chunk_duration
            
            # Check for overlap
            if chunk_end > start_time and ts < end_time:
                matching_chunks.append(chunk)
        
        if not matching_chunks:
            return None
            
        try:
            # Combine chunks and convert to WAV
            audio_data = b"".join(matching_chunks)
            # Verify data is float32 (server sends float32 bytes)
            samples = np.frombuffer(audio_data, dtype=np.float32)
            
            # Convert to 16-bit PCM for WAV
            pcm_samples = (samples * 32767).astype(np.int16)
            
            byte_io = io.BytesIO()
            with wave.open(byte_io, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(pcm_samples.tobytes())
            
            return byte_io.getvalue()
        except Exception as e:
            logger.error(f"Failed to generate audio clip: {e}")
            return None

    def send_email(self, keyword, transcript_data, audio_clip=None):
        """Send an email alert via SMTP with optional context and audio."""
        smtp_config = self.config["smtp"]
        alert_config = self.config["alerts"]
        
        timestamp = transcript_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        speaker = transcript_data.get("speaker", "Unknown")
        text = transcript_data.get("text", "")

        subject = f"{alert_config.get('subject_prefix', '[Alert]')} {keyword}: {text[:50]}"
        if len(text) > 50: subject += "..."
        
        # Build context string
        context_lines = []
        for entry in self.transcript_history:
            if entry.get("timestamp") != timestamp: # Don't repeat the triggering line if it was already buffered
                context_lines.append(f"[{entry.get('timestamp')}] {entry.get('speaker')}: {entry.get('text')}")
        
        context_text = "\n".join(context_lines) if context_lines else "No previous context available."
        
        body = f"""
        Alert triggered for keyword: {keyword}
        
        Triggering Line:
        [{timestamp}] {speaker}: {text}
        
        Context:
        {context_text}
        
        ---
        This is an automated alert.
        """

        msg = MIMEMultipart()
        msg['From'] = smtp_config["from_address"]
        msg['To'] = ", ".join(smtp_config["to_addresses"])
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        if audio_clip:
            filename = f"alert_{int(time.time())}.wav"
            attachment = MIMEApplication(audio_clip, _subtype="wav")
            attachment.add_header('Content-Disposition', 'attachment', filename=filename)
            msg.attach(attachment)

        try:
            server = smtplib.SMTP(smtp_config["server"], smtp_config["port"])
            if smtp_config.get("use_tls", True):
                server.starttls()
            
            if smtp_config.get("username") and smtp_config.get("password"):
                server.login(smtp_config["username"], smtp_config["password"])
            
            server.send_message(msg)
            server.quit()
            logger.info(f"Email alert sent for keyword '{keyword}' (Audio attached: {audio_clip is not None})")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    def check_keywords(self, text):
        """Check transcript text for keywords and trigger alerts if not rate-limited."""
        alert_config = self.config["alerts"]
        keywords = alert_config.get("keywords", [])
        rate_limit = alert_config.get("rate_limit_seconds", 300)
        now = time.time()
        
        matched_keywords = []
        text_lower = text.lower()
        
        for kw in keywords:
            if kw.lower() in text_lower:
                last_time = self.last_alerts.get(kw, 0)
                if now - last_time > rate_limit:
                    matched_keywords.append(kw)
                    self.last_alerts[kw] = now
                else:
                    logger.debug(f"Rate limiting alert for keyword '{kw}'")
        
        return matched_keywords

    def fetch_initial_context(self):
        """Fetch existing transcripts from the server to warm the context buffer."""
        ws_url = self.config["websocket"]["url"]
        # Convert ws:// to http:// and extract the base URL
        http_url = ws_url.replace("ws://", "http://").replace("wss://", "https://").replace("/ws", "/api/session/current")
        
        logger.info(f"Fetching initial context from {http_url}...")
        try:
            with urllib.request.urlopen(http_url, timeout=5) as response:
                data = json.loads(response.read().decode())
                if data.get("active") and "transcripts" in data:
                    transcripts = data["transcripts"]
                    # Get the last N transcripts for the buffer
                    context_count = self.config["alerts"].get("context_count", 3)
                    start_idx = max(0, len(transcripts) - context_count)
                    
                    # Store current content to check for duplicates
                    existing_text = set()
                    for entry in self.transcript_history:
                        existing_text.add((entry.get("timestamp"), entry.get("text")))
                    
                    for t in transcripts[start_idx:]:
                        if (t.get("timestamp"), t.get("text")) not in existing_text:
                            self.transcript_history.append(t)
                    
                    logger.info(f"Buffered {len(list(self.transcript_history))} historical transcripts for context.")
        except Exception as e:
            logger.warning(f"Could not fetch initial context: {e}")

    async def monitor_feed(self):
        """Connect to WebSocket and monitor for keyword matches."""
        uri = self.config["websocket"]["url"]
        attach_audio = self.config["alerts"].get("attach_audio", False)
        
        # Warm the context buffer on startup
        self.fetch_initial_context()
        
        while self.running:
            try:
                logger.info(f"Connecting to WebSocket at {uri}...")
                async with websockets.connect(uri) as websocket:
                    logger.info("Connected to transcription feed.")
                    while True:
                        message = await websocket.recv()
                        
                        # Handle binary audio data
                        if isinstance(message, bytes):
                            # Approximate timing: the speech ended just now
                            # Each chunk is 4096 samples / 16000 Hz = 0.256s
                            ts = time.time() - 0.256 
                            self.audio_history.append((ts, message))
                            
                            # Clean up old audio chunks
                            now = time.time()
                            while self.audio_history and self.audio_history[0][0] < now - AUDIO_BUFFER_SECONDS:
                                self.audio_history.popleft()
                            continue
                            
                        data = json.loads(message)
                        
                        if data.get("type") == "transcript":
                            text = data.get("text", "")
                            matched = self.check_keywords(text)
                            
                            origin_time = data.get("origin_time")
                            duration = data.get("duration")
                            
                            for kw in matched:
                                logger.warning(f"MATCH FOUND: '{kw}' in transcript: {text}")
                                
                                audio_clip = None
                                if attach_audio and origin_time and duration:
                                    audio_clip = self.get_audio_clip(origin_time, duration)
                                
                                self.send_email(kw, data, audio_clip=audio_clip)
                            
                            # Add to history for context
                            self.transcript_history.append(data)
                                
            except (websockets.ConnectionClosed, ConnectionRefusedError):
                logger.warning("WebSocket connection lost or refused. Retrying in 5 seconds...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error in monitor loop: {e}")
                await asyncio.sleep(5)

    def send_test_email(self):
        """Send a test email to verify SMTP configuration."""
        logger.info("Sending test email...")
        test_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "speaker": "Test System",
            "text": "This is a test transcript to verify your email alert settings."
        }
        self.send_email("TEST", test_data)

    def run(self):
        """Run the alert tool."""
        try:
            asyncio.run(self.monitor_feed())
        except KeyboardInterrupt:
            logger.info("Alert tool stopped by user.")
            self.running = False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live Transcription Email Alert Tool")
    parser.add_argument("--config", default="email_config.json", help="Path to configuration file")
    parser.add_argument("--test", action="store_true", help="Send a test email and exit")
    args = parser.parse_args()
    
    tool = EmailAlertTool(config_path=args.config)
    
    if args.test:
        tool.send_test_email()
    else:
        tool.run()
