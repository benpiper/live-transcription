import asyncio
import json
import logging
import smtplib
import time
import wave
import io
import os
import re
import uuid
from collections import deque
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

import websockets
import numpy as np
import urllib.request
import ssl

from alert_rules import AlertRuleEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EmailAlert")

# Constants
SAMPLE_RATE = 16000
AUDIO_BUFFER_SECONDS = 120  # Keep 2 minutes of audio history

class AudioBufferManager:
    """Synchronize client-side audio timestamps with server-side transcript timestamps."""

    def __init__(self, buffer_seconds=120, resync_threshold=0.5):
        self.buffer_seconds = buffer_seconds
        self.resync_threshold = resync_threshold
        self.audio_chunks = deque()  # (server_timestamp, chunk_bytes)
        self.client_to_server_offset = 0.0
        self.recent_syncs = deque(maxlen=10)
        self.offset_initialized = False

    def add_chunk(self, chunk_bytes: bytes, client_time: float):
        """Add audio chunk with client timestamp, convert to server time."""
        server_time = client_time + self.client_to_server_offset
        self.audio_chunks.append((server_time, chunk_bytes))

        # Clean up old chunks
        now = time.time() + self.client_to_server_offset
        while self.audio_chunks and self.audio_chunks[0][0] < now - self.buffer_seconds:
            self.audio_chunks.popleft()

    def sync_with_transcript(self, origin_time: float, client_receive_time: float):
        """Synchronize with transcript timestamp to calibrate offset."""
        new_offset = origin_time - client_receive_time

        if not self.offset_initialized:
            self.client_to_server_offset = new_offset
            self.offset_initialized = True
            self.recent_syncs.append((time.time(), new_offset))
            logger.debug(f"Audio sync initialized: offset={new_offset:.3f}s")
        else:
            # Check for drift
            offset_change = abs(new_offset - self.client_to_server_offset)
            if offset_change > self.resync_threshold:
                logger.debug(f"Audio offset drift detected: {offset_change:.3f}s, updating from {self.client_to_server_offset:.3f}s to {new_offset:.3f}s")
                self.client_to_server_offset = new_offset

            self.recent_syncs.append((time.time(), new_offset))

    def extract_clip(self, origin_time: float, duration: float, padding: float = 0.5) -> bytes:
        """Extract audio clip for time range with padding."""
        start_time = origin_time - padding
        end_time = origin_time + duration + padding

        matching_chunks = []
        for ts, chunk in self.audio_chunks:
            # Each chunk is 4096 samples = 0.256s
            chunk_duration = 4096 / SAMPLE_RATE
            chunk_end = ts + chunk_duration

            # Check for overlap
            if chunk_end > start_time and ts < end_time:
                matching_chunks.append(chunk)

        if not matching_chunks:
            logger.warning(f"No audio chunks found for time range {start_time:.2f}-{end_time:.2f}")
            return None

        try:
            audio_data = b"".join(matching_chunks)
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


class EmailRetryQueue:
    """Persistent queue for failed emails with exponential backoff retry."""

    def __init__(self, config, queue_path="email_retry_queue.json"):
        self.config = config
        self.queue_path = queue_path
        self.queue = []  # List of email records
        self.consecutive_failures = 0
        self.load_queue()

    def load_queue(self):
        """Load queue from disk."""
        try:
            if os.path.exists(self.queue_path):
                with open(self.queue_path, "r") as f:
                    data = json.load(f)
                    self.queue = data.get("emails", [])
                    logger.info(f"Loaded {len(self.queue)} emails from retry queue")
            else:
                self.queue = []
        except Exception as e:
            logger.error(f"Failed to load retry queue: {e}")
            self.queue = []

    def save_queue(self):
        """Persist queue to disk."""
        try:
            with open(self.queue_path, "w") as f:
                json.dump({"emails": self.queue}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save retry queue: {e}")

    def add_failed_email(self, keyword: str, transcript_data: dict, error: str, audio_clip_path: str = None):
        """Add failed email to retry queue."""
        email_id = str(uuid.uuid4())
        retry_config = self.config.get("retry", {})

        email_record = {
            "id": email_id,
            "timestamp": time.time(),
            "retry_count": 0,
            "next_retry_time": time.time() + retry_config.get("initial_backoff_seconds", 60),
            "keyword": keyword,
            "transcript_data": transcript_data,
            "audio_clip_path": audio_clip_path,
            "last_error": error
        }

        # Check queue size
        max_size = retry_config.get("max_queue_size", 100)
        if len(self.queue) >= max_size:
            logger.warning(f"Retry queue at max size ({max_size}). Dropping oldest email.")
            old_email = self.queue.pop(0)
            if old_email.get("audio_clip_path") and os.path.exists(old_email["audio_clip_path"]):
                try:
                    os.remove(old_email["audio_clip_path"])
                except:
                    pass

        self.queue.append(email_record)
        self.save_queue()
        logger.info(f"Added email to retry queue (id={email_id}, keyword={keyword})")

    def get_ready_emails(self) -> list:
        """Return emails ready for retry."""
        now = time.time()
        return [e for e in self.queue if e["next_retry_time"] <= now]

    def mark_success(self, email_id: str):
        """Mark email as successfully sent and remove from queue."""
        email = next((e for e in self.queue if e["id"] == email_id), None)
        if email:
            if email.get("audio_clip_path") and os.path.exists(email["audio_clip_path"]):
                try:
                    os.remove(email["audio_clip_path"])
                except:
                    pass
            self.queue = [e for e in self.queue if e["id"] != email_id]
            self.save_queue()
            self.consecutive_failures = 0
            logger.info(f"Email {email_id} sent successfully, removed from retry queue")

    def mark_failure(self, email_id: str, error: str):
        """Mark email as failed and schedule retry with exponential backoff."""
        email = next((e for e in self.queue if e["id"] == email_id), None)
        if email:
            email["retry_count"] += 1
            email["last_error"] = error

            retry_config = self.config.get("retry", {})
            backoff = self.calculate_backoff(email["retry_count"], retry_config)
            email["next_retry_time"] = time.time() + backoff

            max_retries = retry_config.get("max_retries", 5)
            if email["retry_count"] >= max_retries:
                logger.error(f"Email {email_id} exceeded max retries ({max_retries}). Removing from queue.")
                self.queue = [e for e in self.queue if e["id"] != email_id]

            self.save_queue()

    def calculate_backoff(self, retry_count: int, retry_config: dict) -> float:
        """Calculate exponential backoff time."""
        initial = retry_config.get("initial_backoff_seconds", 60)
        multiplier = retry_config.get("backoff_multiplier", 2.0)
        max_backoff = retry_config.get("max_backoff_seconds", 3600)

        backoff = initial * (multiplier ** (retry_count - 1))
        return min(backoff, max_backoff)

    def cleanup_old_audio_files(self):
        """Delete temp audio files for emails no longer in queue."""
        queue_audio_files = set(e.get("audio_clip_path") for e in self.queue if e.get("audio_clip_path"))

        temp_dir = "/tmp"
        try:
            for filename in os.listdir(temp_dir):
                if filename.startswith("email_alert_") and filename.endswith(".wav"):
                    filepath = os.path.join(temp_dir, filename)
                    if filepath not in queue_audio_files and os.path.exists(filepath):
                        try:
                            os.remove(filepath)
                        except:
                            pass
        except:
            pass

class EmailAlertTool:
    def __init__(self, config_path="email_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.last_alerts = {}  # keyword: timestamp (legacy)
        self.running = True

        # Initialize alert rule engine
        self.alert_engine = AlertRuleEngine(self.config)

        # Buffers for context and audio
        context_count = self.config["alerts"].get("context_count", 3)
        self.transcript_history = deque(maxlen=context_count + 1)

        # Initialize audio buffer manager
        audio_config = self.config.get("audio", {})
        self.audio_buffer = AudioBufferManager(
            buffer_seconds=audio_config.get("buffer_seconds", AUDIO_BUFFER_SECONDS),
            resync_threshold=audio_config.get("resync_threshold_seconds", 0.5)
        )

        # Initialize retry queue if enabled
        retry_config = self.config.get("retry", {})
        if retry_config.get("enabled", True):
            self.retry_queue = EmailRetryQueue(
                self.config,
                queue_path=retry_config.get("persistent_queue_path", "email_retry_queue.json")
            )
        else:
            self.retry_queue = None

        # Heartbeat tracking
        heartbeat_config = self.config.get("heartbeat", {})
        self.last_daily_heartbeat = None
        self.last_connected_time = None
        self.connection_start_time = None
        self.reconnection_failures = deque(maxlen=10)
        self.heartbeat_sent_for_current_outage = False

        # Validate configuration on startup
        self.validate_email_config()
        
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

    def validate_email_config(self) -> list:
        """Validate email configuration. Returns list of errors (empty = valid)."""
        errors = []

        # Required fields
        required_fields = {
            ("smtp", "server"): str,
            ("smtp", "port"): int,
            ("smtp", "from_address"): str,
            ("smtp", "to_addresses"): list,
            ("websocket", "url"): str,
        }

        for (section, field), expected_type in required_fields.items():
            if section not in self.config:
                errors.append(f"Missing required section: {section}")
                continue

            if field not in self.config[section]:
                errors.append(f"Missing required field: {section}.{field}")
                continue

            value = self.config[section][field]
            if not isinstance(value, expected_type):
                errors.append(f"{section}.{field} must be {expected_type.__name__}, got {type(value).__name__}")

        # Validate SMTP port
        try:
            port = self.config.get("smtp", {}).get("port")
            if port and isinstance(port, int):
                if port < 1 or port > 65535:
                    errors.append("SMTP port must be between 1 and 65535")
        except:
            pass

        # Validate email addresses
        email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        try:
            from_addr = self.config.get("smtp", {}).get("from_address", "")
            if "@" in from_addr:
                email_part = from_addr.split("<")[-1].rstrip(">")
                if not re.match(email_regex, email_part):
                    errors.append(f"Invalid from_address email format: {from_addr}")

            for addr in self.config.get("smtp", {}).get("to_addresses", []):
                if not re.match(email_regex, addr):
                    errors.append(f"Invalid to_address email format: {addr}")
        except:
            pass

        # Validate WebSocket URL
        ws_url = self.config.get("websocket", {}).get("url", "")
        if ws_url and not (ws_url.startswith("ws://") or ws_url.startswith("wss://")):
            errors.append("WebSocket URL must start with ws:// or wss://")

        # Validate alert settings
        try:
            alerts = self.config.get("alerts", {})
            rate_limit = alerts.get("rate_limit_seconds")
            if rate_limit is not None and (not isinstance(rate_limit, int) or rate_limit < 0):
                errors.append("rate_limit_seconds must be >= 0")

            context_count = alerts.get("context_count")
            if context_count is not None and (not isinstance(context_count, int) or context_count < 1 or context_count > 100):
                errors.append("context_count must be between 1 and 100")
        except:
            pass

        # Validate credentials (if both provided, both must exist)
        smtp = self.config.get("smtp", {})
        has_user = bool(smtp.get("username"))
        has_pass = bool(smtp.get("password"))
        if (has_user and not has_pass) or (has_pass and not has_user):
            errors.append("Both username and password must be provided together, or neither")

        # Log validation results
        if errors:
            for err in errors:
                logger.warning(f"Config validation: {err}")
        else:
            logger.info("Email configuration validation passed")

        return errors

    def test_smtp_connection(self) -> tuple:
        """Test SMTP connection. Returns (success: bool, message: str)."""
        smtp_config = self.config.get("smtp", {})
        server = smtp_config.get("server")
        port = smtp_config.get("port")
        use_tls = smtp_config.get("use_tls", True)
        username = smtp_config.get("username")
        password = smtp_config.get("password")

        try:
            logger.info(f"Testing SMTP connection to {server}:{port}...")
            smtp = smtplib.SMTP(server, port, timeout=10)

            if use_tls:
                smtp.starttls()

            if username and password:
                smtp.login(username, password)

            smtp.noop()
            smtp.quit()

            msg = f"SMTP connection successful to {server}:{port}"
            logger.info(msg)
            return (True, msg)

        except Exception as e:
            msg = f"SMTP connection failed: {str(e)}"
            logger.error(msg)
            return (False, msg)

    def send_email(self, keyword, transcript_data, audio_clip=None, rule_metadata=None):
        """
        Send an email alert via SMTP with optional context and audio.

        Args:
            keyword: Matched keyword (for backward compatibility)
            transcript_data: Transcript dict with timestamp, speaker, text, etc.
            audio_clip: Optional audio clip bytes or file path
            rule_metadata: Optional dict with rule details (rule_id, description, match_type, tags, etc.)
        """
        smtp_config = self.config["smtp"]
        alert_config = self.config["alerts"]

        timestamp = transcript_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        speaker = transcript_data.get("speaker", "Unknown")
        text = transcript_data.get("text", "")
        confidence = transcript_data.get("confidence")
        duration = transcript_data.get("duration")

        # Build subject with rule info if available
        if rule_metadata:
            rule_desc = rule_metadata.get("description", keyword)
            subject = f"{alert_config.get('subject_prefix', '[Alert]')} {rule_desc}: {text[:40]}"
        else:
            subject = f"{alert_config.get('subject_prefix', '[Alert]')} {keyword}: {text[:50]}"
        if len(text) > 50:
            subject += "..."

        # Build context string
        context_lines = []
        for entry in self.transcript_history:
            if entry.get("timestamp") != timestamp:
                context_lines.append(f"[{entry.get('timestamp')}] {entry.get('speaker')}: {entry.get('text')}")

        context_text = "\n".join(context_lines) if context_lines else "No previous context available."

        # Build email body with rule metadata
        body = f"""
        Alert triggered for keyword: {keyword}
        """

        if rule_metadata:
            body += f"""
        Rule: {rule_metadata.get('description', rule_metadata.get('rule_id', keyword))}
        Match Type: {rule_metadata.get('match_type', 'unknown')}
        """
            if rule_metadata.get('tags'):
                body += f"        Tags: {', '.join(rule_metadata.get('tags', []))}\n"

        body += f"""
        Triggering Line:
        [{timestamp}] {speaker}: {text}
        """

        if confidence is not None:
            body += f"        Confidence: {confidence:.3f}\n"
        if duration is not None:
            body += f"        Duration: {duration:.2f}s\n"

        body += f"""
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

        # Handle audio attachment (either bytes or temp file path)
        audio_clip_path = None
        if audio_clip:
            if isinstance(audio_clip, bytes):
                filename = f"alert_{int(time.time())}.wav"
                attachment = MIMEApplication(audio_clip, _subtype="wav")
                attachment.add_header('Content-Disposition', 'attachment', filename=filename)
                msg.attach(attachment)
            elif isinstance(audio_clip, str) and os.path.exists(audio_clip):
                audio_clip_path = audio_clip
                try:
                    with open(audio_clip, "rb") as f:
                        audio_data = f.read()
                    filename = os.path.basename(audio_clip)
                    attachment = MIMEApplication(audio_data, _subtype="wav")
                    attachment.add_header('Content-Disposition', 'attachment', filename=filename)
                    msg.attach(attachment)
                except Exception as e:
                    logger.error(f"Failed to attach audio file {audio_clip}: {e}")

        try:
            server = smtplib.SMTP(smtp_config["server"], smtp_config["port"])
            if smtp_config.get("use_tls", True):
                server.starttls()

            if smtp_config.get("username") and smtp_config.get("password"):
                server.login(smtp_config["username"], smtp_config["password"])

            server.send_message(msg)
            server.quit()
            logger.info(f"Email alert sent for keyword '{keyword}' (Audio attached: {audio_clip is not None})")

            # Mark retry as successful if this was a retry
            email_id = transcript_data.get("_retry_id")
            if email_id and self.retry_queue:
                self.retry_queue.mark_success(email_id)

        except Exception as e:
            logger.error(f"Failed to send email: {e}")

            # Add to retry queue if enabled
            if self.retry_queue:
                self.retry_queue.add_failed_email(keyword, transcript_data, str(e), audio_clip_path)

    def check_keywords(self, text, transcript_data=None):
        """
        Check transcript text against alert rules and return matched rules.

        Uses the AlertRuleEngine for advanced pattern matching, context filtering,
        and deduplication. Backward compatible with legacy keyword config.

        Args:
            text: Transcript text to check
            transcript_data: Full transcript dict with speaker, confidence, duration, etc.

        Returns:
            List of matched rule dicts with rule_id, description, metadata, etc.
        """
        if transcript_data is None:
            transcript_data = {}

        # Use new rule engine
        matched_rules = self.alert_engine.check_rules(text, transcript_data)
        return matched_rules

    def check_daily_heartbeat(self):
        """Check if daily heartbeat should be sent."""
        heartbeat_config = self.config.get("heartbeat", {})
        if not heartbeat_config.get("enabled", True) or not heartbeat_config.get("daily_status_enabled", True):
            return

        daily_time_str = heartbeat_config.get("daily_status_time", "09:00")
        try:
            target_hour, target_minute = map(int, daily_time_str.split(":"))
        except:
            logger.warning(f"Invalid daily_status_time format: {daily_time_str}")
            return

        now = datetime.now()
        target_time = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)

        # Check if we should send heartbeat (once per day within 1 minute window)
        if self.last_daily_heartbeat is None or \
           (self.last_daily_heartbeat.date() < now.date() and now >= target_time and now < target_time + timedelta(minutes=1)):

            # Calculate uptime stats
            uptime_minutes = 0
            if self.connection_start_time:
                uptime_minutes = int((time.time() - self.connection_start_time) / 60)

            alert_count = len(self.last_alerts)

            details = {
                "uptime_minutes": uptime_minutes,
                "alert_count": alert_count,
                "reconnection_count": len(self.reconnection_failures)
            }

            self.send_heartbeat_email("daily_status", details)
            self.last_daily_heartbeat = now

    def check_connection_health(self, disconnect_duration: float):
        """Monitor connection health and alert on prolonged disconnections."""
        heartbeat_config = self.config.get("heartbeat", {})
        if not heartbeat_config.get("enabled", True):
            return

        threshold_minutes = heartbeat_config.get("connection_failure_threshold_minutes", 15)
        threshold_seconds = threshold_minutes * 60

        if disconnect_duration > threshold_seconds and not self.heartbeat_sent_for_current_outage:
            self.send_heartbeat_email("connection_lost", {
                "duration_minutes": int(disconnect_duration / 60),
                "threshold_minutes": threshold_minutes
            })
            self.heartbeat_sent_for_current_outage = True

    def track_reconnection_failure(self, reason: str):
        """Track reconnection failure attempts and alert if threshold exceeded."""
        heartbeat_config = self.config.get("heartbeat", {})
        if not heartbeat_config.get("enabled", True):
            return

        self.reconnection_failures.append((time.time(), reason))

        # Check failures within window
        window_minutes = heartbeat_config.get("reconnection_failure_window_minutes", 30)
        window_seconds = window_minutes * 60
        now = time.time()

        recent_failures = [f for f in self.reconnection_failures if now - f[0] < window_seconds]

        threshold = heartbeat_config.get("reconnection_failure_threshold", 3)
        if len(recent_failures) >= threshold:
            self.send_heartbeat_email("reconnection_failures", {
                "failure_count": len(recent_failures),
                "threshold": threshold,
                "window_minutes": window_minutes,
                "reasons": [f[1] for f in recent_failures[-3:]]  # Last 3 reasons
            })
            # Clear the list so we don't spam alerts
            self.reconnection_failures.clear()

    def send_heartbeat_email(self, heartbeat_type: str, details: dict):
        """Send heartbeat email with status information."""
        smtp_config = self.config.get("smtp", {})
        heartbeat_config = self.config.get("heartbeat", {})

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if heartbeat_type == "daily_status":
            subject = f"[Heartbeat] Daily Status Report - {timestamp}"
            body = f"""
            Email Alert System Daily Status

            Timestamp: {timestamp}

            Statistics:
            - Uptime: {details.get('uptime_minutes', 0)} minutes
            - Alerts Triggered: {details.get('alert_count', 0)}
            - Reconnections: {details.get('reconnection_count', 0)}

            System is operating normally.
            """

        elif heartbeat_type == "connection_lost":
            subject = f"[Alert] Connection Lost - Disconnected for {details.get('duration_minutes', 0)} minutes"
            body = f"""
            Email Alert System Connection Failure

            Timestamp: {timestamp}

            The WebSocket connection to the transcription server has been disconnected for {details.get('duration_minutes', 0)} minutes (threshold: {details.get('threshold_minutes', 15)} minutes).

            Please check the server status and network connectivity.
            """

        elif heartbeat_type == "reconnection_failures":
            subject = f"[Alert] Repeated Reconnection Failures - {details.get('failure_count', 0)} attempts"
            body = f"""
            Email Alert System Reconnection Failures

            Timestamp: {timestamp}

            Failed to reconnect {details.get('failure_count', 0)} times within the last {details.get('window_minutes', 30)} minutes.
            Threshold: {details.get('threshold', 3)} failures.

            Recent Failures:
            """
            for reason in details.get('reasons', []):
                body += f"\n- {reason}"
            body += """

            Please investigate immediately.
            """

        else:
            return

        msg = MIMEMultipart()
        msg['From'] = smtp_config.get("from_address", "Alert System")
        msg['To'] = ", ".join(smtp_config.get("to_addresses", []))
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP(smtp_config.get("server"), smtp_config.get("port"))
            if smtp_config.get("use_tls", True):
                server.starttls()

            if smtp_config.get("username") and smtp_config.get("password"):
                server.login(smtp_config.get("username"), smtp_config.get("password"))

            server.send_message(msg)
            server.quit()
            logger.info(f"Heartbeat email sent (type={heartbeat_type})")
        except Exception as e:
            logger.error(f"Failed to send heartbeat email: {e}")

    def fetch_initial_context(self):
        """Fetch existing transcripts from the server to warm the context buffer."""
        ws_url = self.config["websocket"]["url"]
        # Convert ws:// to http:// and extract the base URL
        http_url = ws_url.replace("ws://", "http://").replace("wss://", "https://").replace("/ws", "/api/session/current")
        
        logger.info(f"Fetching initial context from {http_url}...")
        
        # Create SSL context to handle self-signed certificates
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        try:
            with urllib.request.urlopen(http_url, timeout=5, context=ssl_context) as response:
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

                # Create SSL context for wss:// if needed
                ssl_context = None
                if uri.startswith("wss://"):
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE

                async with websockets.connect(uri, ssl=ssl_context) as websocket:
                    logger.info("Connected to transcription feed.")
                    self.connection_start_time = time.time()
                    self.last_connected_time = time.time()
                    self.heartbeat_sent_for_current_outage = False

                    while True:
                        # Check daily heartbeat periodically
                        self.check_daily_heartbeat()

                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=30)
                        except asyncio.TimeoutError:
                            # Periodic check without blocking
                            continue

                        # Handle binary audio data
                        if isinstance(message, bytes):
                            self.audio_buffer.add_chunk(message, time.time())
                            self.last_connected_time = time.time()
                            continue

                        data = json.loads(message)

                        if data.get("type") == "transcript":
                            text = data.get("text", "")
                            origin_time = data.get("origin_time")
                            duration = data.get("duration")
                            timestamp = data.get("timestamp")

                            # Sync audio timing with transcript
                            if origin_time:
                                self.audio_buffer.sync_with_transcript(origin_time, time.time())

                            # Check against alert rules with full transcript context
                            matched_rules = self.check_keywords(text, transcript_data=data)

                            for rule in matched_rules:
                                rule_id = rule.get("rule_id", "unknown")
                                rule_desc = rule.get("description", rule_id)
                                logger.warning(f"MATCH FOUND: Rule '{rule_id}' ({rule_desc}) in transcript: {text}")

                                audio_clip = None
                                if attach_audio and origin_time and duration:
                                    audio_config = self.config.get("audio", {})
                                    padding = audio_config.get("extraction_padding_seconds", 0.5)
                                    audio_clip = self.audio_buffer.extract_clip(origin_time, duration, padding)

                                # Send email with rule metadata
                                self.send_email(
                                    keyword=rule.get("pattern", rule_id),
                                    transcript_data=data,
                                    audio_clip=audio_clip,
                                    rule_metadata=rule
                                )

                            # Add to history for context
                            self.transcript_history.append(data)
                            self.last_connected_time = time.time()

            except (websockets.ConnectionClosed, ConnectionRefusedError) as e:
                logger.warning(f"WebSocket connection lost: {e}")
                self.track_reconnection_failure(str(e))

                # Check connection health
                if self.connection_start_time:
                    disconnect_duration = time.time() - self.last_connected_time
                    self.check_connection_health(disconnect_duration)

                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Unexpected error in monitor loop: {e}")
                self.track_reconnection_failure(str(e))
                await asyncio.sleep(5)

    async def process_retry_queue(self):
        """Periodically process failed emails from retry queue."""
        if not self.retry_queue:
            return

        retry_config = self.config.get("retry", {})
        check_interval = 30  # Check every 30 seconds

        while self.running:
            try:
                await asyncio.sleep(check_interval)

                # Get ready emails
                ready_emails = self.retry_queue.get_ready_emails()
                if not ready_emails:
                    continue

                logger.info(f"Processing {len(ready_emails)} emails from retry queue")

                for email_record in ready_emails:
                    try:
                        # Load audio clip from temp file if exists
                        audio_clip = None
                        if email_record.get("audio_clip_path"):
                            audio_clip = email_record["audio_clip_path"]

                        # Add retry metadata
                        transcript_data = email_record["transcript_data"]
                        transcript_data["_retry_id"] = email_record["id"]
                        transcript_data["_retry_count"] = email_record["retry_count"]

                        # Send email
                        self.send_email(email_record["keyword"], transcript_data, audio_clip=audio_clip)

                    except Exception as e:
                        logger.error(f"Failed to process retry email {email_record['id']}: {e}")
                        self.retry_queue.mark_failure(email_record["id"], str(e))

                # Cleanup old audio files
                self.retry_queue.cleanup_old_audio_files()

                # Check for repeated failures
                if retry_config.get("alert_on_repeated_failures", True):
                    threshold = retry_config.get("repeated_failure_threshold", 5)
                    if self.retry_queue.consecutive_failures >= threshold:
                        logger.error(f"Email system has {self.retry_queue.consecutive_failures} consecutive failures!")

            except Exception as e:
                logger.error(f"Error in retry queue processor: {e}")

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
        """Run the alert tool with concurrent monitoring and retry queue tasks."""
        try:
            async def run_both():
                """Run monitor feed and retry queue concurrently."""
                await asyncio.gather(
                    self.monitor_feed(),
                    self.process_retry_queue()
                )

            asyncio.run(run_both())
        except KeyboardInterrupt:
            logger.info("Alert tool stopped by user.")
            self.running = False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live Transcription Email Alert Tool")
    parser.add_argument("--config", default="email_config.json", help="Path to configuration file")
    parser.add_argument("--test", action="store_true", help="Send a test email and exit")
    parser.add_argument("--validate", action="store_true",
                       help="Validate configuration and test SMTP connection, then exit")
    args = parser.parse_args()

    tool = EmailAlertTool(config_path=args.config)

    if args.validate:
        errors = tool.validate_email_config()
        success, msg = tool.test_smtp_connection()
        if errors:
            logger.error(f"Configuration validation failed with {len(errors)} error(s)")
            exit(1)
        if not success:
            logger.error("SMTP connection test failed")
            exit(1)
        logger.info("Configuration validation and SMTP connection test passed")
        exit(0)

    if args.test:
        tool.send_test_email()
    else:
        tool.run()
