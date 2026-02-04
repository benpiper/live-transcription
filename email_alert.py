#!/usr/bin/env python3
"""
Email Alert Tool for Live Transcription.

Monitors the live transcription feed via WebSocket and sends email alerts
when specific keywords are detected.
"""

import asyncio
import json
import logging
import smtplib
import time
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EmailAlert")

class EmailAlertTool:
    def __init__(self, config_path="email_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.last_alerts = {}  # keyword: timestamp
        self.running = True

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

    def send_email(self, keyword, transcript_data):
        """Send an email alert via SMTP."""
        smtp_config = self.config["smtp"]
        alert_config = self.config["alerts"]
        
        timestamp = transcript_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        speaker = transcript_data.get("speaker", "Unknown")
        text = transcript_data.get("text", "")

        subject = f"{alert_config.get('subject_prefix', '[Alert]')} Keyword Detected: {keyword}"
        
        body = f"""
        Alert triggered for keyword: {keyword}
        
        Details:
        - Time: {timestamp}
        - Speaker: {speaker}
        - Transcript: {text}
        
        ---
        This is an automated alert from the Live Transcription System.
        """

        msg = MIMEMultipart()
        msg['From'] = smtp_config["from_address"]
        msg['To'] = ", ".join(smtp_config["to_addresses"])
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP(smtp_config["server"], smtp_config["port"])
            if smtp_config.get("use_tls", True):
                server.starttls()
            
            if smtp_config.get("username") and smtp_config.get("password"):
                server.login(smtp_config["username"], smtp_config["password"])
            
            server.send_message(msg)
            server.quit()
            logger.info(f"Email alert sent for keyword '{keyword}'")
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

    async def monitor_feed(self):
        """Connect to WebSocket and monitor for keyword matches."""
        uri = self.config["websocket"]["url"]
        
        while self.running:
            try:
                logger.info(f"Connecting to WebSocket at {uri}...")
                async with websockets.connect(uri) as websocket:
                    logger.info("Connected to transcription feed.")
                    while True:
                        message = await websocket.recv()
                        
                        # Only process text (JSON) messages; skip binary (audio) data
                        if isinstance(message, bytes):
                            continue
                            
                        data = json.loads(message)
                        
                        if data.get("type") == "transcript":
                            text = data.get("text", "")
                            matched = self.check_keywords(text)
                            
                            for kw in matched:
                                logger.warning(f"MATCH FOUND: '{kw}' in transcript: {text}")
                                self.send_email(kw, data)
                                
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
