"""
Graceful shutdown handler for the live transcription application.

Handles SIGTERM and SIGINT signals to ensure proper resource cleanup,
session persistence, and WebSocket connection closure before exit.
"""

import signal
import sys
import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def register_shutdown_handlers(
    stop_event: threading.Event,
    cleanup_callback: Optional[Callable[[], None]] = None
) -> None:
    """
    Register signal handlers for graceful shutdown.

    Args:
        stop_event: Threading event to signal all threads to stop
        cleanup_callback: Optional callback function to perform cleanup operations
    """

    def _shutdown_handler(signum, frame):
        """Handle shutdown signal."""
        signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        logger.info(f"Received {signal_name}. Starting graceful shutdown...")
        print(f"\n{signal_name} received. Shutting down gracefully...")

        # Set the stop event to signal all threads
        stop_event.set()

        # Call cleanup callback if provided
        if cleanup_callback:
            try:
                cleanup_callback()
            except Exception as e:
                logger.error(f"Error during cleanup callback: {e}")

        # Exit the process
        sys.exit(0)

    # Register handlers for both SIGTERM and SIGINT
    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    logger.debug("Signal handlers registered for SIGTERM and SIGINT")


def wait_for_threads(threads: list, timeout: float = 10.0) -> bool:
    """
    Wait for daemon threads to complete with a timeout.

    Args:
        threads: List of threading.Thread objects to wait for
        timeout: Maximum time to wait in seconds

    Returns:
        True if all threads completed, False if timeout exceeded
    """
    if not threads:
        return True

    logger.info(f"Waiting for {len(threads)} thread(s) to complete (timeout: {timeout}s)...")

    for thread in threads:
        if thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning(f"Thread {thread.name} did not complete within timeout")
                return False

    logger.info("All threads completed")
    return True
