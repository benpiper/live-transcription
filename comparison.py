"""
Session comparison engine for analyzing transcripts across multiple sessions.

This module provides functionality to compare transcripts from different sessions
in two modes: merged (chronological timeline) and side-by-side (columnar view).
"""

import logging
from typing import List, Optional, Dict, Any

from session import load_session_from_file, TranscriptionSession

logger = logging.getLogger(__name__)


class SessionComparator:
    """Compares transcripts across multiple sessions."""

    def __init__(self, session_names: List[str]):
        """
        Initialize the comparator with session names.

        Args:
            session_names: List of session names to compare (2-3 sessions)

        Raises:
            ValueError: If number of sessions is invalid or sessions not found
        """
        if not session_names or len(session_names) < 2 or len(session_names) > 3:
            raise ValueError("Must provide 2-3 session names for comparison")

        self.sessions = []
        self.session_names = session_names

        # Load all sessions
        for name in session_names:
            try:
                session = load_session_from_file(name)
                self.sessions.append(session)
            except FileNotFoundError:
                logger.warning(f"Session not found: {name}")
                raise ValueError(f"Session not found: {name}")

        logger.info(f"Loaded {len(self.sessions)} sessions for comparison")

    def compare_merged(self, speaker_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Merge transcripts from all sessions sorted by timestamp.

        Creates a chronological timeline of all transcripts from all sessions,
        with each transcript tagged with its source session. Useful for analyzing
        events across different sessions chronologically.

        Algorithm:
        1. Collect all transcripts from all sessions
        2. Add 'session_name' field to each transcript
        3. Sort by timestamp (lexicographic, works for ISO 8601 formats)
        4. Apply optional speaker filter
        5. Calculate aggregate statistics

        Args:
            speaker_filter: Optional list of speaker names to include

        Returns:
            Dict with 'transcripts' list and 'stats' dict
        """
        all_transcripts = []

        # Collect transcripts from all sessions
        for session in self.sessions:
            for transcript in session.transcripts:
                enriched = dict(transcript)
                enriched['session_name'] = session.name
                all_transcripts.append(enriched)

        # Sort by timestamp (ISO 8601 sorts correctly lexicographically)
        all_transcripts.sort(key=lambda t: t.get('timestamp', ''))

        # Apply speaker filter if provided
        if speaker_filter:
            all_transcripts = [
                t for t in all_transcripts
                if t.get('speaker') in speaker_filter
            ]

        return {
            "transcripts": all_transcripts,
            "stats": self._calculate_stats(all_transcripts)
        }

    def compare_side_by_side(self, speaker_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Group transcripts by session in parallel columns.

        Creates separate transcript lists for each session, allowing direct
        comparison of what each session contains. Useful for analyzing
        differences between specific sessions.

        Algorithm:
        1. Create per-session transcript lists
        2. Sort transcripts within each session by timestamp
        3. Apply optional speaker filter
        4. Return nested structure for column-based rendering
        5. Calculate aggregate statistics

        Args:
            speaker_filter: Optional list of speaker names to include

        Returns:
            Dict with 'by_session' structure and 'stats' dict
        """
        by_session = {}

        for session in self.sessions:
            transcripts = [dict(t) for t in session.transcripts]

            # Sort by timestamp within session
            transcripts.sort(key=lambda t: t.get('timestamp', ''))

            # Apply speaker filter
            if speaker_filter:
                transcripts = [
                    t for t in transcripts
                    if t.get('speaker') in speaker_filter
                ]

            by_session[session.name] = transcripts

        # Flatten for stats calculation
        all_transcripts = []
        for transcripts in by_session.values():
            all_transcripts.extend(transcripts)

        return {
            "by_session": by_session,
            "stats": self._calculate_stats(all_transcripts)
        }

    def _calculate_stats(self, transcripts: List[Dict]) -> Dict[str, Any]:
        """
        Calculate comparison statistics.

        Args:
            transcripts: List of transcript dicts

        Returns:
            Stats dict with counts, speaker info, and time range
        """
        if not transcripts:
            return {
                "total_transcripts": 0,
                "speaker_count": 0,
                "session_count": len(self.sessions),
                "time_range": None
            }

        # Count unique speakers
        speakers = set()
        for t in transcripts:
            if t.get('speaker'):
                speakers.add(t.get('speaker'))

        # Find time range
        timestamps = [t.get('timestamp') for t in transcripts if t.get('timestamp')]
        time_range = None
        if timestamps:
            time_range = {
                "earliest": min(timestamps),
                "latest": max(timestamps)
            }

        return {
            "total_transcripts": len(transcripts),
            "speaker_count": len(speakers),
            "session_count": len(self.sessions),
            "time_range": time_range
        }


def compare_sessions(
    session_names: List[str],
    mode: str = "merged",
    speaker_filter: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compare multiple sessions and return results.

    Wrapper function that creates a comparator and returns results in the
    format expected by the API response model.

    Args:
        session_names: List of session names to compare (2-3)
        mode: Comparison mode ('merged' or 'side-by-side')
        speaker_filter: Optional list of speaker names to filter

    Returns:
        Dict formatted for SessionComparisonResponse

    Raises:
        ValueError: Invalid parameters or sessions not found
    """
    if mode not in ["merged", "side-by-side"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'merged' or 'side-by-side'")

    comparator = SessionComparator(session_names)

    if mode == "merged":
        comparison_result = comparator.compare_merged(speaker_filter)
    else:
        comparison_result = comparator.compare_side_by_side(speaker_filter)

    return {
        "mode": mode,
        "sessions": session_names,
        "stats": comparison_result["stats"],
        "results": comparison_result
    }
