"""
Pydantic models for API request/response validation and OpenAPI documentation.

This module defines all request and response schemas for the FastAPI application,
providing type safety, automatic validation, and interactive API documentation.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class TranscriptItem(BaseModel):
    """Individual transcript record with speaker and confidence."""
    timestamp: str = Field(..., description="ISO 8601 timestamp of the transcript")
    speaker: Optional[str] = Field(None, description="Identified speaker name or 'Speaker N'")
    text: str = Field(..., description="Transcribed text")
    confidence: float = Field(..., description="Confidence score (-1.0 to 0.0)")
    duration: Optional[float] = Field(None, description="Audio duration in seconds")
    origin_time: Optional[float] = Field(None, description="Original Unix timestamp before correction")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2026-02-08 14:30:45",
                "speaker": "Speaker 1",
                "text": "Engine 5 responding code 3",
                "confidence": -0.35,
                "duration": 2.3,
                "origin_time": "2026-02-08 14:30:43"
            }
        }


class SessionMetadata(BaseModel):
    """Metadata about a saved session."""
    name: str = Field(..., description="Session name")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    updated_at: str = Field(..., description="ISO 8601 last update timestamp")
    transcript_count: int = Field(..., description="Number of transcripts in the session")
    file: Optional[str] = Field(None, description="Filename of the session file")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Morning Shift",
                "created_at": "2026-02-08T08:00:00",
                "updated_at": "2026-02-08T12:00:00",
                "transcript_count": 1084,
                "file": "Morning_Shift.json"
            }
        }


class SessionFull(BaseModel):
    """Complete session data with all transcripts."""
    name: str = Field(..., description="Session name")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    updated_at: str = Field(..., description="ISO 8601 last update timestamp")
    transcript_count: int = Field(..., description="Number of transcripts")
    transcripts: List[TranscriptItem] = Field(default_factory=list, description="List of transcripts")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Morning Shift",
                "created_at": "2026-02-08T08:00:00",
                "updated_at": "2026-02-08T12:00:00",
                "transcript_count": 2,
                "transcripts": [
                    {
                        "timestamp": "2026-02-08 08:15:30",
                        "speaker": "Dispatcher A",
                        "text": "All units clear for shift change",
                        "confidence": -0.25
                    },
                    {
                        "timestamp": "2026-02-08 08:16:00",
                        "speaker": "Dispatcher B",
                        "text": "Acknowledged, shift briefing complete",
                        "confidence": -0.28
                    }
                ]
            }
        }


class SessionListResponse(BaseModel):
    """Response containing list of sessions."""
    sessions: List[SessionMetadata] = Field(..., description="List of saved sessions")


class SessionCreateRequest(BaseModel):
    """Request to create a new session."""
    name: Optional[str] = Field(None, description="Optional session name (auto-generated if not provided)")


class SessionCreateResponse(BaseModel):
    """Response after creating a session."""
    name: str = Field(..., description="The session name")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")


class SessionComparisonRequest(BaseModel):
    """Request to compare multiple sessions."""
    session_names: List[str] = Field(
        ...,
        min_length=2,
        max_length=3,
        description="Names of sessions to compare (2-3 sessions)"
    )
    mode: Literal["merged", "side-by-side"] = Field(
        default="merged",
        description="Comparison mode: 'merged' (chronological) or 'side-by-side' (columnar)"
    )
    speaker_filter: Optional[List[str]] = Field(
        None,
        description="Optional list of speaker names to filter (only these speakers shown)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_names": ["Morning Shift", "Evening Shift"],
                "mode": "merged",
                "speaker_filter": ["Dispatcher A", "Dispatcher B"]
            }
        }


class SessionComparisonStats(BaseModel):
    """Statistics about compared sessions."""
    total_transcripts: int = Field(..., description="Total transcripts across all sessions")
    speaker_count: int = Field(..., description="Unique speakers across all sessions")
    session_count: int = Field(..., description="Number of sessions being compared")
    time_range: Optional[dict] = Field(None, description="Earliest and latest timestamps")

    class Config:
        json_schema_extra = {
            "example": {
                "total_transcripts": 2048,
                "speaker_count": 5,
                "session_count": 2,
                "time_range": {
                    "earliest": "2026-02-08 08:00:00",
                    "latest": "2026-02-08 20:00:00"
                }
            }
        }


class SessionComparisonResponse(BaseModel):
    """Response with comparison results."""
    mode: str = Field(..., description="Comparison mode used")
    sessions: List[str] = Field(..., description="Names of sessions compared")
    stats: SessionComparisonStats = Field(..., description="Comparison statistics")
    results: dict = Field(..., description="Comparison results (format depends on mode)")

    class Config:
        json_schema_extra = {
            "example": {
                "mode": "merged",
                "sessions": ["Morning Shift", "Evening Shift"],
                "stats": {
                    "total_transcripts": 2048,
                    "speaker_count": 5,
                    "session_count": 2,
                    "time_range": {
                        "earliest": "2026-02-08 08:00:00",
                        "latest": "2026-02-08 20:00:00"
                    }
                },
                "results": {
                    "transcripts": [
                        {"session_name": "Morning Shift", "timestamp": "2026-02-08 08:00:00", "speaker": "Dispatcher A", "text": "Good morning dispatch", "confidence": -0.25},
                        {"session_name": "Evening Shift", "timestamp": "2026-02-08 17:00:00", "speaker": "Dispatcher B", "text": "Evening shift beginning", "confidence": -0.28}
                    ]
                }
            }
        }


class HealthCheckResponse(BaseModel):
    """Response for health check endpoint."""
    status: str = Field(..., description="Health status")
    connections: int = Field(..., description="Number of active WebSocket connections")


class CurrentSessionResponse(BaseModel):
    """Response with current active session."""
    active: bool = Field(..., description="Whether a session is currently active")
    name: Optional[str] = Field(None, description="Session name if active")
    created_at: Optional[str] = Field(None, description="Session creation timestamp if active")
    updated_at: Optional[str] = Field(None, description="Session last update timestamp if active")
    transcripts: Optional[List[TranscriptItem]] = Field(None, description="Current transcripts if active")


class DeleteSessionResponse(BaseModel):
    """Response after deleting a session."""
    deleted: bool = Field(..., description="Whether the session was successfully deleted")


class SaveSessionResponse(BaseModel):
    """Response after saving a session."""
    saved: bool = Field(..., description="Whether the session was successfully saved")
    path: str = Field(..., description="Path to the saved session file")


class ArchiveSessionResponse(BaseModel):
    """Response after archiving a session."""
    name: str = Field(..., description="Session name that was archived")
    archived: bool = Field(..., description="Whether the session was successfully archived")
    path: str = Field(..., description="Path to the archived session file")


class RestoreSessionResponse(BaseModel):
    """Response after restoring a session from archive."""
    name: str = Field(..., description="Session name that was restored")
    restored: bool = Field(..., description="Whether the session was successfully restored")
    path: str = Field(..., description="Path to the restored session file")


class SessionRolloverStatus(BaseModel):
    """Session rollover timer status."""
    current_session_name: Optional[str] = Field(None, description="Current active session name")
    created_at: Optional[str] = Field(None, description="Session creation timestamp")
    transcript_count: int = Field(..., description="Number of transcripts in current session")
    time_since_creation_seconds: float = Field(..., description="Seconds elapsed since session creation")
    hours_until_rollover: Optional[float] = Field(None, description="Hours until time-based rollover (if enabled)")
    transcripts_until_rollover: Optional[int] = Field(None, description="Transcripts remaining before count-based rollover (if enabled)")
    will_rollover_by: Optional[Literal["time", "count"]] = Field(None, description="What will trigger rollover: 'time', 'count', or None")

    class Config:
        json_schema_extra = {
            "example": {
                "current_session_name": "Morning Shift",
                "created_at": "2026-02-08T08:00:00",
                "transcript_count": 5234,
                "time_since_creation_seconds": 28800,
                "hours_until_rollover": 19.2,
                "transcripts_until_rollover": 4766,
                "will_rollover_by": None
            }
        }
class EngineStatusResponse(BaseModel):
    """Transcription engine processing status."""
    device: str = Field(..., description="Currently active device ('cuda' or 'cpu')")
    intended_device: str = Field(..., description="The device intended to be used")
    is_fallback: bool = Field(..., description="Whether the engine is currently in fallback mode")
    model_size: str = Field(..., description="The size of the Whisper model being used")
    compute_type: str = Field(..., description="The precision used for computation")
    fallback_count: int = Field(..., description="Number of times fallback has occurred")

    class Config:
        json_schema_extra = {
            "example": {
                "device": "cuda",
                "intended_device": "cuda",
                "is_fallback": False,
                "model_size": "medium.en",
                "compute_type": "int8_float16",
                "fallback_count": 0
            }
        }
