import copy
import json
from datetime import datetime, timedelta

import pytest

import config
import session as session_module
from session import TranscriptionSession


@pytest.fixture
def isolated_session_store(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    archive_dir = sessions_dir / "archive"
    monkeypatch.setattr(session_module, "SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(session_module, "ARCHIVE_DIR", archive_dir)
    monkeypatch.setattr(session_module, "_session", None)

    original_config = copy.deepcopy(config.TRANSCRIPTION_CONFIG)
    yield sessions_dir, archive_dir

    config.TRANSCRIPTION_CONFIG.clear()
    config.TRANSCRIPTION_CONFIG.update(original_config)
    session_module._session = None


def make_transcript(index=0, timestamp=None, speaker="Speaker 1"):
    return {
        "timestamp": timestamp or f"2026-04-29T12:{index:02d}:00",
        "speaker": speaker,
        "text": f"Transcript {index}",
        "confidence": -0.25,
    }


def write_session_file(directory, name, updated_at, transcripts=None):
    directory.mkdir(parents=True, exist_ok=True)
    data = {
        "name": name,
        "created_at": "2026-04-29T08:00:00",
        "updated_at": updated_at,
        "transcript_count": len(transcripts or []),
        "transcripts": transcripts or [],
    }
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
    path = directory / f"{safe_name}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def set_session_management(**overrides):
    config.TRANSCRIPTION_CONFIG["session_management"] = {
        **config.DEFAULT_CONFIG["session_management"],
        **overrides,
    }


def test_add_transcript_caps_history_and_updates_timestamp(isolated_session_store):
    session = TranscriptionSession(session_name="Dispatch")
    original_updated_at = session.updated_at

    for i in range(2001):
        session.add_transcript(make_transcript(i))

    assert len(session.transcripts) == 2000
    assert session.transcripts[0]["text"] == "Transcript 1"
    assert session.transcripts[-1]["text"] == "Transcript 2000"
    assert session.updated_at != original_updated_at


def test_save_session_sanitizes_filename_and_preserves_name(isolated_session_store):
    sessions_dir, _ = isolated_session_store
    session = TranscriptionSession(session_name="Shift: Alpha/Bravo?")
    session.add_transcript(make_transcript())

    path = session_module.save_session(session)

    assert path == str(sessions_dir / "Shift_ Alpha_Bravo_.json")
    assert session.name == "Shift: Alpha/Bravo?"
    saved = json.loads((sessions_dir / "Shift_ Alpha_Bravo_.json").read_text())
    assert saved["name"] == "Shift: Alpha/Bravo?"
    assert saved["transcript_count"] == 1


def test_save_session_with_name_override_updates_name(isolated_session_store):
    sessions_dir, _ = isolated_session_store
    session = TranscriptionSession(session_name="Original")

    path = session_module.save_session(session, name="Renamed Session")

    assert path == str(sessions_dir / "Renamed Session.json")
    assert session.name == "Renamed Session"


def test_save_session_without_active_session_raises(isolated_session_store):
    with pytest.raises(ValueError, match="No session to save"):
        session_module.save_session()


def test_load_session_from_file_restores_data_and_sets_current_session(isolated_session_store):
    sessions_dir, _ = isolated_session_store
    transcripts = [make_transcript(1, speaker="Dispatcher")]
    write_session_file(sessions_dir, "Morning Shift", "2026-04-29T09:00:00", transcripts)

    loaded = session_module.load_session_from_file("Morning Shift")

    assert loaded.name == "Morning Shift"
    assert loaded.transcripts == transcripts
    assert session_module.get_session() is loaded


def test_load_missing_session_raises_file_not_found(isolated_session_store):
    with pytest.raises(FileNotFoundError):
        session_module.load_session_from_file("Missing")


def test_list_sessions_sorts_by_updated_at_descending_and_skips_bad_files(isolated_session_store):
    sessions_dir, _ = isolated_session_store
    write_session_file(sessions_dir, "Older", "2026-04-29T08:00:00")
    write_session_file(sessions_dir, "Newest", "2026-04-29T10:00:00")
    write_session_file(sessions_dir, "Middle", "2026-04-29T09:00:00")
    (sessions_dir / "bad.json").write_text("{not json", encoding="utf-8")

    sessions = session_module.list_sessions()

    assert [s["name"] for s in sessions] == ["Newest", "Middle", "Older"]
    assert sessions[0]["file"] == "Newest.json"


def test_list_sessions_returns_empty_when_directory_missing(isolated_session_store):
    assert session_module.list_sessions() == []


def test_delete_session_removes_saved_file(isolated_session_store):
    sessions_dir, _ = isolated_session_store
    write_session_file(sessions_dir, "Delete Me", "2026-04-29T08:00:00")

    assert session_module.delete_session("Delete Me") is True
    assert not (sessions_dir / "Delete Me.json").exists()
    assert session_module.delete_session("Delete Me") is False


def test_archive_restore_and_list_archived_sessions(isolated_session_store):
    sessions_dir, archive_dir = isolated_session_store
    write_session_file(sessions_dir, "Archive Me", "2026-04-29T08:00:00", [make_transcript()])

    assert session_module.archive_session("Archive Me") is True
    assert not (sessions_dir / "Archive Me.json").exists()
    assert (archive_dir / "Archive Me.json").exists()

    archived = session_module.list_archived_sessions()
    assert [s["name"] for s in archived] == ["Archive Me"]
    assert archived[0]["transcript_count"] == 1

    assert session_module.restore_session("Archive Me") is True
    assert (sessions_dir / "Archive Me.json").exists()
    assert not (archive_dir / "Archive Me.json").exists()


def test_archive_and_restore_missing_sessions_return_false(isolated_session_store):
    assert session_module.archive_session("Missing") is False
    assert session_module.restore_session("Missing") is False


def test_list_archived_sessions_returns_empty_when_archive_missing(isolated_session_store):
    assert session_module.list_archived_sessions() == []


def test_cleanup_old_sessions_archives_only_stale_sessions(isolated_session_store):
    sessions_dir, archive_dir = isolated_session_store
    old_updated_at = (datetime.now() - timedelta(days=45)).isoformat()
    recent_updated_at = (datetime.now() - timedelta(days=3)).isoformat()
    write_session_file(sessions_dir, "Old Shift", old_updated_at)
    write_session_file(sessions_dir, "Recent Shift", recent_updated_at)
    write_session_file(sessions_dir, "No Updated At", "")

    archived_count = session_module.cleanup_old_sessions(days_threshold=30)

    assert archived_count == 1
    assert (archive_dir / "Old Shift.json").exists()
    assert (sessions_dir / "Recent Shift.json").exists()
    assert (sessions_dir / "No Updated At.json").exists()


def test_cleanup_old_sessions_returns_zero_when_directory_missing(isolated_session_store):
    assert session_module.cleanup_old_sessions(days_threshold=30) == 0


def test_rollover_status_without_active_session(isolated_session_store):
    status = session_module.get_session_rollover_status()

    assert status == {
        "current_session_name": None,
        "created_at": None,
        "transcript_count": 0,
        "time_since_creation_seconds": 0,
        "hours_until_rollover": None,
        "transcripts_until_rollover": None,
        "will_rollover_by": None,
    }


def test_rollover_status_when_rollover_disabled(isolated_session_store):
    set_session_management(enable_rollover=False)
    session = TranscriptionSession(session_name="No Rollover")
    session.created_at = (datetime.now() - timedelta(hours=2)).isoformat()
    session.add_transcript(make_transcript())

    status = session_module.get_session_rollover_status(session)

    assert status["current_session_name"] == "No Rollover"
    assert status["transcript_count"] == 1
    assert status["hours_until_rollover"] is None
    assert status["transcripts_until_rollover"] is None
    assert status["will_rollover_by"] is None


def test_rollover_status_time_trigger(isolated_session_store):
    set_session_management(
        enable_rollover=True,
        rollover_time_hours=1,
        rollover_transcript_count=100,
    )
    session = TranscriptionSession(session_name="Time Triggered")
    session.created_at = (datetime.now() - timedelta(hours=2)).isoformat()

    status = session_module.get_session_rollover_status(session)

    assert status["hours_until_rollover"] == 0
    assert status["transcripts_until_rollover"] == 100
    assert status["will_rollover_by"] == "time"


def test_rollover_status_count_trigger(isolated_session_store):
    set_session_management(
        enable_rollover=True,
        rollover_time_hours=24,
        rollover_transcript_count=2,
    )
    session = TranscriptionSession(session_name="Count Triggered")
    session.add_transcript(make_transcript(1))
    session.add_transcript(make_transcript(2))

    status = session_module.get_session_rollover_status(session)

    assert status["hours_until_rollover"] > 23
    assert status["transcripts_until_rollover"] == 0
    assert status["will_rollover_by"] == "count"
