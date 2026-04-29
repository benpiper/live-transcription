import copy
import json

import pytest

import config


@pytest.fixture(autouse=True)
def reset_config_state():
    original_default = copy.deepcopy(config.DEFAULT_CONFIG)
    original_config = copy.deepcopy(config.TRANSCRIPTION_CONFIG)

    config.TRANSCRIPTION_CONFIG.clear()
    config.TRANSCRIPTION_CONFIG.update(copy.deepcopy(config.DEFAULT_CONFIG))
    yield

    config.DEFAULT_CONFIG.clear()
    config.DEFAULT_CONFIG.update(original_default)
    config.TRANSCRIPTION_CONFIG.clear()
    config.TRANSCRIPTION_CONFIG.update(original_config)


def write_config(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_missing_config_file_uses_defaults(tmp_path):
    missing_path = tmp_path / "missing.json"

    loaded = config.load_config(str(missing_path))

    assert loaded == config.TRANSCRIPTION_CONFIG
    assert config.get_setting("model_size") == config.DEFAULT_CONFIG["settings"]["model_size"]
    assert config.get_vocabulary() == []
    assert config.get_corrections() == {}


def test_invalid_json_does_not_crash_and_keeps_defaults(tmp_path):
    invalid_path = tmp_path / "bad.json"
    invalid_path.write_text("{not valid json", encoding="utf-8")

    loaded = config.load_config(str(invalid_path))

    assert loaded == config.TRANSCRIPTION_CONFIG
    assert config.get_setting("device") == "auto"
    assert config.get_setting("beam_size") == 5


def test_user_config_merges_settings_without_dropping_defaults(tmp_path):
    config_path = write_config(
        tmp_path / "config.json",
        {
            "vocabulary": ["Medic 4", "Engine 7"],
            "corrections": {"main straight": "Main Street"},
            "settings": {
                "model_size": "small.en",
                "noise_floor": 0.005,
            },
        },
    )

    loaded = config.load_config(str(config_path))

    assert loaded["vocabulary"] == ["Medic 4", "Engine 7"]
    assert loaded["corrections"] == {"main straight": "Main Street"}
    assert config.get_setting("model_size") == "small.en"
    assert config.get_setting("noise_floor") == 0.005
    assert config.get_setting("device") == "auto"
    assert config.get_setting("beam_size") == 5
    assert config.get_setting("audio_buffer_window_sec") == 7200


def test_user_config_merges_session_management_without_dropping_defaults(tmp_path):
    config_path = write_config(
        tmp_path / "config.json",
        {
            "session_management": {
                "enable_rollover": True,
                "rollover_time_hours": 12,
            }
        },
    )

    loaded = config.load_config(str(config_path))

    assert loaded["session_management"]["enable_rollover"] is True
    assert loaded["session_management"]["rollover_time_hours"] == 12
    assert loaded["session_management"]["rollover_transcript_count"] == 10000
    assert loaded["session_management"]["archive_age_days"] == 30


def test_load_config_does_not_mutate_defaults_or_leak_between_loads(tmp_path):
    first_path = write_config(
        tmp_path / "first.json",
        {
            "settings": {"model_size": "tiny"},
            "session_management": {"rollover_time_hours": 6},
        },
    )
    second_path = write_config(tmp_path / "second.json", {"settings": {"device": "cpu"}})

    config.load_config(str(first_path))
    config.load_config(str(second_path))

    assert config.DEFAULT_CONFIG["settings"]["model_size"] == "medium.en"
    assert config.DEFAULT_CONFIG["session_management"]["rollover_time_hours"] == 24
    assert config.get_setting("model_size") == "medium.en"
    assert config.get_setting("device") == "cpu"
    assert config.get_session_management_setting("rollover_time_hours") == 24


@pytest.mark.parametrize(
    ("setting", "value", "expected_message"),
    [
        ("beam_size", "5", "'beam_size' should be int, got str"),
        ("min_silence_duration_ms", 500.5, "'min_silence_duration_ms' should be int, got float"),
        ("cpu_threads", "4", "'cpu_threads' should be int, got str"),
        ("min_speaker_samples", "16000", "'min_speaker_samples' should be int, got str"),
        ("no_speech_threshold", "0.8", "'no_speech_threshold' should be number, got str"),
        ("log_prob_threshold", "-0.8", "'log_prob_threshold' should be number, got str"),
        ("compression_ratio_threshold", "2.4", "'compression_ratio_threshold' should be number, got str"),
        ("avg_logprob_cutoff", "-0.8", "'avg_logprob_cutoff' should be number, got str"),
        ("no_speech_prob_cutoff", "0.2", "'no_speech_prob_cutoff' should be number, got str"),
        ("extreme_confidence_cutoff", "-0.4", "'extreme_confidence_cutoff' should be number, got str"),
        ("min_window_sec", "1.0", "'min_window_sec' should be number, got str"),
        ("max_window_sec", "5.0", "'max_window_sec' should be number, got str"),
        ("noise_floor", "0.001", "'noise_floor' should be number, got str"),
        ("diarization_threshold", "0.35", "'diarization_threshold' should be number, got str"),
        ("voice_match_threshold", "0.7", "'voice_match_threshold' should be number, got str"),
        ("audio_buffer_window_sec", "7200", "'audio_buffer_window_sec' should be number, got str"),
        ("audio_buffer_chunk_size_sec", "0.256", "'audio_buffer_chunk_size_sec' should be number, got str"),
        ("device", 123, "'device' should be str, got int"),
        ("model_size", 123, "'model_size' should be str, got int"),
        ("compute_type", 123, "'compute_type' should be str, got int"),
        ("voice_profiles_dir", 123, "'voice_profiles_dir' should be str, got int"),
    ],
)
def test_validate_config_catches_invalid_setting_types(setting, value, expected_message):
    config.TRANSCRIPTION_CONFIG["settings"][setting] = value

    errors = config.validate_config()

    assert expected_message in errors


@pytest.mark.parametrize(
    ("setting", "value", "expected_message"),
    [
        ("beam_size", 0, "beam_size should be between 1 and 20"),
        ("beam_size", 21, "beam_size should be between 1 and 20"),
        ("cpu_threads", 0, "cpu_threads must be at least 1"),
        ("noise_floor", -0.1, "noise_floor should be between 0.0 and 1.0"),
        ("noise_floor", 1.1, "noise_floor should be between 0.0 and 1.0"),
        ("diarization_threshold", -0.1, "diarization_threshold should be between 0.0 and 1.0"),
        ("diarization_threshold", 1.1, "diarization_threshold should be between 0.0 and 1.0"),
        ("audio_buffer_window_sec", 299, "audio_buffer_window_sec must be at least 300 seconds (5 minutes)"),
        ("audio_buffer_window_sec", 86401, "audio_buffer_window_sec must be at most 86400 seconds (24 hours)"),
        ("min_speaker_samples", 3999, "min_speaker_samples should be at least 4000 (0.25 seconds)"),
    ],
)
def test_validate_config_catches_invalid_setting_ranges(setting, value, expected_message):
    config.TRANSCRIPTION_CONFIG["settings"][setting] = value

    errors = config.validate_config()

    assert expected_message in errors


def test_validate_config_catches_invalid_window_order():
    config.TRANSCRIPTION_CONFIG["settings"]["min_window_sec"] = 5.0
    config.TRANSCRIPTION_CONFIG["settings"]["max_window_sec"] = 5.0

    errors = config.validate_config()

    assert "max_window_sec (5.0) must be greater than min_window_sec (5.0)" in errors


def test_validate_config_catches_bad_model_size_and_device():
    config.TRANSCRIPTION_CONFIG["settings"]["model_size"] = "xl"
    config.TRANSCRIPTION_CONFIG["settings"]["device"] = "tpu"

    errors = config.validate_config()

    assert any("model_size 'xl' is not valid" in error for error in errors)
    assert "device 'tpu' is not valid. Choose from: auto, cuda, cpu" in errors


@pytest.mark.parametrize(
    ("setting", "value", "expected_message"),
    [
        ("rollover_time_hours", 0, "rollover_time_hours must be greater than 0"),
        ("rollover_time_hours", -1, "rollover_time_hours must be greater than 0"),
        ("rollover_transcript_count", 0, "rollover_transcript_count must be greater than 0"),
        ("rollover_transcript_count", -1, "rollover_transcript_count must be greater than 0"),
        ("archive_age_days", 0, "archive_age_days must be greater than 0"),
        ("archive_age_days", -1, "archive_age_days must be greater than 0"),
    ],
)
def test_validate_config_catches_invalid_session_management_ranges(setting, value, expected_message):
    config.TRANSCRIPTION_CONFIG["session_management"][setting] = value

    errors = config.validate_config()

    assert expected_message in errors


def test_validate_config_catches_invalid_session_management_types():
    config.TRANSCRIPTION_CONFIG["session_management"]["rollover_time_hours"] = "daily"

    errors = config.validate_config()

    assert "session_management settings have invalid types" in errors


def test_validate_config_passes_default_config():
    assert config.validate_config() == []


def test_tests_restore_config_state_between_cases():
    config.TRANSCRIPTION_CONFIG["settings"]["model_size"] = "tiny"
    config.TRANSCRIPTION_CONFIG["vocabulary"].append("temporary")

    assert config.get_setting("model_size") == "tiny"
    assert config.get_vocabulary() == ["temporary"]
