"""
Unit tests for alert_rules.py - AlertRule and AlertRuleEngine classes.

Tests pattern matching, context filtering, deduplication, and config migration.
"""

import pytest
import time
import re
from alert_rules import AlertRule, AlertRuleEngine


class TestAlertRule:
    """Test AlertRule matching logic."""

    def test_substring_match_basic(self):
        """Test basic substring matching."""
        rule = AlertRule(id="test", pattern="fire", match_type="substring")
        matched, meta = rule.matches("structure fire on Main St")
        assert matched is True
        assert meta["matched_text"] == "structure fire on Main St"

    def test_substring_match_case_insensitive(self):
        """Test case-insensitive substring matching."""
        rule = AlertRule(id="test", pattern="FIRE", match_type="substring")
        matched, _ = rule.matches("structure fire on Main St")
        assert matched is True

    def test_substring_no_match(self):
        """Test substring non-match."""
        rule = AlertRule(id="test", pattern="ambulance", match_type="substring")
        matched, meta = rule.matches("structure fire on Main St")
        assert matched is False
        assert meta["reason"] == "pattern_no_match"

    def test_word_boundary_match(self):
        """Test word boundary matching."""
        rule = AlertRule(id="test", pattern="fire", match_type="word")

        # Should match "fire" as standalone word
        matched, _ = rule.matches("structure fire on Main St")
        assert matched is True

        # Should NOT match "firearm"
        matched, meta = rule.matches("officer with firearm")
        assert matched is False
        assert meta["reason"] == "pattern_no_match"

    def test_word_boundary_multiple_keywords(self):
        """Test word matching with multiple keywords."""
        rule = AlertRule(
            id="test",
            pattern=["fire", "burning"],
            match_type="word"
        )

        matched, _ = rule.matches("structure fire detected")
        assert matched is True

        matched, _ = rule.matches("building burning")
        assert matched is True

        matched, _ = rule.matches("firearm found")
        assert matched is False

    def test_regex_match_basic(self):
        """Test basic regex matching."""
        rule = AlertRule(
            id="test",
            pattern=r"\b(Main|Oak|Elm)\s+(?:St|Street|Ave)",
            match_type="regex"
        )

        matched, _ = rule.matches("fire on Main Street")
        assert matched is True

        matched, _ = rule.matches("fire on Elm Ave")
        assert matched is True

        matched, _ = rule.matches("fire on Unknown Road")
        assert matched is False

    def test_regex_case_insensitive(self):
        """Test regex is case-insensitive."""
        rule = AlertRule(id="test", pattern=r"MAIN", match_type="regex")
        matched, _ = rule.matches("incident on main street")
        assert matched is True

    def test_regex_invalid_pattern(self):
        """Test that invalid regex raises error."""
        with pytest.raises(ValueError):
            AlertRule(id="test", pattern=r"[invalid(regex", match_type="regex")

    def test_speaker_filter_match(self):
        """Test speaker filtering matches."""
        rule = AlertRule(
            id="test",
            pattern="fire",
            speaker_filter=["Dispatcher"]
        )

        matched, _ = rule.matches("fire on Main St", speaker="Dispatcher")
        assert matched is True

    def test_speaker_filter_no_match(self):
        """Test speaker filtering rejects wrong speaker."""
        rule = AlertRule(
            id="test",
            pattern="fire",
            speaker_filter=["Dispatcher"]
        )

        matched, meta = rule.matches("fire on Main St", speaker="Unknown")
        assert matched is False
        assert meta["reason"] == "speaker_filtered"

    def test_speaker_filter_multiple(self):
        """Test speaker filter with multiple allowed speakers."""
        rule = AlertRule(
            id="test",
            pattern="fire",
            speaker_filter=["Dispatcher", "Chief"]
        )

        matched, _ = rule.matches("fire on Main St", speaker="Chief")
        assert matched is True

    def test_confidence_filter_meets_threshold(self):
        """Test confidence filtering with sufficient confidence."""
        rule = AlertRule(
            id="test",
            pattern="fire",
            confidence_min=-0.8
        )

        matched, _ = rule.matches("fire on Main St", confidence=-0.5)
        assert matched is True

    def test_confidence_filter_below_threshold(self):
        """Test confidence filtering rejects low confidence."""
        rule = AlertRule(
            id="test",
            pattern="fire",
            confidence_min=-0.8
        )

        matched, meta = rule.matches("fire on Main St", confidence=-0.95)
        assert matched is False
        assert meta["reason"] == "confidence_too_low"

    def test_duration_min_filter(self):
        """Test minimum duration filtering."""
        rule = AlertRule(
            id="test",
            pattern="fire",
            duration_min=1.0
        )

        matched, _ = rule.matches("fire on Main St", duration=2.0)
        assert matched is True

        matched, meta = rule.matches("fire on Main St", duration=0.5)
        assert matched is False
        assert meta["reason"] == "duration_too_short"

    def test_duration_max_filter(self):
        """Test maximum duration filtering."""
        rule = AlertRule(
            id="test",
            pattern="fire",
            duration_max=5.0
        )

        matched, _ = rule.matches("fire on Main St", duration=3.0)
        assert matched is True

        matched, meta = rule.matches("fire on Main St", duration=10.0)
        assert matched is False
        assert meta["reason"] == "duration_too_long"

    def test_combined_filters(self):
        """Test multiple filters working together."""
        rule = AlertRule(
            id="test",
            pattern="fire",
            speaker_filter=["Dispatcher"],
            confidence_min=-0.8,
            duration_min=1.0
        )

        # All filters pass
        matched, _ = rule.matches(
            "fire on Main St",
            speaker="Dispatcher",
            confidence=-0.5,
            duration=2.0
        )
        assert matched is True

        # Speaker fails
        matched, meta = rule.matches(
            "fire on Main St",
            speaker="Unknown",
            confidence=-0.5,
            duration=2.0
        )
        assert matched is False
        assert meta["reason"] == "speaker_filtered"

        # Confidence fails
        matched, meta = rule.matches(
            "fire on Main St",
            speaker="Dispatcher",
            confidence=-0.95,
            duration=2.0
        )
        assert matched is False
        assert meta["reason"] == "confidence_too_low"

    def test_disabled_rule(self):
        """Test that disabled rules never match."""
        rule = AlertRule(id="test", pattern="fire", enabled=False)
        matched, _ = rule.matches("fire on Main St")
        assert matched is False

    def test_regex_compilation_lazy(self):
        """Test that regex is compiled on demand."""
        rule = AlertRule(id="test", pattern=r"\bfire\b", match_type="regex")
        assert rule._compiled_regex is not None  # Compiled in __post_init__


class TestAlertRuleEngine:
    """Test AlertRuleEngine orchestration."""

    def test_load_rules_from_config(self):
        """Test loading rules from config dict."""
        config = {
            "alerts": {
                "rules": [
                    {
                        "id": "fire_rule",
                        "pattern": "fire",
                        "match_type": "substring"
                    }
                ]
            }
        }
        engine = AlertRuleEngine(config)
        assert len(engine.rules) == 1
        assert "fire_rule" in engine.rules

    def test_legacy_keywords_migration(self):
        """Test that legacy keywords are converted to rules."""
        config = {
            "alerts": {
                "keywords": ["fire", "emergency", "ambulance"],
                "rate_limit_seconds": 60
            }
        }
        engine = AlertRuleEngine(config)
        assert len(engine.rules) == 3

        # Check that rules are properly created
        for rule in engine.rules.values():
            assert rule.match_type == "substring"
            assert rule.dedup_cooldown_sec == 60
            assert "legacy" in rule.tags

    def test_mixed_rules_and_keywords(self):
        """Test that new rules take priority, keywords are fallback."""
        config = {
            "alerts": {
                "rules": [
                    {"id": "fire_rule", "pattern": "fire", "match_type": "word"}
                ]
            }
        }
        engine = AlertRuleEngine(config)
        assert len(engine.rules) == 1

    def test_check_rules_basic_match(self):
        """Test basic rule matching via engine."""
        config = {
            "alerts": {
                "rules": [
                    {"id": "fire", "pattern": "fire", "match_type": "substring"}
                ]
            }
        }
        engine = AlertRuleEngine(config)

        matched = engine.check_rules("structure fire on Main St", {"speaker": "Dispatcher"})
        assert len(matched) == 1
        assert matched[0]["rule_id"] == "fire"

    def test_check_rules_no_match(self):
        """Test engine returns empty list for no match."""
        config = {
            "alerts": {
                "rules": [
                    {"id": "fire", "pattern": "fire", "match_type": "substring"}
                ]
            }
        }
        engine = AlertRuleEngine(config)

        matched = engine.check_rules("ambulance call", {})
        assert len(matched) == 0

    def test_per_rule_deduplication(self):
        """Test that per-rule dedup prevents duplicate alerts."""
        config = {
            "alerts": {
                "rules": [
                    {"id": "fire", "pattern": "fire", "dedup_cooldown_sec": 60}
                ]
            }
        }
        engine = AlertRuleEngine(config)

        # First match should succeed
        matched = engine.check_rules("fire", {})
        assert len(matched) == 1

        # Second match immediately should be blocked
        matched = engine.check_rules("fire again", {})
        assert len(matched) == 0

    def test_per_rule_deduplication_timeout(self):
        """Test that dedup timeout expires."""
        config = {
            "alerts": {
                "rules": [
                    {"id": "fire", "pattern": "fire", "dedup_cooldown_sec": 1}
                ]
            }
        }
        engine = AlertRuleEngine(config)

        matched = engine.check_rules("fire", {})
        assert len(matched) == 1

        # Immediate second call blocked
        matched = engine.check_rules("fire again", {})
        assert len(matched) == 0

        # After timeout, should work
        time.sleep(1.1)
        matched = engine.check_rules("fire once more", {})
        assert len(matched) == 1

    def test_group_deduplication(self):
        """Test that grouped rules share cooldown."""
        config = {
            "alerts": {
                "rules": [
                    {"id": "fire1", "pattern": "fire", "dedup_group": "emergency", "dedup_cooldown_sec": 60},
                    {"id": "fire2", "pattern": "burning", "dedup_group": "emergency", "dedup_cooldown_sec": 60},
                    {"id": "fire3", "pattern": "blaze", "dedup_group": "emergency", "dedup_cooldown_sec": 60}
                ]
            }
        }
        engine = AlertRuleEngine(config)

        # First match (fire1)
        matched = engine.check_rules("structure fire", {})
        assert len(matched) == 1
        assert matched[0]["rule_id"] == "fire1"

        # Second match (fire2) immediately - should be blocked by group cooldown
        matched = engine.check_rules("house burning", {})
        assert len(matched) == 0

        # Third match (fire3) immediately - should also be blocked
        matched = engine.check_rules("raging blaze", {})
        assert len(matched) == 0

    def test_add_rule_runtime(self):
        """Test adding rules at runtime."""
        config = {"alerts": {}}
        engine = AlertRuleEngine(config)

        assert len(engine.rules) == 0

        rule = AlertRule(id="new_rule", pattern="ambulance")
        engine.add_rule(rule)

        assert len(engine.rules) == 1
        matched = engine.check_rules("ambulance call", {})
        assert len(matched) == 1

    def test_remove_rule_runtime(self):
        """Test removing rules at runtime."""
        config = {
            "alerts": {
                "rules": [{"id": "fire", "pattern": "fire"}]
            }
        }
        engine = AlertRuleEngine(config)

        assert len(engine.rules) == 1
        success = engine.remove_rule("fire")
        assert success is True
        assert len(engine.rules) == 0

    def test_remove_nonexistent_rule(self):
        """Test removing rule that doesn't exist."""
        config = {"alerts": {}}
        engine = AlertRuleEngine(config)

        success = engine.remove_rule("nonexistent")
        assert success is False

    def test_get_rule(self):
        """Test getting a rule by ID."""
        config = {
            "alerts": {
                "rules": [{"id": "fire", "pattern": "fire"}]
            }
        }
        engine = AlertRuleEngine(config)

        rule = engine.get_rule("fire")
        assert rule is not None
        assert rule.id == "fire"

        rule = engine.get_rule("nonexistent")
        assert rule is None

    def test_list_rules(self):
        """Test listing rules with status."""
        config = {
            "alerts": {
                "rules": [
                    {"id": "fire", "pattern": "fire", "tags": ["emergency"]},
                    {"id": "ambulance", "pattern": "ambulance"}
                ]
            }
        }
        engine = AlertRuleEngine(config)

        rules = engine.list_rules()
        assert len(rules) == 2

        fire_rule = next(r for r in rules if r["id"] == "fire")
        assert fire_rule["tags"] == ["emergency"]
        assert "cooldown_remaining" in fire_rule

    def test_test_rule_without_dedup(self):
        """Test rule testing without affecting dedup state."""
        config = {
            "alerts": {
                "rules": [{"id": "fire", "pattern": "fire", "dedup_cooldown_sec": 60}]
            }
        }
        engine = AlertRuleEngine(config)

        # Test rule shouldn't affect dedup
        matched, result = engine.test_rule("fire", "fire on Main St")
        assert matched is True
        assert result["matched"] is True

        # Now check_rules should still work (dedup not affected)
        matched = engine.check_rules("fire", {})
        assert len(matched) == 1

    def test_multiple_rules_multiple_matches(self):
        """Test multiple rules firing on same transcript."""
        config = {
            "alerts": {
                "rules": [
                    {"id": "fire", "pattern": "fire"},
                    {"id": "structure", "pattern": "structure"}
                ]
            }
        }
        engine = AlertRuleEngine(config)

        matched = engine.check_rules("structure fire", {})
        assert len(matched) == 2
        assert {r["rule_id"] for r in matched} == {"fire", "structure"}

    def test_context_metadata_in_results(self):
        """Test that matched rules include context metadata."""
        config = {
            "alerts": {
                "rules": [{"id": "fire", "pattern": "fire"}]
            }
        }
        engine = AlertRuleEngine(config)

        matched = engine.check_rules(
            "fire on Main St",
            {
                "speaker": "Dispatcher",
                "confidence": -0.5,
                "duration": 2.0
            }
        )

        assert len(matched) == 1
        result = matched[0]
        assert result["speaker"] == "Dispatcher"
        assert result["confidence"] == -0.5
        assert result["duration"] == 2.0
        assert result["match_type"] == "substring"

    def test_invalid_regex_in_config(self):
        """Test that invalid regex in config is logged but doesn't crash."""
        config = {
            "alerts": {
                "rules": [
                    {"id": "bad_regex", "pattern": "[invalid(regex", "match_type": "regex"}
                ]
            }
        }
        # Should not raise, but rule won't be loaded
        engine = AlertRuleEngine(config)
        # Bad rule should not be in engine
        assert "bad_regex" not in engine.rules
