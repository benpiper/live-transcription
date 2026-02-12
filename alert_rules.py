"""
Alert Rule Engine - Advanced pattern matching and context-aware alert triggering.

Supports multiple match types (substring, word boundary, regex), context filtering
(speaker, confidence, duration), and intelligent deduplication (per-rule and grouped).
"""

import re
import time
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("AlertRuleEngine")


@dataclass
class AlertRule:
    """Single alert rule with matching logic and context filters."""

    id: str                                    # Unique rule identifier
    pattern: str | List[str]                  # Regex pattern or list of keywords
    match_type: str = "substring"             # "substring", "word", or "regex"
    speaker_filter: Optional[List[str]] = None # Only match from these speakers
    confidence_min: Optional[float] = None     # Minimum confidence threshold (-1.0 to 0.0)
    duration_min: Optional[float] = None       # Minimum duration in seconds
    duration_max: Optional[float] = None       # Maximum duration in seconds
    dedup_group: Optional[str] = None         # Group ID for shared cooldown
    dedup_cooldown_sec: float = 300           # Cooldown period
    tags: List[str] = field(default_factory=list)  # Rule tags/categories
    description: str = ""                     # Human-readable description
    enabled: bool = True                      # Whether rule is active

    # Compiled regex (lazy-initialized)
    _compiled_regex: Optional[re.Pattern] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Validate and compile regex if needed."""
        if not self.enabled:
            return

        if self.match_type == "regex" and isinstance(self.pattern, str):
            try:
                self._compiled_regex = re.compile(self.pattern, re.IGNORECASE)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern in rule '{self.id}': {e}")

    def get_compiled_regex(self) -> Optional[re.Pattern]:
        """Get compiled regex, compiling if needed."""
        if self._compiled_regex is None and self.match_type == "regex" and isinstance(self.pattern, str):
            try:
                self._compiled_regex = re.compile(self.pattern, re.IGNORECASE)
            except re.error:
                return None
        return self._compiled_regex

    def matches(
        self,
        text: str,
        speaker: Optional[str] = None,
        confidence: Optional[float] = None,
        duration: Optional[float] = None
    ) -> Tuple[bool, Dict]:
        """
        Check if text matches this rule with all context filters applied.

        Returns:
            (matched: bool, metadata: dict with match details)
        """
        if not self.enabled:
            return False, {}

        # Check speaker filter
        if self.speaker_filter and speaker and speaker not in self.speaker_filter:
            return False, {"reason": "speaker_filtered"}

        # Check confidence filter
        if self.confidence_min is not None and confidence is not None:
            if confidence < self.confidence_min:
                return False, {"reason": "confidence_too_low"}

        # Check duration filters
        if duration is not None:
            if self.duration_min is not None and duration < self.duration_min:
                return False, {"reason": "duration_too_short"}
            if self.duration_max is not None and duration > self.duration_max:
                return False, {"reason": "duration_too_long"}

        # Check text pattern
        matched_text = None
        if self.match_type == "substring":
            if any(kw.lower() in text.lower() for kw in ([self.pattern] if isinstance(self.pattern, str) else self.pattern)):
                matched_text = text

        elif self.match_type == "word":
            # Word boundary matching - prevent "fire" matching "firearm"
            keywords = [self.pattern] if isinstance(self.pattern, str) else self.pattern
            for kw in keywords:
                pattern = r"\b" + re.escape(kw) + r"\b"
                if re.search(pattern, text, re.IGNORECASE):
                    matched_text = text
                    break

        elif self.match_type == "regex":
            regex = self.get_compiled_regex()
            if regex and regex.search(text):
                matched_text = text

        if not matched_text:
            return False, {"reason": "pattern_no_match"}

        return True, {
            "matched_text": matched_text,
            "match_type": self.match_type,
            "speaker": speaker,
            "confidence": confidence,
            "duration": duration
        }


class AlertRuleEngine:
    """Orchestrates alert rules with deduplication tracking."""

    def __init__(self, config: Dict):
        """Initialize engine from config dict."""
        self.config = config
        self.rules: Dict[str, AlertRule] = {}
        self.last_rule_alert: Dict[str, float] = {}      # Per-rule dedup
        self.last_group_alert: Dict[str, float] = {}     # Per-group dedup

        # Load rules from config
        self._load_rules_from_config()

    def _load_rules_from_config(self):
        """Load AlertRule objects from config."""
        alerts_config = self.config.get("alerts", {})

        # Load new-style rules if present
        rules_list = alerts_config.get("rules", [])
        for rule_dict in rules_list:
            try:
                rule = self._rule_from_dict(rule_dict)
                if rule:
                    self.rules[rule.id] = rule
            except Exception as e:
                logger.error(f"Failed to load rule: {e}")

        # Legacy fallback: convert old-style keywords to rules
        if not self.rules:
            keywords = alerts_config.get("keywords", [])
            if keywords:
                logger.info(f"Converting {len(keywords)} legacy keywords to rules")
                for i, kw in enumerate(keywords):
                    rule = AlertRule(
                        id=f"legacy_kw_{i}",
                        pattern=kw,
                        match_type="substring",
                        dedup_cooldown_sec=alerts_config.get("rate_limit_seconds", 300),
                        description=f"Legacy keyword: {kw}",
                        tags=["legacy"]
                    )
                    self.rules[rule.id] = rule

        logger.info(f"Loaded {len(self.rules)} alert rules")

    def _rule_from_dict(self, rule_dict: Dict) -> Optional[AlertRule]:
        """Convert dict to AlertRule with validation."""
        try:
            # Extract fields with defaults
            rule_id = rule_dict.get("id")
            if not rule_id:
                logger.error("Rule missing 'id' field")
                return None

            pattern = rule_dict.get("pattern")
            if not pattern:
                logger.error(f"Rule {rule_id} missing 'pattern'")
                return None

            return AlertRule(
                id=rule_id,
                pattern=pattern,
                match_type=rule_dict.get("match_type", "substring"),
                speaker_filter=rule_dict.get("speaker_filter"),
                confidence_min=rule_dict.get("confidence_min"),
                duration_min=rule_dict.get("duration_min"),
                duration_max=rule_dict.get("duration_max"),
                dedup_group=rule_dict.get("dedup_group"),
                dedup_cooldown_sec=rule_dict.get("dedup_cooldown_sec", 300),
                tags=rule_dict.get("tags", []),
                description=rule_dict.get("description", ""),
                enabled=rule_dict.get("enabled", True)
            )
        except Exception as e:
            logger.error(f"Error creating rule from dict: {e}")
            return None

    def check_rules(
        self,
        text: str,
        transcript_data: Dict
    ) -> List[Dict]:
        """
        Check transcript against all rules, respecting deduplication.

        Returns:
            List of matched rules with metadata
        """
        now = time.time()
        matched_rules = []

        speaker = transcript_data.get("speaker")
        confidence = transcript_data.get("confidence")
        duration = transcript_data.get("duration")

        for rule_id, rule in self.rules.items():
            # Check if rule matches content
            matched, metadata = rule.matches(text, speaker, confidence, duration)
            if not matched:
                continue

            # Check per-rule deduplication
            last_alert_time = self.last_rule_alert.get(rule_id, 0)
            if now - last_alert_time < rule.dedup_cooldown_sec:
                logger.debug(f"Rate limiting rule '{rule_id}'")
                continue

            # Check group deduplication if applicable
            if rule.dedup_group:
                last_group_time = self.last_group_alert.get(rule.dedup_group, 0)
                if now - last_group_time < rule.dedup_cooldown_sec:
                    logger.debug(f"Rate limiting rule '{rule_id}' (group: {rule.dedup_group})")
                    continue

            # Record this match for deduplication
            self.last_rule_alert[rule_id] = now
            if rule.dedup_group:
                self.last_group_alert[rule.dedup_group] = now

            # Build result dict
            result = {
                "rule_id": rule_id,
                "description": rule.description or rule_id,
                "pattern": rule.pattern if isinstance(rule.pattern, str) else rule.pattern[0],
                "match_type": rule.match_type,
                "tags": rule.tags,
                "metadata": metadata,
                "confidence": confidence,
                "speaker": speaker,
                "duration": duration
            }
            matched_rules.append(result)

        return matched_rules

    def add_rule(self, rule: AlertRule):
        """Add or update a rule at runtime."""
        if not rule.enabled:
            logger.warning(f"Adding disabled rule '{rule.id}'")
        self.rules[rule.id] = rule
        logger.info(f"Added rule '{rule.id}'")

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule at runtime."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            # Clean up dedup tracking
            if rule_id in self.last_rule_alert:
                del self.last_rule_alert[rule_id]
            logger.info(f"Removed rule '{rule_id}'")
            return True
        return False

    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get a rule by ID."""
        return self.rules.get(rule_id)

    def list_rules(self) -> List[Dict]:
        """List all rules with status."""
        return [
            {
                "id": rule.id,
                "description": rule.description or rule.id,
                "match_type": rule.match_type,
                "enabled": rule.enabled,
                "tags": rule.tags,
                "dedup_cooldown_sec": rule.dedup_cooldown_sec,
                "dedup_group": rule.dedup_group,
                "cooldown_remaining": max(0, rule.dedup_cooldown_sec - (time.time() - self.last_rule_alert.get(rule.id, 0)))
            }
            for rule in self.rules.values()
        ]

    def test_rule(self, rule_id: str, text: str, speaker: Optional[str] = None,
                  confidence: Optional[float] = None, duration: Optional[float] = None) -> Tuple[bool, Dict]:
        """Test a single rule without affecting dedup state."""
        rule = self.get_rule(rule_id)
        if not rule:
            return False, {"error": f"Rule '{rule_id}' not found"}

        matched, metadata = rule.matches(text, speaker, confidence, duration)
        return matched, {
            "matched": matched,
            "metadata": metadata,
            "rule": {
                "id": rule.id,
                "description": rule.description,
                "match_type": rule.match_type
            }
        }
