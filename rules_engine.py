"""
rules_engine.py — Auto-response rules engine for the NIDS/IPS.
Evaluates predictions and triggers prevention actions based on configurable rules.
Rules are persisted in SQLite via database.py.
"""

import logging
from dataclasses import dataclass

import database as db
import blocker

logger = logging.getLogger("rules_engine")


@dataclass
class ActionResult:
    """Result of evaluating a prediction against the rules."""
    action_taken: str           # 'none', 'blocked', 'rate_limited', 'alerted'
    rule_triggered: str | None  # Name of the rule that fired, if any
    details: str                # Human-readable description
    block_duration_hours: float | None  # Duration of block, if applicable


def _get_rule_config(rules: list[dict], rule_name: str) -> dict | None:
    """Get config for a specific rule, if enabled."""
    for r in rules:
        if r["rule_name"] == rule_name and r.get("enabled", True):
            return r.get("config", {})
    return None


def evaluate(prediction: str, confidence: float, source_ip: str,
             attack_type: str = None, features: dict = None) -> ActionResult:
    """
    Evaluate a prediction result against all enabled rules.
    Returns an ActionResult describing what action was taken (if any).
    
    Args:
        prediction: The ML model prediction label (e.g., "Normal", "ddos", "probe")
        confidence: Confidence score (0.0 - 1.0)
        source_ip: Source IP address from the traffic
        attack_type: Specific attack type if available
        features: Raw feature dict from the request
    """
    # Skip evaluation for missing/local IPs
    if not source_ip or source_ip in ("—", "-", "", "unknown", "127.0.0.1", "localhost"):
        return ActionResult("none", None, "Skipped: no valid source IP", None)

    # Already blocked? No need to re-evaluate
    if blocker.is_blocked(source_ip):
        return ActionResult("already_blocked", None,
                            f"IP {source_ip} is already blocked", None)

    rules = db.get_rules()
    pred_lower = (prediction or "").strip().lower()
    attack_lower = (attack_type or pred_lower).strip().lower()

    # ── Rule 1: High-confidence attack ────────────────────────────────────
    cfg = _get_rule_config(rules, "high_confidence_attack")
    if cfg:
        min_conf = cfg.get("min_confidence", 0.90)
        exclude = [l.lower() for l in cfg.get("exclude_labels", ["normal", "benign"])]
        dur = cfg.get("block_duration_hours", 1)

        if confidence >= min_conf and pred_lower not in exclude:
            result = blocker.block_ip(
                source_ip,
                reason=f"High-confidence attack ({confidence:.2%}): {prediction}",
                attack_type=attack_lower,
                blocked_by="auto",
                duration_hours=dur
            )
            logger.info("Rule high_confidence_attack fired for %s (conf=%.2f, pred=%s)",
                        source_ip, confidence, prediction)
            return ActionResult(
                "blocked", "high_confidence_attack",
                f"Blocked {source_ip}: {prediction} with {confidence:.0%} confidence",
                dur
            )

    # ── Rule 2: Critical attack type ──────────────────────────────────────
    cfg = _get_rule_config(rules, "critical_attack_type")
    if cfg:
        critical_types = [t.lower() for t in cfg.get("attack_types", [])]
        dur = cfg.get("block_duration_hours", 24)

        if any(ct in attack_lower for ct in critical_types):
            result = blocker.block_ip(
                source_ip,
                reason=f"Critical attack type: {attack_type or prediction}",
                attack_type=attack_lower,
                blocked_by="auto",
                duration_hours=dur
            )
            logger.info("Rule critical_attack_type fired for %s (type=%s)",
                        source_ip, attack_lower)
            return ActionResult(
                "blocked", "critical_attack_type",
                f"Blocked {source_ip}: critical attack type '{attack_lower}'",
                dur
            )

    # ── Rule 3: Repeated offender ─────────────────────────────────────────
    cfg = _get_rule_config(rules, "repeated_offender")
    if cfg:
        max_incidents = cfg.get("max_incidents", 3)
        window = cfg.get("window_seconds", 300)
        dur = cfg.get("block_duration_hours", 24)

        # Only check for non-normal predictions
        if pred_lower not in ("normal", "benign"):
            recent_count = db.get_recent_incident_count_for_ip(source_ip, window)
            if recent_count >= max_incidents:
                result = blocker.block_ip(
                    source_ip,
                    reason=f"Repeated offender: {recent_count} incidents in {window}s",
                    attack_type=attack_lower,
                    blocked_by="auto",
                    duration_hours=dur
                )
                logger.info("Rule repeated_offender fired for %s (%d incidents in %ds)",
                            source_ip, recent_count, window)
                return ActionResult(
                    "blocked", "repeated_offender",
                    f"Blocked {source_ip}: {recent_count} incidents in {window}s window",
                    dur
                )

    # ── Rule 4: Rate limit ────────────────────────────────────────────────
    cfg = _get_rule_config(rules, "rate_limit")
    if cfg:
        max_conn = cfg.get("max_connections", 100)
        window = cfg.get("window_seconds", 60)
        dur = cfg.get("block_duration_hours", 0.5)

        recent_count = db.get_recent_incident_count_for_ip(source_ip, window)
        if recent_count >= max_conn:
            result = blocker.block_ip(
                source_ip,
                reason=f"Rate limit exceeded: {recent_count} connections in {window}s",
                attack_type="rate_limit",
                blocked_by="auto",
                duration_hours=dur
            )
            logger.info("Rule rate_limit fired for %s (%d conns in %ds)",
                        source_ip, recent_count, window)
            return ActionResult(
                "blocked", "rate_limit",
                f"Blocked {source_ip}: {recent_count} connections in {window}s",
                dur
            )

    # No rule triggered
    return ActionResult("none", None, "No prevention rule triggered", None)
