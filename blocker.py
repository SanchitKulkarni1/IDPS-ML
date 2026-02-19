"""
blocker.py — IP blocking engine for the NIDS/IPS.
Supports simulation mode (default) and live mode (iptables).
Includes a background cleanup scheduler for expired blocks.
"""

import os
import subprocess
import threading
import logging
from datetime import datetime, timedelta

import database as db

logger = logging.getLogger("blocker")

# Mode: False = simulation (default), True = live iptables
LIVE_MODE = os.environ.get("IPS_LIVE_MODE", "0") == "1"

IPTABLES_COMMENT = "NIDS_BLOCK"
CLEANUP_INTERVAL_SECONDS = 60

_cleanup_timer: threading.Timer | None = None


def set_live_mode(enabled: bool):
    """Toggle between simulation and live mode at runtime."""
    global LIVE_MODE
    LIVE_MODE = enabled
    logger.info("Blocker mode set to %s", "LIVE" if enabled else "SIMULATION")


def is_live_mode() -> bool:
    return LIVE_MODE


# ─── iptables helpers (live mode only) ───────────────────────────────────────

def _iptables_block(ip: str) -> bool:
    """Add an iptables DROP rule. Returns True on success."""
    try:
        result = subprocess.run(
            ["iptables", "-A", "INPUT", "-s", ip, "-j", "DROP",
             "-m", "comment", "--comment", IPTABLES_COMMENT],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            logger.info("iptables: blocked %s", ip)
            return True
        logger.error("iptables block failed for %s: %s", ip, result.stderr)
        return False
    except Exception as e:
        logger.error("iptables block exception for %s: %s", ip, e)
        return False


def _iptables_unblock(ip: str) -> bool:
    """Remove the iptables DROP rule. Returns True on success."""
    try:
        result = subprocess.run(
            ["iptables", "-D", "INPUT", "-s", ip, "-j", "DROP",
             "-m", "comment", "--comment", IPTABLES_COMMENT],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            logger.info("iptables: unblocked %s", ip)
            return True
        logger.error("iptables unblock failed for %s: %s", ip, result.stderr)
        return False
    except Exception as e:
        logger.error("iptables unblock exception for %s: %s", ip, e)
        return False


# ─── Public API ──────────────────────────────────────────────────────────────

def block_ip(ip: str, reason: str = "", attack_type: str = "",
             blocked_by: str = "auto", duration_hours: float = None) -> dict:
    """
    Block an IP address.
    Returns a dict with status info.
    """
    expires_at = None
    if duration_hours is not None and duration_hours > 0:
        expires_at = (datetime.utcnow() + timedelta(hours=duration_hours)).isoformat() + "Z"

    # Write to database
    db.block_ip(ip, reason, attack_type, blocked_by, expires_at)

    # Apply iptables rule if in live mode
    iptables_applied = False
    if LIVE_MODE:
        iptables_applied = _iptables_block(ip)

    mode_label = "LIVE" if LIVE_MODE else "SIMULATION"
    logger.info("[%s] Blocked IP %s — reason: %s, duration: %s hours, by: %s",
                mode_label, ip, reason, duration_hours or "permanent", blocked_by)

    return {
        "ip": ip,
        "status": "blocked",
        "mode": mode_label.lower(),
        "iptables_applied": iptables_applied,
        "reason": reason,
        "attack_type": attack_type,
        "blocked_by": blocked_by,
        "expires_at": expires_at
    }


def unblock_ip(ip: str) -> dict:
    """Unblock an IP address."""
    changed = db.unblock_ip(ip)

    iptables_removed = False
    if LIVE_MODE:
        iptables_removed = _iptables_unblock(ip)

    mode_label = "LIVE" if LIVE_MODE else "SIMULATION"
    logger.info("[%s] Unblocked IP %s (db_changed=%s)", mode_label, ip, changed)

    return {
        "ip": ip,
        "status": "unblocked" if changed else "not_found",
        "mode": mode_label.lower(),
        "iptables_removed": iptables_removed
    }


def get_blocked_ips() -> list[dict]:
    """Get all currently active blocked IPs."""
    return db.get_blocked_ips(active_only=True)


def is_blocked(ip: str) -> bool:
    """Check if an IP is currently blocked."""
    return db.is_ip_blocked(ip)


def get_blocked_count() -> int:
    """Get count of currently blocked IPs."""
    return db.get_blocked_count()


# ─── Cleanup scheduler ──────────────────────────────────────────────────────

def cleanup_expired():
    """Remove expired blocks from DB (and iptables in live mode)."""
    # First, find blocks about to expire so we can remove iptables rules
    if LIVE_MODE:
        blocked = db.get_blocked_ips(active_only=True)
        now = datetime.utcnow().isoformat() + "Z"
        for b in blocked:
            if b.get("expires_at") and b["expires_at"] <= now:
                _iptables_unblock(b["ip_address"])

    count = db.cleanup_expired_blocks()
    if count > 0:
        logger.info("Cleaned up %d expired blocks", count)
    return count


def _cleanup_loop():
    """Background timer that calls cleanup_expired periodically."""
    global _cleanup_timer
    try:
        cleanup_expired()
    except Exception as e:
        logger.error("Cleanup error: %s", e)
    # Schedule next run
    _cleanup_timer = threading.Timer(CLEANUP_INTERVAL_SECONDS, _cleanup_loop)
    _cleanup_timer.daemon = True
    _cleanup_timer.start()


def start_cleanup_scheduler():
    """Start the background cleanup scheduler. Call once at server startup."""
    global _cleanup_timer
    if _cleanup_timer is not None:
        return  # Already running
    logger.info("Starting block cleanup scheduler (every %ds)", CLEANUP_INTERVAL_SECONDS)
    _cleanup_timer = threading.Timer(CLEANUP_INTERVAL_SECONDS, _cleanup_loop)
    _cleanup_timer.daemon = True
    _cleanup_timer.start()


def stop_cleanup_scheduler():
    """Stop the background cleanup scheduler."""
    global _cleanup_timer
    if _cleanup_timer is not None:
        _cleanup_timer.cancel()
        _cleanup_timer = None
        logger.info("Stopped block cleanup scheduler")
