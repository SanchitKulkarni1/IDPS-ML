"""
database.py — SQLite database manager for NIDS/IPS.
Tables: incidents, blocked_ips, prevention_rules
"""

import sqlite3
import os
import json
from datetime import datetime
from threading import Lock

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "nids.db")

_lock = Lock()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS incidents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            source_ip   TEXT,
            dest_ip     TEXT,
            protocol    TEXT,
            prediction  TEXT    NOT NULL,
            attack_type TEXT,
            confidence  REAL,
            risk_level  TEXT,
            features    TEXT,
            action_taken TEXT DEFAULT 'none'
        );

        CREATE TABLE IF NOT EXISTS blocked_ips (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ip_address  TEXT    UNIQUE NOT NULL,
            reason      TEXT,
            attack_type TEXT,
            blocked_at  TEXT    NOT NULL,
            blocked_by  TEXT    DEFAULT 'auto',
            expires_at  TEXT,
            active      INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS prevention_rules (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_name   TEXT    UNIQUE NOT NULL,
            enabled     INTEGER DEFAULT 1,
            config      TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_incidents_ts ON incidents(timestamp);
        CREATE INDEX IF NOT EXISTS idx_incidents_ip ON incidents(source_ip);
        CREATE INDEX IF NOT EXISTS idx_blocked_ip ON blocked_ips(ip_address);
    """)
    conn.commit()
    conn.close()


# ─── Incidents ───────────────────────────────────────────────────────────────

def log_incident(source_ip: str, dest_ip: str, protocol: str,
                 prediction: str, attack_type: str, confidence: float,
                 risk_level: str, features: dict, action_taken: str = "none") -> int:
    """Insert an incident and return its id."""
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            """INSERT INTO incidents
               (timestamp, source_ip, dest_ip, protocol, prediction, attack_type,
                confidence, risk_level, features, action_taken)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.utcnow().isoformat() + "Z",
             source_ip, dest_ip, protocol, prediction, attack_type,
             confidence, risk_level, json.dumps(features), action_taken)
        )
        conn.commit()
        row_id = cur.lastrowid
        conn.close()
        return row_id


def get_incidents(limit: int = 100, offset: int = 0,
                  source_ip: str = None, risk_level: str = None) -> list[dict]:
    """Query incidents with optional filters."""
    conn = _get_conn()
    query = "SELECT * FROM incidents WHERE 1=1"
    params = []
    if source_ip:
        query += " AND source_ip = ?"
        params.append(source_ip)
    if risk_level:
        query += " AND risk_level = ?"
        params.append(risk_level)
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_incident_count() -> int:
    conn = _get_conn()
    count = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
    conn.close()
    return count


# ─── Blocked IPs ─────────────────────────────────────────────────────────────

def block_ip(ip_address: str, reason: str = "", attack_type: str = "",
             blocked_by: str = "auto", expires_at: str = None) -> bool:
    """Block an IP. Returns True if newly blocked, False if already blocked."""
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                """INSERT INTO blocked_ips
                   (ip_address, reason, attack_type, blocked_at, blocked_by, expires_at, active)
                   VALUES (?, ?, ?, ?, ?, ?, 1)
                   ON CONFLICT(ip_address) DO UPDATE SET
                       reason=excluded.reason,
                       attack_type=excluded.attack_type,
                       blocked_at=excluded.blocked_at,
                       blocked_by=excluded.blocked_by,
                       expires_at=excluded.expires_at,
                       active=1""",
                (ip_address, reason, attack_type,
                 datetime.utcnow().isoformat() + "Z", blocked_by, expires_at)
            )
            conn.commit()
            conn.close()
            return True
        except Exception:
            conn.close()
            return False


def unblock_ip(ip_address: str) -> bool:
    """Unblock an IP by setting active=0."""
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            "UPDATE blocked_ips SET active=0 WHERE ip_address=? AND active=1",
            (ip_address,)
        )
        conn.commit()
        changed = cur.rowcount > 0
        conn.close()
        return changed


def get_blocked_ips(active_only: bool = True) -> list[dict]:
    conn = _get_conn()
    query = "SELECT * FROM blocked_ips"
    if active_only:
        query += " WHERE active=1"
    query += " ORDER BY blocked_at DESC"
    rows = conn.execute(query).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def is_ip_blocked(ip_address: str) -> bool:
    conn = _get_conn()
    row = conn.execute(
        "SELECT 1 FROM blocked_ips WHERE ip_address=? AND active=1",
        (ip_address,)
    ).fetchone()
    conn.close()
    return row is not None


def get_blocked_count() -> int:
    conn = _get_conn()
    count = conn.execute("SELECT COUNT(*) FROM blocked_ips WHERE active=1").fetchone()[0]
    conn.close()
    return count


def cleanup_expired_blocks() -> int:
    """Deactivate blocks past their expiry. Returns count of expired blocks."""
    now = datetime.utcnow().isoformat() + "Z"
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            "UPDATE blocked_ips SET active=0 WHERE active=1 AND expires_at IS NOT NULL AND expires_at <= ?",
            (now,)
        )
        conn.commit()
        count = cur.rowcount
        conn.close()
        return count


def get_recent_incident_count_for_ip(ip_address: str, window_seconds: int = 300) -> int:
    """Count incidents from an IP within the last `window_seconds`."""
    from datetime import timedelta
    cutoff = (datetime.utcnow() - timedelta(seconds=window_seconds)).isoformat() + "Z"
    conn = _get_conn()
    count = conn.execute(
        "SELECT COUNT(*) FROM incidents WHERE source_ip=? AND timestamp >= ?",
        (ip_address, cutoff)
    ).fetchone()[0]
    conn.close()
    return count



def get_incident_summary() -> dict:
    """Return aggregate stats from all incidents for dashboard/analytics pages."""
    conn = _get_conn()

    total = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]

    # Normal vs attack counts
    normal = conn.execute(
        "SELECT COUNT(*) FROM incidents WHERE LOWER(prediction) IN ('normal','benign')"
    ).fetchone()[0]
    attack = total - normal

    # Attack type breakdown
    attack_rows = conn.execute(
        """SELECT attack_type, COUNT(*) as cnt FROM incidents
           WHERE LOWER(prediction) NOT IN ('normal','benign')
           GROUP BY attack_type ORDER BY cnt DESC"""
    ).fetchall()
    attack_types = {r["attack_type"]: r["cnt"] for r in attack_rows}

    # Prediction distribution (all types)
    pred_rows = conn.execute(
        "SELECT prediction, COUNT(*) as cnt FROM incidents GROUP BY prediction ORDER BY cnt DESC"
    ).fetchall()
    prediction_counts = {r["prediction"]: r["cnt"] for r in pred_rows}

    # Risk level distribution
    risk_rows = conn.execute(
        "SELECT risk_level, COUNT(*) as cnt FROM incidents GROUP BY risk_level ORDER BY cnt DESC"
    ).fetchall()
    risk_levels = {r["risk_level"]: r["cnt"] for r in risk_rows}

    # Action taken distribution
    action_rows = conn.execute(
        "SELECT action_taken, COUNT(*) as cnt FROM incidents GROUP BY action_taken ORDER BY cnt DESC"
    ).fetchall()
    actions = {r["action_taken"]: r["cnt"] for r in action_rows}

    # Top source IPs
    top_ips = conn.execute(
        """SELECT source_ip, COUNT(*) as cnt,
                  SUM(CASE WHEN LOWER(prediction) NOT IN ('normal','benign') THEN 1 ELSE 0 END) as attacks
           FROM incidents GROUP BY source_ip ORDER BY cnt DESC LIMIT 10"""
    ).fetchall()
    top_sources = [{"ip": r["source_ip"], "total": r["cnt"], "attacks": r["attacks"]} for r in top_ips]

    # Hourly timeline (last 24h bucketed by hour)
    timeline_rows = conn.execute(
        """SELECT SUBSTR(timestamp, 1, 13) as hour,
                  COUNT(*) as total,
                  SUM(CASE WHEN LOWER(prediction) NOT IN ('normal','benign') THEN 1 ELSE 0 END) as attacks,
                  SUM(CASE WHEN LOWER(prediction) IN ('normal','benign') THEN 1 ELSE 0 END) as normal
           FROM incidents GROUP BY hour ORDER BY hour DESC LIMIT 24"""
    ).fetchall()
    timeline = [{"hour": r["hour"], "total": r["total"],
                 "attacks": r["attacks"], "normal": r["normal"]} for r in timeline_rows]
    timeline.reverse()  # chronological order

    # Recent incidents (last 20)
    recent = conn.execute(
        "SELECT * FROM incidents ORDER BY id DESC LIMIT 20"
    ).fetchall()

    conn.close()
    return {
        "total_packets": total,
        "normal_count": normal,
        "attack_count": attack,
        "attack_types": attack_types,
        "prediction_counts": prediction_counts,
        "risk_levels": risk_levels,
        "actions": actions,
        "top_sources": top_sources,
        "timeline": timeline,
        "recent": [dict(r) for r in recent],
        "blocked_count": get_blocked_count(),
    }


# ─── Prevention Rules ────────────────────────────────────────────────────────

DEFAULT_RULES = [
    {
        "rule_name": "high_confidence_attack",
        "enabled": True,
        "config": {
            "description": "Block IPs with high-confidence attack predictions",
            "min_confidence": 0.90,
            "block_duration_hours": 1,
            "exclude_labels": ["Normal", "normal", "benign"]
        }
    },
    {
        "rule_name": "critical_attack_type",
        "enabled": True,
        "config": {
            "description": "Block IPs launching critical attack types",
            "attack_types": ["ddos", "u2r", "dos", "neptune", "smurf"],
            "block_duration_hours": 24
        }
    },
    {
        "rule_name": "repeated_offender",
        "enabled": True,
        "config": {
            "description": "Block IPs flagged multiple times in a short window",
            "max_incidents": 3,
            "window_seconds": 300,
            "block_duration_hours": 24
        }
    },
    {
        "rule_name": "rate_limit",
        "enabled": True,
        "config": {
            "description": "Block IPs exceeding connection rate limits",
            "max_connections": 100,
            "window_seconds": 60,
            "block_duration_hours": 0.5
        }
    }
]


def seed_default_rules():
    """Insert default rules if the table is empty."""
    conn = _get_conn()
    count = conn.execute("SELECT COUNT(*) FROM prevention_rules").fetchone()[0]
    if count == 0:
        for rule in DEFAULT_RULES:
            conn.execute(
                "INSERT INTO prevention_rules (rule_name, enabled, config) VALUES (?, ?, ?)",
                (rule["rule_name"], 1 if rule["enabled"] else 0, json.dumps(rule["config"]))
            )
        conn.commit()
    conn.close()


def get_rules() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM prevention_rules").fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["config"] = json.loads(d["config"]) if isinstance(d["config"], str) else d["config"]
        d["enabled"] = bool(d["enabled"])
        result.append(d)
    return result


def update_rule(rule_name: str, enabled: bool = None, config: dict = None) -> bool:
    """Update a rule's enabled state and/or config. Returns True if found."""
    with _lock:
        conn = _get_conn()
        existing = conn.execute(
            "SELECT * FROM prevention_rules WHERE rule_name=?", (rule_name,)
        ).fetchone()
        if not existing:
            conn.close()
            return False
        if enabled is not None:
            conn.execute(
                "UPDATE prevention_rules SET enabled=? WHERE rule_name=?",
                (1 if enabled else 0, rule_name)
            )
        if config is not None:
            conn.execute(
                "UPDATE prevention_rules SET config=? WHERE rule_name=?",
                (json.dumps(config), rule_name)
            )
        conn.commit()
        conn.close()
        return True
