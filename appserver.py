# app.py — load model_pipeline.pkl and serve endpoints
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import io
import json
import time
import random
import logging
from datetime import datetime, timedelta
from sklearn.inspection import permutation_importance

import database as db
import blocker
import rules_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MODELS_DIR = "models"
PIPE_PATH = os.path.join(MODELS_DIR, "model_pipeline.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")
ALLOWED_EXT = {"csv"}

MODEL_PIPE = None
FEATURE_NAMES = None
X_VAL = None
Y_VAL = None  # optional, for permutation importance

def load_artifacts():
    global MODEL_PIPE, FEATURE_NAMES
    # Try multiple model file candidates in order of preference
    model_candidates = [
        PIPE_PATH,                                        # model_pipeline.pkl
        os.path.join(MODELS_DIR, "best_multi_model.pkl"),
        os.path.join(MODELS_DIR, "best_binary_model.pkl"),
    ]
    for path in model_candidates:
        if os.path.exists(path):
            try:
                MODEL_PIPE = joblib.load(path)
                logger.info("Loaded model: %s", path)
                break
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)
    if MODEL_PIPE is None:
        logger.warning("No model pipeline found in %s", MODELS_DIR)

    if os.path.exists(FEATURE_NAMES_PATH):
        try:
            FEATURE_NAMES = joblib.load(FEATURE_NAMES_PATH)
            print("Loaded feature names, count =", len(FEATURE_NAMES))
        except Exception as e:
            print("Failed load feature_names:", e)

# helper
def allowed_file(fname):
    return '.' in fname and fname.rsplit('.',1)[1].lower() in ALLOWED_EXT

# Metadata keys that are NOT model features — strip before prediction
_META_KEYS = {"src_ip", "source_ip", "dst_ip", "dest_ip", "protocol",
              "timestamp", "label", "attack_type"}

# Expected feature names (loaded from X_val at startup)
EXPECTED_FEATURES: list[str] | None = None

def _load_expected_features():
    """Load feature names from validation data so we can align input."""
    global EXPECTED_FEATURES
    for cand in ["X_val_multi_selected.pkl", "X_val_binary_selected.pkl"]:
        p = os.path.join(MODELS_DIR, cand)
        if os.path.exists(p):
            try:
                xv = joblib.load(p)
                if isinstance(xv, pd.DataFrame):
                    EXPECTED_FEATURES = list(xv.columns)
                    logger.info("Loaded %d expected features from %s", len(EXPECTED_FEATURES), cand)
                    return
            except Exception:
                pass

# Convert incoming JSON dict to DataFrame suitable for the model
def prepare_input_df(data: dict) -> pd.DataFrame:
    # Strip metadata keys — they are used by the prevention system, not the model
    features = {k: v for k, v in data.items() if k not in _META_KEYS}

    if EXPECTED_FEATURES:
        # Build a row aligned with the model's expected features
        aligned = {}
        for feat in EXPECTED_FEATURES:
            if feat in features:
                aligned[feat] = features[feat]
            else:
                # Default: 0 for numeric, False for boolean/one-hot columns
                aligned[feat] = 0
        return pd.DataFrame([aligned])

    return pd.DataFrame([features])

# Endpoints
@app.route("/api/predict", methods=["POST"])
def predict():
    if MODEL_PIPE is None:
        return jsonify({"error": "model pipeline not loaded. put model_pipeline.pkl in models/"}), 500
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "invalid json body"}), 400
    try:
        df = prepare_input_df(payload)
        # Pipeline handles preprocessing internally. Call predict / predict_proba on pipeline.
        if hasattr(MODEL_PIPE, "predict_proba"):
            probs = MODEL_PIPE.predict_proba(df)[0]
            pred_idx = int(np.argmax(probs))
            classes = getattr(MODEL_PIPE, "classes_", None)
            # If pipeline wraps an estimator, classes_ usually lives on the final estimator:
            if classes is None and hasattr(MODEL_PIPE, "named_steps"):
                est = MODEL_PIPE.named_steps.get("estimator") or list(MODEL_PIPE.named_steps.values())[-1]
                classes = getattr(est, "classes_", None)
            pred = str(classes[pred_idx]) if classes is not None else str(MODEL_PIPE.predict(df)[0])
            conf = float(np.max(probs))
        else:
            pred = str(MODEL_PIPE.predict(df)[0])
            conf = 1.0

        # ── Prevention: evaluate rules and log incident ──────────────
        source_ip = payload.get("src_ip") or payload.get("source_ip") or "unknown"
        dest_ip = payload.get("dst_ip") or payload.get("dest_ip") or ""
        protocol = payload.get("protocol") or ""

        action = rules_engine.evaluate(
            prediction=pred, confidence=conf,
            source_ip=source_ip, attack_type=pred,
            features=payload
        )

        db.log_incident(
            source_ip=source_ip, dest_ip=dest_ip, protocol=protocol,
            prediction=pred, attack_type=pred, confidence=conf,
            risk_level=_compute_risk(pred, conf),
            features=payload, action_taken=action.action_taken
        )

        result = {
            "prediction": pred,
            "confidence": round(conf, 4),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "action_taken": action.action_taken,
            "rule_triggered": action.rule_triggered,
            "prevention_details": action.details,
            "is_blocked": blocker.is_blocked(source_ip)
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"prediction failed: {str(e)}"}), 500



# ── SSE: stream live predictions from X_val ──────────────────────────
@app.route("/api/stream", methods=["GET"])
def stream_predictions():
    """Server-Sent Events endpoint that continuously streams predictions.
    Samples random rows from X_val, predicts, runs prevention rules,
    and pushes each result as an SSE event.
    """
    def _generate():
        global X_VAL
        # Load X_val lazily if not yet loaded
        if X_VAL is None:
            for cand in ["X_val_multi_selected.pkl", "X_val_binary_selected.pkl", "X_val.joblib"]:
                p = os.path.join(MODELS_DIR, cand)
                if os.path.exists(p):
                    try:
                        X_VAL = joblib.load(p)
                        break
                    except Exception:
                        pass
        if X_VAL is None or MODEL_PIPE is None:
            yield f"data: {json.dumps({'error': 'model or validation data not loaded'})}\n\n"
            return

        idx = 0
        n = len(X_VAL)
        while True:
            try:
                # Pick a random row to simulate live traffic
                row_idx = random.randint(0, n - 1)
                sample = X_VAL.iloc[[row_idx]]
                row_dict = X_VAL.iloc[row_idx].to_dict()

                # Generate a realistic random IP for simulation
                src_ip = f"{random.randint(10,192)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
                dst_ip = f"192.168.{random.randint(1,10)}.{random.randint(1,254)}"

                # Predict
                if hasattr(MODEL_PIPE, "predict_proba"):
                    probs = MODEL_PIPE.predict_proba(sample)[0]
                    pred_idx = int(np.argmax(probs))
                    classes = getattr(MODEL_PIPE, "classes_", None)
                    pred = str(classes[pred_idx]) if classes is not None else str(MODEL_PIPE.predict(sample)[0])
                    conf = float(np.max(probs))
                else:
                    pred = str(MODEL_PIPE.predict(sample)[0])
                    conf = 1.0

                risk = _compute_risk(pred, conf)

                # Run prevention rules
                action = rules_engine.evaluate(
                    prediction=pred, confidence=conf,
                    source_ip=src_ip, attack_type=pred,
                    features=row_dict
                )

                # Log incident
                db.log_incident(
                    source_ip=src_ip, dest_ip=dst_ip, protocol="tcp",
                    prediction=pred, attack_type=pred, confidence=conf,
                    risk_level=risk, features=row_dict,
                    action_taken=action.action_taken
                )

                # Determine flag icon
                is_attack = pred.lower() not in ("normal", "benign")
                flag_icon = "⚠️" if is_attack else "✓"

                event = {
                    "type": "detection",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "source_ip": src_ip,
                    "dst_ip": dst_ip,
                    "bytes": row_dict.get("src_bytes", 0),
                    "protocol": "TCP",
                    "prediction": pred,
                    "confidence": round(conf, 4),
                    "risk_level": risk,
                    "flag_icon": flag_icon,
                    "attack_type": pred,
                    "action_taken": action.action_taken,
                    "rule_triggered": action.rule_triggered,
                    "is_blocked": blocker.is_blocked(src_ip),
                    "input": {
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "src_bytes": row_dict.get("src_bytes", 0),
                        "dst_bytes": row_dict.get("dst_bytes", 0),
                        "protocol": "tcp",
                    }
                }

                yield f"data: {json.dumps(event)}\n\n"
                idx += 1

                # Pace: ~1 event per second
                time.sleep(1)

            except GeneratorExit:
                return
            except Exception as e:
                logger.error("Stream error: %s", e)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                time.sleep(2)

    return Response(
        _generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


def _compute_risk(prediction: str, confidence: float) -> str:
    """Compute a risk level string from prediction and confidence."""
    pred_lower = (prediction or "").lower()
    if pred_lower in ("normal", "benign"):
        return "low"
    if confidence >= 0.95:
        return "critical"
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.70:
        return "medium"
    return "low"

@app.route("/api/predict/csv", methods=["POST"])
def predict_csv():
    if MODEL_PIPE is None:
        return jsonify({"error": "model pipeline not loaded."}), 500
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "no selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "only csv allowed"}), 400
    try:
        content = file.stream.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))
        # Let the pipeline handle preprocessing; but if you used pd.get_dummies originally,
        # ensure CSV contains raw columns expected by pipeline.
        df_in = df  # pipeline will do the preprocessing
        results = []
        if hasattr(MODEL_PIPE, "predict_proba"):
            probs = MODEL_PIPE.predict_proba(df_in)
            preds = MODEL_PIPE.predict(df_in)
            for i, p in enumerate(preds):
                results.append({"prediction": str(p), "confidence": float(np.max(probs[i]))})
        else:
            preds = MODEL_PIPE.predict(df_in)
            for p in preds:
                results.append({"prediction": str(p), "confidence": 1.0})
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"csv predict failed: {str(e)}"}), 500

@app.route("/api/dashboard/stats", methods=["GET"])
def dashboard_stats():
    # If you saved X_val/y_val in models/, load and run predictions to produce counts.
    if MODEL_PIPE is None:
        return jsonify({"error": "model pipeline not loaded."}), 500
    # try load X_val/y_val if present (lazy)
    global X_VAL, Y_VAL
    if X_VAL is None:
        for cand in ["X_val_multi_selected.pkl", "X_val_binary_selected.pkl", "X_val.joblib"]:
            p = os.path.join(MODELS_DIR, cand)
            if os.path.exists(p):
                try:
                    X_VAL = joblib.load(p)
                    break
                except Exception:
                    pass
    if isinstance(X_VAL, pd.DataFrame):
        sample = X_VAL.sample(n=min(500, len(X_VAL)), random_state=42)
        try:
            if hasattr(MODEL_PIPE, "predict_proba"):
                probs = MODEL_PIPE.predict_proba(sample)
                preds = MODEL_PIPE.predict(sample)
                counts = {}
                recents = []
                for i in range(len(preds)):
                    p = str(preds[i])
                    counts[p] = counts.get(p, 0) + 1
                    if len(recents) < 10:
                        recents.append({"prediction": p, "confidence": float(np.max(probs[i])), "input": sample.iloc[i].to_dict()})
                return jsonify({"counts": counts, "recent": recents})
            else:
                preds = MODEL_PIPE.predict(sample)
                counts = {}
                recents = []
                for i,p in enumerate(preds):
                    p = str(p)
                    counts[p] = counts.get(p,0) + 1
                    if len(recents) < 10:
                        recents.append({"prediction": p, "confidence": 1.0, "input": sample.iloc[i].to_dict()})
                return jsonify({"counts": counts, "recent": recents})
        except Exception as e:
            return jsonify({"error": f"dashboard failure: {str(e)}"}), 500
    else:
        classes = []
        # try to extract classes from final estimator
        est = None
        if hasattr(MODEL_PIPE, "named_steps"):
            est = MODEL_PIPE.named_steps.get("estimator") or list(MODEL_PIPE.named_steps.values())[-1]
        classes = list(getattr(est or MODEL_PIPE, "classes_", []))
        counts = {str(c): 0 for c in classes} if classes else {}
        return jsonify({"counts": counts, "recent": []})

@app.route("/api/analytics/feature-importance", methods=["GET"])
def feature_importance():
    if MODEL_PIPE is None:
        return jsonify({"error": "model pipeline not loaded."}), 500
    # try tree-based feature_importances_ on final estimator
    est = None
    try:
        if hasattr(MODEL_PIPE, "named_steps"):
            est = MODEL_PIPE.named_steps.get("estimator") or list(MODEL_PIPE.named_steps.values())[-1]
        if est is not None and hasattr(est, "feature_importances_"):
            fi = list(est.feature_importances_)
            names = FEATURE_NAMES if FEATURE_NAMES else [f"f{i}" for i in range(len(fi))]
            out = [{"feature": names[i], "importance": float(fi[i])} for i in range(len(fi))]
            return jsonify(out)
    except Exception:
        pass

    # fallback: permutation importance if X_val and y_val exist
    # Try load X_val / y_val lazily
    global X_VAL, Y_VAL
    if X_VAL is None:
        for cand in ["X_val_multi_selected.pkl", "X_val_binary_selected.pkl", "X_val.joblib"]:
            p = os.path.join(MODELS_DIR, cand)
            if os.path.exists(p):
                try:
                    X_VAL = joblib.load(p)
                    break
                except Exception:
                    pass
    if Y_VAL is None:
        for cand in ["y_val_binary.pkl", "y_val_multi.pkl", "y_val.pkl", "y_val.joblib"]:
            p = os.path.join(MODELS_DIR, cand)
            if os.path.exists(p):
                try:
                    Y_VAL = joblib.load(p)
                    break
                except Exception:
                    pass

    if X_VAL is None or Y_VAL is None:
        return jsonify({"error": "no native feature_importances_ and missing X_val/y_val for permutation importance"}), 400

    try:
        sample = X_VAL.sample(n=min(200, len(X_VAL)), random_state=42)
        y_sample = Y_VAL.loc[sample.index] if isinstance(Y_VAL, (pd.Series, pd.DataFrame)) else Y_VAL
        perm = permutation_importance(MODEL_PIPE, sample, y_sample, n_repeats=10, random_state=42, n_jobs=1)
        names = sample.columns.tolist()
        out = [{"feature": names[i], "importance": float(perm.importances_mean[i])} for i in range(len(names))]
        out = sorted(out, key=lambda x: x["importance"], reverse=True)
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": f"feature importance failed: {str(e)}"}), 500

# ─── Prevention API endpoints ────────────────────────────────────────────────

@app.route("/api/prevention/status", methods=["GET"])
def prevention_status():
    """Get overall IPS status."""
    rules = db.get_rules()
    return jsonify({
        "mode": "live" if blocker.is_live_mode() else "simulation",
        "blocked_count": blocker.get_blocked_count(),
        "total_incidents": db.get_incident_count(),
        "active_rules": sum(1 for r in rules if r.get("enabled")),
        "total_rules": len(rules)
    })


@app.route("/api/prevention/blocked", methods=["GET"])
def prevention_blocked():
    """List all currently blocked IPs."""
    return jsonify(blocker.get_blocked_ips())


@app.route("/api/prevention/block", methods=["POST"])
def prevention_block():
    """Manually block an IP."""
    data = request.get_json(force=True, silent=True) or {}
    ip = data.get("ip", "").strip()
    if not ip:
        return jsonify({"error": "ip is required"}), 400
    reason = data.get("reason", "Manual block")
    duration = data.get("duration_hours")
    result = blocker.block_ip(
        ip, reason=reason, attack_type="manual",
        blocked_by="manual",
        duration_hours=float(duration) if duration else None
    )
    return jsonify(result)


@app.route("/api/prevention/unblock", methods=["POST"])
def prevention_unblock():
    """Manually unblock an IP."""
    data = request.get_json(force=True, silent=True) or {}
    ip = data.get("ip", "").strip()
    if not ip:
        return jsonify({"error": "ip is required"}), 400
    result = blocker.unblock_ip(ip)
    return jsonify(result)


@app.route("/api/prevention/rules", methods=["GET"])
def prevention_get_rules():
    """Get all prevention rules."""
    return jsonify(db.get_rules())


@app.route("/api/prevention/rules", methods=["PUT"])
def prevention_update_rules():
    """Update a prevention rule."""
    data = request.get_json(force=True, silent=True) or {}
    rule_name = data.get("rule_name", "").strip()
    if not rule_name:
        return jsonify({"error": "rule_name is required"}), 400
    enabled = data.get("enabled")
    config = data.get("config")
    success = db.update_rule(rule_name, enabled=enabled, config=config)
    if not success:
        return jsonify({"error": f"rule '{rule_name}' not found"}), 404
    return jsonify({"status": "updated", "rule_name": rule_name})


@app.route("/api/prevention/incidents", methods=["GET"])
def prevention_incidents():
    """Query incident log."""
    limit = request.args.get("limit", 100, type=int)
    offset = request.args.get("offset", 0, type=int)
    source_ip = request.args.get("source_ip")
    risk_level = request.args.get("risk_level")
    incidents = db.get_incidents(limit=limit, offset=offset,
                                source_ip=source_ip, risk_level=risk_level)
    return jsonify(incidents)


@app.route("/api/prevention/toggle-mode", methods=["POST"])
def prevention_toggle_mode():
    """Toggle between simulation and live mode."""
    data = request.get_json(force=True, silent=True) or {}
    new_mode = data.get("mode", "").strip().lower()
    if new_mode == "live":
        blocker.set_live_mode(True)
    elif new_mode == "simulation":
        blocker.set_live_mode(False)
    else:
        # Toggle
        blocker.set_live_mode(not blocker.is_live_mode())
    return jsonify({"mode": "live" if blocker.is_live_mode() else "simulation"})



# ── Aggregated live stats for Dashboard / Traffic Analytics / Logs ────────
@app.route("/api/incidents/summary", methods=["GET"])
def incidents_summary():
    """Return aggregated stats from live incident data."""
    return jsonify(db.get_incident_summary())


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    load_artifacts()
    _load_expected_features()
    # Initialize prevention system
    db.init_db()
    db.seed_default_rules()
    blocker.start_cleanup_scheduler()
    logger.info("Prevention system initialized (mode=%s)",
                "LIVE" if blocker.is_live_mode() else "SIMULATION")
    app.run(host="0.0.0.0", port=5000, debug=True)
