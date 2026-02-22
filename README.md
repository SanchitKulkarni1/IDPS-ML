<![CDATA[# 🛡️ CyberView — ML-Powered Network Intrusion Detection & Prevention System

> **A real-time NIDS/IPS (Network Intrusion Detection & Prevention System) powered by machine learning, with a modern React dashboard for live traffic monitoring, threat analytics, and automated IP blocking.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Backend — Detailed Breakdown](#backend--detailed-breakdown)
  - [appserver.py — Flask API Server](#appserverpy--flask-api-server)
  - [database.py — SQLite Database Manager](#databasepy--sqlite-database-manager)
  - [rules_engine.py — Auto-Response Rules Engine](#rules_enginepy--auto-response-rules-engine)
  - [blocker.py — IP Blocking Engine](#blockerpy--ip-blocking-engine)
  - [app.py — Streamlit Standalone UI](#apppy--streamlit-standalone-ui)
  - [viewParameters.py — Feature Inspector Utility](#viewparameterspy--feature-inspector-utility)
- [Frontend — Detailed Breakdown](#frontend--detailed-breakdown)
  - [Pages](#pages)
  - [API Layer (api.ts)](#api-layer-apits)
  - [Components & Hooks](#components--hooks)
- [ML Models & Data](#ml-models--data)
- [Database Schema](#database-schema)
- [API Reference](#api-reference)
- [How It All Works Together](#how-it-all-works-together)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)

---

## Overview

CyberView is a full-stack **Intrusion Detection and Prevention System (IDPS)** built as a BE (Bachelor of Engineering) project. It combines:

1. **Machine Learning** — A pre-trained scikit-learn classifier that identifies network traffic as Normal or one of several attack types (DDoS, Probe, U2R, R2L, etc.).
2. **Real-Time Detection** — A Server-Sent Events (SSE) stream that continuously classifies validation data, simulating live network traffic analysis.
3. **Automated Prevention** — A configurable rules engine that automatically blocks malicious IPs based on prediction confidence, attack severity, repeat offenses, and rate limits.
4. **Modern Dashboard** — A React + TypeScript frontend with live charts, traffic analytics, incident logs, explainability views, and full prevention management.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     React Frontend (Vite)                   │
│   Dashboard │ Live Detection │ Analytics │ Prevention │ Logs│
│                        ↕ HTTP / SSE                         │
├─────────────────────────────────────────────────────────────┤
│                   Flask API Server (appserver.py)           │
│  /api/predict │ /api/stream │ /api/prevention/* │ /api/...  │
├──────────┬──────────────┬───────────────┬───────────────────┤
│ ML Model │ Rules Engine │    Blocker    │     Database      │
│ (joblib) │(rules_engine)│ (blocker.py)  │   (database.py)   │
│          │              │ SIM / iptables│    SQLite (WAL)   │
└──────────┴──────────────┴───────────────┴───────────────────┘
```

---

## Tech Stack

| Layer     | Technology                                                                  |
| --------- | --------------------------------------------------------------------------- |
| ML        | scikit-learn, joblib, pandas, numpy                                         |
| Backend   | Python 3, Flask, Flask-CORS, SSE (Server-Sent Events)                      |
| Database  | SQLite 3 (WAL mode, thread-safe with locking)                              |
| Frontend  | React 18, TypeScript, Vite, TailwindCSS, shadcn/ui, Recharts, Lucide Icons |
| Blocking  | iptables (live mode) / simulation mode (default)                           |
| Standalone| Streamlit (alternative single-page UI)                                     |

---

## Project Structure

```
beproject/
├── appserver.py           # 🔧 Main Flask API server (all endpoints)
├── database.py            # 🗄️ SQLite database manager (incidents, blocked IPs, rules)
├── rules_engine.py        # ⚙️ Auto-response rules engine (4 configurable rules)
├── blocker.py             # 🚫 IP blocking engine (simulation + live iptables)
├── app.py                 # 📊 Streamlit standalone UI (Phase 1 demo)
├── viewParameters.py      # 🔍 Utility to inspect model features
├── requirements.txt       # 📦 Python dependencies
├── models/                # 🧠 Pre-trained ML models & validation data
│   ├── best_multi_model.pkl
│   ├── best_binary_model.pkl
│   ├── X_train_multi_selected.pkl
│   ├── X_train_binary_selected.pkl
│   ├── X_val_multi_selected.pkl
│   ├── X_val_binary_selected.pkl
│   └── y_val_binary.pkl
├── data/
│   └── nids.db            # 💾 SQLite database (auto-created)
└── cyberview-dash/        # 🖥️ React frontend dashboard
    ├── src/
    │   ├── App.tsx         # Router & provider setup
    │   ├── pages/          # 9 page components
    │   ├── components/     # Reusable UI components (Layout, StatCard, shadcn/ui)
    │   ├── lib/            # API client (api.ts) & utilities
    │   └── hooks/          # Custom React hooks
    ├── package.json
    ├── vite.config.ts
    └── tailwind.config.ts
```

---

## Backend — Detailed Breakdown

### `appserver.py` — Flask API Server

**The central piece of the backend.** This file initializes the Flask application, loads the ML model, and exposes all REST and SSE endpoints consumed by the frontend.

#### Startup Flow

1. Loads ML model from `models/` (tries `model_pipeline.pkl` → `best_multi_model.pkl` → `best_binary_model.pkl`).
2. Loads expected feature names from validation data (`X_val_multi_selected.pkl`).
3. Initializes the SQLite database (`db.init_db()`), seeds default prevention rules.
4. Starts the background block-cleanup scheduler (expires timed blocks every 60 seconds).
5. Starts Flask on `0.0.0.0:5000`.

#### Key Functions

| Function                | Purpose                                                                                                                                                 |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `load_artifacts()`      | Loads the ML model pipeline from disk using joblib. Tries multiple .pkl candidates.                                                                     |
| `_load_expected_features()` | Loads expected feature column names from validation DataFrames so incoming input can be properly aligned to the model's expected schema.             |
| `prepare_input_df(data)` | Strips metadata keys (`src_ip`, `dst_ip`, `protocol`, `timestamp`, `label`) from incoming JSON, aligns remaining features to model's expected columns, fills missing features with 0. |
| `_compute_risk(prediction, confidence)` | Maps a prediction + confidence score to a risk level: `critical` (≥95%), `high` (≥85%), `medium` (≥70%), or `low`.                      |

#### API Endpoints

| Endpoint                            | Method  | Description                                                                                                                                                                            |
| ----------------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/api/predict`                      | `POST`  | Accepts a JSON body of network traffic features. Runs ML prediction, evaluates prevention rules, logs the incident, and returns `{prediction, confidence, action_taken, is_blocked}`. |
| `/api/stream`                       | `GET`   | **SSE endpoint.** Continuously samples random rows from X_val, predicts, generates synthetic IPs, runs rules, logs incidents, and pushes ~1 event/second as a Server-Sent Event stream. |
| `/api/predict/csv`                  | `POST`  | Accepts a CSV file upload, runs batch predictions on all rows, returns an array of `{prediction, confidence}`.                                                                        |
| `/api/dashboard/stats`              | `GET`   | Samples 500 rows from X_val, predicts, and returns class distribution counts + 10 recent predictions.                                                                                 |
| `/api/analytics/feature-importance` | `GET`   | Returns feature importance scores. Tries tree-based `feature_importances_` first, falls back to permutation importance if X_val/y_val are available.                                   |
| `/api/incidents/summary`            | `GET`   | Returns aggregated stats from the SQLite incident log (totals, attack types, risk levels, timeline, top IPs, recent incidents).                                                       |
| `/api/prevention/status`            | `GET`   | Returns IPS status: current mode (simulation/live), blocked count, incident count, active/total rules.                                                                                |
| `/api/prevention/blocked`           | `GET`   | Lists all currently active blocked IPs with metadata (reason, attack type, when blocked, expiry).                                                                                     |
| `/api/prevention/block`             | `POST`  | Manually block an IP. Accepts `{ip, reason, duration_hours}`.                                                                                                                         |
| `/api/prevention/unblock`           | `POST`  | Manually unblock an IP. Accepts `{ip}`.                                                                                                                                               |
| `/api/prevention/rules`             | `GET`   | Returns all 4 configurable prevention rules with their enabled state and config JSON.                                                                                                 |
| `/api/prevention/rules`             | `PUT`   | Update a rule's `enabled` state or `config`. Accepts `{rule_name, enabled?, config?}`.                                                                                                |
| `/api/prevention/incidents`         | `GET`   | Query incident log with filters: `?limit=100&offset=0&source_ip=x&risk_level=y`.                                                                                                     |
| `/api/prevention/toggle-mode`       | `POST`  | Toggle between simulation and live (iptables) mode. Accepts `{mode: "live"|"simulation"}`.                                                                                            |

---

### `database.py` — SQLite Database Manager

**Manages all persistent data** using thread-safe SQLite with WAL journaling. Three tables store all system state.

#### Tables

| Table                | Purpose                                                                  |
| -------------------- | ------------------------------------------------------------------------ |
| `incidents`          | Every ML prediction is logged as a row — source IP, prediction, confidence, risk level, action taken, timestamped. |
| `blocked_ips`        | Active (and historical) IP blocks with reason, attack type, who blocked it, and optional expiry time. |
| `prevention_rules`   | 4 configurable auto-response rules stored as JSON config blobs.          |

#### Key Functions

| Function                                   | Purpose                                                                                                      |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `init_db()`                                | Creates all three tables + indexes if they don't exist.                                                      |
| `log_incident(...)`                        | Inserts a new incident row with all detection metadata. Called on every prediction.                           |
| `get_incidents(limit, offset, source_ip, risk_level)` | Queries incidents with optional filters and pagination to power the Logs page.                    |
| `get_incident_count()`                     | Returns total number of logged incidents.                                                                    |
| `block_ip(ip, reason, attack_type, ...)`   | Inserts or updates a blocked IP record (uses `ON CONFLICT` upsert).                                         |
| `unblock_ip(ip)`                           | Deactivates a block by setting `active=0`.                                                                   |
| `get_blocked_ips(active_only)`             | Returns list of blocked IPs, optionally filtering to only active blocks.                                     |
| `is_ip_blocked(ip)`                        | Fast lookup to check if an IP is currently blocked.                                                          |
| `cleanup_expired_blocks()`                 | Deactivates blocks past their `expires_at` timestamp. Called every 60s by the background scheduler.          |
| `get_recent_incident_count_for_ip(ip, window)` | Counts incidents from a specific IP within the last N seconds. Used by the repeated-offender and rate-limit rules. |
| `get_incident_summary()`                   | Aggregates all incident data for the Dashboard: totals, attack type breakdown, risk distribution, hourly timeline, top source IPs, recent incidents, and blocked count. |
| `seed_default_rules()`                     | Populates the `prevention_rules` table with 4 default rules on first run.                                   |
| `get_rules()` / `update_rule(...)`         | CRUD operations for prevention rules.                                                                        |

---

### `rules_engine.py` — Auto-Response Rules Engine

**The brain of the prevention system.** After every ML prediction, this engine evaluates the result against 4 configurable rules and decides whether to automatically block the source IP.

#### Evaluation Flow

```
prediction arrives → skip if no valid IP or already blocked
                   → check Rule 1: high_confidence_attack
                   → check Rule 2: critical_attack_type
                   → check Rule 3: repeated_offender
                   → check Rule 4: rate_limit
                   → no rule triggered → return "none"
```

#### The 4 Rules

| Rule                     | Default Config                                            | What It Does                                                                      |
| ------------------------ | --------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `high_confidence_attack` | `min_confidence: 0.90`, block for 1 hour                  | Blocks IPs producing attack predictions with ≥90% confidence.                     |
| `critical_attack_type`   | `attack_types: [ddos, u2r, dos, neptune, smurf]`, 24h ban | Blocks IPs launching specific critical attack categories immediately.             |
| `repeated_offender`      | `max_incidents: 3`, `window: 300s`, 24h ban                | Blocks IPs flagged ≥3 times within a 5-minute window (non-normal predictions).    |
| `rate_limit`             | `max_connections: 100`, `window: 60s`, 30-min ban          | Blocks IPs exceeding 100 incidents in 60 seconds.                                 |

Each rule is independently **enable/disable-able** and has a **configurable JSON config** (thresholds, durations, etc.) editable from the Prevention page.

#### Return Value

Every evaluation returns an `ActionResult` dataclass:
```python
@dataclass
class ActionResult:
    action_taken: str           # 'none', 'blocked', 'already_blocked'
    rule_triggered: str | None  # Name of the rule that fired
    details: str                # Human-readable description
    block_duration_hours: float | None
```

---

### `blocker.py` — IP Blocking Engine

**Handles the actual IP blocking**, supporting two modes:

| Mode              | Behavior                                                                                                    |
| ----------------- | ----------------------------------------------------------------------------------------------------------- |
| **Simulation** (default) | Records blocks in the SQLite database only. No real network changes. Safe for development/demo.       |
| **Live**          | Records blocks in DB **and** adds/removes `iptables DROP` rules to actually block traffic on the host OS.   |

#### Key Features

- **Dual-mode toggling** — Switch between simulation and live at runtime via API (`/api/prevention/toggle-mode`).
- **Timed blocks** — Blocks can have an `expires_at` timestamp. A background cleanup scheduler runs every 60 seconds to deactivate expired blocks (and remove their iptables rules in live mode).
- **Background scheduler** — `start_cleanup_scheduler()` spawns a daemon thread that calls `cleanup_expired()` every 60 seconds.

#### Key Functions

| Function                    | Purpose                                                                                       |
| --------------------------- | --------------------------------------------------------------------------------------------- |
| `block_ip(ip, ...)`        | Records a block in DB + optionally applies `iptables -A INPUT -s IP -j DROP` in live mode.    |
| `unblock_ip(ip)`           | Deactivates block in DB + optionally removes the iptables rule.                               |
| `get_blocked_ips()`        | Returns all currently active blocked IPs from the database.                                   |
| `is_blocked(ip)`           | Checks if an IP is currently blocked.                                                         |
| `cleanup_expired()`        | Deactivates expired blocks and removes their iptables rules if in live mode.                  |
| `start_cleanup_scheduler()` | Starts the background cleanup timer (60s interval). Called once at server startup.            |
| `set_live_mode(enabled)`   | Toggles between simulation and live mode at runtime.                                          |

---

### `app.py` — Streamlit Standalone UI

**A standalone Phase 1 demo interface** built with Streamlit. This was the original UI before the React dashboard was built. It provides:

- **3 input modes**: Manual entry of all 42 network features, random selection from training data, or synthetic random generation based on feature statistics.
- **ML prediction**: Runs the multiclass model and displays the predicted attack type with a confidence bar chart.
- **Feature inspection**: Expandable view showing all input feature values.

> **Note:** This file is a standalone alternative to the React dashboard. It loads `best_multi_model.pkl` directly and does not use the prevention system.

---

### `viewParameters.py` — Feature Inspector Utility

A small utility script that loads the training data (`X_train_multi_selected.pkl`) and prints the number of features and their names. Useful for debugging and understanding what the model expects as input.

---

## Frontend — Detailed Breakdown

The frontend is a **React 18 + TypeScript** SPA built with **Vite**, styled with **TailwindCSS** and **shadcn/ui** components, using **Recharts** for data visualization.

### Pages

| Page                                  | Route                | Description                                                                                                                                               |
| ------------------------------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dashboard** (`Dashboard.tsx`)       | `/`                  | Overview with stat cards (total packets, attacks, normal, blocked IPs), a traffic timeline line chart, and an attack distribution pie chart. Polls `/api/incidents/summary` every 2 seconds. |
| **Live Detection** (`LiveDetection.tsx`) | `/live-detection`  | Connects to the SSE stream (`/api/stream`) and displays each prediction as it arrives in real-time. Shows source/dest IPs, prediction, confidence, risk level, and any prevention action taken. |
| **Live Analysis** (`LiveAnalysis.tsx`) | `/live-analysis`    | Real-time analytics view that visualizes the streaming data with charts and statistics as predictions come in.                                             |
| **Traffic Analytics** (`TrafficAnalytics.tsx`) | `/traffic-analytics` | Detailed analytics with attack type distribution charts, risk level breakdowns, and top source IP tables. Powered by `/api/incidents/summary`.     |
| **Explainability** (`Explainability.tsx`) | `/explainability` | Displays feature importance scores from the ML model (via `/api/analytics/feature-importance`), helping users understand which features drive predictions. |
| **Prevention** (`Prevention.tsx`)     | `/prevention`        | Full IPS management: toggle simulation/live mode, view/block/unblock IPs, configure all 4 prevention rules (enable/disable, edit thresholds), view incident log with risk badges. Polls every 3 seconds. |
| **Logs** (`Logs.tsx`)                 | `/logs`              | Searchable, filterable incident log table powered by `/api/prevention/incidents`. Supports filtering by source IP and risk level.                         |
| **About** (`About.tsx`)              | `/about`             | Project information and documentation page.                                                                                                               |
| **NotFound** (`NotFound.tsx`)        | `*`                  | 404 fallback page.                                                                                                                                        |

### API Layer (`api.ts`)

Located at `src/lib/api.ts`, this module provides a clean abstraction over `fetch` for communicating with the Flask backend on `http://localhost:5000`:

| Function                   | Purpose                                                        |
| -------------------------- | -------------------------------------------------------------- |
| `getJson(path, params?)`   | GET request with optional query string construction.           |
| `jsonPost(path, body)`     | POST request with JSON body.                                   |
| `putJson(path, body)`      | PUT request with JSON body (for updating rules).               |
| `filePost(path, formData)` | POST with `FormData` (for CSV file uploads).                   |
| `buildUrl(path)`           | Constructs full URLs from relative paths.                      |
| `parseResponse(res)`       | Safely parses JSON responses with fallback to raw text.        |

All functions handle error responses and throw structured error objects on non-OK responses.

### Components & Hooks

| Component/Hook     | Purpose                                                                        |
| ------------------ | ------------------------------------------------------------------------------ |
| `Layout.tsx`       | App shell with sidebar navigation and main content area.                       |
| `StatCard.tsx`     | Reusable metric card component (icon, title, value, optional trend indicator). |
| `ui/`              | 49 shadcn/ui component primitives (Button, Card, Table, Dialog, Switch, etc.). |
| `use-mobile.tsx`   | Custom hook for responsive breakpoint detection.                               |
| `use-toast.ts`     | Toast notification management hook.                                            |

---

## ML Models & Data

All model artifacts are stored in the `models/` directory:

| File                             | Description                                                                                  |
| -------------------------------- | -------------------------------------------------------------------------------------------- |
| `best_multi_model.pkl`           | **Primary model** — Multiclass classifier trained on the NSL-KDD / UNSW-NB15 dataset. Predicts specific attack categories (Normal, DoS, Probe, U2R, R2L, etc.). |
| `best_binary_model.pkl`          | Binary classifier (Normal vs. Attack). Used as a fallback.                                   |
| `X_train_multi_selected.pkl`     | Training feature set (multiclass). Used by Streamlit UI for random sample generation.        |
| `X_train_binary_selected.pkl`    | Training feature set (binary).                                                               |
| `X_val_multi_selected.pkl`       | **Validation feature set.** Used by the SSE stream to simulate live traffic.                 |
| `X_val_binary_selected.pkl`      | Validation feature set (binary).                                                             |
| `y_val_binary.pkl`               | Validation labels. Used for permutation importance computation.                              |

The model was trained using scikit-learn and supports `predict_proba()` for confidence scores.

---

## Database Schema

The SQLite database (`data/nids.db`) contains 3 tables:

### `incidents`
| Column         | Type     | Description                                   |
| -------------- | -------- | --------------------------------------------- |
| `id`           | INTEGER  | Auto-incrementing primary key                 |
| `timestamp`    | TEXT     | UTC ISO-8601 timestamp of detection           |
| `source_ip`    | TEXT     | Source IP address                              |
| `dest_ip`      | TEXT     | Destination IP address                         |
| `protocol`     | TEXT     | Network protocol                               |
| `prediction`   | TEXT     | ML model prediction label                      |
| `attack_type`  | TEXT     | Specific attack type classification            |
| `confidence`   | REAL     | Model confidence score (0.0–1.0)               |
| `risk_level`   | TEXT     | Computed risk: low / medium / high / critical  |
| `features`     | TEXT     | Full feature JSON blob                         |
| `action_taken` | TEXT     | Prevention action: none / blocked / already_blocked |

### `blocked_ips`
| Column         | Type     | Description                                    |
| -------------- | -------- | ---------------------------------------------- |
| `id`           | INTEGER  | Auto-incrementing primary key                  |
| `ip_address`   | TEXT     | Blocked IP (UNIQUE constraint)                 |
| `reason`       | TEXT     | Why the IP was blocked                         |
| `attack_type`  | TEXT     | Category of attack that triggered the block    |
| `blocked_at`   | TEXT     | UTC ISO-8601 timestamp                         |
| `blocked_by`   | TEXT     | Who blocked it: 'auto' or 'manual'             |
| `expires_at`   | TEXT     | Optional expiry timestamp (NULL = permanent)   |
| `active`       | INTEGER  | 1 = active block, 0 = expired/unblocked        |

### `prevention_rules`
| Column      | Type     | Description                                       |
| ----------- | -------- | ------------------------------------------------- |
| `id`        | INTEGER  | Auto-incrementing primary key                     |
| `rule_name` | TEXT     | Unique rule identifier                            |
| `enabled`   | INTEGER  | 1 = enabled, 0 = disabled                         |
| `config`    | TEXT     | JSON blob containing rule-specific configuration  |

---

## How It All Works Together

### End-to-End Detection → Prevention → Dashboard Flow

```
1. SSE Stream starts (/api/stream)
   │
2. Random row sampled from X_val (simulating real traffic)
   │
3. ML Model predicts → e.g. "ddos" with 97% confidence
   │
4. Risk computed → "critical" (≥95%)
   │
5. Rules Engine evaluates:
   │  ├─ Rule 1 (high_confidence_attack): 97% ≥ 90% → FIRES → block IP for 1h
   │  └─ (remaining rules skipped — already blocked)
   │
6. Blocker records block in SQLite
   │  └─ (In live mode: also adds iptables DROP rule)
   │
7. Incident logged to DB with all metadata
   │
8. SSE event pushed to frontend with prediction + action
   │
9. Frontend updates:
   ├─ Live Detection page shows the new event
   ├─ Dashboard stat cards update (polls /api/incidents/summary)
   ├─ Prevention page shows new blocked IP
   └─ Logs page shows new incident row
```

### Key Data Flows

- **Live Monitoring**: Frontend connects to `/api/stream` (SSE) → displays each event in real-time on Live Detection and Live Analysis pages.
- **Dashboard Polling**: Dashboard and Traffic Analytics pages poll `/api/incidents/summary` every 2-3 seconds for aggregated stats.
- **Prevention Management**: Prevention page reads rules, blocked IPs, and incidents via REST endpoints. Allows manual block/unblock and rule toggling.
- **Explainability**: The Explainability page calls `/api/analytics/feature-importance` to show which features most influence model predictions.

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Node.js 18+ (with npm or bun)
- Pre-trained model files in `models/` (see [ML Models & Data](#ml-models--data))

### Backend Setup

```bash
# From project root
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

pip install flask flask-cors pandas numpy scikit-learn joblib streamlit matplotlib
```

### Frontend Setup

```bash
cd cyberview-dash
npm install          # or: bun install
```

---

## Running the Project

### 1. Start the Backend

```bash
# From project root (with venv activated)
python appserver.py
```

This starts the Flask server on **http://localhost:5000**.

### 2. Start the Frontend

```bash
cd cyberview-dash
npm run dev
```

This starts the Vite dev server on **http://localhost:5173** (or similar).

### 3. (Optional) Streamlit UI

```bash
# From project root
streamlit run app.py
```

Opens a standalone Streamlit interface on **http://localhost:8501**.

---

## License

Developed as part of a BE (Bachelor of Engineering) Final Year Project.
]]>
