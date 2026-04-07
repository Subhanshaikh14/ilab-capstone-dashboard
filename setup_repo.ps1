# =============================================================================
# Coherence Engine - Output Dashboard Repo Setup
# Team 14-02 | Subhan Farid | System 1 Output Layer
#
# Usage (run in PowerShell):
#   .\setup_repo.ps1
#
# If you get a permissions error, run this first:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
# =============================================================================

$REPO = "ilab-capstone-dashboard"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Setting up: $REPO" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# ── 1. Create folder structure ────────────────────────────────────────────────
New-Item -ItemType Directory -Force -Path "$REPO"                   | Out-Null
New-Item -ItemType Directory -Force -Path "$REPO\pages"             | Out-Null
New-Item -ItemType Directory -Force -Path "$REPO\data\sample"       | Out-Null
New-Item -ItemType Directory -Force -Path "$REPO\tests"             | Out-Null
New-Item -ItemType Directory -Force -Path "$REPO\assets"            | Out-Null
New-Item -ItemType Directory -Force -Path "$REPO\.streamlit"        | Out-Null

Write-Host "  Folders created" -ForegroundColor Green

# ── 2. Write requirements.txt ─────────────────────────────────────────────────
@"
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
"@ | Set-Content "$REPO\requirements.txt" -Encoding UTF8

Write-Host "  requirements.txt written" -ForegroundColor Green

# ── 3. Write .streamlit\config.toml ──────────────────────────────────────────
@"
[theme]
base                     = "light"
backgroundColor          = "#f8f7f4"
secondaryBackgroundColor = "#f1efe8"
textColor                = "#2c2c2a"
font                     = "sans serif"

[server]
headless = true

[browser]
gatherUsageStats = false
"@ | Set-Content "$REPO\.streamlit\config.toml" -Encoding UTF8

Write-Host "  .streamlit\config.toml written" -ForegroundColor Green

# ── 4. Write .gitignore ───────────────────────────────────────────────────────
@"
__pycache__/
*.pyc
*.pyo
.env
.venv/
venv/
*.pkl
*.csv
*.json
!data/sample/*.json
.DS_Store
desktop.ini
"@ | Set-Content "$REPO\.gitignore" -Encoding UTF8

Write-Host "  .gitignore written" -ForegroundColor Green

# ── 5. Write data\sample\example_single_goal.json ────────────────────────────
@"
[
  {
    "goal_id": 3,
    "attainability": 0.18,
    "relevance": 0.29,
    "coherence": 0.36,
    "integrity": 0.41,
    "overall": 0.31,
    "composite_adjusted": 0.31,
    "gp_mean": 0.21,
    "gp_std": 0.062,
    "gp_weight": 0.55,
    "llm_weight": 0.45,
    "baseline": 0.19,
    "uncertain": false,
    "at_risk": true,
    "critical": true,
    "weakest_dim": "attainability",
    "composite_p6": 0.33,
    "composite_p12": 0.36,
    "reasoning": {
      "attainability": "Current ROAS of 0.91 against a target of 3.5. Projected ROAS by period 24 is approximately 1.4.",
      "relevance": "Sitting in orange_high - overfunded relative to current output gains.",
      "coherence": "Allocation drift is moderate. Partially inconsistent with L2 bucket pattern.",
      "integrity": "Output quality is high (0.90) but needle move ratio of 0.33 suggests poor conversion."
    },
    "llm_scores": {
      "llama3":   {"attainability": 0.17, "relevance": 0.30, "coherence": 0.35, "integrity": 0.40, "success": true},
      "gemma3":   {"attainability": 0.19, "relevance": 0.28, "coherence": 0.37, "integrity": 0.43, "success": true},
      "nemotron": {"attainability": 0.18, "relevance": 0.29, "coherence": 0.35, "integrity": 0.40, "success": true}
    },
    "ensemble_meta": {
      "attainability": {"gp_std": 0.062, "gp_weight": 0.55, "llm_weight": 0.45, "uncertain": false}
    },
    "n_llm_ok": 3,
    "status": "ok"
  }
]
"@ | Set-Content "$REPO\data\sample\example_single_goal.json" -Encoding UTF8

Write-Host "  data\sample\example_single_goal.json written" -ForegroundColor Green

# ── 6. Write tests\test_dashboard.py ─────────────────────────────────────────
@"
"""
Basic smoke tests for dashboard helper functions.
Run with:  python -m pytest tests/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest.mock as mock
sys.modules['streamlit'] = mock.MagicMock()

import pandas as pd
from dashboard import (
    score_color, score_bar_class, derive_flags,
    get_goal_meta, get_bucket_path, detect_sibling_conflict,
    DIMS, RISK_THRESHOLD, CRITICAL_THRESHOLD,
    DEMO_GOALS_DATA, DEMO_BUCKETS_DATA, DEMO_RESULTS,
)

def test_score_color():
    assert score_color(0.6)  == "#1d9e75"
    assert score_color(0.4)  == "#ef9f27"
    assert score_color(0.1)  == "#e24b4a"
    assert score_color(0.5)  == "#1d9e75"
    assert score_color(0.35) == "#ef9f27"

def test_score_bar_class():
    assert score_bar_class(0.6) == "dim-bar-green"
    assert score_bar_class(0.4) == "dim-bar-amber"
    assert score_bar_class(0.2) == "dim-bar-red"

def test_derive_flags_critical():
    r = {"overall": 0.15, "attainability": 0.1, "relevance": 0.2,
         "coherence": 0.1, "integrity": 0.2, "status": "ok"}
    r = derive_flags(r)
    assert r["critical"]    is True
    assert r["at_risk"]     is True
    assert r["weakest_dim"] in DIMS

def test_derive_flags_ok():
    r = {"overall": 0.7, "attainability": 0.7, "relevance": 0.7,
         "coherence": 0.7, "integrity": 0.6, "status": "ok"}
    r = derive_flags(r)
    assert r["critical"] is False
    assert r["at_risk"]  is False

def test_get_goal_meta_found():
    meta = get_goal_meta(1, DEMO_GOALS_DATA)
    assert meta["name"]      != "Goal 1"
    assert meta["target"]    == 85.0
    assert meta["bucket_id"] == 19

def test_get_goal_meta_missing():
    meta = get_goal_meta(999, DEMO_GOALS_DATA)
    assert meta["name"] == "Goal 999"

def test_get_bucket_path():
    path = get_bucket_path(19, DEMO_BUCKETS_DATA)
    assert "Marketing"        in path
    assert "Paid Acquisition" in path
    assert "Google Ads Search" in path

def test_detect_sibling_conflict_true():
    conflict, msg = detect_sibling_conflict(
        [DEMO_RESULTS[0], DEMO_RESULTS[1]],
        DEMO_GOALS_DATA,
        DEMO_BUCKETS_DATA
    )
    assert conflict is True
    assert "Paid Acquisition" in msg

def test_detect_sibling_conflict_single():
    conflict, msg = detect_sibling_conflict(
        [DEMO_RESULTS[0]], DEMO_GOALS_DATA, DEMO_BUCKETS_DATA
    )
    assert conflict is False
"@ | Set-Content "$REPO\tests\test_dashboard.py" -Encoding UTF8

Write-Host "  tests\test_dashboard.py written" -ForegroundColor Green

# ── 7. Write README.md ────────────────────────────────────────────────────────
@"
# Coherence Engine — Output Dashboard

**System 1 Output Layer | Team 14-02 | UTS iLab Capstone**

Subhan Farid — ``dashboard.py``

---

## What this is

This repo contains the **output screen** of the Coherence Engine. It receives
scored goal data from System 3 and renders it as a Streamlit dashboard —
showing dimension scores, AI reasoning, trajectory projections, and
cross-goal comparisons.

It is one half of System 1. The input half lives in Padmasri's repo:
``github.com/Psri-01/ilab-capstone-system1``

---

## Repo structure

``````
ilab-capstone-dashboard/
├── dashboard.py                     <- main output screen (entry point)
├── requirements.txt
├── .streamlit/
│   └── config.toml                  <- theme config
├── data/
│   └── sample/
│       └── example_single_goal.json <- example System 3 payload
├── tests/
│   └── test_dashboard.py            <- pytest smoke tests
└── assets/                          <- images, icons if needed
``````

---

## Quickstart

``````bash
pip install -r requirements.txt
streamlit run dashboard.py
``````

Runs in **demo mode** with sample data — no System 2 or System 3 needed.

---

## Integration contract

### For Padmasri (System 1 input screen)

Before navigating to the dashboard, set these two session state keys:

``````python
st.session_state["query"]   = user_query         # str  — original question
st.session_state["results"] = results_from_sys3  # list — score payloads

st.switch_page("dashboard.py")
``````

Then uncomment and update this line in ``dashboard.py``:
``````python
# st.switch_page("app.py")  <- set to your actual entry filename
``````

### For Anupam / Fatemeh (System 3)

Each item in ``results`` must be the dict returned by ``score_goal()``
in ``07_score_goal.py``. Minimum required fields:

| Field | Type | Notes |
|---|---|---|
| ``goal_id`` | int | |
| ``attainability`` | float 0-1 | |
| ``relevance`` | float 0-1 | |
| ``coherence`` | float 0-1 | |
| ``integrity`` | float 0-1 | |
| ``overall`` | float 0-1 | |
| ``gp_std`` | float | |
| ``uncertain`` | bool | |
| ``reasoning`` | dict | keys: attainability, relevance, coherence, integrity |
| ``llm_scores`` | dict | keys: llama3, gemma3, nemotron |
| ``ensemble_meta`` | dict | |
| ``status`` | str | "ok" or "error" |

Optional (auto-derived if missing):
``composite_adjusted``, ``at_risk``, ``critical``, ``weakest_dim``,
``composite_p6``, ``composite_p12``

### Supporting CSV files

Place in the same folder as ``dashboard.py`` for richer output:
- ``goals.csv`` — goal names, targets, units, scenario flags
- ``buckets.csv`` — hierarchy for breadcrumbs and conflict detection
- ``forward_projection_poc.csv`` — trajectory scores at +6 and +12 periods

---

## Routing logic

| Condition | View |
|---|---|
| 1 goal returned | Single goal — full dimensions, trajectory, LLM panel |
| 2+ goals, shared bucket | Portfolio header + conflict alert + tabs + comparison table |
| 2+ goals, different buckets | Tabs + comparison table |

---

## Tests

``````bash
pip install pytest
python -m pytest tests/ -v
``````

---

## Key dates

| Date | Milestone |
|---|---|
| 24 Apr 2026 | Progress Report + supervisor demo |
| 6 May 2026  | Oral presentation |
| 22 May 2026 | Final report |
"@ | Set-Content "$REPO\README.md" -Encoding UTF8

Write-Host "  README.md written" -ForegroundColor Green

# ── 8. Write dashboard.py ─────────────────────────────────────────────────────
# Read the dashboard content from the current directory if it exists,
# otherwise write it inline
$dashboardContent = @'
"""
dashboard.py - Coherence Engine Output Dashboard
System 1 | Subhan Farid | Team 14-02

Entry point:
    streamlit run dashboard.py

Integration:
    Padmasri sets st.session_state["query"] and st.session_state["results"]
    then calls st.switch_page("dashboard.py")

    Runs in demo mode if no session state is present.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="Coherence Engine",
    page_icon="cog",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f8f7f4; }
[data-testid="stAppViewBlockContainer"] { padding-top: 1.5rem; max-width: 1100px; }
[data-testid="stSidebar"] { display: none; }
h1,h2,h3 { font-weight: 500 !important; }
.ce-card { background:#ffffff; border:0.5px solid rgba(0,0,0,0.10); border-radius:12px; padding:16px 18px; margin-bottom:12px; }
.ce-surface { background:#f1efe8; border-radius:8px; padding:12px 14px; margin-bottom:10px; }
.query-banner { background:#ffffff; border:0.5px solid rgba(0,0,0,0.10); border-radius:12px; padding:14px 18px; margin-bottom:20px; }
.query-label { font-size:11px; color:#888780; margin-bottom:4px; }
.query-text  { font-size:14px; font-weight:500; color:#2c2c2a; }
.score-xl { font-size:38px; font-weight:500; line-height:1; }
.score-lg { font-size:26px; font-weight:500; line-height:1; }
.score-md { font-size:18px; font-weight:500; line-height:1; }
.pill { display:inline-block; font-size:11px; padding:3px 10px; border-radius:20px; font-weight:500; }
.pill-crit { background:#fcebeb; color:#a32d2d; }
.pill-risk { background:#faeeda; color:#854f0b; }
.pill-ok   { background:#eaf3de; color:#3b6d11; }
.dim-bar-bg   { height:5px; background:#f1efe8; border-radius:3px; overflow:hidden; margin:6px 0; }
.dim-bar-red  { height:5px; background:#e24b4a; border-radius:3px; }
.dim-bar-amber{ height:5px; background:#ef9f27; border-radius:3px; }
.dim-bar-green{ height:5px; background:#1d9e75; border-radius:3px; }
.alert-warn { background:#faeeda; border:0.5px solid #ba7517; border-radius:8px; padding:10px 14px; font-size:12px; color:#633806; margin-bottom:12px; line-height:1.6; }
.comp-table { width:100%; border-collapse:collapse; font-size:13px; }
.comp-table th { font-size:11px; font-weight:500; color:#888780; padding:8px 10px; text-align:left; border-bottom:0.5px solid rgba(0,0,0,0.10); }
.comp-table td { padding:9px 10px; border-bottom:0.5px solid rgba(0,0,0,0.07); }
.comp-table tr:last-child td { border-bottom:none; }
.section-label { font-size:11px; font-weight:500; color:#888780; text-transform:uppercase; letter-spacing:.06em; margin:16px 0 8px 0; }
.bucket-node        { font-size:11px; padding:3px 9px; border-radius:20px; border:0.5px solid rgba(0,0,0,0.12); color:#888780; background:#f1efe8; }
.bucket-node-active { font-size:11px; padding:3px 9px; border-radius:20px; background:#e6f1fb; border:0.5px solid #185fa5; color:#185fa5; }
.insight-row { display:flex; gap:10px; align-items:flex-start; margin-bottom:9px; }
.idot { width:7px; height:7px; border-radius:50%; flex-shrink:0; margin-top:5px; }
.llm-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:2px; }
.traj-row { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
.traj-label { font-size:12px; color:#888780; width:70px; flex-shrink:0; }
.traj-barwrap { flex:1; height:7px; background:#f1efe8; border-radius:4px; overflow:hidden; }
.traj-val { font-size:12px; font-weight:500; width:36px; text-align:right; }
</style>
""", unsafe_allow_html=True)

DIMS               = ["attainability", "relevance", "coherence", "integrity"]
RISK_THRESHOLD     = 0.35
CRITICAL_THRESHOLD = 0.20

def score_color(s):
    if s >= 0.5:  return "#1d9e75"
    if s >= 0.35: return "#ef9f27"
    return "#e24b4a"

def score_bar_class(s):
    if s >= 0.5:  return "dim-bar-green"
    if s >= 0.35: return "dim-bar-amber"
    return "dim-bar-red"

def status_pill(r):
    if r.get("critical"): return '<span class="pill pill-crit">Critical</span>'
    if r.get("at_risk"):  return '<span class="pill pill-risk">At risk</span>'
    return '<span class="pill pill-ok">On track</span>'

def flag_color(flag):
    return {"underfunded": "#e24b4a", "overfunded": "#ef9f27"}.get(flag, "#1d9e75")

def derive_flags(r):
    score = r.get("composite_adjusted", r.get("overall", 0))
    r.setdefault("at_risk",   score < RISK_THRESHOLD)
    r.setdefault("critical",  score < CRITICAL_THRESHOLD)
    if "weakest_dim" not in r:
        r["weakest_dim"] = min(DIMS, key=lambda d: r.get(d, 1))
    return r

def get_goal_meta(goal_id, goals_df):
    if goals_df is None:
        return {"name": f"Goal {goal_id}", "target": "???", "unit": "", "bucket_id": None, "scenario": "???"}
    row = goals_df[goals_df["goal_id"] == goal_id]
    if row.empty:
        return {"name": f"Goal {goal_id}", "target": "???", "unit": "", "bucket_id": None, "scenario": "???"}
    row = row.iloc[0]
    return {
        "name":      row.get("metric_name",              f"Goal {goal_id}"),
        "target":    row.get("target_value_final_period", "???"),
        "unit":      row.get("metric_unit",              ""),
        "bucket_id": row.get("bucket_id"),
        "scenario":  row.get("scenario_story",           "???"),
    }

def get_bucket_path(bucket_id, buckets_df):
    if bucket_id is None or buckets_df is None:
        return []
    path, current, visited = [], bucket_id, set()
    while current and current not in visited:
        visited.add(current)
        row = buckets_df[buckets_df["bucket_id"] == current]
        if row.empty: break
        row = row.iloc[0]
        path.insert(0, row["bucket_name"])
        parent = row.get("parent_bucket_id")
        current = int(parent) if pd.notna(parent) else None
    return path

def detect_sibling_conflict(results, goals_df, buckets_df):
    if len(results) < 2 or goals_df is None:
        return False, ""
    bucket_ids = [get_goal_meta(r["goal_id"], goals_df)["bucket_id"] for r in results]
    parent_map = {}
    if buckets_df is not None:
        for bid in bucket_ids:
            if bid is None: continue
            row = buckets_df[buckets_df["bucket_id"] == bid]
            if not row.empty:
                p = row.iloc[0].get("parent_bucket_id")
                parent_map[bid] = int(p) if pd.notna(p) else None
    parents = list(set(v for v in parent_map.values() if v))
    shared  = parents[0] if len(parents) == 1 else None
    scenarios = [get_goal_meta(r["goal_id"], goals_df).get("scenario", "") for r in results]
    if shared and any("underfunded" in s for s in scenarios) and any("overfunded" in s for s in scenarios):
        pname = ""
        if buckets_df is not None:
            pr = buckets_df[buckets_df["bucket_id"] == shared]
            if not pr.empty: pname = pr.iloc[0]["bucket_name"]
        return True, (f"Inter-goal coherence issue detected under <strong>{pname}</strong>. "
                      f"One or more goals are underfunded while others are overfunded within the "
                      f"same parent bucket - budget is unevenly distributed across sibling goals.")
    return False, ""

@st.cache_data
def load_support_data(base_dir="."):
    goals_df = buckets_df = proj_df = None
    for fname, key in [("goals.csv","g"),("buckets.csv","b"),("forward_projection_poc.csv","p")]:
        path = os.path.join(base_dir, fname)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if key == "g": goals_df   = df
                if key == "b": buckets_df = df
                if key == "p": proj_df    = df
            except Exception:
                pass
    return goals_df, buckets_df, proj_df

DEMO_QUERY = "How are our paid acquisition channels performing? Are Google Ads and Social ROAS on track?"

DEMO_RESULTS = [
    {"goal_id":1,"attainability":0.22,"relevance":0.19,"coherence":0.31,"integrity":0.41,
     "overall":0.28,"composite_adjusted":0.28,"gp_std":0.041,"gp_weight":0.72,"llm_weight":0.28,
     "uncertain":False,"at_risk":True,"critical":True,"weakest_dim":"relevance",
     "composite_p6":0.30,"composite_p12":0.33,
     "reasoning":{"attainability":"At 21.9/85, this goal is 26% of target at the halfway point. Trajectory does not support reaching 85 by period 24.",
                  "relevance":"Allocation is below the minimum viable band - classic underfunding. Receives least budget in the Paid Acquisition bucket.",
                  "coherence":"Consistently low allocation across all 12 periods. Baseline is wrong - goal has never been adequately funded.",
                  "integrity":"Output quality (0.68) is reasonable but quantity is suppressed by the funding gap. Efficiency ratio of 0.26 is the best in the bucket."},
     "llm_scores":{"llama3":{"attainability":0.21,"relevance":0.18,"coherence":0.30,"integrity":0.40,"success":True},
                   "gemma3":{"attainability":0.23,"relevance":0.20,"coherence":0.33,"integrity":0.43,"success":True},
                   "nemotron":{"attainability":0.22,"relevance":0.19,"coherence":0.29,"integrity":0.39,"success":True}},
     "ensemble_meta":{"attainability":{"gp_std":0.041,"gp_weight":0.72,"llm_weight":0.28,"uncertain":False}},
     "n_llm_ok":3,"status":"ok"},
    {"goal_id":2,"attainability":0.18,"relevance":0.21,"coherence":0.28,"integrity":0.30,
     "overall":0.24,"composite_adjusted":0.24,"gp_std":0.088,"gp_weight":0.38,"llm_weight":0.62,
     "uncertain":True,"at_risk":True,"critical":True,"weakest_dim":"attainability",
     "composite_p6":0.22,"composite_p12":0.21,
     "reasoning":{"attainability":"At 19.8/85, this is the worst performing goal in the bucket. Slope is flat - projecting only ~22 by period 24.",
                  "relevance":"Allocation is in the red_high band - significantly overfunded. Budget is not converting to performance.",
                  "coherence":"Despite high allocation, output gains are minimal. Spend and outcome are structurally disconnected.",
                  "integrity":"Efficiency ratio of 0.10 is the lowest in the bucket. Delivered output is well below expected."},
     "llm_scores":{"llama3":{"attainability":0.17,"relevance":0.22,"coherence":0.27,"integrity":0.29,"success":True},
                   "gemma3":{"attainability":0.19,"relevance":0.20,"coherence":0.30,"integrity":0.32,"success":True},
                   "nemotron":{"attainability":0.18,"relevance":0.21,"coherence":0.27,"integrity":0.29,"success":True}},
     "ensemble_meta":{"attainability":{"gp_std":0.088,"gp_weight":0.38,"llm_weight":0.62,"uncertain":True}},
     "n_llm_ok":3,"status":"ok"},
    {"goal_id":3,"attainability":0.18,"relevance":0.29,"coherence":0.36,"integrity":0.41,
     "overall":0.31,"composite_adjusted":0.31,"gp_std":0.062,"gp_weight":0.55,"llm_weight":0.45,
     "uncertain":False,"at_risk":True,"critical":True,"weakest_dim":"attainability",
     "composite_p6":0.33,"composite_p12":0.36,
     "reasoning":{"attainability":"Current ROAS of 0.91 against a target of 3.5. Projected ROAS by period 24 is approximately 1.4.",
                  "relevance":"Sitting in orange_high - overfunded relative to current output gains.",
                  "coherence":"Allocation drift is moderate. Partially inconsistent with L2 bucket pattern.",
                  "integrity":"Output quality is the highest in the bucket (0.90) but needle move ratio of 0.33 suggests poor conversion."},
     "llm_scores":{"llama3":{"attainability":0.17,"relevance":0.30,"coherence":0.35,"integrity":0.40,"success":True},
                   "gemma3":{"attainability":0.19,"relevance":0.28,"coherence":0.37,"integrity":0.43,"success":True},
                   "nemotron":{"attainability":0.18,"relevance":0.29,"coherence":0.35,"integrity":0.40,"success":True}},
     "ensemble_meta":{"attainability":{"gp_std":0.062,"gp_weight":0.55,"llm_weight":0.45,"uncertain":False}},
     "n_llm_ok":3,"status":"ok"},
]

DEMO_GOALS_DATA = pd.DataFrame([
    {"goal_id":1,"metric_name":"Performance Metric - Google Ads Search", "metric_unit":"score","target_value_final_period":85.0,"bucket_id":19,"scenario_story":"underfunded"},
    {"goal_id":2,"metric_name":"Performance Metric - Google Ads Display","metric_unit":"score","target_value_final_period":85.0,"bucket_id":20,"scenario_story":"overfunded"},
    {"goal_id":3,"metric_name":"ROAS - Social",                          "metric_unit":"ratio","target_value_final_period":3.5, "bucket_id":21,"scenario_story":"overfunded"},
])

DEMO_BUCKETS_DATA = pd.DataFrame([
    {"bucket_id":1, "bucket_name":"Marketing",         "bucket_level":1,"parent_bucket_id":None},
    {"bucket_id":5, "bucket_name":"Paid Acquisition",  "bucket_level":2,"parent_bucket_id":1},
    {"bucket_id":19,"bucket_name":"Google Ads Search", "bucket_level":3,"parent_bucket_id":5},
    {"bucket_id":20,"bucket_name":"Google Ads Display","bucket_level":3,"parent_bucket_id":5},
    {"bucket_id":21,"bucket_name":"Social Media Ads",  "bucket_level":3,"parent_bucket_id":5},
])

def render_query_banner(query, results, goals_df):
    tags = "".join(
        f'<span style="font-size:11px;padding:2px 9px;border-radius:20px;background:#f1efe8;'
        f'border:0.5px solid rgba(0,0,0,0.12);color:#5f5e5a;margin-right:5px">'
        f'{get_goal_meta(r["goal_id"], goals_df)["name"]}</span>'
        for r in results
    )
    st.markdown(f'<div class="query-banner"><div class="query-label">Your question</div>'
                f'<div class="query-text">"{query}"</div>'
                f'<div style="margin-top:8px">{tags}</div></div>', unsafe_allow_html=True)

def render_single_goal(r, goals_df, buckets_df, proj_df):
    r    = derive_flags(r)
    meta = get_goal_meta(r["goal_id"], goals_df)
    score = r.get("composite_adjusted", r.get("overall", 0))

    col_l, col_r = st.columns([3, 1])
    with col_l:
        st.markdown(f"### {meta['name']}")
        st.markdown(f"<span style='font-size:12px;color:#888780'>Goal {r['goal_id']} &nbsp;·&nbsp; Target: {meta['target']} {meta['unit']} &nbsp;·&nbsp; Period 12 of 24</span>", unsafe_allow_html=True)
    with col_r:
        st.markdown(f"<div style='text-align:right'><span class='score-xl' style='color:{score_color(score)}'>{score:.2f}</span><br>{status_pill(r)}</div>", unsafe_allow_html=True)

    if r.get("uncertain"):
        st.markdown(f'<div class="alert-warn">GP model uncertainty elevated (std={r.get("gp_std",0):.3f}) - treat attainability score with caution.</div>', unsafe_allow_html=True)

    if buckets_df is not None:
        path = get_bucket_path(meta["bucket_id"], buckets_df)
        if path:
            nodes = ""
            for i, p in enumerate(path):
                cls = "bucket-node-active" if i == len(path)-1 else "bucket-node"
                nodes += f'<span class="{cls}">{p}</span>'
                if i < len(path)-1: nodes += '<span style="color:#888780;font-size:11px"> &rsaquo; </span>'
            st.markdown(f'<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:14px">{nodes}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Coherence dimensions</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, dim in enumerate(DIMS):
        val    = r.get(dim, 0)
        reason = r.get("reasoning", {}).get(dim, "")
        with cols[i]:
            st.markdown(f'<div class="ce-card"><div style="font-size:11px;color:#888780;margin-bottom:4px">{dim.capitalize()}</div>'
                        f'<div class="score-lg" style="color:{score_color(val)};margin-bottom:4px">{val:.2f}</div>'
                        f'<div class="dim-bar-bg"><div class="{score_bar_class(val)}" style="width:{val*100:.0f}%"></div></div>'
                        f'<div style="font-size:11px;color:#888780;line-height:1.5">{reason}</div></div>', unsafe_allow_html=True)

    col_traj, col_llm = st.columns([3, 2])
    with col_traj:
        st.markdown('<div class="section-label">Trajectory</div>', unsafe_allow_html=True)
        current = score
        p6  = r.get("composite_p6",  current * 1.05)
        p12 = r.get("composite_p12", current * 1.10)
        rows = ""
        for lbl, val in [("Now (P12)", current), ("+6 periods", p6), ("+12 (end)", p12)]:
            rows += f'<div class="traj-row"><span class="traj-label">{lbl}</span><div class="traj-barwrap"><div style="height:100%;width:{min(val*100,100):.0f}%;background:{score_color(val)};border-radius:4px"></div></div><span class="traj-val" style="color:{score_color(val)}">{val:.2f}</span></div>'
        rows += f'<div style="font-size:11px;color:#888780;margin-top:4px">Risk threshold: {RISK_THRESHOLD}</div>'
        st.markdown(f'<div class="ce-card">{rows}</div>', unsafe_allow_html=True)

    with col_llm:
        st.markdown('<div class="section-label">LLM model agreement</div>', unsafe_allow_html=True)
        llm_html = ""
        for model, scores in r.get("llm_scores", {}).items():
            if not scores.get("success"): continue
            dots = "".join(f'<span class="llm-dot" style="background:{score_color(scores.get(d,0))}"></span>' for d in DIMS)
            vals = " ".join(f"{d[0].upper()}={scores.get(d,0):.2f}" for d in DIMS)
            llm_html += f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px"><span style="font-size:11px;color:#888780;width:64px">{model}</span>{dots}<span style="font-size:11px;color:#2c2c2a">{vals}</span></div>'
        em   = r.get("ensemble_meta", {}).get("attainability", {})
        gp_w = em.get("gp_weight", r.get("gp_weight", 0.5))
        lw   = em.get("llm_weight", r.get("llm_weight", 0.5))
        llm_html += f'<div style="border-top:0.5px solid rgba(0,0,0,0.08);margin-top:8px;padding-top:10px;display:flex;gap:20px"><div><div style="font-size:11px;color:#888780">GP weight</div><div style="font-size:16px;font-weight:500;color:#2c2c2a">{gp_w:.0%}</div></div><div><div style="font-size:11px;color:#888780">LLM blend</div><div style="font-size:16px;font-weight:500;color:#2c2c2a">{lw:.0%}</div></div></div>'
        st.markdown(f'<div class="ce-card">{llm_html}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Key insights</div>', unsafe_allow_html=True)
    insights = _build_insights([r], goals_df)
    st.markdown('<div class="ce-card">' + "".join(f'<div class="insight-row"><div class="idot" style="background:{c}"></div><span style="font-size:12px;color:#5f5e5a;line-height:1.6">{t}</span></div>' for c,t in insights) + '</div>', unsafe_allow_html=True)

def render_multi_goal(results, goals_df, buckets_df, proj_df):
    results = [derive_flags(r) for r in results]
    has_conflict, conflict_msg = detect_sibling_conflict(results, goals_df, buckets_df)

    shared_parent_id = None
    if goals_df is not None and buckets_df is not None:
        parent_ids = []
        for r in results:
            bid = get_goal_meta(r["goal_id"], goals_df)["bucket_id"]
            if bid is None: continue
            row = buckets_df[buckets_df["bucket_id"] == bid]
            if not row.empty:
                p = row.iloc[0].get("parent_bucket_id")
                if pd.notna(p): parent_ids.append(int(p))
        if len(set(parent_ids)) == 1:
            shared_parent_id = parent_ids[0]

    st.markdown('<div class="section-label">Portfolio overview</div>', unsafe_allow_html=True)

    if shared_parent_id is not None and buckets_df is not None:
        path  = get_bucket_path(shared_parent_id, buckets_df)
        nodes = "".join(f'<span class="bucket-node-active">{p}</span><span style="color:#888780;font-size:11px"> &rsaquo; </span>' for p in path)
        nodes += f'<span class="bucket-node">{len(results)} goals</span>'
        st.markdown(f'<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:14px">{nodes}</div>', unsafe_allow_html=True)

    port_cols = st.columns(len(results))
    for i, r in enumerate(results):
        meta  = get_goal_meta(r["goal_id"], goals_df)
        score = r.get("composite_adjusted", r.get("overall", 0))
        sc    = score_color(score)
        flag  = meta.get("scenario", "???")
        with port_cols[i]:
            st.markdown(f'<div class="ce-card" style="border-left:3px solid {sc}">'
                        f'<div style="font-size:11px;color:#888780;margin-bottom:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{meta["name"]}</div>'
                        f'<div class="score-lg" style="color:{sc};margin-bottom:6px">{score:.2f}</div>'
                        f'<div class="dim-bar-bg"><div class="{score_bar_class(score)}" style="width:{score*100:.0f}%"></div></div>'
                        f'<div style="font-size:11px;color:{flag_color(flag)};margin-bottom:4px">{flag}</div>'
                        f'{status_pill(r)}</div>', unsafe_allow_html=True)

    if has_conflict:
        st.markdown(f'<div class="alert-warn">{conflict_msg}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Goal breakdown</div>', unsafe_allow_html=True)
    tabs = st.tabs([f"Goal {r["goal_id"]} - {get_goal_meta(r["goal_id"], goals_df)["name"][:28]}" for r in results])

    for tab, r in zip(tabs, results):
        with tab:
            meta  = get_goal_meta(r["goal_id"], goals_df)
            score = r.get("composite_adjusted", r.get("overall", 0))
            col_l, col_r = st.columns([3, 1])
            with col_l:
                st.markdown(f"**{meta['name']}**")
                st.markdown(f"<span style='font-size:12px;color:#888780'>Target: {meta['target']} {meta['unit']} &nbsp;·&nbsp; {meta.get('scenario','???')}</span>", unsafe_allow_html=True)
            with col_r:
                st.markdown(f"<div style='text-align:right'><span class='score-lg' style='color:{score_color(score)}'>{score:.2f}</span><br>{status_pill(r)}</div>", unsafe_allow_html=True)
            if r.get("uncertain"):
                st.markdown(f'<div class="alert-warn" style="font-size:11px">GP uncertainty elevated (std={r.get("gp_std",0):.3f}) - treat attainability with caution.</div>', unsafe_allow_html=True)
            dim_cols = st.columns(4)
            for i, dim in enumerate(DIMS):
                val    = r.get(dim, 0)
                reason = r.get("reasoning", {}).get(dim, "")
                with dim_cols[i]:
                    st.markdown(f'<div class="ce-surface"><div style="font-size:10px;color:#888780;margin-bottom:3px">{dim.capitalize()}</div>'
                                f'<div class="score-md" style="color:{score_color(val)};margin-bottom:5px">{val:.2f}</div>'
                                f'<div class="dim-bar-bg"><div class="{score_bar_class(val)}" style="width:{val*100:.0f}%"></div></div>'
                                f'<div style="font-size:10px;color:#888780;line-height:1.5">{reason}</div></div>', unsafe_allow_html=True)
            wk   = r.get("weakest_dim","attainability")
            flag = meta.get("scenario","")
            fund = f"It is currently <strong>{flag}</strong>. " if flag in ("underfunded","overfunded") else ""
            narrative = f"<strong>{meta['name']}</strong> scores {score:.2f} overall. {fund}Weakest dimension: <strong>{wk}</strong> ({r.get(wk,0):.2f}). {r.get('reasoning',{}).get(wk,'')}"
            st.markdown(f'<div class="ce-surface" style="margin-top:4px"><p style="font-size:12px;color:#5f5e5a;line-height:1.7;margin:0">{narrative}</p></div>', unsafe_allow_html=True)

    _render_comparison_table(results, goals_df)

    st.markdown('<div class="section-label">Key insights</div>', unsafe_allow_html=True)
    insights = _build_insights(results, goals_df)
    st.markdown('<div class="ce-card">' + "".join(f'<div class="insight-row"><div class="idot" style="background:{c}"></div><span style="font-size:12px;color:#5f5e5a;line-height:1.6">{t}</span></div>' for c,t in insights) + '</div>', unsafe_allow_html=True)

def _render_comparison_table(results, goals_df):
    st.markdown('<div class="section-label">Side-by-side comparison</div>', unsafe_allow_html=True)
    header = "<tr><th>Goal</th><th>Overall</th>" + "".join(f"<th>{d.capitalize()}</th>" for d in DIMS) + "<th>Flag</th><th>Weakest</th></tr>"
    rows = ""
    for r in results:
        meta    = get_goal_meta(r["goal_id"], goals_df)
        score   = r.get("composite_adjusted", r.get("overall", 0))
        weakest = r.get("weakest_dim", min(DIMS, key=lambda d: r.get(d,1)))
        flag    = meta.get("scenario","???")
        dcells  = "".join(f'<td><span style="color:{score_color(r.get(d,0))};font-weight:500">{r.get(d,0):.2f}</span><span style="display:inline-block;height:4px;width:{int(r.get(d,0)*36)}px;background:{score_color(r.get(d,0))};border-radius:2px;margin-left:5px;vertical-align:middle"></span></td>' for d in DIMS)
        rows += f'<tr><td style="font-weight:500;color:#2c2c2a">{meta["name"][:32]}</td><td style="color:{score_color(score)};font-weight:500">{score:.2f}</td>{dcells}<td style="font-size:11px;color:{flag_color(flag)}">{flag}</td><td style="font-size:11px;color:#888780">{weakest}</td></tr>'
    st.markdown(f'<div class="ce-card" style="overflow-x:auto"><table class="comp-table"><thead>{header}</thead><tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)

def _build_insights(results, goals_df):
    ins = []
    for r in results:
        meta  = get_goal_meta(r["goal_id"], goals_df)
        name  = meta["name"]
        score = r.get("composite_adjusted", r.get("overall", 0))
        if r.get("critical"):   ins.append(("#e24b4a", f"<strong>{name}</strong> is critical ({score:.2f}). Immediate attention required."))
        elif r.get("at_risk"):  ins.append(("#ef9f27", f"<strong>{name}</strong> is at risk ({score:.2f}). Monitor closely."))
        if meta.get("scenario") == "underfunded": ins.append(("#e24b4a", f"<strong>{name}</strong> is underfunded - allocation is below the minimum viable band."))
        elif meta.get("scenario") == "overfunded": ins.append(("#ef9f27", f"<strong>{name}</strong> is overfunded - additional spend is not converting to output gains."))
        if r.get("uncertain"): ins.append(("#185fa5", f"<strong>{name}</strong> has high GP uncertainty - attainability score should be interpreted cautiously."))
    if len(results) > 1:
        under = [r for r in results if get_goal_meta(r["goal_id"], goals_df).get("scenario") == "underfunded"]
        over  = [r for r in results if get_goal_meta(r["goal_id"], goals_df).get("scenario") == "overfunded"]
        if under and over:
            ins.append(("#854f0b", "Budget reallocation opportunity: redistribute funds from overfunded goals to underfunded goals within the same parent bucket."))
    return ins[:6]

def render_follow_up_buttons(results, goals_df):
    st.markdown('<div class="section-label">Follow-up questions</div>', unsafe_allow_html=True)
    suggestions = ["Which goal should we fix first?", "Show me the full Marketing bucket health"]
    for r in results:
        meta = get_goal_meta(r["goal_id"], goals_df)
        suggestions.append(f"What would it take to improve {meta['name']}?")
    if len(results) > 1:
        suggestions.append("Reallocate budget between these goals")
    cols = st.columns(min(len(suggestions), 4))
    for i, (col, q) in enumerate(zip(cols, suggestions[:4])):
        with col:
            if st.button(f"{q} ???", key=f"follow_{i}"):
                st.session_state["prefill_query"] = q
                st.session_state["results"] = None
                # st.switch_page("app.py")  <- uncomment and set to Padmasri's filename

def main():
    query   = st.session_state.get("query",   DEMO_QUERY)
    results = st.session_state.get("results", None)

    if not results:
        st.info("No live data in session - showing demo output. Padmasri: set st.session_state['results'] and st.session_state['query'] before calling st.switch_page('dashboard.py')", icon="i")
        results = DEMO_RESULTS

    goals_df = buckets_df = proj_df = None
    for base in [".", "../v4", "./v4", "./data/sample"]:
        g, b, p = load_support_data(base)
        if g is not None:
            goals_df, buckets_df, proj_df = g, b, p
            break

    if goals_df is None:
        goals_df   = DEMO_GOALS_DATA
        buckets_df = DEMO_BUCKETS_DATA

    if proj_df is not None:
        proj_map = proj_df.set_index("goal_id").to_dict("index")
        for r in results:
            gid = r["goal_id"]
            if gid in proj_map:
                r.setdefault("composite_p6",  proj_map[gid].get("composite_p6",  r.get("overall",0)))
                r.setdefault("composite_p12", proj_map[gid].get("composite_p12", r.get("overall",0)))

    valid = [r for r in results if r.get("status") == "ok"]
    if not valid:
        st.error("No valid scored goals returned from System 3.")
        return

    col_logo, col_new = st.columns([5, 1])
    with col_logo:
        st.markdown('<span style="font-size:13px;font-weight:500;color:#2c2c2a">Coherence Engine</span>&nbsp;&nbsp;<span style="font-size:11px;color:#888780">Period 12 of 24 &nbsp;&#183;&nbsp; System 1 Output</span>', unsafe_allow_html=True)
    with col_new:
        if st.button("New query"):
            st.session_state["results"] = None
            st.session_state["query"]   = ""
            # st.switch_page("app.py")  <- uncomment and set to Padmasri's filename

    st.divider()
    render_query_banner(query, valid, goals_df)

    if len(valid) == 1:
        render_single_goal(valid[0], goals_df, buckets_df, proj_df)
    else:
        render_multi_goal(valid, goals_df, buckets_df, proj_df)

    st.divider()
    render_follow_up_buttons(valid, goals_df)

if __name__ == "__main__":
    main()
'@

$dashboardContent | Set-Content "$REPO\dashboard.py" -Encoding UTF8

Write-Host "  dashboard.py written" -ForegroundColor Green

# ── 9. Initialise git repo ────────────────────────────────────────────────────
Set-Location $REPO
git init -q
git config user.email "subhan@team1402.com"
git config user.name "Subhan Farid"
git add .
git commit -q -m "Initial commit - System 1 output dashboard (Subhan)"
Set-Location ..

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Done. Repo created: .\$REPO" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  1. cd $REPO"
Write-Host "  2. pip install -r requirements.txt"
Write-Host "  3. streamlit run dashboard.py"
Write-Host ""
Write-Host "  4. Push to GitHub:" -ForegroundColor Yellow
Write-Host "     git remote add origin https://github.com/YOUR_USERNAME/$REPO.git"
Write-Host "     git branch -M main"
Write-Host "     git push -u origin main"
Write-Host ""
Write-Host "  5. Copy goals.csv and buckets.csv from System 2 into .\$REPO\"
Write-Host "     for richer goal names and bucket paths."
Write-Host ""
Write-Host "  6. In dashboard.py, uncomment the two st.switch_page() lines"
Write-Host "     and set the filename to match Padmasri's entry file."
Write-Host ""
