"""
dashboard.py — Coherence Engine Output Dashboard
System 1 | Subhan Farid | Team 14-02

Entry point:
    streamlit run dashboard.py

Integration contract:
    This dashboard expects st.session_state to be populated by Padmasri's
    input layer (conversation layer) before navigating here.

    Required session_state keys:
        st.session_state["query"]        str   — original user question
        st.session_state["results"]      list  — list of score payloads from
                                                  System 3 / 07_score_goal.py
                                                  (one dict per goal)

    Each result dict (from score_goal() in 07_score_goal.py) must contain:
        goal_id         int
        attainability   float 0-1
        relevance       float 0-1
        coherence       float 0-1
        integrity       float 0-1
        overall         float 0-1
        gp_std          float
        uncertain       bool
        reasoning       dict  {attainability, relevance, coherence, integrity}
        llm_scores      dict  {llama3: {attainability,...}, gemma3:..., nemotron:...}
        ensemble_meta   dict
        status          str   "ok" | "error"

    Plus from composite_scores_poc.csv (merged in by System 3 or loaded here):
        weakest_dim     str
        at_risk         bool
        critical        bool
        composite_adjusted float

    Plus from forward_projection_poc.csv (loaded here from System 2 outputs):
        composite_p6    float
        composite_p12   float
        improving_p6    bool
        degrading_p6    bool

    Standalone demo mode (no session state):
        Run as-is — demo data is injected automatically.

Navigation:
    Called from Padmasri's input screen via:
        st.session_state["results"] = results
        st.session_state["query"]   = user_query
        st.switch_page("dashboard.py")

    Returns to input screen via the "New query" button which calls:
        st.switch_page("app.py")   <- adjust to match Padmasri's entry filename
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Coherence Engine",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Stylesheet ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0e0e0f; }
[data-testid="stAppViewBlockContainer"] { padding-top: 1.5rem; max-width: 1100px; }
[data-testid="stSidebar"] { display: none; }
h1,h2,h3 { font-weight: 500 !important; }
.ce-card {
    background: #1f1f20; border: 0.5px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 16px 18px; margin-bottom: 12px;
}
.ce-surface {
    background: #1a1a1b; border-radius: 8px;
    padding: 12px 14px; margin-bottom: 10px;
}
.query-banner {
    background: #1f1f20; border: 0.5px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 14px 18px; margin-bottom: 20px;
}
.query-label { font-size: 11px; color: #6b6967; margin-bottom: 4px; }
.query-text  { font-size: 14px; font-weight: 500; color: #e8e6e0; }
.score-xl   { font-size: 38px; font-weight: 500; line-height: 1; }
.score-lg   { font-size: 26px; font-weight: 500; line-height: 1; }
.score-md   { font-size: 18px; font-weight: 500; line-height: 1; }
.pill { display: inline-block; font-size: 11px; padding: 3px 10px; border-radius: 20px; font-weight: 500; }
.pill-crit { background: #fcebeb; color: #a32d2d; }
.pill-risk { background: #faeeda; color: #854f0b; }
.pill-ok   { background: #eaf3de; color: #3b6d11; }
.pill-info { background: #0c1e36; color: #185fa5; }
.dim-bar-bg  { height: 5px; background: #1a1a1b; border-radius: 3px; overflow: hidden; margin: 6px 0; }
.dim-bar-red    { height: 5px; background: #e24b4a; border-radius: 3px; }
.dim-bar-amber  { height: 5px; background: #ef9f27; border-radius: 3px; }
.dim-bar-green  { height: 5px; background: #1d9e75; border-radius: 3px; }
.alert-warn {
    background: #1e1800; border: 0.5px solid #ba7517;
    border-radius: 8px; padding: 10px 14px;
    font-size: 12px; color: #ef9f27; margin-bottom: 12px; line-height: 1.6;
}
.alert-danger {
    background: #1e0000; border: 0.5px solid #a32d2d;
    border-radius: 8px; padding: 10px 14px;
    font-size: 12px; color: #f09595; margin-bottom: 12px; line-height: 1.6;
}
.alert-info {
    background: #0c1e36; border: 0.5px solid #185fa5;
    border-radius: 8px; padding: 10px 14px;
    font-size: 12px; color: #042c53; margin-bottom: 12px; line-height: 1.6;
}
.comp-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.comp-table th {
    font-size: 11px; font-weight: 500; color: #6b6967;
    padding: 8px 10px; text-align: left;
    border-bottom: 0.5px solid rgba(255,255,255,0.08);
}
.comp-table td { padding: 9px 10px; border-bottom: 0.5px solid rgba(255,255,255,0.06); }
.comp-table tr:last-child td { border-bottom: none; }
.section-label {
    font-size: 11px; font-weight: 500; color: #6b6967;
    text-transform: uppercase; letter-spacing: .06em; margin: 16px 0 8px 0;
}
.bucket-path { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 14px; }
.bucket-node {
    font-size: 11px; padding: 3px 9px; border-radius: 20px;
    border: 0.5px solid rgba(255,255,255,0.10); color: #6b6967; background: #1a1a1b;
}
.bucket-node-active {
    font-size: 11px; padding: 3px 9px; border-radius: 20px;
    background: #0c1e36; border: 0.5px solid #185fa5; color: #185fa5;
}
.insight-row { display: flex; gap: 10px; align-items: flex-start; margin-bottom: 9px; }
.idot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; margin-top: 5px; }
.llm-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 2px; }
.traj-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.traj-label { font-size: 12px; color: #6b6967; width: 70px; flex-shrink: 0; }
.traj-barwrap { flex: 1; height: 7px; background: #1a1a1b; border-radius: 4px; overflow: hidden; }
.traj-val { font-size: 12px; font-weight: 500; width: 36px; text-align: right; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DIMS               = ["attainability", "relevance", "coherence", "integrity"]
WEIGHTS            = {"coherence": 0.35, "attainability": 0.25, "relevance": 0.20, "integrity": 0.20}
RISK_THRESHOLD     = 0.35
CRITICAL_THRESHOLD = 0.20

# ── Helpers ───────────────────────────────────────────────────────────────────
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
        return {"name": f"Goal {goal_id}", "target": "—", "unit": "", "bucket_id": None, "scenario": "—"}
    row = goals_df[goals_df["goal_id"] == goal_id]
    if row.empty:
        return {"name": f"Goal {goal_id}", "target": "—", "unit": "", "bucket_id": None, "scenario": "—"}
    row = row.iloc[0]
    return {
        "name":      row.get("metric_name", f"Goal {goal_id}"),
        "target":    row.get("target_value_final_period", "—"),
        "unit":      row.get("metric_unit", ""),
        "bucket_id": row.get("bucket_id"),
        "scenario":  row.get("scenario_story", "—"),
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
    shared_parent = parents[0] if len(parents) == 1 else None
    scenarios = [get_goal_meta(r["goal_id"], goals_df).get("scenario", "") for r in results]
    has_under = any("underfunded" in s for s in scenarios)
    has_over  = any("overfunded"  in s for s in scenarios)
    if shared_parent and has_under and has_over:
        parent_name = ""
        if buckets_df is not None:
            prow = buckets_df[buckets_df["bucket_id"] == shared_parent]
            if not prow.empty:
                parent_name = prow.iloc[0]["bucket_name"]
        msg = (f"⚠️ Inter-goal coherence issue detected under <strong>{parent_name}</strong>. "
               f"One or more goals are underfunded while others are overfunded within the same "
               f"parent bucket. Budget is unevenly distributed across sibling goals — "
               f"this is suppressing the overall bucket coherence score.")
        return True, msg
    return False, ""

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_support_data(base_dir="."):
    goals_df = buckets_df = proj_df = None
    for fname, attr in [
        ("goals.csv",                  "goals_df"),
        ("buckets.csv",                "buckets_df"),
        ("forward_projection_poc.csv", "proj_df"),
    ]:
        path = os.path.join(base_dir, fname)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if attr == "goals_df":   goals_df   = df
                if attr == "buckets_df": buckets_df = df
                if attr == "proj_df":    proj_df    = df
            except Exception:
                pass
    return goals_df, buckets_df, proj_df

# ── Demo data ─────────────────────────────────────────────────────────────────
DEMO_QUERY = "How are our paid acquisition channels performing? Are Google Ads and Social ROAS on track?"

DEMO_RESULTS = [
    {
        "goal_id": 1, "attainability": 0.22, "relevance": 0.19, "coherence": 0.31, "integrity": 0.41,
        "overall": 0.28, "composite_adjusted": 0.28, "gp_mean": 0.24, "gp_std": 0.041,
        "gp_weight": 0.72, "llm_weight": 0.28, "baseline": 0.21, "uncertain": False,
        "at_risk": True, "critical": True, "weakest_dim": "relevance",
        "composite_p6": 0.30, "composite_p12": 0.33, "improving_p6": True, "degrading_p6": False,
        "reasoning": {
            "attainability": "At 21.9/85, this goal is 26% of target at the halfway point. Trajectory does not support reaching 85 by period 24.",
            "relevance":     "Allocation is below the minimum viable band — classic underfunding. Receives least budget in the Paid Acquisition bucket.",
            "coherence":     "Consistently low allocation across all 12 periods. Drift is low but baseline is wrong — goal has never been adequately funded.",
            "integrity":     "Output quality (0.68) is reasonable but quantity is suppressed by the funding gap. Efficiency ratio of 0.26 is the best in the bucket."
        },
        "llm_scores": {
            "llama3":   {"attainability": 0.21, "relevance": 0.18, "coherence": 0.30, "integrity": 0.40, "success": True},
            "gemma3":   {"attainability": 0.23, "relevance": 0.20, "coherence": 0.33, "integrity": 0.43, "success": True},
            "nemotron": {"attainability": 0.22, "relevance": 0.19, "coherence": 0.29, "integrity": 0.39, "success": True},
        },
        "ensemble_meta": {"attainability": {"gp_std": 0.041, "gp_weight": 0.72, "llm_weight": 0.28, "uncertain": False}},
        "n_llm_ok": 3, "status": "ok",
    },
    {
        "goal_id": 2, "attainability": 0.18, "relevance": 0.21, "coherence": 0.28, "integrity": 0.30,
        "overall": 0.24, "composite_adjusted": 0.24, "gp_mean": 0.20, "gp_std": 0.088,
        "gp_weight": 0.38, "llm_weight": 0.62, "baseline": 0.18, "uncertain": True,
        "at_risk": True, "critical": True, "weakest_dim": "attainability",
        "composite_p6": 0.22, "composite_p12": 0.21, "improving_p6": False, "degrading_p6": True,
        "reasoning": {
            "attainability": "At 19.8/85, this is the worst performing goal in the bucket. Current slope is flat — projecting only ~22 by period 24.",
            "relevance":     "Allocation is in the red_high band — significantly overfunded. Budget is being spent but not converting to performance.",
            "coherence":     "Despite high allocation, output gains are minimal. Spend and outcome are structurally disconnected across periods.",
            "integrity":     "Efficiency ratio of 0.10 is the lowest in the bucket. Delivered output is well below expected given the funding level."
        },
        "llm_scores": {
            "llama3":   {"attainability": 0.17, "relevance": 0.22, "coherence": 0.27, "integrity": 0.29, "success": True},
            "gemma3":   {"attainability": 0.19, "relevance": 0.20, "coherence": 0.30, "integrity": 0.32, "success": True},
            "nemotron": {"attainability": 0.18, "relevance": 0.21, "coherence": 0.27, "integrity": 0.29, "success": True},
        },
        "ensemble_meta": {"attainability": {"gp_std": 0.088, "gp_weight": 0.38, "llm_weight": 0.62, "uncertain": True}},
        "n_llm_ok": 3, "status": "ok",
    },
    {
        "goal_id": 3, "attainability": 0.18, "relevance": 0.29, "coherence": 0.36, "integrity": 0.41,
        "overall": 0.31, "composite_adjusted": 0.31, "gp_mean": 0.21, "gp_std": 0.062,
        "gp_weight": 0.55, "llm_weight": 0.45, "baseline": 0.19, "uncertain": False,
        "at_risk": True, "critical": True, "weakest_dim": "attainability",
        "composite_p6": 0.33, "composite_p12": 0.36, "improving_p6": True, "degrading_p6": False,
        "reasoning": {
            "attainability": "Current ROAS of 0.91 against a target of 3.5. At current slope, projected ROAS by period 24 is approximately 1.4.",
            "relevance":     "Sitting in orange_high — overfunded relative to current output gains. Sibling rank is mid-range.",
            "coherence":     "Allocation drift is moderate. Partially inconsistent with L2 bucket pattern across periods.",
            "integrity":     "Output quality is the highest in the bucket (0.90) but needle move ratio of 0.33 suggests poor spend-to-output conversion."
        },
        "llm_scores": {
            "llama3":   {"attainability": 0.17, "relevance": 0.30, "coherence": 0.35, "integrity": 0.40, "success": True},
            "gemma3":   {"attainability": 0.19, "relevance": 0.28, "coherence": 0.37, "integrity": 0.43, "success": True},
            "nemotron": {"attainability": 0.18, "relevance": 0.29, "coherence": 0.35, "integrity": 0.40, "success": True},
        },
        "ensemble_meta": {"attainability": {"gp_std": 0.062, "gp_weight": 0.55, "llm_weight": 0.45, "uncertain": False}},
        "n_llm_ok": 3, "status": "ok",
    },
]

DEMO_GOALS_DATA = pd.DataFrame([
    {"goal_id": 1, "metric_name": "Performance Metric — Google Ads Search",  "metric_unit": "score",  "target_value_final_period": 85.0, "bucket_id": 19, "scenario_story": "underfunded"},
    {"goal_id": 2, "metric_name": "Performance Metric — Google Ads Display", "metric_unit": "score",  "target_value_final_period": 85.0, "bucket_id": 20, "scenario_story": "overfunded"},
    {"goal_id": 3, "metric_name": "ROAS — Social",                           "metric_unit": "ratio",  "target_value_final_period": 3.5,  "bucket_id": 21, "scenario_story": "overfunded"},
])

DEMO_BUCKETS_DATA = pd.DataFrame([
    {"bucket_id": 1,  "bucket_name": "Marketing",         "bucket_level": 1, "parent_bucket_id": None},
    {"bucket_id": 5,  "bucket_name": "Paid Acquisition",  "bucket_level": 2, "parent_bucket_id": 1},
    {"bucket_id": 19, "bucket_name": "Google Ads Search", "bucket_level": 3, "parent_bucket_id": 5},
    {"bucket_id": 20, "bucket_name": "Google Ads Display","bucket_level": 3, "parent_bucket_id": 5},
    {"bucket_id": 21, "bucket_name": "Social Media Ads",  "bucket_level": 3, "parent_bucket_id": 5},
])

# ── Render functions ──────────────────────────────────────────────────────────
def render_query_banner(query, results, goals_df):
    tags = "".join(
        f'<span style="font-size:11px;padding:2px 9px;border-radius:20px;background:#1a1a1b;'
        f'border:0.5px solid rgba(255,255,255,0.10);color:#a8a6a0;margin-right:5px">'
        f'{get_goal_meta(r["goal_id"], goals_df)["name"]}</span>'
        for r in results
    )
    st.markdown(f"""
    <div class="query-banner">
        <div class="query-label">Your question</div>
        <div class="query-text">"{query}"</div>
        <div style="margin-top:8px">{tags}</div>
    </div>""", unsafe_allow_html=True)


def render_single_goal(r, goals_df, buckets_df, proj_df, api_key=""):
    r = derive_flags(r)
    meta = get_goal_meta(r["goal_id"], goals_df)

    col_left, col_right = st.columns([3, 1])
    with col_left:
        st.markdown(f"### {meta['name']}")
        st.markdown(f"<span style='font-size:12px;color:#6b6967'>Goal {r['goal_id']} &nbsp;·&nbsp; Target: {meta['target']} {meta['unit']} by period 24 &nbsp;·&nbsp; Period 12 of 24</span>", unsafe_allow_html=True)
    with col_right:
        score = r.get("composite_adjusted", r.get("overall", 0))
        st.markdown(f"<div style='text-align:right'><span class='score-xl' style='color:{score_color(score)}'>{score:.2f}</span><br>{status_pill(r)}</div>", unsafe_allow_html=True)

    if r.get("uncertain"):
        st.markdown(f'<div class="alert-warn">⚠️ <strong>GP model uncertainty is elevated</strong> (std = {r.get("gp_std",0):.3f}). Attainability score blends model and LLM estimates — treat with caution.</div>', unsafe_allow_html=True)

    if buckets_df is not None:
        path = get_bucket_path(meta["bucket_id"], buckets_df)
        if path:
            nodes = ""
            for i, p in enumerate(path):
                cls = "bucket-node-active" if i == len(path) - 1 else "bucket-node"
                nodes += f'<span class="{cls}">{p}</span>'
                if i < len(path) - 1: nodes += '<span style="color:#6b6967;font-size:11px"> › </span>'
            st.markdown(f'<div class="bucket-path">{nodes}</div>', unsafe_allow_html=True)

    # ── AI Summary ──
    st.markdown('<div class="section-label">AI summary</div>', unsafe_allow_html=True)
    render_llm_summary(r, meta, api_key)

    st.markdown('<div class="section-label">Coherence dimensions</div>', unsafe_allow_html=True)
    dim_cols = st.columns(4)
    for i, dim in enumerate(DIMS):
        val    = r.get(dim, 0)
        reason = r.get("reasoning", {}).get(dim, "")
        with dim_cols[i]:
            st.markdown(f"""
            <div class="ce-card">
                <div style="font-size:11px;color:#6b6967;margin-bottom:4px">{dim.capitalize()}</div>
                <div class="score-lg" style="color:{score_color(val)};margin-bottom:4px">{val:.2f}</div>
                <div class="dim-bar-bg"><div class="{score_bar_class(val)}" style="width:{val*100:.0f}%"></div></div>
                <div style="font-size:11px;color:#6b6967;line-height:1.5">{reason}</div>
            </div>""", unsafe_allow_html=True)

    # ── Charts row: radar + trajectory ──
    col_radar, col_traj = st.columns(2)
    with col_radar:
        st.markdown('<div class="section-label">Dimension profile</div>', unsafe_allow_html=True)
        st.markdown('<div class="ce-card">', unsafe_allow_html=True)
        render_radar_chart(r)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_traj:
        st.markdown('<div class="section-label">Trajectory</div>', unsafe_allow_html=True)
        st.markdown('<div class="ce-card">', unsafe_allow_html=True)
        render_trajectory_chart(r)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── LLM agreement chart ──
    st.markdown('<div class="section-label">LLM model agreement</div>', unsafe_allow_html=True)
    col_llmchart, col_llmmeta = st.columns([3, 1])
    with col_llmchart:
        st.markdown('<div class="ce-card">', unsafe_allow_html=True)
        render_llm_chart(r)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_llmmeta:
        em    = r.get("ensemble_meta", {}).get("attainability", {})
        gp_w  = em.get("gp_weight",  r.get("gp_weight",  0.5))
        llm_w = em.get("llm_weight", r.get("llm_weight", 0.5))
        st.markdown(f"""
        <div class="ce-card" style="height:100%">
            <div style="font-size:11px;color:#6b6967;margin-bottom:12px">Ensemble weights<br>(attainability)</div>
            <div style="margin-bottom:14px">
                <div style="font-size:11px;color:#6b6967">GP model</div>
                <div style="font-size:22px;font-weight:500;color:#e8e6e0">{gp_w:.0%}</div>
            </div>
            <div>
                <div style="font-size:11px;color:#6b6967">LLM blend</div>
                <div style="font-size:22px;font-weight:500;color:#e8e6e0">{llm_w:.0%}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Key insights</div>', unsafe_allow_html=True)
    insights = _build_insights([r], goals_df)
    st.markdown(
        f'<div class="ce-card">' +
        "".join(f'<div class="insight-row"><div class="idot" style="background:{c}"></div><span style="font-size:12px;color:#a8a6a0;line-height:1.6">{t}</span></div>' for c, t in insights) +
        '</div>', unsafe_allow_html=True
    )


def render_multi_goal(results, goals_df, buckets_df, proj_df, api_key=""):
    results = [derive_flags(r) for r in results]
    has_conflict, conflict_msg = detect_sibling_conflict(results, goals_df, buckets_df)

    shared_parent_id = None
    if goals_df is not None and buckets_df is not None:
        bucket_ids = [get_goal_meta(r["goal_id"], goals_df)["bucket_id"] for r in results]
        parent_ids = []
        for bid in bucket_ids:
            if bid is None: continue
            row = buckets_df[buckets_df["bucket_id"] == bid]
            if not row.empty:
                p = row.iloc[0].get("parent_bucket_id")
                if pd.notna(p): parent_ids.append(int(p))
        if len(set(parent_ids)) == 1:
            shared_parent_id = parent_ids[0]

    st.markdown('<div class="section-label">Portfolio overview</div>', unsafe_allow_html=True)

    if shared_parent_id is not None and buckets_df is not None:
        path = get_bucket_path(shared_parent_id, buckets_df)
        nodes = "".join(f'<span class="bucket-node-active">{p}</span><span style="color:#6b6967;font-size:11px"> › </span>' for p in path)
        nodes += f'<span class="bucket-node">{len(results)} goals</span>'
        st.markdown(f'<div class="bucket-path">{nodes}</div>', unsafe_allow_html=True)

    port_cols = st.columns(len(results))
    for i, r in enumerate(results):
        meta  = get_goal_meta(r["goal_id"], goals_df)
        score = r.get("composite_adjusted", r.get("overall", 0))
        sc    = score_color(score)
        flag  = meta.get("scenario", "—")
        with port_cols[i]:
            st.markdown(f"""
            <div class="ce-card" style="border-left:3px solid {sc}">
                <div style="font-size:11px;color:#6b6967;margin-bottom:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{meta["name"]}</div>
                <div class="score-lg" style="color:{sc};margin-bottom:6px">{score:.2f}</div>
                <div class="dim-bar-bg"><div class="{score_bar_class(score)}" style="width:{score*100:.0f}%"></div></div>
                <div style="font-size:11px;color:{flag_color(flag)};margin-bottom:4px">{flag}</div>
                {status_pill(r)}
            </div>""", unsafe_allow_html=True)

    if has_conflict:
        st.markdown(f'<div class="alert-warn">{conflict_msg}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Goal breakdown</div>', unsafe_allow_html=True)
    tab_labels = [f"Goal {r['goal_id']} — {get_goal_meta(r['goal_id'], goals_df)['name'][:28]}" for r in results]
    tabs = st.tabs(tab_labels)

    for tab, r in zip(tabs, results):
        with tab:
            meta  = get_goal_meta(r["goal_id"], goals_df)
            score = r.get("composite_adjusted", r.get("overall", 0))
            col_l, col_r = st.columns([3, 1])
            with col_l:
                st.markdown(f"**{meta['name']}**")
                st.markdown(f"<span style='font-size:12px;color:#6b6967'>Target: {meta['target']} {meta['unit']} &nbsp;·&nbsp; {meta.get('scenario','—')}</span>", unsafe_allow_html=True)
            with col_r:
                st.markdown(f"<div style='text-align:right'><span class='score-lg' style='color:{score_color(score)}'>{score:.2f}</span><br>{status_pill(r)}</div>", unsafe_allow_html=True)

            if r.get("uncertain"):
                st.markdown(f'<div class="alert-warn" style="font-size:11px">⚠️ GP uncertainty elevated (std = {r.get("gp_std",0):.3f}) — treat attainability score with caution.</div>', unsafe_allow_html=True)

            dim_cols = st.columns(4)
            for i, dim in enumerate(DIMS):
                val    = r.get(dim, 0)
                reason = r.get("reasoning", {}).get(dim, "")
                with dim_cols[i]:
                    st.markdown(f"""
                    <div class="ce-surface">
                        <div style="font-size:10px;color:#6b6967;margin-bottom:3px">{dim.capitalize()}</div>
                        <div class="score-md" style="color:{score_color(val)};margin-bottom:5px">{val:.2f}</div>
                        <div class="dim-bar-bg"><div class="{score_bar_class(val)}" style="width:{val*100:.0f}%"></div></div>
                        <div style="font-size:10px;color:#6b6967;line-height:1.5">{reason}</div>
                    </div>""", unsafe_allow_html=True)

            weakest  = r.get("weakest_dim", "attainability")
            wk_score = r.get(weakest, 0)
            flag     = meta.get("scenario", "")
            fund     = f"It is currently <strong>{flag}</strong>. " if flag in ("underfunded", "overfunded") else ""
            narrative = (f"<strong>{meta['name']}</strong> scores {score:.2f} overall. "
                         f"{fund}The weakest dimension is <strong>{weakest}</strong> ({wk_score:.2f}). "
                         f"{r.get('reasoning', {}).get(weakest, '')}")
            st.markdown(f'<div class="ce-surface" style="margin-top:4px"><p style="font-size:12px;color:#a8a6a0;line-height:1.7;margin:0">{narrative}</p></div>', unsafe_allow_html=True)

    # ── Multi-goal dimension chart ──
    st.markdown('<div class="section-label">Dimension comparison — all goals</div>', unsafe_allow_html=True)
    st.markdown('<div class="ce-card">', unsafe_allow_html=True)
    render_multi_goal_chart(results, goals_df)
    st.markdown('</div>', unsafe_allow_html=True)

    _render_comparison_table(results, goals_df)

    # ── AI Summaries ──
    render_multi_summary(results, goals_df, api_key)

    st.markdown('<div class="section-label">Key insights</div>', unsafe_allow_html=True)
    insights = _build_insights(results, goals_df)
    st.markdown(
        '<div class="ce-card">' +
        "".join(f'<div class="insight-row"><div class="idot" style="background:{c}"></div><span style="font-size:12px;color:#a8a6a0;line-height:1.6">{t}</span></div>' for c, t in insights) +
        '</div>', unsafe_allow_html=True
    )


def _render_comparison_table(results, goals_df):
    st.markdown('<div class="section-label">Side-by-side comparison</div>', unsafe_allow_html=True)
    header = "<tr><th>Goal</th><th>Overall</th>" + "".join(f"<th>{d.capitalize()}</th>" for d in DIMS) + "<th>Flag</th><th>Weakest</th></tr>"
    rows = ""
    for r in results:
        meta    = get_goal_meta(r["goal_id"], goals_df)
        score   = r.get("composite_adjusted", r.get("overall", 0))
        weakest = r.get("weakest_dim", min(DIMS, key=lambda d: r.get(d, 1)))
        flag    = meta.get("scenario", "—")
        dim_cells = "".join(
            f'<td><span style="color:{score_color(r.get(d,0))};font-weight:500">{r.get(d,0):.2f}</span>'
            f'<span style="display:inline-block;height:4px;width:{int(r.get(d,0)*36)}px;background:{score_color(r.get(d,0))};border-radius:2px;margin-left:5px;vertical-align:middle"></span></td>'
            for d in DIMS
        )
        rows += f'<tr><td style="font-weight:500;color:#e8e6e0">{meta["name"][:32]}</td><td style="color:{score_color(score)};font-weight:500">{score:.2f}</td>{dim_cells}<td style="font-size:11px;color:{flag_color(flag)}">{flag}</td><td style="font-size:11px;color:#6b6967">{weakest}</td></tr>'
    st.markdown(f'<div class="ce-card" style="overflow-x:auto"><table class="comp-table"><thead>{header}</thead><tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)


# ── LLM Summary (Groq) ────────────────────────────────────────────────────────

def build_summary_prompt(r: dict, meta: dict) -> str:
    """
    Builds the prompt sent to the LLM for a single goal.
    All the hard analytical work is already done by System 2 —
    the LLM here just converts it into plain English for a manager.
    """
    score    = r.get("composite_adjusted", r.get("overall", 0))
    status   = "critical" if r.get("critical") else "at risk" if r.get("at_risk") else "on track"
    flag     = meta.get("scenario", "unknown")
    weakest  = r.get("weakest_dim", "attainability")
    uncertain_note = (
        "Note: the GP model uncertainty is elevated for this goal — "
        "attainability score should be treated as indicative, not definitive. "
        if r.get("uncertain") else ""
    )
    reasoning = r.get("reasoning", {})

    return f"""You are writing a plain-English goal health summary for a senior manager.
The analytical scoring has already been done — your job is to communicate it clearly.

GOAL: {meta['name']}
TARGET: {meta['target']} {meta['unit']} by period 24 (currently at period 12)
FUNDING STATUS: {flag}
OVERALL COHERENCE SCORE: {score:.2f} out of 1.0 (threshold for healthy = 0.50, at-risk = below 0.35)
STATUS: {status}
{uncertain_note}
DIMENSION SCORES AND ANALYSIS:
- Attainability ({r.get('attainability', 0):.2f}): {reasoning.get('attainability', '')}
- Relevance     ({r.get('relevance',     0):.2f}): {reasoning.get('relevance',     '')}
- Coherence     ({r.get('coherence',     0):.2f}): {reasoning.get('coherence',     '')}
- Integrity     ({r.get('integrity',     0):.2f}): {reasoning.get('integrity',     '')}

WEAKEST DIMENSION: {weakest} ({r.get(weakest, 0):.2f})

Write a summary of exactly 3 sentences for a senior manager:
1. State whether the goal is on track and give the overall score.
2. Identify the most critical problem and what is causing it.
3. Give one specific, actionable recommendation.

Rules:
- No bullet points. Flowing prose only.
- If the goal is underfunded, say so directly and recommend increasing allocation.
- If the goal is overfunded, say the problem is not budget but execution or strategy.
- If uncertain is true, use hedged language: "the data suggests", "may indicate".
- Do not repeat the score numbers already stated in sentence 1.
- Maximum 80 words total.
"""


@st.cache_data(ttl=300, show_spinner=False)
def generate_llm_summary(goal_id: int, score: float, flag: str,
                          attain: float, rel: float, coh: float, integ: float,
                          prompt: str, api_key: str) -> str:
    """
    Calls Groq API to generate a plain-English summary.
    Cached for 5 minutes — same goal won't re-call the API on reruns.
    Uses the official groq package: pip install groq
    """
    if not api_key:
        return (
            "⚠️ No GROQ_API_KEY found. Add it to a `.env` file in the repo root: "
            "`GROQ_API_KEY=gsk_...` then restart Streamlit."
        )

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    except ImportError:
        # Fallback to requests if groq package not installed
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
                "User-Agent":    "Mozilla/5.0",
            }
            payload = {
                "model":       "llama-3.3-70b-versatile",
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  200,
                "temperature": 0.3,
            }
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except ImportError:
            return "⚠️ Install the groq package: pip install groq"
        except Exception as e:
            return f"⚠️ API error: {str(e)[:120]}"

    except Exception as e:
        err = str(e)
        if "401" in err or "invalid_api_key" in err.lower():
            return "⚠️ Invalid GROQ_API_KEY — check your .env file."
        if "429" in err:
            return "⚠️ Groq rate limit hit — try again in a few seconds."
        if "403" in err or "1010" in err:
            return "⚠️ Network blocked Groq API (error 403). Try on a different network or hotspot."
        return f"⚠️ Groq error: {err[:120]}"


def render_llm_summary(r: dict, meta: dict, api_key: str):
    """
    Renders the AI summary card for a single goal.
    Shows a spinner while generating, then displays the result.
    """
    prompt = build_summary_prompt(r, meta)
    score  = r.get("composite_adjusted", r.get("overall", 0))

    with st.spinner("Generating summary..."):
        summary = generate_llm_summary(
            goal_id = r["goal_id"],
            score   = round(score, 4),
            flag    = meta.get("scenario", ""),
            attain  = round(r.get("attainability", 0), 4),
            rel     = round(r.get("relevance",     0), 4),
            coh     = round(r.get("coherence",     0), 4),
            integ   = round(r.get("integrity",     0), 4),
            prompt  = prompt,
            api_key = api_key,
        )

    is_error = summary.startswith("⚠️")
    bg       = "#2a1f1f" if is_error else "#1a2030"
    border   = "#a32d2d" if is_error else "#185fa5"
    label    = "Error" if is_error else "AI summary"
    label_c  = "#e24b4a" if is_error else "#378add"

    st.markdown(f"""
    <div style="background:{bg};border:0.5px solid {border};border-radius:12px;
                padding:16px 18px;margin-bottom:12px">
        <div style="font-size:11px;font-weight:500;color:{label_c};
                    text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">
            {label}
        </div>
        <p style="font-size:13px;color:#e8e6e0;line-height:1.75;margin:0">
            {summary}
        </p>
    </div>""", unsafe_allow_html=True)


def render_multi_summary(results: list, goals_df, api_key: str):
    """
    For multi-goal queries, generates one summary per goal in columns,
    plus a cross-goal summary if goals share a parent bucket.
    """
    # Per-goal summaries in tabs
    st.markdown('<div class="section-label">AI summaries</div>',
                unsafe_allow_html=True)

    tab_labels = [
        f"Goal {r['goal_id']} — {get_goal_meta(r['goal_id'], goals_df)['name'][:20]}"
        for r in results
    ]
    tabs = st.tabs(tab_labels)
    for tab, r in zip(tabs, results):
        with tab:
            meta = get_goal_meta(r["goal_id"], goals_df)
            render_llm_summary(r, meta, api_key)

    # Cross-goal reallocation prompt if conflict exists
    under = [r for r in results
             if get_goal_meta(r["goal_id"], goals_df).get("scenario") == "underfunded"]
    over  = [r for r in results
             if get_goal_meta(r["goal_id"], goals_df).get("scenario") == "overfunded"]

    if under and over:
        under_names = ", ".join(get_goal_meta(r["goal_id"], goals_df)["name"] for r in under)
        over_names  = ", ".join(get_goal_meta(r["goal_id"], goals_df)["name"] for r in over)
        cross_prompt = f"""You are advising a senior manager on budget reallocation.

Underfunded goals (need more budget): {under_names}
Overfunded goals (budget not converting to output): {over_names}

These goals share the same parent budget bucket.

Write 2 sentences only:
1. Explain the imbalance and why it matters.
2. Give one specific reallocation recommendation.
Maximum 60 words. Plain English, no bullet points."""

        with st.spinner("Generating reallocation recommendation..."):
            cross_summary = generate_llm_summary(
                goal_id = -1,
                score   = 0.0,
                flag    = "cross-goal",
                attain  = 0.0, rel=0.0, coh=0.0, integ=0.0,
                prompt  = cross_prompt,
                api_key = api_key,
            )

        st.markdown(f"""
        <div style="background:#1e1e10;border:0.5px solid #ba7517;border-radius:12px;
                    padding:16px 18px;margin-top:8px">
            <div style="font-size:11px;font-weight:500;color:#ef9f27;
                        text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">
                Reallocation recommendation
            </div>
            <p style="font-size:13px;color:#e8e6e0;line-height:1.75;margin:0">
                {cross_summary}
            </p>
        </div>""", unsafe_allow_html=True)


def _build_insights(results, goals_df):
    insights = []
    for r in results:
        meta  = get_goal_meta(r["goal_id"], goals_df)
        name  = meta["name"]
        score = r.get("composite_adjusted", r.get("overall", 0))
        if r.get("critical"):
            insights.append(("#e24b4a", f"<strong>{name}</strong> is critical (score {score:.2f}). Immediate attention required."))
        elif r.get("at_risk"):
            insights.append(("#ef9f27", f"<strong>{name}</strong> is at risk (score {score:.2f}). Monitor closely."))
        if meta.get("scenario") == "underfunded":
            insights.append(("#e24b4a", f"<strong>{name}</strong> is underfunded — allocation below minimum viable band. Consider increasing budget."))
        elif meta.get("scenario") == "overfunded":
            insights.append(("#ef9f27", f"<strong>{name}</strong> is overfunded — additional spend is not converting to output. Consider reallocation."))
        if r.get("uncertain"):
            insights.append(("#185fa5", f"<strong>{name}</strong> has high GP model uncertainty — attainability score should be interpreted cautiously."))
    if len(results) > 1:
        under = [r for r in results if get_goal_meta(r["goal_id"], goals_df).get("scenario") == "underfunded"]
        over  = [r for r in results if get_goal_meta(r["goal_id"], goals_df).get("scenario") == "overfunded"]
        if under and over:
            insights.append(("#854f0b", "Budget reallocation opportunity: funds from overfunded goals could be redistributed to underfunded goals within the same parent bucket."))
    return insights[:6]


# ── Chart functions ───────────────────────────────────────────────────────────

def render_radar_chart(r: dict):
    """Radar chart of the 4 dimension scores for a single goal."""
    import streamlit.components.v1 as components
    import json
    dims   = ["Attainability", "Relevance", "Coherence", "Integrity"]
    scores = [round(r.get(d.lower(), 0), 3) for d in dims]
    html = f"""
    <div style="position:relative;width:100%;height:240px">
      <canvas id="radar_chart"></canvas>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
    <script>
    new Chart(document.getElementById('radar_chart'), {{
      type: 'radar',
      data: {{
        labels: {json.dumps(dims)},
        datasets: [{{
          label: 'Score',
          data: {json.dumps(scores)},
          backgroundColor: 'rgba(24,95,165,0.12)',
          borderColor: '#185fa5',
          pointBackgroundColor: '#185fa5',
          pointRadius: 4,
          borderWidth: 2
        }},{{
          label: 'Threshold',
          data: [0.35,0.35,0.35,0.35],
          backgroundColor: 'rgba(226,75,74,0.05)',
          borderColor: '#e24b4a',
          borderDash: [4,4],
          pointRadius: 0,
          borderWidth: 1.5
        }}]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{ r: {{
          min: 0, max: 1,
          ticks: {{ stepSize: 0.25, font: {{ size: 10 }}, color: '#888780', backdropColor: 'transparent' }},
          pointLabels: {{ font: {{ size: 11 }}, color: '#2c2c2a' }},
          grid: {{ color: 'rgba(255,255,255,0.07)' }},
          angleLines: {{ color: 'rgba(255,255,255,0.07)' }}
        }}}}
      }}
    }});
    </script>
    """
    components.html(html, height=255)


def render_trajectory_chart(r: dict):
    """Line chart showing composite score at now, +6, +12 periods."""
    import streamlit.components.v1 as components
    current = round(r.get("composite_adjusted", r.get("overall", 0)), 3)
    p6      = round(r.get("composite_p6",  current), 3)
    p12     = round(r.get("composite_p12", current), 3)
    html = f"""
    <div style="position:relative;width:100%;height:200px">
      <canvas id="traj_chart"></canvas>
    </div>
    <div style="font-size:11px;color:#6b6967;margin-top:5px">Dashed red line = risk threshold (0.35)</div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
    <script>
    new Chart(document.getElementById('traj_chart'), {{
      type: 'line',
      data: {{
        labels: ['Now (P12)', '+6 periods', '+12 (end)'],
        datasets: [{{
          label: 'Composite score',
          data: [{current}, {p6}, {p12}],
          borderColor: '#185fa5',
          backgroundColor: 'rgba(24,95,165,0.08)',
          pointBackgroundColor: '#185fa5',
          pointRadius: 5, fill: true, tension: 0.3, borderWidth: 2
        }},{{
          label: 'Risk threshold',
          data: [0.35, 0.35, 0.35],
          borderColor: '#e24b4a',
          borderDash: [5,5],
          pointRadius: 0, fill: false, borderWidth: 1.5
        }}]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          y: {{ min: 0, max: 1,
               ticks: {{ stepSize: 0.25, font: {{ size: 10 }}, color: '#888780' }},
               grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
          x: {{ ticks: {{ font: {{ size: 11 }}, color: '#888780' }}, grid: {{ display: false }} }}
        }}
      }}
    }});
    </script>
    """
    components.html(html, height=230)


def render_multi_goal_chart(results: list, goals_df):
    """Grouped bar chart comparing all goals across all 4 dimensions."""
    import streamlit.components.v1 as components
    import json
    COLORS = ["#185fa5", "#ef9f27", "#1d9e75", "#d85a30", "#7f77dd", "#993556"]
    dims   = ["Attainability", "Relevance", "Coherence", "Integrity"]
    datasets, legend_items = [], []
    for i, r in enumerate(results):
        meta  = get_goal_meta(r["goal_id"], goals_df)
        name  = meta["name"][:24]
        color = COLORS[i % len(COLORS)]
        data  = [round(r.get(d.lower(), 0), 3) for d in dims]
        datasets.append({"label": name, "data": data, "backgroundColor": color})
        legend_items.append(f'<span style="display:flex;align-items:center;gap:4px"><span style="width:9px;height:9px;border-radius:2px;background:{color}"></span><span style="font-size:11px;color:#a8a6a0">{name}</span></span>')
    legend_html = "".join(legend_items)
    html = f"""
    <div style="display:flex;flex-wrap:wrap;gap:12px;margin-bottom:10px">{legend_html}</div>
    <div style="position:relative;width:100%;height:220px">
      <canvas id="multi_chart"></canvas>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
    <script>
    new Chart(document.getElementById('multi_chart'), {{
      type: 'bar',
      data: {{ labels: {json.dumps(dims)}, datasets: {json.dumps(datasets)} }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          y: {{ min: 0, max: 1,
               ticks: {{ stepSize: 0.25, font: {{ size: 10 }}, color: '#888780' }},
               grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
          x: {{ ticks: {{ font: {{ size: 11 }}, color: '#888780' }}, grid: {{ display: false }} }}
        }}
      }}
    }});
    </script>
    """
    components.html(html, height=270)


def render_llm_chart(r: dict):
    """Grouped bar chart showing per-model scores across all 4 dimensions."""
    import streamlit.components.v1 as components
    import json
    dims   = ["Attainability", "Relevance", "Coherence", "Integrity"]
    models = {"llama3": "#378add", "gemma3": "#5dcaa5", "nemotron": "#d85a30"}
    datasets, legend_items = [], []
    for model, color in models.items():
        scores = r.get("llm_scores", {}).get(model, {})
        if not scores.get("success"): continue
        data = [round(scores.get(d.lower(), 0), 3) for d in dims]
        datasets.append({"label": model, "data": data, "backgroundColor": color})
        legend_items.append(f'<span style="display:flex;align-items:center;gap:4px"><span style="width:9px;height:9px;border-radius:2px;background:{color}"></span><span style="font-size:11px;color:#a8a6a0">{model}</span></span>')
    if not datasets:
        return
    legend_html = "".join(legend_items)
    html = f"""
    <div style="display:flex;flex-wrap:wrap;gap:12px;margin-bottom:10px">{legend_html}</div>
    <div style="position:relative;width:100%;height:190px">
      <canvas id="llm_chart"></canvas>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
    <script>
    new Chart(document.getElementById('llm_chart'), {{
      type: 'bar',
      data: {{ labels: {json.dumps(dims)}, datasets: {json.dumps(datasets)} }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          y: {{ min: 0, max: 1,
               ticks: {{ stepSize: 0.25, font: {{ size: 10 }}, color: '#888780' }},
               grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
          x: {{ ticks: {{ font: {{ size: 11 }}, color: '#888780' }}, grid: {{ display: false }} }}
        }}
      }}
    }});
    </script>
    """
    components.html(html, height=230)


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
            if st.button(f"{q} ↗", key=f"follow_{i}"):
                st.session_state["prefill_query"] = q
                st.session_state["results"] = None
                # st.switch_page("app.py")  # <- uncomment and point to Padmasri's entry file


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    query   = st.session_state.get("query",   DEMO_QUERY)
    results = st.session_state.get("results", None)

    if not results:
        st.info("No live data in session — showing demo output. Padmasri: set st.session_state['results'] and st.session_state['query'] before calling st.switch_page('dashboard.py')")
        results = DEMO_RESULTS

    # ── Load Groq API key ──
    # Priority: 1) st.secrets (Streamlit Cloud), 2) environment variable, 3) .env file
    api_key = ""

    # 1. Streamlit Cloud secrets
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass

    # 2. Environment variable
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")

    # 3. Local .env file
    if not api_key:
        try:
            env_path = os.path.join(os.path.dirname(__file__), ".env")
            if os.path.exists(env_path):
                with open(env_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip().lstrip("\ufeff")
                        if line.startswith("GROQ_API_KEY="):
                            api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
        except Exception:
            pass

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
                r.setdefault("composite_p6",  proj_map[gid].get("composite_p6",  r.get("overall", 0)))
                r.setdefault("composite_p12", proj_map[gid].get("composite_p12", r.get("overall", 0)))
                r.setdefault("improving_p6",  proj_map[gid].get("improving_p6",  False))
                r.setdefault("degrading_p6",  proj_map[gid].get("degrading_p6",  False))

    valid = [r for r in results if r.get("status") == "ok"]
    if not valid:
        st.error("No valid scored goals returned from System 3.")
        return

    col_logo, col_new = st.columns([5, 1])
    with col_logo:
        st.markdown(
            '<span style="font-size:13px;font-weight:500;color:#e8e6e0">⚙️ Coherence Engine</span>'
            '&nbsp;&nbsp;<span style="font-size:11px;color:#6b6967">Period 12 of 24 &nbsp;·&nbsp; System 1 Output</span>',
            unsafe_allow_html=True
        )
    with col_new:
        if st.button("← New query"):
            st.session_state["results"] = None
            st.session_state["query"]   = ""
            # st.switch_page("app.py")  # <- uncomment and point to Padmasri's entry file

    st.divider()
    render_query_banner(query, valid, goals_df)

    if len(valid) == 1:
        render_single_goal(valid[0], goals_df, buckets_df, proj_df, api_key)
    else:
        render_multi_goal(valid, goals_df, buckets_df, proj_df, api_key)

    st.divider()
    render_follow_up_buttons(valid, goals_df)


if __name__ == "__main__":
    main()
