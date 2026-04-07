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
