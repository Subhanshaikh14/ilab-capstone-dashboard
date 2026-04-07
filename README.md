# Coherence Engine â€” Output Dashboard

**System 1 Output Layer | Team 14-02 | UTS iLab Capstone**

Subhan Farid â€” `dashboard.py`

---

## What this is

This repo contains the **output screen** of the Coherence Engine. It receives
scored goal data from System 3 and renders it as a Streamlit dashboard â€”
showing dimension scores, AI reasoning, trajectory projections, and
cross-goal comparisons.

It is one half of System 1. The input half lives in Padmasri's repo:
`github.com/Psri-01/ilab-capstone-system1`

---

## Repo structure

```
ilab-capstone-dashboard/
â”œâ”€â”€ dashboard.py                     <- main output screen (entry point)
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ .streamlit/
â”‚   â””â”€â”€ config.toml                  <- theme config
â”œâ”€â”€ data/
â”‚   â””â”€â”€ sample/
â”‚       â””â”€â”€ example_single_goal.json <- example System 3 payload
â”œâ”€â”€ tests/
â”‚   â””â”€â”€ test_dashboard.py            <- pytest smoke tests
â””â”€â”€ assets/                          <- images, icons if needed
```

---

## Quickstart

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Runs in **demo mode** with sample data â€” no System 2 or System 3 needed.

---

## Integration contract

### For Padmasri (System 1 input screen)

Before navigating to the dashboard, set these two session state keys:

```python
st.session_state["query"]   = user_query         # str  â€” original question
st.session_state["results"] = results_from_sys3  # list â€” score payloads

st.switch_page("dashboard.py")
```

Then uncomment and update this line in `dashboard.py`:
```python
# st.switch_page("app.py")  <- set to your actual entry filename
```

### For Anupam / Fatemeh (System 3)

Each item in `results` must be the dict returned by `score_goal()`
in `07_score_goal.py`. Minimum required fields:

| Field | Type | Notes |
|---|---|---|
| `goal_id` | int | |
| `attainability` | float 0-1 | |
| `relevance` | float 0-1 | |
| `coherence` | float 0-1 | |
| `integrity` | float 0-1 | |
| `overall` | float 0-1 | |
| `gp_std` | float | |
| `uncertain` | bool | |
| `reasoning` | dict | keys: attainability, relevance, coherence, integrity |
| `llm_scores` | dict | keys: llama3, gemma3, nemotron |
| `ensemble_meta` | dict | |
| `status` | str | "ok" or "error" |

Optional (auto-derived if missing):
`composite_adjusted`, `at_risk`, `critical`, `weakest_dim`,
`composite_p6`, `composite_p12`

### Supporting CSV files

Place in the same folder as `dashboard.py` for richer output:
- `goals.csv` â€” goal names, targets, units, scenario flags
- `buckets.csv` â€” hierarchy for breadcrumbs and conflict detection
- `forward_projection_poc.csv` â€” trajectory scores at +6 and +12 periods

---

## Routing logic

| Condition | View |
|---|---|
| 1 goal returned | Single goal â€” full dimensions, trajectory, LLM panel |
| 2+ goals, shared bucket | Portfolio header + conflict alert + tabs + comparison table |
| 2+ goals, different buckets | Tabs + comparison table |

---

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Key dates

| Date | Milestone |
|---|---|
| 24 Apr 2026 | Progress Report + supervisor demo |
| 6 May 2026  | Oral presentation |
| 22 May 2026 | Final report |
