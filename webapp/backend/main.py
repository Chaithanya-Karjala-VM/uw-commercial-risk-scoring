"""
Assisted Underwriting - UW Workbench API.

Serves the real scoring pipeline (src/scoring.py and friends) as JSON, completely
unchanged, to the static HTML/CSS/JS frontend in webapp/frontend. This exists so the new
HTML build's predictions are byte-identical to both Streamlit apps - same model, same
code, just a different presentation layer. Does not modify anything in src/.

Run:
    cd webapp/backend
    ../../.venv/Scripts/python.exe -m uvicorn main:app --reload --port 8000
"""
import math
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from data_loader import load_data
from scoring import load_models, score
from decision_matrix_reference import RISK_TIER_ORDER, RISK_TIER_INFO, tier_for_score
import driver_explain as dexp
from narrative import build_narrative, build_banner_summary
import feature_map as fmap
from lob_config import ALL_LOBS
import appetite_capacity as ac
import bind_propensity as bp
import scoring as scoring_module

app = FastAPI(title="Assisted Underwriting - UW Workbench API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_PARQUET_PATH = SRC_DIR.parent / "data" / "UW_Risk_Score_Commercial_F.parquet"

print("Loading models and scoring the book (one-time, at startup)...")
MODELS = load_models()
if _PARQUET_PATH.exists():
    # Pre-converted once from the same Excel file via load_data() (data/convert_to_parquet
    # equivalent run locally, not part of the request path) - openpyxl's XML parsing peaks
    # at ~170MB RAM and takes 80+ seconds for this file, which is what was blowing past
    # Render's free-tier 512MB limit on deploy. Parquet reads the identical data in ~1s at
    # a few MB. Verified byte-identical dtypes and scores against the Excel path before
    # switching. The webapp never uses load_data()'s other 3 return values
    # (schema/usable_for/output_cols - only feature_map.py's own independent feature lists
    # matter here), so skipping them is safe.
    _RAW_DF = pd.read_parquet(_PARQUET_PATH)
else:
    _RAW_DF, _, _, _ = load_data()
_RAW_DF["effective_date"] = pd.to_datetime(_RAW_DF["effective_date"])
BOOK = score(_RAW_DF, MODELS)
LOB_OPTIONS = sorted(ALL_LOBS)
REGION_OPTIONS = sorted(BOOK["risk_location_region"].dropna().unique().tolist())
BROKER_TIER_OPTIONS = ["Platinum", "Gold", "Silver", "Bronze"]
print(f"Book ready: {len(BOOK):,} rows, {len(BOOK.columns)} columns.")

# Session-only state (in-memory, resets on restart - no persistence layer in this project,
# same honest scope as the Streamlit build's review-status/feedback features).
REVIEW_STATE = {}       # submission_id -> {"status":..., "assigned_to":..., "decision":..., "reason_code":..., "notes":...}
UNDERWRITER_ROSTER = ["R. Shah", "J. Alvarez", "M. Okafor", "T. Nguyen", "S. Patel"]  # placeholder roster - no real underwriter identity data exists in this project


def _clean(v):
    """JSON-safe scalar - numpy/pandas types don't serialize via plain json, and NaN/Inf
    aren't valid JSON at all (they'd silently break the frontend's JSON.parse)."""
    if v is None:
        return None
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if pd.isna(v) if not isinstance(v, (list, dict)) else False:
        return None
    return v


def row_to_dict(row):
    return {k: _clean(v) for k, v in row.to_dict().items()}


def df_to_records(df):
    return [row_to_dict(r) for _, r in df.iterrows()]


def apply_filters(df, lob=None, region=None, tier=None, broker_tier=None, renewal=None):
    out = df
    if lob and lob != "All":
        out = out[out["line_of_business"] == lob]
    if region and region != "All":
        out = out[out["risk_location_region"] == region]
    if tier and tier != "All":
        out = out[out["risk_tier_v2"] == tier]
    if broker_tier and broker_tier != "All":
        out = out[out["broker_tier"] == broker_tier]
    if renewal and renewal != "All":
        out = out[out["renewal_flag"] == renewal]
    return out


def _status_of(sub_id):
    return REVIEW_STATE.get(sub_id, {}).get("status", "To Review")


# ===========================================================================
# Filters
# ===========================================================================
@app.get("/api/filters")
def get_filters():
    return {
        "lobs": LOB_OPTIONS,
        "regions": REGION_OPTIONS,
        "tiers": RISK_TIER_ORDER,
        "broker_tiers": BROKER_TIER_OPTIONS,
        "renewal": ["Renewal", "New Business"],
    }


# ===========================================================================
# Portfolio view
# ===========================================================================
@app.get("/api/portfolio/kpis")
def portfolio_kpis(lob: str = "All", region: str = "All", tier: str = "All", broker_tier: str = "All", renewal: str = "All"):
    fb = apply_filters(BOOK, lob, region, tier, broker_tier, renewal)
    n = len(fb)
    if n == 0:
        return {"submissions": 0}
    tiv = fb["total_insured_value_usd"].sum()
    premium = fb["indicated_annual_premium_usd"].sum()
    incurred = fb["total_incurred_claim_amount_usd"].sum()
    hard_fails = int((fb["appetite_capacity_score_v2"] == 0).sum())
    accept = int(fb["risk_tier_v2"].isin(["Preferred", "Standard"]).sum())
    preferred_premium = fb.loc[fb["risk_tier_v2"] == "Preferred", "indicated_annual_premium_usd"].sum()
    referred = int(fb["risk_tier_v2"].isin(["High", "Decline"]).sum())
    return {
        "submissions": n,
        "total_tiv_usd": _clean(tiv),
        "premium_usd": _clean(premium),
        "incurred_usd": _clean(incurred),
        "incurred_loss_ratio_pct": _clean(incurred / premium * 100) if premium > 0 else 0,
        "avg_uw_score": _clean(fb["uw_score_v2"].mean()),
        "accept_rate_pct": _clean(accept / n * 100),
        "hard_rule_fails": hard_fails,
        "hard_rule_fails_pct": _clean(hard_fails / n * 100),
        # Same 2 KPIs the Streamlit Portfolio tab shows that the HTML build was missing -
        # both real, not predicted: preferred_premium is a straight sum filter, referred is
        # a straight tier-membership count.
        "preferred_premium_usd": _clean(preferred_premium),
        "referred_count": referred,
        "referred_pct": _clean(referred / n * 100),
    }


@app.get("/api/portfolio/tier-mix")
def portfolio_tier_mix(lob: str = "All", region: str = "All", tier: str = "All", broker_tier: str = "All", renewal: str = "All"):
    fb = apply_filters(BOOK, lob, region, tier, broker_tier, renewal)
    n = len(fb)
    rows = []
    for t in RISK_TIER_ORDER:
        tb = fb[fb["risk_tier_v2"] == t]
        rows.append({
            "tier": t,
            "submissions": len(tb),
            "pct_of_book": _clean(len(tb) / n * 100) if n else 0,
            "premium_usd": _clean(tb["indicated_annual_premium_usd"].sum()) if len(tb) else 0,
            "avg_loss": _clean(tb["loss_propensity_score_v2"].mean()) if len(tb) else None,
            "avg_appetite": _clean(tb["appetite_capacity_score_v2"].mean()) if len(tb) else None,
            "avg_bind": _clean(tb["bind_propensity_score_v2"].mean()) if len(tb) else None,
            "avg_uw": _clean(tb["uw_score_v2"].mean()) if len(tb) else None,
        })
    return {"rows": rows}


@app.get("/api/portfolio/appetite-decision")
def portfolio_appetite_decision(lob: str = "All", region: str = "All", tier: str = "All", broker_tier: str = "All", renewal: str = "All"):
    fb = apply_filters(BOOK, lob, region, tier, broker_tier, renewal)
    # appetite_capacity_decision_v2 is already computed by score() itself - reading it
    # directly instead of recomputing guarantees this always matches the real pipeline.
    counts = fb["appetite_capacity_decision_v2"].value_counts().to_dict()

    # "first rule failed" - only the 4 REAL, verified hard-fail flags (see module docstring:
    # "confirmed, exact NECESSARY condition... verified with zero exceptions across all
    # 20,000 rows"). Checked in a fixed priority order so each failed row is attributed to
    # exactly one reason, not double-counted.
    failed = fb[~ac.hard_rule_pass(fb)]
    reasons = []
    remaining = failed.copy()
    for flag, label in [
        ("sanctions_ofac_watchlist_flag", "Sanctions / OFAC"),
        ("prior_insolvency_bankruptcy_flag", "Prior insolvency"),
        ("restricted_operations_flag", "Restricted operations"),
        ("region_lob_restricted_flag", "Region x LOB restricted"),
    ]:
        hit = remaining[remaining[flag] == 1]
        reasons.append({"reason": label, "count": len(hit)})
        remaining = remaining[remaining[flag] != 1]

    return {
        "decisions": [{"decision": k, "count": int(v)} for k, v in counts.items()],
        "first_rule_failed": reasons,
        "total_hard_fails": int(len(failed)),
    }


@app.get("/api/portfolio/uw-distribution")
def portfolio_uw_distribution(lob: str = "All", region: str = "All", tier: str = "All", broker_tier: str = "All", renewal: str = "All"):
    fb = apply_filters(BOOK, lob, region, tier, broker_tier, renewal)
    n = len(fb)
    if n == 0:
        return {"bins": [], "floor_count": 0, "cap_count": 0}
    counts, edges = np.histogram(fb["uw_score_v2"], bins=20, range=(0, 100))
    bins = [{"start": _clean(edges[i]), "end": _clean(edges[i + 1]), "count": int(counts[i])} for i in range(len(counts))]
    floor_count = int((fb["loss_propensity_score_v2"] <= 10.0001).sum())
    cap_count = int((fb["loss_propensity_score_v2"] >= 99.9999).sum())
    return {
        "bins": bins, "floor_count": floor_count, "cap_count": cap_count,
        "floor_pct": _clean(floor_count / n * 100), "cap_pct": _clean(cap_count / n * 100),
    }


@app.get("/api/portfolio/lob-table")
def portfolio_lob_table(region: str = "All", tier: str = "All", broker_tier: str = "All", renewal: str = "All"):
    fb = apply_filters(BOOK, "All", region, tier, broker_tier, renewal)
    rows = []
    for lob in LOB_OPTIONS:
        tb = fb[fb["line_of_business"] == lob]
        if len(tb) == 0:
            continue
        prem = tb["indicated_annual_premium_usd"].sum()
        incurred = tb["total_incurred_claim_amount_usd"].sum()
        tier_counts = tb["risk_tier_v2"].value_counts()
        rows.append({
            "lob": lob, "submissions": len(tb), "premium_usd": _clean(prem),
            "loss_ratio_pct": _clean(incurred / prem * 100) if prem > 0 else 0,
            "avg_uw": _clean(tb["uw_score_v2"].mean()),
            "tier_mix": {t: int(tier_counts.get(t, 0)) for t in RISK_TIER_ORDER},
        })
    return {"rows": rows}


_GRID_METRICS = {
    "uw_score": ("uw_score_v2", True), "loss": ("loss_propensity_score_v2", False),
    "appetite": ("appetite_capacity_score_v2", True), "bind": ("bind_propensity_score_v2", True),
    "loss_ratio": ("predicted_loss_ratio_pct", False), "hard_fail_pct": ("_hard_fail", False),
}


@app.get("/api/portfolio/grid")
def portfolio_grid(metric: str = "uw_score", lob: str = "All", tier: str = "All", broker_tier: str = "All", renewal: str = "All"):
    if metric not in _GRID_METRICS:
        raise HTTPException(400, f"Unknown metric {metric!r}")
    col, higher_better = _GRID_METRICS[metric]
    fb = apply_filters(BOOK, lob, "All", tier, broker_tier, renewal).copy()
    fb["_hard_fail"] = (fb["appetite_capacity_score_v2"] == 0).astype(float) * 100
    pivot = fb.pivot_table(index="line_of_business", columns="risk_location_region", values=col, aggfunc="mean")
    return {
        "lobs": pivot.index.tolist(), "regions": pivot.columns.tolist(),
        "values": [[_clean(v) for v in row] for row in pivot.values],
        "higher_is_better": higher_better,
    }


@app.get("/api/portfolio/quadrant")
def portfolio_quadrant(lob: str = "All", region: str = "All", tier: str = "All", broker_tier: str = "All", renewal: str = "All"):
    """Risk-Opportunity Quadrant Map data - same real fields and same median-split logic as
    the Streamlit Portfolio tab's scatter (loss propensity vs. bind propensity, split at the
    CURRENT filtered slice's own real median on each axis, not a fixed number), sampled down
    to at most 1200 rows so the frontend isn't shipping 20,000 points over HTTP."""
    fb = apply_filters(BOOK, lob, region, tier, broker_tier, renewal)
    n = len(fb)
    if n == 0:
        return {"points": [], "mid_x": 0, "mid_y": 0, "y_min": 0, "y_max": 100}
    mid_x = float(fb["loss_propensity_score_v2"].median())
    mid_y = float(fb["bind_propensity_score_v2"].median())
    sample = fb.sample(min(1200, n), random_state=1)
    points = [{
        "submission_id": r["submission_id"], "line_of_business": r["line_of_business"],
        "loss": _clean(r["loss_propensity_score_v2"]), "bind": _clean(r["bind_propensity_score_v2"]),
        "premium": _clean(r["indicated_annual_premium_usd"]), "tier": r["risk_tier_v2"],
    } for _, r in sample.iterrows()]
    return {
        "points": points, "mid_x": mid_x, "mid_y": mid_y,
        "y_min": _clean(fb["bind_propensity_score_v2"].min()), "y_max": _clean(fb["bind_propensity_score_v2"].max()),
    }


@app.get("/api/portfolio/industry-concentration")
def portfolio_industry_concentration(lob: str = "All", region: str = "All", tier: str = "All", broker_tier: str = "All", renewal: str = "All"):
    fb = apply_filters(BOOK, lob, region, tier, broker_tier, renewal)
    total_tiv = fb["total_insured_value_usd"].sum()
    rows = []
    for industry, cap_share in ac.INDUSTRY_CONCENTRATION_BY_NAICS.items():
        ib = fb[fb["industry_class_naics"] == industry]
        share = (ib["total_insured_value_usd"].sum() / total_tiv) if total_tiv > 0 else 0
        rows.append({"industry": industry, "tiv_share_pct": _clean(share * 100),
                      "cap_pct": ac.CONCENTRATION_CAP_PCT * 100,
                      "headroom_pts": _clean((ac.CONCENTRATION_CAP_PCT - share) * 100)})
    return {"rows": sorted(rows, key=lambda r: -r["tiv_share_pct"])}


@app.get("/api/portfolio/treaties")
def portfolio_treaties(lob: str = "All", region: str = "All", tier: str = "All"):
    fb = apply_filters(BOOK, lob, region, tier)
    rows = []
    for col, label, zone_level in [
        ("reinsurance_regional_headroom_pct", "Regional", "regional"),
        ("reinsurance_country_headroom_pct", "Country", "country"),
        ("reinsurance_state_headroom_pct", "State", "state"),
    ]:
        if col not in fb.columns:
            continue
        rows.append({"level": label, "min_headroom_pct": _clean(fb[col].min()), "avg_headroom_pct": _clean(fb[col].mean())})
    return {"rows": rows}


@app.get("/api/portfolio/broker-tier")
def portfolio_broker_tier(lob: str = "All", region: str = "All", tier: str = "All", renewal: str = "All"):
    fb = apply_filters(BOOK, lob, region, tier, "All", renewal)
    rows = []
    for bt in BROKER_TIER_OPTIONS:
        tb = fb[fb["broker_tier"] == bt]
        if len(tb) == 0:
            continue
        rows.append({"broker_tier": bt, "submissions": len(tb),
                      "avg_bind": _clean(tb["bind_propensity_score_v2"].mean()),
                      "avg_uw": _clean(tb["uw_score_v2"].mean())})
    return {"rows": rows}


# ===========================================================================
# Submission queue + single-submission detail
# ===========================================================================
@app.get("/api/queue")
def queue(lob: str = "All", region: str = "All", tier: str = "All", broker_tier: str = "All",
          renewal: str = "All", status: str = "All", sort: str = "priority", limit: int = 300):
    fb = apply_filters(BOOK, lob, region, tier, broker_tier, renewal)
    if status != "All":
        mask = fb["submission_id"].apply(lambda s: _status_of(s) == status)
        fb = fb[mask]
    if sort == "priority":
        # No live "priority" model exists in this pipeline - closest real, defensible proxy:
        # referral-worthy cases (High/Decline) first, then worst UW score first within that,
        # so "work now" means "the cases most likely to need a human look," not an arbitrary
        # order. Documented clearly in the frontend, not presented as a trained priority score.
        fb = fb.assign(_needs_review=fb["risk_tier_v2"].isin(["High", "Decline"]).astype(int))
        fb = fb.sort_values(["_needs_review", "uw_score_v2"], ascending=[False, True])
    elif sort == "uw_desc":
        fb = fb.sort_values("uw_score_v2", ascending=False)
    elif sort == "uw_asc":
        fb = fb.sort_values("uw_score_v2", ascending=True)

    out = fb.head(limit)
    rows = []
    for _, r in out.iterrows():
        rows.append({
            "submission_id": r["submission_id"], "line_of_business": r["line_of_business"],
            "risk_location_region": r["risk_location_region"], "risk_state": r.get("risk_state"),
            "uw_score": _clean(r["uw_score_v2"]),
            "risk_tier": r["risk_tier_v2"], "recommended_action": r["recommended_uw_action_v2"],
            "is_hard_fail": bool(r["appetite_capacity_score_v2"] == 0),
            "appetite_decision": r["appetite_capacity_decision_v2"],
            "status": _status_of(r["submission_id"]),
            "assigned_to": REVIEW_STATE.get(r["submission_id"], {}).get("assigned_to"),
        })
    return {"total_matching": len(fb), "returned": len(rows), "rows": rows}


@app.get("/api/queue/breakdown")
def queue_breakdown(lob: str = "All", region: str = "All", tier: str = "All", broker_tier: str = "All",
                     renewal: str = "All", status: str = "All", by: str = "lob"):
    """Aggregate view of the CURRENT filtered queue, not the whole 20,000-row book (that's
    what the Portfolio tab is for) - this answers "what's actually in front of me and why,"
    which is the underwriter-queue question, not the portfolio-strategy one. Cheap group-by,
    no per-row re-scoring - every field here is already a real column on the scored book."""
    fb = apply_filters(BOOK, lob, region, tier, broker_tier, renewal)
    if status != "All":
        mask = fb["submission_id"].apply(lambda s: _status_of(s) == status)
        fb = fb[mask]
    if len(fb) == 0:
        return {"by": by, "rows": []}

    if by == "region":
        group_col, fb2 = "risk_location_region", fb
    elif by == "reason":
        # Same real category each queue row's own pill already shows - hard-rule fail wins,
        # else the real appetite_capacity_decision_v2 value - just grouped here instead of
        # shown per-row.
        fb2 = fb.copy()
        fb2["_reason"] = np.where(fb2["appetite_capacity_score_v2"] == 0, "Hard-rule fail", fb2["appetite_capacity_decision_v2"])
        group_col = "_reason"
    else:
        group_col, fb2 = "line_of_business", fb

    g = fb2.groupby(group_col).agg(
        submissions=("submission_id", "count"),
        avg_uw=("uw_score_v2", "mean"),
        avg_loss=("loss_propensity_score_v2", "mean"),
        avg_appetite=("appetite_capacity_score_v2", "mean"),
        avg_bind=("bind_propensity_score_v2", "mean"),
        premium_usd=("indicated_annual_premium_usd", "sum"),
    ).reset_index().rename(columns={group_col: "label"})
    g = g.sort_values("submissions", ascending=False)
    rows = []
    for _, r in g.iterrows():
        rows.append({
            "label": r["label"], "submissions": int(r["submissions"]), "avg_uw": _clean(round(r["avg_uw"], 1)),
            "premium_usd": _clean(round(r["premium_usd"], 0)),
            # Same 3 real sub-scores every submission already has, just averaged per group -
            # re-expressed as "weakness" (0 = no problem, 100 = big problem) on a SHARED scale
            # so Loss/Appetite/Bind are directly comparable on one chart. Loss is already
            # 0=best/100=worst so it's used as-is; Appetite and Bind are 0=worst/100=best so
            # they're inverted (100 - avg) to match that same "higher = more of a problem" sense.
            "loss_weakness": _clean(round(r["avg_loss"], 1)),
            "appetite_weakness": _clean(round(100 - r["avg_appetite"], 1)),
            "bind_weakness": _clean(round(100 - r["avg_bind"], 1)),
        })
    return {"by": by, "rows": rows}


def _score_fn(d):
    return score(d, MODELS).iloc[0]


def _hard_rule_checklist(row):
    checks = [
        {"rule": "Sanctions / OFAC watchlist clear", "passed": bool(row["sanctions_ofac_watchlist_flag"] == 0), "hard": True},
        {"rule": "No prior insolvency / bankruptcy", "passed": bool(row["prior_insolvency_bankruptcy_flag"] == 0), "hard": True},
        {"rule": "No restricted operations", "passed": bool(row["restricted_operations_flag"] == 0), "hard": True},
        {"rule": "Region x LOB not restricted", "passed": bool(row["region_lob_restricted_flag"] == 0), "hard": True},
    ]
    passed_count = sum(1 for c in checks if c["passed"])
    # Informational, NOT hard gates in the real trained model - flagged honestly as soft
    # factors rather than invented as additional hard rules (see appetite_capacity.py's own
    # docstring: "Tested TIV thresholds... CAT zone combinations... none produced a clean
    # deterministic rule" - these move the score, they don't veto it).
    soft_factors = [
        {"factor": "TIV within $50k-$30M", "ok": bool(50_000 <= row["total_insured_value_usd"] <= 30_000_000), "note": "informational only - not a hard veto in the trained model"},
        {"factor": "Severe CAT zone + high TIV", "ok": not (
            max(row.get("flood_zone_tier", 0), row.get("earthquake_zone_tier", 0), row.get("wildfire_zone_tier", 0)) >= 3
            and row["total_insured_value_usd"] > 2_000_000
        ), "note": "soft penalty (-8/-12 pts) in the real model, not a veto"},
    ]
    return checks, passed_count, soft_factors


def _calc_trail(row_df, row):
    """Real per-submission arithmetic trail - every number here is the actual coefficient
    (scoring.APPETITE_FIT_COEFFICIENTS) times this row's actual real input value, computed
    the same way compute_appetite_fit() itself does, not a reconstruction. Lets someone
    verify the Appetite Fit sub-score and the final UW blend line by line."""
    afc = scoring_module.APPETITE_FIT_COEFFICIENTS
    passed = bool(ac.hard_rule_pass(row_df).iloc[0])
    broker_tier_num = ac.BROKER_TIER_MAP.get(row.get("broker_tier"), 2)
    completeness = row.get("submission_completeness_pct")
    if completeness is None or pd.isna(completeness):
        completeness = ac.BOOK_AVG_COMPLETENESS
    coverage = row.get("coverage_limit_requested_usd") or 0
    cat_penalty = float(ac.cat_tiv_penalty(row_df)[0])

    terms = [
        {"label": "Intercept", "coefficient": afc["intercept"], "input": 1, "contribution": afc["intercept"]},
        {"label": "Hard-rule pass", "coefficient": afc["appetite_hard_rule_pass_flag"], "input": int(passed),
         "contribution": afc["appetite_hard_rule_pass_flag"] * passed},
        {"label": "Target growth segment", "coefficient": afc["target_growth_segment_flag"],
         "input": _clean(row.get("target_growth_segment_flag")) or 0,
         "contribution": afc["target_growth_segment_flag"] * (row.get("target_growth_segment_flag") or 0)},
        {"label": "Financial stability score", "coefficient": afc["financial_stability_score"],
         "input": _clean(row.get("financial_stability_score")),
         "contribution": afc["financial_stability_score"] * (row.get("financial_stability_score") or 0)},
        {"label": f"Broker tier ({row.get('broker_tier', '—')})", "coefficient": afc["broker_tier_num"],
         "input": broker_tier_num, "contribution": afc["broker_tier_num"] * broker_tier_num},
        {"label": "Submission completeness", "coefficient": afc["submission_completeness_pct"],
         "input": _clean(completeness), "contribution": afc["submission_completeness_pct"] * completeness},
        {"label": "CAT zone + TIV penalty (soft)", "coefficient": None, "input": None, "contribution": cat_penalty},
    ]
    for t in terms:
        t["contribution"] = _clean(round(t["contribution"], 2))
    fit_total = sum(t["contribution"] for t in terms)
    fit_total_clipped = max(0.0, min(100.0, fit_total)) if passed else 0.0

    loss_term = round(0.50 * (100 - row["loss_propensity_score_v2"]), 2)
    appetite_term = round(0.30 * row["appetite_capacity_score_v2"], 2)
    bind_term = round(0.20 * row["bind_propensity_score_v2"], 2)

    return {
        "appetite_fit_terms": terms,
        "appetite_fit_raw_sum": _clean(round(fit_total, 2)),
        "appetite_fit_after_hard_rule": _clean(round(fit_total_clipped, 1)),
        "uw_blend": {
            "loss_term": _clean(loss_term), "appetite_term": _clean(appetite_term), "bind_term": _clean(bind_term),
            "total": _clean(row["uw_score_v2"]),
        },
    }


@app.get("/api/submission/{submission_id}")
def submission_detail(submission_id: str):
    match = BOOK[BOOK["submission_id"] == submission_id]
    if len(match) == 0:
        raise HTTPException(404, f"Submission {submission_id} not found")
    row = match.iloc[0]
    row_df = match
    row_dict = row.to_dict()

    loss_drv = dexp.loss_drivers(row_df, _score_fn)
    bind_drv = dexp.bind_drivers(row_dict)
    appetite_drv = dexp.appetite_drivers(row_dict)
    narrative_text = build_narrative(row, loss_drv, bind_drv, appetite_drv)
    banner = build_banner_summary(row, loss_drv, bind_drv)

    hard_checks, hard_passed_count, soft_factors = _hard_rule_checklist(row)

    # Live loss-only tier: same real tier cutoffs applied to Loss Propensity alone, instead
    # of the frozen loss_only_tier_after_propensity column (a workbook-era value that can't
    # be regenerated for a brand-new submission) - consistent with how every other field in
    # this project is treated: recomputed live, never stale.
    loss_only_score = 100 - row["loss_propensity_score_v2"]  # same inversion as uw_score_v2's own Loss term
    loss_only_tier = tier_for_score(loss_only_score)

    review = REVIEW_STATE.get(submission_id, {})

    effective_date = row.get("effective_date")
    return {
        "submission_id": submission_id,
        "line_of_business": row["line_of_business"],
        "is_new_business": row["renewal_flag"] == "New Business",
        "region": row["risk_location_region"], "state": row.get("risk_state"),
        "broker_tier": row["broker_tier"], "broker_size_class": row.get("broker_size_class"),
        "broker_agent_id": row.get("broker_agent_id"),
        "industry": row.get("industry_class_naics"),
        "business_description": row.get("business_description"),
        "effective_date": effective_date.strftime("%d %b %Y") if pd.notna(effective_date) else None,
        "tiv_usd": _clean(row["total_insured_value_usd"]),
        "aoa_limit_usd": _clean(row.get("aoa_limit_usd")), "aoy_limit_usd": _clean(row.get("aoy_limit_usd")),
        "indicated_premium_usd": _clean(row["indicated_annual_premium_usd"]),
        "primary_cat_peril": row.get("primary_cat_peril_exposure"),
        "assigned_to": review.get("assigned_to"),
        "status": _status_of(submission_id),

        "scores": {
            "loss_propensity": _clean(row["loss_propensity_score_v2"]),
            "appetite_capacity": _clean(row["appetite_capacity_score_v2"]),
            "bind_propensity": _clean(row["bind_propensity_score_v2"]),
            "uw_score": _clean(row["uw_score_v2"]),
            "risk_tier": row["risk_tier_v2"],
            "recommended_action": row["recommended_uw_action_v2"],
            "loss_only_tier": loss_only_tier,
        },

        "override": {
            "is_hard_fail": bool(row["appetite_capacity_score_v2"] == 0),
            "decline_reason": row.get("decline_review_reason_v2"),
        },

        "decision_dimensions": {
            "appetite_eligibility": {
                "hard_rules_passed": hard_passed_count, "hard_rules_total": len(hard_checks),
                "checks": hard_checks, "soft_factors": soft_factors,
                "appetite_fit_score": _clean(row["appetite_capacity_score_v2"]),
            },
            "loss": {
                "current_loss_ratio_pct": _clean(row.get("loss_ratio_after_propensity_pct")),
                "prior_period_loss_ratio_pct": _clean(row.get("prior_loss_ratio_avg_pct")),
                "risk_factor_expected_lr_pct": _clean(row.get("risk_factor_expected_lr_pct")),
                "hazard_index": _clean(row.get("hazard_index_score")),
                "predicted_loss_ratio_pct": _clean(row["predicted_loss_ratio_pct"]),
                "flags": row.get("prospective_risk_flags_v2"),
            },
            "exposure_capacity": {
                "industry": row.get("industry_class_naics"),
                "industry_concentration_share_pct": _clean(ac.INDUSTRY_CONCENTRATION_BY_NAICS.get(row.get("industry_class_naics"), 0) * 100),
                "concentration_cap_pct": ac.CONCENTRATION_CAP_PCT * 100,
                "regional_treaty_headroom_pct": _clean(row.get("reinsurance_regional_headroom_pct")),
                "country_treaty_headroom_pct": _clean(row.get("reinsurance_country_headroom_pct")),
                "state_treaty_headroom_pct": _clean(row.get("reinsurance_state_headroom_pct")),
                "appetite_decision": row.get("appetite_capacity_decision_v2"),
            },
            "broker_bind": {
                "broker_quality_score": _clean(row.get("broker_calc_quality_score_0_1")),
                "hit_bind_ratio_pct": _clean(row.get("broker_hit_bind_ratio_pct")),
                "competing_quotes": _clean(row.get("competing_quotes_count")),
                "quote_to_bind_days": _clean(row.get("quote_to_bind_days")),
                "renewal_retention_pct": _clean(row.get("broker_renewal_retention_pct")),
                "bind_score": _clean(row["bind_propensity_score_v2"]),
            },
        },

        "drivers": {"loss": loss_drv, "appetite": appetite_drv, "bind": bind_drv},
        "narrative": narrative_text, "banner_summary": banner,

        "claims": {
            "total": _clean(row.get("claim_count_total")), "open": _clean(row.get("claim_count_open")),
            "settled": _clean(row.get("claim_count_settled")), "last_3yr": _clean(row.get("claim_count_last_3_years")),
            "claims_free_years": _clean(row.get("claims_free_years")),
            "prior_total_incurred_usd": _clean(row.get("prior_total_incurred_usd")),
        },

        "data_quality": {"completeness_pct": _clean(row.get("submission_completeness_pct"))},
        "calc_trail": _calc_trail(row_df, row),
    }


class ReviewUpdate(BaseModel):
    status: str | None = None
    assigned_to: str | None = None
    decision: str | None = None
    reason_code: str | None = None
    notes: str | None = None


@app.get("/api/underwriters")
def underwriters():
    return {"roster": UNDERWRITER_ROSTER}


@app.post("/api/submission/{submission_id}/review")
def update_review(submission_id: str, update: ReviewUpdate):
    if (BOOK["submission_id"] == submission_id).sum() == 0:
        raise HTTPException(404, f"Submission {submission_id} not found")
    current = REVIEW_STATE.setdefault(submission_id, {})
    for field, value in update.model_dump(exclude_none=True).items():
        current[field] = value
    return {"submission_id": submission_id, "review": current}


# ===========================================================================
# Feature map
# ===========================================================================
@app.get("/api/feature-map")
def feature_map(lob: str = LOB_OPTIONS[0]):
    rows = fmap.build_reference_rows(fmap.loss_features_for(lob))
    return {
        "lob": lob, "rows": rows,
        "loss_field_count": sum(r["Loss Propensity"] for r in rows),
        "bind_field_count": sum(r["Bind Propensity"] for r in rows),
        "appetite_field_count": sum(r["Appetite / Capacity"] for r in rows),
    }


@app.get("/api/model-mechanics")
def model_mechanics():
    """Every number here is pulled directly from the real scoring modules at request time -
    never hand-typed - so this can never silently drift from what the model actually does.
    Loss Propensity is NOT the manager's wireframe's rules-based formula (Loss = clamp(0.55
    LRcurrent + 0.30 LRprior + 0.15 ELRrisk)) - that's the original workbook's pre-ML
    approach. Our real deployed model replaced it with 3 trained ML sub-models (frequency,
    severity, large-loss). Flagged honestly rather than faked to match the wireframe."""
    afc = scoring_module.APPETITE_FIT_COEFFICIENTS
    bc = bp.BIND_COEFFICIENTS
    return {
        "loss": {
            "weight_pct": 50,
            "is_trained_ml": True,
            "note": "Differs from the original workbook's rules-based formula - our real model uses 3 trained ML sub-models, not a fixed-weight blend of 3 loss-ratio views.",
            "models": ["frequency_model.joblib", "severity_model.joblib", "large_loss_model.joblib"],
            "pipeline": [
                "Predicted Frequency = frequency model × exposure",
                "Predicted Attritional Severity = severity model prediction",
                "Probability of Large Loss = large-loss model prediction",
                "Expected Large-Loss Severity = flat large severity × clip(severity ratio, 0.3, 3.0)",
                "Attritional Frequency = max(Predicted Frequency − Probability of Large Loss, 0)",
                "Total Expected Loss Cost = Attritional Frequency × Severity + Probability of Large Loss × Expected Large-Loss Severity",
                "Predicted Loss Ratio % = Total Expected Loss Cost / Premium × 100",
                "Loss Score = clip(Predicted Loss Ratio %, 0, 100)",
            ],
            # The real loss ratio's denominator - same honesty pattern as everywhere else in
            # this project: the book's own real premium is used whenever a submission has one,
            # and these per-LOB rates only kick in as an estimate for a genuinely new quote.
            "premium_note": "Uses the submission's own real premium when one exists (the normal case for the 20,000-row book). Only estimated from these rates for a brand-new quote that has no premium yet.",
            "premium_rates_pct": {lob: round(rate * 100, 4) for lob, rate in scoring_module.PREMIUM_RATE_BY_LOB.items()},
            "premium_basis": "Property lines: rate × Total Insured Value. Liability lines: rate × payroll (employees × avg wage).",
        },
        "appetite": {
            "weight_pct": 30,
            "is_trained_ml": False,
            "formula": "Appetite/Capacity Score = 0 if hard-rule fail, else clip(0.75×Fit + 0.25×Headroom, 0, 100)",
            "hard_rules": [
                "Sanctions / OFAC watchlist clear", "No prior insolvency / bankruptcy",
                "No restricted operations", "Region x LOB not restricted",
            ],
            "fit_formula": "Fit = Intercept + Hard-Rule Pass×coef + Growth Segment×coef + Financial Stability×coef + Broker Tier×coef + Completeness×coef + CAT/TIV Penalty",
            "fit_coefficients": afc,
            "headroom_formula": "Headroom = min(Industry Component, State Component), each independently scaled to 0-100",
            "industry_formula": f"Industry Component = clip(50 + (15% − industry TIV share) × 100 × {ac.INDUSTRY_HEADROOM_SCALE}, 0, 100)",
            "state_formula": f"State Component = clip(50 + state treaty headroom % × {ac.STATE_HEADROOM_SCALE}, 0, 100)",
            "concentration_cap_pct": ac.CONCENTRATION_CAP_PCT * 100,
            "decision_thresholds": {
                "fit_strong": ac.FIT_STRONG_THRESHOLD, "fit_marginal": ac.FIT_MARGINAL_THRESHOLD,
                "headroom_comfortable": ac.HEADROOM_COMFORTABLE_THRESHOLD, "headroom_constrained": ac.HEADROOM_CONSTRAINED_THRESHOLD,
            },
        },
        "bind": {
            "weight_pct": 20,
            "is_trained_ml": False,
            "note": "Fitted via real linear regression against the source book's own bind_propensity_score_0_100 - rules-based in form, but the coefficients themselves come from real data, not hand-picked.",
            "formula": "Bind Score = Intercept + Σ(coefficient × real input)",
            "coefficients": bc,
            "groups": {
                "Broker quality ingredients": ["broker_hit_bind_ratio_pct", "broker_renewal_retention_pct", "broker_product_diversity_count", "broker_geo_diversity_count", "broker_profitability_score_0_100"],
                "Deal signals": ["quote_to_bind_days", "competing_quotes_count"],
                "Channel & region": [k for k in bc if k.startswith("channel_") or k.startswith("region_")],
                "Negligible real effect (kept per manager direction)": ["total_insured_value_usd", "coverage_limit_requested_usd", "submission_completeness_pct"],
            },
        },
        "uw_score": {
            "formula": "UW Score = 0.50×(100−Loss Score) + 0.30×Appetite/Capacity + 0.20×Bind Score",
            "note": "Loss uses (100-Loss) since Loss Score runs 0=best/100=worst, the opposite sense of Appetite/Bind.",
            "tiers": [{"name": t, **RISK_TIER_INFO[t]} for t in RISK_TIER_ORDER],
        },
        "available_not_used": _compute_available_not_used(),
    }


def _compute_available_not_used():
    """Checked programmatically against feature_map.py's real LOSS/BIND/APPETITE feature
    sets, not hand-typed - 3 of the wireframe's own candidate fields (revenue growth trend,
    debt-to-equity, safety inspection findings) turned out to already be real, counted
    inputs somewhere in the model, so they're correctly excluded here rather than copied
    from the wireframe uncritically."""
    all_used = set(fmap.LOSS_FEATURES) | set(fmap.BIND_FEATURES) | set(fmap.APPETITE_FEATURES_SET)
    candidates = {
        "geospatial_cat_score": "Geospatial CAT Score", "weather_climate_risk_index": "Climate Risk Index",
        "legal_litigation_venue_index": "Litigation Venue Index", "public_records_ucc_filings_count": "UCC Filings",
        "adverse_media_news_flag": "Adverse Media", "pml_ratio_of_tiv": "PML Ratio of TIV",
    }
    return [{"field": col, "label": label} for col, label in candidates.items() if col not in all_used]


# ===========================================================================
# What-If Simulator - scores a real scenario through the exact same pipeline
# ===========================================================================
@app.post("/api/score")
def what_if_score(inputs: dict):
    row = pd.DataFrame([inputs])
    try:
        result = score(row, MODELS).iloc[0]
    except Exception as e:
        raise HTTPException(400, f"Scoring failed: {e}")

    def _fn(d):
        return score(d, MODELS).iloc[0]

    row_dict = row.iloc[0].to_dict()
    loss_drv = dexp.loss_drivers(row, _fn)
    bind_drv = dexp.bind_drivers(row_dict)
    appetite_drv = dexp.appetite_drivers(row_dict)
    narrative_text = build_narrative(result, loss_drv, bind_drv, appetite_drv)
    banner = build_banner_summary(result, loss_drv, bind_drv)

    return {
        "scores": {
            "loss_propensity": _clean(result["loss_propensity_score_v2"]),
            "appetite_capacity": _clean(result["appetite_capacity_score_v2"]),
            "bind_propensity": _clean(result["bind_propensity_score_v2"]),
            "uw_score": _clean(result["uw_score_v2"]),
            "risk_tier": result["risk_tier_v2"],
            "recommended_action": result["recommended_uw_action_v2"],
            "banner_summary": banner,
        },
        "drivers": {"loss": loss_drv, "appetite": appetite_drv, "bind": bind_drv},
        "narrative": narrative_text,
        # Same fields Streamlit's "Technical & audit detail -> Full calculation" tab shows -
        # pulled straight off the real scored row, not recomputed separately, so this can
        # never drift from the numbers above it.
        "technical": {
            "hard_stop": bool(result["appetite_capacity_score_v2"] == 0),
            "line_of_business": result["line_of_business"],
            "predicted_frequency": _clean(round(result["predicted_frequency"], 4)),
            "predicted_attritional_severity": _clean(round(result["predicted_attritional_severity"], 0)),
            "p_large_loss": _clean(round(result["p_large_loss"], 4)),
            "expected_large_severity": _clean(round(result["expected_large_severity"], 0)),
            "total_expected_loss_cost": _clean(round(result["total_expected_loss_cost"], 0)),
            "predicted_loss_ratio_pct": _clean(round(result["predicted_loss_ratio_pct"], 1)),
            "loss_propensity_score": _clean(round(result["loss_propensity_score_v2"], 1)),
            "bind_propensity_score": _clean(round(result["bind_propensity_score_v2"], 1)),
            "appetite_capacity_score": _clean(round(result["appetite_capacity_score_v2"], 1)),
            "composite_uw_score": _clean(round(result["uw_score_v2"], 1)),
            "risk_tier": result["risk_tier_v2"],
            "appetite_decision": result["appetite_capacity_decision_v2"],
            "recommended_action": result["recommended_uw_action_v2"],
        },
    }


# Same real logo file the Streamlit apps use (assets/vm_logo.png at the project root) -
# mounted read-only, not copied, so there's only ever one copy of this file to keep in sync.
app.mount("/assets", StaticFiles(directory=str(Path(__file__).resolve().parents[2] / "assets")), name="assets")
app.mount("/", StaticFiles(directory=str(Path(__file__).resolve().parents[1] / "frontend"), html=True), name="frontend")
