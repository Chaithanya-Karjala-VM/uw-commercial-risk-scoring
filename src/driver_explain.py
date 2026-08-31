"""
Live, per-submission driver attribution for the What-If Simulator's "Why This Score" panel
and "Reason codes" table. Every number here is derived from the SAME formulas/models already
used for the real score - this module only decomposes them, it never changes a prediction.

Bind Propensity and Appetite/Capacity are exact linear formulas, so each field's real
contribution is computed directly from the fitted coefficients: coefficient x (this
submission's value - the real book's own average value for that field) - exact, not
estimated, no extra scoring calls needed.

Loss Propensity has no such closed form - it's a blend of 3 separately-trained models
(frequency, severity, large-loss), not one linear formula. For that one, each candidate
field's contribution is measured by literally re-scoring the submission with that field
swapped to its real book baseline and reading the change in loss_propensity_score_v2 - the
same perturbation technique already used and verified throughout this project's own
validation work. This naturally handles a composite, multi-model score that a single
off-the-shelf explainer (e.g. SHAP's fast per-model paths) can't reach in one pass, without
adding a new dependency.
"""
import pandas as pd
from lob_config import group_for_lob
from bind_propensity import BIND_COEFFICIENTS, REGION_OPTIONS, CHANNEL_OPTIONS, populate_broker_fields
from appetite_capacity import (
    hard_rule_pass, cat_tiv_penalty, BROKER_TIER_MAP,
    INDUSTRY_HEADROOM_SCALE, STATE_HEADROOM_SCALE, STATE_HEADROOM_AVERAGES,
    INDUSTRY_CONCENTRATION_BY_NAICS, CONCENTRATION_CAP_PCT,
)

# real book medians/most-common values (property+liability combined where applicable) -
# computed 2026-08 directly from the source book:
#   df[<numeric fields>].median()  /  df[<categorical field>].mode()
# the reference point every driver contribution below is measured against - "how does this
# submission differ from a typical real one," not an arbitrary made-up neutral value.
BOOK_BASELINES = {
    "total_insured_value_usd": 472000, "building_age_years": 24, "roof_age_years": 15,
    "num_stories": 3, "square_footage": 3260, "fire_protection_class_iso_ppc": 5,
    "distance_to_fire_station_miles": 1.5, "distance_to_fire_hydrant_miles": 0.24,
    "distance_to_coast_miles": 63, "distance_to_fault_line_miles": 74,
    "distance_to_nearest_water_body_miles": 3.5,
    "flood_zone_tier": "Moderate", "earthquake_zone_tier": "Low",
    "wildfire_zone_tier": "Moderate", "wind_zone_tier": "Moderate",
    "construction_type": "Non-Combustible", "sprinklered_flag": "Y",
    "number_of_employees": 31, "avg_wage_per_employee_usd": 70000, "avg_employee_age": 41,
    "avg_weekly_work_hours": 38.1, "industry_class_naics": "Manufacturing",
    "claim_count_last_3_years": 1, "large_loss_count_last_5_years": 0, "claims_free_years": 2,
    "financial_stability_score": 57, "business_tenure_years": 13, "revenue_growth_trend_pct": 6.0,
    "debt_to_equity_ratio": 1.0,
    # bind-formula ingredients - real book-wide averages (these fields are auto-populated
    # from broker-tier averages for a live submission, not typed in - see bind_propensity.py)
    "broker_hit_bind_ratio_pct": 38.06, "broker_renewal_retention_pct": 95.44,
    "broker_product_diversity_count": 2.90, "broker_geo_diversity_count": 3.75,
    "broker_profitability_score_0_100": 41.04, "quote_to_bind_days": 15.98,
    "competing_quotes_count": 1.80, "submission_completeness_pct": 91.42,
    "broker_tier_num": 2.5,  # midpoint between Silver(2) and Gold(3) - a neutral reference tier
}

FIELD_LABELS = {
    "total_insured_value_usd": "Total Insured Value", "claim_count_last_3_years": "Claim Count (3yr)",
    "large_loss_count_last_5_years": "Large-Loss Count (5yr)", "claims_free_years": "Claims-Free Years",
    "flood_zone_tier": "Flood Zone", "earthquake_zone_tier": "Earthquake Zone",
    "wildfire_zone_tier": "Wildfire Zone", "wind_zone_tier": "Wind Zone",
    "debt_to_equity_ratio": "Debt-to-Equity Ratio", "sprinklered_flag": "Sprinkler Status",
    "construction_type": "Construction Type", "roof_age_years": "Roof Age",
    "financial_stability_score": "Financial Stability Score",
    "distance_to_fire_hydrant_miles": "Distance to Fire Hydrant",
    "business_tenure_years": "Business Tenure", "revenue_growth_trend_pct": "Revenue Growth Trend",
    "number_of_employees": "Number of Employees", "industry_class_naics": "Industry Class",
    "avg_weekly_work_hours": "Avg. Weekly Work Hours", "avg_employee_age": "Avg. Employee Age",
    "avg_wage_per_employee_usd": "Avg. Wage per Employee",
    "competing_quotes_count": "Competing Quotes Count", "quote_to_bind_days": "Quote-to-Bind Days",
    "target_growth_segment_flag": "Target Growth Segment",
}

PROPERTY_LOSS_CANDIDATES = [
    "total_insured_value_usd", "claim_count_last_3_years", "flood_zone_tier", "earthquake_zone_tier",
    "wildfire_zone_tier", "wind_zone_tier", "debt_to_equity_ratio", "sprinklered_flag",
    "construction_type", "roof_age_years", "financial_stability_score",
    "distance_to_fire_hydrant_miles", "business_tenure_years", "claims_free_years",
]
LIABILITY_LOSS_CANDIDATES = [
    "number_of_employees", "claim_count_last_3_years", "financial_stability_score",
    "industry_class_naics", "avg_weekly_work_hours", "avg_employee_age",
    "avg_wage_per_employee_usd", "debt_to_equity_ratio", "claims_free_years",
    "large_loss_count_last_5_years", "business_tenure_years", "revenue_growth_trend_pct",
]


def _normalize(drivers, top_n=5):
    """drivers: list of (label, raw signed contribution - positive means this field is
    pushing the score in the WORSE direction vs. the book-typical case). Returns the top_n
    ranked by size, with weight normalized so the single biggest driver reads as 100."""
    drivers = [d for d in drivers if abs(d[1]) > 1e-6]
    drivers.sort(key=lambda d: -abs(d[1]))
    drivers = drivers[:top_n]
    if not drivers:
        return []
    biggest = abs(drivers[0][1])
    return [
        {"label": label, "direction": "increases" if val > 0 else "decreases",
         "weight": round(abs(val) / biggest * 100, 1)}
        for label, val in drivers
    ]


def loss_drivers(row_df, score_fn, top_n=5):
    """row_df: one-row DataFrame, already scored once by the caller. score_fn: a callable
    taking a one-row DataFrame and returning the scored Series (so this module doesn't need
    to import scoring.py's models directly).

    Compares predicted_loss_ratio_pct (the RAW, uncapped ratio), not loss_propensity_score_v2
    (which is clipped to [0,100] for display). Found 2026-08: a submission whose real loss
    ratio is already, say, 224% stays displayed at exactly 100 no matter which single field
    gets swapped to baseline - almost every field's swap still leaves the ratio well over
    100%, so comparing the CLIPPED score made nearly every real driver look like it had zero
    effect, and falsely spotlighted whichever one field's swap happened to cross the 100%
    boundary as the only "driver." The raw ratio has no such ceiling, so it reflects each
    field's genuine relative contribution even when the account is badly out of range."""
    lob = row_df["line_of_business"].iloc[0]
    group = group_for_lob(lob)
    candidates = PROPERTY_LOSS_CANDIDATES if group == "property" else LIABILITY_LOSS_CANDIDATES

    base_score = score_fn(row_df)["predicted_loss_ratio_pct"]
    results = []
    for field in candidates:
        if field not in row_df.columns or field not in BOOK_BASELINES:
            continue
        if row_df[field].iloc[0] == BOOK_BASELINES[field]:
            continue  # already at baseline - no perturbation needed, saves a call
        neutral_row = row_df.copy()
        neutral_row[field] = BOOK_BASELINES[field]
        neutral_score = score_fn(neutral_row)["predicted_loss_ratio_pct"]
        delta = base_score - neutral_score
        results.append((FIELD_LABELS.get(field, field), delta))
    return _normalize(results, top_n)


def bind_drivers(row_dict, top_n=5):
    """Exact, closed-form - no re-scoring. The 7 broker-detail fields are auto-populated
    entirely from Broker Tier (not independently chosen), so their combined contribution is
    reported as one "Broker Tier" driver rather than 7 redundant near-duplicate lines."""
    # row_dict from the live form only has broker_tier, not the 7 detail fields the formula
    # actually runs on (those are no longer live inputs - see bind_propensity.py) - populate
    # them the exact same way scoring.py does internally, or every one of those 7 terms
    # would silently compare a missing value against itself and read as zero contribution.
    if "broker_hit_bind_ratio_pct" not in row_dict:
        populated = populate_broker_fields(pd.DataFrame([row_dict])).iloc[0]
        row_dict = {**row_dict, **populated.to_dict()}
    b = BOOK_BASELINES
    broker_fields = ["broker_hit_bind_ratio_pct", "broker_renewal_retention_pct",
                      "broker_product_diversity_count", "broker_geo_diversity_count",
                      "broker_profitability_score_0_100", "quote_to_bind_days",
                      "submission_completeness_pct"]
    broker_total = sum(BIND_COEFFICIENTS[f] * (row_dict.get(f, b[f]) - b[f]) for f in broker_fields)

    results = []
    if row_dict.get("channel") == "Direct":
        # No broker relationship on a Direct submission - nothing real to attribute to a
        # "Broker Tier" driver here (the neutral profile behind the scenes is a scoring
        # mechanic, not something the underwriter chose - see bind_propensity.py).
        pass
    else:
        tier = row_dict.get("broker_tier", "Silver")
        results.append((f"Broker Tier: {tier}", broker_total))

    cqc = row_dict.get("competing_quotes_count", b["competing_quotes_count"])
    results.append(("Competing Quotes Count", BIND_COEFFICIENTS["competing_quotes_count"] * (cqc - b["competing_quotes_count"])))

    region = row_dict.get("risk_location_region", "Gulf Coast")
    if region in REGION_OPTIONS[1:]:
        results.append((f"Region: {region}", BIND_COEFFICIENTS[f"region_{region}"]))

    channel = row_dict.get("channel", "Broker")
    if channel in CHANNEL_OPTIONS[1:]:
        results.append((f"Channel: {channel}", BIND_COEFFICIENTS[f"channel_{channel}"]))

    return _normalize(results, top_n)


def appetite_drivers(row_dict, top_n=5):
    """Exact, closed-form. If the hard-rule gate fails, that's the only driver that matters -
    it forces the score to 0 regardless of everything else, so nothing else is shown."""
    row_df = pd.DataFrame([row_dict])
    if not hard_rule_pass(row_df).iloc[0]:
        return [{"label": "Hard-Rule Gate Failed (sanctions/insolvency/restricted ops/region-LOB)",
                  "direction": "increases", "weight": 100.0}]

    from scoring import APPETITE_FIT_COEFFICIENTS
    c = APPETITE_FIT_COEFFICIENTS
    b = BOOK_BASELINES
    FIT_WEIGHT, HEADROOM_WEIGHT = 0.75, 0.25  # must match compute_appetite_capacity_score's own blend

    fsc = row_dict.get("financial_stability_score", b["financial_stability_score"])
    results = [("Financial Stability Score", FIT_WEIGHT * c["financial_stability_score"] * (fsc - b["financial_stability_score"]))]

    if row_dict.get("channel") != "Direct":
        tier_num = BROKER_TIER_MAP.get(row_dict.get("broker_tier"), 2)
        results.append((f"Broker Tier: {row_dict.get('broker_tier', 'Silver')}",
                         FIT_WEIGHT * c["broker_tier_num"] * (tier_num - b["broker_tier_num"])))

    if row_dict.get("target_growth_segment_flag"):
        results.append(("Target Growth Segment", FIT_WEIGHT * c["target_growth_segment_flag"]))

    penalty = cat_tiv_penalty(row_df)[0]
    if penalty != 0:
        results.append(("Severe CAT Zone + High TIV Penalty", FIT_WEIGHT * penalty))

    # capacity headroom - industry concentration and state reinsurance headroom, each
    # measured against their own formula's neutral midpoint (50)
    industry = row_dict.get("industry_class_naics")
    concentration = INDUSTRY_CONCENTRATION_BY_NAICS.get(industry, 0)
    industry_headroom_pct = (CONCENTRATION_CAP_PCT - concentration) * 100
    industry_component = max(0, min(100, 50 + industry_headroom_pct * INDUSTRY_HEADROOM_SCALE))
    results.append((f"Industry Concentration: {industry}", HEADROOM_WEIGHT * (industry_component - 50)))

    state = row_dict.get("risk_state")
    state_headroom_pct = STATE_HEADROOM_AVERAGES.get(state, 0.20) * 100
    state_component = max(0, min(100, 50 + state_headroom_pct * STATE_HEADROOM_SCALE))
    results.append((f"State Capacity: {state}", HEADROOM_WEIGHT * (state_component - 50)))

    # sign convention: these are "fit"/"headroom" contributions where POSITIVE means better
    # appetite (unlike loss/bind above, where positive means worse) - flip here so direction
    # labeling stays consistent everywhere ("increases" always means "moves the risk view up")
    return _normalize([(label, -val) for label, val in results], top_n)
