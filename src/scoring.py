"""
Shared scoring logic - loads the 6 trained sub-models, scores any dataframe of submissions.
Both combine_scores.py and the Streamlit app import this - one implementation only.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import joblib
from lob_config import group_for_lob, encode_ordinal_tiers, region_for_state, CURRENT_YEAR
from bind_propensity import compute_bind_propensity, populate_broker_fields, BIND_COEFFICIENTS
from appetite_capacity import compute_appetite_capacity_score, compute_appetite_decision, hard_rule_pass
from decision_matrix_reference import tier_for_score, RISK_TIER_INFO, RISK_TIER_PREMIUM_LOADING

# fitted appetite coefficients - see appetite_capacity.verify_appetite_formula() docstring
# for how these were derived and re-verify if the source data changes
APPETITE_FIT_COEFFICIENTS = {
    "intercept": -46.0188, "appetite_hard_rule_pass_flag": 56.5342,
    "target_growth_segment_flag": 9.1769, "financial_stability_score": 0.1807,
    "broker_tier_num": 2.3931, "submission_completeness_pct": 0.3138,
    "coverage_limit_requested_usd": 0.0,
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = str(PROJECT_ROOT / "models")

# real per-LOB premium rates - property on TIV, liability on payroll - used only when a
# submission genuinely has no premium yet (a brand-new one); the existing book's own real,
# already-varying premium is always preferred when present
PREMIUM_RATE_BY_LOB = {
    "Commercial Property": 0.004484, "Business Owners Package (BOP)": 0.005045,
    "Inland Marine": 0.004270, "Workers Compensation": 0.004480,
    "General Liability": 0.002230, "Professional Liability": 0.001734,
    "Umbrella/Excess": 0.001372, "Commercial Auto": 0.001348,
}
# fallback only - used if a submission's line_of_business isn't one of the 8 rates above
# (e.g. a genuinely new LOB the book has never seen). Previously a plain dict lookup that
# raised KeyError and crashed scoring outright; this keeps scoring alive with a reasonable
# blended estimate instead.
BOOK_AVG_PREMIUM_RATE = sum(PREMIUM_RATE_BY_LOB.values()) / len(PREMIUM_RATE_BY_LOB)
PROPERTY_LOBS = ["Commercial Property", "Business Owners Package (BOP)", "Inland Marine"]


def load_models():
    return {
        "frequency": joblib.load(f"{MODELS_DIR}/frequency_model.joblib"),
        "severity": joblib.load(f"{MODELS_DIR}/severity_model.joblib"),
        "large_loss": joblib.load(f"{MODELS_DIR}/large_loss_model.joblib"),
    }


def _prep_X(sub_df, cat_cols):
    X = sub_df.copy()
    for c in cat_cols:
        X[c] = X[c].astype(object).where(X[c].notna(), None)
    return X


def compute_premium(lob, total_insured_value_usd=None, number_of_employees=None,
                     avg_wage_per_employee_usd=None):
    rate = PREMIUM_RATE_BY_LOB.get(lob, BOOK_AVG_PREMIUM_RATE)
    if lob in PROPERTY_LOBS:
        return round((total_insured_value_usd or 0) * rate, -1)
    payroll = (number_of_employees or 0) * (avg_wage_per_employee_usd or 0)
    return round(payroll * rate, -1)


# Real regression against the full 20,000-row book (2026-10), not invented weights - fit
# tested against every other candidate field in the Financial & Operational section (safety
# inspection findings, revenue growth trend, debt-to-equity ratio); none added meaningful
# signal over these two alone (combined R^2 moved from 0.1320 to 0.1322 - noise-level), so
# they're the only two used. Per manager direction: the What-If Simulator's Financial
# Stability Score input is now estimated from these two fields instead of a free-standing
# slider the underwriter had to guess a number for. Disclosed honestly in the UI as an
# estimate (R^2=0.13 - genuinely explains only ~13% of the real score's variance), not
# presented as precise - see FINANCIAL_STABILITY_FIT_NOTE.
FINANCIAL_STABILITY_INTERCEPT = 48.89
FINANCIAL_STABILITY_TENURE_COEF = 0.4382
FINANCIAL_STABILITY_SAFETY_COEF = 4.3868
FINANCIAL_STABILITY_FIT_NOTE = "Estimated from Business Tenure and Safety Program status above."


def estimate_financial_stability_score(business_tenure_years, safety_program_flag):
    raw = (FINANCIAL_STABILITY_INTERCEPT
           + FINANCIAL_STABILITY_TENURE_COEF * (business_tenure_years or 0)
           + FINANCIAL_STABILITY_SAFETY_COEF * (1 if safety_program_flag else 0))
    return int(round(min(max(raw, 5), 100)))  # clipped to the real book's own observed 5-100 range


# real, honest indicators a submission can carry BEFORE any claim shows up in its history -
# used for the Emerging Risk Watch List. Replaces the source file's own prospective_risk_flags/
# prospective_risk_flag_count, which are frozen values from the original stored formula and
# can't be regenerated for a brand-new submission or reflect anything from this pipeline.
WATCH_FLAG_DEFINITIONS = [
    ("Low financial stability score", lambda r: (r.get("financial_stability_score") or 100) < 40),
    ("High debt-to-equity ratio", lambda r: (r.get("debt_to_equity_ratio") or 0) > 2.0),
    ("Declining revenue trend", lambda r: (r.get("revenue_growth_trend_pct") or 0) < -5),
    ("Short business tenure (<3 yrs)", lambda r: (r.get("business_tenure_years") or 100) < 3),
    ("Safety inspection deficiencies noted", lambda r: r.get("safety_inspection_findings") in
        ("Moderate Issues – Corrective Action Required", "Major Deficiencies Identified")),
    ("Severe CAT zone exposure", lambda r: any(
        (r.get(c) or 0) >= 3 for c in ["flood_zone_tier", "earthquake_zone_tier", "wildfire_zone_tier", "wind_zone_tier"])),
    ("High competing-quote pressure", lambda r: (r.get("competing_quotes_count") or 0) >= 5),
]


def _watch_flags(row):
    hits = [label for label, test in WATCH_FLAG_DEFINITIONS if test(row)]
    return len(hits), "; ".join(hits) if hits else "None"


def _decline_review_reason(row):
    if row["appetite_capacity_score_v2"] == 0:
        return ("Fails compliance/appetite hard-rule gate (sanctions, insolvency, restricted "
                "operations, or region/LOB restriction) - decline regardless of loss or bind view")
    concerns = []
    if row["loss_propensity_score_v2"] >= 60:
        concerns.append(f"elevated loss propensity (predicted loss ratio {row['predicted_loss_ratio_pct']:.0f}%)")
    if row["appetite_capacity_score_v2"] < 45:
        concerns.append("below-average appetite/capacity fit")
    if row["bind_propensity_score_v2"] < 35:
        concerns.append("lower likelihood to bind")
    if not concerns:
        return "Balanced profile - no single dominant concern, refer for underwriter judgment"
    text = "; ".join(concerns)
    return text[0].upper() + text[1:]


def _priority_strength_reason(row):
    strengths = []
    if row["loss_propensity_score_v2"] <= 25:
        strengths.append("strong loss profile")
    if row["bind_propensity_score_v2"] >= 50:
        strengths.append("high likelihood to bind")
    if row["appetite_capacity_score_v2"] >= 65:
        strengths.append("strong appetite/capacity fit")
    if not strengths:
        return "Solid overall profile, within appetite"
    text = ", ".join(strengths)
    return text[0].upper() + text[1:]


def score(df, models, term_col="policy_term_months"):
    out = df.copy()
    out = encode_ordinal_tiers(out)  # must match training - Low/Moderate/High/Severe -> 0-3

    # broker-detail fields (hit/bind ratio, retention, diversity, profitability, quote-to-
    # bind days, completeness) are no longer collected as live input for a new submission -
    # only auto-populate from broker-tier averages when they're genuinely missing, never
    # overwrite the real per-submission historical values already in the book
    broker_detail_fields = ["broker_hit_bind_ratio_pct", "broker_renewal_retention_pct",
                             "broker_product_diversity_count", "broker_geo_diversity_count",
                             "broker_profitability_score_0_100", "quote_to_bind_days",
                             "submission_completeness_pct"]
    needs_broker_populate = any(f not in out.columns or out[f].isna().any() for f in broker_detail_fields)
    if needs_broker_populate:
        out = populate_broker_fields(out)

    # a new submission only collects State (not Region) - derive Region for whichever
    # model needs it, using the confirmed 1:1 mapping in lob_config.py
    if "risk_location_region" not in out.columns or out["risk_location_region"].isna().any():
        out["risk_location_region"] = out.get("risk_state", pd.Series(index=out.index)).apply(region_for_state)

    # The Streamlit form always derives building_age_years = CURRENT_YEAR - year_built itself
    # before calling score(), so this has never fired for a live submission - found instead
    # while random-testing score() directly with a hand-built row that (like any other future
    # caller - a batch script, an API, a test harness) supplied year_built without separately
    # deriving building_age_years, and got a raw KeyError instead of a usable result. Same
    # defensive-derivation pattern as risk_location_region above, so score() itself is crash-safe
    # for any caller, not just the one path that currently happens to always do this correctly.
    if "year_built" in out.columns and ("building_age_years" not in out.columns or out["building_age_years"].isna().any()):
        derived_age = CURRENT_YEAR - out["year_built"]
        if "building_age_years" in out.columns:
            out["building_age_years"] = out["building_age_years"].fillna(derived_age)
        else:
            out["building_age_years"] = derived_age

    out["_group"] = out["line_of_business"].apply(group_for_lob)

    has_real_premium = ("indicated_annual_premium_usd" in out.columns
                         and out["indicated_annual_premium_usd"].notna().all())

    reset_cols = ["predicted_frequency", "predicted_attritional_severity", "p_large_loss",
                  "expected_large_severity", "total_expected_loss_cost", "predicted_loss_ratio_pct",
                  "loss_propensity_score_v2"]
    if not has_real_premium:
        reset_cols.append("indicated_annual_premium_usd")
    for col in reset_cols:
        out[col] = np.nan

    for group in ["property", "liability"]:
        mask = out["_group"] == group
        if not mask.any():
            continue
        sub = out.loc[mask]

        freq_saved = models["frequency"][group]
        Xf = _prep_X(sub[freq_saved["features"]], freq_saved["cat_cols"])
        exposure_years = (sub[term_col] / 12).clip(lower=0.01)
        # frequency_model.py now predicts a PER-EXPOSURE-UNIT rate (added 2026-08 - see that
        # file's module docstring), so the model's raw output has to be multiplied by both
        # exposure_years AND the exposure unit itself to recover an actual predicted claim
        # count - multiplying by exposure_years alone (the old behavior) understated it by
        # however large the unit value was.
        unit_cfg = freq_saved["exposure_unit"]
        exposure_units = (sub[unit_cfg["col"]] / unit_cfg["scale"]).clip(lower=unit_cfg["floor"])
        predicted_frequency = freq_saved["model"].predict(Xf) * exposure_years * exposure_units

        sev_saved = models["severity"][group]
        Xs = _prep_X(sub[sev_saved["features"]], sev_saved["cat_cols"])
        predicted_attritional_severity = sev_saved["model"].predict(Xs)

        large_saved = models["large_loss"][group]
        Xl = _prep_X(sub[large_saved["features"]], large_saved["cat_cols"])
        p_large = large_saved["occurrence_model"].predict_proba(Xl)[:, 1]
        # .fillna(): a LOB the large-loss model has never seen (not in thresholds_by_lob) used
        # to leave this as NaN, which silently poisoned every downstream number for that row
        # (expected_large_severity, total_expected_loss_cost, the final score - all NaN, no
        # error). Falls back to the MEDIAN THRESHOLD WITHIN THIS SAME GROUP ONLY (property
        # thresholds are PML/TIV-based and run far higher $ than liability's AOA-based ones -
        # blending both scales together would make the fallback meaningless either direction).
        group_thresholds = {lob: v for lob, v in models["large_loss"]["thresholds_by_lob"].items()
                             if group_for_lob(lob) == group}
        group_median_threshold = pd.Series(group_thresholds).median()
        threshold_by_lob = sub["line_of_business"].map(models["large_loss"]["thresholds_by_lob"]) \
            .fillna(group_median_threshold)
        gpd_mean_excess = stats.genpareto.mean(large_saved["gpd_shape"], scale=large_saved["gpd_scale"])
        flat_large_severity = threshold_by_lob + gpd_mean_excess

        # same NaN-on-unknown-LOB risk as threshold_by_lob above, and the same group-scoped fix
        group_avg_severities = {lob: v for lob, v in models["severity"]["lob_avg_severity"].items()
                                 if group_for_lob(lob) == group}
        lob_avg = sub["line_of_business"].map(models["severity"]["lob_avg_severity"]) \
            .fillna(pd.Series(group_avg_severities).median())
        severity_ratio = (predicted_attritional_severity / lob_avg).clip(0.3, 3.0)
        expected_large_severity = flat_large_severity * severity_ratio

        attritional_freq = np.clip(predicted_frequency - p_large, 0, None)
        total_expected_loss_cost = (attritional_freq * predicted_attritional_severity
                                     + p_large * expected_large_severity)

        if has_real_premium:
            premium = sub["indicated_annual_premium_usd"]
        else:
            premium = sub.apply(lambda r: compute_premium(
                r["line_of_business"],
                total_insured_value_usd=r.get("total_insured_value_usd"),
                number_of_employees=r.get("number_of_employees"),
                avg_wage_per_employee_usd=r.get("avg_wage_per_employee_usd"),
            ), axis=1)

        predicted_loss_ratio_pct = (total_expected_loss_cost / premium) * 100
        # 0-100 scale, per confirmed direction: 0 = best risk, 100 = worst
        loss_propensity_score = predicted_loss_ratio_pct.clip(0, 100).round(1)

        out.loc[mask, "predicted_frequency"] = predicted_frequency.round(4)
        out.loc[mask, "predicted_attritional_severity"] = predicted_attritional_severity.round(0)
        out.loc[mask, "p_large_loss"] = p_large.round(4)
        out.loc[mask, "expected_large_severity"] = expected_large_severity.round(0)
        out.loc[mask, "total_expected_loss_cost"] = total_expected_loss_cost.round(0)
        out.loc[mask, "indicated_annual_premium_usd"] = premium.round(0)
        out.loc[mask, "predicted_loss_ratio_pct"] = predicted_loss_ratio_pct.round(1)
        out.loc[mask, "loss_propensity_score_v2"] = loss_propensity_score

    out["bind_propensity_score_v2"] = compute_bind_propensity(out, BIND_COEFFICIENTS)
    out["appetite_capacity_score_v2"] = compute_appetite_capacity_score(out, APPETITE_FIT_COEFFICIENTS)
    out["uw_score_v2"] = np.round(
        0.30 * out["appetite_capacity_score_v2"] + 0.20 * out["bind_propensity_score_v2"]
        + 0.50 * (100 - out["loss_propensity_score_v2"]), 1
    )

    # decision-layer outputs, all derived live from the v2 scores above - NOT the source
    # file's frozen original columns (risk_tier, appetite_capacity_decision, etc.), which
    # never move no matter what changes in the scoring pipeline. See scoring.py's module
    # docstring / project notes for why that distinction matters.
    #
    # real bug found 2026-08 while building the What-If Simulator's narrative panel: the
    # documented rule is "Decline: < 38, OR any hard-rule fail" (decision_matrix_reference.py's
    # own RISK_TIER_INFO text), but tier_for_score() only ever looks at the blended uw_score_v2
    # number. Since appetite is only 30% of that blend, a hard-rule failure (which forces
    # appetite to exactly 0) can still leave uw_score_v2 sitting in Standard/High territory if
    # loss and bind happen to be good enough to carry the other 70% - e.g. a submission with a
    # sanctions hit but otherwise clean loss/bind scores landed in "High" with a "rate
    # adjustment" action instead of "Decline"/"refer," directly contradicting the documented
    # rule. Force Decline whenever the hard-rule gate failed, regardless of what the blended
    # score alone would suggest.
    hard_rule_failed = out["appetite_capacity_score_v2"] == 0
    out["risk_tier_v2"] = np.where(hard_rule_failed, "Decline", out["uw_score_v2"].apply(tier_for_score))
    out["appetite_capacity_decision_v2"] = compute_appetite_decision(out, APPETITE_FIT_COEFFICIENTS)
    out["recommended_uw_action_v2"] = out["risk_tier_v2"].map(lambda t: RISK_TIER_INFO[t]["uw_action"])
    # the technical premium (indicated_annual_premium_usd) has no risk adjustment baked in -
    # this applies the SAME rate loading the app already documents per tier (RISK_TIER_INFO's
    # pricing_guidance text), as a real number rather than only descriptive text. Decline maps
    # to NaN - per its own pricing_guidance ("Refer - do not auto-quote"), there is no adjusted
    # premium to show, since it isn't quoted at all.
    loading = out["risk_tier_v2"].map(RISK_TIER_PREMIUM_LOADING)
    out["premium_loading_pct"] = loading
    out["quoted_premium_usd"] = np.where(
        loading.notna(), (out["indicated_annual_premium_usd"] * (1 + loading.fillna(0))).round(0), np.nan
    )
    out["decline_review_reason_v2"] = out.apply(_decline_review_reason, axis=1)
    out["priority_strength_reason_v2"] = out.apply(_priority_strength_reason, axis=1)
    watch = out.apply(_watch_flags, axis=1, result_type="expand")
    out["prospective_risk_flag_count_v2"] = watch[0]
    out["prospective_risk_flags_v2"] = watch[1]

    return out.drop(columns=["_group"])
