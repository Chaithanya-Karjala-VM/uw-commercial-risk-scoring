"""
Appetite/Capacity - rules-based, not trained (same reasoning as bind_propensity.py: no
historical outcome exists to train against, this is a policy decision with a known
structure, not a pattern to learn).

Hard-rule veto: the 4 named flags (sanctions, insolvency, restricted ops, region x LOB) are
a confirmed, exact NECESSARY condition - any one True means 0% pass, verified with zero
exceptions across all 20,000 rows. They are NOT sufficient - among rows where all 4 are
clear, only 70.4% actually pass. Tested TIV thresholds, Line of Business, and CAT zone
combinations as possible explanations; none produced a clean deterministic rule. Open item,
not resolved - see project notes.

The CAT-zone + TIV interaction is approximated as a soft penalty on the fit score, not a
hard veto, for the same reason - no clean threshold found despite testing.

Concentration cap is confirmed to be a flat 15% for every industry (checked directly - zero
variation across all 10 industry classes), not a per-industry lookup table.
"""
import numpy as np
import pandas as pd

CONCENTRATION_CAP_PCT = 0.15  # flat cap, confirmed identical across every industry class

BROKER_TIER_MAP = {"Platinum": 4, "Gold": 3, "Silver": 2, "Bronze": 1}


def hard_rule_pass(df):
    """Exact, verified deterministic veto - any one of these 4 flags true means automatic fail."""
    return (
        (df["sanctions_ofac_watchlist_flag"] == 0)
        & (df["prior_insolvency_bankruptcy_flag"] == 0)
        & (df["restricted_operations_flag"] == 0)
        & (df["region_lob_restricted_flag"] == 0)
    )


def cat_tiv_penalty(df):
    """Approximated soft penalty, NOT an exact reconstruction - see module docstring."""
    any_severe_cat = (
        (df["flood_zone_tier"] == "Severe") | (df["earthquake_zone_tier"] == "Severe")
        | (df["wildfire_zone_tier"] == "Severe")
    )
    high_tiv = df["total_insured_value_usd"] > 2_000_000
    return np.where(any_severe_cat & high_tiv, -12, np.where(any_severe_cat, -8, 0))


def compute_appetite_fit(df, coefficients):
    passed = hard_rule_pass(df)
    broker_tier_num = df["broker_tier"].map(BROKER_TIER_MAP).fillna(2)
    score = pd.Series(coefficients["intercept"], index=df.index)
    score = score + passed.astype(float) * coefficients["appetite_hard_rule_pass_flag"]
    score = score + df["target_growth_segment_flag"].fillna(0) * coefficients["target_growth_segment_flag"]
    score = score + df["financial_stability_score"].fillna(0) * coefficients["financial_stability_score"]
    score = score + broker_tier_num * coefficients["broker_tier_num"]
    score = score + df["submission_completeness_pct"].fillna(0) * coefficients["submission_completeness_pct"]
    score = score + cat_tiv_penalty(df)
    score = np.where(passed, score.clip(0, 100), 0)  # hard rule failure forces 0 regardless of fit
    return np.round(score, 1)


def compute_capacity_headroom(df, book_for_concentration=None):
    """Concentration is computed against the current book - a brand-new submission's own
    industry share doesn't change the book meaningfully, so it's scored against the existing
    book's concentration, not recomputed with itself included."""
    reference = book_for_concentration if book_for_concentration is not None else df
    concentration_by_industry = (
        reference.groupby("industry_class_naics")["total_insured_value_usd"].sum()
        / reference["total_insured_value_usd"].sum()
    )
    concentration = df["industry_class_naics"].map(concentration_by_industry).fillna(0)
    return (CONCENTRATION_CAP_PCT - concentration) * 100  # headroom %, negative = over cap


def compute_appetite_capacity_score(df, coefficients, book_for_concentration=None):
    fit = compute_appetite_fit(df, coefficients)
    headroom = compute_capacity_headroom(df, book_for_concentration)
    headroom_component = np.clip(50 + headroom * 5, 0, 100)
    passed = hard_rule_pass(df)
    score = np.where(passed, np.clip(0.75 * fit + 0.25 * headroom_component, 0, 100), 0)
    return np.round(score, 1)


def verify_appetite_formula(df):
    """Uses the file's own STORED hard-rule flag (not our reconstruction) to measure the
    fit-score formula's accuracy in isolation - this is what gave R^2=0.977 originally.
    compute_appetite_fit() below has to use our own reconstructed flag instead, since a
    brand-new submission won't have the file's stored value - see module docstring for the
    known limitation that introduces."""
    from sklearn.linear_model import LinearRegression
    broker_tier_num = df["broker_tier"].map(BROKER_TIER_MAP).fillna(2)
    X = pd.DataFrame({
        "appetite_hard_rule_pass_flag": df["appetite_hard_rule_pass_flag"].astype(float),
        "target_growth_segment_flag": df["target_growth_segment_flag"].fillna(0),
        "financial_stability_score": df["financial_stability_score"].fillna(0),
        "broker_tier_num": broker_tier_num,
        "submission_completeness_pct": df["submission_completeness_pct"].fillna(0),
    })
    y = df["appetite_fit_score_0_100"]
    reg = LinearRegression().fit(X, y)
    pred = reg.predict(X)
    r2 = reg.score(X, y)
    mae = np.abs(pred - y).mean()
    coefs = {"intercept": reg.intercept_}
    coefs.update(dict(zip(X.columns, reg.coef_)))
    return {"r2": r2, "mae": mae, "coefficients": coefs}


if __name__ == "__main__":
    from data_loader import load_data
    df, schema, usable_for, output_cols = load_data()

    fit = verify_appetite_formula(df)
    print(f"Appetite Fit regression check: R^2={fit['r2']:.3f}, avg error={fit['mae']:.2f} points")
    print("Coefficients:", {k: round(v, 4) for k, v in fit["coefficients"].items()})

    computed_pass = hard_rule_pass(df)
    mismatch = (computed_pass != df["appetite_hard_rule_pass_flag"]).sum()
    print(f"\nHard-rule veto reconstruction: {mismatch} mismatches out of {len(df)} (should be 0)")

    AFC = fit["coefficients"]
    computed_fit = compute_appetite_fit(df, AFC)
    stored_fit = df["appetite_fit_score_0_100"]
    diff = pd.Series(computed_fit - stored_fit).abs()
    print(f"Appetite Fit reconstruction (with CAT/TIV approximation added): avg diff = {diff.mean():.2f} points")

    computed_final = compute_appetite_capacity_score(df, AFC)
    stored_final = df["appetite_capacity_score_0_100"]
    diff2 = pd.Series(computed_final - stored_final).abs()
    print(f"Appetite/Capacity final score vs stored: avg diff = {diff2.mean():.2f} points, "
          f"correlation = {pd.Series(computed_final).corr(stored_final):.3f}")
