"""
Bind Propensity - rules-based, not trained. No historical bind/no-bind outcome exists in
the source data; every bind-related score column is itself a derived output, not a fact.
Reconstructs the documented formula (broker hit/bind, retention, diversity, profitability,
deal friction) via regression against the file's own stored score.

Per manager direction: Region, TIV, and Coverage Limit are included as inputs even though
they showed negligible improvement to fit quality on this dataset (R^2 0.9272 -> 0.9274,
checked directly) - kept in because real broker/submission data at larger scale may show a
relationship this file doesn't. Their current fitted weights are small, reflecting the
actual (weak) signal in this data - not artificially inflated.

State-level granularity (31 states) was replaced with Region (8 categories) to keep this a
maintainable, readable formula rather than 31 hardcoded coefficients.

Broker detail fields (hit/bind ratio, retention, diversity, profitability, quote-to-bind
days, submission completeness) are no longer collected as live user input - an underwriter
only selects Broker Tier (or Broker Capacity Segment for a new broker). The actual values
used in the formula are auto-populated from real historical averages per tier - see
BROKER_TIER_AVERAGES below. Some of these barely vary by tier in the real data (hit/bind
ratio, quote-to-bind days, completeness sit nearly flat across all 4 tiers) while others
vary a lot (retention 93.7 -> 98.1, profitability 31.6 -> 52.0) - both are honest findings
from the real data, not something to force into a uniform spread.
"""
import numpy as np
import pandas as pd

BIND_INGREDIENTS = [
    "broker_hit_bind_ratio_pct", "broker_renewal_retention_pct",
    "broker_product_diversity_count", "broker_geo_diversity_count",
    "broker_profitability_score_0_100", "quote_to_bind_days",
    "competing_quotes_count", "submission_completeness_pct",
]
REGION_OPTIONS = ["Northeast", "Midwest", "Southeast", "West", "Southwest",
                   "Gulf Coast", "Mountain", "Pacific Northwest"]
NEW_BIND_INGREDIENTS = BIND_INGREDIENTS + ["total_insured_value_usd", "coverage_limit_requested_usd"] \
    + [f"region_{r}" for r in REGION_OPTIONS[1:]]  # first region is the reference level

# real per-tier averages, used to auto-populate the fields no longer collected live -
# computed directly from the source book, not guessed
BROKER_TIER_AVERAGES = {
    "Bronze":   {"broker_hit_bind_ratio_pct": 38.09, "broker_renewal_retention_pct": 93.67,
                 "broker_product_diversity_count": 2.14, "broker_geo_diversity_count": 2.61,
                 "broker_profitability_score_0_100": 31.61, "quote_to_bind_days": 15.94,
                 "submission_completeness_pct": 91.60},
    "Silver":   {"broker_hit_bind_ratio_pct": 38.11, "broker_renewal_retention_pct": 95.01,
                 "broker_product_diversity_count": 3.23, "broker_geo_diversity_count": 4.37,
                 "broker_profitability_score_0_100": 39.14, "quote_to_bind_days": 15.92,
                 "submission_completeness_pct": 91.38},
    "Gold":     {"broker_hit_bind_ratio_pct": 38.21, "broker_renewal_retention_pct": 96.72,
                 "broker_product_diversity_count": 2.96, "broker_geo_diversity_count": 3.67,
                 "broker_profitability_score_0_100": 48.90, "quote_to_bind_days": 16.05,
                 "submission_completeness_pct": 91.43},
    "Platinum": {"broker_hit_bind_ratio_pct": 37.46, "broker_renewal_retention_pct": 98.13,
                 "broker_product_diversity_count": 3.24, "broker_geo_diversity_count": 4.20,
                 "broker_profitability_score_0_100": 51.98, "quote_to_bind_days": 16.09,
                 "submission_completeness_pct": 91.15},
}

# for a brand-new broker with no history yet: most common real tier for that capacity
# segment - an estimate, not a precise derivation (checked directly: every segment
# actually contains all 4 tiers in the real data, just in different proportions)
NEW_BROKER_DEFAULT_TIER = {
    "Large National": "Silver", "Regional": "Silver", "Small Independent": "Bronze",
}


def populate_broker_fields(df, broker_tier_col="broker_tier"):
    """Fills in the 6 no-longer-collected broker-detail fields plus submission_completeness_pct
    from real per-tier averages. competing_quotes_count stays a live, user-entered field -
    it's deal-specific, not a property of the broker relationship."""
    df = df.copy()
    for field in BIND_INGREDIENTS:
        if field == "competing_quotes_count":
            continue
        df[field] = df[broker_tier_col].map(lambda t: BROKER_TIER_AVERAGES.get(t, BROKER_TIER_AVERAGES["Silver"])[field])
    return df

# fitted via verify_bind_formula() against bind_propensity_score_0_100 - rerun and replace
# these if the source data changes
BIND_COEFFICIENTS = {
    "intercept": -251.8211, "broker_hit_bind_ratio_pct": 0.1484, "broker_renewal_retention_pct": 2.9662,
    "broker_product_diversity_count": 0.9209, "broker_geo_diversity_count": 0.2898,
    "broker_profitability_score_0_100": 0.1778, "quote_to_bind_days": -0.1001,
    "competing_quotes_count": -2.0014, "submission_completeness_pct": 0.0017,
    "total_insured_value_usd": -0.0, "coverage_limit_requested_usd": 0.0,
    "region_Midwest": 0.1093, "region_Southeast": 0.0843, "region_West": 0.0489,
    "region_Southwest": 0.0696, "region_Gulf Coast": -0.0, "region_Mountain": 0.1559,
    "region_Pacific Northwest": 0.0207,
}


def _build_design_matrix(df):
    X = df[BIND_INGREDIENTS].fillna(0).copy()
    X["total_insured_value_usd"] = df.get("total_insured_value_usd", pd.Series(0, index=df.index)).fillna(0)
    X["coverage_limit_requested_usd"] = df.get("coverage_limit_requested_usd", pd.Series(0, index=df.index)).fillna(0)
    region_source = df.get("risk_location_region", pd.Series(REGION_OPTIONS[0], index=df.index)).fillna(REGION_OPTIONS[0])
    region_dummies = pd.get_dummies(region_source, prefix="region", drop_first=True)
    for r in REGION_OPTIONS[1:]:
        col = f"region_{r}"
        X[col] = region_dummies[col] if col in region_dummies.columns else 0
    return X[NEW_BIND_INGREDIENTS]


def verify_bind_formula(df):
    """Re-fits the regression fresh and reports fit quality - run any time the source data
    changes. Region/TIV/Coverage Limit are included per manager direction even though they
    added negligible fit improvement on this dataset (checked: R^2 0.9272 -> 0.9274)."""
    from sklearn.linear_model import LinearRegression
    X = _build_design_matrix(df)
    y = df["bind_propensity_score_0_100"]
    reg = LinearRegression().fit(X, y)
    pred = reg.predict(X)
    r2 = reg.score(X, y)
    mae = np.abs(pred - y).mean()
    coefs = {"intercept": reg.intercept_}
    coefs.update(dict(zip(X.columns, reg.coef_)))
    return {"r2": r2, "mae": mae, "coefficients": coefs}


def compute_bind_propensity(df, coefficients):
    X = _build_design_matrix(df)
    result = pd.Series(coefficients["intercept"], index=df.index)
    for col in X.columns:
        result = result + X[col] * coefficients[col]
    return result.clip(0, 100).round(1)


if __name__ == "__main__":
    from data_loader import load_data
    df, schema, usable_for, output_cols = load_data()

    fit = verify_bind_formula(df)
    print(f"Fresh regression check: R^2={fit['r2']:.4f}, avg error={fit['mae']:.2f} points")

    computed = compute_bind_propensity(df, BIND_COEFFICIENTS)
    stored = df["bind_propensity_score_0_100"]
    diff = (computed - stored).abs()
    print(f"Fixed coefficients vs stored: avg diff = {diff.mean():.2f} points, "
          f"{(diff < 3).mean()*100:.1f}% of rows within 3 points")
