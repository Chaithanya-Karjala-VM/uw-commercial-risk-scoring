"""
Bind Propensity - rules-based, not trained. No historical bind/no-bind outcome exists in
the source data; every bind-related score column is itself a derived output, not a fact.
Reconstructs the documented formula (broker hit/bind, retention, diversity, profitability,
deal friction) via regression against the file's own stored score.
Verified: R^2=0.927, avg error 1.54 points on a 0-100 scale.
"""
import numpy as np
import pandas as pd

BIND_INGREDIENTS = [
    "broker_hit_bind_ratio_pct", "broker_renewal_retention_pct",
    "broker_product_diversity_count", "broker_geo_diversity_count",
    "broker_profitability_score_0_100", "quote_to_bind_days",
    "competing_quotes_count", "submission_completeness_pct",
]

# fitted via regression against bind_propensity_score_0_100 - rerun verify_bind_formula()
# if the source data changes
BIND_COEFFICIENTS = {
    "intercept": -251.768038,
    "broker_hit_bind_ratio_pct": 0.148436,
    "broker_renewal_retention_pct": 2.966184,
    "broker_product_diversity_count": 0.919368,
    "broker_geo_diversity_count": 0.288419,
    "broker_profitability_score_0_100": 0.177814,
    "quote_to_bind_days": -0.100091,
    "competing_quotes_count": -2.001480,
    "submission_completeness_pct": 0.001832,
}


def compute_bind_propensity(df):
    score = pd.Series(BIND_COEFFICIENTS["intercept"], index=df.index)
    for field in BIND_INGREDIENTS:
        score = score + df[field].fillna(0) * BIND_COEFFICIENTS[field]
    return score.clip(0, 100).round(1)


def verify_bind_formula(df):
    """Re-fits the regression fresh and reports fit quality - run this any time the source
    data changes to confirm the coefficients above are still accurate."""
    from sklearn.linear_model import LinearRegression
    X = df[BIND_INGREDIENTS].fillna(0)
    y = df["bind_propensity_score_0_100"]
    reg = LinearRegression().fit(X, y)
    pred = reg.predict(X)
    r2 = reg.score(X, y)
    mae = np.abs(pred - y).mean()
    return {"r2": r2, "mae": mae, "intercept": reg.intercept_,
            "coefficients": dict(zip(BIND_INGREDIENTS, reg.coef_))}


if __name__ == "__main__":
    from data_loader import load_data
    df, schema, usable_for, output_cols = load_data()

    fit = verify_bind_formula(df)
    print(f"Fresh regression check: R^2={fit['r2']:.3f}, avg error={fit['mae']:.2f} points")

    computed = compute_bind_propensity(df)
    stored = df["bind_propensity_score_0_100"]
    diff = (computed - stored).abs()
    print(f"Fixed coefficients vs stored: avg diff = {diff.mean():.2f} points, "
          f"{(diff < 3).mean()*100:.1f}% of rows within 3 points")
