"""
Severity model: predicts expected claim $ amount, GIVEN a claim happens, attritional only
(large claims are handled separately in large_loss_model.py).
Trains TWO separate models - property lines and liability lines - see lob_config.py.
Gamma GLM baseline, GBM challenger. Saves both to ../models/severity_model.joblib
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_gamma_deviance, mean_absolute_error
import joblib
from data_loader import load_data
from lob_config import PROPERTY_LOBS, LIABILITY_LOBS, PROPERTY_FEATURES, LIABILITY_FEATURES

# Known real-world direction for continuous/ordinal features - GBM has no built-in guarantee
# that "better financial stability" always means "lower predicted severity" the way a GLM's
# single coefficient does; checked this directly and found financial_stability_score bouncing
# non-monotonically (28.2 -> 27.0 -> 30.8 -> 32.8 -> 30.0 -> 28.9 as it improved from 10 to 99).
# +1 = higher value should push severity up, -1 = higher value should push severity down,
# 0 = no enforced direction (genuinely ambiguous, e.g. employee count, TIV).
MONOTONIC_DIRECTION = {
    "financial_stability_score": -1, "prior_claim_count": 1, "prior_total_incurred_usd": 1,
    "building_age_years": 1, "roof_age_years": 1, "distance_to_coast_miles": -1,
    "distance_to_fault_line_miles": -1, "distance_to_nearest_water_body_miles": -1,
    "fire_protection_class_iso_ppc": 1,
}

df, schema, usable_for, output_cols = load_data()
df["effective_date"] = pd.to_datetime(df["effective_date"])
claims_all = df[df["claim_count_total"] > 0].copy()

GROUPS = [("property", PROPERTY_LOBS, PROPERTY_FEATURES), ("liability", LIABILITY_LOBS, LIABILITY_FEATURES)]
saved = {}
lob_avg_severity = {}

for group_name, lobs, features in GROUPS:
    print(f"\n=== {group_name.upper()} ({', '.join(lobs)}) ===")
    claims = claims_all[claims_all["line_of_business"].isin(lobs)].copy()

    # attritional only - exclude anything above the per-LOB 95th pct threshold (large_loss_model.py)
    thresholds = claims.groupby("line_of_business")["claim_severity_avg_usd"].quantile(0.95)
    claims["_thr"] = claims["line_of_business"].map(thresholds)
    n_before = len(claims)
    claims = claims[claims["claim_severity_avg_usd"] <= claims["_thr"]].drop(columns="_thr")
    print(f"Excluded {n_before - len(claims)} large claims, fitting on {len(claims):,} attritional claims")

    for lob in lobs:
        lob_avg_severity[lob] = claims[claims["line_of_business"] == lob]["claim_severity_avg_usd"].mean()

    y = claims["claim_severity_avg_usd"]
    weight = claims["claim_count_total"]

    claims_sorted = claims.sort_values("effective_date")
    split = claims_sorted.iloc[int(len(claims_sorted) * 0.75)]["effective_date"]
    train_idx = claims_sorted[claims_sorted["effective_date"] < split].index
    test_idx = claims_sorted[claims_sorted["effective_date"] >= split].index

    X = claims[features].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in features if c not in num_cols]
    for c in cat_cols:
        X[c] = X[c].astype(object).where(X[c].notna(), None)

    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ])

    Xtr, Xte = X.loc[train_idx], X.loc[test_idx]
    ytr, yte = y.loc[train_idx], y.loc[test_idx]
    wtr, wte = weight.loc[train_idx], weight.loc[test_idx]

    glm = Pipeline([("prep", pre), ("model", GammaRegressor(alpha=5.0, max_iter=2000))])
    glm.fit(Xtr, ytr, model__sample_weight=wtr)
    glm_pred = glm.predict(Xte)
    glm_mae = mean_absolute_error(yte, glm_pred)
    print(f"GLM (Gamma) - MAE: ${glm_mae:,.0f}")

    # fit the preprocessor once first to get the REAL one-hot output width (raw nunique()
    # would be wrong here - it doesn't account for the "Missing" category the imputer adds,
    # or how OneHotEncoder actually expands each column)
    pre.fit(Xtr)
    n_cat_output_cols = len(pre.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols))
    monotonic_cst = [MONOTONIC_DIRECTION.get(c, 0) for c in num_cols] + [0] * n_cat_output_cols

    gbm = Pipeline([("prep", pre), ("model", HistGradientBoostingRegressor(
        loss="gamma", max_iter=300, max_depth=6, random_state=42, monotonic_cst=monotonic_cst))])
    gbm.fit(Xtr, ytr, model__sample_weight=wtr)
    gbm_pred = np.clip(gbm.predict(Xte), 1, None)
    gbm_mae = mean_absolute_error(yte, gbm_pred)
    print(f"GBM - MAE: ${gbm_mae:,.0f}")

    best_model, best_name = (gbm, "gbm") if gbm_mae < glm_mae else (glm, "glm")
    print(f"Using {best_name}")
    saved[group_name] = {"model": best_model, "features": features, "num_cols": num_cols,
                          "cat_cols": cat_cols, "type": best_name}

saved["lob_avg_severity"] = lob_avg_severity
joblib.dump(saved, "../models/severity_model.joblib")
print("\nSaved both group models to ../models/severity_model.joblib")
print("Per-LOB average attritional severity:", {k: round(v) for k, v in lob_avg_severity.items()})
