"""
Severity model: predicts expected claim $ amount, GIVEN a claim happens, attritional only.
Trains two models (property/liability). Regularization, attritional-only split, and
monotonic constraints all built in from the start.
Saves to ../models/severity_model.joblib

Training only runs under `python severity_model.py`, not on import - see frequency_model.py's
module docstring for why this guard matters (2026-08 reliability audit).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import joblib
from lob_config import PROPERTY_LOBS, LIABILITY_LOBS, PROPERTY_FEATURES, LIABILITY_FEATURES, encode_ordinal_tiers

MONOTONIC_DIRECTION = {
    "financial_stability_score": -1, "claim_count_last_3_years": 1,
    "large_loss_count_last_5_years": 1, "claims_free_years": -1,
    "building_age_years": 1, "roof_age_years": 1, "fire_protection_class_iso_ppc": 1,
    "distance_to_coast_miles": -1, "distance_to_fault_line_miles": -1,
    "distance_to_nearest_water_body_miles": -1, "distance_to_fire_station_miles": 1,
    "distance_to_fire_hydrant_miles": 1, "debt_to_equity_ratio": 1,
    "flood_zone_tier": 1, "earthquake_zone_tier": 1, "wildfire_zone_tier": 1, "wind_zone_tier": 1,
    # added 2026-08 - see frequency_model.py's MONOTONIC_DIRECTION comment for the full
    # reasoning (same audit, same directions). number_of_employees/total_insured_value_usd
    # stay as ordinary constrained features here (unlike frequency_model.py, where they become
    # an explicit exposure multiplier instead) - bigger operations/assets plausibly cost more
    # per claim when one happens, even though claim COUNT needed the exposure-unit fix instead.
    "revenue_growth_trend_pct": -1, "business_tenure_years": -1, "avg_weekly_work_hours": 1,
    "avg_employee_age": 1, "avg_wage_per_employee_usd": -1,
    "number_of_employees": 1, "total_insured_value_usd": 1,
    "num_stories": 1,
}

GROUPS = [("property", PROPERTY_LOBS, PROPERTY_FEATURES), ("liability", LIABILITY_LOBS, LIABILITY_FEATURES)]


if __name__ == "__main__":
    from data_loader import load_data

    df, schema, usable_for, output_cols = load_data()
    df["effective_date"] = pd.to_datetime(df["effective_date"])
    df = encode_ordinal_tiers(df)
    claims_all = df[df["claim_count_total"] > 0].copy()

    saved = {}
    lob_avg_severity = {}

    for group_name, lobs, features in GROUPS:
        print(f"\n=== {group_name.upper()} ({', '.join(lobs)}) ===")
        claims = claims_all[claims_all["line_of_business"].isin(lobs)].copy()

        # attritional only - large claims are handled separately, using this file's own
        # peril-specific threshold rather than a generic percentile rule
        n_before = len(claims)
        claims = claims[claims["large_claim_flag"] == 0]
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
        print(f"GLM (Gamma, alpha=5.0) - MAE: ${glm_mae:,.0f}")

        pre_fitted = pre.fit(Xtr)
        n_cat_output_cols = len(pre_fitted.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols))
        monotonic_cst = [MONOTONIC_DIRECTION.get(c, 0) for c in num_cols] + [0] * n_cat_output_cols

        gbm = Pipeline([("prep", pre), ("model", HistGradientBoostingRegressor(
            loss="gamma", max_iter=300, max_depth=6, random_state=42, monotonic_cst=monotonic_cst))])
        gbm.fit(Xtr, ytr, model__sample_weight=wtr)
        gbm_pred = np.clip(gbm.predict(Xte), 1, None)
        gbm_mae = mean_absolute_error(yte, gbm_pred)
        print(f"GBM (monotonic) - MAE: ${gbm_mae:,.0f}")

        best_model, best_name = (gbm, "gbm") if gbm_mae < glm_mae else (glm, "glm")
        print(f"Using {best_name}")
        saved[group_name] = {"model": best_model, "features": features, "num_cols": num_cols,
                              "cat_cols": cat_cols, "type": best_name}

    saved["lob_avg_severity"] = lob_avg_severity
    joblib.dump(saved, "../models/severity_model.joblib")
    print("\nSaved both group models to ../models/severity_model.joblib")
