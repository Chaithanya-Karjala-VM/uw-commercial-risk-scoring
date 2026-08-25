"""
Frequency model: predicts expected claim count per exposure-year AND per exposure-unit
(number_of_employees for liability, $1M of TIV for property - see EXPOSURE_UNIT below, added
2026-08 after an 8-LOB audit found predicted frequency barely moving across the real range of
either field, while premium scaled linearly with them, collapsing loss ratio toward zero at
scale for purely mechanical reasons).
Trains two models (property/liability groups) - monotonic constraints and exposure
weighting built in from the start this time, not discovered as bugs.
Saves to ../models/frequency_model.joblib
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_poisson_deviance
import joblib
from data_loader import load_data
from lob_config import PROPERTY_LOBS, LIABILITY_LOBS, PROPERTY_FEATURES, LIABILITY_FEATURES, encode_ordinal_tiers

MONOTONIC_DIRECTION = {
    "financial_stability_score": -1, "claim_count_last_3_years": 1,
    "large_loss_count_last_5_years": 1, "claims_free_years": -1,
    "building_age_years": 1, "roof_age_years": 1, "fire_protection_class_iso_ppc": 1,
    "distance_to_coast_miles": -1, "distance_to_fault_line_miles": -1,
    "distance_to_nearest_water_body_miles": -1, "distance_to_fire_station_miles": 1,
    "distance_to_fire_hydrant_miles": 1, "debt_to_equity_ratio": 1,
    "flood_zone_tier": 1, "earthquake_zone_tier": 1, "wildfire_zone_tier": 1, "wind_zone_tier": 1,
    # added 2026-08 - a full 8-LOB audit found these fields swinging the score by up to 76
    # points via non-monotonic, erratic patterns because they had never been given a
    # constraint, unlike every field above. Directions below are the defensible real-world ones:
    "revenue_growth_trend_pct": -1,  # declining revenue -> financial distress -> higher risk
    "business_tenure_years": -1,     # newer business -> less track record -> higher risk
    "avg_weekly_work_hours": 1,      # more hours worked -> more exposure -> higher risk
    # avg_employee_age and avg_wage_per_employee_usd don't have one universally-agreed
    # real-world direction - documented as judgment calls, less certain than the fields above:
    "avg_employee_age": 1,           # real WC pricing commonly loads for older-worker claims
                                      # taking longer/costing more, even though frequency alone
                                      # may not rise with age - judgment call
    "avg_wage_per_employee_usd": -1, # higher average wage commonly proxies for more skilled,
                                      # less physically hazardous work - judgment call
    # number_of_employees / total_insured_value_usd are deliberately NOT constrained here - see
    # this file's exposure-unit handling below for why (the real problem wasn't direction, it
    # was magnitude). They ARE constrained in severity_model.py / large_loss_model.py, where
    # they remain ordinary features rather than an explicit exposure multiplier.
    "number_of_employees": 1, "total_insured_value_usd": 1,
    "num_stories": 1,  # more stories -> more complex building (fire spread, evacuation) -> higher risk
}

df, schema, usable_for, output_cols = load_data()
df["effective_date"] = pd.to_datetime(df["effective_date"])
df = encode_ordinal_tiers(df)  # Low/Moderate/High/Severe -> 0-3, monotonic-constrainable

# Exposure-unit fix, added 2026-08. Root cause found via an 8-LOB audit: predicted_frequency
# barely moved across the ENTIRE real range of number_of_employees (0.28 at 1 employee, 0.18
# at 5,000) or total_insured_value_usd, because both were just ordinary input features the
# model had to learn a relationship for from limited data - not treated as an explicit
# exposure multiplier the way policy_term_months already is via exposure_years. Meanwhile
# premium scales EXACTLY linearly with both, so loss ratio collapsed toward zero at scale for
# purely mechanical reasons regardless of real risk. Standard actuarial practice (and exactly
# how exposure_years already works in this file) is to fit a PER-UNIT rate and multiply by the
# exposure units explicitly, rather than hope a GBM learns proportional scaling on its own from
# ~11,800 liability rows. property's unit is $1M of TIV (a common real rating unit); liability's
# is 1 employee. The field is removed from that group's OWN feature list below since it becomes
# an explicit multiplier rather than something the model fits a coefficient for; it's untouched
# in severity_model.py / large_loss_model.py, where it stays an ordinary constrained feature.
EXPOSURE_UNIT = {
    # floor must sit BELOW the real data's actual minimum, or every value under it gets
    # artificially clamped to the same floor - found immediately when 0.05 ($50K) sat ABOVE
    # the real minimum TIV ($36K), distorting the low end of the property sweep. Real min TIV
    # is $36K, real min employees is 1 - floors set safely under both.
    "property": {"col": "total_insured_value_usd", "scale": 1_000_000, "floor": 0.03},
    "liability": {"col": "number_of_employees", "scale": 1, "floor": 1},
}

GROUPS = [("property", PROPERTY_LOBS, PROPERTY_FEATURES), ("liability", LIABILITY_LOBS, LIABILITY_FEATURES)]
saved = {}

for group_name, lobs, features in GROUPS:
    print(f"\n=== {group_name.upper()} ({', '.join(lobs)}) ===")
    sub = df[df["line_of_business"].isin(lobs)].copy()

    unit_cfg = EXPOSURE_UNIT[group_name]
    features = [f for f in features if f != unit_cfg["col"]]  # becomes an explicit multiplier, not a feature
    print(f"{len(features)} predictors, {len(sub):,} rows (exposure unit: {unit_cfg['col']})")

    exposure_years = (sub["policy_term_months"] / 12).clip(lower=0.01)
    exposure_units = (sub[unit_cfg["col"]] / unit_cfg["scale"]).clip(lower=unit_cfg["floor"])
    total_exposure = exposure_years * exposure_units
    y_count = sub["claim_count_total"]
    y_freq = y_count / total_exposure

    sub_sorted = sub.sort_values("effective_date")
    split = sub_sorted.iloc[int(len(sub_sorted) * 0.75)]["effective_date"]
    train_idx = sub_sorted[sub_sorted["effective_date"] < split].index
    test_idx = sub_sorted[sub_sorted["effective_date"] >= split].index

    X = sub[features].copy()
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
    ytr_freq, yte_freq = y_freq.loc[train_idx], y_freq.loc[test_idx]
    ytr_count = y_count.loc[train_idx]
    etr, ete = total_exposure.loc[train_idx], total_exposure.loc[test_idx]

    glm = Pipeline([("prep", pre), ("model", PoissonRegressor(alpha=0.01, max_iter=500))])
    glm.fit(Xtr, ytr_freq, model__sample_weight=etr)

    # Same safeguard used in large_loss_model.py: a GLM coefficient that contradicts a known
    # real-world direction is a red flag, not real signal - found exactly this with
    # debt_to_equity_ratio (fit negative, i.e. "more debt -> fewer claims", backwards) which
    # was causing a small but real non-monotonic wobble in the combined score. Drop and refit
    # rather than keep a spurious driver.
    ohe_check = glm.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
    feat_names_check = num_cols + list(ohe_check.get_feature_names_out(cat_cols))
    coefs_check = pd.Series(glm.named_steps["model"].coef_.ravel(), index=feat_names_check)
    contradicting = [f for f, d in MONOTONIC_DIRECTION.items()
                      if f in coefs_check.index and np.sign(coefs_check[f]) not in (0, d)]
    if contradicting:
        print(f"Dropping (contradicts known direction): {contradicting}")
        num_cols = [c for c in num_cols if c not in contradicting]
        X = X.drop(columns=contradicting)
        pre = ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                               ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
        ])
        Xtr, Xte = X.loc[train_idx], X.loc[test_idx]
        glm = Pipeline([("prep", pre), ("model", PoissonRegressor(alpha=0.01, max_iter=500))])
        glm.fit(Xtr, ytr_freq, model__sample_weight=etr)

    glm_pred = glm.predict(Xte)
    glm_dev = mean_poisson_deviance(yte_freq.clip(lower=1e-6), glm_pred.clip(min=1e-6), sample_weight=ete)
    print(f"GLM (Poisson) - mean deviance: {glm_dev:.4f}")

    Xtr_dense = pre.fit_transform(Xtr)
    Xtr_sm = sm.add_constant(Xtr_dense)
    poisson_sm = sm.GLM(ytr_count, Xtr_sm, family=sm.families.Poisson(), offset=np.log(etr)).fit()
    dispersion = poisson_sm.deviance / poisson_sm.df_resid
    print(f"Dispersion: {dispersion:.3f}")

    pre_fitted = pre.fit(Xtr)
    n_cat_output_cols = len(pre_fitted.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols))
    monotonic_cst = [MONOTONIC_DIRECTION.get(c, 0) for c in num_cols] + [0] * n_cat_output_cols

    gbm = Pipeline([("prep", pre), ("model", HistGradientBoostingRegressor(
        loss="poisson", max_iter=300, max_depth=6, random_state=42, monotonic_cst=monotonic_cst))])
    gbm.fit(Xtr, ytr_freq, model__sample_weight=etr)
    gbm_pred = gbm.predict(Xte)
    gbm_dev = mean_poisson_deviance(yte_freq.clip(lower=1e-6), np.clip(gbm_pred, 1e-6, None), sample_weight=ete)
    print(f"GBM - mean deviance: {gbm_dev:.4f}")

    best_model, best_name = (gbm, "gbm") if gbm_dev < glm_dev else (glm, "glm")
    print(f"Using {best_name}")
    # exposure_unit saved alongside the model so scoring.py can reconstruct the same multiplier
    # (model.predict(X) is now a PER-UNIT rate, e.g. claims per employee per year, not a
    # per-policy rate - it must be multiplied by both exposure_years AND this unit to recover
    # an actual predicted claim count for the submission)
    saved[group_name] = {"model": best_model, "features": features, "num_cols": num_cols,
                          "cat_cols": cat_cols, "type": best_name, "exposure_unit": unit_cfg}

joblib.dump(saved, "../models/frequency_model.joblib")
print("\nSaved both group models to ../models/frequency_model.joblib")
