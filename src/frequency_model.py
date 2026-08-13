"""
Frequency model: predicts expected claim count per exposure-year.
Trains TWO separate models - one for property lines (building/CAT features), one for
liability lines (industry/workforce features) - see lob_config.py for why.
Poisson GLM baseline per group, NB overdispersion check, GBM challenger.
Saves both to ../models/frequency_model.joblib
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
from lob_config import PROPERTY_LOBS, LIABILITY_LOBS, PROPERTY_FEATURES, LIABILITY_FEATURES

# same defensive monotonicity fix as severity_model.py, applied here too in case GBM ever
# wins the frequency benchmark in a future retrain (GLM won this time, so this isn't
# currently changing behavior, just protecting against it silently regressing later)
MONOTONIC_DIRECTION = {
    "financial_stability_score": -1, "prior_claim_count": 1, "prior_total_incurred_usd": 1,
    "building_age_years": 1, "roof_age_years": 1, "distance_to_coast_miles": -1,
    "distance_to_fault_line_miles": -1, "distance_to_nearest_water_body_miles": -1,
    "fire_protection_class_iso_ppc": 1,
}

df, schema, usable_for, output_cols = load_data()
df["effective_date"] = pd.to_datetime(df["effective_date"])

GROUPS = [("property", PROPERTY_LOBS, PROPERTY_FEATURES), ("liability", LIABILITY_LOBS, LIABILITY_FEATURES)]
saved = {}

for group_name, lobs, features in GROUPS:
    print(f"\n=== {group_name.upper()} ({', '.join(lobs)}) ===")
    sub = df[df["line_of_business"].isin(lobs)].copy()
    print(f"{len(features)} predictors, {len(sub):,} rows")

    exposure_years = (sub["policy_term_months"] / 12).clip(lower=0.01)
    y_count = sub["claim_count_total"]
    y_freq = y_count / exposure_years

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
    etr, ete = exposure_years.loc[train_idx], exposure_years.loc[test_idx]

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
    saved[group_name] = {"model": best_model, "features": features, "num_cols": num_cols,
                          "cat_cols": cat_cols, "type": best_name}

joblib.dump(saved, "../models/frequency_model.joblib")
print("\nSaved both group models to ../models/frequency_model.joblib")
