"""
Shared scoring logic - loads the 3 trained models once, scores any dataframe of submissions
(the whole book, or a single new one). Both combine_scores.py and the Streamlit app import
this instead of duplicating the combination formula.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import joblib

# same fix as data_loader.py - resolved relative to this file, not the working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = str(PROJECT_ROOT / "models")


def load_models():
    return {
        "frequency": joblib.load(f"{MODELS_DIR}/frequency_model.joblib"),
        "severity": joblib.load(f"{MODELS_DIR}/severity_model.joblib"),
        "large_loss": joblib.load(f"{MODELS_DIR}/large_loss_model.joblib"),
    }


def score(df, models, premium_col="indicated_annual_premium_usd", term_col="policy_term_months"):
    """
    df must contain every column in models['frequency']['features'] (the 22 predictors) plus
    line_of_business, policy_term_months, and a premium column. Works on 1 row or 20,000.
    Returns df with score columns appended.
    """
    freq_saved, sev_saved, large_saved = models["frequency"], models["severity"], models["large_loss"]

    X = df[freq_saved["features"]].copy()
    for c in freq_saved["cat_cols"]:
        X[c] = X[c].astype(object).where(X[c].notna(), None)

    exposure_years = (df[term_col] / 12).clip(lower=0.01)
    predicted_freq_rate = freq_saved["model"].predict(X)
    predicted_frequency = predicted_freq_rate * exposure_years

    predicted_attritional_severity = sev_saved["model"].predict(X)

    p_large = large_saved["occurrence_model"].predict_proba(X)[:, 1]
    threshold_by_lob = df["line_of_business"].map(large_saved["thresholds_by_lob"])
    gpd_mean_excess = stats.genpareto.mean(large_saved["gpd_shape"], scale=large_saved["gpd_scale"])
    expected_large_severity = threshold_by_lob + gpd_mean_excess

    attritional_freq = np.clip(predicted_frequency - p_large, 0, None)
    total_expected_loss_cost = (attritional_freq * predicted_attritional_severity
                                 + p_large * expected_large_severity)

    predicted_loss_ratio_pct = (total_expected_loss_cost / df[premium_col]) * 100
    loss_propensity_score = predicted_loss_ratio_pct.clip(10, 100).round(1)

    out = df.copy()
    out["predicted_frequency"] = predicted_frequency.round(4)
    out["predicted_attritional_severity"] = predicted_attritional_severity.round(0)
    out["p_large_loss"] = p_large.round(4)
    out["expected_large_severity"] = expected_large_severity.round(0)
    out["total_expected_loss_cost"] = total_expected_loss_cost.round(0)
    out["predicted_loss_ratio_pct"] = predicted_loss_ratio_pct.round(1)
    out["loss_propensity_score_v2"] = loss_propensity_score
    return out
