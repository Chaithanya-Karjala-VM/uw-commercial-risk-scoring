"""
Combines the three trained models into the Loss Propensity Score (0-100, 0=best, 100=worst)
and reports portfolio-level results. Run: python combine_scores.py
"""
import os
import numpy as np
import pandas as pd
from data_loader import load_data
from scoring import load_models, score

df, schema, usable_for, output_cols = load_data()
df["effective_date"] = pd.to_datetime(df["effective_date"])

models = load_models()
df = score(df, models)

simple_avg = df["predicted_loss_ratio_pct"].mean()
portfolio_lr = df["total_expected_loss_cost"].sum() / df["indicated_annual_premium_usd"].sum() * 100
print(f"Simple average loss ratio: {simple_avg:.1f}% (not weighted - use portfolio figure below)")
print(f"Portfolio loss ratio (premium-weighted): {portfolio_lr:.1f}%")

by_lob = df.groupby("line_of_business").apply(
    lambda g: g["total_expected_loss_cost"].sum() / g["indicated_annual_premium_usd"].sum() * 100,
    include_groups=False
)
print("\nPortfolio loss ratio by Line of Business:")
print(by_lob.round(1).to_string())

stored = df["loss_propensity_score_0_100"]
v2 = df["loss_propensity_score_v2"]
print(f"\nvs. stored score: correlation {v2.corr(stored):.3f}, avg diff {(v2-stored).abs().mean():.1f} points")

print("\nLoss Propensity Score v2 - distribution:")
print(v2.describe().round(1).to_string())

os.makedirs("../reports", exist_ok=True)
df.to_excel("../reports/scored_book.xlsx", index=False)
print("\nFull scored book saved to ../reports/scored_book.xlsx")
