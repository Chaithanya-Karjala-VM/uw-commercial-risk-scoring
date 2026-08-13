"""
Combines the three trained models into the actual Loss Propensity Score, replacing the
workbook's hand-tuned formula end to end. The combination formula itself lives in scoring.py
(shared with the Streamlit app) - this script just runs it against the whole book and reports.
Run: python combine_scores.py
"""
import numpy as np
import pandas as pd
from data_loader import load_data
from scoring import load_models, score

df, schema, usable_for, output_cols = load_data()
df["effective_date"] = pd.to_datetime(df["effective_date"])

models = load_models()
df = score(df, models)

print(f"Simple average of individual predicted loss ratios: {df['predicted_loss_ratio_pct'].mean():.1f}% - "
      f"this number is misleading and shouldn't be trusted on its own. A handful of small-"
      f"premium policies with noisy predictions can swing a plain average heavily even though "
      f"they represent almost nothing in real dollars.")
portfolio_lr = df["total_expected_loss_cost"].sum() / df["indicated_annual_premium_usd"].sum() * 100
print(f"Portfolio loss ratio (total expected cost / total premium): {portfolio_lr:.1f}% - "
      f"this is the trustworthy number.")

by_lob = df.groupby("line_of_business").apply(
    lambda g: g["total_expected_loss_cost"].sum() / g["indicated_annual_premium_usd"].sum() * 100,
    include_groups=False
)
print("\nPortfolio loss ratio by Line of Business:")
print(by_lob.round(1).to_string())

stored = df["loss_propensity_score_0_100"]
v2 = df["loss_propensity_score_v2"]
corr = v2.corr(stored)
mae = (v2 - stored).abs().mean()
print(f"\nAgainst the file's own stored Loss Propensity Score: correlation {corr:.3f}, "
      f"average difference {mae:.1f} points")

print("\nCalibration by v2-score decile - avg predicted vs avg stored:")
calib = pd.DataFrame({"v2": v2, "stored": stored})
calib["decile"] = pd.qcut(calib["v2"], 10, labels=False, duplicates="drop")
print(calib.groupby("decile").agg(avg_v2=("v2", "mean"), avg_stored=("stored", "mean"),
                                    n=("v2", "size")).round(1).to_string())

import os
os.makedirs("../reports", exist_ok=True)
df.to_excel("../reports/scored_book.xlsx", index=False)
print("\nFull scored book saved to ../reports/scored_book.xlsx")
