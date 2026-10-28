"""
Run once locally whenever data/UW_Risk_Score_Commercial_F.parquet or the models change, to
regenerate data/UW_Risk_Score_Commercial_Scored.parquet. Lets main.py skip running all 3
models over 20,000 rows at every startup - see main.py's _SCORED_PARQUET_PATH comment.

Run:
    .venv/Scripts/python.exe webapp/backend/prescore_book.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from scoring import load_models, score

if __name__ == "__main__":
    raw = pd.read_parquet(PROJECT_ROOT / "data" / "UW_Risk_Score_Commercial_F.parquet")
    raw["effective_date"] = pd.to_datetime(raw["effective_date"])
    book = score(raw, load_models())
    out_path = PROJECT_ROOT / "data" / "UW_Risk_Score_Commercial_Scored.parquet"
    book.to_parquet(out_path, index=False)
    print(f"wrote {out_path} ({len(book)} rows, {len(book.columns)} cols)")
