"""
Shared loader - reads schema from Data Dictionary tab, maps Sample Data columns by letter,
returns a clean df + feature metadata. Every script in this project imports this instead of
re-reading the Excel file its own way, so schema logic lives in exactly one place.
"""
import re
from pathlib import Path
import pandas as pd
from openpyxl.utils import column_index_from_string

# resolved relative to THIS FILE's location, not the current working directory - so this
# works whether you run "cd src && python data_audit.py" locally or Streamlit Cloud launches
# the app from the repo root instead
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILE = str(PROJECT_ROOT / "data" / "Copy_of_UW_Risk_Score_Commercial_data.xlsx")


def load_data(path=DEFAULT_FILE):
    raw_dict = pd.read_excel(path, sheet_name="Data Dictionary", header=None)
    rows, section = [], None
    for _, r in raw_dict.iterrows():
        a, b, c, d = r.get(0), r.get(1), r.get(2), r.get(3)
        if isinstance(a, str) and re.match(r"^\d", a) and pd.isna(b):
            section = a
            continue
        if a == "Column Name":
            continue
        if isinstance(a, str) and isinstance(c, str) and re.fullmatch(r"[A-Z]{1,3}", str(c)):
            rows.append({"name": a, "usable_for": d, "col_letter": c, "section": section})
    schema = pd.DataFrame(rows)
    schema["col_index"] = schema["col_letter"].apply(lambda l: column_index_from_string(l) - 1)

    df_raw = pd.read_excel(path, sheet_name="Sample Data")
    rename_map = {}
    for _, r in schema.iterrows():
        if r["col_index"] < len(df_raw.columns):
            rename_map[df_raw.columns[r["col_index"]]] = r["name"]
    df = df_raw.rename(columns=rename_map)

    usable_for = dict(zip(schema["name"], schema["usable_for"]))
    output_cols = [n for n, u in usable_for.items() if isinstance(u, str) and "output" in u.lower()]

    return df, schema, usable_for, output_cols


def features_for(target, usable_for, output_cols, exclude=()):
    """Candidate feature columns tagged for a given model target, outputs and IDs excluded."""
    return [n for n, u in usable_for.items()
            if isinstance(u, str) and target in u and n not in output_cols and n not in exclude]
