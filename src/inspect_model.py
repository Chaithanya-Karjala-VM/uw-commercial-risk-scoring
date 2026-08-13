"""
Opens any .joblib file this project saves and prints what's inside it, regardless of which
script produced it - frequency and severity models, and the two-part large-loss file, are
saved with different shapes, so this checks which one it's looking at first.
Run: python inspect_model.py ../models/frequency_model.joblib
     python inspect_model.py ../models/severity_model.joblib
     python inspect_model.py ../models/large_loss_model.joblib
"""
import sys
import pandas as pd
import joblib

path = sys.argv[1] if len(sys.argv) > 1 else "../models/severity.joblib"
saved = joblib.load(path)


def show_coefficients(pipeline, num_cols, cat_cols):
    model_step = pipeline.named_steps["model"]
    print(f"Underlying sklearn object: {model_step}")
    if hasattr(model_step, "coef_"):
        prep = pipeline.named_steps["prep"]
        ohe = prep.named_transformers_["cat"].named_steps["onehot"]
        feature_names = num_cols + list(ohe.get_feature_names_out(cat_cols))
        coef_values = model_step.coef_.ravel()  # LogisticRegression gives shape (1, n), flatten it
        coefs = pd.Series(coef_values, index=feature_names).sort_values(key=abs, ascending=False)
        print("\nCoefficients, strongest first:")
        print(coefs.head(15).round(4).to_string())
    else:
        print("This is a tree-based model (no coefficients to show) - feature importance for "
              "this one gets printed when the training script itself runs, since it needs the "
              "test data loaded to compute permutation importance.")


if "occurrence_model" in saved:
    # large_loss_model.joblib - a different shape: two parts, not one model
    print("This is the large-loss file - two parts saved together.")
    print(f"\nPart A (occurrence) model type: {saved['occurrence_type']}")
    print(f"Features used: {len(saved['features'])}")
    show_coefficients(saved["occurrence_model"], saved["num_cols"], saved["cat_cols"])

    print(f"\nPer-Line-of-Business large-claim thresholds (95th percentile severity):")
    for lob, v in saved["thresholds_by_lob"].items():
        print(f"  {lob}: ${v:,.0f}")

    print(f"\nPart B (severity given large) - Generalized Pareto fit:")
    print(f"  shape = {saved['gpd_shape']:.3f}, scale = ${saved['gpd_scale']:,.0f}")
    print("  (higher shape = heavier tail; scale sets the typical size of the excess above threshold)")

else:
    # frequency_model.joblib or severity_model.joblib - one model, same shape
    print(f"Model type: {saved['type']}")
    print(f"Number of features: {len(saved['features'])}")
    print(f"Numeric features ({len(saved['num_cols'])}): {saved['num_cols']}")
    print(f"Categorical features ({len(saved['cat_cols'])}): {saved['cat_cols']}")
    print()
    show_coefficients(saved["model"], saved["num_cols"], saved["cat_cols"])
