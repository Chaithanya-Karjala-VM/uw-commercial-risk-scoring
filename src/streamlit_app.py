"""
Streamlit prototype - Explore the existing book, or score a brand-new submission live.
Run: streamlit run streamlit_app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from data_loader import load_data
from scoring import load_models, score
from lob_config import PROPERTY_LOBS, LIABILITY_LOBS, group_for_lob

st.set_page_config(page_title="Commercial UW Risk Intelligence", layout="wide", page_icon="📊")

# resolved relative to this file, not the working directory - same fix as data_loader.py/
# scoring.py, so the logo shows up whether run locally from src/ or launched by Streamlit Cloud
LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "vm_logo.png"
if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), width=180)
    st.sidebar.caption("Built at ValueMomentum")
st.sidebar.markdown("---")

# cache_resource: loaded once per server session, not on every widget interaction
@st.cache_resource
def get_models():
    return load_models()

# cache_data: the scored book is recomputed only if the underlying data file changes.
# Only called inside Portfolio Overview mode below - New Submission mode never needs the
# whole book scored, so it shouldn't pay that cost on every form submission.
@st.cache_data
def get_scored_book():
    df, schema, usable_for, output_cols = load_data()
    df["effective_date"] = pd.to_datetime(df["effective_date"])
    models = get_models()
    return score(df, models)

# lightweight - just the raw columns needed for dropdown options, not the full scoring pass
@st.cache_data
def get_raw_book():
    df, schema, usable_for, output_cols = load_data()
    return df[["line_of_business", "broker_tier"]]

models = get_models()
freq_saved = models["frequency"]
raw_book = get_raw_book()

CATEGORICAL_OPTIONS = {
    "construction_type": ["Fire Resistive", "Frame", "Joisted Masonry", "Masonry Non-Combustible", "Non-Combustible"],
    "sprinklered_flag": ["Y", "N"],
    "fema_flood_zone_code": ["A", "AE", "V", "VE", "X", "X (shaded)"],
    "flood_zone_tier": ["Low", "Moderate", "High", "Severe"],
    "earthquake_zone_tier": ["Low", "Moderate", "High", "Severe"],
    "soil_liquefaction_class": ["A - Hard Rock", "B - Rock", "C - Dense Soil", "D - Stiff Soil", "E - Soft Soil", "F - Special Study Required"],
    "wildfire_zone_tier": ["Low", "Moderate", "High", "Severe"],
    "wind_zone_tier": ["Low", "Moderate", "High", "Severe"],
    "safety_inspection_findings": ["No Issues Noted", "Minor Issues Noted", "Moderate Issues – Corrective Action Required", "Major Deficiencies Identified"],
}
LOB_OPTIONS = raw_book["line_of_business"].unique().tolist()
INDUSTRY_OPTIONS = ["Accommodation & Food Services", "Administrative Services", "Construction",
                     "Health Care", "Manufacturing", "Professional Services", "Real Estate",
                     "Retail Trade", "Transportation & Warehousing", "Wholesale Trade"]

st.title("Commercial UW Risk Intelligence")
st.caption("Model-driven Loss Propensity scoring for commercial submissions")
mode = st.sidebar.radio("View", ["Portfolio Overview", "New Submission"])

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Model scope:** this predicts Loss Propensity only (claim frequency, severity, and "
    "large-loss models). Bind Propensity and Appetite/Capacity are not yet part of this "
    "model, so broker selection does not currently affect the predicted score."
)

# ============================================================================
if mode == "Portfolio Overview":
    st.header("Portfolio Overview")
    book = get_scored_book()

    c1, c2, c3 = st.columns(3)
    lob_filter = c1.selectbox("Line of Business", ["All"] + sorted(LOB_OPTIONS))
    tier_filter = c2.selectbox("Broker Tier", ["All"] + sorted(book["broker_tier"].dropna().unique().tolist()))
    score_range = c3.slider("Loss Propensity Score range", 10, 100, (10, 100))

    filtered = book.copy()
    if lob_filter != "All":
        filtered = filtered[filtered["line_of_business"] == lob_filter]
    if tier_filter != "All":
        filtered = filtered[filtered["broker_tier"] == tier_filter]
    filtered = filtered[filtered["loss_propensity_score_v2"].between(*score_range)]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Submissions", f"{len(filtered):,}")
    m2.metric("Avg Loss Propensity Score", f"{filtered['loss_propensity_score_v2'].mean():.1f}")
    portfolio_lr = (filtered["total_expected_loss_cost"].sum() / filtered["indicated_annual_premium_usd"].sum() * 100
                     if filtered["indicated_annual_premium_usd"].sum() > 0 else 0)
    m3.metric("Portfolio Loss Ratio", f"{portfolio_lr:.1f}%")
    m4.metric("Avg Premium", f"${filtered['indicated_annual_premium_usd'].mean():,.0f}")

    st.subheader("Loss Propensity Score distribution")
    fig_hist = px.histogram(
        filtered, x="loss_propensity_score_v2", nbins=20,
        labels={"loss_propensity_score_v2": "Loss Propensity Score"},
        color_discrete_sequence=["#1F3864"],
    )
    fig_hist.update_layout(
        yaxis_title="Number of Submissions", xaxis_title="Loss Propensity Score (10=best, 100=worst)",
        bargap=0.05, height=350, margin=dict(l=10, r=10, t=10, b=10),
    )
    fig_hist.update_traces(hovertemplate="Score ~%{x}<br>%{y:,} submissions<extra></extra>")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Portfolio Loss Ratio by Line of Business")
    by_lob = filtered.groupby("line_of_business").apply(
        lambda g: g["total_expected_loss_cost"].sum() / g["indicated_annual_premium_usd"].sum() * 100
        if g["indicated_annual_premium_usd"].sum() > 0 else 0, include_groups=False
    )
    by_lob_df = by_lob.reset_index()
    by_lob_df.columns = ["Line of Business", "Loss Ratio (%)"]
    by_lob_df = by_lob_df.sort_values("Loss Ratio (%)", ascending=True)
    fig_lob = px.bar(
        by_lob_df, x="Loss Ratio (%)", y="Line of Business", orientation="h",
        color="Loss Ratio (%)", color_continuous_scale=["#2ca02c", "#f4d03f", "#d62728"],
        text="Loss Ratio (%)",
    )
    fig_lob.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                           hovertemplate="%{y}: %{x:.1f}%<extra></extra>")
    fig_lob.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                           coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(fig_lob, use_container_width=True)

    st.subheader(f"Matching submissions ({len(filtered):,})")
    show_cols = ["submission_id", "line_of_business", "broker_tier", "total_insured_value_usd",
                 "indicated_annual_premium_usd", "predicted_loss_ratio_pct", "loss_propensity_score_v2"]
    st.dataframe(
        filtered[show_cols].head(500),
        use_container_width=True,
        hide_index=True,
        column_config={
            "submission_id": "Submission ID",
            "line_of_business": "Line of Business",
            "broker_tier": "Broker Tier",
            "total_insured_value_usd": st.column_config.NumberColumn("TIV", format="$%d"),
            "indicated_annual_premium_usd": st.column_config.NumberColumn("Premium", format="$%d"),
            "predicted_loss_ratio_pct": st.column_config.NumberColumn("Loss Ratio", format="%.1f%%"),
            "loss_propensity_score_v2": st.column_config.ProgressColumn(
                "Loss Propensity Score", min_value=10, max_value=100, format="%.0f"),
        },
    )

# ============================================================================
else:
    st.header("New Submission")
    st.caption("Enter the submission details below to generate a model-predicted risk score.")

    lob = st.selectbox("Line of Business", sorted(LOB_OPTIONS))
    group = group_for_lob(lob)
    is_property = group == "property"
    st.caption(
        f"Showing **{'property/building' if is_property else 'business operations'}** inputs for "
        f"{lob}. Fields greyed out below aren't part of this line's model - they don't move the "
        f"score, so they're disabled rather than left active but silently ignored."
    )

    with st.form("submission_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Property**")
            construction_type = st.selectbox("Construction Type", CATEGORICAL_OPTIONS["construction_type"], disabled=not is_property)
            year_built = st.number_input("Year Built", 1900, 2026, 2000, disabled=not is_property)
            square_footage = st.number_input("Square Footage", 500, 500000, 5000, disabled=not is_property)
            total_insured_value_usd = st.number_input("Total Insured Value ($)", 50000, 50000000, 500000, step=10000, disabled=not is_property)
            sprinklered_flag = st.selectbox("Sprinklered", CATEGORICAL_OPTIONS["sprinklered_flag"], disabled=not is_property)
            fire_protection_class_iso_ppc = st.slider("Fire Protection Class (ISO PPC, 1=best)", 1, 10, 5, disabled=not is_property)
            roof_age_years = st.number_input("Roof Age (years)", 0, 120, 15, disabled=not is_property)

        with col2:
            st.markdown("**Location Hazard**")
            flood_zone_tier = st.selectbox("Flood Zone Tier", CATEGORICAL_OPTIONS["flood_zone_tier"], disabled=not is_property)
            fema_flood_zone_code = st.selectbox("FEMA Flood Zone Code", CATEGORICAL_OPTIONS["fema_flood_zone_code"], disabled=not is_property)
            earthquake_zone_tier = st.selectbox("Earthquake Zone Tier", CATEGORICAL_OPTIONS["earthquake_zone_tier"], disabled=not is_property)
            soil_liquefaction_class = st.selectbox("Soil Liquefaction Class", CATEGORICAL_OPTIONS["soil_liquefaction_class"], disabled=not is_property)
            wildfire_zone_tier = st.selectbox("Wildfire Zone Tier", CATEGORICAL_OPTIONS["wildfire_zone_tier"], disabled=not is_property)
            wind_zone_tier = st.selectbox("Wind Zone Tier", CATEGORICAL_OPTIONS["wind_zone_tier"], disabled=not is_property)
            distance_to_coast_miles = st.number_input("Distance to Coast (miles)", 0.0, 800.0, 60.0, disabled=not is_property)
            distance_to_fault_line_miles = st.number_input("Distance to Fault Line (miles)", 0.0, 400.0, 60.0, disabled=not is_property)
            distance_to_nearest_water_body_miles = st.number_input("Distance to Nearest Water Body (miles)", 0.0, 50.0, 3.0, disabled=not is_property)

            st.markdown("**Business & Workforce**")
            industry_class_naics = st.selectbox("Industry Class (NAICS)", INDUSTRY_OPTIONS, disabled=is_property)
            number_of_employees = st.number_input("Number of Employees", 1, 20000, 25, disabled=is_property)
            avg_wage_per_employee_usd = st.number_input("Avg. Wage per Employee ($)", 15000, 300000, 55000, step=1000, disabled=is_property)
            avg_employee_age = st.number_input("Avg. Employee Age", 18, 70, 40, disabled=is_property)

        with col3:
            st.markdown("**Financial & History**")
            financial_stability_score = st.slider("Financial Stability Score", 0, 100, 55)
            safety_program_flag = st.selectbox("Safety Program in Place", ["No", "Yes"]) == "Yes"
            safety_inspection_findings = st.selectbox("Safety Inspection Findings", CATEGORICAL_OPTIONS["safety_inspection_findings"])
            prior_claim_count = st.number_input("Prior Claim Count", 0, 20, 0)
            prior_total_incurred_usd = st.number_input("Prior Total Incurred ($)", 0, 5000000, 0, step=1000)
            policy_term_months = st.selectbox("Policy Term (months)", [6, 12, 24, 36], index=1)
            st.caption("Premium is calculated automatically from the inputs above (TIV-based "
                       "for property, payroll-based for liability) once you submit - it isn't "
                       "typed in, since a manually guessed premium was distorting every score.")

        submitted = st.form_submit_button("Score Submission", type="primary")

    if submitted:
        building_age_years = 2026 - year_built
        validation_errors = []
        if is_property and roof_age_years > building_age_years:
            validation_errors.append(
                f"Roof Age ({roof_age_years} years) cannot exceed Building Age ({building_age_years} "
                f"years, based on Year Built) - a roof cannot be older than the building it's on."
            )
        if prior_claim_count == 0 and prior_total_incurred_usd > 0:
            validation_errors.append(
                "Prior Total Incurred is greater than $0 but Prior Claim Count is 0 - there can't be "
                "incurred losses with no prior claims on file."
            )

        if validation_errors:
            for e in validation_errors:
                st.error(e)
            st.stop()

        row = pd.DataFrame([{
            "line_of_business": lob, "construction_type": construction_type, "year_built": year_built,
            "building_age_years": building_age_years, "square_footage": square_footage,
            "total_insured_value_usd": total_insured_value_usd, "sprinklered_flag": sprinklered_flag,
            "fire_protection_class_iso_ppc": fire_protection_class_iso_ppc, "roof_age_years": roof_age_years,
            "flood_zone_tier": flood_zone_tier, "fema_flood_zone_code": fema_flood_zone_code,
            "earthquake_zone_tier": earthquake_zone_tier, "soil_liquefaction_class": soil_liquefaction_class,
            "wildfire_zone_tier": wildfire_zone_tier, "wind_zone_tier": wind_zone_tier,
            "distance_to_coast_miles": distance_to_coast_miles,
            "distance_to_fault_line_miles": distance_to_fault_line_miles,
            "distance_to_nearest_water_body_miles": distance_to_nearest_water_body_miles,
            "industry_class_naics": industry_class_naics, "number_of_employees": number_of_employees,
            "avg_wage_per_employee_usd": avg_wage_per_employee_usd, "avg_employee_age": avg_employee_age,
            "financial_stability_score": financial_stability_score, "safety_program_flag": safety_program_flag,
            "safety_inspection_findings": safety_inspection_findings, "prior_claim_count": prior_claim_count,
            "prior_total_incurred_usd": prior_total_incurred_usd, "policy_term_months": policy_term_months,
        }])

        try:
            result = score(row, models).iloc[0]
        except Exception as e:
            st.error(f"Scoring failed: {e}")
            st.stop()

        st.subheader("Result")
        st.caption(f"Auto-calculated premium used for this score: ${result['indicated_annual_premium_usd']:,.0f}")

        gauge_col, metric_col = st.columns([1, 1])
        with gauge_col:
            score_val = result["loss_propensity_score_v2"]
            gauge_color = "#2ca02c" if score_val < 40 else ("#f4d03f" if score_val < 70 else "#d62728")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score_val,
                number={"font": {"size": 34}},
                gauge={
                    "axis": {"range": [10, 100], "tickwidth": 1},
                    "bar": {"color": gauge_color, "thickness": 0.35},
                    "steps": [
                        {"range": [10, 40], "color": "#eafaf1"},
                        {"range": [40, 70], "color": "#fef9e7"},
                        {"range": [70, 100], "color": "#fdedec"},
                    ],
                    "threshold": {"line": {"color": gauge_color, "width": 4}, "thickness": 0.9, "value": score_val},
                },
                title={"text": "Loss Propensity Score", "font": {"size": 16}},
            ))
            fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with metric_col:
            st.metric("Predicted Loss Ratio", f"{result['predicted_loss_ratio_pct']:.1f}%")
            st.metric("Predicted Probability of Large Loss", f"{result['p_large_loss']*100:.1f}%")

        if result["loss_propensity_score_v2"] < 40:
            st.success("Preferred risk under this model.")
        elif result["loss_propensity_score_v2"] < 70:
            st.warning("Standard risk under this model.")
        else:
            st.error("High risk under this model - review recommended.")
