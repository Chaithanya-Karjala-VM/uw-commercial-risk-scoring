"""
Streamlit prototype - Explore the existing book, or score a brand-new submission live.
Run: streamlit run streamlit_app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from data_loader import load_data
from scoring import load_models, score

st.set_page_config(page_title="UW Risk Scoring", layout="wide")

# cache_resource: loaded once per server session, not on every widget interaction
@st.cache_resource
def get_models():
    return load_models()

# cache_data: the scored book is recomputed only if the underlying data file changes
@st.cache_data
def get_scored_book():
    df, schema, usable_for, output_cols = load_data()
    df["effective_date"] = pd.to_datetime(df["effective_date"])
    models = get_models()
    return score(df, models)

models = get_models()
book = get_scored_book()
freq_saved = models["frequency"]

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
LOB_OPTIONS = book["line_of_business"].unique().tolist()

st.title("UW Risk Scoring")
mode = st.sidebar.radio("Mode", ["Explore Book", "Score a Submission"])

st.sidebar.info(
    "Scope note: only Loss Propensity is live right now (frequency, severity, large-loss "
    "models). Bind Propensity and Appetite/Capacity aren't trained yet, so broker fields "
    "don't affect the score below - that's next."
)

# ============================================================================
if mode == "Explore Book":
    st.header("Existing Book")

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
    st.bar_chart(filtered["loss_propensity_score_v2"].value_counts(bins=10).sort_index())

    st.subheader("Portfolio Loss Ratio by Line of Business")
    by_lob = filtered.groupby("line_of_business").apply(
        lambda g: g["total_expected_loss_cost"].sum() / g["indicated_annual_premium_usd"].sum() * 100
        if g["indicated_annual_premium_usd"].sum() > 0 else 0, include_groups=False
    )
    st.bar_chart(by_lob)

    st.subheader(f"Matching submissions ({len(filtered):,})")
    show_cols = ["submission_id", "line_of_business", "broker_tier", "total_insured_value_usd",
                 "indicated_annual_premium_usd", "predicted_loss_ratio_pct", "loss_propensity_score_v2"]
    st.dataframe(filtered[show_cols].head(500), use_container_width=True)

# ============================================================================
else:
    st.header("Score a New Submission")
    st.caption("Fill in the fields below and score it through the actual trained models - "
               "not a lookup, a live prediction.")

    with st.form("submission_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Property**")
            lob = st.selectbox("Line of Business", sorted(LOB_OPTIONS))
            construction_type = st.selectbox("Construction Type", CATEGORICAL_OPTIONS["construction_type"])
            year_built = st.number_input("Year Built", 1900, 2026, 2000)
            square_footage = st.number_input("Square Footage", 500, 500000, 5000)
            total_insured_value_usd = st.number_input("Total Insured Value ($)", 50000, 50000000, 500000, step=10000)
            sprinklered_flag = st.selectbox("Sprinklered", CATEGORICAL_OPTIONS["sprinklered_flag"])
            fire_protection_class_iso_ppc = st.slider("Fire Protection Class (ISO PPC, 1=best)", 1, 10, 5)

        with col2:
            st.markdown("**Roof & Location Hazard**")
            roof_age_years = st.number_input("Roof Age (years)", 0, 120, 15)
            flood_zone_tier = st.selectbox("Flood Zone Tier", CATEGORICAL_OPTIONS["flood_zone_tier"])
            fema_flood_zone_code = st.selectbox("FEMA Flood Zone Code", CATEGORICAL_OPTIONS["fema_flood_zone_code"])
            earthquake_zone_tier = st.selectbox("Earthquake Zone Tier", CATEGORICAL_OPTIONS["earthquake_zone_tier"])
            soil_liquefaction_class = st.selectbox("Soil Liquefaction Class", CATEGORICAL_OPTIONS["soil_liquefaction_class"])
            wildfire_zone_tier = st.selectbox("Wildfire Zone Tier", CATEGORICAL_OPTIONS["wildfire_zone_tier"])
            wind_zone_tier = st.selectbox("Wind Zone Tier", CATEGORICAL_OPTIONS["wind_zone_tier"])
            distance_to_coast_miles = st.number_input("Distance to Coast (miles)", 0.0, 800.0, 60.0)
            distance_to_fault_line_miles = st.number_input("Distance to Fault Line (miles)", 0.0, 400.0, 60.0)
            distance_to_nearest_water_body_miles = st.number_input("Distance to Nearest Water Body (miles)", 0.0, 50.0, 3.0)

        with col3:
            st.markdown("**Financial & History**")
            financial_stability_score = st.slider("Financial Stability Score", 0, 100, 55)
            safety_program_flag = st.selectbox("Safety Program in Place", ["No", "Yes"]) == "Yes"
            safety_inspection_findings = st.selectbox("Safety Inspection Findings", CATEGORICAL_OPTIONS["safety_inspection_findings"])
            prior_claim_count = st.number_input("Prior Claim Count", 0, 20, 0)
            prior_total_incurred_usd = st.number_input("Prior Total Incurred ($)", 0, 5000000, 0, step=1000)
            policy_term_months = st.selectbox("Policy Term (months)", [6, 12, 24, 36], index=1)
            indicated_annual_premium_usd = st.number_input("Indicated Annual Premium ($)", 100, 2000000, 5000, step=100)

        submitted = st.form_submit_button("Score Submission", type="primary")

    if submitted:
        building_age_years = 2026 - year_built
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
            "financial_stability_score": financial_stability_score, "safety_program_flag": safety_program_flag,
            "safety_inspection_findings": safety_inspection_findings, "prior_claim_count": prior_claim_count,
            "prior_total_incurred_usd": prior_total_incurred_usd, "policy_term_months": policy_term_months,
            "indicated_annual_premium_usd": indicated_annual_premium_usd,
        }])

        result = score(row, models).iloc[0]

        st.subheader("Result")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Loss Propensity Score", f"{result['loss_propensity_score_v2']:.0f}",
                   help="10 = best risk, 100 = worst. Matches the file's own scale.")
        r2.metric("Predicted Loss Ratio", f"{result['predicted_loss_ratio_pct']:.1f}%")
        r3.metric("Predicted Annual Claim Frequency", f"{result['predicted_frequency']:.3f}")
        r4.metric("P(Large Loss)", f"{result['p_large_loss']*100:.1f}%")

        st.write(f"Predicted typical claim size (if one happens): **${result['predicted_attritional_severity']:,.0f}**")
        st.write(f"Predicted large-claim size (if the large-loss event happens): **${result['expected_large_severity']:,.0f}**")
        st.write(f"Total expected annual loss cost: **${result['total_expected_loss_cost']:,.0f}**")

        if result["loss_propensity_score_v2"] < 40:
            st.success("Preferred-range risk under this model.")
        elif result["loss_propensity_score_v2"] < 70:
            st.warning("Standard-range risk.")
        else:
            st.error("High risk under this model - review recommended.")
