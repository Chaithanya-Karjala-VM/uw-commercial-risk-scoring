"""
Commercial UW Risk Intelligence - Streamlit app.
Three views (top tabs): Portfolio Overview, Executive Dashboard, Submission.
Run: streamlit run streamlit_app.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from data_loader import load_data
from scoring import load_models, score
from lob_config import PROPERTY_LOBS, LIABILITY_LOBS, group_for_lob, STATE_TO_REGION
from bind_propensity import NEW_BROKER_DEFAULT_TIER

st.set_page_config(page_title="Commercial UW Risk Intelligence", layout="wide")

# ---------------------------------------------------------------------------
# Palette, fonts, KPI card styling - built to react to the active theme (dark
# by default per .streamlit/config.toml, switchable to light from Settings)
# rather than hardcoded to one mode.
# ---------------------------------------------------------------------------
IS_DARK = st.context.theme.type == "dark"

NAVY = "#5B8DEF" if IS_DARK else "#1F3864"     # primary accent, adapted for contrast
ACCENT = "#7DA6F5" if IS_DARK else "#2E5090"
GOOD = "#3FC97F" if IS_DARK else "#1E8449"
WARN = "#E0B23D" if IS_DARK else "#B7950B"
BAD = "#E5685A" if IS_DARK else "#B03A2E"

CARD_BG = "#1B2532" if IS_DARK else "#FFFFFF"
CARD_BORDER = "#2C394A" if IS_DARK else "#E3E7EF"
TEXT_MAIN = "#E7ECF3" if IS_DARK else "#1A1A1A"
TEXT_MUTED = "#A9B4C2" if IS_DARK else "#33404F"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif; font-size: 16px; }}
    .block-container {{ padding-top: 1rem; padding-bottom: 2rem; max-width: 1500px; }}

    /* KPI cards - larger, higher-contrast text, colors follow the active theme */
    [data-testid="stMetric"] {{
        background: {CARD_BG}; border: 1px solid {CARD_BORDER}; border-radius: 10px;
        padding: 16px 18px 12px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }}
    [data-testid="stMetricLabel"] p {{ color: {TEXT_MUTED} !important; font-size: 1rem !important; font-weight: 600 !important; }}
    [data-testid="stMetricValue"] {{ color: {NAVY}; font-weight: 800; font-size: 2rem; font-variant-numeric: tabular-nums; }}

    h1 {{ color: {NAVY}; font-size: 2.3rem; }}
    h2 {{ color: {NAVY}; font-size: 1.5rem; }}
    h3 {{ color: {NAVY}; font-size: 1.2rem; }}
    p, label, .stCaption {{ font-size: 1rem; }}

    /* top tab bar - bigger, bolder, matches an Excel-sheet-tab look */
    div[data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 2px solid {CARD_BORDER}; }}
    button[data-baseweb="tab"] {{ font-size: 1.05rem; font-weight: 700; height: 46px; color: {TEXT_MUTED}; }}
    button[data-baseweb="tab"][aria-selected="true"] {{ color: {NAVY}; border-bottom: 3px solid {NAVY}; }}

    .section-divider {{ border-top: 2px solid {CARD_BORDER}; margin: 1.6rem 0 1.2rem 0; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_models():
    return load_models()


@st.cache_data
def get_raw_book():
    df, schema, usable_for, output_cols = load_data()
    df["effective_date"] = pd.to_datetime(df["effective_date"])
    return df


@st.cache_data
def get_scored_book():
    df = get_raw_book()
    models = get_models()
    scored = score(df, models)
    scored["loss_only_tier_v2"] = pd.cut(
        scored["loss_propensity_score_v2"], bins=[-1, 25, 50, 75, 101],
        labels=["Preferred", "Standard", "High", "Decline"]
    )
    return scored


models = get_models()
raw_book = get_raw_book()

INDUSTRY_OPTIONS = ["Accommodation & Food Services", "Administrative Services", "Construction",
                     "Health Care", "Manufacturing", "Professional Services", "Real Estate",
                     "Retail Trade", "Transportation & Warehousing", "Wholesale Trade"]
CATEGORICAL_OPTIONS = {
    "construction_type": ["Fire Resistive", "Non-Combustible", "Masonry Non-Combustible", "Joisted Masonry", "Frame"],
    "roof_type": ["Tile", "Metal", "Membrane/TPO", "Shingle", "Flat/Built-Up"],
    "fire_alarm_monitoring_type": ["Central Station Monitored", "Local Alarm Only"],
    "fema_flood_zone_code": ["X", "X (shaded)", "A", "AE", "V", "VE"],
    "soil_liquefaction_class": ["A - Hard Rock", "B - Rock", "C - Dense Soil", "D - Stiff Soil", "E - Soft Soil", "F - Special Study Required"],
    "occupancy_type": ["Office", "Retail Store", "Warehouse", "Distribution Center", "Restaurant", "Mixed Use", "Manufacturing Plant", "Habitational/Apartments"],
    "safety_inspection_findings": ["No Issues Noted", "Minor Issues Noted", "Moderate Issues – Corrective Action Required", "Major Deficiencies Identified"],
    "sprinklered_flag": ["Y", "N"],
    "tier": ["Low", "Moderate", "High", "Severe"],
}
LOB_OPTIONS = sorted(raw_book["line_of_business"].unique().tolist())
TIER_COLORS = {"Preferred": GOOD, "Standard": "#5D8AA8", "High": WARN, "Decline": BAD}

# ---------------------------------------------------------------------------
# Top header - logo beside the project name, tabs directly below (replaces the old
# sidebar navigation with an Excel-sheet-tab-style top nav)
# ---------------------------------------------------------------------------
LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "vm_logo.png"
logo_col, title_col = st.columns([1, 5], vertical_alignment="center")
with logo_col:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=170)
with title_col:
    st.markdown(f"<h1 style='margin-bottom:0;'>Commercial UW Risk Intelligence</h1>", unsafe_allow_html=True)
    st.caption("Model-driven Loss Propensity scoring for commercial submissions")

tab_portfolio, tab_dashboard, tab_submission = st.tabs(
    ["Portfolio Overview", "Executive Dashboard", "Submission"]
)

# ============================================================================

with tab_portfolio:
        st.header("Portfolio Overview")
        book = get_scored_book()

        c1, c2, c3 = st.columns(3)
        lob_filter = c1.selectbox("Line of Business", ["All"] + LOB_OPTIONS, key="po_lob")
        tier_filter = c2.selectbox("Broker Tier", ["All"] + sorted(book["broker_tier"].dropna().unique().tolist()), key="po_broker")
        score_range = c3.slider("Loss Propensity Score range", 0, 100, (0, 100), key="po_score_range")

        filtered = book.copy()
        if lob_filter != "All":
            filtered = filtered[filtered["line_of_business"] == lob_filter]
        if tier_filter != "All":
            filtered = filtered[filtered["broker_tier"] == tier_filter]
        filtered = filtered[filtered["loss_propensity_score_v2"].between(*score_range)]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Submissions", f"{len(filtered):,}")
        m2.metric("Avg Loss Propensity Score", f"{filtered['loss_propensity_score_v2'].mean():.1f}" if len(filtered) else "-")
        portfolio_lr = (filtered["total_expected_loss_cost"].sum() / filtered["indicated_annual_premium_usd"].sum() * 100
                         if len(filtered) and filtered["indicated_annual_premium_usd"].sum() > 0 else 0)
        m3.metric("Portfolio Loss Ratio", f"{portfolio_lr:.1f}%")
        m4.metric("Avg Premium", f"${filtered['indicated_annual_premium_usd'].mean():,.0f}" if len(filtered) else "-")

        if len(filtered):
            st.subheader("Loss Propensity Score distribution")
            fig_hist = px.histogram(filtered, x="loss_propensity_score_v2", nbins=20,
                                     labels={"loss_propensity_score_v2": "Loss Propensity Score"},
                                     color_discrete_sequence=[NAVY])
            fig_hist.update_traces(hovertemplate="Score band %{x}<br>%{y:,} submissions<extra></extra>")
            fig_hist.update_layout(yaxis_title="Number of Submissions", xaxis_title="Loss Propensity Score (0=best, 100=worst)",
                                    bargap=0.05, height=350, margin=dict(l=10, r=10, t=10, b=10),
                                    font=dict(size=14), title_font=dict(size=16))
            st.plotly_chart(fig_hist, width='stretch')

            st.subheader(f"Matching submissions ({len(filtered):,})")
            show_cols = ["submission_id", "line_of_business", "broker_tier", "total_insured_value_usd",
                         "indicated_annual_premium_usd", "predicted_loss_ratio_pct", "loss_propensity_score_v2"]
            st.dataframe(
                filtered[show_cols].head(500), width='stretch', hide_index=True,
                column_config={
                    "submission_id": "Submission ID", "line_of_business": "Line of Business", "broker_tier": "Broker Tier",
                    "total_insured_value_usd": st.column_config.NumberColumn("TIV", format="$%d"),
                    "indicated_annual_premium_usd": st.column_config.NumberColumn("Premium", format="$%d"),
                    "predicted_loss_ratio_pct": st.column_config.NumberColumn("Loss Ratio", format="%.1f%%"),
                    "loss_propensity_score_v2": st.column_config.ProgressColumn("Loss Propensity Score", min_value=0, max_value=100, format="%.0f"),
                },
            )
        else:
            st.info("No submissions match the current filter combination.")

    # ============================================================================

with tab_dashboard:
        st.header("Executive Dashboard")
        book = get_scored_book()

        f1, f2, f3, f4, f5 = st.columns(5)
        lob_f = f1.selectbox("Line of Business", ["All"] + LOB_OPTIONS, key="ed_lob")
        occ_f = f2.selectbox("Occupancy Type", ["All"] + sorted(book["occupancy_type"].dropna().unique().tolist()), key="ed_occ")
        tier_f = f3.selectbox("Loss-Only Tier", ["All", "Preferred", "Standard", "High", "Decline"], key="ed_tier")
        broker_f = f4.selectbox("Broker Tier", ["All"] + sorted(book["broker_tier"].dropna().unique().tolist()), key="ed_broker")
        renewal_f = f5.selectbox("Renewal / New Business", ["All", "Renewal", "New Business"], key="ed_renewal")

        fb = book.copy()
        if lob_f != "All": fb = fb[fb["line_of_business"] == lob_f]
        if occ_f != "All": fb = fb[fb["occupancy_type"] == occ_f]
        if tier_f != "All": fb = fb[fb["loss_only_tier_v2"] == tier_f]
        if broker_f != "All": fb = fb[fb["broker_tier"] == broker_f]
        if renewal_f != "All": fb = fb[fb["renewal_flag"] == renewal_f]

        st.caption(f"Showing {len(fb):,} of {len(book):,} submissions ({len(fb)/len(book)*100:.1f}%) matching current filters")

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Submissions", f"{len(fb):,}")
        k2.metric("Total TIV", f"${fb['total_insured_value_usd'].sum()/1e9:.2f}B" if len(fb) else "$0")
        k3.metric("Total Premium", f"${fb['indicated_annual_premium_usd'].sum()/1e6:.1f}M" if len(fb) else "$0")
        k4.metric("Avg Loss Propensity", f"{fb['loss_propensity_score_v2'].mean():.1f}" if len(fb) else "-")
        needs_review = (fb["loss_only_tier_v2"].isin(["High", "Decline"])).sum() if len(fb) else 0
        k5.metric("Needs Review", f"{needs_review:,}", help="High + Decline tier count")
        k6.metric("Large-Loss Count", f"{fb['large_claim_flag'].sum():,}" if len(fb) else "0")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("Portfolio Visuals")

        if len(fb) == 0:
            st.info("No submissions match the current filter combination.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                tier_counts = fb["loss_only_tier_v2"].value_counts().reindex(["Preferred", "Standard", "High", "Decline"]).fillna(0)
                fig = px.pie(names=tier_counts.index, values=tier_counts.values, hole=0.45,
                             color=tier_counts.index, color_discrete_map=TIER_COLORS,
                             title="Loss-Only Tier Distribution")
                fig.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value:,} submissions (%{percent})<extra></extra>")
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), showlegend=False, font=dict(size=13), title_font=dict(size=15))
                st.plotly_chart(fig, width='stretch')
            with c2:
                renewal_counts = fb["renewal_flag"].value_counts()
                fig = px.pie(names=renewal_counts.index, values=renewal_counts.values, hole=0.45,
                             color_discrete_sequence=[NAVY, "#8FA8C9"],
                             title="Renewal vs. New Business Mix")
                fig.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value:,} submissions (%{percent})<extra></extra>")
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), showlegend=False, font=dict(size=13), title_font=dict(size=15))
                st.plotly_chart(fig, width='stretch')

            c3, c4 = st.columns(2)
            with c3:
                lr_by_lob = fb.groupby("line_of_business").apply(
                    lambda g: g["total_expected_loss_cost"].sum() / g["indicated_annual_premium_usd"].sum() * 100
                    if g["indicated_annual_premium_usd"].sum() > 0 else 0, include_groups=False
                ).sort_values()
                fig = px.bar(x=lr_by_lob.values, y=lr_by_lob.index, orientation="h",
                             color=lr_by_lob.values, color_continuous_scale=[GOOD, WARN, BAD],
                             title="Avg Loss Ratio (Predicted) by LOB",
                             labels={"x": "Loss Ratio (%)", "y": ""})
                fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside", hovertemplate="%{y}: %{x:.1f}% predicted loss ratio<extra></extra>")
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), coloraxis_showscale=False, font=dict(size=13), title_font=dict(size=15))
                st.plotly_chart(fig, width='stretch')
            with c4:
                ct = fb.groupby(["line_of_business", "loss_only_tier_v2"], observed=True).size().reset_index(name="count")
                fig = px.bar(ct, x="count", y="line_of_business", color="loss_only_tier_v2", orientation="h",
                             color_discrete_map=TIER_COLORS, title="Submission Count by LOB x Tier",
                             labels={"count": "Submissions", "line_of_business": "", "loss_only_tier_v2": "Tier"})
                fig.update_traces(hovertemplate="%{y}<br>%{fullData.name}: %{x:,} submissions<extra></extra>")
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), legend_title="", font=dict(size=13), title_font=dict(size=15))
                st.plotly_chart(fig, width='stretch')

            c5, c6 = st.columns(2)
            with c5:
                tiv_region = fb.groupby("risk_location_region")["total_insured_value_usd"].sum().sort_values() / 1e6
                fig = px.bar(x=tiv_region.values, y=tiv_region.index, orientation="h",
                             color_discrete_sequence=[ACCENT],
                             title="Total TIV by Risk Region, $M", labels={"x": "TIV ($M)", "y": ""})
                fig.update_traces(hovertemplate="%{y}: $%{x:,.1f}M total TIV<extra></extra>")
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), font=dict(size=13), title_font=dict(size=15))
                st.plotly_chart(fig, width='stretch')
            with c6:
                large_by_tier = fb.groupby("loss_only_tier_v2", observed=True)["large_claim_flag"].sum().reindex(
                    ["Preferred", "Standard", "High", "Decline"]).fillna(0)
                fig = px.bar(x=large_by_tier.index, y=large_by_tier.values,
                             color=large_by_tier.index, color_discrete_map=TIER_COLORS,
                             title="Large-Loss Count by Tier", labels={"x": "", "y": "Large-Loss Count"})
                fig.update_traces(hovertemplate="%{x}: %{y:,} large losses<extra></extra>")
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), showlegend=False, font=dict(size=13), title_font=dict(size=15))
                st.plotly_chart(fig, width='stretch')

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            c7, c8 = st.columns(2)
            with c7:
                score_by_tier = fb.groupby("loss_only_tier_v2", observed=True).agg(
                    Loss=("loss_propensity_score_v2", "mean"),
                    Bind=("bind_propensity_score_v2", "mean"),
                    Appetite=("appetite_capacity_score_v2", "mean"),
                ).reindex(["Preferred", "Standard", "High", "Decline"]).reset_index()
                score_by_tier_long = score_by_tier.melt(id_vars="loss_only_tier_v2", var_name="Score Type", value_name="Avg Score")
                fig = px.bar(score_by_tier_long, x="loss_only_tier_v2", y="Avg Score", color="Score Type",
                             barmode="group", title="Avg Loss / Bind / Appetite Score by Tier",
                             color_discrete_sequence=[BAD, ACCENT, GOOD], labels={"loss_only_tier_v2": ""})
                fig.update_traces(hovertemplate="%{x}<br>%{fullData.name}: %{y:.1f}<extra></extra>")
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), legend_title="", font=dict(size=13), title_font=dict(size=15))
                st.plotly_chart(fig, width='stretch')
            with c8:
                bind_by_broker = fb.groupby("broker_tier")["bind_propensity_score_v2"].mean().reindex(
                    ["Platinum", "Gold", "Silver", "Bronze"]).dropna()
                fig = px.bar(x=bind_by_broker.index, y=bind_by_broker.values, color_discrete_sequence=[ACCENT],
                             title="Avg Bind Propensity by Broker Tier", labels={"x": "", "y": "Avg Bind Propensity"})
                fig.update_traces(hovertemplate="%{x} brokers: %{y:.1f} avg Bind Propensity<extra></extra>")
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), font=dict(size=13), title_font=dict(size=15))
                st.plotly_chart(fig, width='stretch')

            bind_by_size = fb.groupby("broker_size_class")["bind_propensity_score_v2"].mean().sort_values()
            fig = px.bar(x=bind_by_size.values, y=bind_by_size.index, orientation="h", color_discrete_sequence=[ACCENT],
                         title="Avg Bind Propensity by Broker Size", labels={"x": "Avg Bind Propensity", "y": ""})
            fig.update_traces(hovertemplate="%{y}: %{x:.1f} avg Bind Propensity<extra></extra>")
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=10), font=dict(size=13), title_font=dict(size=15))
            st.plotly_chart(fig, width='stretch')

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.subheader("Claims Overview")
            has_claim = fb[fb["claim_count_total"] > 0]
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Submissions w/ Claim", f"{len(has_claim):,}")
            c2.metric("Total Incurred", f"${fb['total_incurred_claim_amount_usd'].sum()/1e6:.1f}M")
            c3.metric("Open Claims", f"{fb['claim_count_open'].sum():,}")
            c4.metric("Settled Claims", f"{fb['claim_count_settled'].sum():,}")
            c5.metric("Large-Claim Count", f"{fb['large_claim_flag'].sum():,}")
            c6.metric("Catastrophic-Loss Count", f"{fb['catastrophic_loss_flag'].sum():,}")

            st.markdown("**Claims Detail by Line of Business**")
            detail = fb.groupby("line_of_business").agg(
                Submissions_w_Claim=("claim_count_total", lambda x: (x > 0).sum()),
                Total_Incurred=("total_incurred_claim_amount_usd", "sum"),
                Large_Loss_Count=("large_claim_flag", "sum"),
                Litigated_Count=("litigated_claim_flag", "sum"),
            ).reset_index()
            st.dataframe(detail, width='stretch', hide_index=True,
                         column_config={"Total_Incurred": st.column_config.NumberColumn("Total Incurred", format="$%d")})

            st.markdown("**Large-Loss Count by Peril / Cause of Claim**")
            large = fb[fb["large_claim_flag"] == 1]
            if len(large):
                cause = large["primary_cause_of_claim"].value_counts().sort_values()
                fig = px.bar(x=cause.values, y=cause.index, orientation="h", color_discrete_sequence=[BAD],
                             labels={"x": "Large-Loss Count", "y": ""})
                fig.update_traces(hovertemplate="%{y}: %{x:,} large losses<extra></extra>")
                fig.update_layout(height=max(300, len(cause) * 28), margin=dict(l=10, r=10, t=10, b=10), font=dict(size=13))
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No large losses in the current filter.")

    # ============================================================================

with tab_submission:
        st.header("Submission")
        st.caption("Enter the submission details below to generate a model-predicted risk score - "
                   "applies to any submission received, new business or renewal alike.")

        def _reset_tier_to_segment_default():
            # widgets with a key only use their initial value once - after that, index=
            # is silently ignored on rerun, so the tier default must be forced via an
            # on_change callback instead, or it never actually updates when segment changes
            st.session_state["sub_broker_tier"] = NEW_BROKER_DEFAULT_TIER[st.session_state["sub_broker_segment"]]

        lob_col, segment_col = st.columns(2)
        lob = lob_col.selectbox("Line of Business", LOB_OPTIONS, key="sub_lob")
        broker_segment = segment_col.selectbox(
            "Broker Capacity Segment", ["Large National", "Regional", "Small Independent"],
            key="sub_broker_segment", on_change=_reset_tier_to_segment_default,
            help="For a new broker relationship. Broker Tier below defaults to the typical "
                 "tier for this segment, and can be changed if the actual tier is known.")
        group = group_for_lob(lob)
        is_property = group == "property"
        st.caption(
            f"Showing **{'property/building' if is_property else 'business operations'}** inputs for "
            f"{lob}. Fields greyed out below aren't part of this line's model."
        )

        with st.form("submission_form"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown("**Property**")
                construction_type = st.selectbox("Construction Type", CATEGORICAL_OPTIONS["construction_type"], disabled=not is_property)
                year_built = st.number_input("Year Built", 1900, 2026, 2000, disabled=not is_property)
                built_to_code = st.selectbox("Built to Current Code", ["No", "Yes"], disabled=not is_property) == "Yes"
                num_stories = st.number_input("Number of Stories", 1, 50, 2, disabled=not is_property)
                square_footage = st.number_input("Square Footage", 500, 500000, 5000, disabled=not is_property)
                total_insured_value_usd = st.number_input("Total Insured Value ($)", 50000, 50000000, 500000, step=10000, disabled=not is_property)
                occupancy_type = st.selectbox("Occupancy Type", CATEGORICAL_OPTIONS["occupancy_type"], disabled=not is_property)
                roof_age_years = st.number_input("Roof Age (years)", 0, 120, 15, disabled=not is_property)
                roof_type = st.selectbox("Roof Type", CATEGORICAL_OPTIONS["roof_type"], disabled=not is_property)
                sprinklered_flag = st.selectbox("Sprinklered", CATEGORICAL_OPTIONS["sprinklered_flag"], disabled=not is_property)
                fire_protection_class_iso_ppc = st.slider("Fire Protection Class (ISO PPC, 1=best)", 1, 10, 5, disabled=not is_property)

            with col2:
                st.markdown("**Location Hazard**")
                distance_to_fire_station_miles = st.number_input("Distance to Fire Station (miles)", 0.0, 50.0, 5.0, disabled=not is_property)
                distance_to_fire_hydrant_miles = st.number_input("Distance to Fire Hydrant (miles)", 0.0, 10.0, 0.5, disabled=not is_property)
                fire_alarm_monitoring_type = st.selectbox("Fire Alarm Monitoring", CATEGORICAL_OPTIONS["fire_alarm_monitoring_type"], disabled=not is_property)
                flood_zone_tier = st.selectbox("Flood Zone Tier", CATEGORICAL_OPTIONS["tier"], disabled=not is_property)
                fema_flood_zone_code = st.selectbox("FEMA Flood Zone Code", CATEGORICAL_OPTIONS["fema_flood_zone_code"], disabled=not is_property)
                earthquake_zone_tier = st.selectbox("Earthquake Zone Tier", CATEGORICAL_OPTIONS["tier"], disabled=not is_property)
                soil_liquefaction_class = st.selectbox("Soil Liquefaction Class", CATEGORICAL_OPTIONS["soil_liquefaction_class"], disabled=not is_property)
                wildfire_zone_tier = st.selectbox("Wildfire Zone Tier", CATEGORICAL_OPTIONS["tier"], disabled=not is_property)
                wind_zone_tier = st.selectbox("Wind Zone Tier", CATEGORICAL_OPTIONS["tier"], disabled=not is_property)
                distance_to_coast_miles = st.number_input("Distance to Coast (miles)", 0.0, 800.0, 60.0, disabled=not is_property)
                distance_to_fault_line_miles = st.number_input("Distance to Fault Line (miles)", 0.0, 400.0, 60.0, disabled=not is_property)
                distance_to_nearest_water_body_miles = st.number_input("Distance to Nearest Water Body (miles)", 0.0, 50.0, 3.0, disabled=not is_property)

                st.markdown("**Business & Workforce**")
                industry_class_naics = st.selectbox("Industry Class (NAICS)", INDUSTRY_OPTIONS, disabled=is_property)
                number_of_employees = st.number_input("Number of Employees", 1, 20000, 25, disabled=is_property)
                avg_wage_per_employee_usd = st.number_input("Avg. Wage per Employee ($)", 15000, 300000, 55000, step=1000, disabled=is_property)
                avg_employee_age = st.number_input("Avg. Employee Age", 18, 70, 40, disabled=is_property)
                avg_weekly_work_hours = st.number_input("Avg. Weekly Work Hours", 10.0, 60.0, 40.0, disabled=is_property)

            with col3:
                st.markdown("**Claims History**")
                claim_count_last_3_years = st.number_input("Claim Count - Last 3 Years", 0, 20, 0)
                large_loss_count_last_5_years = st.number_input("Large-Loss Count - Last 5 Years", 0, 10, 0)
                claims_free_years = st.number_input("Claims-Free Years", 0, 10, 5)

                st.markdown("**Financial & Operational**")
                financial_stability_score = st.slider("Financial Stability Score", 0, 100, 55)
                safety_program_flag = st.selectbox("Safety Program in Place", ["No", "Yes"]) == "Yes"
                safety_inspection_findings = st.selectbox("Safety Inspection Findings", CATEGORICAL_OPTIONS["safety_inspection_findings"])
                business_tenure_years = st.number_input("Business Tenure (years)", 0, 100, 10)
                revenue_growth_trend_pct = st.number_input("Revenue Growth Trend (%)", -50.0, 100.0, 3.0)
                debt_to_equity_ratio = st.number_input("Debt-to-Equity Ratio", 0.0, 10.0, 1.0)
                policy_term_months = st.selectbox("Policy Term (months)", [6, 12, 24, 36], index=1)
                st.caption("Premium is calculated automatically from the inputs above - not typed in.")

            with col4:
                st.markdown("**Broker**")
                tier_options = ["Platinum", "Gold", "Silver", "Bronze"]
                st.session_state.setdefault("sub_broker_tier", NEW_BROKER_DEFAULT_TIER[broker_segment])
                broker_tier = st.selectbox(
                    "Broker Tier", tier_options, key="sub_broker_tier",
                    help=f"Defaults to the typical tier for {broker_segment}. "
                         f"Change this if the broker's actual tier is already known.")
                competing_quotes_count = st.number_input("Competing Quotes Count", 0, 10, 2)
                risk_state = st.selectbox("State", sorted(STATE_TO_REGION.keys()), index=sorted(STATE_TO_REGION.keys()).index("CA"))
                coverage_limit_requested_usd = st.number_input("Coverage Limit Requested ($)", 50000, 50000000, 500000, step=10000)

                st.markdown("**Compliance & Appetite**")
                sanctions_flag = st.selectbox("Sanctions/OFAC Watchlist Hit", ["No", "Yes"]) == "Yes"
                insolvency_flag = st.selectbox("Prior Insolvency/Bankruptcy", ["No", "Yes"]) == "Yes"
                restricted_ops_flag = st.selectbox("Restricted Operations", ["No", "Yes"]) == "Yes"
                region_lob_restricted_flag = st.selectbox("Region x LOB Restricted", ["No", "Yes"]) == "Yes"
                target_growth_segment_flag = st.selectbox("Target Growth Segment", ["No", "Yes"]) == "Yes"

            submitted = st.form_submit_button("Score Submission", type="primary")

        if submitted:
            building_age_years = 2026 - year_built
            validation_errors = []
            if is_property and roof_age_years > building_age_years:
                validation_errors.append(
                    f"Roof Age ({roof_age_years} years) cannot exceed Building Age ({building_age_years} "
                    f"years, based on Year Built) - a roof cannot be older than the building it's on."
                )
            if claim_count_last_3_years == 0 and large_loss_count_last_5_years > 0:
                validation_errors.append(
                    "Large-Loss Count (Last 5 Years) is greater than 0 but Claim Count (Last 3 Years) is 0 - "
                    "a large loss would also count as a claim if it happened within the last 3 years."
                )
            if validation_errors:
                for e in validation_errors:
                    st.error(e)
                st.stop()

            row = pd.DataFrame([{
                "line_of_business": lob, "construction_type": construction_type, "year_built": year_built,
                "building_age_years": building_age_years, "built_to_current_code_flag": built_to_code,
                "num_stories": num_stories, "square_footage": square_footage,
                "total_insured_value_usd": total_insured_value_usd, "occupancy_type": occupancy_type,
                "roof_age_years": roof_age_years, "roof_type": roof_type, "sprinklered_flag": sprinklered_flag,
                "fire_protection_class_iso_ppc": fire_protection_class_iso_ppc,
                "distance_to_fire_station_miles": distance_to_fire_station_miles,
                "distance_to_fire_hydrant_miles": distance_to_fire_hydrant_miles,
                "fire_alarm_monitoring_type": fire_alarm_monitoring_type,
                "flood_zone_tier": flood_zone_tier, "fema_flood_zone_code": fema_flood_zone_code,
                "earthquake_zone_tier": earthquake_zone_tier, "soil_liquefaction_class": soil_liquefaction_class,
                "wildfire_zone_tier": wildfire_zone_tier, "wind_zone_tier": wind_zone_tier,
                "distance_to_coast_miles": distance_to_coast_miles,
                "distance_to_fault_line_miles": distance_to_fault_line_miles,
                "distance_to_nearest_water_body_miles": distance_to_nearest_water_body_miles,
                "industry_class_naics": industry_class_naics, "number_of_employees": number_of_employees,
                "avg_wage_per_employee_usd": avg_wage_per_employee_usd, "avg_employee_age": avg_employee_age,
                "avg_weekly_work_hours": avg_weekly_work_hours,
                "claim_count_last_3_years": claim_count_last_3_years,
                "large_loss_count_last_5_years": large_loss_count_last_5_years,
                "claims_free_years": claims_free_years,
                "financial_stability_score": financial_stability_score, "safety_program_flag": safety_program_flag,
                "safety_inspection_findings": safety_inspection_findings,
                "business_tenure_years": business_tenure_years, "revenue_growth_trend_pct": revenue_growth_trend_pct,
                "debt_to_equity_ratio": debt_to_equity_ratio, "policy_term_months": policy_term_months,
                "broker_tier": broker_tier, "competing_quotes_count": competing_quotes_count,
                "risk_state": risk_state, "coverage_limit_requested_usd": coverage_limit_requested_usd,
                "sanctions_ofac_watchlist_flag": int(sanctions_flag),
                "prior_insolvency_bankruptcy_flag": int(insolvency_flag),
                "restricted_operations_flag": int(restricted_ops_flag),
                "region_lob_restricted_flag": int(region_lob_restricted_flag),
                "target_growth_segment_flag": int(target_growth_segment_flag),
            }])

            try:
                result = score(row, models).iloc[0]
            except Exception as e:
                st.error(f"Scoring failed: {e}")
                st.stop()

            st.subheader("Result")

            def mini_gauge(value, title, good_dir_low=True):
                if good_dir_low:
                    color = GOOD if value < 30 else (WARN if value < 60 else BAD)
                else:
                    color = BAD if value < 30 else (WARN if value < 60 else GOOD)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=value, number={"font": {"size": 26}},
                    gauge={"axis": {"range": [0, 100], "tickwidth": 1}, "bar": {"color": color, "thickness": 0.35}},
                    title={"text": title, "font": {"size": 13}},
                ))
                fig.update_layout(height=190, margin=dict(l=15, r=15, t=40, b=5))
                return fig

            g1, g2, g3, g4 = st.columns(4)
            g1.plotly_chart(mini_gauge(result["loss_propensity_score_v2"], "Loss Propensity (0=best)"), width='stretch')
            g2.plotly_chart(mini_gauge(result["bind_propensity_score_v2"], "Bind Propensity (100=best)", good_dir_low=False), width='stretch')
            g3.plotly_chart(mini_gauge(result["appetite_capacity_score_v2"], "Appetite/Capacity (100=best)", good_dir_low=False), width='stretch')
            g4.plotly_chart(mini_gauge(result["uw_score_v2"], "Final UW Score (100=best)", good_dir_low=False), width='stretch')

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Annual Premium", f"${result['indicated_annual_premium_usd']:,.0f}")
            m2.metric("Predicted Loss Ratio", f"{result['predicted_loss_ratio_pct']:.1f}%")
            m3.metric("Probability of Large Loss", f"{result['p_large_loss']*100:.1f}%")
            m4.metric("Total Expected Loss Cost", f"${result['total_expected_loss_cost']:,.0f}")

            if result["appetite_capacity_score_v2"] == 0:
                st.error("Outside Appetite - hard-rule gate failed (sanctions, insolvency, restricted operations, or region/LOB restriction). Decline regardless of loss or bind scores.")
            elif result["uw_score_v2"] >= 65:
                st.success("High-priority submission under this model - good risk, in appetite, likely to bind.")
            elif result["uw_score_v2"] >= 45:
                st.warning("Standard priority under this model.")
            else:
                st.error("Low priority under this model - review recommended.")
