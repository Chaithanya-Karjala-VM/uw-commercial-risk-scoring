"""
Assisted Underwriting - Streamlit app.
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
from lob_config import PROPERTY_LOBS, LIABILITY_LOBS, group_for_lob, STATE_TO_REGION, CURRENT_YEAR, PRIMARY_CAT_PERIL_OPTIONS, region_for_state
from bind_propensity import NEW_BROKER_DEFAULT_TIER, KNOWN_BROKER_TIER, tier_for_known_broker
from decision_matrix_reference import RISK_TIER_ORDER, RISK_TIER_INFO
import driver_explain as dexp
from narrative import build_narrative, build_banner_summary

st.set_page_config(page_title="Assisted Underwriting", layout="wide")

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


def _rgba(hex_color, alpha):
    """Hex accent color -> a translucent rgba() of it, so a card can be tinted by an outcome
    color (tier, driver direction) while still sitting on top of the page's own dark/light
    background rather than fighting it with a flat opaque fill."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

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

    /* Drill-down / expander headers ("Technical & audit detail" etc.) - default Streamlit
    renders these as plain, unweighted text with no relation to the app's own palette. Bold,
    accent-colored, and sized to read as a real section heading, matching the card headings
    used everywhere else in the result view. */
    [data-testid="stExpander"] {{
        border: 1px solid {CARD_BORDER}; border-radius: 10px; background: {CARD_BG};
    }}
    [data-testid="stExpander"] summary {{
        font-weight: 700 !important; font-size: 1.02rem !important; color: {NAVY} !important;
        padding: 4px 2px;
    }}
    [data-testid="stExpander"] summary p {{
        font-weight: 700 !important; font-size: 1.02rem !important; color: {NAVY} !important;
    }}
    [data-testid="stExpander"] summary:hover p {{ color: {ACCENT} !important; }}
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
    return score(df, models)  # risk_tier_v2 etc. are computed inside score() itself now


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

def style_risk_tier_column(df, tier_col="Risk Tier"):
    """Colors the Risk Tier column's cell background to match the tier - same palette used
    across every chart, so a tier means the same color everywhere in the app."""
    def _color(val):
        color = TIER_COLORS.get(val)
        if color is None:
            return ""
        return f"background-color: {color}; color: white; font-weight: 600;"
    return df.style.map(_color, subset=[tier_col])

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
    st.markdown(f"<h1 style='margin-bottom:0;'>Assisted Underwriting</h1>", unsafe_allow_html=True)
    st.caption("Model-driven Loss Propensity scoring for commercial submissions")

tab_submission, tab_portfolio, tab_dashboard = st.tabs(
    ["What-If Simulator", "Portfolio Overview", "Executive Dashboard"]
)

# ============================================================================

with tab_portfolio:
        st.header("Portfolio Overview")
        book = get_scored_book()

        c1, c2 = st.columns(2)
        lob_filter = c1.selectbox("Line of Business", ["All"] + LOB_OPTIONS, key="po_lob")
        tier_filter = c2.selectbox("Broker Tier", ["All"] + sorted(book["broker_tier"].dropna().unique().tolist()), key="po_broker")

        filtered = book.copy()
        if lob_filter != "All":
            filtered = filtered[filtered["line_of_business"] == lob_filter]
        if tier_filter != "All":
            filtered = filtered[filtered["broker_tier"] == tier_filter]

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
        tier_f = f3.selectbox("Risk Tier", ["All"] + RISK_TIER_ORDER, key="ed_tier")
        broker_f = f4.selectbox("Broker Tier", ["All"] + sorted(book["broker_tier"].dropna().unique().tolist()), key="ed_broker")
        renewal_f = f5.selectbox("Renewal / New Business", ["All", "Renewal", "New Business"], key="ed_renewal")

        fb = book.copy()
        if lob_f != "All": fb = fb[fb["line_of_business"] == lob_f]
        if occ_f != "All": fb = fb[fb["occupancy_type"] == occ_f]
        if tier_f != "All": fb = fb[fb["risk_tier_v2"] == tier_f]
        if broker_f != "All": fb = fb[fb["broker_tier"] == broker_f]
        if renewal_f != "All": fb = fb[fb["renewal_flag"] == renewal_f]

        st.caption(f"Showing {len(fb):,} of {len(book):,} submissions ({len(fb)/len(book)*100:.1f}%) matching current filters")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Submissions", f"{len(fb):,}")
        k2.metric("Total Premium", f"${fb['indicated_annual_premium_usd'].sum()/1e6:.1f}M" if len(fb) else "$0")
        k3.metric("Avg Loss Propensity", f"{fb['loss_propensity_score_v2'].mean():.1f}" if len(fb) else "-")
        needs_review = (fb["risk_tier_v2"].isin(["High", "Decline"])).sum() if len(fb) else 0
        k4.metric("Needs Review", f"{needs_review:,}", help="High + Decline tier count")
        k5.metric("Large-Loss Count", f"{fb['large_claim_flag'].sum():,}" if len(fb) else "0")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("Underwriting Decision Matrix — Risk Tier Reference")
        if len(fb):
            matrix_rows = []
            for tier in RISK_TIER_ORDER:
                tb = fb[fb["risk_tier_v2"] == tier]
                info = RISK_TIER_INFO[tier]
                matrix_rows.append({
                    "Risk Tier": tier,
                    "Opportunity Premium": tb["indicated_annual_premium_usd"].sum() if len(tb) else 0,
                    "UW Score Band": info["uw_score_band"],
                    "Avg Loss Propensity": tb["loss_propensity_score_v2"].mean() if len(tb) else None,
                    "Avg Claim Count": tb["claim_count_total"].mean() if len(tb) else None,
                    "Avg Bind Propensity": tb["bind_propensity_score_v2"].mean() if len(tb) else None,
                    "Avg Appetite Score": tb["appetite_capacity_score_v2"].mean() if len(tb) else None,
                    "Avg UW Score": tb["uw_score_v2"].mean() if len(tb) else None,
                    "What It Means": info["what_it_means"],
                    "UW Action": info["uw_action"],
                    "Pricing Guidance": info["pricing_guidance"],
                })
            matrix_rows.append({
                "Risk Tier": "Total / Book Average",
                "Opportunity Premium": fb["indicated_annual_premium_usd"].sum(),
                "UW Score Band": "—",
                "Avg Loss Propensity": fb["loss_propensity_score_v2"].mean(),
                "Avg Claim Count": fb["claim_count_total"].mean(),
                "Avg Bind Propensity": fb["bind_propensity_score_v2"].mean(),
                "Avg Appetite Score": fb["appetite_capacity_score_v2"].mean(),
                "Avg UW Score": fb["uw_score_v2"].mean(),
                "What It Means": "", "UW Action": "", "Pricing Guidance": "",
            })
            matrix_df = pd.DataFrame(matrix_rows)
            st.dataframe(
                style_risk_tier_column(matrix_df, "Risk Tier"), width='stretch', hide_index=True,
                column_config={
                    "Opportunity Premium": st.column_config.NumberColumn("Opportunity Premium ($)", format="$%d"),
                    "Avg Loss Propensity": st.column_config.NumberColumn("Avg Loss Propensity", format="%.1f"),
                    "Avg Claim Count": st.column_config.NumberColumn("Avg Claim Count", format="%.2f"),
                    "Avg Bind Propensity": st.column_config.NumberColumn("Avg Bind Propensity", format="%.1f"),
                    "Avg Appetite Score": st.column_config.NumberColumn("Avg Appetite Score", format="%.1f"),
                    "Avg UW Score": st.column_config.NumberColumn("Avg UW Score", format="%.1f"),
                },
            )
        else:
            st.info("No submissions match the current filter combination.")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("Portfolio Visuals")

        if len(fb) == 0:
            st.info("No submissions match the current filter combination.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                tier_counts = fb["risk_tier_v2"].value_counts().reindex(RISK_TIER_ORDER).fillna(0)
                fig = px.pie(names=tier_counts.index, values=tier_counts.values, hole=0.45,
                             color=tier_counts.index, color_discrete_map=TIER_COLORS,
                             title="Risk Tier Distribution")
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
                ct = fb.groupby(["line_of_business", "risk_tier_v2"], observed=True).size().reset_index(name="count")
                fig = px.bar(ct, x="count", y="line_of_business", color="risk_tier_v2", orientation="h",
                             color_discrete_map=TIER_COLORS, title="Submission Count by LOB x Risk Tier",
                             labels={"count": "Submissions", "line_of_business": "", "risk_tier_v2": "Tier"})
                fig.update_traces(hovertemplate="%{y}<br>%{fullData.name}: %{x:,} submissions<extra></extra>")
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), legend_title="", font=dict(size=13), title_font=dict(size=15))
                st.plotly_chart(fig, width='stretch')
            with c4:
                tiv_region = fb.groupby("risk_location_region")["total_insured_value_usd"].sum().sort_values() / 1e6
                fig = px.bar(x=tiv_region.values, y=tiv_region.index, orientation="h",
                             color_discrete_sequence=[ACCENT],
                             title="Total TIV by Risk Region, $M", labels={"x": "TIV ($M)", "y": ""})
                fig.update_traces(hovertemplate="%{y}: $%{x:,.1f}M total TIV<extra></extra>")
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), font=dict(size=13), title_font=dict(size=15))
                st.plotly_chart(fig, width='stretch')

            c5, c6 = st.columns(2)
            with c5:
                large_by_tier = fb.groupby("risk_tier_v2", observed=True)["large_claim_flag"].sum().reindex(RISK_TIER_ORDER).fillna(0)
                fig = px.bar(x=large_by_tier.index, y=large_by_tier.values,
                             color=large_by_tier.index, color_discrete_map=TIER_COLORS,
                             title="Large-Loss Count by Tier", labels={"x": "", "y": "Large-Loss Count"},
                             text=large_by_tier.values)
                fig.update_traces(hovertemplate="%{x}: %{y:,} large losses<extra></extra>",
                                   texttemplate="%{text:,.0f}", textposition="outside")
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=30), showlegend=False, font=dict(size=13), title_font=dict(size=15))
                st.plotly_chart(fig, width='stretch')
            with c6:
                score_by_tier = fb.groupby("risk_tier_v2", observed=True).agg(
                    Loss=("loss_propensity_score_v2", "mean"),
                    Bind=("bind_propensity_score_v2", "mean"),
                    Appetite=("appetite_capacity_score_v2", "mean"),
                ).reindex(RISK_TIER_ORDER).reset_index()
                score_by_tier_long = score_by_tier.melt(id_vars="risk_tier_v2", var_name="Score Type", value_name="Avg Score")
                fig = px.bar(score_by_tier_long, x="risk_tier_v2", y="Avg Score", color="Score Type",
                             barmode="group", title="Avg Loss / Bind / Appetite Score by Tier",
                             color_discrete_sequence=[BAD, ACCENT, GOOD], labels={"risk_tier_v2": ""})
                fig.update_traces(hovertemplate="%{x}<br>%{fullData.name}: %{y:.1f}<extra></extra>")
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), legend_title="", font=dict(size=13), title_font=dict(size=15))
                st.plotly_chart(fig, width='stretch')

            c7, c8 = st.columns(2)
            with c7:
                bind_by_broker = fb.groupby("broker_tier")["bind_propensity_score_v2"].mean().reindex(
                    ["Platinum", "Gold", "Silver", "Bronze"]).dropna()
                fig = px.bar(x=bind_by_broker.index, y=bind_by_broker.values, color_discrete_sequence=[ACCENT],
                             title="Avg Bind Propensity by Broker Tier", labels={"x": "", "y": "Avg Bind Propensity"})
                fig.update_traces(hovertemplate="%{x} brokers: %{y:.1f} avg Bind Propensity<extra></extra>")
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), font=dict(size=13), title_font=dict(size=15))
                st.plotly_chart(fig, width='stretch')
            with c8:
                bind_by_size = fb.groupby("broker_size_class")["bind_propensity_score_v2"].mean().sort_values()
                fig = px.bar(x=bind_by_size.values, y=bind_by_size.index, orientation="h", color_discrete_sequence=[ACCENT],
                             title="Avg Bind Propensity by Broker Size", labels={"x": "Avg Bind Propensity", "y": ""})
                fig.update_traces(hovertemplate="%{y}: %{x:.1f} avg Bind Propensity<extra></extra>")
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), font=dict(size=13), title_font=dict(size=15))
                st.plotly_chart(fig, width='stretch')

            st.markdown("**Broker Size Segmentation**")
            seg = fb.groupby("broker_size_class").agg(
                Count=("submission_id", "count"),
                New_Biz_Pct=("renewal_flag", lambda x: (x == "New Business").mean() * 100),
                Historical_Loss_Ratio_Pct=("loss_ratio_after_propensity_pct", "mean"),
                Renewal_Retention_Pct=("broker_renewal_retention_pct", "mean"),
            ).reset_index()
            st.dataframe(
                seg, width='stretch', hide_index=True,
                column_config={
                    "broker_size_class": "Segment", "Count": "Count",
                    "New_Biz_Pct": st.column_config.NumberColumn("New Biz %", format="%.1f%%"),
                    "Historical_Loss_Ratio_Pct": st.column_config.NumberColumn("Historical Loss Ratio %", format="%.1f%%"),
                    "Renewal_Retention_Pct": st.column_config.NumberColumn("Renewal Retention %", format="%.1f%%"),
                },
            )

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
            detail_lob = fb.groupby("line_of_business").agg(
                Submissions_w_Claim=("claim_count_total", lambda x: (x > 0).sum()),
                Total_Incurred=("total_incurred_claim_amount_usd", "sum"),
                Large_Loss_Count=("large_claim_flag", "sum"),
                Litigated_Count=("litigated_claim_flag", "sum"),
                Avg_Claim_Count=("claim_count_total", "mean"),
            ).reset_index()
            st.dataframe(detail_lob, width='stretch', hide_index=True,
                         column_config={
                             "Total_Incurred": st.column_config.NumberColumn("Total Incurred", format="$%d"),
                             "Avg_Claim_Count": st.column_config.NumberColumn("Avg Claim Count", format="%.2f"),
                         })

            st.markdown("**Claims Detail by Risk Tier**")
            detail_tier = fb.groupby("risk_tier_v2", observed=True).agg(
                Submissions_w_Claim=("claim_count_total", lambda x: (x > 0).sum()),
                Total_Incurred=("total_incurred_claim_amount_usd", "sum"),
                Large_Loss_Count=("large_claim_flag", "sum"),
                Catastrophic_Loss_Count=("catastrophic_loss_flag", "sum"),
                Avg_Claim_Count=("claim_count_total", "mean"),
            ).reindex(RISK_TIER_ORDER).reset_index()
            st.dataframe(style_risk_tier_column(detail_tier, "risk_tier_v2"), width='stretch', hide_index=True,
                         column_config={
                             "risk_tier_v2": "Risk Tier",
                             "Total_Incurred": st.column_config.NumberColumn("Total Incurred", format="$%d"),
                             "Avg_Claim_Count": st.column_config.NumberColumn("Avg Claim Count", format="%.2f"),
                         })

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

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.subheader("Decision Output 1 — Top 20 Referral Queue")
            referral = fb.sort_values("predicted_loss_ratio_pct", ascending=False).head(20)
            referral_display = referral[["submission_id", "line_of_business", "risk_location_region", "total_insured_value_usd",
                          "predicted_loss_ratio_pct", "risk_tier_v2", "large_claim_flag",
                          "loss_propensity_score_v2", "bind_propensity_score_v2", "appetite_capacity_score_v2",
                          "uw_score_v2", "primary_cause_of_claim", "hospital_visit_count",
                          "expected_further_visits", "recommended_uw_action_v2", "decline_review_reason_v2"]].copy()
            referral_display["large_claim_flag"] = referral_display["large_claim_flag"].map({1: "Yes", 0: "No", True: "Yes", False: "No"})
            st.dataframe(
                style_risk_tier_column(referral_display, "risk_tier_v2"),
                width='stretch', hide_index=True,
                column_config={
                    "submission_id": "Submission ID", "line_of_business": "LOB", "risk_location_region": "Risk Region",
                    "total_insured_value_usd": st.column_config.NumberColumn("TIV", format="$%d"),
                    "predicted_loss_ratio_pct": st.column_config.NumberColumn("Predicted Loss Ratio", format="%.1f%%"),
                    "risk_tier_v2": "Risk Tier", "large_claim_flag": "Large Loss?",
                    "loss_propensity_score_v2": "Loss Prop.", "bind_propensity_score_v2": "Bind Prop.",
                    "appetite_capacity_score_v2": "Appetite/Cap.", "uw_score_v2": "UW Score",
                    "primary_cause_of_claim": "Cause of Claim", "hospital_visit_count": "Hosp. Visits",
                    "expected_further_visits": "Exp. Further Visits", "recommended_uw_action_v2": "Recommended UW Action",
                    "decline_review_reason_v2": "Reason for Decline/Refer",
                },
            )

            st.subheader("Decision Output 2 — Top 10 Priority Opportunities")
            priority_pool = fb[fb["appetite_capacity_decision_v2"] == "Within Appetite"]
            priority = priority_pool.sort_values("uw_score_v2", ascending=False).head(10)
            priority_display = priority[["submission_id", "line_of_business", "risk_location_region", "broker_tier",
                          "total_insured_value_usd", "predicted_loss_ratio_pct", "risk_tier_v2",
                          "loss_propensity_score_v2", "bind_propensity_score_v2", "appetite_capacity_score_v2",
                          "uw_score_v2", "appetite_capacity_decision_v2", "recommended_uw_action_v2",
                          "priority_strength_reason_v2"]]
            st.dataframe(
                style_risk_tier_column(priority_display, "risk_tier_v2"),
                width='stretch', hide_index=True,
                column_config={
                    "submission_id": "Submission ID", "line_of_business": "LOB", "risk_location_region": "Risk Region",
                    "broker_tier": "Broker Tier",
                    "total_insured_value_usd": st.column_config.NumberColumn("TIV", format="$%d"),
                    "predicted_loss_ratio_pct": st.column_config.NumberColumn("Predicted Loss Ratio", format="%.1f%%"),
                    "risk_tier_v2": "Risk Tier", "loss_propensity_score_v2": "Loss Prop.",
                    "bind_propensity_score_v2": "Bind Prop.", "appetite_capacity_score_v2": "Appetite/Cap.",
                    "uw_score_v2": "UW Score", "appetite_capacity_decision_v2": "Appetite Decision",
                    "recommended_uw_action_v2": "Recommended UW Action", "priority_strength_reason_v2": "Reason This Is a Priority",
                },
            )

            st.subheader("Decision Output 3 — Emerging Risk Watch List")
            watch_pool = fb[fb["claim_count_total"] <= 1]
            watch = watch_pool.sort_values("prospective_risk_flag_count_v2", ascending=False).head(15)
            watch_display = watch[["submission_id", "line_of_business", "risk_location_region", "broker_tier",
                       "total_insured_value_usd", "risk_tier_v2", "claim_count_total", "loss_propensity_score_v2",
                       "prospective_risk_flag_count_v2", "prospective_risk_flags_v2", "recommended_uw_action_v2"]]
            st.dataframe(
                style_risk_tier_column(watch_display, "risk_tier_v2"),
                width='stretch', hide_index=True,
                column_config={
                    "submission_id": "Submission ID", "line_of_business": "LOB", "risk_location_region": "Risk Region",
                    "broker_tier": "Broker Tier",
                    "total_insured_value_usd": st.column_config.NumberColumn("TIV", format="$%d"),
                    "risk_tier_v2": "Risk Tier", "claim_count_total": "Claim Count",
                    "loss_propensity_score_v2": "Loss Prop.", "prospective_risk_flag_count_v2": "Flag Count",
                    "prospective_risk_flags_v2": "Prospective Risk Factors", "recommended_uw_action_v2": "Recommended UW Action",
                },
            )

    # ============================================================================

with tab_submission:
        st.header("What-If Simulator")
        st.caption("Enter submission details to generate a model-predicted risk score - "
                   "applies to any submission received, new business or renewal alike.")

        def _reset_tier_to_segment_default():
            st.session_state["sub_broker_tier"] = NEW_BROKER_DEFAULT_TIER[st.session_state["sub_broker_segment"]]

        def _reset_tier_to_known_broker():
            broker_id = st.session_state["sub_broker_id"]
            real_tier = tier_for_known_broker(broker_id)
            if real_tier:
                st.session_state["sub_broker_tier"] = real_tier

        def _reset_tier_on_toggle():
            # switching Existing<->New must also force-update tier immediately - same
            # session_state-stickiness issue as segment/broker changes: without this,
            # tier keeps whatever value it had from the OTHER path, which is wrong
            if st.session_state["sub_is_existing_broker"] == "Yes - select broker":
                broker_id = st.session_state.get("sub_broker_id", sorted(KNOWN_BROKER_TIER.keys())[0])
                real_tier = tier_for_known_broker(broker_id)
                if real_tier:
                    st.session_state["sub_broker_tier"] = real_tier
            else:
                segment = st.session_state.get("sub_broker_segment", "Large National")
                st.session_state["sub_broker_tier"] = NEW_BROKER_DEFAULT_TIER[segment]

        lob_col, channel_col = st.columns(2)
        lob = lob_col.selectbox("Line of Business", LOB_OPTIONS, key="sub_lob")
        channel = channel_col.selectbox("Channel", ["Broker", "Direct", "MGA"], key="sub_channel")

        is_existing_broker = st.radio(
            "Is this an existing broker relationship?", ["No - new broker", "Yes - select broker"],
            key="sub_is_existing_broker", on_change=_reset_tier_on_toggle, horizontal=True
        ) == "Yes - select broker"

        if is_existing_broker:
            broker_id = st.selectbox(
                "Broker", sorted(KNOWN_BROKER_TIER.keys()), key="sub_broker_id",
                on_change=_reset_tier_to_known_broker,
                help="Broker Tier below auto-fills with this broker's real, on-file tier.")
            st.session_state.setdefault("sub_broker_tier", tier_for_known_broker(broker_id))
            broker_segment = None
        else:
            broker_segment = st.selectbox(
                "Broker Capacity Segment", ["Large National", "Regional", "Small Independent"],
                key="sub_broker_segment", on_change=_reset_tier_to_segment_default,
                help="Broker Tier below defaults to the typical tier for this segment, and "
                     "can be changed if the actual tier is known.")
            st.session_state.setdefault("sub_broker_tier", NEW_BROKER_DEFAULT_TIER[broker_segment])

        group = group_for_lob(lob)
        is_property = group == "property"
        st.caption(
            f"Showing **{'property/building' if is_property else 'business operations'}** inputs for {lob}."
        )

        with st.form("submission_form"):
            if is_property:
                col1, col2, col3, col4 = st.columns(4)
            else:
                col1, col3, col4 = st.columns(3)

            with col1:
                if is_property:
                    st.markdown("**Property & Workforce**")
                    st.caption("Property")
                    construction_type = st.selectbox("Construction Type", CATEGORICAL_OPTIONS["construction_type"])
                    year_built = st.number_input("Year Built", 1910, CURRENT_YEAR, 2000)
                    built_to_code = st.selectbox("Built to Current Code", ["No", "Yes"]) == "Yes"
                    num_stories = st.number_input("Number of Stories", 1, 11, 2)
                    square_footage = st.number_input("Square Footage", 1000, 268000, 5000)
                    total_insured_value_usd = st.number_input("Total Insured Value ($)", 40000, 25000000, 500000, step=10000)
                    occupancy_type = st.selectbox("Occupancy Type", CATEGORICAL_OPTIONS["occupancy_type"])
                    roof_age_years = st.number_input("Roof Age (years)", 1, 120, 15)
                    roof_type = st.selectbox("Roof Type", CATEGORICAL_OPTIONS["roof_type"])
                    sprinklered_flag = st.selectbox("Sprinklered", CATEGORICAL_OPTIONS["sprinklered_flag"])
                    fire_protection_class_iso_ppc = st.slider("Fire Protection Class (ISO PPC, 1=best)", 1, 10, 5)
                    # this LOB's own model never uses employee/workforce fields - fixed to the
                    # form's previous defaults so nothing downstream changes behavior, just no
                    # longer shown for a line that doesn't need them
                    number_of_employees, avg_wage_per_employee_usd = 25, 55000
                    avg_employee_age, avg_weekly_work_hours = 40, 40.0
                else:
                    st.markdown("**Property & Workforce**")
                    st.caption("Business & Workforce")
                    number_of_employees = st.number_input("Number of Employees", 1, 5000, 25)
                    avg_wage_per_employee_usd = st.number_input("Avg. Wage per Employee ($)", 46000, 94000, 55000, step=1000)
                    avg_employee_age = st.number_input("Avg. Employee Age", 19, 66, 40)
                    avg_weekly_work_hours = st.number_input("Avg. Weekly Work Hours", 18.0, 50.0, 40.0)
                    # this LOB's own model never uses building/property fields - fixed to the
                    # form's previous defaults for the same reason as above. flood/earthquake/
                    # wildfire tier and TIV are the one exception: appetite_capacity.py's
                    # CAT/TIV penalty reads them for every LOB, not just property, so these 4
                    # specifically still need a real (if neutral) value even though this line's
                    # own risk model never looks at them.
                    construction_type, year_built, built_to_code = "Fire Resistive", 2000, False
                    num_stories, square_footage = 2, 5000
                    total_insured_value_usd = 500000
                    occupancy_type, roof_age_years, roof_type = "Office", 15, "Tile"
                    sprinklered_flag, fire_protection_class_iso_ppc = "Y", 5
                    distance_to_fire_station_miles, distance_to_fire_hydrant_miles = 5.0, 0.5
                    fire_alarm_monitoring_type = "Central Station Monitored"
                    flood_zone_tier, earthquake_zone_tier, wildfire_zone_tier, wind_zone_tier = "Low", "Low", "Low", "Low"
                    fema_flood_zone_code, soil_liquefaction_class = "X", "A - Hard Rock"
                    primary_cat_peril_exposure = "Low/Minimal CAT Exposure"
                    distance_to_coast_miles, distance_to_fault_line_miles = 60.0, 60.0
                    distance_to_nearest_water_body_miles = 3.0

            if is_property:
                with col2:
                    st.markdown("**Location Hazard**")
                    distance_to_fire_station_miles = st.number_input("Distance to Fire Station (miles)", 0.1, 22.0, 5.0)
                    distance_to_fire_hydrant_miles = st.number_input("Distance to Fire Hydrant (miles)", 0.0, 3.0, 0.5)
                    fire_alarm_monitoring_type = st.selectbox("Fire Alarm Monitoring", CATEGORICAL_OPTIONS["fire_alarm_monitoring_type"])
                    flood_zone_tier = st.selectbox("Flood Zone Tier", CATEGORICAL_OPTIONS["tier"])
                    fema_flood_zone_code = st.selectbox("FEMA Flood Zone Code", CATEGORICAL_OPTIONS["fema_flood_zone_code"])
                    earthquake_zone_tier = st.selectbox("Earthquake Zone Tier", CATEGORICAL_OPTIONS["tier"])
                    soil_liquefaction_class = st.selectbox("Soil Liquefaction Class", CATEGORICAL_OPTIONS["soil_liquefaction_class"])
                    wildfire_zone_tier = st.selectbox("Wildfire Zone Tier", CATEGORICAL_OPTIONS["tier"])
                    wind_zone_tier = st.selectbox("Wind Zone Tier", CATEGORICAL_OPTIONS["tier"])
                    primary_cat_peril_exposure = st.selectbox(
                        "Primary CAT Peril Exposure", PRIMARY_CAT_PERIL_OPTIONS,
                        help="The dominant catastrophe peril this risk is exposed to.")
                    distance_to_coast_miles = st.number_input("Distance to Coast (miles)", 0.0, 800.0, 60.0)
                    distance_to_fault_line_miles = st.number_input("Distance to Fault Line (miles)", 0.0, 400.0, 60.0)
                    distance_to_nearest_water_body_miles = st.number_input("Distance to Nearest Water Body (miles)", 0.0, 50.0, 3.0)

            with col3:
                st.markdown("**Claims History**")
                claim_count_last_3_years = st.number_input("Claim Count - Last 3 Years", 0, 9, 0)
                large_loss_count_last_5_years = st.number_input("Large-Loss Count - Last 5 Years", 0, 3, 0)
                claims_free_years = st.number_input("Claims-Free Years", 0, 10, 5)

                st.markdown("**Financial & Operational**")
                financial_stability_score = st.slider("Financial Stability Score", 5, 100, 55)
                safety_program_flag = st.selectbox("Safety Program in Place", ["No", "Yes"]) == "Yes"
                safety_inspection_findings = st.selectbox("Safety Inspection Findings", CATEGORICAL_OPTIONS["safety_inspection_findings"])
                business_tenure_years = st.number_input("Business Tenure (years)", 0, 80, 10)
                revenue_growth_trend_pct = st.number_input("Revenue Growth Trend (%)", -30.0, 60.0, 3.0)
                debt_to_equity_ratio = st.number_input("Debt-to-Equity Ratio", 0.0, 5.0, 1.0)
                policy_term_months = st.selectbox("Policy Term (months)", [6, 12, 24, 36], index=1)
                st.caption("Premium is calculated automatically from the inputs above - not typed in.")

            with col4:
                st.markdown("**Broker**")
                tier_options = ["Platinum", "Gold", "Silver", "Bronze"]
                broker_tier = st.selectbox(
                    "Broker Tier", tier_options, key="sub_broker_tier",
                    help="Auto-filled above; change here if the actual tier differs from the default.")
                competing_quotes_count = st.number_input("Competing Quotes Count", 0, 10, 2)
                risk_state = st.selectbox("State", sorted(STATE_TO_REGION.keys()), index=sorted(STATE_TO_REGION.keys()).index("CA"))
                coverage_limit_requested_usd = st.number_input("Coverage Limit Requested ($)", 50000, 50000000, 500000, step=10000)
                industry_class_naics = st.selectbox(
                    "Industry Class (NAICS)", INDUSTRY_OPTIONS,
                    help="Used for portfolio industry-concentration checks - applies to every "
                         "line, not just this submission's own risk model.")

                st.markdown("**Compliance & Appetite**")
                sanctions_flag = st.selectbox("Sanctions/OFAC Watchlist Hit", ["No", "Yes"]) == "Yes"
                insolvency_flag = st.selectbox("Prior Insolvency/Bankruptcy", ["No", "Yes"]) == "Yes"
                restricted_ops_flag = st.selectbox("Restricted Operations", ["No", "Yes"]) == "Yes"
                region_lob_restricted_flag = st.selectbox("Region x LOB Restricted", ["No", "Yes"]) == "Yes"
                target_growth_segment_flag = st.selectbox("Target Growth Segment", ["No", "Yes"]) == "Yes"

            submitted = st.form_submit_button("Calculate", type="primary")

        if submitted:
            building_age_years = CURRENT_YEAR - year_built
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
                "primary_cat_peril_exposure": primary_cat_peril_exposure,
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
                "broker_tier": broker_tier, "channel": channel, "competing_quotes_count": competing_quotes_count,
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

            # driver attribution needs risk_location_region explicitly - score() derives it
            # internally from State but doesn't hand it back to the caller's own row, and
            # bind_drivers() needs the real one (not a silent Gulf Coast default) to identify
            # the actual region driver correctly
            row_dict = row.iloc[0].to_dict()
            row_dict["risk_location_region"] = region_for_state(risk_state)

            def _score_fn(df):
                return score(df, models).iloc[0]

            loss_drv = dexp.loss_drivers(row, _score_fn)
            bind_drv = dexp.bind_drivers(row_dict)
            appetite_drv = dexp.appetite_drivers(row_dict)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            def _section_label(text):
                st.markdown(
                    f'<div style="color:{NAVY}; font-size:0.92rem; font-weight:800; '
                    f'letter-spacing:0.04em; margin-bottom:10px;">{text}</div>', unsafe_allow_html=True
                )

            def _sub_label(text):
                st.markdown(
                    f'<div style="color:{ACCENT}; font-size:0.85rem; font-weight:700; '
                    f'letter-spacing:0.02em; margin-bottom:6px;">{text}</div>', unsafe_allow_html=True
                )

            # ---- Underwriting Result banner ----
            # Full card tinted by the outcome's own tier color (not just the left accent strip)
            # so the result reads at a glance, before a word of text is read - a translucent
            # tint of the tier color over the page's own background, not a flat opaque fill,
            # so it still sits correctly in both light and dark theme.
            tier = result["risk_tier_v2"]
            banner_color = TIER_COLORS[tier]
            banner_bg = _rgba(banner_color, 0.16 if IS_DARK else 0.09)
            banner_border = _rgba(banner_color, 0.5)
            summary_line = build_banner_summary(result, loss_drv, bind_drv)
            st.markdown(f"""
            <div style="background:{banner_bg}; border:1px solid {banner_border}; border-left:6px solid {banner_color};
                        border-radius: 10px; padding: 20px 24px; margin-bottom: 1rem;">
                <div style="color:{TEXT_MUTED}; font-size:0.78rem; font-weight:700; letter-spacing:0.06em;">UNDERWRITING RESULT</div>
                <div style="color:{banner_color}; font-size:1.6rem; font-weight:800; margin: 4px 0 10px 0;">{tier}</div>
                <div style="color:{TEXT_MAIN}; font-size:0.95rem;"><b>Recommended Action:</b> {result['recommended_uw_action_v2']}</div>
                <div style="color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px;">{summary_line}</div>
            </div>
            """, unsafe_allow_html=True)

            # ---- Narrative (deterministic template - see narrative.py) ----
            narrative_text = build_narrative(result, loss_drv, bind_drv, appetite_drv)
            st.markdown(f"""
            <div style="background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:10px;
                        padding:16px 20px; margin-bottom:1rem;">
                <div style="color:{TEXT_MUTED}; font-weight:700; font-size:0.78rem; letter-spacing:0.05em; margin-bottom:8px;">NARRATIVE</div>
                <div style="color:{TEXT_MAIN}; font-size:0.95rem; line-height:1.7;">{narrative_text}</div>
            </div>
            """, unsafe_allow_html=True)

            # ---- Key Factors chips ----
            chip_source = loss_drv[:2] + bind_drv[:2] + appetite_drv[:2]
            seen, chips = set(), []
            for d in chip_source:
                if d["label"] not in seen:
                    chips.append(d["label"]); seen.add(d["label"])
            if chips:
                chip_bg = _rgba(ACCENT, 0.14 if IS_DARK else 0.10)
                chip_border = _rgba(ACCENT, 0.45)
                chip_html = "".join(
                    f'<span style="background:{chip_bg}; border:1px solid {chip_border}; border-radius:20px; '
                    f'padding:6px 16px; margin:3px 6px 3px 0; display:inline-block; font-size:0.85rem; '
                    f'font-weight:600; color:{TEXT_MAIN};">{c}</span>' for c in chips
                )
                with st.container(border=True):
                    _section_label("KEY FACTORS BEHIND THIS DECISION")
                    st.markdown(f'<div style="margin:-2px 0 0 0;">{chip_html}</div>', unsafe_allow_html=True)
                st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)

            # ---- KPI cards ----
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Composite UW Score", f"{result['uw_score_v2']:.1f}/100")
            k2.metric("Loss Propensity Score", f"{result['loss_propensity_score_v2']:.1f}")
            k3.metric("Bind Propensity Score", f"{result['bind_propensity_score_v2']:.1f}")
            k4.metric("Appetite Score", f"{result['appetite_capacity_score_v2']:.1f}")

            p1, p2, p3 = st.columns(3)
            p1.metric("Technical Premium", f"${result['indicated_annual_premium_usd']:,.0f}")
            quoted = result["quoted_premium_usd"]
            p2.metric("Quoted Premium", f"${quoted:,.0f}" if quoted == quoted else "Not Quoted")
            p3.metric("Probability of Large Loss", f"{result['p_large_loss']*100:.1f}%")

            # ---- Why This Score - live driver bar charts, one consistent card ----
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            def _driver_bar(drivers, good_direction):
                if not drivers:
                    return None
                labels = [d["label"] for d in drivers][::-1]
                values = [d["weight"] if d["direction"] != good_direction else -d["weight"] for d in drivers][::-1]
                colors = [BAD if v > 0 else GOOD for v in values]
                fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors))
                fig.update_layout(
                    height=max(200, 44 * len(drivers)), margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(showticklabels=False, zeroline=True, zerolinecolor=CARD_BORDER, zerolinewidth=2),
                    yaxis=dict(tickfont=dict(size=12)), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=TEXT_MAIN),
                )
                return fig

            # st.container(border=True) - the correct, native way to wrap real Streamlit
            # widgets (columns, Plotly charts) in one visually consistent card. An earlier
            # version of this tried to open an HTML <div> in one st.markdown() call and close
            # it in a later one, spanning the widgets in between - that doesn't actually nest
            # in the page's DOM, since each st.markdown()/element call renders as its own
            # isolated fragment, not one continuous HTML document.
            with st.container(border=True):
                _section_label("WHY THIS SCORE")
                st.caption("Top factors behind each score, measured against the real book's own typical "
                           "submission. Red bars push the outcome the wrong way; green bars help it.")
                d1, d2, d3 = st.columns(3)
                with d1:
                    _sub_label("Loss Risk Drivers")
                    fig = _driver_bar(loss_drv, good_direction="decreases")
                    if fig:
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info("No material drivers vs. the book baseline.")
                with d2:
                    _sub_label("Bind Propensity Drivers")
                    fig = _driver_bar(bind_drv, good_direction="increases")
                    if fig:
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info("No material drivers vs. the book baseline.")
                with d3:
                    _sub_label("Appetite / Eligibility Drivers")
                    fig = _driver_bar(appetite_drv, good_direction="increases")
                    if fig:
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info("No eligibility penalties. Clean appetite profile.")

            # ---- Technical & audit detail ----
            with st.expander("Technical & audit detail"):
                st.caption("Every figure below is deterministic and reproducible. The same inputs always "
                           "produce the same output. The narrative above only explains these numbers in "
                           "plain English; it never alters them.")
                tab_calc, tab_reason = st.tabs(["Full calculation", "Reason codes"])

                with tab_calc:
                    loading_pct = result["premium_loading_pct"]
                    calc_rows = [
                        ("Hard Stop", bool(result["appetite_capacity_score_v2"] == 0)),
                        ("Line of Business", result["line_of_business"]),
                        ("Predicted Claim Frequency", round(result["predicted_frequency"], 4)),
                        ("Predicted Attritional Severity ($)", round(result["predicted_attritional_severity"], 0)),
                        ("Probability of Large Loss", round(result["p_large_loss"], 4)),
                        ("Expected Large-Loss Severity ($)", round(result["expected_large_severity"], 0)),
                        ("Total Expected Loss Cost ($)", round(result["total_expected_loss_cost"], 0)),
                        ("Technical Premium ($)", round(result["indicated_annual_premium_usd"], 0)),
                        ("Premium Loading", f"{loading_pct*100:.0f}%" if loading_pct == loading_pct else "N/A - not quoted"),
                        ("Quoted Premium ($)", round(quoted, 0) if quoted == quoted else "Not Quoted"),
                        ("Predicted Loss Ratio (%)", round(result["predicted_loss_ratio_pct"], 1)),
                        ("Loss Propensity Score", round(result["loss_propensity_score_v2"], 1)),
                        ("Bind Propensity Score", round(result["bind_propensity_score_v2"], 1)),
                        ("Appetite/Capacity Score", round(result["appetite_capacity_score_v2"], 1)),
                        ("Composite UW Score", round(result["uw_score_v2"], 1)),
                        ("Risk Tier", result["risk_tier_v2"]),
                        ("Appetite Decision", result["appetite_capacity_decision_v2"]),
                        ("Recommended Action", result["recommended_uw_action_v2"]),
                    ]
                    st.dataframe(pd.DataFrame(calc_rows, columns=["Field", "Value"]), width="stretch", hide_index=True)

                with tab_reason:
                    reason_rows = (
                        [{"Category": "Loss", "Driver": d["label"], "Direction": d["direction"], "Relative Weight": d["weight"]} for d in loss_drv]
                        + [{"Category": "Bind", "Driver": d["label"], "Direction": d["direction"], "Relative Weight": d["weight"]} for d in bind_drv]
                        + [{"Category": "Appetite", "Driver": d["label"], "Direction": d["direction"], "Relative Weight": d["weight"]} for d in appetite_drv]
                    )
                    if reason_rows:
                        st.dataframe(
                            pd.DataFrame(reason_rows), width="stretch", hide_index=True,
                            column_config={"Relative Weight": st.column_config.NumberColumn("Relative Weight", format="%.1f")},
                        )
                    else:
                        st.info("No material drivers to rank for this submission.")
