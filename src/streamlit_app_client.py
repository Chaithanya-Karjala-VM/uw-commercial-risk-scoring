"""
Assisted Underwriting - CLIENT-VIEW build. Entirely separate from streamlit_app.py - the
existing app is never imported, touched, or affected by anything in this file.

Scope, corrected 2026-10 after direct feedback: this is a VISUAL restyle of the existing
What-If Simulator only - same inputs, same Calculate-button flow, same validation, same
scores, same drivers, same narrative, same audit detail. Nothing here is new functionality.
An earlier draft added baseline-vs-scenario comparison, a waterfall chart, a "why not
Preferred" gap analysis, live auto-recalculate, submission lookup, and presets - none of
that exists in the real app, so none of it belongs here either. Removed entirely. The only
things that differ from the existing app are layout, color, and typography, modeled on the
manager-provided wireframe's visual language (used for STRUCTURE reference only - no number
from the wireframe's own mockup is used anywhere; everything shown is computed live from the
real pipeline).

Run on a separate port so both apps can run side by side, existing app untouched:
    streamlit run streamlit_app_client.py --server.port 8502
"""
from pathlib import Path
import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_data
from scoring import load_models, score, estimate_financial_stability_score, FINANCIAL_STABILITY_FIT_NOTE
from lob_config import (PROPERTY_LOBS, LIABILITY_LOBS, ALL_LOBS, group_for_lob, STATE_TO_REGION,
                         CURRENT_YEAR, PRIMARY_CAT_PERIL_OPTIONS, region_for_state)
from bind_propensity import NEW_BROKER_DEFAULT_TIER, KNOWN_BROKER_TIER, tier_for_known_broker
from decision_matrix_reference import RISK_TIER_ORDER, RISK_TIER_INFO, tier_for_score
import driver_explain as dexp
from narrative import build_narrative, build_banner_summary
import feature_map as fmap

st.set_page_config(page_title="Assisted Underwriting - SL", layout="wide")

IS_DARK = st.context.theme.type == "dark"

# ---------------------------------------------------------------------------
# Palette. Triage semantics (red=Decline, amber=High, blue=Standard, green=Preferred) kept
# identical to the existing app on purpose - two apps showing the same decisions should never
# disagree on what a color means.
# ---------------------------------------------------------------------------
NAVY = "#6FA3F2" if IS_DARK else "#1B2A4A"
ACCENT = "#7FB2FF" if IS_DARK else "#3B6EC7"
GOOD = "#3FC97F" if IS_DARK else "#1E8449"
WARN = "#E0B23D" if IS_DARK else "#B7950B"
BAD = "#E5685A" if IS_DARK else "#C0392B"
CARD_BG = "#16213A" if IS_DARK else "#FFFFFF"
CARD_BORDER = "#2A3B57" if IS_DARK else "#E2E6EE"
PAGE_BG = "#0D1420" if IS_DARK else "#F6F8FC"
TEXT_MAIN = "#E9EEF6" if IS_DARK else "#17213A"
TEXT_MUTED = "#9FB0C7" if IS_DARK else "#5B6B85"
TIER_COLORS = {"Preferred": GOOD, "Standard": ACCENT, "High": WARN, "Decline": BAD}


def _rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family:'Inter',-apple-system,sans-serif; font-size:15.5px; }}
    .block-container {{ padding-top:0.8rem; padding-bottom:2rem; max-width:1560px; }}
    [data-testid="stAppViewContainer"] {{ background:{PAGE_BG}; }}
    h1 {{ color:{NAVY}; font-weight:800; font-size:1.85rem; }}
    div[data-baseweb="tab-list"] {{ gap:10px; border-bottom:2px solid {CARD_BORDER}; }}
    button[data-baseweb="tab"] {{
        font-size:1.3rem; font-weight:800; height:54px; color:{TEXT_MUTED};
        border-radius:10px 10px 0 0; padding:0 20px; transition: background 0.2s ease, color 0.2s ease;
    }}
    button[data-baseweb="tab"]:hover {{ color:{NAVY}; background:{_rgba(NAVY, 0.06)}; }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color:{NAVY}; border-bottom:4px solid {NAVY}; background:{_rgba(NAVY, 0.10)};
        animation: tabSelectPop 0.45s ease;
    }}
    @keyframes tabSelectPop {{
        0% {{ transform: scale(0.93); box-shadow: 0 0 0 0 {_rgba(NAVY, 0.45)}; }}
        60% {{ transform: scale(1.03); box-shadow: 0 0 0 9px {_rgba(NAVY, 0)}; }}
        100% {{ transform: scale(1); box-shadow: 0 0 0 0 {_rgba(NAVY, 0)}; }}
    }}
    [data-testid="stMetric"] {{ background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:10px; padding:12px 14px 8px; }}
    [data-testid="stMetricValue"] {{ color:{NAVY}; font-weight:800; font-size:1.7rem; }}
    [data-testid="stExpander"] {{ border:1px solid {CARD_BORDER}; border-radius:10px; background:{CARD_BG}; margin-bottom:0.6rem; }}
    [data-testid="stExpander"] summary {{ font-weight:700; color:{NAVY}; }}
    .cv-card {{ background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:12px; padding:16px 20px; margin-bottom:12px; }}
    .cv-eyebrow {{ color:{TEXT_MUTED}; font-size:0.74rem; font-weight:700; letter-spacing:0.07em; text-transform:uppercase; margin-bottom:8px; }}
    .cv-chip {{ display:inline-block; background:{_rgba(ACCENT,0.12)}; border:1px solid {_rgba(ACCENT,0.4)}; border-radius:16px;
                padding:3px 11px; margin:2px 4px 2px 0; font-size:0.8rem; color:{TEXT_MAIN}; }}
    .cv-sub {{ color:{ACCENT}; font-weight:700; font-size:0.86rem; margin-bottom:4px; }}
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
    return score(get_raw_book(), get_models())


def style_risk_tier_column(df, tier_col="Risk Tier"):
    """Colors the Risk Tier column's cell background to match the tier - identical logic to
    the existing app, same palette, so a tier means the same color in both apps."""
    def _color(val):
        color = TIER_COLORS.get(val)
        if color is None:
            return ""
        return f"background-color: {color}; color: white; font-weight: 600;"
    return df.style.map(_color, subset=[tier_col])


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
LOB_OPTIONS = sorted(ALL_LOBS)  # alphabetical, matching the existing app's own LOB_OPTIONS convention -
# ALL_LOBS's own declared order isn't alphabetical, which silently made the two apps'
# default-selected LOB differ (found during a full old-vs-new parity sweep).


def _reset_tier_to_segment_default():
    st.session_state["cv_broker_tier"] = NEW_BROKER_DEFAULT_TIER[st.session_state["cv_broker_segment"]]


def _reset_tier_to_known_broker():
    broker_id = st.session_state["cv_broker_id"]
    real_tier = tier_for_known_broker(broker_id)
    if real_tier:
        st.session_state["cv_broker_tier"] = real_tier


def _reset_tier_on_toggle():
    if st.session_state["cv_is_existing_broker"] == "Yes - select broker":
        broker_id = st.session_state.get("cv_broker_id", sorted(KNOWN_BROKER_TIER.keys())[0])
        real_tier = tier_for_known_broker(broker_id)
        if real_tier:
            st.session_state["cv_broker_tier"] = real_tier
    else:
        segment = st.session_state.get("cv_broker_segment", "Large National")
        st.session_state["cv_broker_tier"] = NEW_BROKER_DEFAULT_TIER[segment]


# ---------------------------------------------------------------------------
# Top header - real logo, same file the existing app uses
# ---------------------------------------------------------------------------
LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "vm_logo.png"
logo_col, title_col = st.columns([1, 6], vertical_alignment="center")
with logo_col:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=150)
with title_col:
    st.markdown(f"<h1 style='margin-bottom:0;'>Assisted Underwriting - SL</h1>", unsafe_allow_html=True)
    st.caption("Every submission scored, triaged, and explained")

tab_portfolio, tab_dashboard, tab_whatif, tab_feature = st.tabs(
    ["Portfolio", "Dashboard", "What-If Simulator", "Feature Map"]
)

def _section_label(text):
    st.markdown(f'<div class="cv-eyebrow" style="margin-top:0.8rem;">{text}</div>', unsafe_allow_html=True)


def _kpi_card(label, value, subtext=None, help_text=None):
    """A KPI card matching st.metric's own visual style (same CARD_BG/CARD_BORDER/NAVY value
    color), but as one HTML block so a sub-line can sit INSIDE the card, under the number,
    with real contrast (TEXT_MAIN, not a washed-out muted gray) - native st.metric plus a
    separate st.caption renders the caption as a disconnected, low-contrast line below the
    card, not inside it."""
    title_attr = f' title="{help_text}"' if help_text else ""
    sub_html = (f'<div style="color:{TEXT_MAIN}; font-size:0.84rem; margin-top:6px; font-weight:600; line-height:1.3;">{subtext}</div>'
                if subtext else "")
    # min-height (not a fixed height) - a fixed height was too short to fit all 3 lines
    # (label + value + subtext), so the subtext visually overflowed below the card's own
    # border instead of being contained inside it. min-height guarantees the border always
    # grows to fit whatever's inside, never clips or overflows.
    return (f'<div{title_attr} style="background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:10px; '
            f'padding:12px 14px 13px; min-height:112px; box-sizing:border-box;">'
            f'<div style="color:{TEXT_MUTED}; font-size:0.9rem; line-height:1.3;">{label}</div>'
            f'<div style="color:{NAVY}; font-weight:800; font-size:1.7rem; margin-top:3px; line-height:1.3;">{value}</div>'
            f'{sub_html}</div>')


def _ring_gauge_cell(value, label, sub, color, size=76):
    """A small circular progress ring built from plain SVG (not a Plotly chart) - compact,
    sits inline in a single flex row next to other elements. Plotly's go.Indicator gauges
    auto-expand to fill their container (that's what made the first attempt at this row
    render as 3 huge full-width circles stacked above the contribution bar, instead of one
    compact row matching the wireframe) - raw SVG gives full control over exact pixel size."""
    stroke = 8
    r = 50 - stroke / 2
    circumference = 2 * math.pi * r
    pct = max(0.0, min(100.0, value)) / 100.0
    offset = circumference * (1 - pct)
    ring_svg = (
        f'<svg width="{size}" height="{size}" viewBox="0 0 100 100" style="transform:rotate(-90deg);">'
        f'<circle cx="50" cy="50" r="{r}" fill="none" stroke="{CARD_BORDER}" stroke-width="{stroke}"/>'
        f'<circle cx="50" cy="50" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}"/>'
        f'</svg>'
    )
    return (
        f'<div style="display:flex; flex-direction:column; align-items:center; min-width:{size + 14}px;">'
        f'<div style="position:relative; width:{size}px; height:{size}px;">{ring_svg}'
        f'<div style="position:absolute; inset:0; display:flex; align-items:center; justify-content:center; '
        f'font-size:1rem; font-weight:800; color:{TEXT_MAIN};">{value:.1f}</div></div>'
        f'<div style="font-size:0.76rem; font-weight:700; color:{TEXT_MAIN}; margin-top:6px; text-align:center; '
        f'white-space:nowrap;">{label}</div>'
        f'<div style="font-size:0.64rem; color:{TEXT_MUTED}; text-align:center; white-space:nowrap;">{sub}</div></div>'
    )


# ===========================================================================
# PORTFOLIO OVERVIEW - identical data/filters/chart/table to the existing app's Portfolio
# Overview tab, restyled only.
# ===========================================================================
TEAL = "#3FBFB0" if IS_DARK else "#0E8A7D"
VIOLET = "#B98EE8" if IS_DARK else "#7B4FA6"


def _flow_box(label, value, color):
    return (f'<div style="background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:10px; '
            f'padding:8px 14px; text-align:center; min-width:92px;">'
            f'<div style="color:{TEXT_MUTED}; font-size:0.64rem; font-weight:700; text-transform:uppercase; '
            f'letter-spacing:0.04em;">{label}</div>'
            f'<div style="color:{color}; font-size:1.1rem; font-weight:800; margin-top:2px;">{value}</div></div>')


_ARROW = f'<div style="color:{TEXT_MUTED}; font-size:1.2rem; padding:0 2px;">&rarr;</div>'

def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _interp_color(c1, c2, t):
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return f"#{int(r1+(r2-r1)*t):02x}{int(g1+(g2-g1)*t):02x}{int(b1+(b2-b1)*t):02x}"


def _grade_bg(val, vmin, vmax, higher_is_better=True):
    """Cell background color graded red->amber->blue->green by where val sits in [vmin,vmax] -
    hand-interpolated using the exact same BAD/WARN/ACCENT/GOOD colors as the Triage bar and
    flow box elsewhere on this page (full strength, not softened), so the grid reads as the
    same color language as the rest of the page instead of its own washed-out scale."""
    if pd.isna(val) or vmax == vmin:
        return ""
    t = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    if not higher_is_better:
        t = 1 - t
    if t < 1 / 3:
        bg = _interp_color(BAD, WARN, t * 3)
    elif t < 2 / 3:
        bg = _interp_color(WARN, ACCENT, (t - 1 / 3) * 3)
    else:
        bg = _interp_color(ACCENT, GOOD, (t - 2 / 3) * 3)
    return f"background-color:{bg}; color:#FFFFFF; font-weight:700;"


# metric key -> (source column, aggfunc, display format, higher-is-better, column_config printf format)
_GRID_METRICS = {
    "UW Score": ("uw_score_v2", "mean", "{:.1f}", True, "%.1f"),
    "Loss Propensity": ("loss_propensity_score_v2", "mean", "{:.1f}", False, "%.1f"),
    "Appetite": ("appetite_capacity_score_v2", "mean", "{:.1f}", True, "%.1f"),
    "Bind": ("bind_propensity_score_v2", "mean", "{:.1f}", True, "%.1f"),
    "Referral %": ("_is_referral", "mean", "{:.0f}%", False, "%.0f%%"),
}


def _on_grid_select():
    """Clicking a cell in the LOB x Region grid (region 5) sets the Deep-Dive filters
    (region 6) below - one directional, click-to-populate. Clicking the same cell again
    (deselecting) or clicking nothing clears back to All/All, same as the untouched default."""
    sel = st.session_state.get("po_grid_select")
    cells = sel["selection"]["cells"] if sel and sel.get("selection") else []
    if cells:
        row_idx, col_name = cells[0]
        lob_names = st.session_state.get("po_grid_pivot_index", [])
        if 0 <= row_idx < len(lob_names):
            st.session_state["dd_lob_select"] = lob_names[row_idx]
            st.session_state["dd_region_select"] = col_name
    else:
        st.session_state["dd_lob_select"] = "All"
        st.session_state["dd_region_select"] = "All"


def _on_dashboard_inbox_select():
    """Clicking a row in the Dashboard inbox table sets which submission the rest of the
    page (built in a later step) will detail. Clicking nothing/clearing leaves db_selected_id
    unset, which the rest of the page reads as 'show the aggregate view for the whole
    filtered population' - no row selected is a real, valid state, not an error."""
    sel = st.session_state.get("db_inbox_select")
    rows = sel["selection"]["rows"] if sel and sel.get("selection") else []
    ids = st.session_state.get("db_inbox_ids", [])
    if rows and rows[0] < len(ids):
        st.session_state["db_selected_id"] = ids[rows[0]]
    else:
        st.session_state["db_selected_id"] = None


with tab_portfolio:
    book = get_scored_book()
    region_options = sorted(book["risk_location_region"].dropna().unique().tolist())

    # ---- top-level filters - narrow everything below: the flow summary, KPIs, histogram,
    # scatter, and the LOB x Region grid. ----
    c1, c2 = st.columns(2)
    lob_filter = c1.selectbox("Line of Business", ["All"] + LOB_OPTIONS, key="po_lob")
    region_filter = c2.selectbox("Region", ["All"] + region_options, key="po_region")

    filtered = book.copy()
    if lob_filter != "All":
        filtered = filtered[filtered["line_of_business"] == lob_filter]
    if region_filter != "All":
        filtered = filtered[filtered["risk_location_region"] == region_filter]
    n = len(filtered)

    # ---- ① static explainer - how the pipeline works, conceptually. No live numbers here
    # at all, by design: this box documents the SHAPE of the system, not a live data view
    # (that's everything below it). Uses the real 50/30/20 weighting, not a placeholder. Sized
    # generously and each stage boxed consistently so the step-by-step flow (and the arrows
    # joining each step) reads clearly at a glance, not cramped into one dense row. ----
    _section_label("How Every Submission Is Assessed")
    BIG_ARROW = f'<div style="color:{NAVY}; font-size:1.6rem; font-weight:300; padding:0 4px; align-self:center;">&rarr;</div>'
    # Every outer box uses the exact same border treatment (CARD_BORDER, no special-case
    # navy emphasis on any one box) so the row reads as one consistent box "type" throughout.
    BOX = (f"height:178px; box-sizing:border-box; border-radius:12px; display:flex; flex-direction:column; "
           f"justify-content:center; background:{CARD_BG}; border:1px solid {CARD_BORDER};")
    MINI_BOX = f"background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:8px; padding:6px 10px;"
    st.markdown(f"""
    <div class="cv-card" style="display:flex; align-items:stretch; flex-wrap:nowrap; gap:10px; padding:24px 28px; overflow-x:auto;">
        <div style="{BOX} padding:14px 20px; min-width:160px; align-items:center; text-align:center;">
            <div style="color:{NAVY}; font-weight:800; font-size:0.9rem; letter-spacing:0.02em;">SUBMISSION<br>RECEIVED</div>
        </div>
        {BIG_ARROW}
        <div style="{BOX} padding:12px 16px; gap:6px; min-width:300px;">
            <div style="{MINI_BOX}"><b style="color:{BAD}; font-size:0.8rem;">LOSS PROPENSITY</b><br><span style="color:{TEXT_MUTED}; font-size:0.72rem;">expected loss — 0 best / 100 worst</span></div>
            <div style="{MINI_BOX}"><b style="color:{GOOD}; font-size:0.8rem;">APPETITE SCORE</b><br><span style="color:{TEXT_MUTED}; font-size:0.72rem;">fit with appetite &amp; capacity</span></div>
            <div style="{MINI_BOX}"><b style="color:{ACCENT}; font-size:0.8rem;">BIND PROPENSITY</b><br><span style="color:{TEXT_MUTED}; font-size:0.72rem;">likelihood we win it</span></div>
        </div>
        {BIG_ARROW}
        <div style="{BOX} padding:14px 20px; min-width:190px; align-items:center; text-align:center;">
            <div style="color:{NAVY}; font-weight:800; font-size:0.9rem;">UW SCORE</div>
            <div style="color:{TEXT_MUTED}; font-size:0.74rem; margin-top:6px; line-height:1.5;">0–100, weighted blend<br>50% Loss / 30% Appetite / 20% Bind</div>
        </div>
        {BIG_ARROW}
        <div style="{BOX} padding:14px 20px; min-width:280px;">
            <div style="color:{NAVY}; font-weight:800; font-size:0.84rem; margin-bottom:8px; text-align:center;">TRIAGE</div>
            <div style="display:flex; border-radius:6px; overflow:hidden;">
                <div style="flex:1; background:{GOOD}; color:white; text-align:center; font-size:0.68rem; font-weight:700; padding:8px 2px;">Preferred</div>
                <div style="flex:1; background:{ACCENT}; color:white; text-align:center; font-size:0.68rem; font-weight:700; padding:8px 2px;">Standard</div>
                <div style="flex:1; background:{WARN}; color:white; text-align:center; font-size:0.68rem; font-weight:700; padding:8px 2px;">High</div>
                <div style="flex:1; background:{BAD}; color:white; text-align:center; font-size:0.68rem; font-weight:700; padding:8px 2px;">Decline</div>
            </div>
        </div>
        {BIG_ARROW}
        <div style="{BOX} padding:14px 20px; min-width:170px; align-items:center; text-align:center;">
            <div style="color:{NAVY}; font-weight:800; font-size:0.84rem;">DRIVERS</div>
            <div style="color:{TEXT_MUTED}; font-size:0.74rem; margin-top:6px;">Top reasons behind each score</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- ② 5 real KPIs, all computed from the top-filtered slice. Actual Loss Ratio is a
    # genuine historical figure (real incurred claims / real premium), not a prediction. ----
    tier_pct = ((filtered["risk_tier_v2"].value_counts(normalize=True) * 100).reindex(RISK_TIER_ORDER).fillna(0)
                if n else pd.Series(0.0, index=RISK_TIER_ORDER))
    prem_sum = filtered["indicated_annual_premium_usd"].sum() if n else 0
    m1, m2, m3, m4, m5 = st.columns(5)

    m1.markdown(_kpi_card("Submissions Received", f"{n:,}", f"${prem_sum/1e6:.1f}M total premium"), unsafe_allow_html=True)

    straight_through = tier_pct["Preferred"] + tier_pct["Standard"]
    m2.markdown(_kpi_card("Straight-Through %", f"{straight_through:.0f}%", help_text=
                "Preferred + Standard share - the portion that clears without a referral to an underwriter."), unsafe_allow_html=True)

    preferred_prem = filtered.loc[filtered["risk_tier_v2"] == "Preferred", "indicated_annual_premium_usd"].sum() if n else 0
    m3.markdown(_kpi_card("Preferred Submission Premium", f"${preferred_prem/1e6:.1f}M",
                           help_text="Premium from Preferred-tier submissions only."), unsafe_allow_html=True)

    actual_lr = (filtered["total_incurred_claim_amount_usd"].sum() / prem_sum * 100) if prem_sum > 0 else 0
    m4.markdown(_kpi_card("Actual Loss Ratio", f"{actual_lr:.1f}%", help_text=
                "Real historical incurred claims divided by real premium - an actual figure from the book, not a model prediction."), unsafe_allow_html=True)

    referred = int((filtered["risk_tier_v2"].isin(["High", "Decline"])).sum()) if n else 0
    referred_pct = (referred / n * 100) if n else 0
    m5.markdown(_kpi_card("Referred to UW", f"{referred:,}", f"{referred_pct:.1f}% of submissions",
                           help_text="High + Decline tier count - these need underwriter review."), unsafe_allow_html=True)

    if n == 0:
        st.info("No submissions match the current filter combination.")
    else:
        # ---- ③ + ④ side by side ----
        hist_col, scat_col = st.columns(2)
        with hist_col:
            _section_label("Triage Band Distribution, by UW Score")
            fig_hist = px.histogram(filtered, x="uw_score_v2", nbins=20,
                                     labels={"uw_score_v2": "UW Score"},
                                     color_discrete_sequence=[NAVY])
            fig_hist.update_traces(hovertemplate="Score band %{x}<br>%{y:,} submissions<extra></extra>", marker_line_width=0)
            # UW Score's real tier cutoffs (same ones used in the Triage bar and the grid) -
            # 0=worst/100=best, so Decline sits at the low end here, Preferred at the high end.
            for lo, hi, color in [(0, 38, BAD), (38, 55, WARN), (55, 72, ACCENT), (72, 100, GOOD)]:
                fig_hist.add_vrect(x0=lo, x1=hi, fillcolor=color, opacity=0.40 if IS_DARK else 0.30, line_width=0, layer="below")
            fig_hist.update_layout(yaxis_title="Submissions", xaxis_title="UW Score (0=worst, 100=best)",
                                    bargap=0.05, height=320, margin=dict(l=10, r=10, t=10, b=10),
                                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_MAIN, size=11))
            st.plotly_chart(fig_hist, width="stretch")

        with scat_col:
            _section_label("Risk–Opportunity Quadrant Map — bubble = premium, color = tier")
            scatter_df = filtered.sample(min(1200, n), random_state=1)
            x_lo, x_hi = 0, 100
            y_lo, y_hi = filtered["bind_propensity_score_v2"].min(), filtered["bind_propensity_score_v2"].max()
            y_pad = max((y_hi - y_lo) * 0.06, 1)
            # Both axes split at their OWN real median of the current slice - consistent
            # method on both, not a fixed number on one axis and a statistic on the other
            # (that inconsistency was a real issue in an earlier draft of this chart).
            mid_x, mid_y = filtered["loss_propensity_score_v2"].median(), filtered["bind_propensity_score_v2"].median()

            fig_sc = px.scatter(
                scatter_df, x="loss_propensity_score_v2", y="bind_propensity_score_v2",
                size="indicated_annual_premium_usd", color="risk_tier_v2", color_discrete_map=TIER_COLORS,
                category_orders={"risk_tier_v2": RISK_TIER_ORDER},
                size_max=20, opacity=0.78,
                hover_data={"submission_id": True, "line_of_business": True, "indicated_annual_premium_usd": ":$,.0f",
                            "loss_propensity_score_v2": ":.1f", "bind_propensity_score_v2": ":.1f"},
                labels={"loss_propensity_score_v2": "Loss Propensity (worse →)", "bind_propensity_score_v2": "Bind Propensity (more likely →)",
                        "risk_tier_v2": "Tier", "submission_id": "Submission ID", "line_of_business": "Line of Business",
                        "indicated_annual_premium_usd": "Premium"},
            )
            # Shaded quadrant backgrounds, drawn behind the points, in the same semantic
            # colors as the quadrant labels - so color and label reinforce each other instead
            # of the quadrant meaning being guessable only from small corner text.
            for (qx0, qx1, qy0, qy1, qcolor) in [
                (x_lo, mid_x, mid_y, y_hi + y_pad, GOOD),        # WIN: low loss, high bind
                (mid_x, x_hi, mid_y, y_hi + y_pad, WARN),        # PRICE UP: high loss, high bind
                (x_lo, mid_x, y_lo - y_pad, mid_y, ACCENT),      # NURTURE: low loss, low bind
                (mid_x, x_hi, y_lo - y_pad, mid_y, BAD),         # AVOID: high loss, low bind
            ]:
                fig_sc.add_shape(type="rect", xref="x", yref="y", x0=qx0, x1=qx1, y0=qy0, y1=qy1,
                                  fillcolor=_rgba(qcolor, 0.16 if IS_DARK else 0.10), line_width=0, layer="below")
            fig_sc.add_vline(x=mid_x, line_dash="dot", line_color=CARD_BORDER)
            fig_sc.add_hline(y=mid_y, line_dash="dot", line_color=CARD_BORDER)

            label_x_left, label_x_right = x_lo + (x_hi - x_lo) * 0.03, x_hi - (x_hi - x_lo) * 0.03
            label_y_top, label_y_bot = y_hi + y_pad * 0.55, y_lo - y_pad * 0.55
            for text, x, y, anchor in [
                ("WIN — Preferred", label_x_left, label_y_top, "left"),
                ("PRICE UP — refer", label_x_right, label_y_top, "right"),
                ("NURTURE — Broker Push Standard", label_x_left, label_y_bot, "left"),
                ("AVOID — decline", label_x_right, label_y_bot, "right"),
            ]:
                fig_sc.add_annotation(x=x, y=y, text=f"<b>{text}</b>", showarrow=False, align=anchor, xanchor=anchor,
                                       font=dict(size=10, color=TEXT_MAIN),
                                       bgcolor=_rgba(CARD_BG, 0.85), borderpad=3)
            fig_sc.update_xaxes(range=[x_lo, x_hi])
            fig_sc.update_yaxes(range=[y_lo - y_pad, y_hi + y_pad])
            fig_sc.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), legend_title="Tier",
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=9)),
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_MAIN, size=11))
            st.plotly_chart(fig_sc, width="stretch")

        # ---- ⑤ LOB x Region grid - real numbers visible in every cell, color-graded by
        # value, metric switchable via the toggle, every row/cell honestly computed from the
        # top-filtered slice above. ----
        _section_label("Where to Look — by Line of Business x Region")
        metric_choice = st.radio(
            "Metric", list(_GRID_METRICS.keys()), horizontal=True, key="po_grid_metric", label_visibility="collapsed")
        col, agg, fmt, higher_better, cc_fmt = _GRID_METRICS[metric_choice]

        grid_src = filtered.copy()
        grid_src["_is_referral"] = grid_src["risk_tier_v2"].isin(["High", "Decline"]).astype(float) * 100
        pivot = grid_src.pivot_table(index="line_of_business", columns="risk_location_region", values=col, aggfunc=agg)

        if pivot.empty:
            st.info("No data for this combination.")
        else:
            vmin, vmax = pivot.values[~pd.isna(pivot.values)].min(), pivot.values[~pd.isna(pivot.values)].max()
            styled = (pivot.style
                      .map(lambda v: _grade_bg(v, vmin, vmax, higher_better))
                      .format(lambda v: fmt.format(v) if pd.notna(v) else "—"))

            cap_col, clear_col = st.columns([5, 1])
            cap_col.caption("Click a cell to select that Line of Business x Region combination - it fills in the Segment Deep-Dive filters at the bottom of the page.")
            # Re-clicking an already-selected cell in Streamlit's grid doesn't reliably clear
            # the selection (it behaves like a plain text-select click instead) - so clearing
            # gets its own explicit button rather than relying on that re-click gesture.
            if clear_col.button("Clear selection", key="po_clear_grid_btn", width="stretch"):
                st.session_state["po_grid_select"] = {"selection": {"rows": [], "columns": [], "cells": []}}
                st.session_state["dd_lob_select"] = "All"
                st.session_state["dd_region_select"] = "All"

            st.session_state["po_grid_pivot_index"] = list(pivot.index)
            st.dataframe(
                styled, width="stretch", height=min(260, 44 * len(pivot) + 44),
                on_select=_on_grid_select, selection_mode="single-cell", key="po_grid_select",
            )

    # ---- ⑥ Segment Deep-Dive - moved to the bottom, full width. No standalone filters here
    # by design: this section is purely click-driven from the grid above (_on_grid_select
    # sets dd_lob_select/dd_region_select in session_state) and the grid's Clear selection
    # button resets it back to the whole book - no independent filter widgets to keep in sync. ----
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    _section_label("Segment Deep-Dive")
    dd_lob = st.session_state.get("dd_lob_select", "All")
    dd_region = st.session_state.get("dd_region_select", "All")

    seg = book.copy()
    if dd_lob != "All":
        seg = seg[seg["line_of_business"] == dd_lob]
    if dd_region != "All":
        seg = seg[seg["risk_location_region"] == dd_region]
    seg_label = " › ".join([x for x in [dd_lob if dd_lob != "All" else None, dd_region if dd_region != "All" else None] if x]) or "All Lines, All Regions"
    st.caption(f"Portfolio › {seg_label} — click a grid cell above to filter, or Clear selection to reset. "
               f"{len(seg):,} submissions, ${seg['indicated_annual_premium_usd'].sum()/1e6:.1f}M premium")

    if len(seg) == 0:
        st.info("No submissions in this segment.")
    else:
        dd_col1, dd_col2 = st.columns([1.4, 1])
        with dd_col1:
            sg1, sg2, sg3, sg4 = st.columns(4)
            sg1.metric("UW Score", f"{seg['uw_score_v2'].mean():.1f}")
            sg2.metric("Loss Prop.", f"{seg['loss_propensity_score_v2'].mean():.1f}")
            sg3.metric("Bind Prop.", f"{seg['bind_propensity_score_v2'].mean():.1f}")
            sg4.metric("Appetite", f"{seg['appetite_capacity_score_v2'].mean():.1f}")

            seg_tier_pct = (seg["risk_tier_v2"].value_counts(normalize=True) * 100).reindex(RISK_TIER_ORDER).fillna(0)
            bar_html = "".join(
                f'<div style="flex:{max(seg_tier_pct[t],0.5)}; background:{TIER_COLORS[t]}; height:22px; '
                f'display:flex; align-items:center; justify-content:center; color:white; font-size:0.68rem; font-weight:700;">'
                f'{seg_tier_pct[t]:.0f}%</div>'
                for t in RISK_TIER_ORDER
            )
            st.markdown(f'<div style="display:flex; border-radius:6px; overflow:hidden; margin:10px 0 2px;">{bar_html}</div>', unsafe_allow_html=True)
            st.caption(" / ".join(RISK_TIER_ORDER))

            st.markdown('<div class="cv-sub" style="margin-top:0.8rem;">Vs. Book Average — What Drives the Score Here</div>', unsafe_allow_html=True)
            for label, colname in [("Loss Propensity", "loss_propensity_score_v2"), ("Bind Propensity", "bind_propensity_score_v2"),
                                    ("Appetite/Capacity", "appetite_capacity_score_v2")]:
                seg_avg, book_avg = seg[colname].mean(), book[colname].mean()
                delta = seg_avg - book_avg
                favorable = (delta < 0) if colname == "loss_propensity_score_v2" else (delta > 0)
                dcolor = GOOD if favorable else BAD
                bar_w = min(abs(delta) * 4, 100)
                st.markdown(
                    f'<div style="display:flex; align-items:center; gap:8px; font-size:0.82rem; padding:3px 0;">'
                    f'<span style="color:{TEXT_MUTED}; width:140px;">{label}</span>'
                    f'<div style="flex:1; background:{CARD_BORDER}; border-radius:3px; height:10px; position:relative;">'
                    f'<div style="width:{bar_w}%; background:{dcolor}; height:10px; border-radius:3px;"></div></div>'
                    f'<span style="color:{dcolor}; font-weight:700; width:50px; text-align:right;">{delta:+.1f}</span></div>',
                    unsafe_allow_html=True)

        with dd_col2:
            st.markdown(f'<div class="cv-sub">Submission Detail — {len(seg):,} Record(s) in This Segment</div>', unsafe_allow_html=True)
            seg_table = seg.sort_values("submission_id")[["submission_id", "risk_tier_v2", "uw_score_v2", "indicated_annual_premium_usd"]]
            st.dataframe(
                style_risk_tier_column(seg_table, "risk_tier_v2"), width="stretch", hide_index=True, height=420,
                column_config={"submission_id": "ID", "risk_tier_v2": "Tier",
                                "uw_score_v2": st.column_config.NumberColumn("UW", format="%.1f"),
                                "indicated_annual_premium_usd": st.column_config.NumberColumn("Premium", format="$%d")},
            )

# ===========================================================================
# EXECUTIVE DASHBOARD - identical data/filters/KPIs/decision matrix/8 charts/broker
# segmentation/claims overview/3 decision-output tables to the existing app, restyled only.
# ===========================================================================
with tab_dashboard:
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

    # =======================================================================
    # Region 1 of the manager's wireframe - a submission inbox. Built full-width (the
    # wireframe itself splits this with a detail panel on the right, but that doesn't exist
    # yet - it's a later step, built once this region is confirmed). Emerging-Risk Watch
    # reuses the exact same real prospective_risk_flag_count_v2 system already live in
    # Dashboard's old Decision Queues and the Portfolio tab's Case Review draft - a
    # submission earns a flag from WATCH_FLAG_DEFINITIONS in scoring.py (low financial
    # stability, high debt-to-equity, declining revenue, short tenure, safety deficiencies,
    # severe CAT exposure, heavy competing-quote pressure), computed fresh from real data,
    # not a stored/frozen number. Status (To review / On watch 90-day / Reviewed) is real,
    # session-only state - there's no persistence layer in this project, so it resets on
    # restart; nothing sets a status yet since the feedback form that would do that (region
    # 5) isn't built yet either - that's expected for this step, not a bug.
    # =======================================================================
    if "db_reviews" not in st.session_state:
        st.session_state["db_reviews"] = {}

    def _db_status_of(sub_id):
        return st.session_state["db_reviews"].get(sub_id, {}).get("status", "To Review")

    inbox_view = st.radio(
        "Inbox view", [f"Submission Inbox ({len(fb):,})", "\U0001F6A9 Emerging-Risk Watch"],
        horizontal=True, key="db_inbox_view", label_visibility="collapsed",
    )
    is_watch_view = inbox_view != f"Submission Inbox ({len(fb):,})"

    st.markdown(
        f'<div style="background:{_rgba(WARN,0.1)}; border:1px solid {_rgba(WARN,0.4)}; border-radius:8px; '
        f'padding:8px 14px; font-size:0.84rem; color:{TEXT_MAIN}; margin:6px 0 10px;">'
        f'<b>Inbox:</b> all submissions matching the filters above, by triage band. '
        f'<b>Watch:</b> model score and real forward-looking signals disagree — UW relooks.</div>',
        unsafe_allow_html=True)

    if is_watch_view:
        flag_thresh = st.selectbox("Flags ≥", [1, 2, 3, 4], index=2, key="db_flag_thresh",
                                    help="A submission earns a flag for each real forward-looking risk signal it carries "
                                         "(low financial stability, high debt-to-equity, declining revenue, short business "
                                         "tenure, safety deficiencies, severe CAT exposure, heavy competing-quote pressure) - "
                                         "the same definition already used in this app's watch-list logic elsewhere, not a new rule.")
        pool = fb[(fb["claim_count_total"] <= 1) & (fb["prospective_risk_flag_count_v2"] >= flag_thresh)]
        st.caption(f"{len(pool):,} of {len(fb):,} filtered submissions carry {flag_thresh}+ real risk flags "
                   f"but 0–1 claims on file — nothing in their claims history would catch this; these flags "
                   f"are the only early warning the model has.")
    else:
        pool = fb

    # ---- Status + the two table-local filters (ID search, Region) share one row to save
    # vertical space. Status is applied first, then Region, then the ID dropdown's own
    # option list is built from whatever's left - so it only ever offers IDs actually
    # reachable under flags/status/Region, and narrows live as those change, instead of
    # always listing the full unfiltered book. ----
    status_col, search_col, region_col = st.columns([1, 2, 1])

    # "On Watch 90d" / "Reviewed" are deliberately not shown yet - nothing sets those
    # statuses until the feedback form (region 5) exists, so they'd always be empty tabs
    # that look like a working feature but silently do nothing. Re-add once region 5 is built.
    status_tab = status_col.radio("Status", ["All", "To Review"], horizontal=True,
                                   key=f"db_status_filter_{is_watch_view}")
    if status_tab != "All":
        keep_mask = pool["submission_id"].apply(lambda s: _db_status_of(s) == status_tab)
        pool = pool[keep_mask]

    region_opts = sorted(fb["risk_location_region"].dropna().unique().tolist())
    region_filter = region_col.selectbox("Region", ["All"] + region_opts, key="db_region_filter")
    if region_filter != "All":
        pool = pool[pool["risk_location_region"] == region_filter]

    id_options = ["All"] + sorted(pool["submission_id"].tolist())
    id_search = search_col.selectbox("Search Submission ID", id_options, key="db_id_search",
                                      help="Type to filter the list, or pick directly from the dropdown.")
    if id_search != "All":
        pool = pool[pool["submission_id"] == id_search]

    show_n = min(len(pool), 300)
    display = pool.head(show_n)[["submission_id", "line_of_business", "risk_location_region",
                                  "risk_tier_v2", "loss_propensity_score_v2", "prospective_risk_flag_count_v2"]].copy()
    display["Status"] = display["submission_id"].apply(_db_status_of)
    display.columns = ["Submission", "LOB", "Region", "Model Tier", "Loss", "Flags", "Status"]
    st.session_state["db_inbox_ids"] = display["Submission"].tolist()

    more_note = f" (showing first {show_n:,})" if len(pool) > show_n else ""
    st.caption(f"{len(pool):,} matching submission(s){more_note}. Click a row to select it for detail below (built next) - "
               f"with nothing selected, the rest of the page will show the aggregate view for all {len(pool):,} filtered here.")

    # Height sized to the actual row count (capped) - a fixed height left a wall of empty
    # rows below the real data whenever a filter narrowed the result down to a handful.
    table_height = min(460, 42 + 35 * max(len(display), 1))
    st.dataframe(
        style_risk_tier_column(display, "Model Tier"), width="stretch", hide_index=True, height=table_height,
        on_select=_on_dashboard_inbox_select, selection_mode="single-row", key="db_inbox_select",
        column_config={"Loss": st.column_config.NumberColumn(format="%.1f")},
    )

    # Selection persists against the full book, not just whatever page of the table is
    # currently shown - changing a table-local filter after selecting a row shouldn't
    # silently drop the selection just because that row scrolled out of the visible page.
    sel_id = st.session_state.get("db_selected_id")
    sel_row = None
    if sel_id:
        sel_match = book[book["submission_id"] == sel_id]
        if len(sel_match):
            sel_row = sel_match.iloc[0]

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if len(pool) == 0:
        st.info("No submissions match the current filters - nothing to show below.")
    else:
        # =======================================================================
        # Regions 2-4 of the wireframe. With a submission selected, these show that one
        # real submission's actual scores/triage/drivers - the exact same driver_explain.py
        # engine the What-If Simulator uses, just pointed at a real book row instead of a
        # typed-in scenario. With nothing selected, they show the real AGGREGATE for
        # whatever's currently filtered in the inbox above (pool) - population means for
        # the scores/triage, and for drivers, the real submission closest to that mean UW
        # score stands in as a representative example (never a synthetic/blended row -
        # every number shown is always a real record's actual data).
        # =======================================================================
        if sel_row is not None:
            is_watch = (sel_row["claim_count_total"] <= 1) and (sel_row["prospective_risk_flag_count_v2"] >= 1)
            watch_badge = ""
            if is_watch:
                watch_badge = (f' <span class="cv-chip" style="background:{_rgba(WARN,0.15)}; '
                                f'border-color:{_rgba(WARN,0.5)}; color:{WARN};">\U0001F6A9 On watch list '
                                f'· {int(sel_row["prospective_risk_flag_count_v2"])} signal(s)</span>')
            st.markdown(f'<div style="color:{TEXT_MAIN}; font-size:0.86rem; font-weight:600;">'
                        f'Dashboard › {"Emerging-Risk Watch" if is_watch else "Submission Inbox"} › {sel_id}</div>',
                        unsafe_allow_html=True)
            st.markdown(f"#### {sel_id} · {sel_row['line_of_business']}{watch_badge}", unsafe_allow_html=True)
            st.markdown(f'<div style="color:{TEXT_MAIN}; font-size:0.9rem; font-weight:600; margin-bottom:6px;">'
                        f'{sel_row["risk_location_region"]} · {sel_row["broker_tier"]} broker · '
                        f'TIV ${sel_row["total_insured_value_usd"]:,.0f} · {int(sel_row["claim_count_total"])} prior claim(s)</div>',
                        unsafe_allow_html=True)

            raw_col, _raw_spacer = st.columns([1, 2])
            with raw_col.expander("View raw inputs"):
                show_fields = ["line_of_business", "risk_location_region", "broker_tier", "total_insured_value_usd",
                                "construction_type", "year_built", "occupancy_type", "number_of_employees",
                                "claims_free_years", "safety_program_flag", "business_tenure_years",
                                "revenue_growth_trend_pct", "debt_to_equity_ratio", "coverage_limit_requested_usd"]
                sel_dict = sel_row.to_dict()
                raw_tbl = pd.DataFrame([(k, str(sel_dict.get(k))) for k in show_fields if k in sel_dict], columns=["Field", "Value"])
                st.dataframe(raw_tbl, width="stretch", hide_index=True, height=250)

            loss_v, app_v, bind_v = sel_row["loss_propensity_score_v2"], sel_row["appetite_capacity_score_v2"], sel_row["bind_propensity_score_v2"]
            uw_v, tier = sel_row["uw_score_v2"], sel_row["risk_tier_v2"]
        else:
            st.caption(f"Dashboard › Aggregate view — {len(pool):,} filtered submission(s) (select a row above for single-submission detail)")
            loss_v, app_v, bind_v = pool["loss_propensity_score_v2"].mean(), pool["appetite_capacity_score_v2"].mean(), pool["bind_propensity_score_v2"].mean()
            uw_v = pool["uw_score_v2"].mean()
            tier = tier_for_score(uw_v)

        # ---- Steps 1-3: 3 component scores combining into the UW Score, full width. Real
        # 50/30/20 weights, verified to foot exactly against uw_score_v2 before building this
        # (Loss uses 100 minus Loss Propensity, since that score runs 0=best/100=worst, the
        # opposite sense of Appetite/Bind). ----
        _section_label("Steps 1–3 · Three Analytical Scores Combine Into One UW Score")
        suffix = "" if sel_row is not None else " (avg)"

        loss_contrib = 0.50 * (100 - loss_v)
        app_contrib = 0.30 * app_v
        bind_contrib = 0.20 * bind_v
        tier_color = TIER_COLORS[tier]
        contrib_bar = "".join(
            f'<div style="flex:{max(v,1.5)}; background:{c}; color:white; text-align:center; '
            f'font-size:0.74rem; font-weight:700; padding:7px 2px; white-space:nowrap; overflow:hidden;">{lbl} {v:.1f}</div>'
            for lbl, v, c in [("Loss", loss_contrib, GOOD), ("App", app_contrib, ACCENT), ("Bind", bind_contrib, NAVY)]
        )
        plus_sign = f'<div style="font-size:1.3rem; color:{TEXT_MUTED}; align-self:center; padding:0 2px;">+</div>'
        equals_sign = f'<div style="font-size:1.3rem; color:{TEXT_MUTED}; align-self:center; padding:0 8px;">=</div>'
        band_text = RISK_TIER_INFO[tier]["uw_score_band"]

        st.markdown(f"""
        <div class="cv-card" style="display:flex; align-items:center; gap:4px; padding:18px 24px; flex-wrap:nowrap; overflow-x:auto;">
            {_ring_gauge_cell(loss_v, f"Loss Propensity{suffix}", "0 best / 100 worst", BAD)}
            {plus_sign}
            {_ring_gauge_cell(app_v, f"Appetite Score{suffix}", "higher = better fit", GOOD)}
            {plus_sign}
            {_ring_gauge_cell(bind_v, f"Bind Propensity{suffix}", "higher = more likely", ACCENT)}
            {equals_sign}
            <div style="flex:1; min-width:280px;">
                <div style="font-size:0.74rem; color:{TEXT_MUTED}; margin-bottom:5px;">Contribution — real weights 50% Loss / 30% Appetite / 20% Bind</div>
                <div style="display:flex; border-radius:6px; overflow:hidden;">{contrib_bar}</div>
                <div style="margin-top:7px; display:flex; align-items:baseline; gap:8px;">
                    <span style="font-weight:800; font-size:1.5rem; color:{NAVY};">= {uw_v:.1f}</span>
                    <span class="cv-chip" style="margin:0; background:{_rgba(tier_color, 0.15)}; border-color:{_rgba(tier_color, 0.5)}; color:{tier_color}; font-weight:700;">{tier}{suffix}</span>
                </div>
                <div style="font-size:0.72rem; color:{TEXT_MUTED}; margin-top:2px;">UW score {band_text} · {tier}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ---- Steps 4-5 side by side: Triage on the left, Drivers on the right. ----
        tri_col, drv_col = st.columns(2)
        with tri_col:
            _section_label("Step 4 · Triage")
            band_order = [("Decline", 0, 38), ("High", 38, 55), ("Standard", 55, 72), ("Preferred", 72, 100)]
            triage_bar = "".join(
                f'<div style="flex:{end-start}; background:{TIER_COLORS[name]}; height:26px; display:flex; '
                f'align-items:center; justify-content:center; color:white; font-size:0.72rem; font-weight:700;">'
                f'{name} ({start}–{end})</div>' for name, start, end in band_order
            )
            marker_pct = min(max(uw_v, 0), 100)
            st.markdown(f"""
            <div style="position:relative; margin:10px 0 2px;">
                <div style="display:flex; border-radius:6px; overflow:hidden;">{triage_bar}</div>
                <div style="position:absolute; left:{marker_pct}%; top:-4px; width:3px; height:34px; background:{TEXT_MAIN};"></div>
            </div>
            <div style="color:{TEXT_MUTED}; font-size:0.78rem;">UW Score: {uw_v:.1f} / 100</div>
            """, unsafe_allow_html=True)
            if sel_row is not None:
                st.markdown(f"**Model recommends:** {sel_row['recommended_uw_action_v2']}")
                if is_watch:
                    st.markdown(f'<div style="background:{_rgba(WARN,0.1)}; border:1px solid {_rgba(WARN,0.4)}; '
                                f'border-radius:8px; padding:8px 12px; font-size:0.82rem; color:{TEXT_MAIN};">'
                                f'\U0001F6A9 Held for UW relook — {int(sel_row["prospective_risk_flag_count_v2"])} real forward-looking '
                                f'signal(s) not strongly weighted by the blended model: <i>{sel_row["prospective_risk_flags_v2"]}</i></div>',
                                unsafe_allow_html=True)
            else:
                st.caption(f"Average position for the {len(pool):,} filtered submission(s) shown above.")

        with drv_col:
            _section_label("Step 5 · What's Driving the Score")
            driver_view = st.radio("Driver view", ["UW Score", "Loss", "Appetite", "Bind"], horizontal=True,
                                    key="db_driver_view", label_visibility="collapsed")

            def _db_score_fn(d):
                return score(d, get_models()).iloc[0]

            if sel_row is not None:
                drv_row_df = book[book["submission_id"] == sel_id]
                drv_row_dict = drv_row_df.iloc[0].to_dict()
                drv_source = sel_row
            else:
                # Real submission closest to the pool's own mean UW score stands in as a
                # representative example - never a synthetic/blended row.
                closest_idx = (pool["uw_score_v2"] - uw_v).abs().idxmin()
                drv_row_df = pool.loc[[closest_idx]]
                drv_row_dict = drv_row_df.iloc[0].to_dict()
                drv_source = drv_row_df.iloc[0]
                st.caption(f"Representative example — **{drv_source['submission_id']}**, the real submission closest "
                           f"to this segment's average UW score ({uw_v:.1f}).")

            loss_drv = dexp.loss_drivers(drv_row_df, _db_score_fn)
            bind_drv = dexp.bind_drivers(drv_row_dict)
            appetite_drv = dexp.appetite_drivers(drv_row_dict)
            if driver_view == "UW Score":
                tagged = [(d, "Loss") for d in loss_drv[:2]] + [(d, "Appetite") for d in appetite_drv[:2]] + [(d, "Bind") for d in bind_drv[:2]]
                tagged = sorted(tagged, key=lambda t: -t[0]["weight"])[:6]
            elif driver_view == "Loss":
                tagged = [(d, "Loss") for d in loss_drv[:6]]
            elif driver_view == "Appetite":
                tagged = [(d, "Appetite") for d in appetite_drv[:6]]
            else:
                tagged = [(d, "Bind") for d in bind_drv[:6]]

            tag_color = {"Loss": BAD, "Appetite": GOOD, "Bind": ACCENT}
            # debt_to_equity_ratio is a real, declared candidate feature that was tested
            # against real claims and found too weak/confounded to trust - dropped from the
            # trained model (feature_map.py's own documented finding).
            for d, score_type in tagged:
                is_low_weight = "debt-to-equity" in d["label"].lower()
                dcolor = GOOD if d["direction"] == "decreases" else BAD
                low_tag = f'<span style="color:{WARN}; font-size:0.68rem; margin-left:6px;">low model weight</span>' if is_low_weight else ""
                st.markdown(
                    f'<div style="display:flex; align-items:center; gap:8px; font-size:0.82rem; padding:3px 0;">'
                    f'<span style="color:{TEXT_MAIN}; width:190px;">{d["label"]}{low_tag}</span>'
                    f'<div style="flex:1; background:{CARD_BORDER}; border-radius:3px; height:10px;">'
                    f'<div style="width:{d["weight"]:.0f}%; background:{dcolor}; height:10px; border-radius:3px;"></div></div>'
                    f'<span class="cv-chip" style="margin:0; background:{_rgba(tag_color[score_type],0.15)}; '
                    f'border-color:{_rgba(tag_color[score_type],0.5)}; color:{tag_color[score_type]};">{score_type}</span></div>',
                    unsafe_allow_html=True)

            summary_line = build_banner_summary(drv_source, loss_drv, bind_drv)
            st.caption(f"**Summary:** {summary_line}")

# ===========================================================================
# FEATURE MAP - identical data/logic to the existing app's Feature Map tab (reuses
# feature_map.py directly, unchanged), restyled only.
# ===========================================================================
with tab_feature:
    st.caption("Which real input fields feed each of the three scores - grouped by category, "
               "with a one-line reason for each.")

    fm_lob = st.selectbox("Line of Business", fmap.LOB_OPTIONS, index=0, key="fm_lob_filter",
                           help="Bind and Appetite never change by line - only Loss Propensity's "
                                "own column narrows to that line's real feature set.")
    fm_rows = fmap.build_reference_rows(fmap.loss_features_for(fm_lob))

    fm_loss_n = sum(r["Loss Propensity"] for r in fm_rows)
    fm_bind_n = sum(r["Bind Propensity"] for r in fm_rows)
    fm_app_n = sum(r["Appetite / Capacity"] for r in fm_rows)
    m1, m2, m3 = st.columns(3)
    m1.metric("Loss Propensity", f"{fm_loss_n} fields")
    m2.metric("Bind Propensity", f"{fm_bind_n} fields")
    m3.metric("Appetite / Capacity", f"{fm_app_n} fields")

    st.dataframe(
        pd.DataFrame(fm_rows), width="stretch", hide_index=True, height=1250,
        column_config={
            "Category": st.column_config.TextColumn("Category", width="small"),
            "Field": st.column_config.TextColumn("Field", width="medium"),
            "Loss Propensity": st.column_config.CheckboxColumn("Loss Propensity"),
            "Bind Propensity": st.column_config.CheckboxColumn("Bind Propensity"),
            "Appetite / Capacity": st.column_config.CheckboxColumn("Appetite / Capacity"),
            "Why": st.column_config.TextColumn("Why", width="large"),
        },
    )

# ===========================================================================
# WHAT-IF SIMULATOR - same fields, same flow, same Calculate button as the existing app
# ===========================================================================
with tab_whatif:
    business_type = st.radio("New Business or Renewal?", ["New Business", "Renewal"],
                              key="cv_business_type", horizontal=True)
    is_renewal = business_type == "Renewal"

    prior_direct = False
    if is_renewal:
        prior_relationship = st.radio(
            "Prior relationship on this account, before this renewal",
            ["Broker or MGA", "Direct (no broker)"], key="cv_renewal_prior", horizontal=True,
        )
        prior_direct = prior_relationship == "Direct (no broker)"

    lob_col, channel_col = st.columns(2)
    lob = lob_col.selectbox("Line of Business", LOB_OPTIONS, key="cv_lob")

    if prior_direct:
        channel = "Direct"
        channel_col.selectbox("Channel", ["Direct"], disabled=True, key="cv_channel_locked")
    else:
        channel_options = ["Broker", "MGA"] if is_renewal else ["Broker", "Direct", "MGA"]
        if st.session_state.get("cv_channel") not in channel_options:
            st.session_state["cv_channel"] = channel_options[0]
        channel = channel_col.selectbox("Channel", channel_options, key="cv_channel")

    show_broker_fields = channel != "Direct"
    if show_broker_fields:
        is_existing_broker = st.radio(
            "Is this an existing broker relationship?", ["No - new broker", "Yes - select broker"],
            key="cv_is_existing_broker", on_change=_reset_tier_on_toggle, horizontal=True,
        ) == "Yes - select broker"
        if is_existing_broker:
            broker_id = st.selectbox("Broker", sorted(KNOWN_BROKER_TIER.keys()), key="cv_broker_id",
                                      on_change=_reset_tier_to_known_broker)
            st.session_state.setdefault("cv_broker_tier", tier_for_known_broker(broker_id))
        else:
            broker_segment = st.selectbox("Broker Capacity Segment", ["Large National", "Regional", "Small Independent"],
                                           key="cv_broker_segment", on_change=_reset_tier_to_segment_default)
            st.session_state.setdefault("cv_broker_tier", NEW_BROKER_DEFAULT_TIER[broker_segment])
    else:
        st.caption("Direct channel - no broker relationship applies to this submission.")

    group = group_for_lob(lob)
    is_property = group == "property"
    st.caption(f"Showing **{'property/building' if is_property else 'business operations'}** inputs for {lob}.")

    with st.form("cv_submission_form"):
        if is_property:
            col1, col2, col3, col4 = st.columns(4)
        else:
            col1, col3, col4 = st.columns(3)

        with col1:
            st.markdown('<div class="cv-sub">Property &amp; Workforce</div>', unsafe_allow_html=True)
            if is_property:
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
                number_of_employees, avg_wage_per_employee_usd = 25, 55000
                avg_employee_age, avg_weekly_work_hours = 40, 40.0
            else:
                number_of_employees = st.number_input("Number of Employees", 1, 5000, 25)
                avg_wage_per_employee_usd = st.number_input("Avg. Wage per Employee ($)", 46000, 94000, 55000, step=1000)
                avg_employee_age = st.number_input("Avg. Employee Age", 19, 66, 40)
                avg_weekly_work_hours = st.number_input("Avg. Weekly Work Hours", 18.0, 50.0, 40.0)
                construction_type, year_built, built_to_code = "Fire Resistive", 2000, False
                num_stories, square_footage, total_insured_value_usd = 2, 5000, 500000
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
                st.markdown('<div class="cv-sub">Location Hazard</div>', unsafe_allow_html=True)
                distance_to_fire_station_miles = st.number_input("Distance to Fire Station (mi)", 0.1, 22.0, 5.0)
                distance_to_fire_hydrant_miles = st.number_input("Distance to Fire Hydrant (mi)", 0.0, 3.0, 0.5)
                fire_alarm_monitoring_type = st.selectbox("Fire Alarm Monitoring", CATEGORICAL_OPTIONS["fire_alarm_monitoring_type"])
                flood_zone_tier = st.selectbox("Flood Zone Tier", CATEGORICAL_OPTIONS["tier"])
                fema_flood_zone_code = st.selectbox("FEMA Flood Zone Code", CATEGORICAL_OPTIONS["fema_flood_zone_code"])
                earthquake_zone_tier = st.selectbox("Earthquake Zone Tier", CATEGORICAL_OPTIONS["tier"])
                soil_liquefaction_class = st.selectbox("Soil Liquefaction Class", CATEGORICAL_OPTIONS["soil_liquefaction_class"])
                wildfire_zone_tier = st.selectbox("Wildfire Zone Tier", CATEGORICAL_OPTIONS["tier"])
                wind_zone_tier = st.selectbox("Wind Zone Tier", CATEGORICAL_OPTIONS["tier"])
                primary_cat_peril_exposure = st.selectbox("Primary CAT Peril Exposure", PRIMARY_CAT_PERIL_OPTIONS)
                distance_to_coast_miles = st.number_input("Distance to Coast (mi)", 0.0, 800.0, 60.0)
                distance_to_fault_line_miles = st.number_input("Distance to Fault Line (mi)", 0.0, 400.0, 60.0)
                distance_to_nearest_water_body_miles = st.number_input("Distance to Water Body (mi)", 0.0, 50.0, 3.0)

        with col3:
            st.markdown('<div class="cv-sub">Claims History</div>', unsafe_allow_html=True)
            claim_count_last_3_years = st.number_input("Claim Count - Last 3 Years", 0, 9, 0)
            large_loss_count_last_5_years = st.number_input("Large-Loss Count - Last 5 Years", 0, 3, 0)
            claims_free_years = st.number_input("Claims-Free Years", 0, 10, 5)

            st.markdown('<div class="cv-sub" style="margin-top:0.8rem;">Financial &amp; Operational</div>', unsafe_allow_html=True)
            safety_program_flag = st.selectbox("Safety Program in Place", ["No", "Yes"]) == "Yes"
            safety_inspection_findings = st.selectbox("Safety Inspection Findings", CATEGORICAL_OPTIONS["safety_inspection_findings"])
            business_tenure_years = st.number_input("Business Tenure (years)", 0, 80, 10)
            revenue_growth_trend_pct = st.number_input("Revenue Growth Trend (%)", -30.0, 60.0, 3.0)
            debt_to_equity_ratio = st.number_input("Debt-to-Equity Ratio", 0.0, 5.0, 1.0)
            policy_term_months = st.selectbox("Policy Term (months)", [6, 12, 24, 36], index=1)
            financial_stability_score = estimate_financial_stability_score(business_tenure_years, safety_program_flag)
            st.markdown(f"""
            <div style="margin-bottom:0.4rem;">
                <div style="color:{TEXT_MUTED}; font-size:0.9rem; margin-bottom:0.3rem;">Financial Stability Score (estimated)</div>
                <div style="background:{PAGE_BG}; border:1px solid {CARD_BORDER}; border-radius:8px; padding:0.45rem 0.7rem; color:{TEXT_MAIN};">{financial_stability_score}</div>
            </div>""", unsafe_allow_html=True)
            st.caption(FINANCIAL_STABILITY_FIT_NOTE)
            st.caption("Premium is calculated automatically from the inputs above - not typed in.")

        with col4:
            st.markdown('<div class="cv-sub">Broker</div>', unsafe_allow_html=True)
            if show_broker_fields:
                broker_tier = st.selectbox("Broker Tier", ["Platinum", "Gold", "Silver", "Bronze"], key="cv_broker_tier")
            else:
                broker_tier = None
                st.caption("No broker relationship (Direct channel) - not applicable.")
            competing_quotes_count = st.number_input("Competing Quotes Count", 0, 10, 2)
            risk_state = st.selectbox("State", sorted(STATE_TO_REGION.keys()), index=sorted(STATE_TO_REGION.keys()).index("CA"))
            coverage_limit_requested_usd = st.number_input("Coverage Limit Requested ($)", 50000, 50000000, 500000, step=10000)
            industry_class_naics = st.selectbox("Industry Class (NAICS)", INDUSTRY_OPTIONS)

            st.markdown('<div class="cv-sub" style="margin-top:0.8rem;">Compliance &amp; Appetite</div>', unsafe_allow_html=True)
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
            "broker_tier": broker_tier, "channel": channel, "renewal_flag": business_type,
            "competing_quotes_count": competing_quotes_count,
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

        row_dict = row.iloc[0].to_dict()
        row_dict["risk_location_region"] = region_for_state(risk_state)

        def _score_fn(df):
            return score(df, models).iloc[0]

        loss_drv = dexp.loss_drivers(row, _score_fn)
        bind_drv = dexp.bind_drivers(row_dict)
        appetite_drv = dexp.appetite_drivers(row_dict)

        tier = result["risk_tier_v2"]
        tier_color = TIER_COLORS[tier]
        banner_bg = _rgba(tier_color, 0.15 if IS_DARK else 0.08)
        summary_line = build_banner_summary(result, loss_drv, bind_drv)

        # ---- Result banner + triage bar (one marker - this submission's score) ----
        st.markdown('<div class="cv-eyebrow" style="margin-top:0.6rem;">Underwriting Result</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="cv-card" style="border-left:6px solid {tier_color}; background:{banner_bg};">
            <div style="color:{tier_color}; font-size:1.55rem; font-weight:800;">{tier}</div>
            <div style="color:{TEXT_MAIN}; font-size:0.95rem; margin-top:4px;"><b>Recommended Action:</b> {result['recommended_uw_action_v2']}</div>
            <div style="color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px;">{summary_line}</div>
        """, unsafe_allow_html=True)
        band_order = [("Decline", 0, 38), ("High", 38, 55), ("Standard", 55, 72), ("Preferred", 72, 100)]
        bar_html = "".join(
            f'<div style="flex:{end-start}; background:{TIER_COLORS[name]}; height:24px; '
            f'display:flex; align-items:center; justify-content:center; color:white; font-size:0.7rem; font-weight:700;'
            f'{"border-radius:6px 0 0 6px;" if name=="Decline" else ""}{"border-radius:0 6px 6px 0;" if name=="Preferred" else ""}">{name}</div>'
            for name, start, end in band_order
        )
        marker_pct = min(max(result["uw_score_v2"], 0), 100)
        st.markdown(f"""
            <div style="position:relative; margin:14px 0 2px;">
                <div style="display:flex;">{bar_html}</div>
                <div style="position:absolute; left:{marker_pct}%; top:-4px; width:3px; height:32px; background:{TEXT_MAIN};"></div>
            </div>
            <div style="color:{TEXT_MUTED}; font-size:0.78rem;">UW Score: {result['uw_score_v2']:.1f} / 100</div>
        </div>
        """, unsafe_allow_html=True)

        # ---- KPI row - visible immediately, not hidden behind a click ----
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Composite UW Score", f"{result['uw_score_v2']:.1f}/100")
        k2.metric("Loss Propensity Score", f"{result['loss_propensity_score_v2']:.1f}")
        k3.metric("Bind Propensity Score", f"{result['bind_propensity_score_v2']:.1f}")
        k4.metric("Appetite Score", f"{result['appetite_capacity_score_v2']:.1f}")

        # ---- Key factors chips ----
        chip_source = loss_drv[:2] + bind_drv[:2] + appetite_drv[:2]
        seen, chips = set(), []
        for d in chip_source:
            if d["label"] not in seen:
                chips.append(d["label"]); seen.add(d["label"])
        if chips:
            chip_html = "".join(f'<span class="cv-chip">{c}</span>' for c in chips)
            st.markdown(f"""
            <div class="cv-card">
                <div class="cv-eyebrow">Key Factors Behind This Decision</div>
                {chip_html}
            </div>""", unsafe_allow_html=True)

        # ---- Why this score - 3 driver panels, always visible, value labeled on each bar ----
        def _driver_bar(drivers, good_direction):
            if not drivers:
                return None
            lbls = [d["label"] for d in drivers][::-1]
            vals = [d["weight"] if d["direction"] != good_direction else -d["weight"] for d in drivers][::-1]
            colors = [BAD if v > 0 else GOOD for v in vals]
            fig = go.Figure(go.Bar(
                x=vals, y=lbls, orientation="h", marker_color=colors,
                text=[f"{v:+.0f}" for v in vals], textposition="outside",
            ))
            fig.update_layout(height=max(170, 38 * len(drivers)), margin=dict(l=10, r=30, t=10, b=10),
                               xaxis=dict(showticklabels=False, zeroline=True, zerolinecolor=CARD_BORDER),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_MAIN, size=12))
            return fig

        with st.container():
            st.markdown(f'<div class="cv-eyebrow">Why This Score</div>', unsafe_allow_html=True)
            st.caption("Top factors behind each score, measured against the real book's own typical submission.")
            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown(f"<div class='cv-sub'>Loss Risk Drivers</div>", unsafe_allow_html=True)
                fig = _driver_bar(loss_drv, "decreases")
                if fig:
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.caption("No material drivers vs. book baseline.")
            with d2:
                st.markdown(f"<div class='cv-sub'>Bind Propensity Drivers</div>", unsafe_allow_html=True)
                fig = _driver_bar(bind_drv, "increases")
                if fig:
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.caption("No material drivers vs. book baseline.")
            with d3:
                st.markdown(f"<div class='cv-sub'>Appetite / Eligibility Drivers</div>", unsafe_allow_html=True)
                fig = _driver_bar(appetite_drv, "increases")
                if fig:
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.caption("Clean appetite profile.")

        # ---- Narrative ----
        narrative_text = build_narrative(result, loss_drv, bind_drv, appetite_drv)
        st.markdown(f"""
        <div class="cv-card">
            <div class="cv-eyebrow">Narrative</div>
            <div style="color:{TEXT_MAIN}; font-size:0.95rem; line-height:1.7;">{narrative_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # ---- Technical & audit detail ----
        with st.expander("Technical & audit detail"):
            st.caption("Every figure below is deterministic and reproducible. The same inputs always "
                       "produce the same output. The narrative above only explains these numbers in "
                       "plain English; it never alters them.")
            tab_calc, tab_reason = st.tabs(["Full calculation", "Reason codes"])
            with tab_calc:
                loading_pct = result["premium_loading_pct"]
                quoted = result["quoted_premium_usd"]
                calc_rows = [
                    ("Hard Stop", bool(result["appetite_capacity_score_v2"] == 0)),
                    ("Line of Business", result["line_of_business"]),
                    ("Predicted Claim Frequency", round(result["predicted_frequency"], 4)),
                    ("Predicted Attritional Severity ($)", round(result["predicted_attritional_severity"], 0)),
                    ("Probability of Large Loss", round(result["p_large_loss"], 4)),
                    ("Expected Large-Loss Severity ($)", round(result["expected_large_severity"], 0)),
                    ("Total Expected Loss Cost ($)", round(result["total_expected_loss_cost"], 0)),
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
