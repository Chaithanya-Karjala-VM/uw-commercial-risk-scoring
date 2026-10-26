const API_BASE = ""; // same origin - FastAPI serves this frontend directly

async function apiGet(path, params = {}) {
    const qs = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== ""));
    const url = `${API_BASE}${path}${qs.toString() ? "?" + qs.toString() : ""}`;
    const res = await fetch(url);
    if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`GET ${path} failed: ${res.status} ${text}`);
    }
    return res.json();
}

async function apiPost(path, body) {
    const res = await fetch(`${API_BASE}${path}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`POST ${path} failed: ${res.status} ${text}`);
    }
    return res.json();
}

const TIER_COLORS = { Preferred: "var(--good)", Standard: "var(--accent)", High: "var(--warn)", Decline: "var(--bad)" };
const TIER_ORDER = ["Preferred", "Standard", "High", "Decline"];

function fmtMoney(v, digits = 1) {
    if (v === null || v === undefined) return "—";
    const abs = Math.abs(v);
    if (abs >= 1e9) return `$${(v / 1e9).toFixed(digits)}B`;
    if (abs >= 1e6) return `$${(v / 1e6).toFixed(digits)}M`;
    if (abs >= 1e3) return `$${(v / 1e3).toFixed(0)}K`;
    return `$${v.toFixed(0)}`;
}
function fmtNum(v, digits = 1) {
    if (v === null || v === undefined) return "—";
    return v.toFixed(digits);
}
function fmtPct(v, digits = 1) {
    if (v === null || v === undefined) return "—";
    return `${v.toFixed(digits)}%`;
}
function fmtInt(v) {
    if (v === null || v === undefined) return "—";
    return Math.round(v).toLocaleString();
}
function escapeHtml(s) {
    if (s === null || s === undefined) return "";
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function tierChip(tier, extra = "") {
    return `<span class="tier-badge tier-bg-${tier}">${escapeHtml(tier)}${extra}</span>`;
}

// Small numbered circle badge used in front of section/card titles on pages built to mirror
// a numbered wireframe (Model Note, Underwriting File submission detail).
function mechNum(n) {
    return `<span class="mech-num">${n}</span>`;
}

// Resolves a "var(--xxx)" reference to its concrete computed color - Plotly draws into its
// own SVG/canvas layer, not the page's CSS cascade, so chart code needs the actual resolved
// value rather than the var() reference itself.
function cssVal(v) {
    return getComputedStyle(document.documentElement).getPropertyValue(v.replace("var(", "").replace(")", "")).trim();
}

// Small circular SVG ring gauge - same technique as the Streamlit build's ring gauges:
// hand-built SVG for exact pixel control, no charting library overhead for something this small.
function ringGaugeHtml(value, label, sub, colorVar, size = 76) {
    const stroke = 8, r = 50 - stroke / 2, circumference = 2 * Math.PI * r;
    const pct = Math.max(0, Math.min(100, value || 0)) / 100;
    const offset = circumference * (1 - pct);
    return `
    <div class="ring-wrap" style="min-width:${size + 14}px;">
        <div style="position:relative; width:${size}px; height:${size}px;">
            <svg width="${size}" height="${size}" viewBox="0 0 100 100" style="transform:rotate(-90deg);">
                <circle cx="50" cy="50" r="${r}" fill="none" stroke="var(--card-border)" stroke-width="${stroke}"/>
                <circle cx="50" cy="50" r="${r}" fill="none" stroke="${colorVar}" stroke-width="${stroke}"
                    stroke-linecap="round" stroke-dasharray="${circumference.toFixed(2)}" stroke-dashoffset="${offset.toFixed(2)}"/>
            </svg>
            <div style="position:absolute; inset:0; display:flex; align-items:center; justify-content:center;" class="ring-value">
                ${value === null || value === undefined ? "—" : value.toFixed(1)}
            </div>
        </div>
        <div class="ring-label">${escapeHtml(label)}</div>
        <div class="ring-sub">${escapeHtml(sub)}</div>
    </div>`;
}

function driverRowHtml(driver, scoreType, tagColorVar, goodDirection = "decreases") {
    const dcolor = driver.direction === goodDirection ? "var(--good)" : "var(--bad)";
    return `
    <div class="driver-row">
        <span class="driver-label">${escapeHtml(driver.label)}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${driver.weight}%; background:${dcolor};"></div></div>
        <span class="chip" style="border-color:${tagColorVar}; color:${tagColorVar};">${scoreType}</span>
    </div>`;
}

// Zero-centered diverging bar - matches the Streamlit What-If Simulator's driver chart
// exactly: direction === goodDirection means this field is pulling the score the FAVORABLE
// way (rendered as a green bar growing left of center), anything else is unfavorable (red,
// grows right). goodDirection is "decreases" for Loss (lower loss ratio is better) and
// "increases" for Bind/Appetite (higher score is better) - see driver_explain.py's own
// direction-semantics note for why this isn't just "decreases=good" everywhere.
function divergingBarHtml(driver, goodDirection) {
    const favorable = driver.direction === goodDirection;
    const val = favorable ? -driver.weight : driver.weight;
    const color = val > 0 ? "var(--bad)" : "var(--good)";
    const halfWidth = Math.min(Math.abs(val), 100) / 2;
    const side = val >= 0 ? `left:50%; width:${halfWidth}%;` : `right:50%; width:${halfWidth}%;`;
    return `
    <div class="diverge-row">
        <span class="diverge-label">${escapeHtml(driver.label)}</span>
        <div class="diverge-track"><div class="diverge-fill" style="${side} background:${color};"></div></div>
        <span class="diverge-value" style="color:${color};">${val >= 0 ? "+" : ""}${val.toFixed(0)}</span>
    </div>`;
}
