// Readable labels for the raw column/coefficient names the real scoring modules use
// internally - the manager's wireframe shows plain-English tags ("Hit/bind ratio", not
// "broker_hit_bind_ratio_pct"), and the real coefficients deserve the same treatment
// instead of leaking Python identifiers into a page underwriters will read.
const MECH_FIELD_LABELS = {
    intercept: "Intercept",
    appetite_hard_rule_pass_flag: "Hard-Rule Pass",
    target_growth_segment_flag: "Target Growth Segment",
    financial_stability_score: "Financial Stability Score",
    broker_tier_num: "Broker Tier",
    submission_completeness_pct: "Submission Completeness",
    coverage_limit_requested_usd: "Coverage Limit Requested",
    broker_hit_bind_ratio_pct: "Hit/Bind Ratio",
    broker_renewal_retention_pct: "Renewal Retention",
    broker_product_diversity_count: "Product Diversity",
    broker_geo_diversity_count: "Geographic Diversity",
    broker_profitability_score_0_100: "Profitability Score",
    quote_to_bind_days: "Quote-to-Bind Days",
    competing_quotes_count: "Competing Quotes",
    total_insured_value_usd: "Total Insured Value",
};
function mechLabel(key) {
    if (MECH_FIELD_LABELS[key]) return MECH_FIELD_LABELS[key];
    if (key.startsWith("region_")) return `Region: ${key.slice(7)}`;
    if (key.startsWith("channel_")) return `Channel: ${key.slice(8)}`;
    return key;
}
async function initModelNote() {
    const root = document.getElementById("page-model-note");
    const [mech, fm] = await Promise.all([
        apiGet("/api/model-mechanics"),
        apiGet("/api/feature-map", { lob: "All Lines" }),
    ]);

    root.innerHTML = `
        <div class="caption" style="margin-bottom:10px;">How each of the 3 scores is actually built - weights, hard rules, and the real fitted coefficients, pulled live from the scoring modules.</div>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:14px;">
            ${renderLossCard(mech.loss)}
            ${renderAppetiteCard(mech.appetite)}
            ${renderBindCard(mech.bind)}
            ${renderUwCard(mech.uw_score)}
        </div>

        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:14px;">
            <div class="mech-card">
                <div class="mech-title">${mechNum(5)}Outputs per submission</div>
                <div style="font-size:0.84rem; line-height:1.6;">Reason for decline/refer &middot; priority reason &middot; emerging-risk flags (financial stability, debt-to-equity, revenue trend, business tenure, safety, CAT zone, competing quotes) &middot; flag count</div>
            </div>
            <div class="mech-card">
                <div class="mech-title">${mechNum(6)}Available, not yet used</div>
                <div>${mech.available_not_used.map((f) => `<span class="mech-chip">${escapeHtml(f.label)}</span>`).join("")}</div>
                <div class="caption" style="margin-top:6px;">Real columns in the data, confirmed not used as an input to Loss, Appetite, or Bind today - checked programmatically against the real feature lists, not assumed.</div>
            </div>
        </div>

        <div class="card">
            <div class="mech-title" style="margin-bottom:10px;">${mechNum(7)}Field list &mdash; by category and score</div>
            <div class="caption" style="margin-bottom:10px;">Real input fields, rolled up by category (same categories as the Feature Map tab) - which score each category's fields feed, at a glance. Per-field detail with reasons lives on the Feature Map tab.</div>
            ${renderFieldSummaryTable(fm.rows)}
        </div>
    `;
}

function renderLossCard(loss) {
    return `
    <div class="mech-card" style="border-color:var(--bad);">
        <div class="mech-title" style="color:var(--bad);">${mechNum(1)}Loss propensity &mdash; weight ${loss.weight_pct}% &middot; 0 best, 100 worst</div>
        <div class="mech-question">How much loss do we expect from this risk?</div>
        <div class="override-banner" style="margin-bottom:10px;">&#9888; <b>Trained ML, not a fixed formula.</b> ${escapeHtml(loss.note)}</div>
        <div class="mech-group-label">Real pipeline (3 trained models)</div>
        <div>${loss.models.map((m) => `<span class="mech-chip">${escapeHtml(m)}</span>`).join("")}</div>
        <div class="formula-box">${loss.pipeline.map((l) => escapeHtml(l)).join("<br>")}</div>
        <div class="mech-group-label">Premium (denominator) &mdash; real per-LOB base rates</div>
        <div>${Object.entries(loss.premium_rates_pct).map(([lob, pct]) => `<span class="mech-chip">${escapeHtml(lob)}: ${pct}%</span>`).join("")}</div>
        <div class="caption" style="margin-top:4px;">${escapeHtml(loss.premium_basis)}</div>
        <div class="caption">${escapeHtml(loss.premium_note)}</div>
    </div>`;
}

function renderAppetiteCard(ap) {
    const fitChips = Object.entries(ap.fit_coefficients).map(([k, v]) => `<span class="mech-chip">${escapeHtml(mechLabel(k))}: ${v}</span>`).join("");
    return `
    <div class="mech-card" style="border-color:var(--good);">
        <div class="mech-title" style="color:var(--good);">${mechNum(2)}Appetite / capacity &mdash; weight ${ap.weight_pct}% &middot; 0-100</div>
        <div class="mech-question">May we write it, and do we have room?</div>
        <div class="mech-group-label">Hard-rule gate (fail &rarr; score 0, Decline)</div>
        <div>${ap.hard_rules.map((r) => `<span class="mech-chip">${escapeHtml(r)}</span>`).join("")}</div>
        <div class="mech-group-label">Appetite Fit &mdash; real fitted coefficients</div>
        <div>${fitChips}</div>
        <div class="mech-group-label">Capacity Headroom &mdash; tightest of industry / state</div>
        <div style="font-size:0.78rem;">Concentration cap: <b>${ap.concentration_cap_pct}%</b> of book TIV per industry</div>
        <div class="formula-box">
            ${escapeHtml(ap.industry_formula)}<br>${escapeHtml(ap.state_formula)}<br>${escapeHtml(ap.headroom_formula)}<br><br>${escapeHtml(ap.formula)}
        </div>
        <div class="caption" style="margin-top:6px;">Decision thresholds &mdash; Fit: strong &ge;${ap.decision_thresholds.fit_strong}, marginal &lt;${ap.decision_thresholds.fit_marginal}. Headroom: comfortable &ge;${ap.decision_thresholds.headroom_comfortable}, constrained &lt;${ap.decision_thresholds.headroom_constrained}.</div>
    </div>`;
}

function renderBindCard(bind) {
    const groupHtml = Object.entries(bind.groups).map(([label, fields]) => `
        <div class="mech-group-label">${escapeHtml(label)}</div>
        <div>${fields.map((f) => `<span class="mech-chip">${escapeHtml(mechLabel(f))}: ${bind.coefficients[f]}</span>`).join("")}</div>
    `).join("");
    return `
    <div class="mech-card" style="border-color:var(--accent);">
        <div class="mech-title" style="color:var(--accent);">${mechNum(3)}Bind propensity &mdash; weight ${bind.weight_pct}% &middot; 0-100</div>
        <div class="mech-question">If we quote, will the broker bind with us?</div>
        <div class="caption">${escapeHtml(bind.note)}</div>
        ${groupHtml}
        <div class="formula-box">${escapeHtml(bind.formula)}<br>Intercept: ${bind.coefficients.intercept}</div>
    </div>`;
}

function renderUwCard(uw) {
    const tierRows = uw.tiers.map((t) => `
        <div style="margin-bottom:8px;">
            <div class="tier-badge tier-bg-${t.name}" style="display:inline-block; padding:5px 10px;">
                ${escapeHtml(t.name)} (${escapeHtml(t.uw_score_band)}) &mdash; ${escapeHtml(t.uw_action)}
            </div>
            <div style="font-size:0.78rem; margin-top:3px;">${escapeHtml(t.what_it_means)} &middot; Pricing: <b>${escapeHtml(t.pricing_guidance)}</b></div>
        </div>`).join("");
    return `
    <div class="mech-card" style="border-color:var(--navy);">
        <div class="mech-title" style="color:var(--navy);">${mechNum(4)}UW score &rarr; triage</div>
        <div class="formula-box">${escapeHtml(uw.formula)}</div>
        <div class="caption" style="margin:4px 0 10px;">${escapeHtml(uw.note)}</div>
        ${tierRows}
    </div>`;
}

function renderFieldSummaryTable(rows) {
    const order = ["Claims & Financial History", "Property & Building", "Catastrophe & Location", "Workforce", "Broker & Deal", "Compliance & Strategy"];
    const groups = {};
    for (const r of rows) {
        const cat = r.Category;
        if (!groups[cat]) groups[cat] = { loss: [], bind: [], appetite: [] };
        if (r["Loss Propensity"]) groups[cat].loss.push(r.Field);
        if (r["Bind Propensity"]) groups[cat].bind.push(r.Field);
        if (r["Appetite / Capacity"]) groups[cat].appetite.push(r.Field);
    }
    const cats = order.filter((c) => groups[c]).concat(Object.keys(groups).filter((c) => !order.includes(c)));
    const cell = (list) => list.length ? escapeHtml(list.join(", ")) : '<span class="muted">&mdash;</span>';
    return `<table class="data-table fm-table">
        <thead><tr><th>Category</th><th>Loss</th><th>Bind</th><th>Appetite</th></tr></thead>
        <tbody>
            ${cats.map((c) => `<tr><td><b>${escapeHtml(c)}</b></td><td>${cell(groups[c].loss)}</td><td>${cell(groups[c].bind)}</td><td>${cell(groups[c].appetite)}</td></tr>`).join("")}
        </tbody>
    </table>`;
}
