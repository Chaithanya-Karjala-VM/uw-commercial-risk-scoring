const ufState = { lob: "All", region: "All", tier: "All", broker_tier: "All", renewal: "All", status: "All", sort: "priority", selectedId: null };

// Short-form appetite decision text, reused on both the queue-card pill and the submission
// detail's Exposure & capacity pill.
const UF_SHORT_DECISION = {
    "Within Appetite": "within appetite", "Refer – Marginal Fit": "marginal fit",
    "Refer – Capacity Constrained": "capacity constrained", "Decline – Outside Appetite": "outside appetite",
};
// Same real cutoffs already documented on the Model Note page's Bind card (bind.coefficients
// aren't tiered there, but the UW-score tier text "Bind tier: >=55 High * 35-54 Medium * <35
// Low" is) - reused here instead of inventing a second threshold set.
function ufBindTier(score) {
    if (score === null || score === undefined) return { label: "—", color: "var(--text-muted)" };
    if (score >= 55) return { label: "High", color: "var(--good)" };
    if (score >= 35) return { label: "Medium", color: "var(--accent)" };
    return { label: "Low", color: "var(--bad)" };
}
// Solid-filled status pill (reuses the same tier-badge look already used for risk tiers)
// for an arbitrary color - the wireframe's decision-dimension cards use a solid colored
// pill for their status, not the lighter outlined chip style used elsewhere in this app.
function ufPill(text, color) {
    return `<span class="tier-badge" style="background:${color};">${escapeHtml(text)}</span>`;
}

async function initUnderwritingFile() {
    const root = document.getElementById("page-underwriting");
    const filters = await apiGet("/api/filters");
    const mkFilter = (key, label, options) => `
        <div class="filter-item"><label class="field-label">${label}</label>
        <select id="uf-filter-${key}"><option value="All">All</option>
        ${options.map((o) => `<option value="${escapeHtml(o)}">${escapeHtml(o)}</option>`).join("")}
        </select></div>`;
    root.innerHTML = `
        <div class="grid-2fr-1fr" style="grid-template-columns: 380px 1fr; align-items: start;">
            <div class="card" style="display:flex; flex-direction:column; box-sizing:border-box; padding:14px 16px;
                position:sticky; top:14px; max-height:calc(100vh - 28px); overflow:hidden;">
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                    <div class="eyebrow" style="margin-bottom:0;">Submission queue</div>
                    <button type="button" id="uf-filters-toggle" class="btn" style="padding:3px 9px; font-size:0.72rem;">Hide filters &#9660;</button>
                </div>
                <div id="uf-filters-panel" style="flex-shrink:0;">
                    <div class="filter-row" style="margin-bottom:6px;">
                        ${mkFilter("lob", "LOB", filters.lobs)}
                        ${mkFilter("region", "Region", filters.regions)}
                        ${mkFilter("tier", "Tier", filters.tiers)}
                    </div>
                    <div class="filter-row" style="margin-bottom:6px;">
                        ${mkFilter("broker_tier", "Broker", filters.broker_tiers)}
                        ${mkFilter("renewal", "Renewal", filters.renewal)}
                        <div class="filter-item"><label class="field-label">Status</label>
                            <select id="uf-filter-status"><option value="All">All</option><option value="To Review">To Review</option></select></div>
                    </div>
                </div>
                <div class="filter-item" style="min-width:100%; margin-bottom:6px; flex-shrink:0;"><label class="field-label">Sort</label>
                    <select id="uf-sort"><option value="priority">Priority (needs review first)</option><option value="uw_asc">UW score, worst first</option><option value="uw_desc">UW score, best first</option></select></div>
                <div class="caption" id="uf-queue-kpis" style="margin-bottom:4px; flex-shrink:0;"></div>
                <div class="caption" id="uf-queue-count" style="margin-bottom:4px; flex-shrink:0;"></div>
                <div id="uf-queue-list" style="flex:1; overflow-y:auto; min-height:0;"></div>
                <div style="flex-shrink:0; margin-top:6px; padding-top:6px; border-top:1px solid var(--card-border);">
                    <button type="button" id="uf-triage-toggle" class="btn" style="padding:3px 9px; font-size:0.72rem; width:100%;">Triage bands &#9654;</button>
                    <div id="uf-triage-legend" style="display:none; margin-top:6px;"></div>
                </div>
            </div>
            <div id="uf-detail"></div>
        </div>
    `;
    document.getElementById("uf-filters-toggle").addEventListener("click", () => {
        const panel = document.getElementById("uf-filters-panel");
        const btn = document.getElementById("uf-filters-toggle");
        const hidden = panel.style.display === "none";
        panel.style.display = hidden ? "block" : "none";
        btn.innerHTML = hidden ? "Hide filters &#9660;" : "Show filters &#9654;";
    });
    document.getElementById("uf-triage-toggle").addEventListener("click", () => {
        const panel = document.getElementById("uf-triage-legend");
        const btn = document.getElementById("uf-triage-toggle");
        const hidden = panel.style.display === "none";
        panel.style.display = hidden ? "block" : "none";
        btn.innerHTML = hidden ? "Triage bands &#9660;" : "Triage bands &#9654;";
    });
    ["lob", "region", "tier", "broker_tier", "renewal", "status"].forEach((key) => {
        document.getElementById(`uf-filter-${key}`).addEventListener("change", (e) => {
            ufState[key] = e.target.value;
            refreshQueue();
        });
    });
    document.getElementById("uf-sort").addEventListener("change", (e) => { ufState.sort = e.target.value; refreshQueue(); });
    renderTriageLegend();
    await refreshQueue();
}

function renderTriageLegend() {
    // Real tier cutoffs and real recommended actions - same RISK_TIER_INFO text already
    // shown in the Model Note and used throughout scoring.py, not wireframe-invented wording.
    const bands = [
        ["Preferred", "72-100", "Fast-track - no rate action"],
        ["Standard", "55-71", "Rate at technical price"],
        ["High", "38-54", "Rate adjustment + deductible review"],
        ["Decline", "< 38, or any hard-rule fail", "Refer to senior underwriter"],
    ];
    document.getElementById("uf-triage-legend").innerHTML = `
        ${bands.map(([tier, band, action]) => `
            <div style="display:flex; align-items:center; gap:7px; padding:2px 0; font-size:0.76rem;">
                <span style="width:10px; height:10px; border-radius:3px; background:${TIER_COLORS[tier]}; flex-shrink:0;"></span>
                <b style="width:66px; flex-shrink:0;">${tier}</b>
                <span class="caption" style="width:110px; flex-shrink:0;">${band}</span>
                <span class="caption">${action}</span>
            </div>`).join("")}
    `;
}

async function refreshQueue() {
    const q = await apiGet("/api/queue", { lob: ufState.lob, region: ufState.region, tier: ufState.tier,
        broker_tier: ufState.broker_tier, renewal: ufState.renewal, status: ufState.status, sort: ufState.sort, limit: 60 });
    document.getElementById("uf-queue-kpis").innerHTML =
        `<b style="color:var(--text-main);">${fmtInt(q.total_matching)}</b> open &middot; <b style="color:var(--text-main);">${fmtInt(q.rows.filter((r) => r.risk_tier === "High").length)}</b> capacity refer in this page`;
    document.getElementById("uf-queue-count").textContent = `Showing ${q.returned.toLocaleString()} of ${q.total_matching.toLocaleString()} matching submissions.`;
    // Matches the manager's wireframe card: bordered card, filled score circle top-right,
    // LOB + state line, and one solid tier-colored pill carrying "Tier · why" together -
    // same real fields as before, just laid out the way the wireframe shows it.
    document.getElementById("uf-queue-list").innerHTML = q.rows.map((r) => {
        const statusText = r.is_hard_fail ? "hard-rule fail" : (UF_SHORT_DECISION[r.appetite_decision] || r.appetite_decision || "");
        const selected = r.submission_id === ufState.selectedId;
        return `
        <div class="queue-item" data-id="${escapeHtml(r.submission_id)}" style="padding:9px 11px; margin-bottom:7px; border-radius:10px; cursor:pointer;
            border:1px solid ${selected ? "var(--accent)" : "var(--card-border)"};
            background:${selected ? "color-mix(in srgb, var(--accent) 8%, transparent)" : "var(--card-bg)"};">
            <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:8px;">
                <div style="min-width:0;">
                    <div style="font-weight:700; font-size:0.86rem;">${escapeHtml(r.submission_id)}</div>
                    <div class="caption" style="font-size:0.76rem; margin-top:1px;">${escapeHtml(r.line_of_business)}${r.risk_state ? ` &middot; ${escapeHtml(r.risk_state)}` : ""}</div>
                </div>
                <div style="width:30px; height:30px; border-radius:50%; background:${TIER_COLORS[r.risk_tier]}; color:#fff; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:0.72rem; flex-shrink:0;">${Math.round(r.uw_score)}</div>
            </div>
            <span class="tier-badge tier-bg-${r.risk_tier}" style="display:inline-block; margin-top:6px; font-size:0.7rem; padding:2px 9px;">${escapeHtml(r.risk_tier)} &middot; ${escapeHtml(statusText)}</span>
        </div>`;
    }).join("");
    document.querySelectorAll(".queue-item").forEach((item) => item.addEventListener("click", () => {
        const alreadySelected = ufState.selectedId === item.dataset.id;
        document.querySelectorAll(".queue-item").forEach((i) => i.style.background = "");
        if (alreadySelected) {
            // Click the already-open row again to close it back out to the aggregate view -
            // selection is a toggle, not a one-way door.
            ufState.selectedId = null;
            renderAggregateDetail();
        } else {
            ufState.selectedId = item.dataset.id;
            item.style.background = "color-mix(in srgb, var(--accent) 10%, transparent)";
            loadSubmissionDetail(item.dataset.id);
        }
    }));
    // Default state (nothing ever clicked, or a filter was just applied/changed) shows an
    // aggregate KPI view, not any one submission's detail - a specific submission only shows
    // once the user actually picks one from the queue. The queue list itself always reflects
    // every matching submission regardless (filters narrow it, they never collapse it to 1 row).
    if (ufState.selectedId) {
        loadSubmissionDetail(ufState.selectedId);
    } else {
        await renderAggregateDetail();
    }
}

async function renderAggregateDetail() {
    const filters = { lob: ufState.lob, region: ufState.region, tier: ufState.tier,
        broker_tier: ufState.broker_tier, renewal: ufState.renewal, status: ufState.status };
    const k = await apiGet("/api/portfolio/kpis", filters);
    document.getElementById("uf-detail").innerHTML = `
        <div class="card">
            <div class="eyebrow">No submission selected</div>
            <div class="kpi-row">
                <div class="kpi-card"><div class="label">Submissions</div><div class="value">${fmtInt(k.submissions)}</div></div>
                <div class="kpi-card"><div class="label">Avg UW Score</div><div class="value">${fmtNum(k.avg_uw_score)}</div></div>
                <div class="kpi-card"><div class="label">Accept Rate</div><div class="value">${fmtPct(k.accept_rate_pct)}</div><div class="sub">Preferred + Standard</div></div>
                <div class="kpi-card"><div class="label">Total TIV</div><div class="value">${fmtMoney(k.total_tiv_usd)}</div></div>
                <div class="kpi-card"><div class="label">Premium</div><div class="value">${fmtMoney(k.premium_usd)}</div></div>
                <div class="kpi-card"><div class="label">Hard-Rule Fails</div><div class="value">${fmtInt(k.hard_rule_fails)}</div><div class="sub">${fmtPct(k.hard_rule_fails_pct)} of submissions</div></div>
            </div>
        </div>

        <div class="card">
            <div class="mech-title">Average UW Score by line of business, for this queue</div>
            <div id="uf-agg-breakdown"></div>
        </div>
    `;
    await refreshUfAggBreakdown();
}

async function refreshUfAggBreakdown() {
    const data = await apiGet("/api/queue/breakdown", { lob: ufState.lob, region: ufState.region, tier: ufState.tier,
        broker_tier: ufState.broker_tier, renewal: ufState.renewal, status: ufState.status, by: "lob" });
    const el = document.getElementById("uf-agg-breakdown");
    if (!data.rows.length) {
        el.innerHTML = `<div class="caption">No submissions match the current filters.</div>`;
        return;
    }
    // Worst-scoring line at the top, matching "where do I look first."
    const rows = [...data.rows].sort((a, b) => a.avg_uw - b.avg_uw).reverse();
    const tierColorFor = (uw) => uw >= 72 ? cssVal("var(--good)") : uw >= 55 ? cssVal("var(--accent)") : uw >= 38 ? cssVal("var(--warn)") : cssVal("var(--bad)");
    Plotly.newPlot("uf-agg-breakdown", [{
        x: rows.map((r) => r.avg_uw), y: rows.map((r) => `${r.label} (${fmtInt(r.submissions)})`),
        type: "bar", orientation: "h", marker: { color: rows.map((r) => tierColorFor(r.avg_uw)) },
        text: rows.map((r) => r.avg_uw.toFixed(1)), textposition: "outside", cliponaxis: false,
        hovertemplate: "%{y}<br>Avg UW Score %{x:.1f}/100<extra></extra>",
    }], {
        // range runs a bit past 100 (not just [0,100]) purely so the "outside" score label on
        // a near-100 bar always has room to render instead of getting clipped at the edge -
        // the axis itself is still read as the real 0-100 scale, same as every other UW Score
        // on this page.
        height: Math.max(220, 44 * rows.length + 70), margin: { l: 10, r: 46, t: 10, b: 36 },
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: { title: "Avg UW Score (0 = worst, 100 = best)", range: [0, 112], tick0: 0, dtick: 20 }, yaxis: { automargin: true, ticksuffix: "  " },
        font: { color: cssVal("var(--text-main)"), size: 11 },
    }, { displayModeBar: false, responsive: true });
    let note = document.getElementById("uf-agg-breakdown-note");
    if (!note) {
        note = document.createElement("div");
        note.id = "uf-agg-breakdown-note";
        note.className = "caption";
        note.style.marginTop = "6px";
        el.after(note);
    }
    note.textContent = "Bar color: green = avg ≥ 72 (Preferred), blue = 55-71 (Standard), amber = 38-54 (High), red = below 38 (Decline) - the same cutoffs used on every triage bar in this app. Sorted worst to best, top to bottom.";
}

function renderNextBestActions(drivers) {
    // "direction: increases" means a driver is pushing the score in the WORSE direction
    // (driver_explain.py's own documented convention) - the weakest real drivers, not a
    // separate invented ranking.
    const all = [...drivers.loss.map((d) => ({ ...d, type: "Loss" })),
                 ...drivers.appetite.map((d) => ({ ...d, type: "Appetite" })),
                 ...drivers.bind.map((d) => ({ ...d, type: "Bind" }))]
        .filter((d) => d.direction === "increases")
        .sort((a, b) => b.weight - a.weight)
        .slice(0, 3);
    if (!all.length) {
        return `<div class="caption">No driver is currently pulling the score down - nothing weak to flag.</div>`;
    }
    return `<ul style="margin:6px 0 0; padding-left:20px; font-size:0.84rem; line-height:1.5;">
        ${all.map((d) => `<li style="margin-bottom:8px;">Address <b>${escapeHtml(d.label)}</b> (${escapeHtml(d.type)}) - the strongest factor currently pulling this score down.</li>`).join("")}
    </ul>`;
}

async function loadSubmissionDetail(id) {
    const detailEl = document.getElementById("uf-detail");
    detailEl.innerHTML = `<div class="loading">Loading ${escapeHtml(id)}...</div>`;
    const d = await apiGet(`/api/submission/${encodeURIComponent(id)}`);
    const s = d.scores;

    const overrideHtml = d.override.is_hard_fail ? `
        <div class="override-banner" style="margin-bottom:12px;">
            &#9888; <b>Override &mdash; Decline.</b> ${escapeHtml(d.override.decline_reason)}
        </div>` : "";

    const bandOrder = [["Decline", 0, 38], ["High", 38, 55], ["Standard", 55, 72], ["Preferred", 72, 100]];
    const triageBar = bandOrder.map(([name, lo, hi]) =>
        `<div style="flex:${hi - lo}; background:${TIER_COLORS[name]}; height:22px; display:flex; align-items:center; justify-content:center; color:white; font-size:0.68rem; font-weight:700;">${name} (${lo}-${hi})</div>`).join("");
    const markerPct = Math.min(Math.max(s.uw_score, 0), 100);

    // Colored dot bullets (green = pass/ok, red/amber = not) instead of checkmark glyphs -
    // same real pass/ok booleans, just the wireframe's own visual language for them.
    // Hanging indent (padding-left + negative text-indent) instead of flex - flex was
    // treating the bullet, the factor text, and the note as separate items that could each
    // wrap independently, which is what produced the broken-looking gap between a short
    // factor name and its note. This way it reads and wraps as one normal paragraph, with
    // continuation lines lining up under the text instead of under the bullet.
    const hardChecks = d.decision_dimensions.appetite_eligibility.checks.map((c) =>
        `<div style="font-size:0.8rem; padding:2px 0 2px 15px; text-indent:-15px;"><span style="color:${c.passed ? "var(--good)" : "var(--bad)"}; font-size:0.7rem;">&#9679;</span>&nbsp; ${escapeHtml(c.rule)}</div>`).join("");
    const softFactors = d.decision_dimensions.appetite_eligibility.soft_factors.map((f) =>
        `<div class="caption" style="padding:2px 0 2px 15px; text-indent:-15px;"><span style="color:${f.ok ? "var(--good)" : "var(--warn)"}; font-size:0.7rem;">&#9679;</span>&nbsp; ${escapeHtml(f.factor)} <span class="muted">(${escapeHtml(f.note)})</span></div>`).join("");

    const loss = d.decision_dimensions.loss, exp = d.decision_dimensions.exposure_capacity, bb = d.decision_dimensions.broker_bind;

    const hardAllPass = d.decision_dimensions.appetite_eligibility.hard_rules_passed === d.decision_dimensions.appetite_eligibility.hard_rules_total;
    // Same real rule the Portfolio page already documents - loss ratio highlighted red above 80%.
    const lrColor = (v) => (v === null || v === undefined) ? "inherit" : (v > 80 ? "var(--bad)" : "var(--text-main)");
    const headroomColor = (v) => (v === null || v === undefined) ? "inherit" : (v < 0 ? "var(--bad)" : "var(--good)");
    const apDecisionColorMap = { "Within Appetite": "var(--good)", "Refer – Marginal Fit": "var(--warn)", "Refer – Capacity Constrained": "var(--warn)", "Decline – Outside Appetite": "var(--bad)" };
    const apDecisionColor = apDecisionColorMap[exp.appetite_decision] || "var(--text-muted)";
    const apDecisionShort = UF_SHORT_DECISION[exp.appetite_decision] || exp.appetite_decision || "—";
    const bindTier = ufBindTier(bb.bind_score);

    // Manager direction: when a sub-score has been forced to its extreme auto-fail/auto-cap
    // value (Loss 100, Appetite 0, or Bind 0), the composite UW Score number isn't a
    // meaningful number to show - one input already dominates it. Show why instead, in
    // priority order (the hard-rule veto first, since it overrides Loss/Bind regardless).
    const failedRules = d.decision_dimensions.appetite_eligibility.checks.filter((c) => !c.passed).map((c) => c.rule);
    let uwSuppressReason = null;
    if (s.appetite_capacity === 0 && failedRules.length) {
        uwSuppressReason = `Appetite/Capacity is 0 &mdash; fails hard rule${failedRules.length > 1 ? "s" : ""}: <b>${failedRules.map(escapeHtml).join(", ")}</b>. Decline regardless of Loss or Bind.`;
    } else if (s.loss_propensity === 100) {
        uwSuppressReason = loss.flags
            ? `Loss Score is at its maximum (100) &mdash; driven by: <b>${escapeHtml(loss.flags)}</b>.`
            : `Loss Score is at its maximum (100) &mdash; predicted loss ratio exceeds 100% of premium.`;
    } else if (s.bind_propensity === 0) {
        const topBindDrv = [...d.drivers.bind].filter((dr) => dr.direction === "decreases").sort((a, b) => b.weight - a.weight)[0];
        uwSuppressReason = topBindDrv
            ? `Bind Score is at its floor (0) &mdash; largest factor: <b>${escapeHtml(topBindDrv.label)}</b>.`
            : `Bind Score is at its floor (0).`;
    }

    // Real, honest capacity signal distinct from the hard-fail override above: when treaty/
    // concentration headroom is tight the real model does NOT automatically downgrade the
    // tier (only the 4 hard rules force a Decline) - flagged here as advisory information an
    // underwriter should weigh, not represented as something the engine already acted on.
    const capacityNoteHtml = (!d.override.is_hard_fail && (exp.appetite_decision || "").includes("Capacity Constrained")) ? `
        <div class="override-banner" style="margin-bottom:12px;">
            &#9888; <b>Capacity note &mdash; ${escapeHtml(exp.appetite_decision)}.</b>
            ${escapeHtml(exp.industry || "This industry")}'s headroom is tight (regional ${fmtPct(exp.regional_treaty_headroom_pct)}, country ${fmtPct(exp.country_treaty_headroom_pct)}, state ${fmtPct(exp.state_treaty_headroom_pct)}).
            The tier above is not automatically downgraded for this - worth a capacity check before binding.
        </div>` : "";

    const rosterData = await apiGet("/api/underwriters");
    const assignOptionsHtml = [`<option value="">Unassigned</option>`]
        .concat(rosterData.roster.map((u) => `<option ${u === d.assigned_to ? "selected" : ""}>${escapeHtml(u)}</option>`))
        .join("");

    detailEl.innerHTML = `
        <div class="card" style="padding:14px 18px;">
            <div style="display:flex; justify-content:space-between; align-items:start; flex-wrap:wrap; gap:10px;">
                <div>
                    <div style="font-size:1.2rem; font-weight:800;">${escapeHtml(d.submission_id)} &middot; ${escapeHtml(d.industry || d.line_of_business)}</div>
                    <div class="caption" title="${escapeHtml(d.business_description || "")}">${escapeHtml(d.line_of_business)} &middot; ${d.is_new_business ? "New business" : "Renewal"}${d.effective_date ? ` &middot; effective ${escapeHtml(d.effective_date)}` : ""}${d.broker_agent_id ? ` &middot; broker ${escapeHtml(d.broker_agent_id)}` : ""}</div>
                </div>
                <div style="text-align:right;">
                    <div class="stat-label">Assigned</div>
                    <select id="uf-assign-select" style="margin-top:3px;">${assignOptionsHtml}</select>
                </div>
            </div>
            <div class="uf-stat-grid">
                <div class="uf-stat-box"><div class="stat-label">Risk location</div><div class="stat-value">${escapeHtml(d.region)} &middot; ${escapeHtml(d.state || "—")}</div></div>
                <div class="uf-stat-box"><div class="stat-label">Broker</div><div class="stat-value">${escapeHtml(d.broker_tier)} &middot; ${escapeHtml(d.broker_size_class || "—")}</div></div>
                <div class="uf-stat-box"><div class="stat-label">TIV</div><div class="stat-value">${fmtMoney(d.tiv_usd)}</div></div>
                <div class="uf-stat-box"><div class="stat-label">AOA / AOY Limit</div><div class="stat-value">${fmtMoney(d.aoa_limit_usd, 0)} / ${fmtMoney(d.aoy_limit_usd, 0)}</div></div>
                <div class="uf-stat-box"><div class="stat-label">Indicated Premium</div><div class="stat-value">${fmtMoney(d.indicated_premium_usd, 1)}</div></div>
                <div class="uf-stat-box"><div class="stat-label">Primary CAT Peril</div><div class="stat-value">${escapeHtml(d.primary_cat_peril || "—")}</div></div>
            </div>
        </div>

        ${overrideHtml}
        ${capacityNoteHtml}

        <div class="card" style="padding:14px 18px;">
            <div class="eyebrow">UW score</div>
            <div style="display:grid; grid-template-columns:7fr 3fr; gap:20px; align-items:start;">
                <div>
                    <div style="display:grid; grid-template-columns:420px 1fr; align-items:center; gap:14px;">
                        <div class="gauge-row" style="flex-wrap:nowrap; overflow-x:auto;">
                            ${ringGaugeHtml(s.loss_propensity, "Loss Score", "0 best / 100 worst", "var(--bad)", 62)}
                            <div class="plus-sign">+</div>
                            ${ringGaugeHtml(s.appetite_capacity, "Appetite / Capacity", "higher = better fit", "var(--good)", 62)}
                            <div class="plus-sign">+</div>
                            ${ringGaugeHtml(s.bind_propensity, "Bind Score", "higher = more likely", "var(--accent)", 62)}
                            <div class="equals-sign">=</div>
                        </div>
                        <div>
                            <div class="caption" style="font-size:0.76rem;">Composite UW Score</div>
                            <div style="display:flex; align-items:center; gap:10px;">
                                ${uwSuppressReason ? "" : `<span style="font-weight:800; font-size:1.7rem; color:var(--navy);">${fmtNum(s.uw_score)}</span>`}
                                ${tierChip(s.risk_tier)}
                            </div>
                            ${uwSuppressReason ? `<div class="caption" style="margin-top:4px;">${uwSuppressReason}</div>` : ""}
                        </div>
                    </div>
                    ${uwSuppressReason ? "" : `<div style="margin-top:14px; position:relative;">
                        <div style="display:flex; border-radius:6px; overflow:hidden;">${triageBar}</div>
                        <div style="position:absolute; left:${markerPct}%; top:-3px; width:3px; height:28px; background:var(--text-main);"></div>
                    </div>`}
                    <div class="caption" style="margin-top:6px;">Model recommends: <b>${escapeHtml(s.recommended_action)}</b></div>
                </div>
                <div>
                    <div class="sub-label" style="margin-top:0;">UW Score blend</div>
                    <div class="formula-box">
                        Loss: 0.50 &times; (100 &minus; ${fmtNum(s.loss_propensity)}) = ${d.calc_trail.uw_blend.loss_term}<br>
                        Appetite: 0.30 &times; ${fmtNum(s.appetite_capacity)} = ${d.calc_trail.uw_blend.appetite_term}<br>
                        Bind: 0.20 &times; ${fmtNum(s.bind_propensity)} = ${d.calc_trail.uw_blend.bind_term}<br>
                        <b>UW Score = ${d.calc_trail.uw_blend.total}</b>
                    </div>
                </div>
            </div>
        </div>

        <div class="eyebrow" style="margin:4px 0 2px;">Decision dimensions</div>
        <div class="decision-dim-row">
            <div class="card">
                <div class="uf-dim-header">
                    <div class="eyebrow">1 &middot; Appetite &amp; eligibility</div>
                    ${ufPill(`${d.decision_dimensions.appetite_eligibility.hard_rules_passed} of ${d.decision_dimensions.appetite_eligibility.hard_rules_total} hard rules pass`, hardAllPass ? "var(--good)" : "var(--bad)")}
                </div>
                <div class="caption" style="margin-bottom:6px;">Can we write it?</div>
                <div>${hardChecks}</div>
                <div class="sub-label" style="margin-top:8px;">Capacity / informational</div>
                ${softFactors}
                <div class="uf-dim-field" style="display:flex; justify-content:space-between; margin-top:10px; padding-top:8px; border-top:1px solid var(--card-border);">
                    <span>Appetite fit</span><b>${fmtNum(d.calc_trail.appetite_fit_after_hard_rule, 0)}</b>
                </div>
            </div>
            <div class="card">
                <div class="uf-dim-header">
                    <div class="eyebrow">2 &middot; Exposure &amp; capacity</div>
                    ${ufPill(apDecisionShort, apDecisionColor)}
                </div>
                <div class="caption" style="margin-bottom:6px;">Do we have room?</div>
                <div class="uf-dim-field">Industry: <b>${escapeHtml(exp.industry || "—")}</b></div>
                <div class="uf-dim-field">${fmtPct(exp.industry_concentration_share_pct)} of book TIV vs ${fmtPct(exp.concentration_cap_pct)} cap</div>
                <div class="uf-dim-field" style="margin-top:4px;">Regional treaty headroom: <b style="color:${headroomColor(exp.regional_treaty_headroom_pct)};">${fmtPct(exp.regional_treaty_headroom_pct)}</b></div>
                <div class="uf-dim-field">Country treaty headroom: <b style="color:${headroomColor(exp.country_treaty_headroom_pct)};">${fmtPct(exp.country_treaty_headroom_pct)}</b></div>
                <div class="uf-dim-field">State treaty headroom: <b style="color:${headroomColor(exp.state_treaty_headroom_pct)};">${fmtPct(exp.state_treaty_headroom_pct)}</b></div>
            </div>
            <div class="card">
                <div class="uf-dim-header">
                    <div class="eyebrow">3 &middot; Loss &amp; complexity</div>
                    ${ufPill(`Loss score ${fmtNum(s.loss_propensity, 0)} · loss tier ${s.loss_only_tier}`, TIER_COLORS[s.loss_only_tier])}
                </div>
                <div class="caption" style="margin-bottom:6px;">How risky is it?</div>
                <div class="uf-dim-field">Current LR (CAT-dampened): <b style="color:${lrColor(loss.current_loss_ratio_pct)};">${fmtPct(loss.current_loss_ratio_pct)}</b></div>
                <div class="uf-dim-field">Prior-period LR (5yr): <b style="color:${lrColor(loss.prior_period_loss_ratio_pct)};">${fmtPct(loss.prior_period_loss_ratio_pct)}</b></div>
                <div class="uf-dim-field">Hazard index: <b>${fmtNum(loss.hazard_index, 3)}</b></div>
                <div class="uf-dim-field">Predicted LR (model, uncapped): <b style="color:${lrColor(loss.predicted_loss_ratio_pct)};">${fmtPct(loss.predicted_loss_ratio_pct)}</b></div>
                ${loss.flags ? `<div class="caption" style="margin-top:6px;">Flags: <i>${escapeHtml(loss.flags)}</i></div>` : ""}
            </div>
            <div class="card">
                <div class="uf-dim-header">
                    <div class="eyebrow">4 &middot; Broker &amp; bind</div>
                    ${ufPill(`Bind ${fmtNum(bb.bind_score, 0)} · ${bindTier.label}`, bindTier.color)}
                </div>
                <div class="caption" style="margin-bottom:6px;">Will it convert?</div>
                <div class="uf-dim-field">Broker quality score: <b>${fmtNum(bb.broker_quality_score, 2)}</b></div>
                <div class="uf-dim-field">Hit/bind ratio: <b>${fmtPct(bb.hit_bind_ratio_pct)}</b></div>
                <div class="uf-dim-field">Competing quotes: <b>${fmtInt(bb.competing_quotes)}</b></div>
                <div class="uf-dim-field">Quote-to-bind days: <b>${fmtInt(bb.quote_to_bind_days)}</b></div>
                <div class="uf-dim-field">Renewal retention: <b>${fmtPct(bb.renewal_retention_pct)}</b></div>
            </div>
        </div>

        <div class="card">
            <div class="eyebrow">Why this score &mdash; drivers in UW points vs the typical submission</div>
            <div class="caption" style="margin-bottom:6px;">Exact linear split of the workbook formulas - positive pulls toward Decline, negative toward Preferred.</div>
            <div class="sub-label">Loss</div>${d.drivers.loss.length ? d.drivers.loss.map((dr) => divergingBarHtml(dr, "decreases")).join("") : `<div class="caption">No material drivers vs. book baseline.</div>`}
            <div class="sub-label" style="margin-top:8px;">Appetite</div>${d.drivers.appetite.length ? d.drivers.appetite.map((dr) => divergingBarHtml(dr, "increases")).join("") : `<div class="caption">Clean appetite profile.</div>`}
            <div class="sub-label" style="margin-top:8px;">Bind</div>${d.drivers.bind.length ? d.drivers.bind.map((dr) => divergingBarHtml(dr, "increases")).join("") : `<div class="caption">No material drivers vs. book baseline.</div>`}
            <div class="narrative-box" style="margin-top:10px;">${escapeHtml(d.narrative)}</div>
        </div>

        <div class="card">
            <div class="eyebrow">Evidence</div>
            <div class="btn-row" id="uf-evidence-tabs" style="margin-bottom:10px;">
                <button class="btn active-toggle" data-tab="claims">Claims history</button>
                <button class="btn" data-tab="quality">Data quality</button>
            </div>
            <div id="uf-evidence-claims">
                <div class="uf-dim-field">Total: <b>${fmtInt(d.claims.total)}</b> &middot; Open: <b>${fmtInt(d.claims.open)}</b> &middot; Settled: <b>${fmtInt(d.claims.settled)}</b></div>
                <div class="uf-dim-field">Last 3 yrs: <b>${fmtInt(d.claims.last_3yr)}</b> &middot; Claims-free years: <b>${fmtInt(d.claims.claims_free_years)}</b></div>
                <div class="uf-dim-field">Prior total incurred: <b>${fmtMoney(d.claims.prior_total_incurred_usd)}</b></div>
            </div>
            <div id="uf-evidence-quality" style="display:none;">
                <div class="uf-dim-field">Submission completeness: <b>${fmtPct(d.data_quality.completeness_pct)}</b></div>
                <div class="caption" style="margin-top:4px;">This barely varies across the real book (mean sits 90-92% regardless of segment) - a real, honestly-reported figure, just not a very differentiating one.</div>
            </div>
        </div>

        <div class="card">
            <div class="eyebrow">Next best actions</div>
            ${renderNextBestActions(d.drivers)}
        </div>

        <div class="card">
            <div class="eyebrow">Decision</div>
            <div class="btn-row" id="uf-decision-buttons" style="margin-bottom:8px;">
                <button class="btn" data-decision="Approve to quote">Approve to quote</button>
                <button class="btn" data-decision="Quote with conditions">Quote with conditions</button>
                <button class="btn" data-decision="Refer (capacity)">Refer (capacity)</button>
                <button class="btn" data-decision="Decline">Decline</button>
            </div>
            <label class="field-label">Reason code</label>
            <select id="uf-reason-code">
                <option>Agree with engine</option><option>Capacity constraint</option><option>Risk quality concern</option>
                <option>Pricing adequacy</option><option>Broker relationship</option><option>Other</option>
            </select>
            <div id="uf-decision-status" class="caption" style="margin-top:8px;"></div>
        </div>
    `;

    document.querySelectorAll("#uf-evidence-tabs button[data-tab]").forEach((btn) => btn.addEventListener("click", () => {
        if (btn.disabled) return;
        document.querySelectorAll("#uf-evidence-tabs button").forEach((b) => b.classList.remove("active-toggle"));
        btn.classList.add("active-toggle");
        document.getElementById("uf-evidence-claims").style.display = btn.dataset.tab === "claims" ? "block" : "none";
        document.getElementById("uf-evidence-quality").style.display = btn.dataset.tab === "quality" ? "block" : "none";
    }));

    document.querySelectorAll("#uf-decision-buttons button").forEach((btn) => btn.addEventListener("click", async () => {
        const reasonCode = document.getElementById("uf-reason-code").value;
        await apiPost(`/api/submission/${encodeURIComponent(id)}/review`, { decision: btn.dataset.decision, reason_code: reasonCode, status: "Reviewed" });
        document.getElementById("uf-decision-status").textContent = `Logged: ${btn.dataset.decision} (${reasonCode}) - session only, no persistence layer yet.`;
    }));
}
