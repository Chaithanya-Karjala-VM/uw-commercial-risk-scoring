async function initFeatureMap() {
    const root = document.getElementById("page-feature-map");
    const filters = await apiGet("/api/filters");
    // "All Lines" first, same as the Streamlit page's fmap.LOB_OPTIONS - the backend's
    // loss_features_for() already special-cases this string, so no extra endpoint work needed.
    const lobOptions = ["All Lines", ...filters.lobs];
    root.innerHTML = `
        <div class="card">
            <div class="caption" style="margin-bottom:10px;">Which real input fields feed each of the three scores - grouped by category, with a one-line reason for each.</div>
            <div class="filter-row">
                <div class="filter-item"><label class="field-label">Line of Business</label>
                    <select id="fm-lob" title="Bind and Appetite never change by line - only Loss Propensity's own column narrows to that line's real feature set.">
                        ${lobOptions.map((l) => `<option>${escapeHtml(l)}</option>`).join("")}
                    </select></div>
            </div>
            <div class="kpi-row" id="fm-kpis"></div>
            <table class="data-table fm-table">
                <thead><tr><th>Category</th><th>Field</th><th class="fm-check">Loss Propensity</th><th class="fm-check">Bind Propensity</th><th class="fm-check">Appetite / Capacity</th><th>Why</th></tr></thead>
                <tbody id="fm-rows"></tbody>
            </table>
        </div>
    `;
    document.getElementById("fm-lob").addEventListener("change", refreshFeatureMap);
    await refreshFeatureMap();
}

async function refreshFeatureMap() {
    const lob = document.getElementById("fm-lob").value;
    const data = await apiGet("/api/feature-map", { lob });
    document.getElementById("fm-kpis").innerHTML = `
        <div class="kpi-card"><div class="label">Loss Propensity</div><div class="value">${data.loss_field_count} fields</div></div>
        <div class="kpi-card"><div class="label">Bind Propensity</div><div class="value">${data.bind_field_count} fields</div></div>
        <div class="kpi-card"><div class="label">Appetite / Capacity</div><div class="value">${data.appetite_field_count} fields</div></div>
    `;
    const check = (b) => b ? `<span class="fm-dot fm-dot-yes">&#10003;</span>` : `<span class="fm-dot fm-dot-no"></span>`;
    document.getElementById("fm-rows").innerHTML = data.rows.map((r) => `
        <tr><td>${escapeHtml(r.Category)}</td><td><b>${escapeHtml(r.Field)}</b></td>
        <td class="fm-check">${check(r["Loss Propensity"])}</td><td class="fm-check">${check(r["Bind Propensity"])}</td><td class="fm-check">${check(r["Appetite / Capacity"])}</td>
        <td class="fm-why">${escapeHtml(r.Why)}</td></tr>`).join("");
}
