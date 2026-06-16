/* ============================================================
   BVL QUANT ENGINE — Institutional JavaScript Layer
   ============================================================ */

'use strict';

// ── GLOBAL STATE ──────────────────────────────────────────────
const state = {
    data: null,
    confidence: 0.95,
    capital: 3200,
    selectedAsset: null,
    loading: false,
};

// ── PLOTLY THEME ──────────────────────────────────────────────
const THEME = {
    bg: 'rgba(0,0,0,0)',
    plot: 'rgba(0,0,0,0)',
    gridColor: 'rgba(255,255,255,0.04)',
    zerolineColor: 'rgba(255,255,255,0.07)',
    fontColor: '#64748B',
    fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
    fontSize: 10,
    blue:   '#3B7EF7',
    green:  '#10B981',
    red:    '#F43F5E',
    orange: '#F59E0B',
    gold:   '#D4A843',
    purple: '#8B5CF6',
    teal:   '#14B8A6',
};

const baseLayout = () => ({
    paper_bgcolor: THEME.bg,
    plot_bgcolor: THEME.plot,
    font: { family: THEME.fontFamily, color: THEME.fontColor, size: THEME.fontSize },
    margin: { t: 20, r: 20, l: 55, b: 50 },
    xaxis: {
        gridcolor: THEME.gridColor,
        zerolinecolor: THEME.zerolineColor,
        linecolor: 'rgba(255,255,255,0.05)',
        tickfont: { family: THEME.fontFamily, size: 10, color: '#4A5A72' },
    },
    yaxis: {
        gridcolor: THEME.gridColor,
        zerolinecolor: THEME.zerolineColor,
        linecolor: 'rgba(255,255,255,0.05)',
        tickfont: { family: THEME.fontFamily, size: 10, color: '#4A5A72' },
    },
    legend: {
        font: { family: THEME.fontFamily, size: 10, color: '#8A9BB5' },
        bgcolor: 'rgba(8,12,20,0.7)',
        bordercolor: 'rgba(255,255,255,0.07)',
        borderwidth: 1,
        borderradius: 6,
    },
    hoverlabel: {
        bgcolor: '#111927',
        bordercolor: 'rgba(255,255,255,0.1)',
        font: { family: THEME.fontFamily, size: 11, color: '#E8EDF5' },
    },
});

const plotConfig = { responsive: true, displayModeBar: false };

// Palette for multi-series
const PALETTE = ['#3B7EF7','#10B981','#D4A843','#8B5CF6','#14B8A6','#F43F5E','#60A0FF','#34D399','#F0C060','#A78BFA'];

// ── FORMAT UTILS ──────────────────────────────────────────────
const fmt = {
    pct:  (v) => (v * 100).toFixed(2) + '%',
    pct4: (v) => (v * 100).toFixed(4) + '%',
    pen:  (v) => 'S/. ' + Math.abs(v).toLocaleString('es-PE', { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
    num:  (v) => v.toFixed(3),
    mono: (v, d=4) => v.toFixed(d),
};

// ── CLOCK ─────────────────────────────────────────────────────
function startClock() {
    const el = document.getElementById('clockDisplay');
    setInterval(() => {
        const now = new Date();
        el.textContent = now.toLocaleTimeString('en-US', { hour12: false, hour:'2-digit', minute:'2-digit', second:'2-digit' });
    }, 1000);
}

// ── LOADER ANIMATION ──────────────────────────────────────────
let loaderVal = 0;
const loaderMessages = [
    'Connecting to data pipeline...',
    'Downloading BVL market data...',
    'Standardizing to PEN...',
    'Applying liquidity filter...',
    'Fitting GJR-GARCH models (8 specs/asset)...',
    'Computing EWMA dynamic correlations...',
    'Running 10,000 correlated Monte Carlo paths...',
    'Minimizing CVaR via SLSQP...',
    'Building institutional dashboard...',
];
let msgIdx = 0;

function advanceLoader(pct, msg) {
    const bar = document.getElementById('loaderBar');
    const status = document.getElementById('loaderStatus');
    if (bar) bar.style.width = pct + '%';
    if (status && msg) status.textContent = msg;
}

function hideLoader() {
    const loader = document.getElementById('loader');
    if (loader) {
        loader.style.opacity = '0';
        loader.style.visibility = 'hidden';
        loader.style.pointerEvents = 'none';
    }
}

// ── NAVIGATION ────────────────────────────────────────────────
function initNav() {
    document.querySelectorAll('.nav-tab').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const panelId = 'panel-' + btn.dataset.panel;
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            const target = document.getElementById(panelId);
            if (target) {
                target.classList.add('active');
                // Re-render charts for active panel when switching
                if (state.data) {
                    if (btn.dataset.panel === 'overview') renderOverview();
                    if (btn.dataset.panel === 'risk') renderRisk();
                    if (btn.dataset.panel === 'montecarlo') renderMonteCarlo();
                    if (btn.dataset.panel === 'validation') loadValidationPanel();
                }
            }
        });
    });

    // Stress test button
    const stressBtn = document.getElementById('runStressBtn');
    if (stressBtn) stressBtn.addEventListener('click', runStressTest);
}

// ── CONTROL BAR ───────────────────────────────────────────────
function initControls() {
    // Segmented confidence
    document.querySelectorAll('.seg-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.confidence = parseFloat(btn.dataset.val);
        });
    });

    // Capital input
    const capitalEl = document.getElementById('capitalBase');
    capitalEl.addEventListener('change', () => {
        state.capital = parseFloat(capitalEl.value) || 3200;
    });

    // Run button
    document.getElementById('runBtn').addEventListener('click', runSimulation);
}

// ── SET STATUS ────────────────────────────────────────────────
function setStatus(type, text) {
    const pill = document.getElementById('statusPill');
    const textEl = document.getElementById('statusText');
    pill.className = 'status-pill ' + type;
    textEl.textContent = text;
}

// ── FETCH & SIMULATE ──────────────────────────────────────────
async function runSimulation() {
    if (state.loading) return;
    state.loading = true;
    const btn = document.getElementById('runBtn');
    btn.disabled = true;
    btn.innerHTML = `<span class="spin-icon">⟳</span> COMPUTING...`;
    setStatus('loading', 'COMPUTING');

    try {
        const res = await fetch('/api/calculate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                confidence_level: state.confidence,
                capital_base: state.capital,
            }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        state.data = await res.json();

        renderAll();
        updateLastRun();
        setStatus('', 'LIVE');
    } catch (err) {
        console.error('Simulation error:', err);
        setStatus('loading', 'ERROR');
        showErrorBanner(err.message);
    } finally {
        state.loading = false;
        btn.disabled = false;
        btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> RUN SIMULATION`;
    }
}

function updateLastRun() {
    const el = document.getElementById('lastUpdated');
    if (el) el.textContent = 'Last run: ' + new Date().toLocaleTimeString('en-US', { hour12: false });
}

function showErrorBanner(msg) {
    const existing = document.getElementById('errorBanner');
    if (existing) existing.remove();
    const banner = document.createElement('div');
    banner.id = 'errorBanner';
    banner.style.cssText = 'position:fixed;bottom:24px;right:24px;background:#1a0a0a;border:1px solid rgba(239,68,68,0.4);border-radius:8px;padding:14px 20px;font-family:JetBrains Mono,monospace;font-size:11px;color:#EF4444;max-width:360px;z-index:999;';
    banner.innerHTML = `<strong>ENGINE FAULT</strong><br>${msg}<br><small style="color:#4A5A72;margin-top:6px;display:block;">Check terminal for traceback.</small>`;
    document.body.appendChild(banner);
    setTimeout(() => banner.remove(), 8000);
}

// ── RENDER ALL ────────────────────────────────────────────────
function renderAll() {
    renderKPIs();
    renderOverview();
    renderRisk();
    renderMonteCarlo();
}

// ── KPIs ──────────────────────────────────────────────────────
function renderKPIs() {
    const d = state.data;
    const m = d.metrics;
    const capital = state.capital;
    const conf = state.confidence;

    // Capital — always gold
    document.getElementById('kpi-v-capital').textContent = fmt.pen(capital);
    document.getElementById('kpi-v-capital').className = 'kpi-value gold-val';

    // Return
    const retEl = document.getElementById('kpi-v-return');
    retEl.textContent = fmt.pct(m.expected_return);
    if (m.expected_return >= 0) {
        retEl.className = 'kpi-value positive';
        document.getElementById('kpi-b-return').textContent = '▲ GAIN';
        document.getElementById('kpi-b-return').className = 'kpi-badge pos';
    } else {
        retEl.className = 'kpi-value negative';
        document.getElementById('kpi-b-return').textContent = '▼ LOSS';
        document.getElementById('kpi-b-return').className = 'kpi-badge neg';
    }

    // Volatility — already annualized by the backend
    const annVol = m.expected_volatility;
    document.getElementById('kpi-v-vol').textContent = fmt.pct(annVol);
    document.getElementById('kpi-v-vol').className = 'kpi-value gradient';

    // VaR
    const varPEN = m.var * capital;
    document.getElementById('kpi-v-var').textContent = fmt.pen(varPEN);
    document.getElementById('kpi-v-var').className = 'kpi-value danger';
    document.getElementById('kpi-b-var').textContent = (conf * 100).toFixed(0) + '%';

    // CVaR
    const cvarPEN = m.cvar * capital;
    document.getElementById('kpi-v-cvar').textContent = fmt.pen(cvarPEN);
    document.getElementById('kpi-v-cvar').className = 'kpi-value danger';
    document.getElementById('kpi-b-cvar').textContent = (conf * 100).toFixed(0) + '%';

    // Sharpe
    const sharpeEl = document.getElementById('kpi-v-sharpe');
    sharpeEl.textContent = fmt.num(m.sharpe_ratio);
    if (m.sharpe_ratio >= 1) {
        sharpeEl.className = 'kpi-value positive';
        document.getElementById('kpi-b-sharpe').textContent = 'STRONG';
        document.getElementById('kpi-b-sharpe').className = 'kpi-badge pos';
    } else if (m.sharpe_ratio >= 0) {
        sharpeEl.className = 'kpi-value gradient';
        document.getElementById('kpi-b-sharpe').textContent = 'FAIR';
        document.getElementById('kpi-b-sharpe').className = 'kpi-badge neutral';
    } else {
        sharpeEl.className = 'kpi-value negative';
        document.getElementById('kpi-b-sharpe').textContent = 'WEAK';
        document.getElementById('kpi-b-sharpe').className = 'kpi-badge neg';
    }
}

// ── OVERVIEW PANEL ────────────────────────────────────────────
function renderOverview() {
    const d = state.data;
    if (!d) return;
    renderDonutAndTable(d.assets, d.weights, state.capital);
    renderScatterPlot(d);
    renderCorrHeatmap(d);
}

function renderDonutAndTable(assets, weights, capital) {
    // Only show assets with meaningful weight
    const threshold = 0.0005;
    const filtered = assets.map((a, i) => ({ asset: a.replace('.LM',''), weight: weights[i], raw: a }))
        .filter(x => x.weight > threshold)
        .sort((a, b) => b.weight - a.weight);

    // Donut
    const donut = [{
        type: 'pie',
        labels: filtered.map(x => x.asset),
        values: filtered.map(x => x.weight),
        hole: 0.65,
        marker: {
            colors: PALETTE,
            line: { color: '#080C14', width: 2 },
        },
        textinfo: 'none',
        hovertemplate: '<b>%{label}</b><br>Weight: %{value:.2%}<br>PEN: S/. %{customdata:,.2f}<extra></extra>',
        customdata: filtered.map(x => x.weight * capital),
        direction: 'clockwise',
        sort: false,
    }];

    const donutLayout = {
        ...baseLayout(),
        margin: { t: 10, r: 10, l: 10, b: 10 },
        showlegend: false,
        annotations: [{
            text: `<b>${filtered.length}</b><br><span style="font-size:9px">ASSETS</span>`,
            x: 0.5, y: 0.5, xref: 'paper', yref: 'paper',
            showarrow: false,
            font: { family: THEME.fontFamily, size: 14, color: '#E8EDF5' },
        }],
    };
    Plotly.newPlot('chart-donut', donut, donutLayout, plotConfig);

    // Weights table
    const tbody = document.getElementById('weightsTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';
    filtered.forEach((item, i) => {
        const pct = (item.weight * 100).toFixed(2);
        const pen = (item.weight * capital).toFixed(2);
        const barPct = (item.weight / 0.15 * 100).toFixed(1);
        const color = PALETTE[i % PALETTE.length];
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color};margin-right:7px;"></span>${item.asset}</td>
            <td>${pct}%</td>
            <td>${fmt.pen(parseFloat(pen))}</td>
            <td class="wt-bar-cell">
                <div class="wt-bar-bg">
                    <div class="wt-bar-fill" style="width:${barPct}%;background:${color};"></div>
                </div>
            </td>`;
        tbody.appendChild(tr);
    });
}

function renderScatterPlot(d) {
    const assets = d.assets;
    const weights = d.weights;
    const timeseries = d.timeseries;

    // Compute avg volatility and avg returns per asset
    const avgVols = [], avgRets = [], labels = [], colors = [];
    assets.forEach((asset, i) => {
        const vols = timeseries.volatility[asset];
        const rets = timeseries.log_returns[asset];
        if (!vols || !rets) return;
        const cleanVols = vols.filter(v => v !== null && !isNaN(v));
        const cleanRets = rets.filter(r => r !== null && !isNaN(r));
        if (cleanVols.length === 0 || cleanRets.length === 0) return;
        const avgV = cleanVols.reduce((a,b)=>a+b,0)/cleanVols.length;
        const avgR = cleanRets.reduce((a,b)=>a+b,0)/cleanRets.length * 252;
        avgVols.push(avgV * Math.sqrt(252)); // annualized
        avgRets.push(avgR);
        labels.push(asset.replace('.LM',''));
        const sz = 8 + weights[i] * 200;
        colors.push(PALETTE[i % PALETTE.length]);
    });

    const trace = {
        type: 'scatter',
        mode: 'markers+text',
        x: avgVols,
        y: avgRets,
        text: labels,
        textposition: 'top center',
        textfont: { family: THEME.fontFamily, size: 9, color: '#8A9BB5' },
        marker: {
            color: colors,
            size: assets.map((a, i) => 8 + weights[i] * 200),
            opacity: 0.85,
            line: { color: 'rgba(255,255,255,0.1)', width: 1 },
        },
        hovertemplate: '<b>%{text}</b><br>Ann. Vol: %{x:.2%}<br>Ann. Return: %{y:.2%}<extra></extra>',
    };

    const layout = {
        ...baseLayout(),
        xaxis: { ...baseLayout().xaxis, title: { text: 'ANNUALIZED VOLATILITY (GARCH)', font: { size: 9, color: '#4A5A72' }, standoff: 10 }, tickformat: '.1%' },
        yaxis: { ...baseLayout().yaxis, title: { text: 'ANN. RETURN', font: { size: 9, color: '#4A5A72' }, standoff: 10 }, tickformat: '.1%' },
        shapes: [{
            type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 0, y1: 0,
            line: { color: 'rgba(255,255,255,0.1)', width: 1, dash: 'dot' }
        }],
    };
    Plotly.newPlot('chart-scatter', [trace], layout, plotConfig);
}

function renderCorrHeatmap(d) {
    const assets = d.assets;
    const logReturns = d.timeseries.log_returns;
    const shortNames = assets.map(a => a.replace('.LM',''));
    const n = assets.length;

    // Build correlation matrix
    const matrix = [];
    const allRets = assets.map(a => logReturns[a]?.filter(v => v !== null && !isNaN(v)) || []);
    const minLen = Math.min(...allRets.map(r => r.length));

    for (let i = 0; i < n; i++) {
        const row = [];
        for (let j = 0; j < n; j++) {
            const ri = allRets[i].slice(-minLen);
            const rj = allRets[j].slice(-minLen);
            row.push(pearsonCorr(ri, rj));
        }
        matrix.push(row);
    }

    const texts = matrix.map(row => row.map(v => v.toFixed(2)));

    const trace = {
        type: 'heatmap',
        z: matrix,
        x: shortNames, y: shortNames,
        text: texts,
        texttemplate: '%{text}',
        textfont: { family: THEME.fontFamily, size: 9 },
        colorscale: [
            [0.0,  '#B91C1C'],
            [0.25, '#7C1D1D'],
            [0.5,  '#111927'],
            [0.75, '#1a3060'],
            [1.0,  '#1d4ed8'],
        ],
        zmid: 0, zmin: -1, zmax: 1,
        showscale: true,
        colorbar: {
            thickness: 12, len: 0.8,
            tickfont: { family: THEME.fontFamily, size: 9, color: '#4A5A72' },
            tickformat: '.1f',
        },
        hovertemplate: '<b>%{x} ↔ %{y}</b><br>Correlation: %{z:.4f}<extra></extra>',
    };

    const layout = {
        ...baseLayout(),
        margin: { t: 10, r: 80, l: 80, b: 70 },
        xaxis: { ...baseLayout().xaxis, tickfont: { family: THEME.fontFamily, size: 9, color: '#8A9BB5' } },
        yaxis: { ...baseLayout().yaxis, tickfont: { family: THEME.fontFamily, size: 9, color: '#8A9BB5' }, autorange: 'reversed' },
    };
    Plotly.newPlot('chart-corr', [trace], layout, plotConfig);
}

function pearsonCorr(x, y) {
    const n = Math.min(x.length, y.length);
    if (n < 2) return 0;
    const mx = x.slice(0,n).reduce((a,b)=>a+b,0)/n;
    const my = y.slice(0,n).reduce((a,b)=>a+b,0)/n;
    let num = 0, dx = 0, dy = 0;
    for (let i=0; i<n; i++) {
        const ex = x[i]-mx, ey = y[i]-my;
        num += ex*ey; dx += ex*ex; dy += ey*ey;
    }
    return (dx===0||dy===0) ? 0 : num/Math.sqrt(dx*dy);
}

// ── RISK ANALYTICS PANEL ──────────────────────────────────────
function renderRisk() {
    const d = state.data;
    if (!d) return;
    renderAssetList(d.assets, d.weights);
    renderDrawdown(d);
    // Select first asset by default
    if (d.assets.length > 0 && !state.selectedAsset) {
        selectAsset(d.assets[0]);
    } else if (state.selectedAsset) {
        renderGARCH(state.selectedAsset);
    }
}

function renderAssetList(assets, weights) {
    const container = document.getElementById('assetList');
    if (!container) return;
    container.innerHTML = '';
    assets.forEach((asset, i) => {
        const div = document.createElement('div');
        div.className = 'asset-item' + (asset === state.selectedAsset ? ' active' : '');
        div.innerHTML = `
            <span>${asset.replace('.LM','')}</span>
            <span class="asset-item-label">${(weights[i]*100).toFixed(2)}%</span>
        `;
        div.addEventListener('click', () => selectAsset(asset));
        container.appendChild(div);
    });
}

function selectAsset(asset) {
    state.selectedAsset = asset;
    document.querySelectorAll('.asset-item').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.asset-item').forEach(el => {
        if (el.querySelector('span')?.textContent === asset.replace('.LM','')) el.classList.add('active');
    });
    document.getElementById('garchAssetLabel').textContent = asset;
    renderGARCH(asset);
    renderGARCHParams(asset);
}

function renderGARCH(asset) {
    const d = state.data;
    if (!d || !d.timeseries) return;
    const dates = d.timeseries.dates;
    const rets = d.timeseries.log_returns[asset];
    const vols = d.timeseries.volatility[asset];
    if (!rets || !vols) return;

    const retTrace = {
        x: dates, y: rets,
        type: 'scatter', mode: 'lines',
        name: 'Log Return',
        line: { color: 'rgba(138,155,181,0.3)', width: 1 },
        fill: 'tozeroy',
        fillcolor: 'rgba(45,126,247,0.03)',
        yaxis: 'y',
        hovertemplate: '%{x}<br>Return: %{y:.4f}<extra>Return</extra>',
    };

    const volTrace = {
        x: dates, y: vols,
        type: 'scatter', mode: 'lines',
        name: 'σ GARCH(1,1)',
        line: { color: THEME.blue, width: 2 },
        yaxis: 'y',
        hovertemplate: '%{x}<br>Cond. Vol: %{y:.4f}<extra>GARCH Vol</extra>',
    };

    const negVolTrace = {
        x: dates, y: vols.map(v => v !== null ? -v : null),
        type: 'scatter', mode: 'lines',
        name: '−σ GARCH(1,1)',
        line: { color: THEME.blue, width: 2, dash: 'dot' },
        yaxis: 'y',
        showlegend: false,
        hoverinfo: 'skip',
    };

    const layout = {
        ...baseLayout(),
        margin: { t: 10, r: 20, l: 60, b: 60 },
        hovermode: 'x unified',
        xaxis: { ...baseLayout().xaxis, showgrid: false, rangeslider: { visible: true, thickness: 0.04, bgcolor: 'rgba(13,20,32,0.5)', bordercolor: 'rgba(255,255,255,0.05)' } },
        yaxis: { ...baseLayout().yaxis, tickformat: '.4f' },
        legend: { ...baseLayout().legend, orientation: 'h', y: 1.05, x: 0 },
    };
    Plotly.newPlot('chart-garch', [retTrace, volTrace, negVolTrace], layout, plotConfig);
}

function renderGARCHParams(asset) {
    const d = state.data;
    const garchInfo = d.garch_params?.[asset];
    if (!garchInfo) return;
    const grid = document.getElementById('paramsGrid');
    if (!grid) return;
    const header = document.getElementById('garchParamsHeader');
    if (header) {
        header.innerHTML = `Best Fit: <strong style="color: #F0C040">${garchInfo.dist ? garchInfo.dist.toUpperCase() : 'T'}</strong> &nbsp;|&nbsp; BIC: ${garchInfo.bic ? garchInfo.bic.toFixed(1) : '--'}`;
    }

    const params = [
        { label: 'μ (MU)', val: garchInfo.mu, pval: null },
        { label: 'ω (OMEGA)', val: garchInfo.omega, pval: null },
        { label: 'α ARCH', val: garchInfo.alpha, pval: garchInfo.alpha_pval },
        { label: 'β GARCH', val: garchInfo.beta, pval: garchInfo.beta_pval },
    ];
    if (garchInfo.nu !== null && garchInfo.nu !== undefined) {
        params.push({ label: 'ν (df)', val: garchInfo.nu, pval: null });
    }
    if (garchInfo.lam !== null && garchInfo.lam !== undefined) {
        params.push({ label: 'λ (skew)', val: garchInfo.lam, pval: null });
    }
    grid.innerHTML = params.map(p => {
        const pvalHtml = p.pval !== null ?
            `<div class="param-pval ${p.pval < 0.05 ? 'sig' : 'insig'}">p=${p.pval.toFixed(4)} ${p.pval < 0.05 ? '✓' : '✗'}</div>` : '';
        return `<div class="param-item">
            <div class="param-label">${p.label}</div>
            <div class="param-val">${p.val.toFixed(5)}</div>
            ${pvalHtml}
        </div>`;
    }).join('');
}

function renderDrawdown(d) {
    const { assets, weights, timeseries } = d;
    const dates = timeseries.dates;
    const n = dates.length;

    // Compute portfolio daily returns
    let portRets = new Array(n).fill(0);
    assets.forEach((asset, i) => {
        const rets = timeseries.log_returns[asset];
        if (!rets) return;
        for (let t = 0; t < Math.min(n, rets.length); t++) {
            if (rets[t] !== null && !isNaN(rets[t])) {
                portRets[t] += weights[i] * rets[t];
            }
        }
    });

    // Compute drawdown
    let peak = 1, nav = 1;
    const drawdowns = portRets.map(r => {
        nav *= (1 + r);
        if (nav > peak) peak = nav;
        return (nav - peak) / peak;
    });

    const trace = {
        x: dates, y: drawdowns,
        type: 'scatter', mode: 'lines',
        name: 'Drawdown',
        line: { color: THEME.red, width: 1.5 },
        fill: 'tozeroy',
        fillcolor: 'rgba(239,68,68,0.08)',
        hovertemplate: '%{x}<br>Drawdown: %{y:.2%}<extra></extra>',
    };

    const layout = {
        ...baseLayout(),
        margin: { t: 10, r: 20, l: 65, b: 50 },
        yaxis: { ...baseLayout().yaxis, tickformat: '.1%' },
        xaxis: { ...baseLayout().xaxis, showgrid: false },
    };
    Plotly.newPlot('chart-drawdown', [trace], layout, plotConfig);
}

// ── MONTE CARLO PANEL ─────────────────────────────────────────
function renderMonteCarlo() {
    const d = state.data;
    if (!d) return;
    renderMCHistogram(d);
    renderMCPaths(d);
    renderRiskDecomp(d);
}

function renderMCHistogram(d) {
    const portRets = d.monte_carlo.portfolio_returns;
    const conf = state.confidence;
    const capital = state.capital;

    const losses = portRets.map(r => -r);
    const var95 = quantile(losses, conf);
    const cvarVal = losses.filter(l => l >= var95).reduce((a,b)=>a+b,0) / losses.filter(l => l >= var95).length;
    const values = portRets.map(r => capital * (1 + r));
    const varValue = capital * (1 - var95);

    // Update mc meta
    const meta = document.getElementById('mcMeta');
    if (meta) meta.textContent = `VaR ${(conf*100).toFixed(0)}%: ${fmt.pen(varValue)} · CVaR: ${fmt.pen(capital*(1-cvarVal))} · Capital: ${fmt.pen(capital)}`;

    const safe   = values.filter(v => v >= varValue);
    const tail   = values.filter(v => v < varValue);
    const mean   = values.reduce((a,b)=>a+b,0)/values.length;

    const traces = [
        {
            x: safe, type: 'histogram', name: 'Normal Scenarios',
            nbinsx: 120,
            marker: { color: 'rgba(45,126,247,0.4)', line: { color: 'rgba(45,126,247,0.6)', width: 0.5 } },
            opacity: 0.85,
        },
        {
            x: tail, type: 'histogram', name: 'Tail Risk (CVaR Zone)',
            nbinsx: 30,
            marker: { color: 'rgba(239,68,68,0.6)', line: { color: THEME.red, width: 0.5 } },
            opacity: 0.95,
        },
    ];

    const layout = {
        ...baseLayout(),
        barmode: 'overlay',
        margin: { t: 20, r: 20, l: 70, b: 60 },
        xaxis: { ...baseLayout().xaxis, title: { text: 'PORTFOLIO VALUE AT 20D (PEN)', font: { size: 9, color: '#4A5A72' }, standoff: 10 },
            tickformat: ',.0f', tickprefix: 'S/. ' },
        yaxis: { ...baseLayout().yaxis, title: { text: 'FREQUENCY', font: { size: 9, color: '#4A5A72' }, standoff: 10 } },
        shapes: [
            { type:'line', x0:varValue, x1:varValue, y0:0, y1:1, yref:'paper',
              line: { color: '#F59E0B', width: 2, dash: 'dash' } },
            { type:'line', x0:capital*(1-cvarVal), x1:capital*(1-cvarVal), y0:0, y1:1, yref:'paper',
              line: { color: THEME.red, width: 2, dash: 'dot' } },
            { type:'line', x0:mean, x1:mean, y0:0, y1:1, yref:'paper',
              line: { color: THEME.green, width: 1.5, dash: 'dash' } },
            { type:'line', x0:capital, x1:capital, y0:0, y1:1, yref:'paper',
              line: { color: 'rgba(255,255,255,0.2)', width: 1.5 } },
        ],
        annotations: [
            { x: varValue, y: 1, yref:'paper', xanchor:'right', text: `VaR ${(conf*100).toFixed(0)}%`, font:{color:'#F59E0B',size:9,family:THEME.fontFamily}, showarrow:false, bgcolor:'rgba(245,158,11,0.1)', bordercolor:'rgba(245,158,11,0.3)', borderwidth:1, borderpad:4, xshift:-6 },
            { x: capital*(1-cvarVal), y: 0.85, yref:'paper', xanchor:'right', text: `CVaR`, font:{color:THEME.red,size:9,family:THEME.fontFamily}, showarrow:false, bgcolor:'rgba(239,68,68,0.1)', bordercolor:'rgba(239,68,68,0.3)', borderwidth:1, borderpad:4, xshift:-6 },
            { x: mean, y: 0.7, yref:'paper', xanchor:'left', text: `Mean`, font:{color:THEME.green,size:9,family:THEME.fontFamily}, showarrow:false, bgcolor:'rgba(34,197,94,0.08)', bordercolor:'rgba(34,197,94,0.2)', borderwidth:1, borderpad:4, xshift:6 },
            { x: capital, y: 0.55, yref:'paper', xanchor:'left', text: `Initial`, font:{color:'rgba(255,255,255,0.4)',size:9,family:THEME.fontFamily}, showarrow:false, xshift:6 },
        ],
    };
    Plotly.newPlot('chart-mc-hist', traces, layout, plotConfig);
}

function renderMCPaths(d) {
    const portRets = d.monte_carlo.portfolio_returns;
    const capital = state.capital;
    const values = portRets.map(r => capital * (1 + r)).sort((a,b)=>a-b);
    const n = values.length;

    // Percentile buckets
    const buckets = [
        { pct: 1,  label: 'Worst 1%',    color: THEME.red },
        { pct: 5,  label: 'Worst 5%',    color: '#F97316' },
        { pct: 25, label: '25th Pct',    color: THEME.orange },
        { pct: 50, label: 'Median',       color: THEME.blue },
        { pct: 75, label: '75th Pct',     color: THEME.teal },
        { pct: 95, label: 'Best 5%',      color: THEME.green },
    ];

    // Bar chart of scenario distribution
    const labels = ['P1', 'P5', 'P25', 'Median', 'P75', 'P95', 'P99'];
    const pctValues = [1,5,25,50,75,95,99].map(p => quantile(values, p/100));
    const colors = [THEME.red, '#F97316', THEME.orange, THEME.blue, THEME.teal, THEME.green, '#34D399'];

    const trace = {
        type: 'bar',
        x: labels,
        y: pctValues,
        marker: {
            color: colors,
            opacity: 0.85,
            line: { color: 'rgba(255,255,255,0.08)', width: 1 },
        },
        hovertemplate: '<b>%{x}</b><br>Portfolio Value: S/. %{y:,.2f}<extra></extra>',
    };

    // Reference line
    const layout = {
        ...baseLayout(),
        margin: { t: 10, r: 20, l: 70, b: 50 },
        xaxis: { ...baseLayout().xaxis, showgrid: false },
        yaxis: { ...baseLayout().yaxis, tickformat: ',.0f', tickprefix: 'S/. ' },
        shapes: [{
            type: 'line', x0: -0.5, x1: labels.length - 0.5, y0: capital, y1: capital, xref: 'x', yref: 'y',
            line: { color: 'rgba(255,255,255,0.2)', width: 1, dash: 'dot' }
        }],
    };
    Plotly.newPlot('chart-mc-paths', [trace], layout, plotConfig);
}

function renderRiskDecomp(d) {
    const { assets, weights } = d;
    const portRets = d.monte_carlo.portfolio_returns;
    const conf = state.confidence;

    const losses = portRets.map(r => -r);
    const var95 = quantile(losses, conf);

    // Marginal CVaR: simplified Euler allocation
    // For each asset, compute contribution as weight * avg return in tail scenarios
    const tailIdx = losses.map((l,i) => l >= var95 ? i : -1).filter(i=>i>=0);
    const mcRets = d.monte_carlo.portfolio_returns;
    // Approximate individual scenario returns via correlation structure
    const contribs = assets.map((asset, ai) => {
        // Weight-scaled contribution
        const logRets = d.timeseries.log_returns[asset];
        if (!logRets) return { asset: asset.replace('.LM',''), contrib: 0 };
        // Use historical avg loss (rough approximation)
        const n = logRets.filter(v=>v!==null&&!isNaN(v)).length;
        const tail = Math.floor(n * (1-conf));
        const sorted = logRets.filter(v=>v!==null&&!isNaN(v)).sort((a,b)=>a-b);
        const avgTailLoss = -sorted.slice(0, tail).reduce((a,b)=>a+b,0) / (tail||1);
        return { asset: asset.replace('.LM',''), contrib: weights[ai] * avgTailLoss };
    }).sort((a,b) => b.contrib - a.contrib);

    const trace = {
        type: 'bar',
        x: contribs.map(c => c.asset),
        y: contribs.map(c => c.contrib),
        marker: {
            color: contribs.map((c,i) => PALETTE[i % PALETTE.length]),
            opacity: 0.85,
            line: { color: 'rgba(255,255,255,0.08)', width: 1 },
        },
        hovertemplate: '<b>%{x}</b><br>Marginal CVaR Contrib: %{y:.4%}<extra></extra>',
    };

    const layout = {
        ...baseLayout(),
        margin: { t: 10, r: 20, l: 65, b: 60 },
        xaxis: { ...baseLayout().xaxis, showgrid: false, tickangle: -30 },
        yaxis: { ...baseLayout().yaxis, tickformat: '.2%' },
    };
    Plotly.newPlot('chart-mc-risk', [trace], layout, plotConfig);
}

// ── UTILS ─────────────────────────────────────────────────────
function quantile(sorted, q) {
    const arr = [...sorted].sort((a, b) => a - b);
    const pos = (arr.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    if (base + 1 < arr.length) return arr[base] + rest * (arr[base + 1] - arr[base]);
    return arr[base];
}

// ── INTRO ANIMATION ───────────────────────────────────────────
function animateIn() {
    // Force app visible immediately — no opacity tricks
    document.getElementById('app').style.opacity = '1';
    // Force all KPI cards to full opacity before any animation
    document.querySelectorAll('.kpi-card').forEach(el => el.style.opacity = '1');
    // Subtle entrance animations — position only, never touch opacity
    if (typeof gsap !== 'undefined') {
        gsap.from('.topbar',     { y: -16, duration: 0.4, delay: 0.05 });
        gsap.from('.control-bar', { y: -8, duration: 0.35, delay: 0.1 });
        gsap.from('.kpi-card',   { y: 10, duration: 0.4, stagger: 0.05, delay: 0.15, ease: 'power2.out' });
        gsap.from('.chart-card', { y: 14, duration: 0.45, stagger: 0.08, delay: 0.3, ease: 'power2.out' });
    }
}

// ── BOOT ──────────────────────────────────────────────────────
async function boot() {
    startClock();
    initNav();
    initMobileNav();
    initControls();

    // Simulate loading steps while fetching
    const steps = loaderMessages.length;
    let step = 0;
    const interval = setInterval(() => {
        if (step < steps - 1) {
            step++;
            advanceLoader(Math.round(step / steps * 85), loaderMessages[step]);
        }
    }, 900);

    try {
        const res = await fetch('/api/calculate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ confidence_level: 0.95, capital_base: state.capital }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        state.data = await res.json();
        clearInterval(interval);
        advanceLoader(100, 'Dashboard ready.');
        setTimeout(() => {
            hideLoader();
            animateIn();
            renderAll();
            updateLastRun();
        }, 400);
    } catch (err) {
        clearInterval(interval);
        const status = document.getElementById('loaderStatus');
        if (status) status.textContent = 'ERROR: ' + err.message + ' — Is api.py running?';
        status.style.color = '#EF4444';
        setTimeout(() => {
            hideLoader();
            document.getElementById('app').style.opacity = '1';
            showErrorBanner(err.message);
        }, 2000);
    }
}

// ── VALIDATION & STRESS PANEL ─────────────────────────────────
let _btData = null;
let _btSelectedAsset = null;

async function loadValidationPanel() {
    // Load EWMA correlation
    try {
        const corrResp = await fetch('/api/dynamic-correlation');
        if (corrResp.ok) {
            const corrData = await corrResp.json();
            renderEWMAHeatmap(corrData);
        }
    } catch(e) { console.warn('EWMA corr failed:', e); }

    // Load backtesting (can take time — show loading)
    const btChart = document.getElementById('chart-backtest');
    if (btChart) btChart.innerHTML = '<div style="color:#8A9BB5; text-align:center; padding:40px; font-size:12px;">⏳ Running walk-forward VaR backtest... (may take a few minutes)</div>';
    try {
        const btResp = await fetch(`/api/backtest?confidence=${state.confidence}`);
        if (btResp.ok) {
            _btData = await btResp.json();
            if (btChart) btChart.innerHTML = ''; // Clear loading text before rendering
            renderBacktestPanel(_btData);
        } else {
            if (btChart) btChart.innerHTML = '<div style="color:#F43F5E; text-align:center; padding:40px; font-size:12px;">Error loading backtest data.</div>';
        }
    } catch(e) {
        if (btChart) btChart.innerHTML = `<div style="color:#F43F5E; text-align:center; padding:40px; font-size:12px;">Backtest error: ${e.message}</div>`;
    }
}

function renderBacktestPanel(btData) {
    const assets = Object.keys(btData.assets || {});
    if (assets.length === 0) return;

    // Asset selector buttons
    const selector = document.getElementById('btAssetSelector');
    if (selector) {
        selector.innerHTML = assets.map((a, i) =>
            `<button class="seg-btn ${i===0?'active':''}" data-asset="${a}" onclick="selectBtAsset('${a}')">${a.replace('.LM','')}</button>`
        ).join('');
    }

    _btSelectedAsset = assets[0];
    renderBacktestChart(_btSelectedAsset, btData.assets);
    renderTrafficLight(_btSelectedAsset, btData.assets);
    renderBtStatsTable(_btSelectedAsset, btData.assets);
}

function selectBtAsset(ticker) {
    _btSelectedAsset = ticker;
    document.querySelectorAll('#btAssetSelector .seg-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.asset === ticker);
    });
    if (_btData) {
        renderBacktestChart(ticker, _btData.assets);
        renderTrafficLight(ticker, _btData.assets);
        renderBtStatsTable(ticker, _btData.assets);
    }
}

function renderBacktestChart(ticker, assetsData) {
    const d = assetsData[ticker];
    if (!d) return;

    const dates = d.dates;
    const varEst = d.var_estimates.map(v => v * 100);  // to %
    const actRet = d.actual_returns.map(r => r * 100);
    const viols = d.violations;

    // Violation markers
    const violDates = dates.filter((_, i) => viols[i] === 1);
    const violRets  = actRet.filter((_, i) => viols[i] === 1);

    const traces = [
        {
            x: dates, y: actRet,
            type: 'scatter', mode: 'lines',
            name: 'Retorno Real',
            line: { color: THEME.blue, width: 1 },
        },
        {
            x: dates, y: varEst.map(v => -v),
            type: 'scatter', mode: 'lines',
            name: 'VaR (−)',
            line: { color: THEME.orange, width: 1.5, dash: 'dot' },
        },
        {
            x: violDates, y: violRets,
            type: 'scatter', mode: 'markers',
            name: 'Violación',
            marker: { color: THEME.red, size: 7, symbol: 'circle' },
        },
    ];

    const layout = {
        ...baseLayout(),
        hovermode: 'x unified',
        xaxis: { ...baseLayout().xaxis, title: '' },
        yaxis: { ...baseLayout().yaxis, tickformat: '.2f', ticksuffix: '%', title: 'Retorno %' },
        legend: { ...baseLayout().legend, orientation: 'h', y: 1.08, x: 0 },
        shapes: [{
            type: 'line', x0: dates[0], x1: dates[dates.length-1],
            y0: 0, y1: 0,
            line: { color: 'rgba(255,255,255,0.1)', width: 1 }
        }],
        title: { text: `${ticker} — Walk-Forward VaR Backtest (${d.kupiec.n_violations} violaciones / ${d.kupiec.n_obs} días)`, font: { size: 11, color: '#8A9BB5' }, x: 0.01 }
    };

    Plotly.newPlot('chart-backtest', traces, layout, plotConfig);
}

function renderTrafficLight(ticker, assetsData) {
    const d = assetsData[ticker];
    if (!d) return;
    const tl = d.traffic_light;
    const colorMap = { green: '#10B981', yellow: '#F59E0B', red: '#F43F5E' };
    const color = colorMap[tl.color] || '#8A9BB5';

    const el = document.getElementById('trafficLight');
    if (!el) return;
    el.innerHTML = `
        <div style="text-align:center;">
            <div style="width:80px; height:80px; border-radius:50%; background:${color};
                        margin:0 auto 16px; display:flex; align-items:center; justify-content:center;
                        font-size:30px; box-shadow: 0 0 30px ${color}60;">
                ${tl.color === 'green' ? '✅' : tl.color === 'yellow' ? '⚠️' : '❌'}
            </div>
            <div style="font-size:18px; font-weight:700; color:${color}; margin-bottom:6px;">${tl.label}</div>
            <div style="font-size:12px; color:#64748B;">${d.kupiec.n_violations} violaciones en ${d.kupiec.n_obs} días</div>
            <div style="font-size:11px; color:#64748B; margin-top:4px;">Mult. capital: ×${tl.capital_multiplier}</div>
            <hr style="border-color:rgba(255,255,255,0.07); margin:16px 0;">
            <div style="font-size:10px; color:#8A9BB5; line-height:1.8;">
                Tasa violaciones: <strong style="color:#F0C040;">${(d.kupiec.violation_rate*100).toFixed(1)}%</strong><br>
                Esperado: <strong>${(d.kupiec.expected_rate*100).toFixed(1)}%</strong>
            </div>
        </div>`;
}

function renderBtStatsTable(ticker, assetsData) {
    const d = assetsData[ticker];
    if (!d) return;
    const el = document.getElementById('btStatsTable');
    if (!el) return;

    const rows = [
        ['Test', 'Estadístico LR', 'p-valor', 'Resultado'],
        ['Kupiec POF (Cobertura Incond.)', d.kupiec.lr_uc.toFixed(4), d.kupiec.p_value_uc.toFixed(4), d.kupiec.reject_h0_uc ? '❌ Rechazar H₀' : '✅ No rechazar H₀'],
        ['Christoffersen (Independencia)', d.christoffersen.lr_ind.toFixed(4), d.christoffersen.p_value_ind.toFixed(4), d.christoffersen.reject_h0_ind ? '❌ Violaciones agrupadas' : '✅ Independientes'],
        ['Cobertura Condicional (CC)', d.conditional_coverage.lr_cc.toFixed(4), d.conditional_coverage.p_value_cc.toFixed(4), d.conditional_coverage.reject ? '❌ Modelo débil' : '✅ Modelo válido'],
    ];

    el.innerHTML = `<table style="width:100%; border-collapse:collapse; font-size:11px;">
        ${rows.map((r, i) => `<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
            ${r.map((c, j) => `<${i===0?'th':'td'} style="padding:8px 6px; color:${i===0?'#8A9BB5':j===0?'#C8D8EC':'#F0F4F8'}; text-align:${j===0?'left':'center'};">${c}</${i===0?'th':'td'}>`).join('')}
        </tr>`).join('')}
    </table>`;
}

function renderEWMAHeatmap(corrData) {
    const { tickers, matrix } = corrData;
    const shortTickers = tickers.map(t => t.replace('.LM',''));

    const trace = {
        z: matrix,
        x: shortTickers, y: shortTickers,
        type: 'heatmap',
        colorscale: [
            [0, '#0F1923'], [0.5, '#1E40AF'], [1, '#F0C040']
        ],
        zmin: -1, zmax: 1,
        text: matrix.map(row => row.map(v => v.toFixed(2))),
        texttemplate: '%{text}',
        textfont: { size: 9, color: '#fff' },
        showscale: true,
        colorbar: { thickness: 12, len: 0.8, tickfont: { size: 9, color: '#64748B' } }
    };

    const layout = {
        ...baseLayout(),
        margin: { t: 10, r: 80, l: 70, b: 70 },
        xaxis: { ...baseLayout().xaxis, tickfont: { size: 9 } },
        yaxis: { ...baseLayout().yaxis, tickfont: { size: 9 }, autorange: 'reversed' },
    };

    Plotly.newPlot('chart-ewma-corr', [trace], layout, plotConfig);
}

async function runStressTest() {
    const btn = document.getElementById('runStressBtn');
    if (btn) { btn.disabled = true; btn.textContent = 'Running...'; }
    const el = document.getElementById('stressTestTable');
    if (el) el.innerHTML = '<div style="color:#8A9BB5; text-align:center; padding:20px; font-size:12px;">⏳ Descargando datos históricos de crisis...</div>';

    try {
        const capital = state.capital * 1000;
        const resp = await fetch(`/api/stress-test?capital=${capital}&confidence=${state.confidence}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        renderStressTable(data.scenarios, capital);
    } catch(e) {
        if (el) el.innerHTML = `<div style="color:#F43F5E; padding:20px; font-size:12px;">Error: ${e.message}</div>`;
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = 'RUN STRESS TEST'; }
    }
}

function renderStressTable(scenarios, capital) {
    const el = document.getElementById('stressTestTable');
    if (!el) return;

    const rows = scenarios.map(s => {
        const ret = s.cum_return;
        const pnl = s.pnl_pen;
        const color = ret === null ? '#8A9BB5' : ret < -0.10 ? '#F43F5E' : ret < -0.05 ? '#F59E0B' : '#10B981';
        return `<tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
            <td style="padding:8px 6px; color:#C8D8EC; font-size:10px;">${s.scenario}</td>
            <td style="padding:8px 6px; color:#8A9BB5; font-size:9px;">${s.period}</td>
            <td style="padding:8px 6px; color:${color}; font-weight:700; font-size:11px; text-align:right;">
                ${ret === null ? 'N/A' : (ret*100).toFixed(2)+'%'}
            </td>
            <td style="padding:8px 6px; color:${color}; font-size:10px; text-align:right;">
                ${pnl === null ? 'N/A' : 'S/. '+(pnl).toLocaleString('es-PE', {maximumFractionDigits:0})}
            </td>
        </tr>`;
    });

    el.innerHTML = `<table style="width:100%; border-collapse:collapse; font-size:11px;">
        <thead><tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
            <th style="padding:8px 6px; color:#8A9BB5; text-align:left;">Escenario</th>
            <th style="padding:8px 6px; color:#8A9BB5; text-align:left;">Período</th>
            <th style="padding:8px 6px; color:#8A9BB5; text-align:right;">Retorno</th>
            <th style="padding:8px 6px; color:#8A9BB5; text-align:right;">P&amp;L (PEN)</th>
        </tr></thead>
        <tbody>${rows.join('')}</tbody>
    </table>`;
}

document.addEventListener('DOMContentLoaded', boot);


// ── MOBILE NAV ────────────────────────────────────────────────
function initMobileNav() {
    const mobileNav = document.getElementById('mobileNav');
    if (!mobileNav) return;

    const showMobileNav = () => {
        if (window.innerWidth <= 900) {
            mobileNav.style.display = 'flex';
        } else {
            mobileNav.style.display = 'none';
        }
    };
    showMobileNav();
    window.addEventListener('resize', showMobileNav);

    // Sync mobile nav with desktop tabs
    mobileNav.querySelectorAll('.nav-tab').forEach(btn => {
        btn.addEventListener('click', () => {
            mobileNav.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            // Also update desktop tabs
            const panelId = 'panel-' + btn.dataset.panel;
            document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('[data-panel="' + btn.dataset.panel + '"]').forEach(b => b.classList.add('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            const target = document.getElementById(panelId);
            if (target) {
                target.classList.add('active');
                if (state.data) {
                    if (btn.dataset.panel === 'overview') renderOverview();
                    if (btn.dataset.panel === 'risk') renderRisk();
                    if (btn.dataset.panel === 'montecarlo') renderMonteCarlo();
                }
            }
        });
    });
}
