import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from arch import arch_model

# Import our custom modules
from data_pipeline import get_bvl_data
from econometrics import get_pen_prices_and_volumes, filter_assets_by_volume, fit_garch_models
from optimization import simulate_garch_paths, optimize_portfolio, calculate_cvar

# Page configuration
st.set_page_config(
    page_title="Institutional Quant Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- INSTITUTIONAL UI CSS (Salt / Quant Design) -----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap');

/* Hide Streamlit default UI elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #030712;
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.2);
}

/* Base typography */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
    background-color: #030712 !important;
    background: radial-gradient(circle at 50% 10%, #0c142b 0%, #030712 100%) !important;
    color: #F3F4F6 !important;
    font-size: 13px;
}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background-color: #030712 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* Floating Glassmorphic Header */
.terminal-header {
    background: rgba(15, 23, 42, 0.65);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 18px 28px;
    margin-top: -30px;
    margin-bottom: 28px;
    margin-left: 0;
    margin-right: 0;
    font-family: 'Inter', sans-serif;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.terminal-header h1 {
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0;
    padding: 0;
    letter-spacing: 0.5px;
    background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-transform: uppercase;
}

.terminal-header span {
    font-size: 0.8rem;
    font-family: 'Roboto Mono', monospace;
    color: #3B82F6;
    background-color: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.2);
    padding: 6px 12px;
    border-radius: 6px;
    text-shadow: 0 0 10px rgba(59, 130, 246, 0.3);
}

/* Weightless Glass Panels */
.panel {
    background: rgba(15, 23, 42, 0.45);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    padding: 24px;
    margin-bottom: 20px;
    border-radius: 12px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.25);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}

.panel:hover {
    transform: translateY(-4px);
    border-color: rgba(59, 130, 246, 0.3);
    box-shadow: 0 30px 60px rgba(59, 130, 246, 0.15), 0 20px 40px rgba(0, 0, 0, 0.35);
}

.panel-header {
    font-size: 0.75rem;
    font-weight: 700;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 20px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    padding-bottom: 8px;
}

/* Metrics with subtle inner depth */
.metric-container {
    display: flex;
    flex-direction: column;
}

.metric-value {
    font-family: 'Roboto Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    color: #FFFFFF;
    line-height: 1.2;
    background: linear-gradient(135deg, #FFFFFF 0%, #CBD5E1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.color-negative {
    background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

.color-positive {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

.metric-context {
    font-size: 0.75rem;
    color: #64748B;
    margin-top: 6px;
    font-weight: 500;
}

/* Tabs Override */
.stTabs [data-baseweb="tab-list"] {
    gap: 16px;
    background: rgba(15, 23, 42, 0.3);
    padding: 6px 12px;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 24px;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    color: #64748B !important;
    padding-top: 8px;
    padding-bottom: 8px;
    border: none !important;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #94A3B8 !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #3B82F6 !important;
    text-shadow: 0 0 10px rgba(59, 130, 246, 0.3);
}

/* Dataframe font */
[data-testid="stDataFrame"] {
    font-family: 'Roboto Mono', monospace;
    font-size: 12px;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1rem !important;
    max-width: 1400px;
}
</style>
""", unsafe_allow_html=True)
# ----------------- CACHING HEAVY COMPUTATIONS -----------------

@st.cache_data(show_spinner=False)
def load_and_preprocess_data():
    raw_df = get_bvl_data()
    prices_pen, volume_pen = get_pen_prices_and_volumes(raw_df)
    prices_filtered, volume_filtered, valid_tickers = filter_assets_by_volume(prices_pen, volume_pen)
    return prices_filtered, volume_filtered, valid_tickers

@st.cache_resource(show_spinner=False)
def run_econometrics_layer(prices_filtered):
    log_returns, std_residuals, garch_models, cond_vols = fit_garch_models(prices_filtered)
    return log_returns, std_residuals, garch_models, cond_vols

# ----------------- APP LOGIC -----------------

st.sidebar.markdown("**SYSTEM CONTROLS**")
st.sidebar.markdown("---")
confidence_level = st.sidebar.slider("CVaR Confidence Level (%)", 90, 99, 95, 1) / 100.0
capital_inicial = st.sidebar.number_input("Capital Base (PEN)", 100.0, 1000000.0, 3200.0, 100.0)
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 11px; color: #64748B; line-height: 1.5;">
<strong>ENGINE SPECS</strong><br>
Risk Metric: Expected Shortfall (CVaR)<br>
Simulations: 10,000 Paths (20D)<br>
Vol Model: GARCH(1,1)<br>
Optimization: SLSQP<br>
Constraint: Max 15% per asset
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="terminal-header">
    <h1>Quantitative Risk & Portfolio Allocation</h1>
    <span>STATUS: ONLINE | ENGINE: GARCH-BOOTSTRAP</span>
</div>
""", unsafe_allow_html=True)

with st.spinner("Initializing quant engine... fetching data and calibrating econometric models..."):
    try:
        prices_filtered, volume_filtered, valid_tickers = load_and_preprocess_data()
        log_returns, std_residuals, garch_models, cond_vols = run_econometrics_layer(prices_filtered)
    except Exception as e:
        st.error(f"SYSTEM FAULT: {e}")
        st.stop()

with st.spinner("Running 10,000 Monte Carlo paths and optimizing constraints..."):
    np.random.seed(42)
    simulated_returns = simulate_garch_paths(garch_models, std_residuals, n_sims=10000, n_days=20)
    w_opt, opt_result = optimize_portfolio(simulated_returns, valid_tickers, confidence_level)
    
    portfolio_cvar = calculate_cvar(w_opt, simulated_returns, confidence_level)
    portfolio_loss_soles = portfolio_cvar * capital_inicial

# ----------------- TABS PRESENTATION -----------------
tab1, tab2, tab3 = st.tabs(["📊 EXECUTIVE SUMMARY", "🔬 RISK & ECONOMETRICS", "🎲 MONTE CARLO ENGINE"])

# --- TAB 1: EXECUTIVE SUMMARY ---
with tab1:
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-header">BASE CAPITAL</div>
            <div class="metric-container">
                <div class="metric-value">S/. {capital_inicial:,.2f}</div>
                <div class="metric-context">Investable Equity</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        hist_ret = log_returns.mean().values
        exp_port_ret = np.sum(w_opt * hist_ret) * 20
        ret_color_class = "color-positive" if exp_port_ret >= 0 else "color-negative"
        st.markdown(f"""
        <div class="panel">
            <div class="panel-header">EXP. RETURN (20D PROJ)</div>
            <div class="metric-container">
                <div class="metric-value {ret_color_class}">{exp_port_ret:,.4%}</div>
                <div class="metric-context">Historical drift based</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-header">TAIL RISK (CVAR {confidence_level:.0%})</div>
            <div class="metric-container">
                <div class="metric-value">{portfolio_cvar:,.4%}</div>
                <div class="metric-context">Expected Shortfall (20D)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_m4:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-header">MAX EXPECTED LOSS</div>
            <div class="metric-container">
                <div class="metric-value color-negative">S/. {portfolio_loss_soles:,.2f}</div>
                <div class="metric-context">Capital at risk beyond VaR limit</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    col_c1, col_c2 = st.columns([1.2, 1.5])
    
    weights_df = pd.DataFrame({
        "Asset": valid_tickers,
        "Weight": w_opt,
        "Allocation (PEN)": w_opt * capital_inicial
    })
    
    with col_c1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">OPTIMAL ALLOCATION TOPOLOGY</div>', unsafe_allow_html=True)
        pie_df = weights_df[weights_df["Weight"] > 0.0001].copy()
        corp_colors = ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899', '#06B6D4', '#14B8A6', '#6366F1']
        if not pie_df.empty:
            fig = px.pie(pie_df, names="Asset", values="Weight", hole=0.75, color_discrete_sequence=corp_colors)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_family="Roboto Mono", font_size=11, font_color="#E2E8F0",
                showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=300
            )
            fig.update_traces(textposition='outside', textinfo='label+percent', 
                              marker=dict(line=dict(color='#030712', width=2)),
                              textfont=dict(color='#E2E8F0'))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">POSITION SIZING GRID (HEATMAP)</div>', unsafe_allow_html=True)
        
        display_df = weights_df.copy()
        display_df = display_df[display_df["Weight"] > 0.00001].sort_values("Weight", ascending=False)
        
        format_dict = {
            'Weight': '{:.4%}',
            'Allocation (PEN)': 'S/. {:,.2f}'
        }
        
        st.dataframe(
            display_df.style.format(format_dict).background_gradient(subset=['Weight'], cmap='Blues', vmin=0, vmax=0.15),
            use_container_width=True,
            hide_index=True,
            height=300
        )
        st.markdown('</div>', unsafe_allow_html=True)


# --- TAB 2: RISK & ECONOMETRICS ---
with tab2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-header">CONDITIONAL VOLATILITY AUDIT (GARCH 1,1)</div>', unsafe_allow_html=True)
    
    col_a1, col_a2 = st.columns([1, 3])
    with col_a1:
        inspected_ticker = st.selectbox("Select Asset Identifier:", valid_tickers, key="garch_select")
        if inspected_ticker:
            res_model = garch_models[inspected_ticker]
            params_df = pd.DataFrame({
                "Param": res_model.params.index,
                "Estimate": res_model.params.values,
                "P-Value": res_model.pvalues.values
            })
            st.markdown("<br><strong>STATISTICAL SIGNIFICANCE</strong>", unsafe_allow_html=True)
            # Use Pandas 2.1+ compatible styler map
            st.dataframe(params_df.style.format({"Estimate": "{:.5f}", "P-Value": "{:.4f}"}).map(lambda v: 'color: #EF4444;' if v > 0.05 else 'color: #10B981;', subset=['P-Value']), use_container_width=True, hide_index=True)
            
    with col_a2:
        if inspected_ticker:
            plot_df = pd.DataFrame({
                "Returns": log_returns[inspected_ticker],
                "Vol": cond_vols[inspected_ticker]
            }, index=log_returns.index)
            
            fig_plot = go.Figure()
            fig_plot.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Returns"], mode="lines", name="Log Return", line=dict(color="rgba(148, 163, 184, 0.25)", width=1.0)))
            fig_plot.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Vol"], mode="lines", name="Conditional Volatility (GARCH)", line=dict(color="#3B82F6", width=2.0)))
            
            fig_plot.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_family="Roboto Mono", font_size=10, font_color="#94A3B8",
                margin=dict(t=10, b=20, l=10, r=10), height=300,
                legend=dict(orientation="h", y=1.1, x=0.0, font=dict(color="#E2E8F0")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", zeroline=True, zerolinecolor="rgba(255, 255, 255, 0.1)"),
                xaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_plot, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# --- TAB 3: MONTE CARLO ENGINE ---
with tab3:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-header">STOCHASTIC SIMULATION: JOINT HISTORICAL BOOTSTRAP (10,000 SCENARIOS)</div>', unsafe_allow_html=True)
    
    st.markdown(f"Visualizing the distribution of **10,000 possible final portfolio values** after a 20-day horizon. The red tail highlights the worst {(1-confidence_level):.0%} of scenarios (CVaR zone).")
    
    # Calculate portfolio simulated returns (10,000,)
    portfolio_sims = np.dot(simulated_returns, w_opt)
    # Calculate portfolio values
    portfolio_values = capital_inicial * (1 + portfolio_sims)
    
    # Identify VaR cutoff
    var_threshold = np.percentile(portfolio_values, (1 - confidence_level) * 100)
    
    fig_hist = go.Figure()
    
    # Safe scenarios (Blue)
    safe_values = portfolio_values[portfolio_values >= var_threshold]
    fig_hist.add_trace(go.Histogram(
        x=safe_values,
        name='Normal Scenarios',
        marker_color='#3B82F6',
        opacity=0.6,
        nbinsx=100
    ))
    
    # Tail Risk scenarios (Red)
    tail_values = portfolio_values[portfolio_values < var_threshold]
    fig_hist.add_trace(go.Histogram(
        x=tail_values,
        name='Tail Risk (CVaR Zone)',
        marker_color='#EF4444',
        opacity=0.8,
        nbinsx=20
    ))
    
    # Value at Risk Line
    fig_hist.add_vline(x=var_threshold, line_width=2, line_dash="dash", line_color="#F3F4F6",
                       annotation_text=f"VaR {confidence_level:.0%}: S/. {var_threshold:,.2f}",
                       annotation_font_color="#F3F4F6",
                       annotation_position="top left")
                       
    # Initial Capital Line
    fig_hist.add_vline(x=capital_inicial, line_width=1, line_dash="solid", line_color="#10B981",
                       annotation_text="Initial Capital", 
                       annotation_font_color="#10B981",
                       annotation_position="top right")
    
    fig_hist.update_layout(
        barmode='overlay',
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_family="Roboto Mono", font_size=11, font_color="#94A3B8",
        margin=dict(t=30, b=20, l=10, r=10), height=400,
        xaxis_title="Final Portfolio Value (PEN)",
        yaxis_title="Frequency (Out of 10,000 Sims)",
        legend=dict(orientation="h", y=1.1, x=0.0, font=dict(color="#E2E8F0")),
        yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)"),
        xaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
