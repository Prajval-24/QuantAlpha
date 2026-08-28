import streamlit as st
import pandas as pd
import numpy as np
from src.portfolio.research import PortfolioExperimentConfig, run_portfolio_experiment

# 1. Page Configuration
st.set_page_config(
    page_title="QuantAlpha | Institutional Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Injection for Prime Brand Aesthetic
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .stMetric label {
        color: #8b949e !important;
        font-weight: 600;
    }
    .stMetric .st-emotion-cache-1wivap2 {
        color: #58a6ff !important;
    }
    h1, h2, h3 {
        color: #f0f6fc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. App Header & Brand Banner
st.title("📈 QuantAlpha — Institutional Quantitative Platform")
st.markdown("### End-to-End Multi-Asset Research, Machine Learning, and Path-Dependent Backtesting Engine")
st.markdown("---")

# 4. Professional Overview & Platform Architecture Section
with st.container():
    st.markdown("### 🏛️ About the Platform")
    col_intro1, col_intro2 = st.columns([2, 1])
    
    with col_intro1:
        st.markdown("""
        **QuantAlpha** is an institutional-grade quantitative finance research platform built to bridge the gap between heavy financial data ingestion, out-of-sample machine learning prediction, and rigorous portfolio risk execution. 
        
        * **What We Do:** We ingest historical financial time-series data, engineer leakage-free technical features, train rolling walk-forward supervised models (Scikit-Learn Random Forests), and simulate path-dependent portfolio allocations with real-world transaction friction.
        * **How to Access & Use:** Configure your parameters in the sidebar to the left. Select your asset universe, choose between momentum rules, mean-reversion, or machine learning alpha signals, adjust transaction friction, and click **Execute Backtest** to run simulations instantly.
        """)
        
    with col_intro2:
        st.markdown("""
        <div class="card">
            <h4>⚡ Quick Links</h4>
            <p>🔗 <b><a href="https://github.com/Prajval-24/QuantAlpha" target="_blank">GitHub Repository</a></b></p>
            <p>🛠️ <b>Tech Stack:</b> Python, FastAPI, Streamlit, Scikit-Learn, SQLite, Pandas</p>
            <p>👤 <b>Lead Engineer:</b> Prajval Patil</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# 5. Sidebar Controls
st.sidebar.header("⚙️ Experiment Controls")

with st.sidebar.expander("🌍 Asset Universe", expanded=True):
    selected_symbols = st.multiselect(
        "Select Equities", 
        ["RELIANCE", "TCS", "INFY"], 
        default=["RELIANCE", "TCS", "INFY"]
    )

with st.sidebar.expander("🤖 Alpha & Portfolio Strategy", expanded=True):
    strategy_name = st.selectbox(
        "Alpha Signal Generator", 
        ["momentum", "mean_reversion", "ml_supervised"],
        format_func=lambda x: {
            "momentum": "Momentum Strategy (Moving Averages)",
            "mean_reversion": "Mean Reversion Strategy (Z-Score)",
            "ml_supervised": "Scikit-Learn Random Forest (Walk-Forward ML)"
        }[x]
    )
    
    portfolio_method = st.selectbox(
        "Portfolio Weighting Allocation", 
        ["equal_weight", "inverse_volatility", "volatility_target", "maximum_diversification"],
        format_func=lambda x: x.replace("_", " ").title()
    )

with st.sidebar.expander("⚖️ Risk & Friction Settings", expanded=True):
    transaction_cost_pct = st.slider("Transaction Cost / Slippage (%)", 0.0, 0.5, 0.1, 0.05)
    transaction_cost = transaction_cost_pct / 100.0

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Execute Backtest", type="primary", use_container_width=True)

# 6. Main Execution Flow
if run_button:
    if not selected_symbols:
        st.error("⚠️ Please select at least one asset symbol from the sidebar.")
    else:
        with st.spinner("🔄 Running path-dependent portfolio simulation & ML training loop..."):
            try:
                config = PortfolioExperimentConfig(
                    symbols=selected_symbols,
                    strategy_name=strategy_name,
                    portfolio_method=portfolio_method,
                    transaction_cost=transaction_cost
                )
                result_df, metrics = run_portfolio_experiment(config)
                
                st.success("✅ Backtest executed successfully!")
                
                # --- Metrics Dashboard Cards ---
                st.subheader("📊 Performance Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                total_ret = metrics.get('total_return', 0.0) * 100
                sharpe = metrics.get('sharpe_ratio', 0.0)
                max_dd = metrics.get('max_drawdown', 0.0) * 100
                trades = metrics.get('trades', 0)
                
                col1.metric("Total Return", f"{total_ret:.2f}%")
                col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
                col3.metric("Max Drawdown", f"{max_dd:.2f}%")
                col4.metric("Total Trades", f"{trades}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # --- Advanced Layout Tabs ---
                tab_chart, tab_data, tab_metrics = st.tabs(["📈 Equity Curve", "📋 Trade Data & Holdings", "🔍 Raw Metrics JSON"])
                
                with tab_chart:
                    st.subheader("Cumulative Portfolio Equity Growth")
                    if "Portfolio_Equity" in result_df.columns:
                        st.line_chart(result_df["Portfolio_Equity"], use_container_width=True)
                    else:
                        st.line_chart(result_df.iloc[:, 0], use_container_width=True)
                        
                with tab_data:
                    st.subheader("Historical Simulation Time Series")
                    st.dataframe(result_df, use_container_width=True)
                    
                with tab_metrics:
                    st.subheader("Complete Performance Metrics Dictionary")
                    st.json(metrics)
                    
            except Exception as e:
                st.error(f"❌ Error during experiment execution: {e}")
else:
    # Landing View When App Loads
    st.info("👈 **Configure your strategy parameters in the sidebar and click 'Execute Backtest' to launch the quantitative engine.**")
    
    col_a, col_b, col_c = st.columns(3)
    col_a.markdown("""
    <div class="card">
        <h3>⚡ FastAPI & SQLite</h3>
        <p>Ingests and serves historical financial time-series data with optimized database queries for low-latency access.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_b.markdown("""
    <div class="card">
        <h3>🌲 Scikit-Learn ML</h3>
        <p>Implements rolling walk-forward Random Forest classifiers to predict out-of-sample directional returns without look-ahead bias.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_c.markdown("""
    <div class="card">
        <h3>📊 Risk & Execution</h3>
        <p>Simulates path-dependent portfolio turnover, asset volatility weighting, and dynamic transaction cost friction.</p>
    </div>
    """, unsafe_allow_html=True)