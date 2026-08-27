import streamlit as st
import pandas as pd
from src.portfolio.research import PortfolioExperimentConfig, run_portfolio_experiment

st.set_page_config(page_title="QuantAlpha Analytics Platform", layout="wide")

st.title("🚀 QuantAlpha: ML-Driven Market Analytics & Backtesting Platform")
st.markdown("Interactive exploration of quantitative signals, multi-asset portfolios, and risk-adjusted backtests.")

# Sidebar Controls for User Interaction
st.sidebar.header("Experiment Parameters")
selected_symbols = st.sidebar.multiselect("Asset Universe", ["RELIANCE", "TCS", "INFY"], default=["RELIANCE", "TCS", "INFY"])
strategy_name = st.sidebar.selectbox("Alpha Strategy", ["momentum", "mean_reversion", "ml_supervised"])
portfolio_method = st.sidebar.selectbox("Allocation Method", ["equal_weight", "inverse_volatility", "volatility_target", "maximum_diversification"])
transaction_cost = st.sidebar.slider("Transaction Cost (%)", 0.0, 0.5, 0.1) / 100.0

if st.sidebar.button("Run Quantitative Backtest"):
    if not selected_symbols:
        st.error("Please select at least one symbol.")
    else:
        with st.spinner("Executing path-dependent portfolio engine & ML pipeline..."):
            try:
                config = PortfolioExperimentConfig(
                    symbols=selected_symbols,
                    strategy_name=strategy_name,
                    portfolio_method=portfolio_method,
                    transaction_cost=transaction_cost
                )
                result_df, metrics = run_portfolio_experiment(config)
                
                st.success("Backtest executed successfully!")
                
                # Display Metrics Summary Cards
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Return", f"{metrics['total_return']*100:.2f}%")
                col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                col3.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
                col4.metric("Total Trades", metrics['trades'])
                
                # Plot Equity Curve
                st.subheader("Portfolio Equity Curve")
                if "Portfolio_Equity" in result_df.columns:
                    st.line_chart(result_df["Portfolio_Equity"])
                else:
                    st.line_chart(result_df.iloc[:, 0])
                    
                # Raw Metrics View
                with st.expander("View Full Performance Metrics Dictionary"):
                    st.json(metrics)
                    
            except Exception as e:
                st.error(f"Error running experiment: {e}")
else:
    st.info("Configure your parameters in the sidebar and click **Run Quantitative Backtest** to launch the analysis.")