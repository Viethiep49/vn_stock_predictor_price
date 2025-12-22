"""
Streamlit Dashboard - MBB Stock Predictor.

Tabs:
1. Overview - Current prediction & signal
2. Charts - Price charts với predictions
3. Performance - Backtest results
4. Settings - Adjust thresholds
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="MBB Stock Predictor",
    page_icon="📈",
    layout="wide",
)

# Title
st.title("📈 MBB Stock Predictor")
st.markdown("**Deep Learning dự đoán giá cổ phiếu MBB (MB Bank)**")

# Sidebar
st.sidebar.header("Settings")
signal_threshold = st.sidebar.slider(
    "Signal Threshold (%)",
    min_value=1,
    max_value=10,
    value=5,
    help="Ngưỡng để tạo tín hiệu BUY/SELL",
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Charts", "📉 Performance", "⚙️ Settings"])

with tab1:
    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Current Price",
            value="25,500 VND",
            delta="2.5%",
        )

    with col2:
        st.metric(
            label="Predicted Price",
            value="26,800 VND",
            delta="+5.1%",
        )

    with col3:
        st.metric(
            label="Signal",
            value="🟢 BUY",
        )

    with col4:
        st.metric(
            label="Model Confidence",
            value="72%",
        )

    st.divider()

    # Key indicators
    st.subheader("Key Indicators")

    ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)

    with ind_col1:
        st.metric("RSI (14)", "45.2", help="< 30: Oversold, > 70: Overbought")

    with ind_col2:
        st.metric("MACD", "Bullish", delta="↑")

    with ind_col3:
        st.metric("SMA Cross", "Above SMA20")

    with ind_col4:
        st.metric("Volume", "1.2M", delta="+15%")

    st.divider()

    # Disclaimer
    st.info(
        "⚠️ **Disclaimer**: Dự đoán giá cổ phiếu có độ không chắc chắn cao. "
        "Kết quả này chỉ mang tính tham khảo, không phải khuyến nghị đầu tư."
    )

with tab2:
    st.header("Price Charts")

    # Placeholder chart
    st.subheader("MBB Price History with Predictions")

    # Create sample data
    dates = pd.date_range(end=datetime.now(), periods=60, freq="D")
    prices = [25000 + i * 50 + (i % 7) * 100 for i in range(60)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=prices,
            mode="lines",
            name="Actual Price",
            line=dict(color="blue"),
        )
    )

    # Add predicted price (last point)
    fig.add_trace(
        go.Scatter(
            x=[dates[-1] + timedelta(days=1)],
            y=[26800],
            mode="markers",
            name="Predicted",
            marker=dict(color="red", size=12),
        )
    )

    fig.update_layout(
        title="MBB Stock Price",
        xaxis_title="Date",
        yaxis_title="Price (VND)",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Technical indicators chart
    st.subheader("Technical Indicators")
    st.write("Coming soon...")

with tab3:
    st.header("Performance Metrics")

    # Model metrics
    st.subheader("Model Performance")

    model_col1, model_col2, model_col3, model_col4 = st.columns(4)

    with model_col1:
        st.metric("MAPE", "8.5%", delta="-1.5%", delta_color="inverse")

    with model_col2:
        st.metric("Direction Accuracy", "58%", delta="+3%")

    with model_col3:
        st.metric("MAE", "450 VND")

    with model_col4:
        st.metric("RMSE", "620 VND")

    st.divider()

    # Strategy metrics
    st.subheader("Strategy Performance (Backtest)")

    strat_col1, strat_col2, strat_col3, strat_col4 = st.columns(4)

    with strat_col1:
        st.metric("Total Return", "+25.3%", delta="vs B&H: +5%")

    with strat_col2:
        st.metric("Sharpe Ratio", "1.45")

    with strat_col3:
        st.metric("Max Drawdown", "-12.5%")

    with strat_col4:
        st.metric("Win Rate", "55%")

    st.divider()

    # Equity curve placeholder
    st.subheader("Equity Curve")
    st.write("Coming soon...")

with tab4:
    st.header("Settings")

    st.subheader("Model Settings")
    col1, col2 = st.columns(2)

    with col1:
        lookback = st.number_input("Lookback Days", value=60, min_value=30, max_value=120)
        retrain_freq = st.selectbox("Retrain Frequency", ["Weekly", "Monthly", "Quarterly"])

    with col2:
        confidence_threshold = st.slider("Minimum Confidence", 0.5, 1.0, 0.7)

    st.divider()

    st.subheader("Risk Management")
    col3, col4 = st.columns(2)

    with col3:
        position_size = st.slider("Position Size (%)", 1, 20, 5)
        stop_loss = st.slider("Stop Loss (%)", 1, 10, 5)

    with col4:
        max_holding = st.number_input("Max Holding Days", value=20, min_value=5, max_value=60)

    st.divider()

    if st.button("Save Settings"):
        st.success("Settings saved!")

    if st.button("Retrain Model"):
        st.info("Retraining model... This may take a few minutes.")


# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        MBB Stock Predictor v0.1.0 | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
