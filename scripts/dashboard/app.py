# app_dashboard_modern.py
import streamlit as st
import pandas as pd
import sqlite3
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# ---------------------------
# 🎨 PAGE CONFIG & STYLING
# ---------------------------
st.set_page_config(
    page_title="Tech Stocks Dashboard",
    layout="wide",
    page_icon="📈",
)

# Enhanced CSS for a modern, sleek, and organized look with improved text contrast
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --bg-start: #f6f7fb;
            --bg-end: #e9eef5;
            --text: #0f172a;
            --muted: #475569;
            --heading: #111827;
            --card-bg: #ffffff;
            --card-bg-soft: #f8fafc;
            --border: #e5e7eb;
            --shadow: 0 10px 30px rgba(2, 6, 23, 0.06);
            --brand: #2563eb; /* primary */
            --brand-2: #7c3aed; /* secondary */
            --accent: #06b6d4;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }

        body {
            background: linear-gradient(135deg, var(--bg-start) 0%, var(--bg-end) 100%);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: var(--text);
        }
        h1, h2, h3 {
            color: var(--heading);
            font-weight: 700;
            letter-spacing: -0.01em;
            text-shadow: none;
        }
        p, span, li {
            color: var(--text);
            line-height: 1.6;
            font-weight: 500;
        }
        .subtitle {
            color: var(--muted);
            font-size: 16px;
            font-style: italic;
            font-weight: 500;
        }
        .card {
            background: linear-gradient(145deg, var(--card-bg), var(--card-bg-soft));
            border-radius: 20px;
            padding: 25px;
            margin: 15px 0;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
            transition: transform 0.2s;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .metric-card {
            background: var(--card-bg);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
            border-left: 5px solid var(--brand);
            margin: 10px;
        }
        .metric-card h4 {
            color: var(--heading);
            font-weight: bold;
            margin-bottom: 10px;
        }
        .metric-card p {
            color: var(--text);
            font-size: 18px;
            font-weight: 600;
            margin: 0;
        }
        .overview-card {
            background: var(--card-bg);
            border-radius: 20px;
            padding: 30px;
            margin: 20px 0;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
        }
        .overview-card h4 {
            color: var(--heading);
            font-weight: bold;
        }
        .overview-card p {
            color: var(--text);
            font-size: 16px;
        }
        .divider {
            border: none;
            height: 2px;
            background: linear-gradient(90deg, var(--brand), var(--brand-2));
            margin: 30px 0;
        }
        .tab-content {
            background: rgba(255,255,255,0.9);
            border-radius: 15px;
            padding: 20px;
            margin-top: 10px;
            color: var(--text);
        }
        .footer {
            text-align: center;
            color: var(--muted);
            font-size: 14px;
            margin-top: 50px;
            padding: 20px;
            background: rgba(255,255,255,0.8);
            border-radius: 10px;
        }
        .stButton>button {
            background: var(--brand);
            color: #ffffff !important;
            border: 1px solid var(--brand);
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: 600;
            letter-spacing: 0.2px;
            transition: all 0.25s ease;
        }
        .stDownloadButton>button {
            background: var(--brand-2) !important;
            color: #ffffff !important;
            border: 1px solid var(--brand-2) !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 8px 24px rgba(37, 99, 235, 0.35);
        }
        .stSelectbox, .stSlider {
            border-radius: 10px;
        }
        .highlight {
            background: #ffffff;
            padding: 10px;
            border-radius: 10px;
            border-left: 5px solid var(--warning);
            margin: 10px 0;
            box-shadow: var(--shadow);
            color: #111827 !important;
        }
        .highlight b {
            color: #7c2d12 !important;
        }
        .highlight p, .highlight span {
            color: #111827 !important;
        }
        .highlight * {
            color: #111827 !important;
        }
        /* Improve tabs contrast */
        div[data-baseweb="tab-list"] button[role="tab"] {
            color: #334155 !important;
            background: #ffffff !important;
            border: 1px solid var(--border) !important;
            border-bottom: none !important;
            border-top-left-radius: 10px !important;
            border-top-right-radius: 10px !important;
            padding: 10px 14px !important;
            font-weight: 600 !important;
        }
        div[data-baseweb="tab-list"] button[aria-selected="true"] {
            color: var(--brand) !important;
            font-weight: 700 !important;
            background: #f8fafc !important;
            box-shadow: inset 0 -2px 0 var(--brand) !important;
        }
        div[data-baseweb="tab-panel"] {
            background: transparent !important;
        }
        /* Explanation text style for dark surfaces */
        .explanation {
            color: #e5e7eb !important; /* light gray for readability */
            background: transparent !important;
            font-weight: 500;
            line-height: 1.7;
        }
        .explanation b, .explanation strong {
            color: #fef08a !important; /* soft yellow for the label */
            font-weight: 700;
        }
        /* Light labels for dark sections */
        .field-label {
            color: #e5e7eb !important;
            font-weight: 600;
            margin: 6px 0 4px 2px;
            display: block;
            letter-spacing: .2px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# 📊 HEADER
# ---------------------------
st.title("📈 Tech Stocks Analysis & Prediction Dashboard")
st.markdown("<p class='subtitle'>Data Engineering & Machine Learning | مشروع تحليل الأسهم التقنية</p>",
            unsafe_allow_html=True)


# ---------------------------
# 📥 LOAD DATA
# ---------------------------
@st.cache_data
def load_clean_data():
    if os.path.exists("../data/db/tech_stocks.db"):
        conn = sqlite3.connect("../data/db/tech_stocks.db")
        df = pd.read_sql("SELECT * FROM tech_stocks", conn)
        conn.close()
    else:
        df = pd.read_csv("../data/clean/clean_tech_stocks.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


@st.cache_data
def load_predictions():
    # Match project layout: scripts/data/predictions.csv relative to this app
    path = "../data/predictions.csv"
    if os.path.exists(path):
        dfp = pd.read_csv(path)
        dfp["Date"] = pd.to_datetime(dfp["Date"], dayfirst=True, errors="coerce")
        return dfp

    # Auto-generate baseline predictions if file is missing, using clean data/DB
    # This avoids needing scikit-learn or external training just to preview predictions.
    # Load clean data similar to load_clean_data
    if os.path.exists("../data/db/tech_stocks.db"):
        conn = sqlite3.connect("../data/db/tech_stocks.db")
        dfc = pd.read_sql("SELECT * FROM tech_stocks", conn)
        conn.close()
    else:
        dfc = pd.read_csv("../data/clean/clean_tech_stocks.csv")

    # Ensure types and minimal features
    dfc["Date"] = pd.to_datetime(dfc["Date"], errors="coerce")
    dfc = dfc.dropna(subset=["Date"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    if "Daily_Return" not in dfc.columns or dfc["Daily_Return"].isna().all():
        if "Close" in dfc.columns:
            dfc["Daily_Return"] = dfc.groupby("Ticker")["Close"].pct_change()
    if "Volatility_30" not in dfc.columns or dfc["Volatility_30"].isna().all():
        dfc["Volatility_30"] = dfc.groupby("Ticker")["Daily_Return"].transform(lambda s: s.rolling(30, min_periods=1).std())
    if "MA_7" not in dfc.columns or dfc["MA_7"].isna().all():
        dfc["MA_7"] = dfc.groupby("Ticker")["Close"].transform(lambda s: s.rolling(7, min_periods=1).mean())

    base = dfc.copy()
    base = base.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    base["Actual_Next_Close"] = base.groupby("Ticker")["Close"].shift(-1)
    # Baseline next close: 7-day moving average shifted by 1 (yesterday's MA)
    base["GBR_Pred_Next_Close"] = base.groupby("Ticker")["MA_7"].shift(1)
    # Directions
    if "Daily_Return" in base.columns and not base["Daily_Return"].isna().all():
        base["Actual_Direction"] = (base["Daily_Return"] > 0).astype(int)
    else:
        base["Actual_Direction"] = (base["Close"].diff() > 0).astype(int)
    base["RFC_Pred_Direction"] = ((base["GBR_Pred_Next_Close"] - base["Close"]) > 0).astype(int)
    base["GBC_Pred_Direction"] = base["RFC_Pred_Direction"]
    # Monthly metrics fallbacks
    if "Monthly_Return" not in base.columns or base["Monthly_Return"].isna().all():
        base["Monthly_Return"] = base.groupby("Ticker")["Close"].transform(lambda s: s.pct_change(30))
    if "Monthly_Volatility" not in base.columns or base["Monthly_Volatility"].isna().all():
        base["Monthly_Volatility"] = base.groupby("Ticker")["Daily_Return"].transform(lambda s: s.rolling(30, min_periods=1).std())
    base["Pred_Monthly_Return"] = base["Monthly_Return"]
    base["Pred_Monthly_Volatility"] = base["Monthly_Volatility"]

    preds = base.dropna(subset=["Actual_Next_Close", "GBR_Pred_Next_Close"]).copy()
    # Keep only columns used by the dashboard
    cols = [
        "Date", "Ticker", "Actual_Next_Close", "GBR_Pred_Next_Close", "RFC_Pred_Next_Close",
        "Actual_Direction", "RFC_Pred_Direction", "GBC_Pred_Direction",
        "Actual_Monthly_Return", "Pred_Monthly_Return",
        "Actual_Monthly_Volatility", "Pred_Monthly_Volatility"
    ]
    # Derive RFC_Pred_Next_Close from baseline (same as GBR here)
    preds["RFC_Pred_Next_Close"] = preds["GBR_Pred_Next_Close"]
    preds["Actual_Monthly_Return"] = preds["Monthly_Return"]
    preds["Actual_Monthly_Volatility"] = preds["Monthly_Volatility"]
    preds = preds[[c for c in cols if c in preds.columns]].copy()

    # Save for subsequent runs and return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    preds.to_csv(path, index=False)
    return preds


data = load_clean_data()
pred_df = load_predictions()

# Shared brief for key terms (simple English)
glossary_html = """
<div class='overview-card'>
  <h4>ℹ️ Key Terms (Brief)</h4>
  <ul>
    <li><b>Close:</b> The last traded price of the day.</li>
    <li><b>Moving Average (MA):</b> Average of prices over recent days to smooth noise.</li>
    <li><b>Daily Return:</b> % change from yesterday’s close to today’s close.</li>
    <li><b>Volatility:</b> How much price moves around; higher = more risk and swings.</li>
    <li><b>MACD:</b> A momentum signal from two moving averages (fast vs slow).</li>
    <li><b>Signal Line:</b> A smoothed line used to confirm MACD turns.</li>
    <li><b>Bollinger Bands:</b> A typical range around price; edges suggest stretched moves.</li>
    <li><b>RMSE:</b> Typical prediction error size (lower is better).</li>
    <li><b>R²:</b> How much of price changes the model explains (closer to 1 is better).</li>
    <li><b>Direction Accuracy:</b> % of days model correctly called up or down.</li>
    <li><b>Confusion Matrix:</b> Table showing correct vs wrong direction calls.</li>
    <li><b>Baseline:</b> Simple estimate from moving averages when no model output exists.</li>
    <li><b>Model (GBR/RFC):</b> Machine‑learning predictions from Gradient Boosting / Random Forest.</li>
  </ul>
</div>
"""
# ---------------------------
# ⚙️ SIDEBAR CONTROLS
# ---------------------------
st.sidebar.header("⚙️ Control Panel")
st.sidebar.markdown("---")

# (Reload predictions button removed per request)

tickers = sorted(data["Ticker"].unique())
st.sidebar.markdown("<span class='field-label'>📈 Select Stock:</span>", unsafe_allow_html=True)
selected_ticker = st.sidebar.selectbox("", tickers, help="Choose a tech stock to analyze", label_visibility="collapsed")

ticker_data_full = data[data["Ticker"] == selected_ticker].sort_values("Date").reset_index(drop=True)

min_date = ticker_data_full["Date"].min()
max_date = ticker_data_full["Date"].max()

# ✅ SLIDER FOR DATE RANGE (fixed)
st.sidebar.markdown("<span class='field-label'>📅 Select Time Range:</span>", unsafe_allow_html=True)
selected_range = st.sidebar.slider(
    "",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    help="Adjust the date range for analysis",
    label_visibility="collapsed"
)

ticker_data = ticker_data_full[
    (ticker_data_full["Date"] >= selected_range[0]) &
    (ticker_data_full["Date"] <= selected_range[1])
    ]

if ticker_data.empty:
    st.warning("🚨 No data available for the selected time range. Try widening the range.")
    st.stop()

# ---------------------------
# 🧭 NAVIGATION TABS
# ---------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Overview", "📉 Historical Data", "🔮 Predictions", "🧾 Raw Data", "🧠 Model Insights"])

# --------------------------------------------------
# 🏠 OVERVIEW TAB
# --------------------------------------------------
with tab1:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    st.subheader(f"📘 Overview — {selected_ticker}")

    # Key Metrics in organized cards
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><h4>Data Points</h4><p>{len(ticker_data)}</p></div>",
                    unsafe_allow_html=True)
    with col2:
        last_close = ticker_data['Close'].iloc[-1]
        st.markdown(f"<div class='metric-card'><h4>Last Close</h4><p>${last_close:.2f}</p></div>",
                    unsafe_allow_html=True)
    with col3:
        max_close = ticker_data['Close'].max()
        st.markdown(f"<div class='metric-card'><h4>Max Close</h4><p>${max_close:.2f}</p></div>", unsafe_allow_html=True)
    with col4:
        min_close = ticker_data['Close'].min()
        st.markdown(f"<div class='metric-card'><h4>Min Close</h4><p>${min_close:.2f}</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Detailed Insights in organized cards
    st.markdown("### 🔍 Detailed Insights")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f"""
        <div class='overview-card'>
            <h4>📅 Time Range Summary</h4>
            <p><b>Selected Range:</b> {selected_range[0].date()} → {selected_range[1].date()}</p>
            <p><b>Duration:</b> {(selected_range[1] - selected_range[0]).days} days</p>
            <p><b>Average Close:</b> ${ticker_data['Close'].mean():.2f}</p>
            <p><b>Median Close:</b> ${ticker_data['Close'].median():.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown(f"""
        <div class='overview-card'>
            <h4>📈 Performance Highlights</h4>
            <p><b>Total Change:</b> ${ticker_data['Close'].iloc[-1] - ticker_data['Close'].iloc[0]:.2f} ({((ticker_data['Close'].iloc[-1] / ticker_data['Close'].iloc[0]) - 1) * 100:.2f}%)</p>
            <p><b>Average Volatility (30-day):</b> {ticker_data['Volatility_30'].mean():.4f}</p>
            <p><b>Highest Volatility:</b> {ticker_data['Volatility_30'].max():.4f}</p>
            <p><b>Lowest Volatility:</b> {ticker_data['Volatility_30'].min():.4f}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Quick Chart Preview
    st.markdown("### 📊 Quick Price Trend Preview")
    fig_overview = px.line(ticker_data.tail(30), x="Date", y="Close",
                           title=f"Last 30 Days Closing Price for {selected_ticker}",
                           color_discrete_sequence=["#3b82f6"])
    fig_overview.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=300)
    st.plotly_chart(fig_overview, use_container_width=True)

    st.markdown(
        "<div class='highlight'>💡 <b>Tip:</b> Use the tabs above to explore historical data, predictions, and raw data in detail.</div>",
        unsafe_allow_html=True)

    st.markdown("### ℹ️ Key Terms")
    st.markdown(glossary_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# 📉 HISTORICAL TAB
# --------------------------------------------------
with tab2:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    st.subheader("📉 Historical Analysis")

    # Closing Price
    st.markdown("### 1️⃣ Closing Price Over Time")
    fig1 = px.line(ticker_data, x="Date", y="Close", title=f"{selected_ticker} — Closing Price",
                   color_discrete_sequence=["#3b82f6"])
    fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown(
        "<p class='explanation'>💬 <b>Explanation:</b> This graph shows how the stock's closing price has changed over time, helping identify long-term trends and key turning points.</p>",
        unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Moving Averages
    ma_cols = [c for c in ["MA_7", "MA_30", "MA_50"] if c in ticker_data.columns]
    if ma_cols:
        st.markdown("### 2️⃣ Moving Averages Comparison")
        fig2 = px.line(ticker_data, x="Date", y=ma_cols, title="Short-term vs Long-term Trends (MAs)",
                       color_discrete_map={"MA_7": "#10b981", "MA_30": "#f59e0b", "MA_50": "#ef4444"})
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(
            "<p class='explanation'>💬 <b>Explanation:</b> Moving averages smooth daily fluctuations and help spot potential buy/sell signals based on crossovers.</p>",
            unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Daily Return
    if "Daily_Return" in ticker_data.columns:
        st.markdown("### 3️⃣ Daily Return Analysis")
        fig3 = px.line(ticker_data, x="Date", y="Daily_Return", title="Daily Return (%)",
                       color_discrete_sequence=["#8b5cf6"])
        fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("<p class='explanation'>💬 <b>Explanation:</b> Shows daily percentage changes to measure short-term volatility and momentum.</p>", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Volatility
    if "Volatility_30" in ticker_data.columns:
        st.markdown("### 4️⃣ 30-Day Rolling Volatility")
        fig4 = px.area(ticker_data, x="Date", y="Volatility_30", title="Volatility (30 Days)",
                       color_discrete_sequence=["#f97316"])
        fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown(
            "<p class='explanation'>💬 <b>Explanation:</b> Indicates how much the stock's price fluctuates. Higher volatility means higher risk and potential reward.</p>",
            unsafe_allow_html=True)

    st.markdown("### ℹ️ Key Terms")
    st.markdown(glossary_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# 🔮 PREDICTIONS TAB
# --------------------------------------------------
with tab3:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    st.subheader("🔮 Predictions and Model Insights")

    # Case-insensitive ticker matching to avoid name mismatches
    selected_up = str(selected_ticker).upper()
    pred_tickers_up = set([] if pred_df is None else pred_df["Ticker"].astype(str).str.upper().unique())

    if pred_df is not None and selected_up in pred_tickers_up:
        preds = pred_df[pred_df["Ticker"].astype(str).str.upper() == selected_up].sort_values("Date")
        source_label = "Model (GBR/RFC)"

        last = preds.iloc[-1]
        actual = last.get("Actual_Next_Close", float("nan"))
        predicted = last.get("GBR_Pred_Next_Close", float("nan"))

        # Simple, robust forecast signal based on recent behavior (last 30 trading days)
        recent = ticker_data_full.sort_values("Date").tail(35).copy()
        recent["ret"] = recent["Close"].pct_change()
        recent_rets = recent["ret"].dropna().tail(30)

        # Chance Up = fraction of up days in last 30
        up_prob = float((recent_rets > 0).mean()) if len(recent_rets) else float("nan")

        # Trend decision from Chance Up (probability band to avoid bias)
        if np.isfinite(up_prob):
            if up_prob > 0.55:
                trend = "⬆️ Expected Increase"
            elif up_prob < 0.45:
                trend = "⬇️ Expected Decrease"
            else:
                trend = "➡️ Stable"
        else:
            trend = "➡️ Stable"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric-card'><h4>Last Actual Close</h4><p>${actual:.2f}</p></div>",
                        unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><h4>Predicted Close (GBR)</h4><p>${predicted:.2f}</p></div>",
                        unsafe_allow_html=True)
        with col3:
            extra = []
            if np.isfinite(up_prob):
                extra.append(f"Chance Up: {up_prob:.0%}")
            extra_txt = " • ".join(extra) if extra else ""
            st.markdown(f"<div class='metric-card'><h4>Model Forecast</h4><p>{trend}{' • ' + extra_txt if extra_txt else ''}</p></div>", unsafe_allow_html=True)
        # Range card removed per request

        # Show prediction source for clarity
        st.markdown(f"<div class='highlight'><b>Source:</b> {source_label}</div>", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # 1) Actual vs Predicted (primary view)
        st.markdown("### 1️⃣ Predicted vs Actual Closing Prices")
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=preds["Date"], y=preds["Actual_Next_Close"], mode='lines', name='Actual',
                                      line=dict(color='#2e77ff')))
        fig_pred.add_trace(
            go.Scatter(x=preds["Date"], y=preds["GBR_Pred_Next_Close"], mode='lines', name='Predicted (GBR)',
                       line=dict(color='#ff6f61', dash='dot')))
        fig_pred.update_layout(title="Model Comparison — Actual vs Predicted Close", xaxis_title="Date",
                               yaxis_title="Price ($)",
                               plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pred, use_container_width=True)
        st.markdown(
            "<p class='explanation'>💬 <b>Explanation:</b> This chart compares the model’s predicted closing price to what actually happened. The closer the dotted line is to the solid line, the better the model is tracking reality.</p>",
            unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # 2) Volatility Comparison
        if "Actual_Monthly_Volatility" in preds.columns and "Pred_Monthly_Volatility" in preds.columns:
            st.markdown("### 2️⃣ Predicted vs Actual Monthly Volatility")
            fig_vol = px.line(preds, x="Date", y=["Actual_Monthly_Volatility", "Pred_Monthly_Volatility"],
                              title="Volatility Prediction",
                              color_discrete_map={"Actual_Monthly_Volatility": "#06b6d4",
                                                  "Pred_Monthly_Volatility": "#ec4899"})
            fig_vol.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_vol, use_container_width=True)
            st.markdown("<p class='explanation'>💬 <b>Explanation:</b> Volatility tells you how much prices swing. Here we compare real monthly volatility with the model’s forecast to see if it captures the ups and downs.</p>", unsafe_allow_html=True)
        else:
            st.info("ℹ️ Volatility columns not found in predictions data.")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # 3) Model Metrics (computed from predictions.csv)
        st.markdown("### 3️⃣ Model Quality Metrics")
        preds_clean = preds.dropna(subset=["Actual_Next_Close", "GBR_Pred_Next_Close"])

        def compute_rmse(y_true, y_pred):
            diffs = np.array(y_true) - np.array(y_pred)
            return float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) else float("nan")

        def compute_r2(y_true, y_pred):
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            if len(y_true) == 0:
                return float("nan")
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")

        rmse_close = compute_rmse(preds_clean["Actual_Next_Close"], preds_clean["GBR_Pred_Next_Close"])
        r2_close = compute_r2(preds_clean["Actual_Next_Close"], preds_clean["GBR_Pred_Next_Close"])

        acc_rfc = None
        acc_gbc = None
        if {"Actual_Direction", "RFC_Pred_Direction"}.issubset(preds.columns):
            d = preds.dropna(subset=["Actual_Direction", "RFC_Pred_Direction"])
            if len(d) > 0:
                acc_rfc = float((d["Actual_Direction"] == d["RFC_Pred_Direction"]).mean())
        if {"Actual_Direction", "GBC_Pred_Direction"}.issubset(preds.columns):
            d = preds.dropna(subset=["Actual_Direction", "GBC_Pred_Direction"])
            if len(d) > 0:
                acc_gbc = float((d["Actual_Direction"] == d["GBC_Pred_Direction"]).mean())

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='metric-card'><h4>RMSE (Next Close)</h4><p>{rmse_close:.3f}</p></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><h4>R² (Next Close)</h4><p>{r2_close:.3f}</p></div>", unsafe_allow_html=True)
        with c3:
            if acc_rfc is not None or acc_gbc is not None:
                acc_txt = f"RFC: {acc_rfc:.2%}" if acc_rfc is not None else ""
                if acc_gbc is not None:
                    acc_txt = (acc_txt + " • " if acc_txt else "") + f"GBC: {acc_gbc:.2%}"
                st.markdown(f"<div class='metric-card'><h4>Direction Accuracy</h4><p>{acc_txt}</p></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='metric-card'><h4>Direction Accuracy</h4><p>—</p></div>", unsafe_allow_html=True)

        st.markdown("<p class='explanation'>💬 <b>What do these mean?</b> RMSE measures the typical prediction error (lower is better). R² shows how much of the price movement the model explains (closer to 1 is stronger). Direction accuracy is the share of days where the model correctly said “up” or “down.”</p>", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # 4) Confusion Matrix (Direction)
        if {"Actual_Direction", "RFC_Pred_Direction"}.issubset(preds.columns):
            st.markdown("### 4️⃣ Direction Confusion Matrix (RFC)")
            d = preds.dropna(subset=["Actual_Direction", "RFC_Pred_Direction"])
            if len(d) > 0:
                labels = [0, 1]
                matrix = np.zeros((2, 2), dtype=int)
                for i, a in enumerate(labels):
                    for j, p in enumerate(labels):
                        matrix[i, j] = int(((d["Actual_Direction"] == a) & (d["RFC_Pred_Direction"] == p)).sum())
                fig_cm = go.Figure(data=go.Heatmap(
                    z=matrix,
                    x=["Pred:Down", "Pred:Up"],
                    y=["Actual:Down", "Actual:Up"],
                    colorscale="Blues",
                    showscale=True
                ))
                fig_cm.update_layout(height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_cm, use_container_width=True)
                st.markdown("<p class='explanation'>💬 <b>Explanation:</b> The diagonal cells show correct calls. Bigger diagonal values mean the model is better at guessing direction.</p>", unsafe_allow_html=True)
    else:
        # Build baseline predictions so every ticker has a prediction view
        base = ticker_data_full.copy().sort_values("Date").reset_index(drop=True)
        base["Actual_Next_Close"] = base["Close"].shift(-1)
        # Baseline next close: 7-day moving average shifted by 1 (yesterday's MA)
        if "MA_7" in base.columns and not base["MA_7"].isna().all():
            base["GBR_Pred_Next_Close"] = base["MA_7"].shift(1)
        else:
            base["GBR_Pred_Next_Close"] = base["Close"].rolling(7).mean().shift(1)
        # Monthly volatility/return fallbacks
        if "Volatility_30" in base.columns:
            base["Actual_Monthly_Volatility"] = base["Volatility_30"]
            base["Pred_Monthly_Volatility"] = base["Volatility_30"]
        if "Monthly_Return" in base.columns:
            base["Actual_Monthly_Return"] = base["Monthly_Return"]
            base["Pred_Monthly_Return"] = base["Monthly_Return"]
        # Directions
        if "Daily_Return" in base.columns and not base["Daily_Return"].isna().all():
            base["Actual_Direction"] = (base["Daily_Return"] > 0).astype(int)
        else:
            base["Actual_Direction"] = (base["Close"].diff() > 0).astype(int)
        base["RFC_Pred_Direction"] = ((base["GBR_Pred_Next_Close"] - base["Close"]) > 0).astype(int)
        base["GBC_Pred_Direction"] = base["RFC_Pred_Direction"]
        preds = base.dropna(subset=["Actual_Next_Close", "GBR_Pred_Next_Close"]).copy()
        source_label = "Baseline (moving-average)"
        st.info("ℹ️ No trained-model predictions found for this ticker. Showing baseline estimates derived from moving averages so you still get a useful preview.")
        st.markdown(f"<div class='highlight'><b>Source:</b> {source_label}</div>", unsafe_allow_html=True)

    st.markdown("### ℹ️ Key Terms")
    st.markdown(glossary_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# 🧾 RAW DATA TAB
# --------------------------------------------------
with tab4:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    st.subheader("🧾 Raw Data Explorer")

    # Optional filters
    colf1, colf2 = st.columns(2)
    with colf1:
        st.markdown("<span class='field-label'>Select columns to display</span>", unsafe_allow_html=True)
        show_cols = st.multiselect("", options=list(ticker_data.columns), label_visibility="collapsed",
                                   default=["Date", "Open", "High", "Low", "Close", "Volume"] if set(["Open","High","Low","Close","Volume"]).issubset(ticker_data.columns) else list(ticker_data.columns)[:8])
    with colf2:
        st.markdown("<span class='field-label'>Sort by</span>", unsafe_allow_html=True)
        sort_by = st.selectbox("", options=["Date", "Close"] if "Close" in ticker_data.columns else ["Date"], index=0, label_visibility="collapsed")
        st.markdown("<span class='field-label'>Ascending sort</span>", unsafe_allow_html=True)
        ascending = st.checkbox("", value=True, label_visibility="collapsed")

    df_view = ticker_data.sort_values(sort_by, ascending=ascending)
    st.dataframe(df_view[show_cols], use_container_width=True)

    # Download
    csv_bytes = df_view.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered CSV", data=csv_bytes, file_name=f"{selected_ticker}_filtered.csv", mime="text/csv")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# 🧠 MODEL INSIGHTS TAB
# --------------------------------------------------
with tab5:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    st.subheader("🧠 Model Insights and Technical Indicators")

    # Compute indicators safely if not present
    tmp = ticker_data.copy().sort_values("Date")
    if "EMA_12" not in tmp.columns:
        tmp["EMA_12"] = tmp["Close"].ewm(span=12, adjust=False).mean()
    if "EMA_26" not in tmp.columns:
        tmp["EMA_26"] = tmp["Close"].ewm(span=26, adjust=False).mean()
    if "MACD" not in tmp.columns:
        tmp["MACD"] = tmp["EMA_12"] - tmp["EMA_26"]
    if "Signal_Line" not in tmp.columns:
        tmp["Signal_Line"] = tmp["MACD"].ewm(span=9, adjust=False).mean()
    if "MA_20" not in tmp.columns:
        tmp["MA_20"] = tmp["Close"].rolling(window=20).mean()
    if "Rolling_Std_20" not in tmp.columns:
        tmp["Rolling_Std_20"] = tmp["Close"].rolling(window=20).std()
    if "Upper_Band" not in tmp.columns:
        tmp["Upper_Band"] = tmp["MA_20"] + 2 * tmp["Rolling_Std_20"]
    if "Lower_Band" not in tmp.columns:
        tmp["Lower_Band"] = tmp["MA_20"] - 2 * tmp["Rolling_Std_20"]

    st.markdown("### 1️⃣ MACD and Signal")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=tmp["Date"], y=tmp["MACD"], name="MACD", line=dict(color="#06b6d4")))
    fig_macd.add_trace(go.Scatter(x=tmp["Date"], y=tmp["Signal_Line"], name="Signal", line=dict(color="#f59e0b", dash="dot")))
    fig_macd.update_layout(title="MACD vs Signal", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_macd, use_container_width=True)
    st.markdown("<p class='explanation'>💬 <b>Explanation:</b> MACD tracks price momentum by comparing short and long moving averages. When the MACD line rises above the signal line, momentum is turning upward; falling below hints at fading strength.</p>", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("### 2️⃣ Bollinger Bands")
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=tmp["Date"], y=tmp["Upper_Band"], name="Upper Band", line=dict(color="#94a3b8")))
    fig_bb.add_trace(go.Scatter(x=tmp["Date"], y=tmp["MA_20"], name="MA 20", line=dict(color="#6366f1")))
    fig_bb.add_trace(go.Scatter(x=tmp["Date"], y=tmp["Lower_Band"], name="Lower Band", line=dict(color="#94a3b8")))
    fig_bb.add_trace(go.Scatter(x=tmp["Date"], y=tmp["Close"], name="Close", line=dict(color="#22c55e")))
    fig_bb.update_layout(title="Bollinger Bands with Closing Price", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_bb, use_container_width=True)
    st.markdown("<p class='explanation'>💬 <b>Explanation:</b> Bollinger Bands outline a typical price range. Touching the upper band means price is stretched higher than usual, while the lower band highlights potential oversold levels. Use them as context rather than automatic trading signals.</p>", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Top/Bottom daily returns
    st.markdown("### 3️⃣ Best and Worst Daily Returns")
    if "Daily_Return" not in tmp.columns or tmp["Daily_Return"].isna().all():
        tmp["Daily_Return"] = tmp["Close"].pct_change() * 100.0
    top5 = tmp[["Date", "Close", "Daily_Return"]].dropna().sort_values("Daily_Return", ascending=False).head(5)
    bottom5 = tmp[["Date", "Close", "Daily_Return"]].dropna().sort_values("Daily_Return", ascending=True).head(5)
    cL, cR = st.columns(2)
    with cL:
        st.markdown("#### 🔼 Top 5 Days")
        st.dataframe(top5.reset_index(drop=True), use_container_width=True)
    with cR:
        st.markdown("#### 🔽 Worst 5 Days")
        st.dataframe(bottom5.reset_index(drop=True), use_container_width=True)
    st.markdown("<p class='explanation'>💬 <b>Explanation:</b> These tables show the biggest up and down days in your selection. They reveal which sessions drove most gains or losses so you can judge impact and risk.</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# 📌 FOOTER
# --------------------------------------------------
st.markdown(
    "<div class='footer'>Developed by Data Engineering Team — DEPI Graduation Project 2025 🎓"
    " • Last updated: "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M')}</div>",
    unsafe_allow_html=True)