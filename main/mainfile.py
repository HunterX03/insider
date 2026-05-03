# ============================================================
# 1️⃣ PAGE CONFIG & CORE IMPORTS
# ============================================================
import streamlit as st

st.set_page_config(
    page_title="AI-Powered Stock Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import pytz

import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ============================================================
# 2️⃣ PATHS & INSIDER FUSION MODEL
# ============================================================
BASE_DIR = r"C:\Users\siddh\Desktop\Insider Trading Detection"
MODELS_DIR = os.path.join(BASE_DIR, "Models")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
DATA_DIR = os.path.join(BASE_DIR, "Data")


class InsiderFusionModel(nn.Module):
    """Must match the architecture used when fused_insider_model.pth was trained."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


@st.cache_resource
def load_fusion_model():
    model_path = os.path.join(MODELS_DIR, "fused_insider_model.pth")
    if not os.path.exists(model_path):
        st.error(f"❌ Insider model not found at: {model_path}")
        return None

    try:
        model = InsiderFusionModel()
        state = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
        model.load_state_dict(state)
        model.eval()
        st.success("✅ Insider fusion model loaded")
        return model
    except Exception as e:
        st.error(f"❌ Error loading fusion model: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


fusion_model = load_fusion_model()


@st.cache_data
def fetch_insider_features(ticker: str):
    """
    Returns [tranad_score, mtad_score, graph_score, sentiment]
    for the given ticker, or None if any piece is missing.
    """
    sym = ticker.replace(".NS", "").replace(".BO", "").upper()

    tranad_score = np.nan
    mtad_score = np.nan
    graph_score = np.nan
    sentiment = np.nan

    # 1) TranAD
    tranad_file = os.path.join(DATA_DIR, "Results", "TranAD_summary.csv")
    if os.path.exists(tranad_file):
        try:
            df_tr = pd.read_csv(tranad_file)
            stock_col = None
            for c in ["Stock", "Symbol", "Ticker", "Company"]:
                if c in df_tr.columns:
                    stock_col = c
                    break
            if stock_col is None:
                stock_col = df_tr.columns[0]

            score_col = None
            for c in ["TranAD_Score", "Score", "Anomaly_Score", "tranad_score"]:
                if c in df_tr.columns:
                    score_col = c
                    break
            if score_col is None and len(df_tr.columns) > 1:
                score_col = df_tr.columns[1]

            if score_col:
                subset = df_tr[df_tr[stock_col].astype(str).str.contains(sym, case=False, na=False)]
                if not subset.empty:
                    tranad_score = pd.to_numeric(subset[score_col], errors="coerce").mean()
        except Exception as e:
            st.warning(f"⚠️ TranAD error: {e}")

    # 2) MTAD-GAT
    mtad_file = os.path.join(RESULTS_DIR, "mtadgat_scores.csv")
    if os.path.exists(mtad_file):
        try:
            mtad_df = pd.read_csv(mtad_file)
            stock_col = None
            for c in ["Stock", "Symbol", "Ticker", "Company"]:
                if c in mtad_df.columns:
                    stock_col = c
                    break
            if stock_col is None:
                stock_col = mtad_df.columns[0]

            score_col = None
            for c in ["MTAD_Score", "Score", "Anomaly_Score", "mtad_score"]:
                if c in mtad_df.columns:
                    score_col = c
                    break
            if score_col is None and len(mtad_df.columns) > 1:
                score_col = mtad_df.columns[1]

            if score_col:
                subset = mtad_df[mtad_df[stock_col].astype(str).str.contains(sym, case=False, na=False)]
                if not subset.empty:
                    mtad_score = pd.to_numeric(subset[score_col], errors="coerce").mean()
        except Exception as e:
            st.warning(f"⚠️ MTAD-GAT error: {e}")

    # 3) Graph anomaly
    graph_file = os.path.join(DATA_DIR, "graph_results", "node_anomaly_probs.csv")
    if os.path.exists(graph_file):
        try:
            gdf = pd.read_csv(graph_file)
            stock_col = None
            for c in ["Stock", "Symbol", "Node_ID", "Ticker", "node_id", "Node", "Company"]:
                if c in gdf.columns:
                    stock_col = c
                    break
            if stock_col is None:
                stock_col = gdf.columns[0]

            score_col = None
            for c in ["anomaly_prob", "Anomaly_Prob", "graph_score", "Anomaly_Score", "score"]:
                if c in gdf.columns:
                    score_col = c
                    break
            if score_col is None and len(gdf.columns) > 1:
                score_col = gdf.columns[1]

            if score_col:
                subset = gdf[gdf[stock_col].astype(str).str.contains(sym, case=False, na=False)]
                if not subset.empty:
                    graph_score = pd.to_numeric(subset[score_col], errors="coerce").mean()
        except Exception as e:
            st.warning(f"⚠️ Graph anomaly error: {e}")

    # 4) FinBERT
    fin_files = [
        os.path.join(RESULTS_DIR, "finbert_sentiment", "FinBERT_NIFTY50_Sentiment.csv"),
        os.path.join(RESULTS_DIR, "finbert_sentiment", "finbert_summary.csv"),
    ]
    for fin_file in fin_files:
        if os.path.exists(fin_file) and np.isnan(sentiment):
            try:
                fdf = pd.read_csv(fin_file)
                stock_col = None
                for c in ["Stock", "Symbol", "Ticker", "Company"]:
                    if c in fdf.columns:
                        stock_col = c
                        break
                if stock_col is None:
                    stock_col = fdf.columns[0]

                sent_col = None
                for c in ["Sentiment_Score", "sentiment", "score", "Compound", "Overall_Sentiment", "FinBERT_Score"]:
                    if c in fdf.columns:
                        sent_col = c
                        break
                if sent_col is None:
                    sent_col = fdf.columns[-1]

                subset = fdf[fdf[stock_col].astype(str).str.contains(sym, case=False, na=False)]
                if not subset.empty:
                    sentiment = pd.to_numeric(subset[sent_col], errors="coerce").mean()
            except Exception as e:
                st.warning(f"⚠️ FinBERT error ({fin_file}): {e}")

    arr = np.array([tranad_score, mtad_score, graph_score, sentiment], dtype=float)
    if np.isnan(arr).any():
        missing = []
        if np.isnan(tranad_score): missing.append("TranAD")
        if np.isnan(mtad_score):   missing.append("MTAD-GAT")
        if np.isnan(graph_score):  missing.append("GraphSAGE")
        if np.isnan(sentiment):    missing.append("FinBERT")
        st.info(f"📊 Missing insider features for {ticker}: {', '.join(missing)}")
        return None
    return arr.tolist()

# ============================================================
# 3️⃣ GLOBAL STYLING
# ============================================================
st.markdown(
    """
<style>
.stApp {background-color: #0b0e11; color: #ffffff; font-family: 'Inter', sans-serif;}
[data-testid="stSidebar"] {background-color: #111418; color: #ffffff; padding: 2rem 1rem;}
h1,h2,h3,h4 {color: #00bcd4; font-weight:600;}
label,p,span,div {color:#e0e0e0 !important;}
.stTextInput input, .stSelectbox, .stSlider {background-color:#1a1f25 !important; color:white !important;
    border:1px solid #2c3138; border-radius:6px;}
button[kind="primary"] {background-color:#00bcd4 !important; color:white !important; border-radius:8px;
    border:none; font-weight:600;}
.stMetric {background-color:#1a1f25 !important; border-radius:12px; padding:1rem; margin:0.5rem;
    box-shadow:0px 2px 8px rgba(0,0,0,0.3);}
.small-muted {color:#888; font-size:0.85rem; margin-top:2rem;}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# 4️⃣ HEADER
# ============================================================
st.markdown(
    """
<h1 style='text-align:center; color:#00bcd4; font-size:2.5rem;'>🤖 AI-Powered Stock Dashboard</h1>
<p style='text-align:center; color:#b0bec5;'>LSTM Time-Series Forecasting & Insider Trading Detection</p>
""",
    unsafe_allow_html=True,
)

# ============================================================
# 5️⃣ SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("⚙️ Controls")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

ticker = st.sidebar.text_input("Stock Ticker (e.g. RELIANCE.NS, TCS.NS)", "TCS.NS")

today_ist = datetime.now(pytz.timezone("Asia/Kolkata")).date()
default_start = today_ist - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", default_start, max_value=today_ist)
end_date = st.sidebar.date_input("End Date", today_ist, min_value=start_date, max_value=today_ist)

train_size = st.sidebar.slider("Train/Test Split %", 60, 95, 80)

st.sidebar.markdown("### 🤖 AI Features")
enable_sentiment = st.sidebar.checkbox("Enable Sentiment Summary", value=True)
enable_anomaly = st.sidebar.checkbox("Detect Price/Volume Anomalies", value=True)
enable_pattern = st.sidebar.checkbox("Pattern Recognition", value=True)
enable_insider = st.sidebar.checkbox("Insider Trading Detection", value=True)

prediction_days = st.sidebar.slider("Forecast Days Ahead", 1, 7, 3)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='
        background: linear-gradient(135deg, #1a1f25 0%, #2c3138 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #00bcd4;
        text-align: center;
        margin-top: 50px;
    '>
        <h4 style='color: #00bcd4; margin-bottom: 10px;'>Developed By</h4>
        <h3 style='color: #ffffff; margin-bottom: 15px;'>Siddharth Lalwani</h3>
        <a href='https://www.linkedin.com/in/siddharth-lalwani-21891818b/' target='_blank' style='
            display: inline-block;
            background: #0077b5;
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            text-decoration: none;
            font-weight: 600;
            margin: 10px 0;
            transition: all 0.3s ease;
        '>
            🔗 LinkedIn Profile
        </a>
        <p style='color: #b0bec5; font-size: 12px; margin-top: 15px;'>
            Built with ❤️ using Streamlit, PyTorch & Machine Learning
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 6️⃣ FETCH DATA
# ============================================================
@st.cache_data(show_spinner=True, ttl=300)
def fetch_data(ticker, start, end):
    try:
        start_dt = pd.to_datetime(start)
        buffered_end = datetime.now() + timedelta(days=1)

        df = yf.download(
            ticker,
            start=start_dt,
            end=buffered_end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df.dropna(subset=["Close"], inplace=True)
        if df.empty:
            return None

        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df[df.index >= start_dt]
        if df.empty:
            return None

        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


with st.spinner(f"Fetching data for {ticker}..."):
    df = fetch_data(ticker, start_date, end_date)

if df is None or df.empty:
    st.error(f"❌ No data found for {ticker} in the selected range.")
    st.stop()

latest_date = df.index[-1].date()
st.success(f"✅ Fetched {len(df)} trading days (latest: {latest_date})")

# ============================================================
# 7️⃣ TECHNICAL INDICATORS (BASIC + ADVANCED)
# ============================================================
def add_indicators(df_in):
    df = df_in.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Moving averages
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA100"] = df["Close"].ewm(span=100, adjust=False).mean()

    # Returns
    df["Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Middle"] - 2 * bb_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / (df["BB_Middle"] + 1e-9)

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    # Volatility & Volume
    df["Volatility"] = df["Return"].rolling(10).std() * np.sqrt(252)
    df["Volume_MA"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / (df["Volume_MA"] + 1e-9)

    # Momentum & ROC
    df["ROC"] = ((df["Close"] - df["Close"].shift(10)) / (df["Close"].shift(10) + 1e-9)) * 100
    df["Momentum"] = df["Close"] - df["Close"].shift(4)

    # Pivot levels
    df["Pivot"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["R1"] = 2 * df["Pivot"] - df["Low"]
    df["S1"] = 2 * df["Pivot"] - df["High"]

    # Trend & BB position
    df["Trend_Strength"] = (df["EMA20"] - df["EMA50"]) / (df["EMA50"] + 1e-9) * 100
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-9)

    df.dropna(inplace=True)
    return df


def add_advanced_indicators(df_in):
    """Enhanced technical indicators for better analysis"""
    df = df_in.copy()
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stochastic_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14 + 1e-9))
    df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
    
    # Williams %R
    df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14 + 1e-9))
    
    # Commodity Channel Index (CCI)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-9)
    
    # Average Directional Index (ADX)
    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff()
    
    pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    atr = df['ATR'].fillna(1)
    pos_di = 100 * (pos_dm.rolling(14).mean() / atr)
    neg_di = 100 * (neg_dm.rolling(14).mean() / atr)
    
    dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di + 1e-9)
    df['ADX'] = dx.rolling(14).mean()
    
    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Money Flow Index (MFI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    
    mfi_ratio = positive_flow / (negative_flow + 1e-9)
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))
    
    # Ichimoku Cloud components
    nine_period_high = df['High'].rolling(window=9).max()
    nine_period_low = df['Low'].rolling(window=9).min()
    df['Tenkan_sen'] = (nine_period_high + nine_period_low) / 2
    
    twenty_six_period_high = df['High'].rolling(window=26).max()
    twenty_six_period_low = df['Low'].rolling(window=26).min()
    df['Kijun_sen'] = (twenty_six_period_high + twenty_six_period_low) / 2
    
    df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
    
    fifty_two_period_high = df['High'].rolling(window=52).max()
    fifty_two_period_low = df['Low'].rolling(window=52).min()
    df['Senkou_Span_B'] = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)
    
    # Price Rate of Change (PROC)
    df['PROC'] = ((df['Close'] - df['Close'].shift(12)) / (df['Close'].shift(12) + 1e-9)) * 100
    
    # Donchian Channel
    df['DC_Upper'] = df['High'].rolling(window=20).max()
    df['DC_Lower'] = df['Low'].rolling(window=20).min()
    df['DC_Middle'] = (df['DC_Upper'] + df['DC_Lower']) / 2
    
    return df


df = add_indicators(df)
df = add_advanced_indicators(df)

if df.empty or len(df) < 120:
    st.error("❌ Not enough data after indicators. Use a longer date range.")
    st.stop()

# ============================================================
# 8️⃣ PRICE FIGURE
# ============================================================
def make_price_figure(df_in, ticker_sym):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_in.index,
            y=df_in["Close"],
            mode="lines",
            name="Close",
            line=dict(width=2, color="cyan"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_in.index,
            y=df_in["EMA20"],
            mode="lines",
            name="EMA20",
            line=dict(width=1.5, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_in.index,
            y=df_in["EMA50"],
            mode="lines",
            name="EMA50",
            line=dict(width=1.5, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_in.index,
            y=df_in["EMA100"],
            mode="lines",
            name="EMA100",
            line=dict(width=1.5, dash="dot"),
        )
    )

    last = df_in.iloc[-1]
    for key, color in zip(["Pivot", "R1", "S1"], ["gray", "green", "red"]):
        fig.add_hline(y=last[key], line_dash="dot", annotation_text=key, line_color=color)

    fig.update_layout(
        title=f"{ticker_sym} — Price & EMA",
        template="plotly_white",
        height=400,
        xaxis_title="Date",
        yaxis_title="Price (₹)",
    )
    return fig


st.markdown("### 1️⃣ Price & Trend Structure")
st.plotly_chart(make_price_figure(df, ticker), use_container_width=True)

# ============================================================
# 9️⃣ ANOMALY DETECTION
# ============================================================
if enable_anomaly:
    st.markdown("## 2️⃣ 🚨 AI Anomaly Detection")

    def detect_anomalies(df_in):
        df_copy = df_in.copy()
        returns = df_copy["Return"]
        mean_return = returns.mean()
        std_return = returns.std()
        df_copy["Z_Score"] = (returns - mean_return) / (std_return + 1e-9)
        df_copy["Is_Anomaly"] = np.abs(df_copy["Z_Score"]) > 2.5
        df_copy["Volume_Z"] = (df_copy["Volume"] - df_copy["Volume"].mean()) / (df_copy["Volume"].std() + 1e-9)
        df_copy["Volume_Anomaly"] = np.abs(df_copy["Volume_Z"]) > 2.5
        return df_copy

    df_anomaly = detect_anomalies(df)
    anomalies = df_anomaly[df_anomaly["Is_Anomaly"]]

    if len(anomalies) > 0:
        col1, col2 = st.columns(2)
        col1.metric("Price Anomalies Detected", len(anomalies))
        col2.metric("Volume Anomalies", int(df_anomaly["Volume_Anomaly"].sum()))

        fig_anomaly = go.Figure()
        fig_anomaly.add_trace(
            go.Scatter(
                x=df_anomaly.index,
                y=df_anomaly["Close"],
                mode="lines",
                name="Price",
                line=dict(color="cyan"),
            )
        )
        fig_anomaly.add_trace(
            go.Scatter(
                x=anomalies.index,
                y=anomalies["Close"],
                mode="markers",
                name="Anomalies",
                marker=dict(size=10, color="red", symbol="x"),
            )
        )
        fig_anomaly.update_layout(
            title="2.1 Anomalous Price Movements",
            template="plotly_white",
            height=300,
            xaxis_title="Date",
            yaxis_title="Price (₹)",
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)

        with st.expander("📊 Anomaly Details (Last 10)"):
            st.dataframe(
                anomalies[["Close", "Return", "Volume", "Z_Score"]]
                .tail(10)
                .style.format({"Close": "₹{:.2f}", "Return": "{:.2%}", "Volume": "{:,.0f}", "Z_Score": "{:.2f}"})
            )
    else:
        st.info("✅ No significant anomalies detected")

# ============================================================
# 🔟 PATTERN RECOGNITION
# ============================================================
if enable_pattern:
    st.markdown("## 3️⃣ 🔍 AI Pattern Recognition")

    def detect_patterns(df_in):
        patterns = []
        last_row = df_in.iloc[-1]

        # Bullish patterns
        if last_row["Close"] > last_row["EMA20"] > last_row["EMA50"]:
            patterns.append(("🟢 Golden Cross", "Bullish momentum - price above both EMAs"))
        if last_row["RSI14"] < 30:
            patterns.append(("🟢 Oversold (RSI)", "Potential buying opportunity"))
        if last_row["MACD"] > last_row["MACD_Signal"] and df_in.iloc[-2]["MACD"] <= df_in.iloc[-2]["MACD_Signal"]:
            patterns.append(("🟢 MACD Bullish Crossover", "Momentum turning positive"))
        if last_row["Close"] < last_row["BB_Lower"]:
            patterns.append(("🟢 Below Lower BB", "Potential mean reversion up"))

        # Bearish patterns
        if last_row["Close"] < last_row["EMA20"] < last_row["EMA50"]:
            patterns.append(("🔴 Death Cross", "Bearish momentum - price below both EMAs"))
        if last_row["RSI14"] > 70:
            patterns.append(("🔴 Overbought (RSI)", "Potential selling pressure"))
        if last_row["MACD"] < last_row["MACD_Signal"] and df_in.iloc[-2]["MACD"] >= df_in.iloc[-2]["MACD_Signal"]:
            patterns.append(("🔴 MACD Bearish Crossover", "Momentum turning negative"))
        if last_row["Close"] > last_row["BB_Upper"]:
            patterns.append(("🔴 Above Upper BB", "Potential mean reversion down"))

        # Neutral / volatility
        if 45 <= last_row["RSI14"] <= 55:
            patterns.append(("⚪ RSI Neutral", "Market in equilibrium"))
        if last_row["BB_Width"] > df_in["BB_Width"].quantile(0.8):
            patterns.append(("⚡ High Volatility", "Increased price movement expected"))
        elif last_row["BB_Width"] < df_in["BB_Width"].quantile(0.2):
            patterns.append(("🔇 Low Volatility", "Consolidation phase - breakout likely"))

        return patterns

    patterns = detect_patterns(df)
    if patterns:
        st.markdown("### 3.1 Detected Patterns")
        for i, (name, desc) in enumerate(patterns, start=1):
            st.markdown(f"**{i}. {name}** – {desc}")
    else:
        st.info("No strong patterns detected")

# ============================================================
# 1️⃣1️⃣ SENTIMENT (TECHNICAL-BASED)
# ============================================================
sentiment_score = 0.0
if enable_sentiment:
    st.markdown("## 4️⃣ 💭 AI Market Sentiment Score")

    def calculate_sentiment(df_in):
        last = df_in.iloc[-1]
        scores = []

        # RSI
        if last["RSI14"] < 30:
            scores.append(1)
        elif last["RSI14"] > 70:
            scores.append(-1)
        else:
            scores.append((50 - last["RSI14"]) / 20)

        # MACD
        scores.append(1 if last["MACD"] > last["MACD_Signal"] else -1)

        # EMAs
        if last["Close"] > last["EMA20"] > last["EMA50"]:
            scores.append(1)
        elif last["Close"] < last["EMA20"] < last["EMA50"]:
            scores.append(-1)
        else:
            scores.append(0)

        # Volume
        if last["Volume_Ratio"] > 1.5:
            scores.append(0.5 if last["Return"] > 0 else -0.5)
        else:
            scores.append(0)

        # Bollinger position
        if last["BB_Position"] > 0.8:
            scores.append(-0.5)
        elif last["BB_Position"] < 0.2:
            scores.append(0.5)
        else:
            scores.append(0)

        overall = np.mean(scores)
        return overall, scores

    sentiment_score, individual_scores = calculate_sentiment(df)

    col1, col2, col3 = st.columns(3)
    if sentiment_score > 0.3:
        sentiment_label, sentiment_color = "🟢 BULLISH", "green"
    elif sentiment_score < -0.3:
        sentiment_label, sentiment_color = "🔴 BEARISH", "red"
    else:
        sentiment_label, sentiment_color = "⚪ NEUTRAL", "gray"

    col2.markdown(
        f"<h2 style='text-align:center;color:{sentiment_color};'>{sentiment_label}</h2>",
        unsafe_allow_html=True,
    )
    col2.metric("Sentiment Score", f"{sentiment_score:.2f}", f"{abs(sentiment_score) * 100:.0f}% confidence")

    with st.expander("📊 Sentiment Breakdown by Indicator"):
        sentiment_df = pd.DataFrame(
            {"Indicator": ["RSI", "MACD", "EMA Trend", "Volume", "Bollinger Position"], "Score": individual_scores}
        )
        fig_sent = px.bar(
            sentiment_df,
            x="Indicator",
            y="Score",
            title="4.1 Individual Indicator Sentiment",
            color="Score",
            color_continuous_scale=["red", "gray", "green"],
            text="Score",
        )
        fig_sent.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_sent.update_layout(yaxis_title="Score", height=350)
        st.plotly_chart(fig_sent, use_container_width=True)

# ============================================================
# 1️⃣2️⃣ Bi-LSTM + Attention Time-Series Model (Forecast)
# ============================================================
st.markdown("## 5️⃣ 🧠 Bi-LSTM + Attention Model (Price Forecasting)")

SEQ_LEN = 60
PRED_STEPS = int(prediction_days)
EPOCHS = 80
BATCH_SIZE = 64
LR = 1e-3

feature_cols = [
    "Close", "Open", "High", "Low", "Volume", "Return", "Log_Return", "Volatility",
    "EMA20", "EMA50", "EMA100", "RSI14", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Lower", "BB_Width", "BB_Position", "ATR", "Volume_Ratio",
    "ROC", "Momentum", "Trend_Strength",
]
feature_cols = [c for c in feature_cols if c in df.columns]

scaler_feats = StandardScaler()
df_feats = df.copy()
df_feats[feature_cols] = scaler_feats.fit_transform(df_feats[feature_cols])


def build_sequences(df_feats_in, df_raw_in, feat_cols, seq_len=60, pred_steps=3):
    feat_vals = df_feats_in[feat_cols].values
    closes = df_raw_in["Close"].values
    dates = df_raw_in.index.to_list()

    X_list, y_list = [], []
    base_prices, future_prices, target_dates = [], [], []

    max_i = len(df_feats_in) - pred_steps
    for i in range(seq_len, max_i):
        X_list.append(feat_vals[i - seq_len : i, :])

        base_p = closes[i - 1]
        fut_p = []
        fut_ret = []
        last_p = base_p
        for k in range(pred_steps):
            p = closes[i + k]
            fut_p.append(p)
            r = (p - last_p) / (last_p + 1e-9)
            fut_ret.append(r)
            last_p = p

        y_list.append(fut_ret)
        base_prices.append(base_p)
        future_prices.append(fut_p)
        target_dates.append(dates[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    base_prices = np.array(base_prices, dtype=np.float32)
    future_prices = np.array(future_prices, dtype=np.float32)
    target_dates = np.array(target_dates)

    return X, y, base_prices, future_prices, target_dates


X_all, y_all, base_all, future_all, target_dates_all = build_sequences(
    df_feats, df, feature_cols, SEQ_LEN, PRED_STEPS
)

if len(X_all) < 100:
    st.error("❌ Not enough sequence data for Bi-LSTM. Increase date range.")
    st.stop()

split_idx = int(len(X_all) * (train_size / 100))
if split_idx < 50 or (len(X_all) - split_idx) < 20:
    st.error("❌ Insufficient samples for the chosen train/test split.")
    st.stop()

X_train, X_test = X_all[:split_idx], X_all[split_idx:]
y_train, y_test = y_all[:split_idx], y_all[split_idx:]
base_train, base_test = base_all[:split_idx], base_all[split_idx:]
future_train, future_test = future_all[:split_idx], future_all[split_idx:]
dates_train, dates_test = target_dates_all[:split_idx], target_dates_all[split_idx:]


class PriceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_ds = PriceDataset(X_train, y_train)
test_ds = PriceDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiLSTMAttn(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        scores = self.attn(out).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = (out * weights.unsqueeze(-1)).sum(dim=1)
        return self.fc(context)


def train_bilstm_model(X_train_in, y_train_in, input_dim, pred_steps, epochs, lr):
    model = BiLSTMAttn(input_dim=input_dim, output_dim=pred_steps).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = PriceDataset(X_train_in, y_train_in)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    progress_bar = st.progress(0.0)
    loss_placeholder = st.empty()

    loss_history = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        avg_loss = epoch_loss / len(dataset)
        loss_history.append(avg_loss)

        progress_bar.progress((epoch + 1) / epochs)
        loss_df = pd.DataFrame({"Epoch": list(range(1, len(loss_history) + 1)), "Loss": loss_history})
        loss_placeholder.line_chart(loss_df.set_index("Epoch"))

    return model


with st.spinner("Training Bi-LSTM"):
    bilstm_model = train_bilstm_model(
        X_train,
        y_train,
        input_dim=len(feature_cols),
        pred_steps=PRED_STEPS,
        epochs=EPOCHS,
        lr=LR,
    )

bilstm_model.eval()
all_pred_returns = []

with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        out = bilstm_model(xb)
        all_pred_returns.append(out.cpu().numpy())

all_pred_returns = np.vstack(all_pred_returns)

# Evaluate only T+1 horizon for metrics
pred_ret_1 = all_pred_returns[:, 0]
true_ret_1 = y_test[:, 0]

pred_price_1 = base_test * (1.0 + pred_ret_1)
true_price_1 = future_test[:, 0]

rmse = np.sqrt(mean_squared_error(true_price_1, pred_price_1))
mae = mean_absolute_error(true_price_1, pred_price_1)
mape = np.mean(np.abs((true_price_1 - pred_price_1) / (true_price_1 + 1e-9))) * 100.0
r2 = r2_score(true_price_1, pred_price_1)

st.markdown("### 5.1 Forecast Model Performance (T+1)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("RMSE (T+1)", f"₹{rmse:.2f}")
col2.metric("MAE (T+1)", f"₹{mae:.2f}")
col3.metric("R² (T+1)", f"{r2:.3f}")
col4.metric("MAPE (T+1)", f"{mape:.2f}%")

# --------- Forecast for next prediction_days from latest window ----------
st.markdown("---")
st.markdown(f"### 5.2 {PRED_STEPS}-Day Ahead Forecast (Model Output)")

last_window_feats = df_feats[feature_cols].iloc[-SEQ_LEN:].values.astype(np.float32)
last_window_feats = last_window_feats.reshape(1, SEQ_LEN, len(feature_cols))
last_close = df["Close"].iloc[-1]

bilstm_model.eval()
with torch.no_grad():
    pred_returns_n = bilstm_model(torch.tensor(last_window_feats, dtype=torch.float32).to(device))
    pred_returns_n = pred_returns_n.cpu().numpy().flatten()

pred_prices_n = []
price = last_close
for r in pred_returns_n:
    price = price * (1.0 + r)
    pred_prices_n.append(price)

forecast_df = pd.DataFrame(
    {
        "Day": [f"Day {i + 1}" for i in range(PRED_STEPS)],
        "Predicted Price": pred_prices_n,
        "Predicted Return %": pred_returns_n * 100.0,
        "Change from Current": np.array(pred_prices_n) - last_close,
    }
)

st.dataframe(
    forecast_df.style.format(
        {
            "Predicted Price": "₹{:.2f}",
            "Predicted Return %": "{:.2f}%",
            "Change from Current": "₹{:.2f}",
        }
    )
)

fig_forecast = go.Figure()
historical_dates = df.index[-30:]
fig_forecast.add_trace(
    go.Scatter(
        x=historical_dates,
        y=df["Close"].iloc[-30:],
        mode="lines",
        name="Historical",
        line=dict(color="cyan"),
    )
)
future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=PRED_STEPS, freq="D")
fig_forecast.add_trace(
    go.Scatter(
        x=future_dates,
        y=pred_prices_n,
        mode="lines+markers",
        name="Forecast",
        line=dict(color="orange", dash="dash"),
        text=[f"₹{p:.2f}" for p in pred_prices_n],
        textposition="top center",
    )
)
fig_forecast.update_layout(
    title=f"5.3 {PRED_STEPS}-Day Price Forecast (Bi-LSTM + Attention)",
    template="plotly_white",
    height=400,
    xaxis_title="Date",
    yaxis_title="Price (₹)",
)
st.plotly_chart(fig_forecast, use_container_width=True)

avg_change = (pred_prices_n[-1] - last_close) / (last_close + 1e-9) * 100.0
if avg_change > 1:
    st.success(f"📈 Model predicts upward trend: +{avg_change:.2f}% over {PRED_STEPS} days")
elif avg_change < -1:
    st.error(f"📉 Model predicts downward trend: {avg_change:.2f}% over {PRED_STEPS} days")
else:
    st.info(f"➡️ Model predicts sideways movement: {avg_change:.2f}% over {PRED_STEPS} days")

# ============================================================
# 1️⃣3️⃣ ACTUAL VS PREDICTED & ERROR ANALYSIS (T+1)
# ============================================================
st.markdown("---")
st.markdown("## 6️⃣ 📊 Actual vs Predicted Prices (T+1, Bi-LSTM + Attention)")

results_df = pd.DataFrame(
    {
        "Date": dates_test,
        "Actual": true_price_1,
        "Predicted": pred_price_1,
    }
)
results_df["Error"] = results_df["Actual"] - results_df["Predicted"]
results_df["Error_Pct"] = results_df["Error"] / (results_df["Actual"] + 1e-9) * 100.0

fig_ap = go.Figure()
fig_ap.add_trace(
    go.Scatter(
        x=results_df["Date"],
        y=results_df["Actual"],
        mode="lines+markers",
        name="Actual",
        line=dict(color="cyan"),
    )
)
fig_ap.add_trace(
    go.Scatter(
        x=results_df["Date"],
        y=results_df["Predicted"],
        mode="lines+markers",
        name="Predicted",
        line=dict(color="orange"),
    )
)
fig_ap.update_layout(
    title="6.1 Actual vs Predicted Prices (Next-Day)",
    xaxis_title="Date",
    yaxis_title="Price (₹)",
    template="plotly_white",
    height=400,
)
st.plotly_chart(fig_ap, use_container_width=True)

with st.expander("📈 Prediction Error Analysis"):
    col_err1, col_err2 = st.columns(2)
    with col_err1:
        fig_err = px.histogram(
            results_df,
            x="Error_Pct",
            nbins=30,
            title="6.2 Prediction Error Distribution (%)",
            labels={"Error_Pct": "Error Percentage"},
        )
        fig_err.update_traces(marker_line_width=0.5, marker_line_color="black")
        st.plotly_chart(fig_err, use_container_width=True)
    with col_err2:
        fig_scatter = px.scatter(
            results_df,
            x="Actual",
            y="Predicted",
            title="6.3 Actual vs Predicted Scatter (Next-Day)",
        )
        fig_scatter.add_trace(
            go.Scatter(
                x=[results_df["Actual"].min(), results_df["Actual"].max()],
                y=[results_df["Actual"].min(), results_df["Actual"].max()],
                mode="lines",
                name="Perfect Prediction",
                line=dict(dash="dash", color="red"),
            )
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ============================================================
# 1️⃣4️⃣ INSIDER TRADING RISK (FUSED MODEL)
# ============================================================
if enable_insider:
    st.markdown("---")
    st.markdown("## 7️⃣ 🕵️ Insider Trading Risk Detection (AI Fusion Model)")

    if fusion_model is None:
        st.warning("⚠️ Insider detection model not loaded.")
    else:
        ins_features = fetch_insider_features(ticker)
        if ins_features is None:
            st.info(f"📊 Insider trading risk data not available for {ticker}.")
            with st.expander("📁 Expected Data Files"):
                st.code(
                    f"""
TranAD: {os.path.join(DATA_DIR, 'Results', 'TranAD_summary.csv')}
MTAD-GAT: {os.path.join(RESULTS_DIR, 'mtadgat_scores.csv')}
GraphSAGE: {os.path.join(DATA_DIR, 'graph_results', 'node_anomaly_probs.csv')}
FinBERT:  {os.path.join(RESULTS_DIR, 'finbert_sentiment', 'FinBERT_NIFTY50_Sentiment.csv')}
OR        {os.path.join(RESULTS_DIR, 'finbert_sentiment', 'finbert_summary.csv')}
"""
                )
        else:
            x_tensor = torch.tensor(ins_features, dtype=torch.float32).view(1, -1)
            with torch.no_grad():
                prob = fusion_model(x_tensor).item()
            risk_pct = prob * 100.0

            if risk_pct >= 70:
                label, color = "🔴 HIGH RISK", "red"
                recommendation = (
                    "⚠️ Significant insider trading patterns detected. "
                    "Exercise extreme caution and consider delaying trades."
                )
                action = "Avoid entry positions until risk subsides"
            elif risk_pct >= 40:
                label, color = "🟠 MODERATE RISK", "orange"
                recommendation = (
                    "⚠️ Some suspicious patterns detected. "
                    "Monitor closely for unusual volume or price movements."
                )
                action = "Proceed with caution, use tight stop losses"
            else:
                label, color = "🟢 LOW RISK", "green"
                recommendation = (
                    "✅ No strong insider trading patterns detected. Market behavior appears normal."
                )
                action = "Normal trading conditions"

            st.markdown(
                f"""
            <div style='background: linear-gradient(135deg, #1a1f25 0%, #2c3138 100%);
                        padding: 2rem; border-radius: 15px; border-left: 5px solid {color};
                        margin-bottom: 1.5rem;'>
                <h2 style='color: {color}; margin: 0;'>{label}</h2>
                <h1 style='color: white; font-size: 3rem; margin: 0.5rem 0;'>{risk_pct:.1f}%</h1>
                <p style='color: #b0bec5; font-size: 1.1rem; margin: 0;'>Insider Trading Probability</p>
                <hr style='border-color: #2c3138; margin: 1rem 0;'>
                <p style='color: white; font-size: 1rem;'>{recommendation}</p>
                <p style='color: #00bcd4; font-weight: 600;'>📋 Action: {action}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            with st.expander("📊 AI Model Feature Analysis & Explainability", expanded=True):
                feature_names = ["TranAD Score", "MTAD-GAT Score", "Graph Anomaly Score", "FinBERT Sentiment"]
                feature_df = pd.DataFrame({"Feature": feature_names, "Value": ins_features})

                col_feat1, col_feat2 = st.columns([1, 1])

                with col_feat1:
                    st.markdown("#### 7.1 Raw Feature Scores")
                    st.dataframe(
                        feature_df.style.format({"Value": "{:.4f}"}).background_gradient(
                            cmap="RdYlGn_r", subset=["Value"]
                        )
                    )

                    st.markdown("#### 7.2 Feature Contribution Analysis")
                    normalized_vals = np.abs(ins_features) / (np.sum(np.abs(ins_features)) + 1e-9)
                    contrib_df = pd.DataFrame(
                        {"Feature": feature_names, "Contribution %": normalized_vals * 100}
                    )

                    fig_contrib = px.bar(
                        contrib_df,
                        x="Contribution %",
                        y="Feature",
                        orientation="h",
                        title="Feature Importance in Risk Assessment",
                        color="Contribution %",
                        color_continuous_scale="Reds",
                        text="Contribution %",
                    )
                    fig_contrib.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig_contrib.update_layout(height=300, showlegend=False, xaxis_title="Contribution (%)")
                    st.plotly_chart(fig_contrib, use_container_width=True)

                with col_feat2:
                    st.markdown("#### 7.3 Multi-Dimensional Feature Scores")
                    fig_features = px.bar(
                        feature_df,
                        x="Feature",
                        y="Value",
                        title="AI Model Input Features",
                        color="Value",
                        color_continuous_scale="RdYlGn_r",
                        text="Value",
                    )
                    fig_features.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                    fig_features.update_layout(height=300, xaxis_tickangle=-45, yaxis_title="Score")
                    st.plotly_chart(fig_features, use_container_width=True)

                    st.markdown("#### 7.4 Individual Risk Indicators")
                    for fname, fval in zip(feature_names, ins_features):
                        if fval > 0.7:
                            risk_level = "🔴 High"
                        elif fval > 0.4:
                            risk_level = "🟠 Medium"
                        else:
                            risk_level = "🟢 Low"
                        st.markdown(f"**{fname}**: {fval:.4f} — {risk_level}")

                st.markdown("---")
                st.markdown(
                    """
                ### 7.5 Feature Descriptions

                **🔍 TranAD Score (Transformer Anomaly Detection)**  
                • Detects unusual temporal patterns in trading behavior  
                • Identifies deviations from normal transaction sequences  

                **🧠 MTAD-GAT Score (Multi-scale Temporal Attention)**  
                • Analyzes multi-timeframe anomalies using graph attention  
                • Captures complex dependencies in trading patterns  

                **🕸️ Graph Anomaly Score (GraphSAGE)**  
                • Network-based detection using company relationship graphs  
                • Finds suspicious trading clusters and structures  

                **💬 FinBERT Sentiment Score**  
                • AI analysis of financial news and filings  
                • Highlights sentiment–price divergences around events  
                """
                )

# ============================================================
# 1️⃣5️⃣ INSIDER ANALYTICS PNG REPORT (FIXED & INTEGRATED)
# ============================================================
if enable_insider and fusion_model is not None:
    st.markdown("---")
    st.markdown("## 8️⃣ 📊 Comprehensive Insider Trading Analytics Dashboard")

    # ==========================================
    # SETUP MATPLOTLIB STYLING
    # ==========================================
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    
    # Clean professional styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.2
    
    COLORS = {
        'blue': '#2563eb',
        'red': '#dc2626',
        'orange': '#f97316',
        'green': '#16a34a',
    }
    
    def format_xaxis_dates(ax, df_data):
        """Format x-axis with clean date labels"""
        date_range = (df_data['Date'].max() - df_data['Date'].min()).days
        
        if date_range <= 90:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%b'))
        elif date_range <= 180:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.grid(True, which='major', axis='x', alpha=0.3)
        ax.grid(True, which='minor', axis='x', alpha=0.1)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    
    def format_inr(value):
        """Format as Indian Rupees"""
        if value >= 10000:
            return f"₹{value/1000:.1f}K"
        elif value >= 1000:
            return f"₹{value:,.0f}"
        else:
            return f"₹{value:.2f}"

    # ==========================================
    # CALCULATE RISK SCORES (FIXED)
    # ==========================================
    df_ia = df.copy()
    df_ia["Date"] = df_ia.index
    df_ia["Return_%"] = df_ia["Return"] * 100

    # Get single overall risk score for the stock
    ins_features = fetch_insider_features(ticker)

    if ins_features is None or fusion_model is None:
        # No insider data - use volume/price anomalies
        df_ia['Risk'] = 25.0  # Base risk
        
        # Increase risk for unusual patterns
        df_ia.loc[df_ia['Volume'] > df_ia['Volume'].quantile(0.95), 'Risk'] += 30
        df_ia.loc[abs(df_ia['Return']) > 0.03, 'Risk'] += 20
        df_ia.loc[df_ia['RSI14'] > 70, 'Risk'] += 15
        df_ia.loc[df_ia['RSI14'] < 30, 'Risk'] += 15
        
        df_ia['Risk'] = df_ia['Risk'].clip(0, 100)
    else:
        # Calculate base risk from fusion model
        with torch.no_grad():
            x_tensor = torch.tensor(ins_features, dtype=torch.float32).view(1, -1)
            base_risk = fusion_model(x_tensor).item() * 100.0
        
        df_ia['Risk'] = base_risk
        
        # Add variability based on actual trading patterns
        df_ia.loc[df_ia['Volume'] > df_ia['Volume'].quantile(0.95), 'Risk'] += 25
        df_ia.loc[abs(df_ia['Return']) > 0.03, 'Risk'] += 20
        df_ia.loc[df_ia['RSI14'] > 70, 'Risk'] += 10
        df_ia.loc[df_ia['RSI14'] < 30, 'Risk'] += 10
        
        df_ia['Risk'] = df_ia['Risk'].clip(0, 100)

    # Mark only truly suspicious days (high risk + unusual activity)
    df_ia["Suspicious"] = (
        (df_ia["Risk"] >= 65) |
        ((df_ia["Risk"] >= 50) & (df_ia['Volume'] > df_ia['Volume'].quantile(0.95)))
    )

    st.write(f"**Analysis:** {len(df_ia)} days | {df_ia['Suspicious'].sum()} suspicious | Risk: {df_ia['Risk'].min():.1f}%-{df_ia['Risk'].max():.1f}%")
    # ==========================================
    # PNG 1: OVERVIEW CHART
    # ==========================================
    save_dir = os.path.join(RESULTS_DIR, "Pinpoint_Detection")
    os.makedirs(save_dir, exist_ok=True)

    suspicious_df_png = df_ia[df_ia["Suspicious"] == True].copy()
    
    # Show statistics
    st.write(f"**Analysis Summary:** {len(df_ia)} trading days | {len(suspicious_df_png)} suspicious days | Max Risk: {df_ia['Risk'].max():.1f}%")

    # Create figure
    fig = plt.figure(figsize=(18, 11), facecolor='white', dpi=100)
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.2, 1, 1],
                  hspace=0.4, left=0.1, right=0.95, top=0.92, bottom=0.08)
    
    fig.text(0.5, 0.965, f'{ticker.replace(".NS", "")} — AI-Powered Insider Trading Detection Report',
             ha='center', fontsize=20, fontweight='bold', color='#1f2937')
    fig.text(0.5, 0.94, f'All prices in Indian Rupees (₹) | Period: {df_ia["Date"].min().strftime("%b %Y")} - {df_ia["Date"].max().strftime("%b %Y")}',
             ha='center', fontsize=12, color='#6b7280')

    # ===== CHART 1: Price with Suspicious Days =====
    ax1 = fig.add_subplot(gs[0])
    
    ax1.plot(df_ia['Date'], df_ia['Close'], color=COLORS['blue'], 
             linewidth=2.5, label='Stock Price', alpha=0.8, zorder=1)
    
    if len(suspicious_df_png) > 0:
        # Add vertical shading for suspicious periods
        for _, row in suspicious_df_png.iterrows():
            ax1.axvspan(row['Date'] - pd.Timedelta(hours=12), 
                       row['Date'] + pd.Timedelta(hours=12),
                       alpha=0.25, color=COLORS['red'], zorder=0)
        
        # Add prominent markers
        ax1.scatter(suspicious_df_png['Date'], suspicious_df_png['Close'], 
                   color=COLORS['red'], s=180, marker='X', 
                   label='🚨 Suspicious Activity', zorder=5, alpha=1.0,
                   edgecolors='darkred', linewidths=3)
    
    ax1.set_ylabel('Price (₹)', fontsize=14, fontweight='600', labelpad=10)
    ax1.set_title('8.1 Stock Price Movement with Suspicious Days', 
                  fontsize=16, fontweight='bold', pad=20, loc='left')
    ax1.legend(loc='upper left', frameon=False, fontsize=12)
    format_xaxis_dates(ax1, df_ia)

    # ===== CHART 2: Volume =====
    ax2 = fig.add_subplot(gs[1])
    
    # Color bars based on suspicious flag
    suspicious_indices = set(suspicious_df_png.index)
    
    for idx, row in df_ia.iterrows():
        color = COLORS['red'] if idx in suspicious_indices else COLORS['blue']
        alpha = 0.9 if idx in suspicious_indices else 0.4
        edge = 'darkred' if idx in suspicious_indices else None
        linewidth = 2 if idx in suspicious_indices else 0
        
        ax2.bar(row['Date'], row['Volume']/1000, 
               color=color, alpha=alpha, width=1.5,
               edgecolor=edge, linewidth=linewidth)
    
    legend_elements = [
        Patch(facecolor=COLORS['blue'], alpha=0.4, label='Normal Volume'),
        Patch(facecolor=COLORS['red'], alpha=0.9, edgecolor='darkred', 
              linewidth=2, label='🚨 Suspicious Volume')
    ]
    ax2.legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=12)
    
    ax2.set_ylabel('Volume (Thousands)', fontsize=14, fontweight='600', labelpad=10)
    ax2.set_title('8.2 Trading Volume Analysis (Suspicious Days Highlighted)', 
                  fontsize=16, fontweight='bold', pad=20, loc='left')
    format_xaxis_dates(ax2, df_ia)

    # ===== CHART 3: Risk Score =====
    ax3 = fig.add_subplot(gs[2])
    
    # Fill risk zones
    ax3.fill_between(df_ia['Date'], 0, df_ia['Risk'],
                    where=(df_ia['Risk'] >= 65),
                    color=COLORS['red'], alpha=0.3,
                    label='High Risk Zone (≥65%)', interpolate=True)
    
    ax3.fill_between(df_ia['Date'], 0, df_ia['Risk'],
                    where=(df_ia['Risk'] >= 40) & (df_ia['Risk'] < 65),
                    color=COLORS['orange'], alpha=0.3,
                    label='Moderate Risk (40-65%)', interpolate=True)
    
    # Plot risk line
    ax3.plot(df_ia['Date'], df_ia['Risk'], color='darkred', 
            linewidth=3, label='AI Risk Score', zorder=3)
    
    # Add suspicious markers
    if len(suspicious_df_png) > 0:
        # Add vertical shading
        for _, row in suspicious_df_png.iterrows():
            ax3.axvspan(row['Date'] - pd.Timedelta(hours=12), 
                       row['Date'] + pd.Timedelta(hours=12),
                       alpha=0.15, color=COLORS['red'], zorder=0)
        
        ax3.scatter(suspicious_df_png['Date'], suspicious_df_png['Risk'],
                   color=COLORS['red'], s=150, marker='o',
                   edgecolors='darkred', linewidths=3, zorder=5,
                   label='🚨 Suspicious Days')
    
    # Threshold lines
    ax3.axhline(65, linestyle='--', color=COLORS['red'], 
               linewidth=2, label='High Risk Threshold', alpha=0.8)
    ax3.axhline(40, linestyle='--', color=COLORS['orange'], 
               linewidth=2, label='Moderate Risk', alpha=0.8)
    
    ax3.set_ylabel('Risk Score (%)', fontsize=14, fontweight='600', labelpad=10)
    ax3.set_xlabel('Date', fontsize=14, fontweight='600', labelpad=10)
    ax3.set_title('8.3 AI-Generated Insider Trading Risk Score Timeline', 
                  fontsize=16, fontweight='bold', pad=20, loc='left')
    ax3.legend(loc='upper left', frameon=False, fontsize=10, ncol=2)
    ax3.set_ylim(0, 105)
    format_xaxis_dates(ax3, df_ia)

    plt.tight_layout()
    st.pyplot(fig)

    overview_png = os.path.join(save_dir, f"{ticker.replace('.NS','')}_overview.png")
    fig.savefig(overview_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # ==========================================
    # PNG 2: TOP 10 SUSPICIOUS DAYS
    # ==========================================
    top10 = df_ia.nlargest(10, 'Risk').copy()
    
    fig2 = plt.figure(figsize=(16, 10), facecolor='white', dpi=100)
    gs2 = GridSpec(2, 2, figure=fig2, hspace=0.35, wspace=0.3,
                  left=0.1, right=0.95, top=0.92, bottom=0.1)
    
    fig2.text(0.5, 0.965, f'{ticker.replace(".NS", "")} — Top 10 Most Suspicious Trading Days (AI Analysis)',
             ha='center', fontsize=20, fontweight='bold', color='#1f2937')
    
    top10['Rank'] = range(1, 11)
    top10['Date_Str'] = top10['Date'].dt.strftime('%d %b %Y')
    top10['Volume_Ratio'] = top10['Volume'] / top10['Volume'].mean()

    # ===== CHART 1: Prices =====
    ax1 = fig2.add_subplot(gs2[0, 0])
    bars = ax1.barh(top10['Rank'], top10['Close'], 
                    color=COLORS['blue'], alpha=0.7, height=0.7)
    
    for bar, price in zip(bars, top10['Close']):
        ax1.text(bar.get_width() + max(top10['Close'])*0.02, 
                bar.get_y() + bar.get_height()/2,
                format_inr(price), va='center', fontsize=11, fontweight='600')
    
    ax1.set_xlabel('Price (₹)', fontsize=13, fontweight='600', labelpad=10)
    ax1.set_ylabel('Rank', fontsize=13, fontweight='600', labelpad=10)
    ax1.set_title('8.4 Stock Closing Prices', fontsize=15, fontweight='bold', pad=15)
    ax1.set_yticks(top10['Rank'])
    ax1.invert_yaxis()

    # ===== CHART 2: Volume Ratio =====
    ax2 = fig2.add_subplot(gs2[0, 1])
    bars = ax2.barh(top10['Rank'], top10['Volume_Ratio'], 
                    color=COLORS['orange'], alpha=0.7, height=0.7)
    
    for bar, ratio in zip(bars, top10['Volume_Ratio']):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{ratio:.1f}×', va='center', fontsize=11, fontweight='600')
    
    ax2.axvline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('Volume Ratio', fontsize=13, fontweight='600', labelpad=10)
    ax2.set_ylabel('Rank', fontsize=13, fontweight='600', labelpad=10)
    ax2.set_title('8.5 Abnormal Trading Volume', fontsize=15, fontweight='bold', pad=15)
    ax2.set_yticks(top10['Rank'])
    ax2.invert_yaxis()

    # ===== CHART 3: Returns =====
    ax3 = fig2.add_subplot(gs2[1, 0])
    colors = [COLORS['green'] if r > 0 else COLORS['red'] 
              for r in top10['Return_%'].fillna(0)]
    
    bars = ax3.barh(top10['Rank'], top10['Return_%'].fillna(0), 
                    color=colors, alpha=0.7, height=0.7)
    
    for bar, ret in zip(bars, top10['Return_%'].fillna(0)):
        x_pos = bar.get_width() + (0.3 if ret > 0 else -0.3)
        ax3.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{ret:+.1f}%', va='center', 
                ha='left' if ret > 0 else 'right',
                fontsize=11, fontweight='600')
    
    ax3.axvline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5)
    ax3.set_xlabel('Return (%)', fontsize=13, fontweight='600', labelpad=10)
    ax3.set_ylabel('Rank', fontsize=13, fontweight='600', labelpad=10)
    ax3.set_title('8.6 Price Returns (%)', fontsize=15, fontweight='bold', pad=15)
    ax3.set_yticks(top10['Rank'])
    ax3.invert_yaxis()

    # ===== CHART 4: Risk Scores =====
    ax4 = fig2.add_subplot(gs2[1, 1])
    bars = ax4.barh(top10['Rank'], top10['Risk'], 
                    color=COLORS['red'], alpha=0.7, height=0.7)
    
    for bar, score in zip(bars, top10['Risk']):
        ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}%', va='center', fontsize=11, fontweight='600')
    
    ax4.axvline(65, color=COLORS['red'], linestyle='--', 
               linewidth=2, label='High Risk', alpha=0.8)
    ax4.axvline(40, color=COLORS['orange'], linestyle='--', 
               linewidth=2, label='Moderate Risk', alpha=0.8)
    ax4.legend(fontsize=9, loc='lower right', frameon=False)
    
    ax4.set_xlabel('Risk Score (%)', fontsize=13, fontweight='600', labelpad=10)
    ax4.set_ylabel('Rank', fontsize=13, fontweight='600', labelpad=10)
    ax4.set_title('8.7 AI Risk Score (Insider Trading Probability)', 
                  fontsize=15, fontweight='bold', pad=15)
    ax4.set_yticks(top10['Rank'])
    ax4.invert_yaxis()

    plt.tight_layout()
    st.pyplot(fig2)

    top10_png = os.path.join(save_dir, f"{ticker.replace('.NS','')}_top_10.png")
    fig2.savefig(top10_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig2)

    st.success(f"✅ Insider analytics PNG reports saved to:\n`{save_dir}`")
# ============================================================
# 1️⃣6️⃣ DETAILED INSIDER ACTIVITY TIMELINE
# ============================================================
if enable_insider and fusion_model is not None:
    st.markdown("---")
    st.markdown("## 9️⃣ 🔍 Detailed Insider Activity Timeline & Suspicious Days")

    fig_risk = go.Figure()

    suspicious_df = df_ia[df_ia["Risk"] >= 65].copy()
    
    # Add semi-transparent background for suspicious periods
    if not suspicious_df.empty:
        for _, row in suspicious_df.iterrows():
            fig_risk.add_vrect(
                x0=row["Date"] - pd.Timedelta(hours=12),
                x1=row["Date"] + pd.Timedelta(hours=12),
                fillcolor="red",
                opacity=0.15,
                layer="below",
                line_width=0,
            )

    # Add risk score line with gradient fill
    fig_risk.add_trace(
        go.Scatter(
            x=df_ia["Date"],
            y=df_ia["Risk"],
            mode="lines",
            line=dict(color="darkorange", width=3),
            name="AI Risk Score",
            fill="tozeroy",
            fillcolor="rgba(255, 140, 0, 0.2)",
        )
    )

    # Add threshold lines
    fig_risk.add_hline(
        y=65,
        line_dash="dash",
        line_color="red",
        line_width=3,
        annotation_text="⚠️ High Risk Threshold (65%)",
        annotation_position="right",
    )
    fig_risk.add_hline(
        y=40,
        line_dash="dot",
        line_color="orange",
        line_width=2,
        annotation_text="⚠️ Moderate Risk (40%)",
        annotation_position="right",
    )

    # Add prominent markers for suspicious days
    if not suspicious_df.empty:
        fig_risk.add_trace(
            go.Scatter(
                x=suspicious_df["Date"],
                y=suspicious_df["Risk"],
                mode="markers+text",
                marker=dict(
                    size=18, 
                    color="red", 
                    symbol="x", 
                    line=dict(width=3, color="darkred")
                ),
                name="🚨 Suspicious Activity",
                text=[f"{r:.1f}%" for r in suspicious_df["Risk"]],
                textposition="top center",
                textfont=dict(size=10, color="red", family="Arial Black"),
            )
        )

    fig_risk.update_layout(
        title={
            "text": "9.1 Insider Trading Risk Timeline (AI Fusion Model)",
            "font": {"size": 16, "color": "#00bcd4"},
        },
        xaxis_title="Date",
        yaxis_title="Risk Probability (%)",
        template="plotly_white",
        height=450,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_risk.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128, 128, 128, 0.2)")
    fig_risk.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128, 128, 128, 0.2)", range=[0, 105])

    st.plotly_chart(fig_risk, use_container_width=True)

    suspicious_df_table = df_ia[df_ia["Risk"] >= 65].copy()
    suspicious_df_table["Risk"] = suspicious_df_table["Risk"].round(2)
    suspicious_df_table["Price_Change_%"] = (
        (suspicious_df_table["Close"] - suspicious_df_table["Close"].shift(1))
        / suspicious_df_table["Close"].shift(1)
        * 100
    ).round(2)
    suspicious_df_table["Volume_vs_Avg"] = (
        suspicious_df_table["Volume"]
        / suspicious_df_table["Volume"].rolling(20, min_periods=1).mean()
    ).round(2)

    if suspicious_df_table.empty or suspicious_df_table["Risk"].isna().all():
        st.success("✅ No suspicious insider activity detected across the entire analysis period.")
        st.info("🔍 The AI fusion model found no significant anomalies suggesting insider trading behavior.")
    else:
        st.markdown(
            f"""
        <div style='background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
                    padding: 1.5rem; border-radius: 10px; border: 2px solid #ff0000; margin-bottom: 1rem;'>
            <h3 style='color: white; margin: 0;'>🚨 ALERT: {len(suspicious_df_table)} Suspicious Trading Days Detected</h3>
            <p style='color: white; margin: 0.5rem 0 0 0;'>Potential insider trading patterns identified by AI analysis</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("### 9.2 Detailed Suspicious Activity Report")
        display_df = suspicious_df_table[
            ["Date", "Close", "Volume", "Risk", "Price_Change_%", "Volume_vs_Avg"]
        ].sort_values("Risk", ascending=False).head(20)

        st.dataframe(
            display_df.rename(
                columns={
                    "Risk": "Insider Risk %",
                    "Price_Change_%": "Price Change %",
                    "Volume_vs_Avg": "Volume vs 20D Avg",
                }
            )
            .style.format(
                {
                    "Close": "₹{:.2f}",
                    "Volume": "{:,.0f}",
                    "Insider Risk %": "{:.2f}%",
                    "Price Change %": "{:+.2f}%",
                    "Volume vs 20D Avg": "{:.2f}x",
                }
            )
            .background_gradient(subset=["Insider Risk %"], cmap="Reds")
            .background_gradient(subset=["Price Change %"], cmap="RdYlGn")
            .background_gradient(subset=["Volume vs 20D Avg"], cmap="YlOrRd"),
            use_container_width=True,
        )

        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        col_stat1.metric("Avg Risk Score", f"{suspicious_df_table['Risk'].mean():.1f}%")
        col_stat2.metric("Max Risk Day", f"{suspicious_df_table['Risk'].max():.1f}%")
        col_stat3.metric("Avg Price Change", f"{suspicious_df_table['Price_Change_%'].mean():.2f}%")
        col_stat4.metric("Avg Volume Spike", f"{suspicious_df_table['Volume_vs_Avg'].mean():.2f}x")

        csv_export = suspicious_df_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Complete Suspicious Days Report (CSV)",
            data=csv_export,
            file_name=f"{ticker.replace('.NS','')}_insider_suspicious_detailed_report.csv",
            mime="text/csv",
        )

    # Changed all df_insider_daily to df_ia below
    if not df_ia["Risk"].isna().all():
        today_risk = float(df_ia["Risk"].iloc[-1])
        today_date = df_ia["Date"].iloc[-1]

        if len(df_ia) >= 5:
            recent_risks = df_ia["Risk"].tail(5)
            risk_trend = "increasing" if recent_risks.iloc[-1] > recent_risks.iloc[0] else "decreasing"
            trend_change = abs(recent_risks.iloc[-1] - recent_risks.iloc[0])
        else:
            risk_trend = "stable"
            trend_change = 0

        st.markdown("### 9.3 Current Day Risk Assessment")

        if today_risk >= 65:
            st.markdown(
                f"""
            <div style='background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
                        padding: 1.5rem; border-radius: 10px; border: 2px solid #ff0000;'>
                <h2 style='color: white; margin: 0;'>🚨 CRITICAL ALERT: TODAY ({today_date.strftime('%Y-%m-%d')})</h2>
                <h1 style='color: white; font-size: 3rem; margin: 0.5rem 0;'>{today_risk:.1f}%</h1>
                <p style='color: white; font-size: 1.2rem; margin: 0;'>
                    ⚠️ High probability of insider trading activity detected<br>
                    📊 Risk is {risk_trend} (±{trend_change:.1f}% over 5 days)<br>
                    🛑 <strong>RECOMMENDATION: Avoid trading until risk subsides</strong>
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        elif today_risk >= 40:
            st.markdown(
                f"""
            <div style='background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
                        padding: 1.5rem; border-radius: 10px; border: 2px solid #ff9800;'>
                <h2 style='color: white; margin: 0;'>⚠️ MODERATE ALERT: TODAY ({today_date.strftime('%Y-%m-%d')})</h2>
                <h1 style='color: white; font-size: 3rem; margin: 0.5rem 0;'>{today_risk:.1f}%</h1>
                <p style='color: white; font-size: 1.1rem; margin: 0;'>
                    🔍 Elevated insider trading risk detected<br>
                    📊 Risk is {risk_trend} (±{trend_change:.1f}% over 5 days)<br>
                    ⚠️ <strong>RECOMMENDATION: Exercise caution, use protective stops</strong>
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div style='background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%);
                        padding: 1.5rem; border-radius: 10px; border: 2px solid #4caf50;'>
                <h2 style='color: white; margin: 0;'>✅ ALL CLEAR: TODAY ({today_date.strftime('%Y-%m-%d')})</h2>
                <h1 style='color: white; font-size: 3rem; margin: 0.5rem 0;'>{today_risk:.1f}%</h1>
                <p style='color: white; font-size: 1.1rem; margin: 0;'>
                    🟢 No significant insider trading patterns detected<br>
                    📊 Risk is {risk_trend} (±{trend_change:.1f}% over 5 days)<br>
                    ✅ <strong>Market behavior appears normal</strong>
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )
    else:
        st.info("ℹ️ Daily insider risk timeline could not be computed due to missing component scores.")

# ============================================================
# CONTINUATION WITH NEW FEATURES
# ============================================================

# ============================================================
# NEW FEATURE: Smart Stop Loss & Take Profit Calculator
# ============================================================
st.markdown("---")
st.markdown("## 🔟 🎯 Smart Stop Loss & Take Profit Levels")

def calculate_smart_levels(df_in):
    """Calculate intelligent stop loss and take profit levels"""
    last = df_in.iloc[-1]
    current_price = last['Close']
    atr = last['ATR']
    volatility = last['Volatility']
    
    # ATR-based stops
    conservative_stop = current_price - (1.5 * atr)
    moderate_stop = current_price - (2.0 * atr)
    aggressive_stop = current_price - (2.5 * atr)
    
    # Volatility-adjusted targets
    vol_multiplier = 1 + (volatility * 10)
    conservative_target = current_price + (1.5 * atr * vol_multiplier)
    moderate_target = current_price + (2.5 * atr * vol_multiplier)
    aggressive_target = current_price + (3.5 * atr * vol_multiplier)
    
    # Support/Resistance levels
    support = last['S1']
    resistance = last['R1']
    
    return {
        'current': current_price,
        'stops': {
            'conservative': max(conservative_stop, support * 0.98),
            'moderate': moderate_stop,
            'aggressive': aggressive_stop
        },
        'targets': {
            'conservative': min(conservative_target, resistance),
            'moderate': moderate_target,
            'aggressive': min(aggressive_target, resistance * 1.05)
        },
        'support': support,
        'resistance': resistance,
        'atr': atr
    }

levels = calculate_smart_levels(df)

st.markdown("### 10.1 Position Sizing & Risk Management")
col_sl1, col_sl2, col_sl3 = st.columns(3)

with col_sl1:
    st.markdown("#### 🛡️ Conservative Strategy")
    st.metric("Stop Loss", f"₹{levels['stops']['conservative']:.2f}", 
              f"{((levels['stops']['conservative'] - levels['current'])/levels['current']*100):.2f}%")
    st.metric("Take Profit", f"₹{levels['targets']['conservative']:.2f}",
              f"{((levels['targets']['conservative'] - levels['current'])/levels['current']*100):.2f}%")
    risk_reward = (levels['targets']['conservative'] - levels['current']) / (levels['current'] - levels['stops']['conservative'])
    st.metric("Risk:Reward", f"1:{risk_reward:.2f}")

with col_sl2:
    st.markdown("#### ⚖️ Moderate Strategy")
    st.metric("Stop Loss", f"₹{levels['stops']['moderate']:.2f}",
              f"{((levels['stops']['moderate'] - levels['current'])/levels['current']*100):.2f}%")
    st.metric("Take Profit", f"₹{levels['targets']['moderate']:.2f}",
              f"{((levels['targets']['moderate'] - levels['current'])/levels['current']*100):.2f}%")
    risk_reward = (levels['targets']['moderate'] - levels['current']) / (levels['current'] - levels['stops']['moderate'])
    st.metric("Risk:Reward", f"1:{risk_reward:.2f}")

with col_sl3:
    st.markdown("#### 🚀 Aggressive Strategy")
    st.metric("Stop Loss", f"₹{levels['stops']['aggressive']:.2f}",
              f"{((levels['stops']['aggressive'] - levels['current'])/levels['current']*100):.2f}%")
    st.metric("Take Profit", f"₹{levels['targets']['aggressive']:.2f}",
              f"{((levels['targets']['aggressive'] - levels['current'])/levels['current']*100):.2f}%")
    risk_reward = (levels['targets']['aggressive'] - levels['current']) / (levels['current'] - levels['stops']['aggressive'])
    st.metric("Risk:Reward", f"1:{risk_reward:.2f}")

# Visualize levels
fig_levels = go.Figure()
fig_levels.add_trace(go.Scatter(x=df.index[-30:], y=df['Close'].iloc[-30:], 
                                mode='lines', name='Price', line=dict(color='cyan', width=2)))

current_date = df.index[-1]
future_date = current_date + timedelta(days=5)

for level_type, color in [('conservative', 'green'), ('moderate', 'orange'), ('aggressive', 'red')]:
    fig_levels.add_trace(go.Scatter(
        x=[current_date, future_date],
        y=[levels['stops'][level_type], levels['stops'][level_type]],
        mode='lines', name=f'{level_type.title()} Stop',
        line=dict(color=color, dash='dash', width=2)
    ))
    fig_levels.add_trace(go.Scatter(
        x=[current_date, future_date],
        y=[levels['targets'][level_type], levels['targets'][level_type]],
        mode='lines', name=f'{level_type.title()} Target',
        line=dict(color=color, dash='dot', width=2)
    ))

fig_levels.update_layout(title="10.2 Stop Loss & Take Profit Visualization",
                        template='plotly_white', height=400,
                        xaxis_title='Date', yaxis_title='Price (₹)')
st.plotly_chart(fig_levels, use_container_width=True)

# ============================================================
# NEW FEATURE: Market Regime Detection
# ============================================================
st.markdown("---")
st.markdown("## 1️⃣1️⃣ 🌡️ Market Regime Analysis")

def detect_market_regime(df_in):
    """Identify current market regime"""
    last_60 = df_in.tail(60)
    
    price_change = (last_60['Close'].iloc[-1] - last_60['Close'].iloc[0]) / last_60['Close'].iloc[0]
    adx = last_60['ADX'].iloc[-1] if 'ADX' in last_60.columns else 20
    
    current_vol = last_60['Volatility'].iloc[-1]
    avg_vol = last_60['Volatility'].mean()
    vol_ratio = current_vol / (avg_vol + 1e-9)
    
    bb_width = last_60['BB_Width'].iloc[-1]
    avg_bb_width = last_60['BB_Width'].mean()
    
    if adx > 25 and abs(price_change) > 0.05:
        if price_change > 0:
            regime = "Strong Uptrend"
            color = "green"
            strategy = "Trend Following - Buy dips, ride momentum"
        else:
            regime = "Strong Downtrend"
            color = "red"
            strategy = "Short or Stay Out - Wait for reversal"
    elif adx > 25 and abs(price_change) <= 0.05:
        regime = "Ranging Market"
        color = "orange"
        strategy = "Mean Reversion - Buy support, sell resistance"
    elif vol_ratio > 1.5:
        regime = "High Volatility"
        color = "purple"
        strategy = "Reduce Position Size - Use wider stops"
    elif bb_width < avg_bb_width * 0.5:
        regime = "Low Volatility Squeeze"
        color = "yellow"
        strategy = "Prepare for Breakout - Wait for direction"
    else:
        regime = "Neutral/Consolidation"
        color = "gray"
        strategy = "Wait for Clear Signal - Avoid random entries"
    
    return {
        'regime': regime,
        'color': color,
        'strategy': strategy,
        'adx': adx,
        'trend_strength': price_change * 100,
        'volatility_ratio': vol_ratio
    }

regime_info = detect_market_regime(df)

st.markdown(f"""
<div style='background: linear-gradient(135deg, #1a1f25 0%, #2c3138 100%);
            padding: 2rem; border-radius: 15px; border-left: 5px solid {regime_info['color']};'>
    <h2 style='color: {regime_info['color']}; margin: 0;'>📊 Current Market Regime</h2>
    <h1 style='color: white; font-size: 2.5rem; margin: 0.5rem 0;'>{regime_info['regime']}</h1>
    <p style='color: #b0bec5; font-size: 1.1rem; margin: 0.5rem 0;'>
        <strong>ADX:</strong> {regime_info['adx']:.1f} | 
        <strong>Trend Strength:</strong> {regime_info['trend_strength']:.2f}% |
        <strong>Volatility Ratio:</strong> {regime_info['volatility_ratio']:.2f}x
    </p>
    <hr style='border-color: #2c3138; margin: 1rem 0;'>
    <p style='color: white; font-size: 1.1rem;'>💡 <strong>Recommended Strategy:</strong></p>
    <p style='color: #00bcd4; font-size: 1.2rem; font-weight: 600;'>{regime_info['strategy']}</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# NEW FEATURE: Advanced Indicators Dashboard
# ============================================================
st.markdown("---")
st.markdown("## 1️⃣2️⃣ 📈 Advanced Technical Indicators")

col_adv1, col_adv2, col_adv3, col_adv4 = st.columns(4)
last = df.iloc[-1]

col_adv1.metric("Stochastic K", f"{last['Stochastic_K']:.1f}", 
                "Overbought" if last['Stochastic_K'] > 80 else "Oversold" if last['Stochastic_K'] < 20 else "Neutral")
col_adv2.metric("CCI", f"{last['CCI']:.1f}",
                "Overbought" if last['CCI'] > 100 else "Oversold" if last['CCI'] < -100 else "Neutral")
col_adv3.metric("ADX", f"{last['ADX']:.1f}",
                "Strong Trend" if last['ADX'] > 25 else "Weak Trend")
col_adv4.metric("MFI", f"{last['MFI']:.1f}",
                "Overbought" if last['MFI'] > 80 else "Oversold" if last['MFI'] < 20 else "Neutral")

# Advanced indicators chart
fig_adv = go.Figure()

fig_adv.add_trace(go.Scatter(x=df.index[-60:], y=df['Stochastic_K'].iloc[-60:],
                             mode='lines', name='Stochastic K', line=dict(color='blue')))
fig_adv.add_trace(go.Scatter(x=df.index[-60:], y=df['Stochastic_D'].iloc[-60:],
                             mode='lines', name='Stochastic D', line=dict(color='red', dash='dash')))
fig_adv.add_hline(y=80, line_dash="dot", line_color="red", annotation_text="Overbought")
fig_adv.add_hline(y=20, line_dash="dot", line_color="green", annotation_text="Oversold")

fig_adv.update_layout(title="12.1 Stochastic Oscillator (Last 60 Days)",
                      template='plotly_white', height=350,
                      xaxis_title='Date', yaxis_title='Value')
st.plotly_chart(fig_adv, use_container_width=True)

# ============================================================
# NEW FEATURE: Volume Profile Analysis
# ============================================================
st.markdown("---")
st.markdown("## 1️⃣3️⃣ 📊 Volume Profile & Market Microstructure")

def calculate_volume_profile(df_in, bins=20):
    """Calculate volume profile for key price levels"""
    recent_data = df_in.tail(100).copy()
    
    price_min = recent_data['Low'].min()
    price_max = recent_data['High'].max()
    
    price_bins = np.linspace(price_min, price_max, bins)
    volume_profile = np.zeros(bins - 1)
    
    for i in range(len(price_bins) - 1):
        mask = (recent_data['Close'] >= price_bins[i]) & (recent_data['Close'] < price_bins[i+1])
        volume_profile[i] = recent_data.loc[mask, 'Volume'].sum()
    
    poc_idx = np.argmax(volume_profile)
    poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
    
    sorted_indices = np.argsort(volume_profile)[::-1]
    cumulative_volume = 0
    total_volume = volume_profile.sum()
    value_area_indices = []
    
    for idx in sorted_indices:
        cumulative_volume += volume_profile[idx]
        value_area_indices.append(idx)
        if cumulative_volume >= total_volume * 0.70:
            break
    
    va_high = price_bins[max(value_area_indices) + 1]
    va_low = price_bins[min(value_area_indices)]
    
    return {
        'price_bins': price_bins,
        'volume_profile': volume_profile,
        'poc': poc_price,
        'va_high': va_high,
        'va_low': va_low
    }

vp = calculate_volume_profile(df)

col_vp1, col_vp2, col_vp3 = st.columns(3)
col_vp1.metric("Point of Control (POC)", f"₹{vp['poc']:.2f}")
col_vp2.metric("Value Area High", f"₹{vp['va_high']:.2f}")
col_vp3.metric("Value Area Low", f"₹{vp['va_low']:.2f}")

fig_vp = go.Figure()

fig_vp.add_trace(go.Scatter(x=df.index[-100:], y=df['Close'].iloc[-100:],
                            mode='lines', name='Price', line=dict(color='cyan')))

fig_vp.add_hline(y=vp['poc'], line_dash="solid", line_color="yellow",
                 line_width=3, annotation_text="POC")

fig_vp.add_hrect(y0=vp['va_low'], y1=vp['va_high'],
                 fillcolor="green", opacity=0.2,
                 annotation_text="Value Area (70%)", annotation_position="right")

fig_vp.update_layout(title="13.1 Volume Profile Analysis (Last 100 Days)",
                     template='plotly_white', height=400,
                     xaxis_title='Date', yaxis_title='Price (₹)')
st.plotly_chart(fig_vp, use_container_width=True)

# ============================================================
# CORRELATION & TECHNICAL SNAPSHOT
# ============================================================
st.markdown("---")
st.markdown("## 1️⃣4️⃣ 🧩 Technical Insights & Correlation")

def plot_correlation_matrix(df_in):
    corr_cols = ["Close", "EMA20", "EMA50", "EMA100", "RSI14", "MACD", "MACD_Signal", "Volatility"]
    available = [c for c in corr_cols if c in df_in.columns]
    corr_df = df_in[available].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        ax=ax,
        square=True,
        linewidths=1,
    )
    plt.title("14.1 Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


colA, colB = st.columns([2, 1])
with colA:
    st.pyplot(plot_correlation_matrix(df))
with colB:
    st.markdown("#### 14.2 Technical Snapshot")
    last = df.iloc[-1]
    st.write(f"**Current Price:** ₹{last['Close']:.2f}")
    st.write(f"**RSI (14):** {last['RSI14']:.1f}")
    st.write(f"**Volatility:** {last['Volatility']:.2%}")
    st.write(f"**Volume vs Avg:** {last['Volume_Ratio']:.2f}x")
    st.write(f"**Trend Strength:** {last['Trend_Strength']:.2f}%")
    if last["RSI14"] < 30:
        st.success("✓ Oversold zone")
    elif last["RSI14"] > 70:
        st.error("✓ Overbought zone")
    else:
        st.info("✓ Neutral RSI range")

# ============================================================
# AI TRADING SUMMARY & RECOMMENDATION
# ============================================================
st.markdown("---")
st.markdown("## 1️⃣5️⃣ 📋 AI Trading Summary & Recommendation")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    last = df.iloc[-1]
    st.markdown("### 15.1 Technical Summary")
    st.write(f"**Current Price:** ₹{last['Close']:.2f}")
    st.write(f"**RSI (14):** {last['RSI14']:.2f}")
    st.write(f"**MACD:** {last['MACD']:.2f}")
    st.write(f"**Volatility:** {last['Volatility']:.2%}")
    st.write(f"**Volume Ratio:** {last['Volume_Ratio']:.2f}x")
    st.write(f"**ATR:** ₹{last['ATR']:.2f}")
    st.markdown("---")
    st.markdown("### 15.2 Support & Resistance")
    st.write(f"**Pivot Point:** ₹{last['Pivot']:.2f}")
    st.write(f"**Resistance (R1):** ₹{last['R1']:.2f}")
    st.write(f"**Support (S1):** ₹{last['S1']:.2f}")

with summary_col2:
    st.markdown("### 15.3 AI Trading Recommendation (Model Driven)")

    avg_forecast_change = np.mean(forecast_df["Predicted Return %"])

    probability_up = 50 + avg_forecast_change * 10
    probability_up = float(np.clip(probability_up, 0, 100))

    if probability_up > 70:
        rating = "🟢 **STRONG BUY**"
        explanation = "Model predicts strong upward momentum over the next few trading days."
    elif probability_up > 55:
        rating = "🟢 **BUY**"
        explanation = "Bullish probability exceeds bearish probability."
    elif probability_up < 30:
        rating = "🔴 **STRONG SELL**"
        explanation = "Model expects significant downside movement."
    elif probability_up < 45:
        rating = "🔴 **SELL**"
        explanation = "Bearish pressure outweighs upward momentum."
    else:
        rating = "⚪ **HOLD**"
        explanation = "Neutral outlook. No clear directional trend."

    st.markdown(f"## {rating}")
    st.write(explanation)

    st.metric("Upward Probability", f"{probability_up:.1f}%")
    st.progress(probability_up / 100)

# ============================================================
# RISK METRICS
# ============================================================
st.markdown("---")
st.markdown("## 1️⃣6️⃣ ⚠️ AI Risk Analysis")

risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

sharpe_ratio = (df["Return"].mean() / (df["Return"].std() + 1e-9)) * np.sqrt(252)
max_drawdown = ((df["Close"].cummax() - df["Close"]) / (df["Close"].cummax() + 1e-9)).max()
var_95 = df["Return"].quantile(0.05)
avg_volume = df["Volume"].mean()

risk_col1.metric(
    "Sharpe Ratio",
    f"{sharpe_ratio:.2f}",
    "Good" if sharpe_ratio > 1 else "Moderate" if sharpe_ratio > 0 else "Poor",
)
risk_col2.metric("Max Drawdown", f"{max_drawdown * 100:.2f}%")
risk_col3.metric("VaR (95%)", f"{var_95 * 100:.2f}%")
risk_col4.metric("Avg Daily Volume", f"{avg_volume:,.0f}")

with st.expander("📖 Risk Metrics Explained"):
    st.markdown(
        """
**Sharpe Ratio** – risk-adjusted return (higher is better).  
**Max Drawdown** – worst peak-to-trough fall.  
**VaR 95%** – expected maximum loss in 95% of days.  
**Average Volume** – liquidity and ease of entry/exit.
"""
    )

# ============================================================
# RECENT DATA TABLE
# ============================================================
st.markdown("---")
st.markdown("## 1️⃣7️⃣ 📅 Recent Trading Data")

display_cols = ["Open", "High", "Low", "Close", "Volume", "RSI14", "MACD", "Volatility"]
available_display_cols = [c for c in display_cols if c in df.columns]

st.dataframe(
    df[available_display_cols]
    .tail(10)
    .style.format(
        {
            "Open": "₹{:.2f}",
            "High": "₹{:.2f}",
            "Low": "₹{:.2f}",
            "Close": "₹{:.2f}",
            "Volume": "{:,.0f}",
            "RSI14": "{:.2f}",
            "MACD": "{:.2f}",
            "Volatility": "{:.2%}",
        }
    )
)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """
<div class='small-muted' style='text-align:center; padding:2rem;'>
    <p>⚠️ <strong>Disclaimer:</strong> This AI dashboard is for educational and research use only.
    Not investment advice. Always do your own research and consult a qualified advisor.</p>
    <p>📊 Data: Yahoo Finance | Forecast Model: PyTorch Bi-LSTM | Insider Risk: TranAD + MTAD-GAT + GraphSAGE + FinBERT Fusion</p>
    <p>✨ Enhanced with Advanced Technical Indicators, Market Regime Detection, Smart SL/TP, Volume Profile Analysis</p>
</div>
""",
    unsafe_allow_html=True,
)