import io, streamlit as st, yfinance as yf, pandas as pd, numpy as np, mplfinance as mpf, matplotlib.pyplot as plt

st.set_page_config(page_title="US Stock Quant Dashboard", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""<style>
.stApp { background-color:#0B0E14; color:#E2E8F0; }
header[data-testid="stHeader"] { background-color:#0B0E14; }
.block-container { padding-top:2.8rem!important; padding-bottom:2rem!important; padding-left:0.8rem!important; padding-right:0.8rem!important; }
.kpi-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:6px; margin:6px 0 8px 0; }
.metric-card { background-color:#131722; border:1px solid #1E293B; border-radius:8px; padding:9px 10px; }
.metric-title { font-size:0.72rem; color:#94A3B8; margin-bottom:2px; }
.metric-value { font-size:1.1rem; font-weight:700; color:#F8FAFC; }
.badge-buy { background-color:rgba(16,185,129,0.15); color:#10B981; border:1px solid #10B981; padding:2px 6px; border-radius:5px; font-size:0.72rem; font-weight:600; display:inline-block; }
.badge-sell { background-color:rgba(239,68,68,0.15); color:#EF4444; border:1px solid #EF4444; padding:2px 6px; border-radius:5px; font-size:0.72rem; font-weight:600; display:inline-block; }
.badge-neutral { background-color:rgba(148,163,184,0.15); color:#94A3B8; border:1px solid #475569; padding:2px 6px; border-radius:5px; font-size:0.72rem; font-weight:600; display:inline-block; }
.ohlc-bar { display:grid; grid-template-columns:repeat(4,1fr); gap:4px; background-color:#131722; border:1px solid #1E293B; border-radius:8px; padding:7px 6px; margin-bottom:8px; text-align:center; }
.ohlc-lbl { font-size:0.68rem; color:#64748B; margin-bottom:1px; }
.ohlc-val { font-size:0.82rem; font-weight:700; color:#F8FAFC; }
div[data-testid="stExpander"], div[data-testid="stExpander"] details, div[data-testid="stExpander"] summary { background-color:#131722!important; color:#F8FAFC!important; border-radius:8px!important; }
div[data-testid="stExpander"] { border:1px solid #1E293B!important; }
div[data-testid="stExpander"] summary p, div[data-testid="stExpander"] summary svg { color:#F8FAFC!important; fill:#F8FAFC!important; font-size:0.82rem!important; font-weight:600!important; }
div[data-baseweb="input"]>div, div[data-baseweb="select"]>div { background-color:#F8FAFC!important; color:#0B0E14!important; border-radius:8px!important; }
div[data-baseweb="input"] input { color:#0B0E14!important; font-weight:600!important; }
label p, .stRadio label p, button[data-baseweb="tab"] p { color:#E2E8F0!important; font-size:0.82rem!important; }
.stButton>button { background-color:#1E293B!important; color:#F8FAFC!important; border:1px solid #334155!important; border-radius:8px!important; font-size:0.8rem!important; }
</style>""", unsafe_allow_html=True)

with st.expander("⚙️ 종목 / 봉 기준 / 전략 설정 (눌러서 열기)", expanded=False):
    ticker = st.text_input("티커 (예: QQQ, NVDA, TSLA)", value="QQQ").upper().strip()
    timeframe = st.radio("봉 기준", ["30분봉", "1시간봉", "일봉"], horizontal=True)
    strategy_filter = st.selectbox("전략 필터", ["전체 전략 통합 (고승률 엄선)", "1. ORB (오프닝 레인지 돌파)", "2. VWAP + RSI 눌림목", "3. EMA 정배열 회귀", "4. 갭 채우기 (Gap Fill) 매도"], label_visibility="collapsed")
    if st.button("🔄 데이터 새로고침", use_container_width=True):
        st.cache_data.clear()

def analyze_df(raw_df, tf, strat_filter):
    if raw_df is None or raw_df.empty: return None
    df = raw_df.dropna().copy()
    if len(df) < 5: return None
    if df.index.tz is None: df.index = df.index.tz_localize('UTC')
    df_ny = df.tz_convert('America/New_York')
    df['Date_NY'], df['Time_NY'] = df_ny.index.date, df_ny.index.strftime('%H:%M')
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    d = df['Close'].diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    l = (-d.where(d < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + g / l.replace(0, np.nan)))
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Vol'] = tp * df['Volume']
    df['VWAP'] = (df.groupby('Date_NY')['TP_Vol'].cumsum() / df.groupby('Date_NY')['Volume'].cumsum()) if tf in ["30분봉", "1시간봉"] else (df['TP_Vol'].rolling(20).sum() / df['Volume'].rolling(20).sum())
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Buy_Signal'], df['Sell_Signal'], df['Signal_Reason'] = False, False, ""
