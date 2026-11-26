import streamlit as st, yfinance as yf, pandas as pd, numpy as np, plotly.graph_objects as go
from plotly.subplots import make_subplots

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
.list-card { background-color:#131722; border:1px solid #1E293B; border-radius:8px; padding:9px 11px; margin-bottom:6px; }
.list-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; }
.list-sub { font-size:0.75rem; color:#94A3B8; display:flex; justify-content:space-between; }
div[data-testid="stExpander"], div[data-testid="stExpander"] details, div[data-testid="stExpander"] summary { background-color:#131722!important; color:#F8FAFC!important; border-radius:8px!important; }
div[data-testid="stExpander"] { border:1px solid #1E293B!important; }
div[data-testid="stExpander"] summary p, div[data-testid="stExpander"] summary svg { color:#F8FAFC!important; fill:#F8FAFC!important; font-size:0.82rem!important; font-weight:600!important; }
div[data-baseweb="input"]>div, div[data-baseweb="select"]>div { background-color:#F8FAFC!important; color:#0B0E14!important; border-radius:8px!important; }
div[data-baseweb="input"] input { color:#0B0E14!important; font-weight:600!important; }
label p, .stRadio label p, button[data-baseweb="tab"] p { color:#E2E8F0!important; font-size:0.82rem!important; }
.stButton>button { background-color:#1E293B!important; color:#F8FAFC!important; border:1px solid #334155!important; border-radius:8px!important; font-size:0.8rem!important; }
</style>""", unsafe_allow_html=True)

with st.expander("⚙️ 종목 / 분봉 설정 (눌러서 열기)", expanded=False):
    ticker = st.text_input("티커 (예: QQQ, NVDA, SOXL)", value="QQQ").upper().strip()
    timeframe = st.radio("봉 기준 (단타 전용)", ["30분봉", "1시간봉"], horizontal=True)
    strategy_filter = st.selectbox("전략 모드", ["30m·1h 최고승률 스나이퍼 (골든타임+추세+수급)"], label_visibility="collapsed")
    if st.button("🔄 데이터 새로고침", use_container_width=True): st.cache_data.clear()

@st.cache_data(ttl=60)
def load_single(symbol, tf, strat_filter):
    p, iv = ("1mo", "30m") if tf == "30분봉" else ("3mo", "1h")
    raw = yf.download(symbol, period=p, interval=iv, progress=False, auto_adjust=True)
    if raw is None or raw.empty: return None
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna().copy()
    if len(df) < 25: return None
    if df.index.tz is None: df.index = df.index.tz_localize('UTC')
    df_ny = df.tz_convert('America/New_York')
    df['Date_NY'], df['Time_NY'] = df_ny.index.date, df_ny.index.strftime('%H:%M')
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    d = df['Close'].diff()
    g, l = d.where(d > 0, 0).rolling(14).mean(), (-d.where(d < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + g / l.replace(0, np.nan)))
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Vol'] = tp * df['Volume']
    df['VWAP'] = df.groupby('Date_NY')['TP_Vol'].cumsum() / df.groupby('Date_NY')['Volume'].cumsum()
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Buy_Signal'], df['Sell_Signal'], df['Signal_Reason'] = False, False, ""
    fb, pc = df.groupby('Date_NY').first(), df.groupby('Date_NY')['Close'].last().shift(1)
    orb_h, d_open, f_close, p_close = fb['High'].to_dict(), fb['Open'].to_dict(), fb['Close'].to_dict(), pc.to_dict()
    for i in range(2, len(df)):
        idx, r, pv, pv2 = df.index[i], df.iloc[i], df.iloc[i-1], df.iloc[i-2]
        dt, tm = r['Date_NY'], r['Time_NY']
        oh, do, pcv = orb_h.get(dt, np.nan), d_open.get(dt, np.nan), p_close.get(dt, np.nan)
        is_golden = ("09:30" <= tm <= "11:30") or ("14:30" <= tm <= "15:30")
        trend_ok = (r['Close'] > r['VWAP']) and (r['Close'] > r['EMA50']) and (r['Close'] > r['EMA200']) and (r['EMA20'] > r['EMA50'])
        rng = max(r['High'] - r['Low'], 0.0001)
        strong_candle = (r['Close'] > r['Open']) and ((r['High'] - r['Close']) / rng <= 0.35)
        rsi_ok = (42 <= r['RSI'] <= 66)
        vol_ok = (r['Volume'] > r['Vol_MA20'] * 1.4)
        rb, rs = [], []
        if tm > "09:30" and r['Close'] > oh and pv['Close'] <= oh and r['Volume'] > r['Vol_MA20'] * 1.6: rb.append("ORB 돌파")
        if (pv['RSI'] <= 45 or pv2['RSI'] <= 45) and (r['Low'] <= r['EMA20'] * 1.003 or r['Low'] <= r['VWAP'] * 1.003): rb.append("VWAP·EMA 눌림")
        if (r['Low'] <= r['EMA20'] * 1.002) and ((min(r['Open'], r['Close']) - r['Low']) > abs(r['Close'] - r['Open']) * 0.6): rb.append("정배열 꼬리반등")
        if is_golden and trend_ok and strong_candle and rsi_ok and vol_ok and len(rb) >= 1:
            df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, "★A+강력매수: " + ", ".join(rb)
        if (r['Close'] < r['VWAP'] and pv['Close'] >= pv['VWAP'] and r['Close'] < r['EMA20']) or (pd.notna(pcv) and pcv > 0 and (0 < (do - pcv)/pcv <= 0.01) and (f_close.get(dt, np.nan) < do) and r['Close'] < r['VWAP']):
            rs.append("VWAP·단기추세 이탈")
        if rs: df.at[idx, 'Sell_Signal'], df.at[idx, 'Signal_Reason'] = True, ", ".join(rs)
    return df
import streamlit as st, yfinance as yf, pandas as pd, numpy as np, plotly.graph_objects as go
from plotly.subplots import make_subplots

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
.list-card { background-color:#131722; border:1px solid #1E293B; border-radius:8px; padding:9px 11px; margin-bottom:6px; }
.list-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; }
.list-sub { font-size:0.75rem; color:#94A3B8; display:flex; justify-content:space-between; }
div[data-testid="stExpander"], div[data-testid="stExpander"] details, div[data-testid="stExpander"] summary { background-color:#131722!important; color:#F8FAFC!important; border-radius:8px!important; }
div[data-testid="stExpander"] { border:1px solid #1E293B!important; }
div[data-testid="stExpander"] summary p, div[data-testid="stExpander"] summary svg { color:#F8FAFC!important; fill:#F8FAFC!important; font-size:0.82rem!important; font-weight:600!important; }
div[data-baseweb="input"]>div, div[data-baseweb="select"]>div { background-color:#F8FAFC!important; color:#0B0E14!important; border-radius:8px!important; }
div[data-baseweb="input"] input { color:#0B0E14!important; font-weight:600!important; }
label p, .stRadio label p, button[data-baseweb="tab"] p { color:#E2E8F0!important; font-size:0.82rem!important; }
.stButton>button { background-color:#1E293B!important; color:#F8FAFC!important; border:1px solid #334155!important; border-radius:8px!important; font-size:0.8rem!important; }
</style>""", unsafe_allow_html=True)

with st.expander("⚙️ 종목 / 분봉 설정 (눌러서 열기)", expanded=False):
    ticker = st.text_input("티커 (예: QQQ, NVDA, SOXL)", value="QQQ").upper().strip()
    timeframe = st.radio("봉 기준 (단타 전용)", ["30분봉", "1시간봉"], horizontal=True)
    strategy_filter = st.selectbox("전략 모드", ["30m·1h 최고승률 스나이퍼 (골든타임+추세+수급)"], label_visibility="collapsed")
    if st.button("🔄 데이터 새로고침", use_container_width=True): st.cache_data.clear()

@st.cache_data(ttl=60)
def load_single(symbol, tf, strat_filter):
    p, iv = ("1mo", "30m") if tf == "30분봉" else ("3mo", "1h")
    raw = yf.download(symbol, period=p, interval=iv, progress=False, auto_adjust=True)
    if raw is None or raw.empty: return None
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna().copy()
    if len(df) < 25: return None
    if df.index.tz is None: df.index = df.index.tz_localize('UTC')
    df_ny = df.tz_convert('America/New_York')
    df['Date_NY'], df['Time_NY'] = df_ny.index.date, df_ny.index.strftime('%H:%M')
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    d = df['Close'].diff()
    g, l = d.where(d > 0, 0).rolling(14).mean(), (-d.where(d < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + g / l.replace(0, np.nan)))
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Vol'] = tp * df['Volume']
    df['VWAP'] = df.groupby('Date_NY')['TP_Vol'].cumsum() / df.groupby('Date_NY')['Volume'].cumsum()
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Buy_Signal'], df['Sell_Signal'], df['Signal_Reason'] = False, False, ""
    fb, pc = df.groupby('Date_NY').first(), df.groupby('Date_NY')['Close'].last().shift(1)
    orb_h, d_open, f_close, p_close = fb['High'].to_dict(), fb['Open'].to_dict(), fb['Close'].to_dict(), pc.to_dict()
    for i in range(2, len(df)):
        idx, r, pv, pv2 = df.index[i], df.iloc[i], df.iloc[i-1], df.iloc[i-2]
        dt, tm = r['Date_NY'], r['Time_NY']
        oh, do, pcv = orb_h.get(dt, np.nan), d_open.get(dt, np.nan), p_close.get(dt, np.nan)
        is_golden = ("09:30" <= tm <= "11:30") or ("14:30" <= tm <= "15:30")
        trend_ok = (r['Close'] > r['VWAP']) and (r['Close'] > r['EMA50']) and (r['Close'] > r['EMA200']) and (r['EMA20'] > r['EMA50'])
        rng = max(r['High'] - r['Low'], 0.0001)
        strong_candle = (r['Close'] > r['Open']) and ((r['High'] - r['Close']) / rng <= 0.35)
        rsi_ok = (42 <= r['RSI'] <= 66)
        vol_ok = (r['Volume'] > r['Vol_MA20'] * 1.4)
        rb, rs = [], []
        if tm > "09:30" and r['Close'] > oh and pv['Close'] <= oh and r['Volume'] > r['Vol_MA20'] * 1.6: rb.append("ORB 돌파")
        if (pv['RSI'] <= 45 or pv2['RSI'] <= 45) and (r['Low'] <= r['EMA20'] * 1.003 or r['Low'] <= r['VWAP'] * 1.003): rb.append("VWAP·EMA 눌림")
        if (r['Low'] <= r['EMA20'] * 1.002) and ((min(r['Open'], r['Close']) - r['Low']) > abs(r['Close'] - r['Open']) * 0.6): rb.append("정배열 꼬리반등")
        if is_golden and trend_ok and strong_candle and rsi_ok and vol_ok and len(rb) >= 1:
            df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, "★A+강력매수: " + ", ".join(rb)
        if (r['Close'] < r['VWAP'] and pv['Close'] >= pv['VWAP'] and r['Close'] < r['EMA20']) or (pd.notna(pcv) and pcv > 0 and (0 < (do - pcv)/pcv <= 0.01) and (f_close.get(dt, np.nan) < do) and r['Close'] < r['VWAP']):
            rs.append("VWAP·단기추세 이탈")
        if rs: df.at[idx, 'Sell_Signal'], df.at[idx, 'Signal_Reason'] = True, ", ".join(rs)
    return df
