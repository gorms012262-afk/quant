import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="US Stock Quant Dashboard", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""<style>
.stApp { background-color: #0B0E14; color: #E2E8F0; }
.block-container { padding-top: 0.6rem !important; padding-bottom: 1rem !important; padding-left: 0.6rem !important; padding-right: 0.6rem !important; }
header[data-testid="stHeader"] { background-color: #0B0E14 !important; height: 2.2rem !important; }
.kpi-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; margin-bottom: 8px; }
.metric-card { background-color: #131722; border: 1px solid #1E293B; border-radius: 8px; padding: 8px 10px; }
.metric-title { font-size: 0.72rem; color: #94A3B8; margin-bottom: 2px; }
.metric-value { font-size: 1.1rem; font-weight: 700; color: #F8FAFC; }
.badge-buy { background-color: rgba(16,185,129,0.15); color: #10B981; border: 1px solid #10B981; padding: 2px 6px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; display: inline-block; }
.badge-sell { background-color: rgba(239,68,68,0.15); color: #EF4444; border: 1px solid #EF4444; padding: 2px 6px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; display: inline-block; }
.badge-neutral { background-color: rgba(148,163,184,0.15); color: #94A3B8; border: 1px solid #475569; padding: 2px 6px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; display: inline-block; }
div[data-testid="stExpander"] { background-color: #131722 !important; border: 1px solid #1E293B !important; border-radius: 8px !important; margin-bottom: 8px !important; }
div[data-testid="stExpander"] summary p { color: #E2E8F0 !important; font-size: 0.85rem !important; font-weight: 600 !important; }
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { background-color: #0B0E14 !important; color: #F8FAFC !important; border-color: #334155 !important; }
div[data-baseweb="input"] input { color: #F8FAFC !important; }
label p, .stRadio label p { color: #E2E8F0 !important; font-size: 0.82rem !important; }
.stButton > button { background-color: #1E293B !important; color: #F8FAFC !important; border: 1px solid #334155 !important; }
</style>""", unsafe_allow_html=True)

with st.expander("⚙️ 종목 검색 / 봉 기준 / 전략 필터", expanded=False):
    c1, c2 = st.columns([1.2, 1.8])
    with c1:
        ticker = st.text_input("티커 (예: QQQ, NVDA, TSLA)", value="QQQ").upper().strip()
    with c2:
        timeframe = st.radio("봉 기준", ["30분봉", "1시간봉", "일봉"], horizontal=True)
    c3, c4 = st.columns([2.2, 1])
    with c3:
        strategy_filter = st.selectbox("전략 필터", ["전체 전략 통합", "1. ORB (오프닝 레인지 돌파)", "2. VWAP + RSI 눌림목", "3. EMA 정배열 회귀", "4. 갭 채우기 (Gap Fill) 매도"], label_visibility="collapsed")
    with c4:
        if st.button("🔄 새로고침", use_container_width=True):
            st.cache_data.clear()

@st.cache_data(ttl=60)
def load_and_process_data(symbol, tf):
    period, interval = ("1mo", "30m") if tf == "30분봉" else (("3mo", "1h") if tf == "1시간봉" else ("2y", "1d"))
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna().copy()

    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df_ny = df.tz_convert('America/New_York')
    df['Date_NY'] = df_ny.index.date
    df['Time_NY'] = df_ny.index.strftime('%H:%M')

    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()

    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Vol'] = df['Typical_Price'] * df['Volume']
    if tf in ["30분봉", "1시간봉"]:
        df['VWAP'] = df.groupby('Date_NY')['TP_Vol'].cumsum() / df.groupby('Date_NY')['Volume'].cumsum()
    else:
        df['VWAP'] = df['TP_Vol'].rolling(20).sum() / df['Volume'].rolling(20).sum()

    df['Buy_Signal'] = False
    df['Sell_Signal'] = False
    df['Signal_Reason'] = ""

    if tf in ["30분봉", "1시간봉"]:
        first_bars = df.groupby('Date_NY').first()
        prev_closes = df.groupby('Date_NY')['Close'].last().shift(1)
        daily_orb_high = first_bars['High'].to_dict()
        daily_open = first_bars['Open'].to_dict()
        daily_first_close = first_bars['Close'].to_dict()
        daily_prev_close = prev_closes.to_dict()

        for i in range(1, len(df)):
            idx = df.index[i]
            row, prev = df.iloc[i], df.iloc[i - 1]
            d, t = row['Date_NY'], row['Time_NY']
            orb_h, day_open, prev_c = daily_orb_high.get(d, np.nan), daily_open.get(d, np.nan), daily_prev_close.get(d, np.nan)
            reasons_buy, reasons_sell = [], []

            if t > "09:30" and row['Close'] > orb_h and prev['Close'] <= orb_h and row['Volume'] > row['Vol_MA20'] * 1.5:
                reasons_buy.append("ORB 돌파")
            if row['Close'] > row['VWAP'] and prev['RSI'] <= 40 and row['Close'] > row['Open']:
                reasons_buy.append("VWAP+RSI 눌림")
            is_aligned = (row['EMA20'] > row['EMA50']) and (row['EMA50'] > row['EMA200'])
            touched_ema = (row['Low'] <= row['EMA20'] * 1.002) or (row['Low'] <= row['EMA50'] * 1.002)
            lower_wick = min(row['Open'], row['Close']) - row['Low']
            body = abs(row['Close'] - row['Open'])
            if is_aligned and touched_ema and (row['Close'] > row['Open']) and (lower_wick > body * 0.5):
                reasons_buy.append("EMA 정배열 회귀")
            if pd.notna(prev_c) and prev_c > 0:
                gap_pct = (day_open - prev_c) / prev_c
                first_c = daily_first_close.get(d, np.nan)
                if (0 < gap_pct <= 0.01) and (first_c < day_open) and (row['Close'] < row['VWAP']) and (prev['Close'] >= prev['VWAP']):
                    reasons_sell.append("갭 채우기 이탈")

            if strategy_filter == "1. ORB (오프닝 레인지 돌파)":
                reasons_buy, reasons_sell = [r for r in reasons_buy if "ORB" in r], []
            elif strategy_filter == "2. VWAP + RSI 눌림목":
                reasons_buy, reasons_sell = [r for r in reasons_buy if "VWAP" in r], []
            elif strategy_filter == "3. EMA 정배열 회귀":
                reasons_buy, reasons_sell = [r for r in reasons_buy if "EMA" in r], []
            elif strategy_filter == "4. 갭 채우기 (Gap Fill) 매도":
                reasons_buy = []

            if reasons_buy:
                df.at[idx, 'Buy_Signal'] = True
                df.at[idx, 'Signal_Reason'] = ", ".join(reasons_buy)
            if reasons_sell:
                df.at[idx, 'Sell_Signal'] = True
                df.at[idx, 'Signal_Reason'] = ", ".join(reasons_sell)
    else:
        for i in range(1, len(df)):
            idx = df.index[i]
            row, prev = df.iloc[i], df.iloc[i - 1]
            if row['Close'] > row['VWAP'] and prev['RSI'] <= 40 and row['Close'] > row['Open']:
                df.at[idx, 'Buy_Signal'] = True
                df.at[idx, 'Signal_Reason'] = "일봉 VWAP+RSI 눌림"
            elif (row['EMA20'] > row['EMA50'] > row['EMA200']) and (row['Low'] <= row['EMA20']) and (row['Close'] > row['Open']):
                df.at[idx, 'Buy_Signal'] = True
                df.at[idx, 'Signal_Reason'] = "일봉 EMA20 지지"
    return df

df = load_and_process_data(ticker, timeframe)

if df is None or len(df) < 5:
    st.error(f"❌ '{ticker}' 종목 데이터를 불러올 수 없습니다.")
    st.stop()

latest, prev_bar = df.iloc[-1], df.iloc[-2]
price_diff = latest['Close'] - prev_bar['Close']
price_pct = (price_diff / prev_bar['Close']) * 100
buy_df, sell_df = df[df['Buy_Signal']], df[df['Sell_Signal']]
last_buy_str = buy_df.index[-1].strftime('%m-%d %H:%M') if not buy_df.empty else "없음"
last_buy_reason = buy_df.iloc[-1]['Signal_Reason'] if not buy_df.empty else "-"
last_sell_str = sell_df.index[-1].strftime('%m-%d %H:%M') if not sell_df.empty else "없음"

if latest['Buy_Signal']:
    status_html = f'<span class="badge-buy">🟢 {latest["Signal_Reason"]}</span>'
elif latest['Sell_Signal']:
    status_html = f'<span class="badge-sell">🔴 {latest["Signal_Reason"]}</span>'
elif latest['Close'] > latest['VWAP']:
    status_html = '<span class="badge-buy">📈 VWAP 상단</span>'
else:
    status_html = '<span class="badge-neutral">⏳ VWAP 하단</span>'

color_p = "#10B981" if price_diff >= 0 else "#EF4444"
sign = "+" if price_diff >= 0 else ""
rsi_val = latest['RSI']
rsi_col = "#EF4444" if rsi_val >= 70 else ("#10B981" if rsi_val <= 30 else "#F8FAFC")

st.markdown(f"""
<div class="kpi-grid">
    <div class="metric-card">
        <div class="metric-title">{ticker} ({timeframe})</div>
        <div class="metric-value">${latest["Close"]:,.2f} <span style="font-size:0.78rem;color:{color_p};">({sign}{price_pct:.2f}%)</span></div>
    </div>
    <div class="metric-card">
        <div class="metric-title">현재 추세 판정</div>
        <div style="margin-top:2px;">{status_html}</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">최근 매수 ({last_buy_str})</div>
        <div class="metric-value" style="font-size:0.9rem;color:#10B981;">{last_buy_reason}</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">RSI(14) / 매도({last_sell_str})</div>
        <div class="metric-value" style="color:{rsi_col};">{rsi_val:.1f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

plot_df = df.tail(60).copy().reset_index(drop=True)
x_idx = list(range(len(plot_df)))
tick_step = max(1, len(plot_df) // 5)
