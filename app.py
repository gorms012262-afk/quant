import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json

st.set_page_config(page_title="US Stock Quant Dashboard Pro", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""<style>
.stApp { background-color:#0B0E14; color:#E2E8F0; }
header[data-testid="stHeader"] { background-color:#0B0E14; }
.block-container { padding-top:2.8rem!important; padding-bottom:2rem!important; padding-left:0.8rem!important; padding-right:0.8rem!important; }
.kpi-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:6px; margin:6px 0 8px 0; }
.metric-card { background-color:#131722; border:1px solid #1E293B; border-radius:8px; padding:9px 10px; }
.metric-title { font-size:0.72rem; color:#94A3B8; margin-bottom:2px; }
.metric-value { font-size:1.05rem; font-weight:700; color:#F8FAFC; }
.badge-buy { background-color:rgba(16,185,129,0.15); color:#10B981; border:1px solid #10B981; padding:2px 6px; border-radius:5px; font-size:0.72rem; font-weight:600; display:inline-block; }
.badge-sell { background-color:rgba(239,68,68,0.15); color:#EF4444; border:1px solid #EF4444; padding:2px 6px; border-radius:5px; font-size:0.72rem; font-weight:600; display:inline-block; }
.badge-neutral { background-color:rgba(148,163,184,0.15); color:#94A3B8; border:1px solid #475569; padding:2px 6px; border-radius:5px; font-size:0.72rem; font-weight:600; display:inline-block; }
.ohlc-bar { display:grid; grid-template-columns:repeat(4,1fr); gap:4px; background-color:#131722; border:1px solid #1E293B; border-radius:8px; padding:7px 6px; margin-bottom:8px; text-align:center; }
.ohlc-lbl { font-size:0.68rem; color:#64748B; margin-bottom:1px; }
.ohlc-val { font-size:0.82rem; font-weight:700; color:#F8FAFC; }
.tpsl-bar { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; background-color:#131722; border:1px solid #334155; border-radius:8px; padding:8px; margin-bottom:8px; text-align:center; }
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

# --- 설정 영역 ---
with st.expander("⚙️ 종목 / 전략 / ATR 리스크 / 알림 설정 (눌러서 열기)", expanded=False):
    ticker = st.text_input("메인 분석 티커 (예: QQQ, NVDA, TSLA)", value="QQQ").upper().strip()
    timeframe = st.radio("봉 기준", ["30분봉", "1시간봉", "일봉"], horizontal=True)
    strategy_filter = st.selectbox(
        "전략 필터",
        ["전체 전략 통합 (고승률 엄선)", "ORB 돌파", "VWAP+RSI 눌림", "EMA 정배열 회귀", "갭 채우기 이탈"]
    )
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        hold_bars = st.slider("백테스트 보유봉", 1, 20, 5)
    with col_s2:
        sl_mult = st.number_input("손절 ATR 배수", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    with col_s3:
        tp_mult = st.number_input("익절 ATR 배수", min_value=1.0, max_value=10.0, value=3.0, step=0.5)

    watchlist_str = st.text_input("스캐너 관심종목 (쉼표 구분)", value="QQQ, SPY, NVDA, TSLA, AAPL, MSFT, AMD, META")
    st.markdown("---")
    discord_webhook = st.text_input("디스코드 웹훅 URL (선택)", type="password", placeholder="https://discord.com...")
    kakao_token = st.text_input("카카오톡 나에게 보내기 Access Token (선택)", type="password", placeholder="Kakao REST API Access Token")

    if st.button("🔄 데이터 새로고침", use_container_width=True):
        st.cache_data.clear()

# --- 알림 전송 함수 ---
def send_discord_alert(webhook_url, msg):
    if not webhook_url:
        return False, "디스코드 웹훅 URL이 입력되지 않았습니다."
    try:
        res = requests.post(webhook_url, json={"content": msg}, timeout=5)
        if res.status_code in:
            return True, "디스코드 알림 전송 성공!"
        return False, f"디스코드 전송 실패 (코드: {res.status_code})"
    except Exception as e:
        return False, f"오류 발생: {e}"

def send_kakao_alert(access_token, msg):
    if not access_token:
        return False, "카카오톡 액세스 토큰이 입력되지 않았습니다."
    url = "https://kakao.com"
    headers = {"Authorization": f"Bearer {access_token}"}
    template = {
        "object_type": "text",
        "text": msg[:200],
        "link": {"web_url": "https://yahoo.com", "mobile_web_url": "https://yahoo.com"},
        "button_title": "차트 확인"
    }
    try:
        res = requests.post(url, headers=headers, data={"template_object": json.dumps(template)}, timeout=5)
        if res.status_code == 200:
            return True, "카카오톡 나에게 보내기 성공!"
        return False, f"카카오톡 전송 실패: {res.text}"
    except Exception as e:
        return False, f"오류 발생: {e}"

# --- 데이터 로드 및 지표/시그널 연산 ---
@st.cache_data(ttl=60)
def load_single(symbol, tf, strat_filter):
    p, iv = ("1mo", "30m") if tf == "30분봉" else (("3mo", "1h") if tf == "1시간봉" else ("2y", "1d"))
    raw = yf.download(symbol, period=p, interval=iv, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna().copy()
    if len(df) < 15:
        return None

    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df_ny = df.tz_convert('America/New_York')
    df['Date_NY'], df['Time_NY'] = df_ny.index.date, df_ny.index.strftime('%H:%M')

    # 이동평균선
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # RSI (14)
    d = df['Close'].diff()
    g, l = d.where(d > 0, 0).rolling(14).mean(), (-d.where(d < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + g / l.replace(0, np.nan)))

    # ATR (14) 계산
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean().bfill()

    # VWAP & 거래량 이평
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Vol'] = tp * df['Volume']
    df['VWAP'] = (
        (df.groupby('Date_NY')['TP_Vol'].cumsum() / df.groupby('Date_NY')['Volume'].cumsum())
        if tf in ["30분봉", "1시간봉"]
        else (df['TP_Vol'].rolling(20).sum() / df['Volume'].rolling(20).sum())
    )
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Buy_Signal'], df['Sell_Signal'], df['Signal_Reason'] = False, False, ""

    if tf in ["30분봉", "1시간봉"]:
        fb, pc = df.groupby('Date_NY').first(), df.groupby('Date_NY')['Close'].last().shift(1)
        orb_h, d_open, f_close, p_close = fb['High'].to_dict(), fb['Open'].to_dict(), fb['Close'].to_dict(), pc.to_dict()
        for i in range(1, len(df)):
            idx, r, pv = df.index[i], df.iloc[i], df.iloc[i - 1]
            dt, tm = r['Date_NY'], r['Time_NY']
            oh, do, pcv = orb_h.get(dt, np.nan), d_open.get(dt, np.nan), p_close.get(dt, np.nan)
            rb, rs = [], []

            if tm > "09:30" and r['Close'] > oh and pv['Close'] <= oh and r['Volume'] > r['Vol_MA20'] * 1.5:
                rb.append("ORB 돌파")
            if r['Close'] > r['VWAP'] and pv['RSI'] <= 40 and r['Close'] > r['Open']:
                rb.append("VWAP+RSI 눌림")
            if (r['EMA20'] > r['EMA50'] > r['EMA200']) and ((r['Low'] <= r['EMA20'] * 1.002) or (r['Low'] <= r['EMA50'] * 1.002)) and (r['Close'] > r['Open']) and ((min(r['Open'], r['Close']) - r['Low']) > abs(r['Close'] - r['Open']) * 0.5):
                rb.append("EMA 정배열 회귀")
            if pd.notna(pcv) and pcv > 0 and (0 < (do - pcv) / pcv <= 0.01) and (f_close.get(dt, np.nan) < do) and (r['Close'] < r['VWAP']) and (pv['Close'] >= pv['VWAP']):
                rs.append("갭 채우기 이탈")

            if not ((r['Close'] > r['EMA50'] or r['EMA20'] > r['EMA50']) and (r['RSI'] < 68)):
                rb = []
            if not (r['Close'] < r['EMA20'] and r['RSI'] > 32):
                rs = []

            if strat_filter != "전체 전략 통합 (고승률 엄선)":
                rb = [x for x in rb if strat_filter in x]
                rs = [x for x in rs if strat_filter in x]
                if rb:
                    df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, "매수: " + ", ".join(rb)
                if rs:
                    df.at[idx, 'Sell_Signal'], df.at[idx, 'Signal_Reason'] = True, ", ".join(rs)
            else:
                if rb and (len(rb) >= 2 or r['Volume'] > r['Vol_MA20'] * 1.8):
                    df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, "★강력매수: " + ", ".join(rb)
                if rs:
                    df.at[idx, 'Sell_Signal'], df.at[idx, 'Signal_Reason'] = True, ", ".join(rs)
    else:
        for i in range(1, len(df)):
