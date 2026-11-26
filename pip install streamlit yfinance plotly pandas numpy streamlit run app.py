import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 페이지 설정 및 다크 테마 커스텀 CSS (#0B0E14)
st.set_page_config(page_title="US Stock Quant Dashboard", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp {
        background-color: #0B0E14;
        color: #E2E8F0;
    }
    .metric-card {
        background-color: #131722;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .badge-buy {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid #10B981;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-sell {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid #EF4444;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-neutral {
        background-color: rgba(148, 163, 184, 0.15);
        color: #94A3B8;
        border: 1px solid #475569;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    div[data-baseweb="input"] > div {
        background-color: #131722 !important;
        color: white !important;
        border-color: #334155 !important;
    }
</style>
""", unsafe_allow_html=True)

# 2. 상단 컨트롤 패널 (종목 검색 & 타임프레임 & 새로고침)
col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([2, 2, 2, 1])

with col_ctrl1:
    ticker = st.text_input("🔍 미국 주식 티커 검색 (예: QQQ, NVDA, TSLA, AAPL, SOXL)", value="QQQ").upper().strip()

with col_ctrl2:
    timeframe = st.radio("⏱️ 조회 봉 기준", ["30분봉", "1시간봉", "일봉"], horizontal=True)

with col_ctrl3:
    strategy_filter = st.selectbox(
        "🎯 적용 전략 필터",
        ["전체 전략 통합", "1. ORB (오프닝 레인지 돌파)", "2. VWAP + RSI 눌림목", "3. EMA 정배열 회귀", "4. 갭 채우기 (Gap Fill) 매도"]
    )

with col_ctrl4:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 데이터 새로고침", use_container_width=True):
        st.cache_data.clear()

# 3. 데이터 로드 및 지표 계산 함수
@st.cache_data(ttl=60)
def load_and_process_data(symbol, tf):
    if tf == "30분봉":
        period, interval = "1mo", "30m"
    elif tf == "1시간봉":
        period, interval = "3mo", "1h"
    else:
        period, interval = "2y", "1d"

    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        return None

    # 멀티인덱스 컬럼 정리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().copy()

    # 시간대 변환 (미국 뉴욕 시간 기준으로 세션 구분)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df_ny = df.tz_convert('America/New_York')
    df['Date_NY'] = df_ny.index.date
    df['Time_NY'] = df_ny.index.strftime('%H:%M')

    # 기본 기술적 지표 계산
    # EMA (20, 50, 200)
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # 거래량 이동평균 (20)
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()

    # 일별 VWAP 계산 (분봉일 경우 당일 누적, 일봉일 경우 20일 롤링 VWAP)
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Vol'] = df['Typical_Price'] * df['Volume']

    if tf in ["30분봉", "1시간봉"]:
        df['Cum_TP_Vol'] = df.groupby('Date_NY')['TP_Vol'].cumsum()
        df['Cum_Vol'] = df.groupby('Date_NY')['Volume'].cumsum()
        df['VWAP'] = df['Cum_TP_Vol'] / df['Cum_Vol']
    else:
        df['VWAP'] = df['TP_Vol'].rolling(20).sum() / df['Volume'].rolling(20).sum()

    # 신호 컬럼 초기화
    df['Buy_Signal'] = False
    df['Sell_Signal'] = False
    df['Signal_Reason'] = ""

    # 전략별 시그널 계산
    if tf in ["30분봉", "1시간봉"]:
        # 일별 첫 번째 봉(09:30 NY) 정보 추출 (ORB 및 갭 계산용)
        first_bars = df.groupby('Date_NY').first()
        prev_closes = df.groupby('Date_NY')['Close'].last().shift(1)

        daily_orb_high = first_bars['High'].to_dict()
        daily_open = first_bars['Open'].to_dict()
        daily_first_close = first_bars['Close'].to_dict()
        daily_prev_close = prev_closes.to_dict()

        for i in range(1, len(df)):
            idx = df.index[i]
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            d = row['Date_NY']
            t = row['Time_NY']

            orb_h = daily_orb_high.get(d, np.nan)
            day_open = daily_open.get(d, np.nan)
            prev_c = daily_prev_close.get(d, np.nan)

            reasons_buy = []
            reasons_sell = []

            # [전략 1] ORB (오프닝 레인지 돌파): 첫 봉(09:30) 이후 당일 첫 봉 고점을 거래량 1.5배 이상 실어 돌파
            if t > "09:30" and row['Close'] > orb_h and prev['Close'] <= orb_h and row['Volume'] > row['Vol_MA20'] * 1.5:
                reasons_buy.append("ORB 돌파")

            # [전략 2] VWAP + RSI 눌림목: 주가가 VWAP 위 & 직전 봉 RSI <= 40 과매도 눌림 후 양봉 전환
            if row['Close'] > row['VWAP'] and prev['RSI'] <= 40 and row['Close'] > row['Open']:
                reasons_buy.append("VWAP+RSI 눌림")

            # [전략 3] EMA 정배열 회귀: 20 > 50 > 200 정배열 + 주가가 20 또는 50 EMA 터치 + 아래꼬리 양봉
            is_aligned = (row['EMA20'] > row['EMA50']) and (row['EMA50'] > row['EMA200'])
            touched_ema = (row['Low'] <= row['EMA20'] * 1.002) or (row['Low'] <= row['EMA50'] * 1.002)
            lower_wick = min(row['Open'], row['Close']) - row['Low']
            body = abs(row['Close'] - row['Open'])
            if is_aligned and touched_ema and (row['Close'] > row['Open']) and (lower_wick > body * 0.5):
                reasons_buy.append("EMA 정배열 회귀")

            # [전략 4] 갭 채우기 (Gap Fill) 역추세 매도(S): 시초가 0~1% 갭상승 + 첫 봉 음봉 마감 + VWAP 하향 이탈
            if pd.notna(prev_c) and prev_c > 0:
                gap_pct = (day_open - prev_c) / prev_c
                first_c = daily_first_close.get(d, np.nan)
                if (0 < gap_pct <= 0.01) and (first_c < day_open) and (row['Close'] < row['VWAP']) and (prev['Close'] >= prev['VWAP']):
                    reasons_sell.append("갭 채우기 이탈")

            # 필터 적용
            if strategy_filter == "1. ORB (오프닝 레인지 돌파)":
                reasons_buy = [r for r in reasons_buy if "ORB" in r]
                reasons_sell = []
            elif strategy_filter == "2. VWAP + RSI 눌림목":
                reasons_buy = [r for r in reasons_buy if "VWAP" in r]
                reasons_sell = []
            elif strategy_filter == "3. EMA 정배열 회귀":
                reasons_buy = [r for r in reasons_buy if "EMA" in r]
                reasons_sell = []
            elif strategy_filter == "4. 갭 채우기 (Gap Fill) 매도":
                reasons_buy = []

            if reasons_buy:
                df.at[idx, 'Buy_Signal'] = True
                df.at[idx, 'Signal_Reason'] = ", ".join(reasons_buy)
            if reasons_sell:
                df.at[idx, 'Sell_Signal'] = True
                df.at[idx, 'Signal_Reason'] = ", ".join(reasons_sell)
    else:
        # 일봉 선택 시: EMA 정배열 눌림목 & VWAP+RSI 눌림목 자동 적용
        for i in range(1, len(df)):
            idx = df.index[i]
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            if row['Close'] > row['VWAP'] and prev['RSI'] <= 40 and row['Close'] > row['Open']:
                df.at[idx, 'Buy_Signal'] = True
                df.at[idx, 'Signal_Reason'] = "일봉 VWAP+RSI 눌림"
            elif (row['EMA20'] > row['EMA50'] > row['EMA200']) and (row['Low'] <= row['EMA20']) and (row['Close'] > row['Open']):
                df.at[idx, 'Buy_Signal'] = True
                df.at[idx, 'Signal_Reason'] = "일봉 EMA20 지지"

    return df

# 데이터 불러오기 실행
with st.spinner(f"'{ticker}' 실시간 주가 및 퀀트 데이터 분석 중..."):
    df = load_and_process_data(ticker, timeframe)

if df is None or len(df) < 5:
    st.error(f"❌ '{ticker}' 종목 데이터를 불러올 수 없습니다. 정확한 미국 주식 티커(예: AAPL, NVDA, QQQ, TQQQ)를 입력해 주세요.")
    st.stop()

# 4. 상단 요약 KPI 카드 4개
latest = df.iloc[-1]
prev_bar = df.iloc[-2]
price_diff = latest['Close'] - prev_bar['Close']
price_pct = (price_diff / prev_bar['Close']) * 100

buy_df = df[df['Buy_Signal']]
sell_df = df[df['Sell_Signal']]

last_buy_str = buy_df.index[-1].strftime('%m-%d %H:%M') if not buy_df.empty else "신호 없음"
last_buy_reason = buy_df.iloc[-1]['Signal_Reason'] if not buy_df.empty else "-"
last_sell_str = sell_df.index[-1].strftime('%m-%d %H:%M') if not sell_df.empty else "신호 없음"

# 현재 상태 판정 배지
if latest['Buy_Signal']:
    status_html = f'<span class="badge-buy">🟢 매수 신호 발생 ({latest["Signal_Reason"]})</span>'
elif latest['Sell_Signal']:
    status_html = f'<span class="badge-sell">🔴 매도/숏 신호 발생 ({latest["Signal_Reason"]})</span>'
elif latest['Close'] > latest['VWAP']:
    status_html = '<span class="badge-buy">📈 VWAP 상단 (매수 우위 구간)</span>'
else:
    status_html = '<span class="badge-neutral">⏳ 관망 구간 (VWAP 하단)</span>'

c1, c2, c3, c4 = st.columns(4)

with c1:
    color_p = "#10B981" if price_diff >= 0 else "#EF4444"
    sign = "+" if price_diff >= 0 else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{ticker} 현재가 ({timeframe})</div>
        <div class="metric-value">${latest['Close']:,.2f} <span style="font-size:0.95rem; color:{color_p};">({sign}{price_pct:.2f}%)</span></div>
