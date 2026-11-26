import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="US Stock Quant Dashboard", layout="wide", initial_sidebar_state="collapsed")

with st.container(border=True):
    ticker = st.text_input("티커 (예: QQQ, SOXL, NVDA, TSLA)", value="QQQ").upper().strip()
    timeframe = st.radio("봉 기준", ["30분봉", "1시간봉", "일봉"], horizontal=True, label_visibility="collapsed")
    strategy_filter = st.selectbox("전략 필터", ["전체 전략 통합", "1. ORB (오프닝 레인지 돌파)", "2. VWAP + RSI 눌림목", "3. EMA 정배열 회귀", "4. 갭 채우기 (Gap Fill) 매도"], label_visibility="collapsed")
    if st.button("🔄 새로고침", use_container_width=True):
        st.cache_data.clear()

@st.cache_data(ttl=60)
def load_data(symbol, tf):
    p, iv = ("1mo", "30m") if tf == "30분봉" else (("3mo", "1h") if tf == "1시간봉" else ("2y", "1d"))
    df = yf.download(symbol, period=p, interval=iv, progress=False, auto_adjust=True)
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
    d = df['Close'].diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    l = (-d.where(d < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + g / l.replace(0, np.nan)))
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Vol'] = tp * df['Volume']
    df['VWAP'] = (df.groupby('Date_NY')['TP_Vol'].cumsum() / df.groupby('Date_NY')['Volume'].cumsum()) if tf in ["30분봉", "1시간봉"] else (df['TP_Vol'].rolling(20).sum() / df['Volume'].rolling(20).sum())
    df['Buy_Signal'] = False
    df['Sell_Signal'] = False
    df['Signal_Reason'] = ""

    if tf in ["30분봉", "1시간봉"]:
        fb = df.groupby('Date_NY').first()
        pc = df.groupby('Date_NY')['Close'].last().shift(1)
        orb_h = fb['High'].to_dict()
        d_open = fb['Open'].to_dict()
        f_close = fb['Close'].to_dict()
        p_close = pc.to_dict()
        for i in range(1, len(df)):
            idx = df.index[i]
            r = df.iloc[i]
            pv = df.iloc[i-1]
            dt = r['Date_NY']
            tm = r['Time_NY']
            oh = orb_h.get(dt, np.nan)
            do = d_open.get(dt, np.nan)
            pcv = p_close.get(dt, np.nan)
            rb = []
            rs = []
            if tm > "09:30" and r['Close'] > oh and pv['Close'] <= oh and r['Volume'] > r['Vol_MA20'] * 1.5:
                rb.append("ORB 돌파")
            if r['Close'] > r['VWAP'] and pv['RSI'] <= 40 and r['Close'] > r['Open']:
                rb.append("VWAP+RSI 눌림")
            if (r['EMA20'] > r['EMA50'] > r['EMA200']) and ((r['Low'] <= r['EMA20']*1.002) or (r['Low'] <= r['EMA50']*1.002)) and (r['Close'] > r['Open']) and ((min(r['Open'], r['Close']) - r['Low']) > abs(r['Close'] - r['Open']) * 0.5):
                rb.append("EMA 정배열 회귀")
            if pd.notna(pcv) and pcv > 0 and (0 < (do - pcv)/pcv <= 0.01) and (f_close.get(dt, np.nan) < do) and (r['Close'] < r['VWAP']) and (pv['Close'] >= pv['VWAP']):
                rs.append("갭 채우기 이탈")
            if strategy_filter == "1. ORB (오프닝 레인지 돌파)":
                rb = [x for x in rb if "ORB" in x]
                rs = []
            elif strategy_filter == "2. VWAP + RSI 눌림목":
                rb = [x for x in rb if "VWAP" in x]
                rs = []
            elif strategy_filter == "3. EMA 정배열 회귀":
                rb = [x for x in rb if "EMA" in x]
                rs = []
            elif strategy_filter == "4. 갭 채우기 (Gap Fill) 매도":
                rb = []
            if rb:
                df.at[idx, 'Buy_Signal'] = True
                df.at[idx, 'Signal_Reason'] = ", ".join(rb)
            if rs:
                df.at[idx, 'Sell_Signal'] = True
                df.at[idx, 'Signal_Reason'] = ", ".join(rs)
    else:
        for i in range(1, len(df)):
            idx = df.index[i]
            r = df.iloc[i]
            pv = df.iloc[i-1]
            if r['Close'] > r['VWAP'] and pv['RSI'] <= 40 and r['Close'] > r['Open']:
                df.at[idx, 'Buy_Signal'] = True
                df.at[idx, 'Signal_Reason'] = "일봉 VWAP+RSI 눌림"
            elif (r['EMA20'] > r['EMA50'] > r['EMA200']) and (r['Low'] <= r['EMA20']) and (r['Close'] > r['Open']):
                df.at[idx, 'Buy_Signal'] = True
                df.at[idx, 'Signal_Reason'] = "일봉 EMA20 지지"
    return df

df = load_data(ticker, timeframe)
if df is None or len(df) < 5:
    st.error(f"❌ '{ticker}' 종목 데이터를 불러올 수 없습니다.")
    st.stop()

latest = df.iloc[-1]
prev_bar = df.iloc[-2]
p_diff = latest['Close'] - prev_bar['Close']
p_pct = (p_diff / prev_bar['Close']) * 100

c1, c2 = st.columns(2)
c1.metric(f"{ticker} ({timeframe})", f"${latest['Close']:,.2f}", f"{p_pct:+.2f}%")
c2.metric("RSI(14) / VWAP", f"{latest['RSI']:.1f}", f"VWAP ${latest['VWAP']:,.2f}")

b_df = df[df['Buy_Signal']]
if latest['Buy_Signal']:
    st.success(f"🟢 현재 매수 시그널 발생: {latest['Signal_Reason']}")
elif latest['Sell_Signal']:
    st.error(f"🔴 현재 매도 시그널 발생: {latest['Signal_Reason']}")
elif not b_df.empty:
    st.info(f"📌 최근 매수 시그널 ({b_df.index[-1].strftime('%m-%d %H:%M')}): {b_df.iloc[-1]['Signal_Reason']}")

st.subheader(f"🕯️ {ticker} 실시간 캔들 차트 ({timeframe})")

pdf = df.tail(50).copy()
fmt = '%m/%d' if timeframe == "일봉" else '%m/%d %H:%M'
pdf['DateStr'] = pdf.index.strftime(fmt)
pdf['Color'] = np.where(pdf['Close'] >= pdf['Open'], '#EF4444', '#3B82F6')

y_min = float(pdf['Low'].min()) * 0.998
y_max = float(pdf['High'].max()) * 1.002
y_dom = list((y_min, y_max))

base = alt.Chart(pdf).encode(
    x=alt.X('DateStr:N', sort=None, axis=alt.Axis(title=None, labelAngle=0, labelOverlap=True))
)

wick = base.mark_rule(size=1.5).encode(
    y=alt.Y('Low:Q', scale=alt.Scale(domain=y_dom), axis=alt.Axis(orient='right', title=None)),
    y2='High:Q',
    color=alt.Color('Color:N', scale=None)
)

body = base.mark_bar(size=5).encode(
    y='Open:Q',
    y2='Close:Q',
    color=alt.Color('Color:N', scale=None),
    tooltip=list(('DateStr:N', 'Open:Q', 'High:Q', 'Low:Q', 'Close:Q', 'Volume:Q'))
)

candle_chart = (wick + body).properties(height=300)
st.altair_chart(candle_chart, use_container_width=True)

st.subheader("📊 거래량 (Volume)")
vol_chart = alt.Chart(pdf).mark_bar(size=5).encode(
    x=alt.X('DateStr:N', sort=None, axis=alt.Axis(title=None, labelAngle=0, labelOverlap=True)),
    y=alt.Y('Volume:Q', axis=alt.Axis(orient='right', title=None, format='~s')),
    color=alt.Color('Color:N', scale=None)
).properties(height=120)

st.altair_chart(vol_chart, use_container_width=True)
