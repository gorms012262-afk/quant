import io, base64, streamlit as st, yfinance as yf, pandas as pd, numpy as np, mplfinance as mpf, matplotlib.pyplot as plt

st.set_page_config(page_title="US Stock Quant Dashboard", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""<style>
.stApp { background-color:#0B0E14; color:#E2E8F0; }
header[data-testid="stHeader"] { background-color:#0B0E14; }
.block-container { padding-top:3.5rem!important; padding-bottom:5rem!important; padding-left:0.9rem!important; padding-right:0.9rem!important; }
.kpi-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:8px; margin:10px 0; }
.metric-card { background-color:#131722; border:1px solid #1E293B; border-radius:10px; padding:12px; }
.metric-title { font-size:0.78rem; color:#94A3B8; margin-bottom:4px; }
.metric-value { font-size:1.2rem; font-weight:700; color:#F8FAFC; }
.badge-buy { background-color:rgba(16,185,129,0.15); color:#10B981; border:1px solid #10B981; padding:3px 8px; border-radius:6px; font-size:0.78rem; font-weight:600; display:inline-block; }
.badge-sell { background-color:rgba(239,68,68,0.15); color:#EF4444; border:1px solid #EF4444; padding:3px 8px; border-radius:6px; font-size:0.78rem; font-weight:600; display:inline-block; }
.badge-neutral { background-color:rgba(148,163,184,0.15); color:#94A3B8; border:1px solid #475569; padding:3px 8px; border-radius:6px; font-size:0.78rem; font-weight:600; display:inline-block; }
.ohlc-bar { display:grid; grid-template-columns:repeat(4,1fr); gap:6px; background-color:#131722; border:1px solid #1E293B; border-radius:10px; padding:10px 8px; margin-bottom:10px; text-align:center; }
.ohlc-lbl { font-size:0.72rem; color:#64748B; margin-bottom:2px; }
.ohlc-val { font-size:0.88rem; font-weight:700; color:#F8FAFC; }
div[data-testid="stVerticalBlockBorderWrapper"] { background-color:#131722; border:1px solid #1E293B; border-radius:10px; padding:12px; }
div[data-baseweb="input"]>div, div[data-baseweb="select"]>div { background-color:#F8FAFC!important; color:#0B0E14!important; border-radius:8px!important; }
div[data-baseweb="input"] input { color:#0B0E14!important; font-weight:600!important; }
label p, .stRadio label p { color:#E2E8F0!important; font-size:0.85rem!important; }
.stButton>button { background-color:#1E293B!important; color:#F8FAFC!important; border:1px solid #334155!important; border-radius:8px!important; }
</style>""", unsafe_allow_html=True)

with st.container(border=True):
    ticker = st.text_input("티커 (예: QQQ, NVDA, TSLA)", value="QQQ").upper().strip()
    timeframe = st.radio("봉 기준", ["30분봉", "1시간봉", "일봉"], horizontal=True)
    strategy_filter = st.selectbox("전략 필터", ["전체 전략 통합", "1. ORB (오프닝 레인지 돌파)", "2. VWAP + RSI 눌림목", "3. EMA 정배열 회귀", "4. 갭 채우기 (Gap Fill) 매도"], label_visibility="collapsed")
    if st.button("🔄 새로고침", use_container_width=True):
        st.cache_data.clear()

@st.cache_data(ttl=60)
def load_data(symbol, tf):
    p, iv = ("1mo", "30m") if tf == "30분봉" else (("3mo", "1h") if tf == "1시간봉" else ("2y", "1d"))
    df = yf.download(symbol, period=p, interval=iv, progress=False, auto_adjust=True)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.dropna().copy()
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

    if tf in ["30분봉", "1시간봉"]:
        fb = df.groupby('Date_NY').first()
        pc = df.groupby('Date_NY')['Close'].last().shift(1)
        orb_h, d_open, f_close, p_close = fb['High'].to_dict(), fb['Open'].to_dict(), fb['Close'].to_dict(), pc.to_dict()
        for i in range(1, len(df)):
            idx = df.index[i]
            r, pv = df.iloc[i], df.iloc[i-1]
            dt, tm = r['Date_NY'], r['Time_NY']
            oh, do, pcv = orb_h.get(dt, np.nan), d_open.get(dt, np.nan), p_close.get(dt, np.nan)
            rb, rs = [], []
            if tm > "09:30" and r['Close'] > oh and pv['Close'] <= oh and r['Volume'] > r['Vol_MA20'] * 1.5: rb.append("ORB 돌파")
            if r['Close'] > r['VWAP'] and pv['RSI'] <= 40 and r['Close'] > r['Open']: rb.append("VWAP+RSI 눌림")
            if (r['EMA20'] > r['EMA50'] > r['EMA200']) and ((r['Low'] <= r['EMA20']*1.002) or (r['Low'] <= r['EMA50']*1.002)) and (r['Close'] > r['Open']) and ((min(r['Open'], r['Close']) - r['Low']) > abs(r['Close'] - r['Open']) * 0.5): rb.append("EMA 정배열 회귀")
            if pd.notna(pcv) and pcv > 0 and (0 < (do - pcv)/pcv <= 0.01) and (f_close.get(dt, np.nan) < do) and (r['Close'] < r['VWAP']) and (pv['Close'] >= pv['VWAP']): rs.append("갭 채우기 이탈")
            if strategy_filter == "1. ORB (오프닝 레인지 돌파)": rb, rs = [x for x in rb if "ORB" in x], []
            elif strategy_filter == "2. VWAP + RSI 눌림목": rb, rs = [x for x in rb if "VWAP" in x], []
            elif strategy_filter == "3. EMA 정배열 회귀": rb, rs = [x for x in rb if "EMA" in x], []
            elif strategy_filter == "4. 갭 채우기 (Gap Fill) 매도": rb = []
            if rb: df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, ", ".join(rb)
            if rs: df.at[idx, 'Sell_Signal'], df.at[idx, 'Signal_Reason'] = True, ", ".join(rs)
    else:
        for i in range(1, len(df)):
            idx = df.index[i]
            r, pv = df.iloc[i], df.iloc[i-1]
            if r['Close'] > r['VWAP'] and pv['RSI'] <= 40 and r['Close'] > r['Open']:
                df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, "일봉 VWAP+RSI 눌림"
            elif (r['EMA20'] > r['EMA50'] > r['EMA200']) and (r['Low'] <= r['EMA20']) and (r['Close'] > r['Open']):
                df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, "일봉 EMA20 지지"
    return df

df = load_data(ticker, timeframe)
if df is None or len(df) < 5:
    st.error(f"❌ '{ticker}' 종목 데이터를 불러올 수 없습니다.")
    st.stop()

latest, prev_bar = df.iloc[-1], df.iloc[-2]
p_diff = latest['Close'] - prev_bar['Close']
p_pct = (p_diff / prev_bar['Close']) * 100
b_df, s_df = df[df['Buy_Signal']], df[df['Sell_Signal']]
lb_str = b_df.index[-1].strftime('%m-%d %H:%M') if not b_df.empty else "없음"
lb_rsn = b_df.iloc[-1]['Signal_Reason'] if not b_df.empty else "-"
ls_str = s_df.index[-1].strftime('%m-%d %H:%M') if not s_df.empty else "없음"

if latest['Buy_Signal']: st_html = f'<span class="badge-buy">🟢 {latest["Signal_Reason"]}</span>'
elif latest['Sell_Signal']: st_html = f'<span class="badge-sell">🔴 {latest["Signal_Reason"]}</span>'
elif latest['Close'] > latest['VWAP']: st_html = '<span class="badge-buy">📈 VWAP 상단</span>'
else: st_html = '<span class="badge-neutral">⏳ VWAP 하단</span>'

cp = "#EF4444" if p_diff >= 0 else "#3B82F6"
sgn = "+" if p_diff >= 0 else ""
rv = latest['RSI']
rc = "#EF4444" if rv >= 70 else ("#10B981" if rv <= 30 else "#F8FAFC")

st.markdown(f'<div class="kpi-grid"><div class="metric-card"><div class="metric-title">{ticker} ({timeframe})</div><div class="metric-value">${latest["Close"]:,.2f} <span style="font-size:0.8rem;color:{cp};">({sgn}{p_pct:.2f}%)</span></div></div><div class="metric-card"><div class="metric-title">현재 추세 판정</div><div style="margin-top:2px;">{st_html}</div></div><div class="metric-card"><div class="metric-title">최근 매수 ({lb_str})</div><div class="metric-value" style="font-size:0.95rem;color:#10B981;">{lb_rsn}</div></div><div class="metric-card"><div class="metric-title">RSI(14) / 매도({ls_str})</div><div class="metric-value" style="color:{rc};">{rv:.1f}</div></div></div>', unsafe_allow_html=True)

st.markdown(f'<div class="ohlc-bar"><div><div class="ohlc-lbl">시가</div><div class="ohlc-val">${latest["Open"]:,.2f}</div></div><div><div class="ohlc-lbl">고가</div><div class="ohlc-val" style="color:#EF4444;">${latest["High"]:,.2f}</div></div><div><div class="ohlc-lbl">저가</div><div class="ohlc-val" style="color:#3B82F6;">${latest["Low"]:,.2f}</div></div><div><div class="ohlc-lbl">VWAP</div><div class="ohlc-val" style="color:#38BDF8;">${latest["VWAP"]:,.2f}</div></div></div>', unsafe_allow_html=True)

pdf = df.tail(50).copy()
pdf.index = pdf.index.tz_convert('America/New_York')

mc = mpf.make_marketcolors(up='#EF4444', down='#3B82F6', edge='inherit', wick='inherit', volume={'up':'#EF4444', 'down':'#3B82F6'})
style = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, facecolor='#0B0E14', figcolor='#131722', gridcolor='#1E293B', gridstyle='-', y_on_right=True)

apds = [
    mpf.make_addplot(pdf['EMA20'], color='#38BDF8', width=1.1),
    mpf.make_addplot(pdf['EMA50'], color='#F59E0B', width=1.1),
    mpf.make_addplot(pdf['VWAP'], color='#10B981', width=1.2, linestyle='--'),
    mpf.make_addplot(pdf['RSI'], panel=2, color='#818CF8', width=1.2, ylabel='RSI')
]

buy_y = np.where(pdf['Buy_Signal'], pdf['Low'] * 0.996, np.nan)
sell_y = np.where(pdf['Sell_Signal'], pdf['High'] * 1.004, np.nan)
if not np.all(np.isnan(buy_y)):
    apds.append(mpf.make_addplot(buy_y, type='scatter', markersize=70, marker='^', color='#10B981'))
if not np.all(np.isnan(sell_y)):
    apds.append(mpf.make_addplot(sell_y, type='scatter', markersize=70, marker='v', color='#EF4444'))

buf = io.BytesIO()
