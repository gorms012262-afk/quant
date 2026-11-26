import json, streamlit as st, streamlit.components.v1 as components, yfinance as yf, pandas as pd, numpy as np

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

with st.expander("⚙️ 종목 / 봉 기준 설정 (눌러서 열기)", expanded=False):
    ticker = st.text_input("티커 (예: QQQ, NVDA, TSLA)", value="QQQ").upper().strip()
    timeframe = st.radio("봉 기준", ["30분봉", "1시간봉", "일봉"], horizontal=True)
    strategy_filter = st.selectbox("전략 필터", ["전체 전략 통합 (고승률 엄선)"], label_visibility="collapsed")
    if st.button("🔄 데이터 새로고침", use_container_width=True):
        st.cache_data.clear()

@st.cache_data(ttl=60)
def load_single(symbol, tf, strat_filter):
    p, iv = ("1mo", "30m") if tf == "30분봉" else (("3mo", "1h") if tf == "1시간봉" else ("2y", "1d"))
    raw = yf.download(symbol, period=p, interval=iv, progress=False, auto_adjust=True)
    if raw is None or raw.empty: return None
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna().copy()
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
            if not ((r['Close'] > r['EMA50'] or r['EMA20'] > r['EMA50']) and (r['RSI'] < 68)): rb = []
            if not (r['Close'] < r['EMA20'] and r['RSI'] > 32): rs = []
            if rb and (len(rb) >= 2 or r['Volume'] > r['Vol_MA20'] * 1.8):
                df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, "★강력매수: " + ", ".join(rb)
            if rs: df.at[idx, 'Sell_Signal'], df.at[idx, 'Signal_Reason'] = True, ", ".join(rs)
    else:
        for i in range(1, len(df)):
            idx = df.index[i]
            r, pv = df.iloc[i], df.iloc[i-1]
            rb, rs = [], []
            if r['Close'] > r['VWAP'] and pv['RSI'] <= 40 and r['Close'] > r['Open']: rb.append("일봉 VWAP+RSI 눌림")
            if (r['EMA20'] > r['EMA50'] > r['EMA200']) and (r['Low'] <= r['EMA20']) and (r['Close'] > r['Open']): rb.append("일봉 EMA20 지지")
            if (r['Open'] > pv['Close']) and (r['Close'] < pv['Close']) and (r['Close'] < r['VWAP']): rs.append("일봉 갭하락/VWAP 이탈")
            if not ((r['Close'] > r['EMA50']) and (r['RSI'] < 68)): rb = []
            if not (r['Close'] < r['EMA20']): rs = []
            if rb and (len(rb) >= 2 or r['Volume'] > r['Vol_MA20'] * 1.5):
                df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, "★강력매수: " + ", ".join(rb)
            if rs: df.at[idx, 'Sell_Signal'], df.at[idx, 'Signal_Reason'] = True, ", ".join(rs)
    return df

def render_tv_chart(df_in, tf):
    pdf = df_in.tail(120).copy()
    ny_idx = pdf.index.tz_convert('America/New_York')
    candles, ema20, ema50, vwap, vols, markers = [], [], [], [], [], []
    for ts, r in zip(ny_idx, pdf.itertuples()):
        t_val = ts.strftime('%Y-%m-%d') if tf == "일봉" else int(ts.tz_localize(None).timestamp())
        candles.append({"time": t_val, "open": round(float(r.Open), 2), "high": round(float(r.High), 2), "low": round(float(r.Low), 2), "close": round(float(r.Close), 2)})
        if pd.notna(r.EMA20): ema20.append({"time": t_val, "value": round(float(r.EMA20), 2)})
        if pd.notna(r.EMA50): ema50.append({"time": t_val, "value": round(float(r.EMA50), 2)})
        if pd.notna(r.VWAP): vwap.append({"time": t_val, "value": round(float(r.VWAP), 2)})
        v_col = "rgba(239, 68, 68, 0.45)" if r.Close >= r.Open else "rgba(59, 130, 246, 0.45)"
        vols.append({"time": t_val, "value": float(r.Volume), "color": v_col})
        if r.Buy_Signal:
            markers.append({"time": t_val, "position": "belowBar", "color": "#10B981", "shape": "arrowUp", "text": "BUY"})
        elif r.Sell_Signal:
            markers.append({"time": t_val, "position": "aboveBar", "color": "#EF4444", "shape": "arrowDown", "text": "SELL"})

    html_code = f"""
    <div style="background:#131722;border:1px solid #1E293B;border-radius:8px;padding:6px 4px 2px 4px;">
      <div style="display:flex;gap:10px;font-size:11px;color:#94A3B8;padding:0 8px 4px 8px;font-family:sans-serif;">
        <span><b style="color:#38BDF8;">━</b> EMA20</span>
        <span><b style="color:#F59E0B;">━</b> EMA50</span>
        <span><b style="color:#10B981;">┄</b> VWAP</span>
        <span style="margin-left:auto;color:#64748B;">터치·줌 가능</span>
      </div>
      <div id="tvchart" style="width:100%;height:390px;"></div>
    </div>
    <script src="https://unpkg.com"></script>
    <script>
      const chart = LightweightCharts.createChart(document.getElementById('tvchart'), {{
        layout: {{ background: {{ type: 'solid', color: '#131722' }}, textColor: '#94A3B8', fontSize: 11 }},
        grid: {{ vertLines: {{ color: '#1E293B' }}, horzLines: {{ color: '#1E293B' }} }},
        crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
        rightPriceScale: {{ borderColor: '#1E293B' }},
        timeScale: {{ borderColor: '#1E293B', timeVisible: true, secondsVisible: false }}
      }});
      const cSeries = chart.addCandlestickSeries({{
        upColor: '#EF4444', downColor: '#3B82F6', borderVisible: false, wickUpColor: '#EF4444', wickDownColor: '#3B82F6'
      }});
      cSeries.setData({json.dumps(candles)});
      cSeries.setMarkers({json.dumps(markers)});
      const vSeries = chart.addHistogramSeries({{ priceFormat: {{ type: 'volume' }}, priceScaleId: '' }});
      vSeries.priceScale().applyOptions({{ scaleMargins: {{ top: 0.82, bottom: 0 }} }});
      vSeries.setData({json.dumps(vols)});
      const e20 = chart.addLineSeries({{ color: '#38BDF8', lineWidth: 1.5, priceLineVisible: false }});
      e20.setData({json.dumps(ema20)});
      const e50 = chart.addLineSeries({{ color: '#F59E0B', lineWidth: 1.5, priceLineVisible: false }});
      e50.setData({json.dumps(ema50)});
      const vw = chart.addLineSeries({{ color: '#10B981', lineWidth: 1.5, lineStyle: 2, priceLineVisible: false }});
      vw.setData({json.dumps(vwap)});
      chart.timeScale().fitContent();
      new ResizeObserver(entries => {{
        if (entries.length === 0 || entries.target !== document.getElementById('tvchart')) return;
        const newRect = entries.contentRect;
        chart.applyOptions({{ width: newRect.width, height: newRect.height }});
      }}).observe(document.getElementById('tvchart'));
    </script>
    """
    components.html(html_code, height=425)

tab1, tab2 = st.tabs(["📊 종목 차트 분석", "📡 관심 종목 스캐너"])

with tab1:
    df = load_single(ticker, timeframe, strategy_filter)
    if df is None:
        st.error(f"❌ '{ticker}' 종목 데이터를 불러올 수 없습니다.")
        st.stop()

    wins, losses = 0, 0
    for bi in np.flatnonzero(df['Buy_Signal']):
        if bi + 1 >= len(df): continue
        entry = df['Close'].iloc[bi]
        future = df.iloc[bi+1 : min(bi+8, len(df))]
        hit_tp, hit_sl = (future['High'] >= entry * 1.015).any(), (future['Low'] <= entry * 0.990).any()
        if hit_tp and not hit_sl: wins += 1
        elif hit_sl and not hit_tp: losses += 1
        elif future['Close'].iloc[-1] > entry: wins += 1
        else: losses += 1
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    latest, prev_bar = df.iloc[-1], df.iloc[-2]
    p_diff = latest['Close'] - prev_bar['Close']
    p_pct = (p_diff / prev_bar['Close']) * 100
    b_df = df[df['Buy_Signal']]
    lb_str = b_df.index[-1].tz_convert('America/New_York').strftime('%m-%d %H:%M') if not b_df.empty else "없음"
    lb_rsn = b_df.iloc[-1]['Signal_Reason'] if not b_df.empty else "-"

    if latest['Buy_Signal']: st_html = f'<span class="badge-buy">🟢 {latest["Signal_Reason"]}</span>'
    elif latest['Sell_Signal']: st_html = f'<span class="badge-sell">🔴 {latest["Signal_Reason"]}</span>'
    elif latest['Close'] > latest['VWAP']: st_html = '<span class="badge-buy">📈 VWAP 상단</span>'
    else: st_html = '<span class="badge-neutral">⏳ VWAP 하단</span>'

    cp, sgn = ("#EF4444" if p_diff >= 0 else "#3B82F6"), ("+" if p_diff >= 0 else "")
    rv = latest['RSI']
    rc = "#EF4444" if rv >= 70 else ("#10B981" if rv <= 30 else "#F8FAFC")

    st.markdown(f'<div class="kpi-grid"><div class="metric-card"><div class="metric-title">{ticker} ({timeframe})</div><div class="metric-value">${latest["Close"]:,.2f} <span style="font-size:0.75rem;color:{cp};">({sgn}{p_pct:.2f}%)</span></div></div><div class="metric-card"><div class="metric-title">현재 추세 판정</div><div style="margin-top:2px;">{st_html}</div></div><div class="metric-card"><div class="metric-title">최근 강력매수 ({lb_str})</div><div class="metric-value" style="font-size:0.84rem;color:#10B981;">{lb_rsn}</div></div><div class="metric-card"><div class="metric-title">강력매수 승률 ({total_trades}회) / RSI</div><div class="metric-value" style="color:{rc};">{win_rate:.1f}% <span style="font-size:0.78rem;color:#94A3B8;">(RSI {rv:.1f})</span></div></div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="ohlc-bar"><div><div class="ohlc-lbl">시가</div><div class="ohlc-val">${latest["Open"]:,.2f}</div></div><div><div class="ohlc-lbl">고가</div><div class="ohlc-val" style="color:#EF4444;">${latest["High"]:,.2f}</div></div><div><div class="ohlc-lbl">저가</div><div class="ohlc-val" style="color:#3B82F6;">${latest["Low"]:,.2f}</div></div><div><div class="ohlc-lbl">VWAP</div><div class="ohlc-val" style="color:#38BDF8;">${latest["VWAP"]:,.2f}</div></div></div>', unsafe_allow_html=True)

    render_tv_chart(df, timeframe)

    sig_df = df[df['Buy_Signal'] | df['Sell_Signal']].tail(8).iloc[::-1].copy()
    st.markdown("<div style='font-size:0.85rem;font-weight:700;color:#F8FAFC;margin:10px 0 6px 2px;'>📋 최근 강력매수 및 손절/익절 가이드</div>", unsafe_allow_html=True)
    if not sig_df.empty:
        sig_ny = sig_df.index.tz_convert('America/New_York')
