import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests, json

st.set_page_config(page_title="US Stock Quant Pro", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
.stApp{background:#0B0E14;color:#E2E8F0}header[data-testid="stHeader"]{background:#0B0E14}
.block-container{padding:2.6rem .8rem 2rem .8rem!important}
.kpi-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:6px;margin:6px 0 8px 0}
.metric-card,.list-card{background:#131722;border:1px solid #1E293B;border-radius:8px;padding:9px 10px;margin-bottom:6px}
.metric-title,.ohlc-lbl{font-size:.72rem;color:#94A3B8}
.metric-value{font-size:1.05rem;font-weight:700;color:#F8FAFC}
.ohlc-val{font-size:.82rem;font-weight:700;color:#F8FAFC}
.badge-buy{background:rgba(16,185,129,.15);color:#10B981;border:1px solid #10B981;padding:2px 6px;border-radius:5px;font-size:.72rem;font-weight:600}
.badge-sell{background:rgba(239,68,68,.15);color:#EF4444;border:1px solid #EF4444;padding:2px 6px;border-radius:5px;font-size:.72rem;font-weight:600}
.badge-neutral{background:rgba(148,163,184,.15);color:#94A3B8;border:1px solid #475569;padding:2px 6px;border-radius:5px;font-size:.72rem;font-weight:600}
.tpsl-bar{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;background:#131722;border:1px solid #334155;border-radius:8px;padding:8px;margin-bottom:8px;text-align:center}
.opt-box{background:#131722;border:1px solid #10B981;border-radius:8px;padding:10px;margin:6px 0 10px 0}
.list-top,.list-sub{display:flex;justify-content:space-between;align-items:center}
.list-sub{font-size:.75rem;color:#94A3B8;margin-top:4px}
</style>""", unsafe_allow_html=True)

with st.expander("⚙️ 종목 / 전략 / ATR / 카톡 설정", expanded=False):
    ticker = st.text_input("메인 분석 티커", value="QQQ").upper().strip()
    timeframe = st.radio("봉 기준", ["30분봉", "1시간봉", "일봉"], horizontal=True)
    strategy_filter = st.selectbox("전략 필터", ["전체 전략 통합 (최고승률 엄선)", "VWAP+RSI 눌림", "EMA 정배열 회귀", "MACD+VWAP 반전", "볼린저 하단 반등", "ORB 돌파", "갭 채우기 이탈"])
    c1, c2, c3 = st.columns(3)
    hold_bars = c1.slider("최대 보유봉", 2, 24, 8)
    sl_mult = c2.number_input("손절 ATR", 0.5, 5.0, 1.5, 0.1)
    tp_mult = c3.number_input("익절 ATR", 1.0, 10.0, 2.5, 0.5)
    use_adx_filter = st.checkbox("ADX>20 & EMA200 상승추세 필터", value=True)
    watchlist_str = st.text_input("스캐너 관심종목", value="QQQ, SPY, NVDA, TSLA, AAPL, MSFT, AMD, META")
    kakao_token = st.text_input("카카오톡 Access Token", type="password")
    if st.button("🔄 데이터 새로고침", use_container_width=True):
        st.cache_data.clear()

def send_kakao(token, msg):
    if not token:
        return False, "카카오톡 토큰을 입력해 주세요."
    try:
        r = requests.post("https://kakao.com", headers={"Authorization": f"Bearer {token}"}, data={"template_object": json.dumps({"object_type": "text", "text": msg[:200], "link": {"web_url": "https://yahoo.com", "mobile_web_url": "https://yahoo.com"}})}, timeout=5)
        return (True, "카톡 전송 완료!") if r.status_code == 200 else (False, f"전송 실패: {r.text}")
    except Exception as e:
        return False, str(e)

@st.cache_data(ttl=60)
def load_single(sym, tf, s_filter, adx_on):
    p, iv = ("1mo", "30m") if tf == "30분봉" else (("3mo", "1h") if tf == "1시간봉" else ("2y", "1d"))
    raw = yf.download(sym, period=p, interval=iv, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna().copy()
    if len(df) < 25:
        return None
    df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index
    ny = df.tz_convert('America/New_York')
    df['Date_NY'], df['Time_NY'] = ny.index.date, ny.index.strftime('%H:%M')
    df['EMA20'], df['EMA50'], df['EMA200'] = df['Close'].ewm(span=20).mean(), df['Close'].ewm(span=50).mean(), df['Close'].ewm(span=200).mean()
    d = df['Close'].diff()
    df['RSI'] = 100 - (100 / (1 + d.where(d > 0, 0).rolling(14).mean() / (-d.where(d < 0, 0)).rolling(14).mean().replace(0, np.nan)))
    hd, ld = df['High'].diff(), -df['Low'].diff()
    pdm, mdm = hd.where((hd > ld) & (hd > 0), 0.0), ld.where((ld > hd) & (ld > 0), 0.0)
    df['TR'] = pd.concat([df['High'] - df['Low'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean().bfill()
    atrs = df['TR'].ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan)
    pdi, mdi = 100 * (pdm.ewm(alpha=1/14, adjust=False).mean() / atrs), 100 * (mdm.ewm(alpha=1/14, adjust=False).mean() / atrs)
    df['ADX'] = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).fillna(0).ewm(alpha=1/14, adjust=False).mean()
    macd = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['MACD_Hist'] = macd - macd.ewm(span=9).mean()
    df['BB_Low'] = df['Close'].rolling(20).mean() - 2 * df['Close'].rolling(20).std()
    df['TP_Vol'] = ((df['High'] + df['Low'] + df['Close']) / 3) * df['Volume']
    df['VWAP'] = (df.groupby('Date_NY')['TP_Vol'].cumsum() / df.groupby('Date_NY')['Volume'].cumsum()) if tf != "일봉" else (df['TP_Vol'].rolling(20).sum() / df['Volume'].rolling(20).sum())
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Buy_Signal'], df['Sell_Signal'], df['Signal_Reason'] = False, False, ""
    if tf != "일봉":
        fb, pc = df.groupby('Date_NY').first(), df.groupby('Date_NY')['Close'].last().shift(1)
        orb_h, d_open, f_close, p_close = fb['High'].to_dict(), fb['Open'].to_dict(), fb['Close'].to_dict(), pc.to_dict()
    for i in range(2, len(df)):
        idx, r, pv, pv2 = df.index[i], df.iloc[i], df.iloc[i-1], df.iloc[i-2]
        rb, rs = [], []
        if r['Close'] > r['VWAP'] and (pv['RSI'] <= 42 or pv2['RSI'] <= 40) and r['Close'] > r['Open']: rb.append("VWAP+RSI 눌림")
        if r['EMA20'] > r['EMA50'] and (r['Low'] <= r['EMA20']*1.003 or r['Low'] <= r['EMA50']*1.003) and r['Close'] > r['Open']: rb.append("EMA 정배열 회귀")
        if r['Close'] > r['VWAP'] and r['MACD_Hist'] > 0 and pv['MACD_Hist'] <= 0 and r['Close'] > r['Open']: rb.append("MACD+VWAP 반전")
        if (pv['Low'] <= pv['BB_Low'] or r['Low'] <= r['BB_Low']) and r['Close'] > r['BB_Low'] and r['Close'] > r['Open'] and r['RSI'] < 52: rb.append("볼린저 하단 반등")
        if tf != "일봉":
            dt, tm = r['Date_NY'], r['Time_NY']
            oh, do, pcv = orb_h.get(dt, np.nan), d_open.get(dt, np.nan), p_close.get(dt, np.nan)
            if tm > "09:30" and r['Close'] > oh and pv['Close'] <= oh and r['Volume'] > r['Vol_MA20']*1.4: rb.append("ORB 돌파")
            if pd.notna(pcv) and pcv > 0 and (0 < (do - pcv)/pcv <= 0.012) and (f_close.get(dt, np.nan) < do) and r['Close'] < r['VWAP'] and pv['Close'] >= pv['VWAP']: rs.append("갭 채우기 이탈")
        elif r['Open'] > pv['Close'] and r['Close'] < pv['Close'] and r['Close'] < r['VWAP']: rs.append("갭 채우기 이탈")
        if not ((r['Close'] > r['EMA200'] and r['ADX'] >= 20 and r['RSI'] < 68) if adx_on else (r['Close'] > r['EMA50'] and r['RSI'] < 70)): rb = []
        if not (r['Close'] < r['EMA20'] and r['RSI'] > 30): rs = []
        if s_filter != "전체 전략 통합 (최고승률 엄선)":
            rb, rs = [x for x in rb if s_filter in x], [x for x in rs if s_filter in x]
            if rb: df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, "매수: " + ", ".join(rb)
            if rs: df.at[idx, 'Sell_Signal'], df.at[idx, 'Signal_Reason'] = True, "매도: " + ", ".join(rs)
        else:
            if rb and (len(rb) >= 2 or r['Volume'] > r['Vol_MA20']*1.5): df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, "★최고승률 매수: " + ", ".join(rb)
            if rs: df.at[idx, 'Sell_Signal'], df.at[idx, 'Signal_Reason'] = True, "매도: " + ", ".join(rs)
    return df
def run_bt(df_in, n_bars, tp_m, sl_m):
    trades, c_arr, h_arr, l_arr, a_arr, idx_arr = [], df_in['Close'].to_numpy(float), df_in['High'].to_numpy(float), df_in['Low'].to_numpy(float), df_in['ATR'].to_numpy(float), df_in.index
    for pos in np.flatnonzero(df_in['Buy_Signal'].to_numpy()):
        p = int(pos)
        if p + 1 >= len(df_in): continue
        ep, av = c_arr[p], a_arr[p]
        tp_t, sl_t, end_p = ep + av * tp_m, ep - av * sl_m, min(p + n_bars, len(df_in) - 1)
        xp, xdt = c_arr[end_p], idx_arr[end_p]
        for s in range(p + 1, end_p + 1):
            if l_arr[s] <= sl_t: xp, xdt = sl_t, idx_arr[s]; break
            if h_arr[s] >= tp_t: xp, xdt = tp_t, idx_arr[s]; break
        trades.append({"date": xdt, "ret": (xp - ep) / ep})
    if not trades: return 0, 0.0, 0.0, 0.0, 0.0, pd.DataFrame()
    tdf = pd.DataFrame(trades)
    tdf['eq'] = (1 + tdf['ret']).cumprod()
    tdf['cum'] = (tdf['eq'] - 1) * 100
    tdf['dd'] = ((tdf['eq'] - tdf['eq'].cummax()) / tdf['eq'].cummax()) * 100
    return len(tdf), float((tdf['ret'] > 0).mean()*100), float(tdf['ret'].mean()*100), float(tdf['cum'].iloc[-1]), float(tdf['dd'].min()), tdf

tab1, tab2 = st.tabs(["📊 종목 분석 & 자동 최적화", "🔍 멀티 스캐너 & 카톡"])
with tab1:
    df = load_single(ticker, timeframe, strategy_filter, use_adx_filter)
    if df is None: st.error(f"'{ticker}' 데이터를 불러올 수 없습니다.")
    else:
        last, prev = df.iloc[-1], df.iloc[-2]
        chg_pct = ((float(last['Close']) - float(prev['Close'])) / float(prev['Close'])) * 100
        tp_p, sl_p = float(last['Close']) + float(last['ATR'])*tp_mult, float(last['Close']) - float(last['ATR'])*sl_mult
        st_html = f'<span class="badge-buy">{last["Signal_Reason"]}</span>' if last['Buy_Signal'] else (f'<span class="badge-sell">{last["Signal_Reason"]}</span>' if last['Sell_Signal'] else '<span class="badge-neutral">관망</span>')
        st.markdown(f"### 📈 {ticker} ({timeframe}) {st_html}", unsafe_allow_html=True)
        cnt, win, avg, tot, mdd, tdf = run_bt(df, hold_bars, tp_mult, sl_mult)
        st.markdown(f"""<div class="kpi-grid">
        <div class="metric-card"><div class="metric-title">현재가</div><div class="metric-value">${last['Close']:,.2f} ({chg_pct:+.2f}%)</div></div>
        <div class="metric-card"><div class="metric-title">RSI / ADX</div><div class="metric-value">{last['RSI']:.1f} / {last['ADX']:.1f}</div></div>
        <div class="metric-card"><div class="metric-title">백테스트 승률 ({cnt}회)</div><div class="metric-value">{win:.1f}% (평균 {avg:+.2f}%)</div></div>
        <div class="metric-card"><div class="metric-title">누적 수익 / MDD</div><div class="metric-value">{tot:+.2f}% ({mdd:.2f}%)</div></div>
        </div><div class="tpsl-bar">
        <div><div class="ohlc-lbl">🎯 익절(TP)</div><div class="ohlc-val" style="color:#10B981;">${tp_p:,.2f}</div></div>
        <div><div class="ohlc-lbl">🛡️ 손절(SL)</div><div class="ohlc-val" style="color:#EF4444;">${sl_p:,.2f}</div></div>
        <div><div class="ohlc-lbl">⚖️ 손익비</div><div class="ohlc-val" style="color:#F59E0B;">1 : {tp_mult/sl_mult:.2f}</div></div>
        </div>""", unsafe_allow_html=True)
        if st.button(f"✨ {ticker} 최적 손절/익절 자동 탐색", use_container_width=True):
            opts = []
            for sm in (1.0, 1.5, 2.0):
                for tm in (2.0, 2.5, 3.0, 4.0):
                    if tm > sm:
                        c, w, a, t, m, _ = run_bt(df, hold_bars, tm, sm)
                        if c > 0: opts.append((t*0.6 + w*0.3 - abs(m)*0.4, sm, tm, w, t, m))
            if opts:
                opts.sort(reverse=True)
                _, b_sl, b_tp, b_w, b_t, b_m = opts[0]
                st.markdown(f'<div class="opt-box"><strong style="color:#10B981;">🏆 최적 비율: 손절 {b_sl} ATR / 익절 {b_tp} ATR</strong><br>승률 {b_w:.1f}% | 누적수익 {b_t:+.2f}% | MDD {b_m:.2f}%</div>', unsafe_allow_html=True)
            else: st.warning("탐색할 매수 이력이 부족합니다.")
        pdf = df.tail(75)
        xl = pdf.index.tz_convert('America/New_York').strftime('%m-%d %H:%M' if timeframe != "일봉" else '%Y-%m-%d').tolist()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.78, 0.22])
        fig.add_trace(go.Candlestick(x=xl, open=pdf['Open'], high=pdf['High'], low=pdf['Low'], close=pdf['Close'], name="가격"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xl, y=pdf['EMA20'], line=dict(color='#F59E0B', width=1), name="EMA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xl, y=pdf['VWAP'], line=dict(color='#A855F7', width=1, dash='dot'), name="VWAP"), row=1, col=1)
        fig.add_trace(go.Bar(x=xl, y=pdf['Volume'], name="거래량", showlegend=False), row=2, col=1)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0B0E14", plot_bgcolor="#131722", height=400, margin=dict(l=5,r=5,t=10,b=10), xaxis_rangeslider_visible=False)
        fig.update_xaxes(type='category', nticks=6)
        st.plotly_chart(fig, use_container_width=True)
        if st.button("💛 카톡으로 현재 분석 전송", use_container_width=True):
            ok, m = send_kakao(kakao_token, f"[{ticker} {timeframe}] ${last['Close']:,.2f} ({chg_pct:+.2f}%)\n신호: {last['Signal_Reason'] or '관망'}\nTP: ${tp_p:,.2f} / SL: ${sl_p:,.2f}")
            st.toast(m)
        if not tdf.empty:
            st.line_chart(tdf.set_index('date')['cum'], height=180)

with tab2:
    syms = [s.strip().upper() for s in watchlist_str.split(",") if s.strip()]
    if st.button("🚀 전체 관심종목 스캔", use_container_width=True):
        scan_list = []
        for s in syms:
            sdf = load_single(s, timeframe, strategy_filter, use_adx_filter)
            if sdf is not None and len(sdf) >= 2:
                sl, sp = sdf.iloc[-1], sdf.iloc[-2]
                cp = ((float(sl['Close']) - float(sp['Close'])) / float(sp['Close'])) * 100
                sig = sl['Signal_Reason'] if (sl['Buy_Signal'] or sl['Sell_Signal']) else "관망"
                scan_list.append({"sym": s, "p": float(sl['Close']), "cp": cp, "sig": sig, "rsi": float(sl['RSI'])})
        for it in scan_list:
            bcls = "badge-buy" if "매수" in it['sig'] else ("badge-sell" if "매도" in it['sig'] else "badge-neutral")
            st.markdown(f'<div class="list-card"><div class="list-top"><strong>{it["sym"]} ${it["p"]:,.2f} ({it["cp"]:+.2f}%)</strong><span class="{bcls}">{it["sig"]}</span></div><div class="list-sub"><span>RSI: {it["rsi"]:.1f}</span></div></div>', unsafe_allow_html=True)
