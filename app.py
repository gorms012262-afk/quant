import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests, json

st.set_page_config(page_title="Global Quant Pro - Auto WinRate", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
.stApp{background:#0B0E14;color:#E2E8F0}header[data-testid="stHeader"]{background:#0B0E14}
.block-container{padding:2.6rem .8rem 2rem .8rem!important}
.kpi-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:6px;margin:6px 0 8px 0}
.metric-card,.list-card{background:#131722;border:1px solid #1E293B;border-radius:10px;padding:10px 12px;margin-bottom:8px}
.metric-title,.ohlc-lbl{font-size:.72rem;color:#94A3B8}
.metric-value{font-size:1.05rem;font-weight:700;color:#F8FAFC}
.ohlc-val{font-size:.82rem;font-weight:700;color:#F8FAFC}
.badge-buy{background:rgba(16,185,129,.18);color:#10B981;border:1px solid #10B981;padding:2px 7px;border-radius:6px;font-size:.75rem;font-weight:700;white-space:nowrap}
.badge-sell{background:rgba(239,68,68,.18);color:#EF4444;border:1px solid #EF4444;padding:2px 7px;border-radius:6px;font-size:.75rem;font-weight:700;white-space:nowrap}
.badge-neutral{background:rgba(148,163,184,.15);color:#94A3B8;border:1px solid #475569;padding:2px 7px;border-radius:6px;font-size:.75rem;font-weight:600;white-space:nowrap}
.tpsl-bar{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;background:#131722;border:1px solid #334155;border-radius:8px;padding:8px;margin-bottom:8px;text-align:center}
.opt-box{background:#131722;border:1px solid #10B981;border-radius:8px;padding:10px;margin:6px 0 10px 0;font-size:.82rem}
.list-top{display:flex;justify-content:space-between;align-items:flex-start;gap:8px}
.list-left{display:flex;align-items:flex-start;gap:7px;flex:1;min-width:0}
.strat-txt{font-size:.85rem;font-weight:700;color:#F8FAFC;line-height:1.35;word-break:keep-all}
.price-txt{font-size:.9rem;font-weight:700;color:#F8FAFC;white-space:nowrap}
.list-sub{display:flex;justify-content:space-between;align-items:center;font-size:.74rem;color:#94A3B8;margin-top:6px}
div[data-testid="stExpander"]{background:#131722!important;border:1px solid #1E293B!important;border-radius:8px!important}
div[data-testid="stExpander"] summary p,label p,.stRadio label p,.stCheckbox label p{color:#F8FAFC!important;font-size:.82rem!important}
</style>""", unsafe_allow_html=True)

def send_kakao(token, msg):
    if not token: return False, "카카오톡 토큰을 입력해 주세요."
    try:
        r = requests.post("https://kakao.com", headers={"Authorization": f"Bearer {token}"}, data={"template_object": json.dumps({"object_type": "text", "text": msg[:200], "link": {"web_url": "https://yahoo.com", "mobile_web_url": "https://yahoo.com"}})}, timeout=5)
        return (True, "카톡 전송 완료!") if r.status_code == 200 else (False, f"전송 실패: {r.text}")
    except Exception as e:
        return False, str(e)

@st.cache_data(ttl=60)
def load_quant_data(sym, tf, s_filter, trend_on):
    p, iv = ("1mo", "30m") if tf == "30분봉" else ("3mo", "1h")
    raw = yf.download(sym, period=p, interval=iv, progress=False, auto_adjust=True)
    if raw is None or raw.empty: return None
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna().copy()
    if len(df) < 30: return None
    df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index
    ny = df.tz_convert('America/New_York')
    df['Date_NY'], df['Time_NY'] = ny.index.date, ny.index.strftime('%H:%M')

    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    d = df['Close'].diff()
    df['RSI'] = 100 - (100 / (1 + d.where(d > 0, 0).rolling(14).mean() / (-d.where(d < 0, 0)).rolling(14).mean().replace(0, np.nan)))

    hd, ld = df['High'].diff(), -df['Low'].diff()
    pdm, mdm = hd.where((hd > ld) & (hd > 0), 0.0), ld.where((ld > hd) & (ld > 0), 0.0)
    df['TR'] = pd.concat([df['High'] - df['Low'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean().bfill()
    atrs = df['TR'].ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan)
    pdi, mdi = 100 * (pdm.ewm(alpha=1/14, adjust=False).mean() / atrs), 100 * (mdm.ewm(alpha=1/14, adjust=False).mean() / atrs)
    df['ADX'] = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).fillna(0).ewm(alpha=1/14, adjust=False).mean()

    macd = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = macd - macd.ewm(span=9, adjust=False).mean()
    bb_m, bb_s = df['Close'].rolling(20).mean(), df['Close'].rolling(20).std()
    df['BB_Low'], df['BB_Up'] = bb_m - 2.0 * bb_s, bb_m + 2.0 * bb_s
    df['ST_Up'] = df['Close'] > (df['EMA20'] - 1.5 * df['ATR'])
    df['TP_Vol'] = ((df['High'] + df['Low'] + df['Close']) / 3) * df['Volume']
    df['VWAP'] = df.groupby('Date_NY')['TP_Vol'].cumsum() / df.groupby('Date_NY')['Volume'].cumsum().replace(0, np.nan)
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Buy_Signal'], df['Sell_Signal'], df['Signal_Reason'] = False, False, ""

    fb = df.groupby('Date_NY').first()
    orb_h, orb_l = fb['High'].to_dict(), fb['Low'].to_dict()

    for i in range(2, len(df)):
        idx, r, pv, pv2 = df.index[i], df.iloc[i], df.iloc[i-1], df.iloc[i-2]
        rb, rs = [], []
        if r['Close'] > r['VWAP'] and (pv['RSI'] <= 43 or pv2['RSI'] <= 41) and r['Close'] > r['Open']: rb.append("VWAP+RSI 눌림")
        if r['EMA20'] > r['EMA50'] and r['ST_Up'] and (r['Low'] <= r['EMA20']*1.003) and r['Close'] > r['Open']: rb.append("EMA 정배열 회귀")
        if r['Close'] > r['VWAP'] and r['MACD_Hist'] > 0 and pv['MACD_Hist'] <= 0 and r['Volume'] > r['Vol_MA20']*1.2: rb.append("MACD+VWAP 반전")
        if (pv['Low'] <= pv['BB_Low'] or r['Low'] <= r['BB_Low']) and r['Close'] > r['BB_Low'] and r['Close'] > r['Open'] and r['RSI'] < 48: rb.append("볼린저 하단 반등")
        oh, ol = orb_h.get(r['Date_NY'], np.nan), orb_l.get(r['Date_NY'], np.nan)
        if r['Time_NY'] > "09:30" and r['Close'] > oh and pv['Close'] <= oh and r['Volume'] > r['Vol_MA20']*1.4: rb.append("ORB 돌파")

        if r['Close'] < r['VWAP'] and (pv['RSI'] >= 62 or pv2['RSI'] >= 65) and r['Close'] < r['Open']: rs.append("VWAP 저항+RSI 과열")
        if r['EMA20'] < r['EMA50'] and (r['High'] >= r['EMA20']*0.997) and r['Close'] < r['Open']: rs.append("EMA 역배열 저항")
        if r['Close'] < r['VWAP'] and r['MACD_Hist'] < 0 and pv['MACD_Hist'] >= 0: rs.append("MACD+VWAP 하락반전")
        if (pv['High'] >= pv['BB_Up'] or r['High'] >= r['BB_Up']) and r['Close'] < r['BB_Up'] and r['Close'] < r['Open'] and r['RSI'] > 55: rs.append("볼린저 상단 이탈")
        if r['Time_NY'] > "09:30" and r['Close'] < ol and pv['Close'] >= ol and r['Volume'] > r['Vol_MA20']*1.3: rs.append("ORB 하단 이탈")

        if trend_on:
            if not (r['Close'] > r['EMA200'] and r['ADX'] >= 20.0 and r['RSI'] < 68.0): rb = []
            if not (r['Close'] < r['EMA200'] and r['ADX'] >= 20.0 and r['RSI'] > 32.0): rs = []

        if s_filter.startswith("🏆"):
            if rb:
                tag = "★강력매수: " if len(rb) >= 2 else ""
                df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, tag + ", ".join(rb)
            if rs and not df.at[idx, 'Buy_Signal']:
                tag = "★강력매도: " if len(rs) >= 2 else ""
                df.at[idx, 'Sell_Signal'], df.at[idx, 'Signal_Reason'] = True, tag + ", ".join(rs)
        else:
            key = s_filter.split(".").pop().strip()
            mb = [x for x in rb if key in x]
            ms = [x for x in rs if key in x]
            if mb: df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, ", ".join(mb)
            if ms and not mb: df.at[idx, 'Sell_Signal'], df.at[idx, 'Signal_Reason'] = True, ", ".join(ms)
    return df

def run_bt(df_in, n_bars, tp_m, sl_m):
    trades, c_arr, h_arr, l_arr, a_arr, idx_arr = [], df_in['Close'].to_numpy(float), df_in['High'].to_numpy(float), df_in['Low'].to_numpy(float), df_in['ATR'].to_numpy(float), df_in.index
    b_arr, s_arr = df_in['Buy_Signal'].to_numpy(), df_in['Sell_Signal'].to_numpy()
    for p in range(len(df_in) - 1):
        if not (b_arr[p] or s_arr[p]): continue
        ep, av, end_p = c_arr[p], a_arr[p], min(p + n_bars, len(df_in) - 1)
        xp, xdt = c_arr[end_p], idx_arr[end_p]
        if b_arr[p]:
            tp_t, sl_t = ep + av * tp_m, ep - av * sl_m
            for s in range(p + 1, end_p + 1):
                if l_arr[s] <= sl_t: xp, xdt = sl_t, idx_arr[s]; break
                if h_arr[s] >= tp_t: xp, xdt = tp_t, idx_arr[s]; break
            trades.append({"date": xdt, "ret": (xp - ep) / ep})
        else:
            tp_t, sl_t = ep - av * tp_m, ep + av * sl_m
            for s in range(p + 1, end_p + 1):
                if h_arr[s] >= sl_t: xp, xdt = sl_t, idx_arr[s]; break
                if l_arr[s] <= tp_t: xp, xdt = tp_t, idx_arr[s]; break
            trades.append({"date": xdt, "ret": (ep - xp) / ep})
    if not trades: return 0, 0.0, 0.0, 0.0, 0.0, pd.DataFrame()
    tdf = pd.DataFrame(trades)
    tdf['eq'] = (1 + tdf['ret']).cumprod()
    tdf['cum'] = (tdf['eq'] - 1) * 100
    tdf['dd'] = ((tdf['eq'] - tdf['eq'].cummax()) / tdf['eq'].cummax()) * 100
    return len(tdf), float((tdf['ret'] > 0).mean()*100), float(tdf['ret'].mean()*100), float(tdf['cum'].iloc[-1]), float(tdf['dd'].min()), tdf

def find_best_params(df_in):
    best = None
    for hb in (4, 6, 8, 12, 16, 20):
        for sm in (0.8, 1.0, 1.2, 1.5, 2.0):
            for tm in (1.2, 1.5, 2.0, 2.5, 3.0):
                c, w, a, t, m, _ = run_bt(df_in, hb, tm, sm)
                if c > 0:
if "hold_bars" not in st.session_state: st.session_state.hold_bars = 8
if "sl_mult" not in st.session_state: st.session_state.sl_mult = 1.2
if "tp_mult" not in st.session_state: st.session_state.tp_mult = 2.0

with st.expander("⚙️ 글로벌 고승률 30분/1시간봉 설정 (티커 입력 시 자동 최적화)", expanded=False):
    ticker = st.text_input("분석 티커 (입력 시 최고승률 보유봉·ATR 자동 세팅)", value="SOXL").upper().strip()
    timeframe = st.radio("최적화 타임프레임", ["30분봉", "1시간봉"], horizontal=True)
    strategy_filter = st.selectbox("글로벌 검증 알고리즘", [
        "🏆 최고승률 컨플루언스 (전체 통합)",
        "1. VWAP+RSI 눌림",
        "2. EMA 정배열 회귀",
        "3. MACD+VWAP 반전",
        "4. 볼린저 하단 반등",
        "5. ORB 돌파"
    ])
    strict_trend = st.checkbox("상위 추세(EMA200) & ADX>20 횡보장 차단 필터", value=True)

    df_pre = load_quant_data(ticker, timeframe, strategy_filter, strict_trend)
    opt_key = f"{ticker}_{timeframe}_{strategy_filter}_{strict_trend}"
    if df_pre is not None and st.session_state.get("last_opt_key") != opt_key:
        best_p = find_best_params(df_pre)
        if best_p:
            st.session_state.hold_bars = int(best_p["hb"])
            st.session_state.sl_mult = float(best_p["sl"])
            st.session_state.tp_mult = float(best_p["tp"])
            st.session_state.opt_summary = best_p
        else:
            st.session_state.opt_summary = None
        st.session_state.last_opt_key = opt_key

    c1, c2, c3 = st.columns(3)
    hold_bars = c1.slider("최대 보유봉", 3, 24, key="hold_bars")
    sl_mult = c2.number_input("손절 ATR", 0.5, 4.0, step=0.1, key="sl_mult")
    tp_mult = c3.number_input("익절 ATR", 1.0, 8.0, step=0.5, key="tp_mult")
    watchlist_str = st.text_input("글로벌 스캐너 종목", value="SOXL, TQQQ, QQQ, NVDA, TSLA, AAPL, MSFT, AMD, META")
    kakao_token = st.text_input("카카오톡 Access Token (선택)", type="password")
    if st.button("🔄 실시간 데이터 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.session_state.last_opt_key = ""

tab1, tab2 = st.tabs(["📊 30분/1시간 매수·매도 분석", "🌍 글로벌 고승률 스캐너"])
with tab1:
    df = df_pre
    if df is None: st.error(f"'{ticker}' 데이터를 불러올 수 없습니다. 티커명을 확인해 주세요.")
    else:
        last, prev = df.iloc[-1], df.iloc[-2]
        chg_pct = ((float(last['Close']) - float(prev['Close'])) / float(prev['Close'])) * 100
        if last['Sell_Signal']:
            tp_p, sl_p = float(last['Close']) - float(last['ATR'])*tp_mult, float(last['Close']) + float(last['ATR'])*sl_mult
            st_html = f'<span class="badge-sell">🔴 매도 · {last["Signal_Reason"]}</span>'
        elif last['Buy_Signal']:
            tp_p, sl_p = float(last['Close']) + float(last['ATR'])*tp_mult, float(last['Close']) - float(last['ATR'])*sl_mult
            st_html = f'<span class="badge-buy">🟢 매수 · {last["Signal_Reason"]}</span>'
        else:
            tp_p, sl_p = float(last['Close']) + float(last['ATR'])*tp_mult, float(last['Close']) - float(last['ATR'])*sl_mult
            st_html = '<span class="badge-neutral">청산/관망</span>'

        st.markdown(f"### 📈 {ticker} ({timeframe}) {st_html}", unsafe_allow_html=True)

        opt_info = st.session_state.get("opt_summary")
        if opt_info:
            st.markdown(f'<div class="opt-box"><strong style="color:#10B981;">✨ [{ticker} {timeframe}] AI 최고승률 자동 세팅 적용됨</strong><br>최적 설정: <strong>최대 {opt_info["hb"]}봉 보유</strong> · 손절 <strong>{opt_info["sl"]} ATR</strong> · 익절 <strong>{opt_info["tp"]} ATR</strong> (최고승률 <strong>{opt_info["win"]:.1f}%</strong>)</div>', unsafe_allow_html=True)

        cnt, win, avg, tot, mdd, tdf = run_bt(df, hold_bars, tp_mult, sl_mult)
        st.markdown(f"""<div class="kpi-grid">
        <div class="metric-card"><div class="metric-title">현재가 (USD)</div><div class="metric-value">${last['Close']:,.2f} ({chg_pct:+.2f}%)</div></div>
        <div class="metric-card"><div class="metric-title">RSI / ADX 추세강도</div><div class="metric-value">{last['RSI']:.1f} / {last['ADX']:.1f}</div></div>
        <div class="metric-card"><div class="metric-title">검증 승률 ({cnt}회 매매)</div><div class="metric-value" style="color:#10B981;">{win:.1f}% (평균 {avg:+.2f}%)</div></div>
        <div class="metric-card"><div class="metric-title">누적 수익률 / MDD</div><div class="metric-value">{tot:+.2f}% ({mdd:.2f}%)</div></div>
        </div><div class="tpsl-bar">
        <div><div class="ohlc-lbl">🎯 목표가</div><div class="ohlc-val" style="color:#10B981;">${tp_p:,.2f}</div></div>
        <div><div class="ohlc-lbl">🛡️ 손절가</div><div class="ohlc-val" style="color:#EF4444;">${sl_p:,.2f}</div></div>
        <div><div class="ohlc-lbl">⚖️ 손익비(R:R)</div><div class="ohlc-val" style="color:#F59E0B;">1 : {tp_mult/sl_mult:.2f}</div></div>
        </div>""", unsafe_allow_html=True)

        pdf = df.tail(80)
        xl = pdf.index.tz_convert('America/New_York').strftime('%m-%d %H:%M').tolist()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.78, 0.22])
        fig.add_trace(go.Candlestick(x=xl, open=pdf['Open'], high=pdf['High'], low=pdf['Low'], close=pdf['Close'], name="가격"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xl, y=pdf['EMA20'], line=dict(color='#F59E0B', width=1.1), name="EMA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xl, y=pdf['EMA200'], line=dict(color='#64748B', width=1.1, dash='dash'), name="EMA200"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xl, y=pdf['VWAP'], line=dict(color='#A855F7', width=1.2, dash='dot'), name="VWAP"), row=1, col=1)
        buys = pdf[pdf['Buy_Signal']]
        if not buys.empty:
            fig.add_trace(go.Scatter(x=buys.index.tz_convert('America/New_York').strftime('%m-%d %H:%M').tolist(), y=(buys['Low']*0.997).tolist(), mode='markers', marker=dict(symbol='triangle-up', size=11, color='#10B981'), text=buys['Signal_Reason'].tolist(), name="매수"), row=1, col=1)
        sells = pdf[pdf['Sell_Signal']]
        if not sells.empty:
            fig.add_trace(go.Scatter(x=sells.index.tz_convert('America/New_York').strftime('%m-%d %H:%M').tolist(), y=(sells['High']*1.003).tolist(), mode='markers', marker=dict(symbol='triangle-down', size=11, color='#EF4444'), text=sells['Signal_Reason'].tolist(), name="매도"), row=1, col=1)
        fig.add_trace(go.Bar(x=xl, y=pdf['Volume'], name="거래량", showlegend=False), row=2, col=1)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0B0E14", plot_bgcolor="#131722", height=400, margin=dict(l=5,r=5,t=10,b=10), xaxis_rangeslider_visible=False)
        fig.update_xaxes(type='category', nticks=6)
        st.plotly_chart(fig, use_container_width=True)

        if st.button("💛 카톡으로 현재 분석 전송", use_container_width=True):
            ok, m = send_kakao(kakao_token, f"[{ticker} {timeframe}] ${last['Close']:,.2f} ({chg_pct:+.2f}%)\n최적보유:{hold_bars}봉 | 승률:{win:.1f}%\n신호: {last['Signal_Reason'] or '청산/관망'}\n목표: ${tp_p:,.2f} / 손절: ${sl_p:,.2f}")
            st.toast(m)

        st.markdown("#### 🔔 최근 매수·매도 시그널 타임라인")
        for idx, row in df.tail(18).iloc[::-1].iterrows():
            ny_str = idx.tz_convert('America/New_York').strftime('%m-%d %H:%M (NY)')
            price_v, atr_v = float(row['Close']), float(row['ATR'])
            if row['Buy_Signal']:
                t_p, s_p = price_v + atr_v * tp_mult, price_v - atr_v * sl_mult
                st.markdown(f"""<div class="list-card"><div class="list-top"><div class="list-left"><span class="badge-buy">🟢 매수</span><span class="strat-txt">{row['Signal_Reason']}</span></div><span class="price-txt">${price_v:,.2f}</span></div><div class="list-sub"><span>🕒 {ny_str}</span><span>목표 <strong style="color:#EF4444;">${t_p:,.2f}</strong> · 손절 <strong style="color:#3B82F6;">${s_p:,.2f}</strong></span></div></div>""", unsafe_allow_html=True)
            elif row['Sell_Signal']:
                t_p, s_p = price_v - atr_v * tp_mult, price_v + atr_v * sl_mult
                st.markdown(f"""<div class="list-card"><div class="list-top"><div class="list-left"><span class="badge-sell">🔴 매도</span><span class="strat-txt">{row['Signal_Reason']}</span></div><span class="price-txt">${price_v:,.2f}</span></div><div class="list-sub"><span>🕒 {ny_str}</span><span>목표 <strong style="color:#3B82F6;">${t_p:,.2f}</strong> · 손절 <strong style="color:#EF4444;">${s_p:,.2f}</strong></span></div></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="list-card" style="padding:7px 12px; opacity:0.75;"><div class="list-sub" style="margin-top:0;"><span>🕒 {ny_str}</span><span>청산/관망</span></div></div>""", unsafe_allow_html=True)

with tab2:
    syms = [s.strip().upper() for s in watchlist_str.split(",") if s.strip()]
    if st.button("🚀 30분봉 + 1시간봉 종목별 최고승률 자동 스캔", use_container_width=True):
        scan_list = []
        bar = st.progress(0, text="종목별 최고승률 파라미터 탐색 및 스캔 중...")
        for idx_s, s in enumerate(syms):
            bar.progress((idx_s + 1) / max(len(syms), 1), text=f"🔍 {s} 최고승률 시뮬레이션 중... ({idx_s+1}/{len(syms)})")
            d30 = load_quant_data(s, "30분봉", strategy_filter, strict_trend)
            d60 = load_quant_data(s, "1시간봉", strategy_filter, strict_trend)
            if d30 is not None and d60 is not None and len(d30) >= 2:
                sl, sp = d30.iloc[-1], d30.iloc[-2]
                cp = ((float(sl['Close']) - float(sp['Close'])) / float(sp['Close'])) * 100
                bp60 = find_best_params(d60)
                w60 = bp60["win"] if bp60 else 0.0
                hb60 = bp60["hb"] if bp60 else hold_bars
                s_tp_m = bp60["tp"] if bp60 else tp_mult
                s_sl_m = bp60["sl"] if bp60 else sl_mult
                sig_type, sig_txt = "NEUTRAL", "청산/관망"
                tp_v, sl_v = float(sl['Close']) + float(sl['ATR']) * s_tp_m, float(sl['Close']) - float(sl['ATR']) * s_sl_m
                for sub_df, lbl in [(d30.tail(3), "30m"), (d60.tail(3), "1h")]:
                    sig_rows = sub_df[sub_df['Buy_Signal'] | sub_df['Sell_Signal']]
                    if not sig_rows.empty:
