import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests, json

st.set_page_config(page_title="Global Quant Pro - 30m/1h High WinRate", layout="wide", initial_sidebar_state="collapsed")
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

with st.expander("⚙️ 글로벌 고승률 30분/1시간봉 설정", expanded=False):
    ticker = st.text_input("분석 티커 (미국주식/ETF/지수: QQQ, NVDA, TSLA, SPY, AAPL, BTC-USD)", value="QQQ").upper().strip()
    timeframe = st.radio("최적화 타임프레임", ["30분봉", "1시간봉"], horizontal=True)
    strategy_filter = st.selectbox("글로벌 검증 알고리즘", [
        "🏆 최고승률 컨플루언스 (2개 이상 동시 충족)",
        "1. VWAP 기관수급 눌림목",
        "2. EMA 정배열 + SuperTrend 반등",
        "3. MACD 골든크로스 + 거래량 돌파",
        "4. 볼린저밴드 하단 과매도 복귀",
        "5. 장초반 ORB 30분 고점 돌파"
    ])
    c1, c2, c3 = st.columns(3)
    hold_bars = c1.slider("최대 보유봉", 3, 24, 10)
    sl_mult = c2.number_input("손절 ATR", 0.5, 4.0, 1.2, 0.1)
    tp_mult = c3.number_input("익절 ATR", 1.0, 8.0, 2.5, 0.5)
    strict_trend = st.checkbox("상위 추세(EMA200 위) & ADX>22 횡보장 차단 필터", value=True)
    watchlist_str = st.text_input("글로벌 스캐너 종목", value="QQQ, SPY, NVDA, TSLA, AAPL, MSFT, AMD, META, AMZN, GOOGL")
    kakao_token = st.text_input("카카오톡 Access Token (선택)", type="password")
    if st.button("🔄 실시간 데이터 새로고침", use_container_width=True):
        st.cache_data.clear()

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

    # 핵심 추세 및 모멘텀 지표
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    d = df['Close'].diff()
    df['RSI'] = 100 - (100 / (1 + d.where(d > 0, 0).rolling(14).mean() / (-d.where(d < 0, 0)).rolling(14).mean().replace(0, np.nan)))

    # ATR 및 ADX 추세 강도
    hd, ld = df['High'].diff(), -df['Low'].diff()
    pdm, mdm = hd.where((hd > ld) & (hd > 0), 0.0), ld.where((ld > hd) & (ld > 0), 0.0)
    df['TR'] = pd.concat([df['High'] - df['Low'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean().bfill()
    atrs = df['TR'].ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan)
    pdi, mdi = 100 * (pdm.ewm(alpha=1/14, adjust=False).mean() / atrs), 100 * (mdm.ewm(alpha=1/14, adjust=False).mean() / atrs)
    df['ADX'] = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).fillna(0).ewm(alpha=1/14, adjust=False).mean()

    # MACD, 볼린저밴드, 기관 VWAP, SuperTrend(단순화)
    macd = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = macd - macd.ewm(span=9, adjust=False).mean()
    df['BB_Low'] = df['Close'].rolling(20).mean() - 2.0 * df['Close'].rolling(20).std()
    df['ST_Up'] = df['Close'] > (df['EMA20'] - 1.5 * df['ATR'])
    df['TP_Vol'] = ((df['High'] + df['Low'] + df['Close']) / 3) * df['Volume']
    df['VWAP'] = df.groupby('Date_NY')['TP_Vol'].cumsum() / df.groupby('Date_NY')['Volume'].cumsum().replace(0, np.nan)
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Buy_Signal'], df['Sell_Signal'], df['Signal_Reason'] = False, False, ""

    fb = df.groupby('Date_NY').first()
    orb_h = fb['High'].to_dict()

    for i in range(2, len(df)):
        idx, r, pv, pv2 = df.index[i], df.iloc[i], df.iloc[i-1], df.iloc[i-2]
        rb, rs = [], []
        if r['Close'] > r['VWAP'] and (pv['RSI'] <= 43 or pv2['RSI'] <= 41) and r['Close'] > r['Open']: rb.append("VWAP기관눌림")
        if r['EMA20'] > r['EMA50'] and r['ST_Up'] and (r['Low'] <= r['EMA20']*1.003) and r['Close'] > r['Open']: rb.append("EMA+ST반등")
        if r['Close'] > r['VWAP'] and r['MACD_Hist'] > 0 and pv['MACD_Hist'] <= 0 and r['Volume'] > r['Vol_MA20']*1.2: rb.append("MACD수급돌파")
        if (pv['Low'] <= pv['BB_Low'] or r['Low'] <= r['BB_Low']) and r['Close'] > r['BB_Low'] and r['Close'] > r['Open'] and r['RSI'] < 48: rb.append("BB하단복귀")
        oh = orb_h.get(r['Date_NY'], np.nan)
        if r['Time_NY'] > "09:30" and r['Close'] > oh and pv['Close'] <= oh and r['Volume'] > r['Vol_MA20']*1.4: rb.append("ORB돌파")

        if r['Close'] < r['VWAP'] and r['Close'] < r['EMA20'] and pv['Close'] >= pv['VWAP'] and r['RSI'] > 32: rs.append("VWAP이탈매도")

        if trend_on and not (r['Close'] > r['EMA200'] and r['ADX'] >= 22.0 and r['RSI'] < 68.0): rb = []
        if s_filter.startswith("🏆"):
            if len(rb) >= 2 or (rb and r['Volume'] > r['Vol_MA20'] * 1.6):
                df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, "★고승률매수: " + "+".join(rb)
            if rs: df.at[idx, 'Sell_Signal'], df.at[idx, 'Signal_Reason'] = True, "매도경고: " + "+".join(rs)
        else:
            key = s_filter.split(".")[1].strip()[:4]
            matched = [x for x in rb if key[:2] in x]
            if matched: df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, "매수: " + "+".join(matched)
            if rs: df.at[idx, 'Sell_Signal'], df.at[idx, 'Signal_Reason'] = True, "매도: " + "+".join(rs)
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

tab1, tab2 = st.tabs(["📊 30분/1시간 정밀 분석", "🌍 글로벌 고승률 스캐너"])
with tab1:
    df = load_quant_data(ticker, timeframe, strategy_filter, strict_trend)
    if df is None: st.error(f"'{ticker}' 데이터를 불러올 수 없습니다.")
    else:
        last, prev = df.iloc[-1], df.iloc[-2]
        chg_pct = ((float(last['Close']) - float(prev['Close'])) / float(prev['Close'])) * 100
        tp_p, sl_p = float(last['Close']) + float(last['ATR'])*tp_mult, float(last['Close']) - float(last['ATR'])*sl_mult
        st_html = f'<span class="badge-buy">{last["Signal_Reason"]}</span>' if last['Buy_Signal'] else (f'<span class="badge-sell">{last["Signal_Reason"]}</span>' if last['Sell_Signal'] else '<span class="badge-neutral">관망 / 타점대기</span>')
        st.markdown(f"### 📈 {ticker} ({timeframe}) {st_html}", unsafe_allow_html=True)
        cnt, win, avg, tot, mdd, tdf = run_bt(df, hold_bars, tp_mult, sl_mult)
        st.markdown(f"""<div class="kpi-grid">
        <div class="metric-card"><div class="metric-title">현재가 (USD)</div><div class="metric-value">${last['Close']:,.2f} ({chg_pct:+.2f}%)</div></div>
        <div class="metric-card"><div class="metric-title">RSI / ADX 추세강도</div><div class="metric-value">{last['RSI']:.1f} / {last['ADX']:.1f}</div></div>
        <div class="metric-card"><div class="metric-title">검증 승률 ({cnt}회 매매)</div><div class="metric-value" style="color:#10B981;">{win:.1f}% (평균 {avg:+.2f}%)</div></div>
        <div class="metric-card"><div class="metric-title">누적 수익률 / MDD</div><div class="metric-value">{tot:+.2f}% ({mdd:.2f}%)</div></div>
        </div><div class="tpsl-bar">
        <div><div class="ohlc-lbl">🎯 목표 익절(TP)</div><div class="ohlc-val" style="color:#10B981;">${tp_p:,.2f}</div></div>
        <div><div class="ohlc-lbl">🛡️ 손절 라인(SL)</div><div class="ohlc-val" style="color:#EF4444;">${sl_p:,.2f}</div></div>
        <div><div class="ohlc-lbl">⚖️ 손익비(R:R)</div><div class="ohlc-val" style="color:#F59E0B;">1 : {tp_mult/sl_mult:.2f}</div></div>
        </div>""", unsafe_allow_html=True)
        if st.button(f"✨ {ticker} ({timeframe}) 최고승률 손절/익절 자동 탐색", use_container_width=True):
            opts = []
            for sm in (1.0, 1.2, 1.5, 2.0):
                for tm in (2.0, 2.5, 3.0, 4.0):
                    if tm > sm:
                        c, w, a, t, m, _ = run_bt(df, hold_bars, tm, sm)
                        if c > 0: opts.append((w*0.5 + t*0.5 - abs(m)*0.3, sm, tm, w, t, m))
            if opts:
                opts.sort(reverse=True)
                _, b_sl, b_tp, b_w, b_t, b_m = opts
                st.markdown(f'<div class="opt-box"><strong style="color:#10B981;">🏆 AI 추천 최적 비율: 손절 {b_sl} ATR / 익절 {b_tp} ATR</strong><br>승률 {b_w:.1f}% | 누적수익 {b_t:+.2f}% | MDD {b_m:.2f}%</div>', unsafe_allow_html=True)
            else: st.warning("탐색할 매수 이력이 부족합니다. (설정에서 추세 필터를 해제해 보세요.)")
        pdf = df.tail(80)
        xl = pdf.index.tz_convert('America/New_York').strftime('%m-%d %H:%M').tolist()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.78, 0.22])
        fig.add_trace(go.Candlestick(x=xl, open=pdf['Open'], high=pdf['High'], low=pdf['Low'], close=pdf['Close'], name="가격"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xl, y=pdf['EMA20'], line=dict(color='#F59E0B', width=1.1), name="EMA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xl, y=pdf['EMA200'], line=dict(color='#64748B', width=1.1, dash='dash'), name="EMA200"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xl, y=pdf['VWAP'], line=dict(color='#A855F7', width=1.2, dash='dot'), name="VWAP"), row=1, col=1)
        buys = pdf[pdf['Buy_Signal']]
        if not buys.empty:
            bx = buys.index.tz_convert('America/New_York').strftime('%m-%d %H:%M').tolist()
            fig.add_trace(go.Scatter(x=bx, y=(buys['Low']*0.997).tolist(), mode='markers', marker=dict(symbol='triangle-up', size=11, color='#10B981'), text=buys['Signal_Reason'].tolist(), name="매수타점"), row=1, col=1)
        fig.add_trace(go.Bar(x=xl, y=pdf['Volume'], name="거래량", showlegend=False), row=2, col=1)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0B0E14", plot_bgcolor="#131722", height=410, margin=dict(l=5,r=5,t=10,b=10), xaxis_rangeslider_visible=False)
        fig.update_xaxes(type='category', nticks=6)
        st.plotly_chart(fig, use_container_width=True)
        if st.button("💛 카톡으로 현재 분석 전송", use_container_width=True):
            ok, m = send_kakao(kakao_token, f"[{ticker} {timeframe}] ${last['Close']:,.2f} ({chg_pct:+.2f}%)\n신호: {last['Signal_Reason'] or '관망'}\nTP: ${tp_p:,.2f} / SL: ${sl_p:,.2f}")
            st.toast(m)
        if not tdf.empty:
            st.markdown("#### 📈 누적 수익 곡선 (%)")
            st.line_chart(tdf.set_index('date')['cum'], height=180)

with tab2:
    syms = [s.strip().upper() for s in watchlist_str.split(",") if s.strip()]
    if st.button("🚀 30분봉 + 1시간봉 동시 고승률 스캔", use_container_width=True):
        scan_list = []
        for s in syms:
            d30 = load_quant_data(s, "30분봉", strategy_filter, strict_trend)
            d60 = load_quant_data(s, "1시간봉", strategy_filter, strict_trend)
            if d30 is not None and d60 is not None and len(d30) >= 2:
                sl, sp = d30.iloc[-1], d30.iloc[-2]
                cp = ((float(sl['Close']) - float(sp['Close'])) / float(sp['Close'])) * 100
                s30 = sl['Signal_Reason'] if (sl['Buy_Signal'] or sl['Sell_Signal']) else "관망"
                s60 = d60.iloc[-1]['Signal_Reason'] if (d60.iloc[-1]['Buy_Signal'] or d60.iloc[-1]['Sell_Signal']) else "관망"
                _, w60, _, _, _, _ = run_bt(d60, hold_bars, tp_mult, sl_mult)
                scan_list.append({"sym": s, "p": float(sl['Close']), "cp": cp, "s30": s30, "s60": s60, "w60": w60, "rsi": float(sl['RSI'])})
        scan_list.sort(key=lambda x: (0 if "매수" in x['s30'] or "매수" in x['s60'] else 1, -x['w60']))
        for it in scan_list:
            has_buy = ("매수" in it['s30']) or ("매수" in it['s60'])
            bcls = "badge-buy" if has_buy else "badge-neutral"
            badge_txt = f"30m:{it['s30']} | 1h:{it['s60']}" if has_buy else "관망"
            st.markdown(f'<div class="list-card"><div class="list-top"><strong>{it["sym"]} ${it["p"]:,.2f} ({it["cp"]:+.2f}%)</strong><span class="{bcls}">{badge_txt}</span></div><div class="list-sub"><span>1시간봉 검증승률: <strong style="color:#10B981;">{it["w60"]:.1f}%</strong></span><span>RSI: {it["rsi"]:.1f}</span></div></div>', unsafe_allow_html=True)
