import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests, json, time

st.set_page_config(page_title="Global Quant Pro", layout="wide", initial_sidebar_state="collapsed")
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
.alert-tp{background:rgba(16,185,129,.22);border:1.5px solid #10B981;border-radius:10px;padding:12px;margin:8px 0;color:#ECFDF5;font-size:.86rem}
.alert-sl{background:rgba(239,68,68,.22);border:1.5px solid #EF4444;border-radius:10px;padding:12px;margin:8px 0;color:#FEF2F2;font-size:.86rem}
.alert-run{background:rgba(56,189,248,.14);border:1.5px solid #38BDF8;border-radius:10px;padding:12px;margin:8px 0;color:#F0F9FF;font-size:.86rem}
.list-top{display:flex;justify-content:space-between;align-items:flex-start;gap:8px}
.list-left{display:flex;align-items:flex-start;gap:7px;flex:1;min-width:0}
.strat-txt{font-size:.85rem;font-weight:700;color:#F8FAFC;line-height:1.35;word-break:keep-all}
.price-txt{font-size:.9rem;font-weight:700;color:#F8FAFC;white-space:nowrap}
.list-sub{display:flex;justify-content:space-between;align-items:center;font-size:.74rem;color:#94A3B8;margin-top:6px}
div[data-testid="stExpander"]{background:#131722!important;border:1px solid #1E293B!important;border-radius:8px!important}
div[data-testid="stExpander"] summary p,label p,.stRadio label p,.stCheckbox label p{color:#F8FAFC!important;font-size:.82rem!important}
</style>""", unsafe_allow_html=True)

def send_discord(webhook_url, msg):
    if not webhook_url: return False, "디스코드 웹훅 URL을 입력해 주세요."
    try:
        r = requests.post(webhook_url.strip(), json={"content": msg}, timeout=5)
        return (True, "디스코드 전송 완료!") if r.status_code in (200, 204) else (False, f"전송 실패 ({r.status_code}): {r.text}")
    except Exception as e:
        return False, str(e)
@st.cache_data(ttl=15)
def load_quant_data(sym, tf, s_filter, trend_on, long_only=False):
    tf_map = {"15분봉": ("1mo", "15m"), "30분봉": ("1mo", "30m"), "1시간봉": ("3mo", "1h")}
    p, iv = tf_map.get(tf, ("1mo", "15m"))
    raw = yf.download(sym, prepost=True, period=p, interval=iv, progress=False, auto_adjust=True)
    if raw is None or raw.empty: return None
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna().copy()
    if len(df) < 30: return None

    is_krx = sym.endswith((".KS", ".KQ"))
    df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index
    loc = df.tz_convert('Asia/Seoul' if is_krx else 'America/New_York')
    df['Date_Loc'], df['Time_Loc'] = loc.index.date, loc.index.strftime('%H:%M')
    open_time = "09:00" if is_krx else "09:30"

    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    d = df['Close'].diff()
    df['RSI'] = 100 - (100 / (1 + d.where(d > 0, 0).rolling(14).mean() / (-d.where(d < 0, 0)).rolling(14).mean().replace(0, np.nan)))

    hd, ld = df['High'].diff(), -df['Low'].diff()
    pdm, mdm = hd.where((hd > ld) & (hd > 0), 0.0), ld.where((ld > hd) & (ld > 0), 0.0)
    df['TR'] = pd.concat((df['High'] - df['Low'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()), axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean().bfill()
    atrs = df['TR'].ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan)
    pdi, mdi = 100 * (pdm.ewm(alpha=1/14, adjust=False).mean() / atrs), 100 * (mdm.ewm(alpha=1/14, adjust=False).mean() / atrs)
    df['ADX'] = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).fillna(0).ewm(alpha=1/14, adjust=False).mean()

    macd = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = macd - macd.ewm(span=9, adjust=False).mean()
    bb_m, bb_s = df['Close'].rolling(20).mean(), df['Close'].rolling(20).std()
    df['BB_Low'], df['BB_Up'] = bb_m - 1.95 * bb_s, bb_m + 1.95 * bb_s
    df['TP_Vol'] = ((df['High'] + df['Low'] + df['Close']) / 3) * df['Volume']
    df['VWAP'] = df.groupby('Date_Loc')['TP_Vol'].cumsum() / df.groupby('Date_Loc')['Volume'].cumsum().replace(0, np.nan)
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Buy_Signal'], df['Sell_Signal'], df['Signal_Reason'] = False, False, ""
    fb = df.groupby('Date_Loc').first()
    orb_h, orb_l = fb['High'].to_dict(), fb['Low'].to_dict()
    is_15m = (tf == "15분봉")

    for i in range(2, len(df)):
        idx, r, pv, pv2 = df.index[i], df.iloc[i], df.iloc[i-1], df.iloc[i-2]
        rb, rs = list(), list()

        if r['Close'] > r['VWAP'] and (pv['RSI'] <= (45 if is_15m else 47) or pv2['RSI'] <= (43 if is_15m else 45)) and r['Close'] > r['Open']:
            rb.append("VWAP+RSI 눌림")
        if r['EMA20'] > r['EMA50'] and (r['Low'] <= r['EMA20']*1.005) and r['Close'] > r['Open']:
            rb.append("EMA 정배열 회귀")
        if r['Close'] > r['VWAP'] and r['MACD_Hist'] > 0 and pv['MACD_Hist'] <= 0 and r['Volume'] > r['Vol_MA20']*1.1:
            rb.append("MACD+VWAP 반전")
        if (pv['Low'] <= pv['BB_Low'] or r['Low'] <= r['BB_Low']) and r['Close'] > r['BB_Low'] and r['Close'] > r['Open'] and r['RSI'] < 50:
            rb.append("볼린저 하단 반등")
        oh, ol = orb_h.get(r['Date_Loc'], np.nan), orb_l.get(r['Date_Loc'], np.nan)
        if r['Time_Loc'] > open_time and r['Close'] > oh and pv['Close'] <= oh and r['Volume'] > r['Vol_MA20']*1.3:
            rb.append("ORB 돌파")

        v_guard = (r['Close'] > r['VWAP']) and (r['EMA20'] > pv['EMA20'])
        if not long_only and not v_guard:
            if r['Close'] < r['VWAP'] and (pv['RSI'] >= (60 if is_15m else 58) or pv2['RSI'] >= (62 if is_15m else 60)) and r['Close'] < r['Open']:
                rs.append("VWAP 저항+RSI 과열")
            if r['EMA20'] < r['EMA50'] and (r['High'] >= r['EMA20']*0.995) and r['Close'] < r['Open']:
                rs.append("EMA 역배열 저항")
            if r['Close'] < r['VWAP'] and r['MACD_Hist'] < 0 and pv['MACD_Hist'] >= 0:
                rs.append("MACD+VWAP 하락반전")
            if (pv['High'] >= pv['BB_Up'] or r['High'] >= pv['BB_Up']) and r['Close'] < r['BB_Up'] and r['Close'] < r['Open'] and r['RSI'] > 52:
                rs.append("볼린저 상단 이탈")
            if r['Time_Loc'] > open_time and r['Close'] < ol and pv['Close'] >= ol and r['Volume'] > r['Vol_MA20']*1.3:
                rs.append("ORB 하단 이탈")

        if trend_on:
            if not (r['Close'] > r['EMA100'] and r['ADX'] >= 15.0 and r['RSI'] < 70.0):
                rb = list()
            if not (r['Close'] < r['EMA100'] and r['ADX'] >= 15.0 and r['RSI'] > 30.0):
                rs = list()

        is_strong_only = s_filter.startswith("🔥")
        min_cnt = 2 if is_strong_only else 1
        allow_all = s_filter.startswith("🏆") or is_strong_only
        key = s_filter.split(".").pop().split(" ").pop(0)

        if len(rb) >= min_cnt and (allow_all or any(key in x for x in rb)):
            tag = "★강력매수: " if len(rb) >= 2 else ""
            df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, tag + ", ".join(rb)
        if len(rs) >= min_cnt and not df.at[idx, 'Buy_Signal'] and (allow_all or any(key in x for x in rs)):
            tag = "★강력매도: " if len(rs) >= 2 else ""
            df.at[idx, 'Sell_Signal'], df.at[idx, 'Signal_Reason'] = True, tag + ", ".join(rs)
    return df
def run_bt(df_in, n_bars, tp_m, sl_m, no_overlap=True, use_be=True, intraday_exit=True, fee_pct=0.10):
    trades, last_i, active_until = list(), len(df_in) - 1, -1
    c_arr, h_arr, l_arr, a_arr = df_in['Close'].to_numpy(float), df_in['High'].to_numpy(float), df_in['Low'].to_numpy(float), df_in['ATR'].to_numpy(float)
    b_arr, s_arr, rsn_arr, idx_arr = df_in['Buy_Signal'].to_numpy(bool), df_in['Sell_Signal'].to_numpy(bool), df_in['Signal_Reason'].to_numpy(str), df_in.index
    t_arr = df_in['Time_Loc'].to_numpy(str)

    for i in range(len(df_in)):
        if (no_overlap and i <= active_until) or not (b_arr[i] or s_arr[i]):
            continue
        side, entry, atr = ("BUY" if b_arr[i] else "SELL"), c_arr[i], a_arr[i]
        tp = entry + tp_m * atr if side == "BUY" else entry - tp_m * atr
        sl = entry - sl_m * atr if side == "BUY" else entry + sl_m * atr
        be_trigger = entry + 0.8 * atr if side == "BUY" else entry - 0.8 * atr
        be_armed = False

        exit_idx = min(i + n_bars, last_i)
        res, exit_p = ("⏳진행중" if i + n_bars > last_i else "보유만기"), c_arr[exit_idx]

        for j in range(i + 1, min(i + 1 + n_bars, len(df_in))):
            is_live = (j == last_i)
            if side == "BUY":
                if l_arr[j] <= sl:
                    tag = "🛡️본절청산" if be_armed else ("🚨장중손절" if is_live else "손절")
                    res, exit_p, exit_idx = tag, sl, j
                    break
                if h_arr[j] >= tp:
                    res, exit_p, exit_idx = ("🎯장중익절" if is_live else "익절"), tp, j
                    break
                if use_be and h_arr[j] >= be_trigger:
                    be_armed = True
                    sl = entry * (1 + fee_pct / 100)
            else:
                if h_arr[j] >= sl:
                    tag = "🛡️본절청산" if be_armed else ("🚨장중손절" if is_live else "손절")
                    res, exit_p, exit_idx = tag, sl, j
                    break
                if l_arr[j] <= tp:
                    res, exit_p, exit_idx = ("🎯장중익절" if is_live else "익절"), tp, j
                    break
                if use_be and l_arr[j] <= be_trigger:
                    be_armed = True
                    sl = entry * (1 - fee_pct / 100)

            if intraday_exit and t_arr[j] >= "15:30":
                res, exit_p, exit_idx = "🌙당일청산", c_arr[j], j
                break

        active_until = exit_idx
        raw_pnl = ((exit_p - entry) / entry * 100) if side == "BUY" else ((entry - exit_p) / entry * 100)
        trades.append({"time": idx_arr[i], "side": side, "reason": rsn_arr[i], "entry": entry, "tp": tp, "sl": sl, "res": res, "pnl": raw_pnl - fee_pct})
    return pd.DataFrame(trades)
if "sent_alerts" not in st.session_state:
    st.session_state["sent_alerts"] = set()

t1, t2 = st.columns((2.2, 1.0))
t1.markdown("#### ⚡ Global Quant Pro")
if t2.button("🔄 실시간 새로고침", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

with st.expander("⚙️ 종목 · 분봉 · 레버리지 단타 설정", expanded=True):
    c1, c2 = st.columns((1.2, 1.8))
    sym = c1.text_input("티커 (예: TQQQ, SOXL, NVDA)", value="TQQQ").upper().strip()
    tf = c2.radio("타임프레임", ("15분봉", "30분봉", "1시간봉"), index=0, horizontal=True)
    watchlist_raw = st.text_input("📋 워치리스트 (쉼표 구분)", value="TQQQ, SOXL, NVDL, TSLL, NVDA, QQQ")
    s_filter = st.selectbox("시그널 필터", ("🏆 전체 시그널 (일반+강력)", "🔥 강력신호만 (★)", "1. VWAP+RSI", "2. EMA", "3. MACD+VWAP", "4. 볼린저", "5. ORB"))

    c3, c4 = st.columns(2)
    trend_on = c3.checkbox("EMA100 + ADX 추세필터", value=True)
    auto_opt = c4.checkbox("기대수익×손익비 자동탐색", value=True)

    c5, c6 = st.columns(2)
    long_only = c5.checkbox("현물 전용 (매수만)", value=False)
    use_be = c6.checkbox("🛡️ +0.8 ATR 본절보호", value=True)

    intraday_exit = st.checkbox("🌙 당일 종가 청산 (오버나이트 갭 회피)", value=True)

    b1, b2, b3 = st.columns(3)
    n_bars = b1.number_input("최대보유(봉)", 3, 48, 12 if tf == "15분봉" else 8)
    tp_m = b2.number_input("익절(ATR)", 0.5, 5.0, 2.0, 0.5)
    sl_m = b3.number_input("손절(ATR)", 0.5, 3.0, 1.0, 0.5)

    k1, k2 = st.columns((1.8, 1.2))
    discord_url = k1.text_input("디스코드 Webhook URL", type="password")
    auto_discord = k2.checkbox("신규·익절·손절 즉시 알림", value=True)

tab1, tab2 = st.tabs(("📊 메인 종목 분석", "🔍 워치리스트 스캔"))

with tab1:
    df = load_quant_data(sym, tf, s_filter, trend_on, long_only)
    if df is None:
        st.error("데이터를 불러올 수 없습니다. 티커명을 확인해 주세요.")
        st.stop()

    is_krx = sym.endswith((".KS", ".KQ"))
    cur_fmt = "{:,.0f}원" if is_krx else "${:,.2f}"
    if auto_opt:
        best_score, best_cfg, best_wr = -999.0, (n_bars, tp_m, sl_m), 0.0
        for nb in ((6, 8, 12, 16) if tf == "15분봉" else (4, 6, 8, 12)):
            for tm in (1.5, 2.0, 2.5):
                for sm in (0.8, 1.0, 1.2):
                    tmp = run_bt(df, nb, tm, sm, True, use_be, intraday_exit)
                    if len(tmp) >= 2:
                        wr = (tmp['pnl'] > 0).mean() * 100
                        gw = tmp.loc[tmp['pnl'] > 0, 'pnl'].sum()
                        gl = abs(tmp.loc[tmp['pnl'] < 0, 'pnl'].sum())
                        pf_tmp = (gw / gl) if gl > 0 else 3.0
                        score = tmp['pnl'].mean() * min(pf_tmp, 4.0)
                        if score > best_score:
                            best_score, best_cfg, best_wr = score, (nb, tm, sm), wr
        n_bars, tp_m, sl_m = best_cfg
        if best_score > -999.0:
            st.markdown(f"<div class='opt-box'>🎯 <b>{tf} 단타 최적화 완료:</b> 보유 {n_bars}봉 | 익절 {tp_m} ATR | 손절 {sl_m} ATR (승률 {best_wr:.1f}%)</div>", unsafe_allow_html=True)

    bt = run_bt(df, n_bars, tp_m, sl_m, True, use_be, intraday_exit)
    last = df.iloc[-1]

    if not bt.empty and bt.iloc[-1]['res'] in ("⏳진행중", "🎯장중익절", "🚨장중손절", "🛡️본절청산"):
        lv = bt.iloc[-1]
        a_key = f"{sym}_{tf}_{lv['time']}_{lv['res']}"
        if lv['res'] == "🎯장중익절":
            st.markdown(f"<div class='alert-tp'>🎉 <b>[봉 마감 전 장중 익절 도달!] {sym} ({lv['side']})</b><br>진입 {cur_fmt.format(lv['entry'])} → 목표가 <b>{cur_fmt.format(lv['tp'])}</b> 돌파! (<b>{lv['pnl']:+.2f}%</b>)</div>", unsafe_allow_html=True)
        elif lv['res'] in ("🚨장중손절", "🛡️본절청산"):
            st.markdown(f"<div class='alert-sl'>🚨 <b>[봉 마감 전 {lv['res']}!] {sym} ({lv['side']})</b><br>진입 {cur_fmt.format(lv['entry'])} → 청산가 <b>{cur_fmt.format(lv['sl'])}</b> 터치! (<b>{lv['pnl']:+.2f}%</b>)</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='alert-run'>⏳ <b>[진행 중 포지션 실시간 감시] {sym} ({lv['side']})</b><br>진입 {cur_fmt.format(lv['entry'])} | 현재 {cur_fmt.format(last['Close'])} (<b>{lv['pnl']:+.2f}%</b>) · 목표 {cur_fmt.format(lv['tp'])} / 손절 {cur_fmt.format(lv['sl'])}</div>", unsafe_allow_html=True)

        if auto_discord and discord_url and a_key not in st.session_state["sent_alerts"]:
            side_kr = "🟢 매수" if lv['side'] == "BUY" else "🔴 매도"
            d_msg = f"⚡ **[{sym} {tf} {lv['res']} - {side_kr}]**\n• 전략: {lv['reason']}\n• 진입가: `{cur_fmt.format(lv['entry'])}`\n• 목표가: `{cur_fmt.format(lv['tp'])}` / 손절가: `{cur_fmt.format(lv['sl'])}`\n• 현재수익: **{lv['pnl']:+.2f}%**"
            ok, _ = send_discord(discord_url, d_msg)
            if ok: st.session_state["sent_alerts"].add(a_key)

    win_rate = (bt['pnl'] > 0).mean() * 100 if not bt.empty else 0.0
    tot_pnl = bt['pnl'].sum() if not bt.empty else 0.0
    gw = bt.loc[bt['pnl'] > 0, 'pnl'].sum() if not bt.empty else 0.0
    gl = abs(bt.loc[bt['pnl'] < 0, 'pnl'].sum()) if not bt.empty else 0.0
    pf = (gw / gl) if gl > 0 else (99.9 if gw > 0 else 0.0)
    mdd = (((1 + bt['pnl'] / 100).cumprod() / (1 + bt['pnl'] / 100).cumprod().cummax() - 1) * 100).min() if not bt.empty else 0.0

    st.markdown(f"""
    <div class='kpi-grid'>
      <div class='metric-card'><div class='metric-title'>{sym} 현재가 ({tf})</div><div class='metric-value'>{cur_fmt.format(last['Close'])}</div></div>
      <div class='metric-card'><div class='metric-title'>시그널 승률 ({len(bt)}건)</div><div class='metric-value'>{win_rate:.1f}% <span style='font-size:.75rem;color:#94A3B8'>({tot_pnl:+.1f}%)</span></div></div>
      <div class='metric-card'><div class='metric-title'>손익비 (PF)</div><div class='metric-value' style='color:{"#10B981" if pf >= 1.5 else "#F8FAFC"}'>{pf:.2f}</div></div>
      <div class='metric-card'><div class='metric-title'>최대 낙폭 (MDD)</div><div class='metric-value' style='color:#EF4444'>{mdd:.2f}%</div></div>
    </div>""", unsafe_allow_html=True)

    tp_val = last['Close'] + tp_m * last['ATR']
    sl_val = last['Close'] - sl_m * last['ATR']
    st.markdown(f"""
    <div class='tpsl-bar'>
      <div><div class='ohlc-lbl'>신규 진입 기준가</div><div class='ohlc-val'>{cur_fmt.format(last['Close'])}</div></div>
      <div><div class='ohlc-lbl'>신규 익절가 (+{tp_m} ATR)</div><div class='ohlc-val' style='color:#10B981'>{cur_fmt.format(tp_val)}</div></div>
      <div><div class='ohlc-lbl'>신규 손절가 (-{sl_m} ATR)</div><div class='ohlc-val' style='color:#EF4444'>{cur_fmt.format(sl_val)}</div></div>
    </div>""", unsafe_allow_html=True)

    plot_df = df.tail(100)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=(0.75, 0.25), vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close']), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA20'], line=dict(color='#38BDF8', width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['VWAP'], line=dict(color='#F59E0B', width=1.2, dash='dot')), row=1, col=1)
    buys, sells = plot_df.loc[plot_df['Buy_Signal']], plot_df.loc[plot_df['Sell_Signal']]
    fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.996, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#10B981')), row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.004, mode='markers', marker=dict(symbol='triangle-down', size=12, color='#EF4444')), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], line=dict(color='#A855F7', width=1.2)), row=2, col=1)
    fig.update_layout(height=400, margin=dict(l=5, r=5, t=10, b=10), paper_bgcolor='#0B0E14', plot_bgcolor='#131722', font=dict(color='#E2E8F0'), xaxis_rangeslider_visible=False, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("##### 📌 최근 발생 시그널 내역")
    if bt.empty:
        st.info("선택한 조건에서 발생한 시그널이 없습니다. 추세필터 체크를 잠시 해제해 보세요.")
    else:
        tz_str = 'Asia/Seoul' if is_krx else 'America/New_York'
        tz_lbl = 'KST' if is_krx else 'NY'
        recent = bt.iloc[::-1].head(15)
        for _, row in recent.iterrows():
            badge = "<span class='badge-buy'>🟢 매수</span>" if row['side'] == "BUY" else "<span class='badge-sell'>🔴 매도</span>"
            p_col = "#10B981" if row['pnl'] > 0 else "#EF4444"
            t_str = row['time'].tz_convert(tz_str).strftime(f'%m-%d %H:%M ({tz_lbl})')
            st.markdown(f"""
            <div class='list-card'>
              <div class='list-top'>
                <div class='list-left'>{badge}<div class='strat-txt'>{row['reason']}</div></div>
                <div class='price-txt'>{cur_fmt.format(row['entry'])}</div>
              </div>
              <div class='list-sub'>
                <span>🕒 {t_str}</span>
                <span>목표 <b style='color:#10B981'>{cur_fmt.format(row['tp'])}</b> · 손절 <b style='color:#EF4444'>{cur_fmt.format(row['sl'])}</b></span>
              </div>
              <div class='list-sub'>
                <span>결과: <b>{row['res']}</b></span>
                <span>수익률: <b style='color:{p_col}'>{row['pnl']:+.2f}%</b></span>
              </div>
            </div>""", unsafe_allow_html=True)

        if st.button("📲 최신 시그널 디스코드 전송"):
            ls = recent.head(1).squeeze()
            side_kr = "🟢 매수" if ls['side'] == "BUY" else "🔴 매도"
            ok, info = send_discord(discord_url, f"⚡ **[{sym} {tf} {side_kr} 시그널]**\n• 전략: {ls['reason']}\n• 진입가: `{cur_fmt.format(ls['entry'])}`\n• 목표가: `{cur_fmt.format(ls['tp'])}` / 손절가: `{cur_fmt.format(ls['sl'])}`\n• 현재결과: {ls['res']} (**{ls['pnl']:+.2f}%**)")
            if ok:
                st.success(info)
            else:
                st.warning(info)

with tab2:
    st.markdown(f"##### 🔍 워치리스트 일괄 스캔 ({tf})")
    for tk in tuple(t.strip().upper() for t in watchlist_raw.split(",") if t.strip()):
        w_df = load_quant_data(tk, tf, s_filter, trend_on, long_only)
        if w_df is None: continue
        w_bt = run_bt(w_df, n_bars, tp_m, sl_m, True, use_be, intraday_exit)
        w_fmt = "{:,.0f}원" if tk.endswith((".KS", ".KQ")) else "${:,.2f}"
        w_wr = (w_bt['pnl'] > 0).mean() * 100 if not w_bt.empty else 0.0
        w_pnl = w_bt['pnl'].sum() if not w_bt.empty else 0.0

        if not w_bt.empty and w_bt.iloc[-1]['res'] in ("⏳진행중", "🎯장중익절", "🚨장중손절", "🛡️본절청산"):
            lv = w_bt.iloc[-1]
            badge = "<span class='badge-buy'>🟢 매수</span>" if lv['side'] == "BUY" else "<span class='badge-sell'>🔴 매도</span>"
            rsn = f"{lv['res']} ({lv['pnl']:+.2f}%) · {lv['reason']}"
        else:
            badge, rsn = "<span class='badge-neutral'>관망</span>", f"진행중 시그널 없음 (RSI {w_df['RSI'].iloc[-1]:.1f})"

        st.markdown(f"""
        <div class='list-card'>
          <div class='list-top'>
            <div class='list-left'>{badge}<div class='strat-txt'><b>{tk}</b> · {rsn}</div></div>
            <div class='price-txt'>{w_fmt.format(w_df['Close'].iloc[-1])}</div>
          </div>
          <div class='list-sub'>
            <span>승률: <b>{w_wr:.1f}%</b> ({len(w_bt)}건)</span>
            <span>누적수익: <b>{w_pnl:+.1f}%</b></span>
          </div>
        </div>""", unsafe_allow_html=True)
