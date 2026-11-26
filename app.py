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
        url = "https://kakao.com"
        payload = {
            "object_type": "text",
            "text": msg[:200],
            "link": {"web_url": "https://yahoo.com", "mobile_web_url": "https://yahoo.com"}
        }
        r = requests.post(url, headers={"Authorization": f"Bearer {token}"}, data={"template_object": json.dumps(payload)}, timeout=5)
        return (True, "카톡 전송 완료!") if r.status_code == 200 else (False, f"전송 실패: {r.text}")
    except Exception as e:
        return False, str(e)
@st.cache_data(ttl=60)
def load_quant_data(sym, tf, s_filter, trend_on):
    tf_map = {
        "15분봉": ("1mo", "15m"),
        "30분봉": ("1mo", "30m"),
        "1시간봉": ("3mo", "1h")
    }
    p, iv = tf_map.get(tf, ("3mo", "1h"))

    raw = yf.download(sym, period=p, interval=iv, progress=False, auto_adjust=True)
    if raw is None or raw.empty: return None
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna().copy()
    if len(df) < 30: return None

    is_krx = sym.endswith((".KS", ".KQ"))
    local_tz = 'Asia/Seoul' if is_krx else 'America/New_York'
    open_time = "09:00" if is_krx else "09:30"

    df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index
    loc = df.tz_convert(local_tz)
    df['Date_Loc'], df['Time_Loc'] = loc.index.date, loc.index.strftime('%H:%M')

    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
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
    df['BB_Low'], df['BB_Up'] = bb_m - 2.0 * bb_s, bb_m + 2.0 * bb_s
    df['ST_Up'] = df['Close'] > (df['EMA20'] - 1.5 * df['ATR'])
    df['TP_Vol'] = ((df['High'] + df['Low'] + df['Close']) / 3) * df['Volume']
    df['VWAP'] = df.groupby('Date_Loc')['TP_Vol'].cumsum() / df.groupby('Date_Loc')['Volume'].cumsum().replace(0, np.nan)
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Buy_Signal'], df['Sell_Signal'], df['Signal_Reason'] = False, False, ""

    fb = df.groupby('Date_Loc').first()
    orb_h, orb_l = fb['High'].to_dict(), fb['Low'].to_dict()

    is_15m = (tf == "15분봉")
    rsi_buy_1, rsi_buy_2 = (40, 38) if is_15m else (43, 41)
    rsi_sell_1, rsi_sell_2 = (65, 68) if is_15m else (62, 65)
    vol_orb_b = 1.6 if is_15m else 1.4
    vol_orb_s = 1.5 if is_15m else 1.3
    adx_min = 22.0 if is_15m else 20.0

    for i in range(2, len(df)):
        idx, r, pv, pv2 = df.index[i], df.iloc[i], df.iloc[i-1], df.iloc[i-2]
        rb, rs = list(), list()
        if r['Close'] > r['VWAP'] and (pv['RSI'] <= rsi_buy_1 or pv2['RSI'] <= rsi_buy_2) and r['Close'] > r['Open']:
            rb.append("VWAP+RSI 눌림")
        if r['EMA20'] > r['EMA50'] and r['ST_Up'] and (r['Low'] <= r['EMA20']*1.003) and r['Close'] > r['Open']:
            rb.append("EMA 정배열 회귀")
        if r['Close'] > r['VWAP'] and r['MACD_Hist'] > 0 and pv['MACD_Hist'] <= 0 and r['Volume'] > r['Vol_MA20']*1.2:
            rb.append("MACD+VWAP 반전")
        if (pv['Low'] <= pv['BB_Low'] or r['Low'] <= r['BB_Low']) and r['Close'] > r['BB_Low'] and r['Close'] > r['Open'] and r['RSI'] < 48:
            rb.append("볼린저 하단 반등")
        oh, ol = orb_h.get(r['Date_Loc'], np.nan), orb_l.get(r['Date_Loc'], np.nan)
        if r['Time_Loc'] > open_time and r['Close'] > oh and pv['Close'] <= oh and r['Volume'] > r['Vol_MA20']*vol_orb_b:
            rb.append("ORB 돌파")

        if r['Close'] < r['VWAP'] and (pv['RSI'] >= rsi_sell_1 or pv2['RSI'] >= rsi_sell_2) and r['Close'] < r['Open']:
            rs.append("VWAP 저항+RSI 과열")
        if r['EMA20'] < r['EMA50'] and (r['High'] >= r['EMA20']*0.997) and r['Close'] < r['Open']:
            rs.append("EMA 역배열 저항")
        if r['Close'] < r['VWAP'] and r['MACD_Hist'] < 0 and pv['MACD_Hist'] >= 0:
            rs.append("MACD+VWAP 하락반전")
        if (pv['High'] >= pv['BB_Up'] or r['High'] >= r['BB_Up']) and r['Close'] < r['BB_Up'] and r['Close'] < r['Open'] and r['RSI'] > 55:
            rs.append("볼린저 상단 이탈")
        if r['Time_Loc'] > open_time and r['Close'] < ol and pv['Close'] >= ol and r['Volume'] > r['Vol_MA20']*vol_orb_s:
            rs.append("ORB 하단 이탈")

        if trend_on:
            if not (r['Close'] > r['EMA200'] and r['ADX'] >= adx_min and r['RSI'] < 68.0): rb = list()
            if not (r['Close'] < r['EMA200'] and r['ADX'] >= adx_min and r['RSI'] > 32.0): rs = list()

        if s_filter.startswith("🏆"):
            if rb:
                tag = "★강력매수: " if len(rb) >= 2 else ""
                df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, tag + ", ".join(rb)
            if rs and not df.at[idx, 'Buy_Signal']:
                tag = "★강력매도: " if len(rs) >= 2 else ""
                df.at[idx, 'Sell_Signal'], df.at[idx, 'Signal_Reason'] = True, tag + ", ".join(rs)
        else:
            key = s_filter.split(".").pop().strip()
            mb = tuple(x for x in rb if key in x)
            ms = tuple(x for x in rs if key in x)
            if mb: df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, ", ".join(mb)
            if ms and not mb: df.at[idx, 'Sell_Signal'], df.at[idx, 'Signal_Reason'] = True, ", ".join(ms)
    return df

def run_bt(df_in, n_bars, tp_m, sl_m):
    trades = list()
    c_arr, h_arr, l_arr, a_arr = df_in['Close'].to_numpy(float), df_in['High'].to_numpy(float), df_in['Low'].to_numpy(float), df_in['ATR'].to_numpy(float)
    b_arr, s_arr = df_in['Buy_Signal'].to_numpy(bool), df_in['Sell_Signal'].to_numpy(bool)
    rsn_arr, idx_arr = df_in['Signal_Reason'].to_numpy(str), df_in.index

    for i in range(len(df_in) - 1):
        if not (b_arr[i] or s_arr[i]): continue
        side = "BUY" if b_arr[i] else "SELL"
        entry, atr = c_arr[i], a_arr[i]
        tp = entry + tp_m * atr if side == "BUY" else entry - tp_m * atr
        sl = entry - sl_m * atr if side == "BUY" else entry + sl_m * atr
        res, exit_p = "보유만기", c_arr[min(i + n_bars, len(df_in) - 1)]

        for j in range(i + 1, min(i + 1 + n_bars, len(df_in))):
            if side == "BUY":
                if l_arr[j] <= sl: res, exit_p = "손절", sl; break
                if h_arr[j] >= tp: res, exit_p = "익절", tp; break
            else:
                if h_arr[j] >= sl: res, exit_p = "손절", sl; break
                if l_arr[j] <= tp: res, exit_p = "익절", tp; break

        pnl = ((exit_p - entry) / entry * 100) if side == "BUY" else ((entry - exit_p) / entry * 100)
        trades.append({"time": idx_arr[i], "side": side, "reason": rsn_arr[i], "entry": entry, "exit": exit_p, "res": res, "pnl": pnl})
    return pd.DataFrame(trades)
st.markdown("#### ⚡ Global Quant Pro")

with st.expander("⚙️ 종목 · 분봉 · 전략 설정", expanded=True):
    c1, c2 = st.columns((1.2, 1.8))
    sym = c1.text_input("티커 (예: NVDA, 005930.KS)", value="NVDA").upper().strip()
    tf = c2.radio("타임프레임", ("15분봉", "30분봉", "1시간봉"), index=0, horizontal=True)

    s_filter = st.selectbox("전략 필터", (
        "🏆 전체 복합 전략 (강력신호 우선)",
        "1. VWAP+RSI",
        "2. EMA",
        "3. MACD+VWAP",
        "4. 볼린저",
        "5. ORB"
    ))
    c3, c4 = st.columns(2)
    trend_on = c3.checkbox("EMA200 + ADX 추세필터", value=True)
    auto_opt = c4.checkbox("승률 최적화 자동 탐색", value=True)

    default_bars = 16 if tf == "15분봉" else (12 if tf == "30분봉" else 8)
    b1, b2, b3 = st.columns(3)
    n_bars = b1.number_input("최대보유(봉)", 3, 48, default_bars)
    tp_m = b2.number_input("익절(ATR배수)", 0.5, 5.0, 2.0, 0.5)
    sl_m = b3.number_input("손절(ATR배수)", 0.5, 3.0, 1.0, 0.5)

    if auto_opt:
        st.caption("🔧 자동 탐색 후보군 커스터마이징")
        o1, o2, o3 = st.columns(3)
        default_nb = (8, 12, 16, 24) if tf == "15분봉" else (6, 8, 12, 16)
        nb_opts = o1.multiselect("보유봉 후보", (6, 8, 12, 16, 20, 24), default=default_nb)
        tp_opts = o2.multiselect("익절 배수 후보", (1.0, 1.5, 2.0, 2.5, 3.0), default=(1.5, 2.0, 2.5))
        sl_opts = o3.multiselect("손절 배수 후보", (0.8, 1.0, 1.2, 1.5), default=(1.0, 1.5))

    kakao_token = st.text_input("카카오톡 Access Token (선택)", type="password")

df = load_quant_data(sym, tf, s_filter, trend_on)
if df is None:
    st.error("데이터를 불러올 수 없습니다. 티커명(한국 주식은 .KS 또는 .KQ 포함)을 확인해 주세요.")
    st.stop()

is_krx = sym.endswith((".KS", ".KQ"))
cur_fmt = "{:,.0f}원" if is_krx else "${:,.2f}"

if auto_opt and nb_opts and tp_opts and sl_opts:
    best_wr, best_cfg = -1.0, (n_bars, tp_m, sl_m)
    for nb in nb_opts:
        for tm in tp_opts:
            for sm in sl_opts:
                tmp = run_bt(df, nb, tm, sm)
                if len(tmp) >= 5:
                    wr = (tmp['pnl'] > 0).mean() * 100
                    if wr > best_wr:
                        best_wr, best_cfg = wr, (nb, tm, sm)
    n_bars, tp_m, sl_m = best_cfg
    if best_wr >= 0:
        st.markdown(f"<div class='opt-box'>🎯 <b>{tf} 자동 최적화 완료:</b> 보유 {n_bars}봉 | 익절 {tp_m} ATR | 손절 {sl_m} ATR (최고 승률 {best_wr:.1f}%)</div>", unsafe_allow_html=True)
    else:
        st.warning("유효 표본(5건 이상)이 부족해 기본 설정값을 유지합니다.")

bt = run_bt(df, n_bars, tp_m, sl_m)
last = df.iloc[-1]
win_rate = (bt['pnl'] > 0).mean() * 100 if not bt.empty else 0.0
tot_pnl = bt['pnl'].sum() if not bt.empty else 0.0

st.markdown(f"""
<div class='kpi-grid'>
  <div class='metric-card'><div class='metric-title'>{sym} 현재가 ({tf})</div><div class='metric-value'>{cur_fmt.format(last['Close'])}</div></div>
  <div class='metric-card'><div class='metric-title'>백테스트 승률 ({len(bt)}건)</div><div class='metric-value'>{win_rate:.1f}% <span style='font-size:.75rem;color:#94A3B8'>(누적 {tot_pnl:+.1f}%)</span></div></div>
</div>
""", unsafe_allow_html=True)

tp_val = last['Close'] + tp_m * last['ATR']
sl_val = last['Close'] - sl_m * last['ATR']
st.markdown(f"""
<div class='tpsl-bar'>
  <div><div class='ohlc-lbl'>기준 진입가</div><div class='ohlc-val'>{cur_fmt.format(last['Close'])}</div></div>
  <div><div class='ohlc-lbl'>목표 익절가 (+{tp_m} ATR)</div><div class='ohlc-val' style='color:#10B981'>{cur_fmt.format(tp_val)}</div></div>
  <div><div class='ohlc-lbl'>기준 손절가 (-{sl_m} ATR)</div><div class='ohlc-val' style='color:#EF4444'>{cur_fmt.format(sl_val)}</div></div>
</div>
""", unsafe_allow_html=True)

plot_df = df.tail(100)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=(0.75, 0.25), vertical_spacing=0.03)
fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="Price"), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA20'], line=dict(color='#38BDF8', width=1.2), name="EMA20"), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['VWAP'], line=dict(color='#F59E0B', width=1.2, dash='dot'), name="VWAP"), row=1, col=1)

buys = plot_df.loc[plot_df['Buy_Signal']]
sells = plot_df.loc[plot_df['Sell_Signal']]
fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.996, mode='markers', marker=dict(symbol='triangle-up', size=11, color='#10B981'), name="매수신호"), row=1, col=1)
fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.004, mode='markers', marker=dict(symbol='triangle-down', size=11, color='#EF4444'), name="매도신호"), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], line=dict(color='#A855F7', width=1.2), name="RSI"), row=2, col=1)
fig.update_layout(height=420, margin=dict(l=5, r=5, t=10, b=10), paper_bgcolor='#0B0E14', plot_bgcolor='#131722', font=dict(color='#E2E8F0'), xaxis_rangeslider_visible=False, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown("##### 📌 최근 발생 시그널 내역")
if bt.empty:
    st.info("선택한 조건에서 발생한 최근 시그널이 없습니다.")
else:
    recent = bt.iloc[::-1].head(10)
    for _, row in recent.iterrows():
        badge = "<span class='badge-buy'>BUY</span>" if row['side'] == "BUY" else "<span class='badge-sell'>SELL</span>"
        p_color = "#10B981" if row['pnl'] > 0 else "#EF4444"
        t_str = row['time'].tz_convert('Asia/Seoul').strftime('%m/%d %H:%M KST')
        st.markdown(f"""
        <div class='list-card'>
          <div class='list-top'>
            <div class='list-left'>{badge}<div class='strat-txt'>{row['reason']}</div></div>
            <div class='price-txt'>{cur_fmt.format(row['entry'])}</div>
          </div>
          <div class='list-sub'>
            <span>{t_str}</span>
            <span>결과: {row['res']} (<b style='color:{p_color}'>{row['pnl']:+.2f}%</b>)</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("📲 최신 시그널 카카오톡 전송"):
        latest_sig = recent.head(1).squeeze()
        msg = f"[{sym} {tf} 시그널]\n유형: {latest_sig['side']}\n전략: {latest_sig['reason']}\n진입가: {cur_fmt.format(latest_sig['entry'])}"
        ok, info = send_kakao(kakao_token, msg)
        st.success(info) if ok else st.warning(info)
