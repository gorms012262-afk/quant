import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests, json

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
.alert-tp{background:rgba(16,185,129,.22);border:1.5px solid #10B981;border-radius:10px;padding:12px;margin:8px 0;color:#ECFDF5;font-size:.86rem;line-height:1.45}
.alert-sl{background:rgba(239,68,68,.22);border:1.5px solid #EF4444;border-radius:10px;padding:12px;margin:8px 0;color:#FEF2F2;font-size:.86rem;line-height:1.45}
.alert-run{background:rgba(56,189,248,.14);border:1.5px solid #38BDF8;border-radius:10px;padding:12px;margin:8px 0;color:#F0F9FF;font-size:.86rem;line-height:1.45}
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
        payload = {"object_type": "text", "text": msg[:200], "link": {"web_url": "https://yahoo.com", "mobile_web_url": "https://yahoo.com"}}
        r = requests.post(url, headers={"Authorization": f"Bearer {token}"}, data={"template_object": json.dumps(payload)}, timeout=5)
        return (True, "카톡 전송 완료!") if r.status_code == 200 else (False, f"전송 실패: {r.text}")
    except Exception as e:
        return False, str(e)
@st.cache_data(ttl=15)
def load_quant_data(sym, tf, s_filter, trend_on, long_only=False):
    tf_map = {"15분봉": ("1mo", "15m"), "30분봉": ("1mo", "30m"), "1시간봉": ("3mo", "1h")}
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

        v_rebound_guard = (r['EMA20'] > pv['EMA20']) or bool(r['ST_Up']) or (r['Close'] > r['VWAP'])
        if not long_only and not v_rebound_guard:
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
            if len(rb) >= 2:
                df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, "★강력매수: " + ", ".join(rb)
            if len(rs) >= 2 and not df.at[idx, 'Buy_Signal']:
                df.at[idx, 'Sell_Signal'], df.at[idx, 'Signal_Reason'] = True, "★강력매도: " + ", ".join(rs)
        else:
            key = s_filter.split(".").pop().strip()
            if len(rb) >= 2 and any(key in x for x in rb):
                df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, "★강력매수: " + ", ".join(rb)
            if len(rs) >= 2 and any(key in x for x in rs) and not df.at[idx, 'Buy_Signal']:
                df.at[idx, 'Sell_Signal'], df.at[idx, 'Signal_Reason'] = True, "★강력매도: " + ", ".join(rs)
    return df
def run_bt(df_in, n_bars, tp_m, sl_m, no_overlap=True):
    trades = list()
    c_arr, h_arr, l_arr, a_arr = df_in['Close'].to_numpy(float), df_in['High'].to_numpy(float), df_in['Low'].to_numpy(float), df_in['ATR'].to_numpy(float)
    b_arr, s_arr = df_in['Buy_Signal'].to_numpy(bool), df_in['Sell_Signal'].to_numpy(bool)
    rsn_arr, idx_arr = df_in['Signal_Reason'].to_numpy(str), df_in.index
    last_i = len(df_in) - 1

    active_until = -1
    for i in range(len(df_in)):
        if no_overlap and i <= active_until:
            continue
        if not (b_arr[i] or s_arr[i]):
            continue

        side = "BUY" if b_arr[i] else "SELL"
        entry, atr = c_arr[i], a_arr[i]
        tp = entry + tp_m * atr if side == "BUY" else entry - tp_m * atr
        sl = entry - sl_m * atr if side == "BUY" else entry + sl_m * atr

        is_still_open = (i + n_bars > last_i)
        exit_idx = min(i + n_bars, last_i)
        res = "진행중" if is_still_open else "보유만기"
        exit_p = c_arr[exit_idx]

        for j in range(i + 1, min(i + 1 + n_bars, len(df_in))):
            is_live_bar = (j == last_i)
            if side == "BUY":
                if l_arr[j] <= sl:
                    res = "🚨장중손절" if is_live_bar else "손절"
                    exit_p, exit_idx = sl, j
                    break
                if h_arr[j] >= tp:
                    res = "🎯장중익절" if is_live_bar else "익절"
                    exit_p, exit_idx = tp, j
                    break
            else:
                if h_arr[j] >= sl:
                    res = "🚨장중손절" if is_live_bar else "손절"
                    exit_p, exit_idx = sl, j
                    break
                if l_arr[j] <= tp:
                    res = "🎯장중익절" if is_live_bar else "익절"
                    exit_p, exit_idx = tp, j
                    break

        active_until = exit_idx
        pnl = ((exit_p - entry) / entry * 100) if side == "BUY" else ((entry - exit_p) / entry * 100)
        trades.append({
            "time": idx_arr[i], "side": side, "reason": rsn_arr[i],
            "entry": entry, "exit": exit_p, "tp": tp, "sl": sl, "res": res, "pnl": pnl
        })
    return pd.DataFrame(trades)

def check_live_tpsl(df_in, n_bars, tp_m, sl_m):
    sigs = df_in.loc[df_in['Buy_Signal'] | df_in['Sell_Signal']]
    if sigs.empty:
        return None
    last_sig_idx = sigs.index[-1]
    sig_pos = df_in.index.get_loc(last_sig_idx)
    if isinstance(sig_pos, slice):
        sig_pos = sig_pos.start

    elapsed = len(df_in) - 1 - sig_pos
    if elapsed > n_bars:
        return None

    sig_row = df_in.iloc[sig_pos]
    side = "BUY" if bool(sig_row['Buy_Signal']) else "SELL"
    entry, atr = float(sig_row['Close']), float(sig_row['ATR'])
    tp = entry + tp_m * atr if side == "BUY" else entry - tp_m * atr
    sl = entry - sl_m * atr if side == "BUY" else entry + sl_m * atr

    sub = df_in.iloc[sig_pos:]
    cur_p = float(df_in['Close'].iloc[-1])

    if len(sub) == 1:
        hit_tp = (cur_p >= tp) if side == "BUY" else (cur_p <= tp)
        hit_sl = (cur_p <= sl) if side == "BUY" else (cur_p >= sl)
    else:
        after = df_in.iloc[sig_pos + 1:]
        hit_tp = (after['High'].max() >= tp) if side == "BUY" else (after['Low'].min() <= tp)
        hit_sl = (after['Low'].min() <= sl) if side == "BUY" else (after['High'].max() >= sl)

    cur_pnl = ((cur_p - entry) / entry * 100) if side == "BUY" else ((entry - cur_p) / entry * 100)
    status = "TP" if hit_tp else ("SL" if hit_sl else "ACTIVE")
    return {
        "status": status, "side": side, "reason": sig_row['Signal_Reason'],
        "time": last_sig_idx, "entry": entry, "tp": tp, "sl": sl,
        "cur_p": cur_p, "cur_pnl": cur_pnl, "elapsed": elapsed, "n_bars": n_bars
    }

def calc_bt_metrics(bt_df):
    if bt_df.empty:
        return 0.0, 0.0, 0.0, 0.0
    win_rate = (bt_df['pnl'] > 0).mean() * 100
    tot_pnl = bt_df['pnl'].sum()
    gross_win = bt_df.loc[bt_df['pnl'] > 0, 'pnl'].sum()
    gross_loss = abs(bt_df.loc[bt_df['pnl'] < 0, 'pnl'].sum())
    pf = (gross_win / gross_loss) if gross_loss > 0 else (99.9 if gross_win > 0 else 0.0)
    cum_curve = (1 + bt_df['pnl'] / 100).cumprod()
    peak = cum_curve.cummax()
    mdd = ((cum_curve - peak) / peak * 100).min()
    return win_rate, tot_pnl, pf, mdd
if "sent_alerts" not in st.session_state:
    st.session_state["sent_alerts"] = set()

top_l, top_r = st.columns((2.2, 1.0))
top_l.markdown("#### ⚡ Global Quant Pro")
if top_r.button("🔄 실시간 새로고침", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

with st.expander("⚙️ 종목 · 분봉 · 전략 설정", expanded=True):
    c1, c2 = st.columns((1.2, 1.8))
    sym = c1.text_input("메인 분석 티커", value="NVDA").upper().strip()
    tf = c2.radio("타임프레임", ("15분봉", "30분봉", "1시간봉"), index=0, horizontal=True)
    watchlist_raw = st.text_input("📋 워치리스트 (쉼표 구분)", value="NVDA, TSLA, AAPL, QQQ, 005930.KS")
    s_filter = st.selectbox("강력신호 포함 전략 필터", (
        "🏆 전체 강력신호 (2개 이상 전략 중첩)",
        "1. VWAP+RSI 포함 강력신호",
        "2. EMA 포함 강력신호",
        "3. MACD+VWAP 포함 강력신호",
        "4. 볼린저 포함 강력신호",
        "5. ORB 포함 강력신호"
    ))
    c3, c4 = st.columns(2)
    trend_on = c3.checkbox("EMA200 + ADX 추세필터", value=True)
    auto_opt = c4.checkbox("승률 최적화 자동 탐색", value=True)
    c5, c6 = st.columns(2)
    long_only = c5.checkbox("현물 전용 (★강력매수만 보기)", value=False)
    no_overlap = c6.checkbox("포지션 중 중복신호 차단", value=True)

    default_bars = 16 if tf == "15분봉" else (12 if tf == "30분봉" else 8)
    b1, b2, b3 = st.columns(3)
    n_bars = b1.number_input("최대보유(봉)", 3, 48, default_bars)
    tp_m = b2.number_input("익절(ATR배수)", 0.5, 5.0, 2.0, 0.5)
    sl_m = b3.number_input("손절(ATR배수)", 0.5, 3.0, 1.0, 0.5)

    if auto_opt:
        o1, o2, o3 = st.columns(3)
        default_nb = (8, 12, 16, 24) if tf == "15분봉" else (6, 8, 12, 16)
        nb_opts = o1.multiselect("보유봉 후보", (6, 8, 12, 16, 20, 24), default=default_nb)
        tp_opts = o2.multiselect("익절 배수 후보", (1.0, 1.5, 2.0, 2.5, 3.0), default=(1.5, 2.0, 2.5))
        sl_opts = o3.multiselect("손절 배수 후보", (0.8, 1.0, 1.2, 1.5), default=(1.0, 1.5))

    k1, k2 = st.columns((1.8, 1.2))
    kakao_token = k1.text_input("카카오톡 Access Token", type="password")
    auto_kakao = k2.checkbox("장중 익절·손절 즉시 카톡발송", value=False)

tab1, tab2 = st.tabs(("📊 메인 종목 정밀 분석", "🔍 워치리스트 일괄 스캔"))
