import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests, json

st.set_page_config(page_title="US Leverage DayQuant Pro", layout="wide", initial_sidebar_state="collapsed")
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
def load_quant_data(sym, tf, s_filter, trend_on, avoid_lunch=True, long_only=False):
    tf_map = {"15분봉": ("1mo", "15m"), "30분봉": ("1mo", "30m"), "1시간봉": ("3mo", "1h")}
    p, iv = tf_map.get(tf, ("1mo", "15m"))
    raw = yf.download(sym, period=p, interval=iv, progress=False, auto_adjust=True)
    if raw is None or raw.empty: return None
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna().copy()
    if len(df) < 30: return None

    df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index
    ny = df.tz_convert('America/New_York')
    df['Date_NY'], df['Time_NY'] = ny.index.date, ny.index.strftime('%H:%M')

    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
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
    df['VWAP'] = df.groupby('Date_NY')['TP_Vol'].cumsum() / df.groupby('Date_NY')['Volume'].cumsum().replace(0, np.nan)
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['RVOL'] = df['Volume'] / df['Vol_MA20'].replace(0, np.nan)
    df['Body_Ratio'] = (df['Close'] - df['Open']).abs() / (df['High'] - df['Low']).replace(0, np.nan)
    df['Buy_Signal'], df['Sell_Signal'], df['Signal_Reason'] = False, False, ""

    fb = df.groupby('Date_NY').first()
    orb_h, orb_l = fb['High'].to_dict(), fb['Low'].to_dict()
    is_15m = (tf == "15분봉")

    for i in range(2, len(df)):
        idx, r, pv, pv2 = df.index[i], df.iloc[i], df.iloc[i-1], df.iloc[i-2]
        t_ny = r['Time_NY']

        # 장 마감 30분 전(15:30 EST 이후) 신규 진입 금지 & 점심시간(12:00~14:00 EST) 거래량 저조 구간 회피
        if t_ny >= "15:30":
            continue
        if avoid_lunch and ("12:00" <= t_ny < "14:00"):
            continue
        if r['Body_Ratio'] < 0.45 or r['RVOL'] < 0.95:
            continue

        rb, rs = list(), list()
        if r['Close'] > r['VWAP'] and (pv['RSI'] <= (41 if is_15m else 44) or pv2['RSI'] <= (39 if is_15m else 42)) and r['Close'] > r['Open']:
            rb.append("VWAP+RSI 눌림")
        if r['EMA9'] > r['EMA20'] > r['EMA50'] and r['ST_Up'] and (r['Low'] <= r['EMA20']*1.004) and r['Close'] > r['Open']:
            rb.append("EMA 정배열 회귀")
        if r['Close'] > r['VWAP'] and r['MACD_Hist'] > 0 and pv['MACD_Hist'] <= 0 and r['RVOL'] >= 1.25:
            rb.append("MACD+VWAP 반전")
        if (pv['Low'] <= pv['BB_Low'] or r['Low'] <= r['BB_Low']) and r['Close'] > r['BB_Low'] and r['Close'] > r['Open'] and r['RSI'] < 48:
            rb.append("볼린저 하단 반등")
        oh, ol = orb_h.get(r['Date_NY'], np.nan), orb_l.get(r['Date_NY'], np.nan)
        if t_ny > "09:30" and r['Close'] > oh and pv['Close'] <= oh and r['RVOL'] >= (1.5 if is_15m else 1.3):
            rb.append("ORB 돌파")

        v_guard = (r['EMA20'] > pv['EMA20']) or bool(r['ST_Up']) or (r['Close'] > r['VWAP'])
        if not long_only and not v_guard:
            if r['Close'] < r['VWAP'] and (pv['RSI'] >= (64 if is_15m else 61) or pv2['RSI'] >= (67 if is_15m else 64)) and r['Close'] < r['Open']:
                rs.append("VWAP 저항+RSI 과열")
            if r['EMA9'] < r['EMA20'] < r['EMA50'] and (r['High'] >= r['EMA20']*0.996) and r['Close'] < r['Open']:
                rs.append("EMA 역배열 저항")
            if r['Close'] < r['VWAP'] and r['MACD_Hist'] < 0 and pv['MACD_Hist'] >= 0 and r['RVOL'] >= 1.2:
                rs.append("MACD+VWAP 하락반전")
            if (pv['High'] >= pv['BB_Up'] or r['High'] >= r['BB_Up']) and r['Close'] < r['BB_Up'] and r['Close'] < r['Open'] and r['RSI'] > 55:
                rs.append("볼린저 상단 이탈")
            if t_ny > "09:30" and r['Close'] < ol and pv['Close'] >= ol and r['RVOL'] >= (1.5 if is_15m else 1.3):
                rs.append("ORB 하단 이탈")

        if trend_on:
            adx_m = 21.0 if is_15m else 19.0
            ema200_up = r['EMA200'] >= df['EMA200'].iloc[max(0, i-5)]
            if not (r['Close'] > r['EMA200'] and ema200_up and r['ADX'] >= adx_m and r['RSI'] < 69.0):
                rb = list()
            if not (r['Close'] < r['EMA200'] and (not ema200_up) and r['ADX'] >= adx_m and r['RSI'] > 31.0):
                rs = list()

        key = s_filter.split(".").pop().split(" ").pop(0)
        if len(rb) >= 2 and (s_filter.startswith("🏆") or any(key in x for x in rb)):
            df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, "★강력매수: " + ", ".join(rb)
        if len(rs) >= 2 and not df.at[idx, 'Buy_Signal'] and (s_filter.startswith("🏆") or any(key in x for x in rs)):
            df.at[idx, 'Sell_Signal'], df.at[idx, 'Signal_Reason'] = True, "★강력매도: " + ", ".join(rs)
    return df

def run_bt(df_in, n_bars, tp_m, sl_m, eod_close=True, use_be=True, fee_pct=0.05):
    trades, last_i, active_until = list(), len(df_in) - 1, -1
    c_arr, h_arr, l_arr, a_arr = df_in['Close'].to_numpy(float), df_in['High'].to_numpy(float), df_in['Low'].to_numpy(float), df_in['ATR'].to_numpy(float)
    b_arr, s_arr, rsn_arr, idx_arr = df_in['Buy_Signal'].to_numpy(bool), df_in['Sell_Signal'].to_numpy(bool), df_in['Signal_Reason'].to_numpy(str), df_in.index
    d_arr, t_arr = df_in['Date_NY'].to_numpy(), df_in['Time_NY'].to_numpy(str)

    for i in range(len(df_in)):
        if i <= active_until or not (b_arr[i] or s_arr[i]):
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
            # 당일 장 마감 강제청산(오버나이트 갭 방어)
            if eod_close and (d_arr[j] != d_arr[i] or t_arr[j] >= "15:45"):
                target_j = j - 1 if d_arr[j] != d_arr[i] else j
                res, exit_p, exit_idx = "🌙당일마감청산", c_arr[target_j], target_j
                break

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

        active_until = exit_idx
        raw_pnl = ((exit_p - entry) / entry * 100) if side == "BUY" else ((entry - exit_p) / entry * 100)
        net_pnl = raw_pnl - fee_pct
        trades.append({"time": idx_arr[i], "side": side, "reason": rsn_arr[i], "entry": entry, "tp": tp, "sl": sl, "res": res, "pnl": net_pnl})
    return pd.DataFrame(trades)
