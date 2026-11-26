import os, requests
import yfinance as yf
import pandas as pd
import numpy as np

WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")
WATCHLIST = ["TQQQ", "SOXL", "NVDL", "TSLL", "NVDA", "QQQ"]
TIMEFRAMES = [
    ("30분봉", "1mo", "30m"),
    ("1시간봉", "3mo", "1h")
]

def check_symbol(sym, tf_name, p, iv):
    raw = yf.download(sym, period=p, interval=iv, progress=False, auto_adjust=True)
    if raw is None or raw.empty: return None
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna().copy()
    if len(df) < 30: return None

    df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index
    loc = df.tz_convert('America/New_York')
    df['Date_Loc'], df['Time_Loc'] = loc.index.date, loc.index.strftime('%H:%M')

    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    d = df['Close'].diff()
    df['RSI'] = 100 - (100 / (1 + d.where(d > 0, 0).rolling(14).mean() / (-d.where(d < 0, 0)).rolling(14).mean().replace(0, np.nan)))
    df['TR'] = pd.concat((df['High'] - df['Low'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()), axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean().bfill()
    macd = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = macd - macd.ewm(span=9, adjust=False).mean()
    bb_m, bb_s = df['Close'].rolling(20).mean(), df['Close'].rolling(20).std()
    df['BB_Low'], df['BB_Up'] = bb_m - 1.95 * bb_s, bb_m + 1.95 * bb_s
    df['TP_Vol'] = ((df['High'] + df['Low'] + df['Close']) / 3) * df['Volume']
    df['VWAP'] = df.groupby('Date_Loc')['TP_Vol'].cumsum() / df.groupby('Date_Loc')['Volume'].cumsum().replace(0, np.nan)
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()

    r, pv, pv2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    rb, rs = [], []

    if r['Close'] > r['VWAP'] and (pv['RSI'] <= 47 or pv2['RSI'] <= 45) and r['Close'] > r['Open']: rb.append("VWAP+RSI 눌림")
    if r['EMA20'] > r['EMA50'] and (r['Low'] <= r['EMA20']*1.005) and r['Close'] > r['Open']: rb.append("EMA 정배열 회귀")
    if r['Close'] > r['VWAP'] and r['MACD_Hist'] > 0 and pv['MACD_Hist'] <= 0 and r['Volume'] > r['Vol_MA20']*1.1: rb.append("MACD+VWAP 반전")
    if (pv['Low'] <= pv['BB_Low'] or r['Low'] <= r['BB_Low']) and r['Close'] > r['BB_Low'] and r['Close'] > r['Open'] and r['RSI'] < 50: rb.append("볼린저 하단 반등")

    v_guard = (r['Close'] > r['VWAP']) and (r['EMA20'] > pv['EMA20'])
    if not v_guard:
        if r['Close'] < r['VWAP'] and (pv['RSI'] >= 58 or pv2['RSI'] >= 60) and r['Close'] < r['Open']: rs.append("VWAP 저항+RSI 과열")
        if r['EMA20'] < r['EMA50'] and (r['High'] >= r['EMA20']*0.995) and r['Close'] < r['Open']: rs.append("EMA 역배열 저항")
        if r['Close'] < r['VWAP'] and r['MACD_Hist'] < 0 and pv['MACD_Hist'] >= 0: rs.append("MACD+VWAP 하락반전")
        if (pv['High'] >= pv['BB_Up'] or r['High'] >= pv['BB_Up']) and r['Close'] < r['BB_Up'] and r['Close'] < r['Open'] and r['RSI'] > 52: rs.append("볼린저 상단 이탈")

    if rb and r['Close'] > r['EMA100'] and r['RSI'] < 70:
        tp, sl = r['Close'] + 2.0 * r['ATR'], r['Close'] - 1.0 * r['ATR']
        tag = "★강력매수: " if len(rb) >= 2 else ""
        return f"🟢 **[{sym} {tf_name} 매수 시그널]**\n• 전략: {tag}{', '.join(rb)}\n• 진입가: `${r['Close']:.2f}` | 목표: `${tp:.2f}` | 손절: `${sl:.2f}`"
    if rs and r['Close'] < r['EMA100'] and r['RSI'] > 30:
        tp, sl = r['Close'] - 2.0 * r['ATR'], r['Close'] + 1.0 * r['ATR']
        tag = "★강력매도: " if len(rs) >= 2 else ""
        return f"🔴 **[{sym} {tf_name} 매도 시그널]**\n• 전략: {tag}{', '.join(rs)}\n• 진입가: `${r['Close']:.2f}` | 목표: `${tp:.2f}` | 손절: `${sl:.2f}`"
    return None

if __name__ == "__main__" and WEBHOOK:
    msgs = []
    for sym in WATCHLIST:
        for tf_name, p, iv in TIMEFRAMES:
            m = check_symbol(sym, tf_name, p, iv)
            if m: msgs.append(m)
    if msgs:
        requests.post(WEBHOOK, json={"content": "\n\n".join(msgs)[:1950]}, timeout=10)
