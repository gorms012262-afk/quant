import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json

st.set_page_config(page_title="US Stock Quant Pro - Auto Optimizer", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""<style>
.stApp { background-color:#0B0E14; color:#E2E8F0; }
header[data-testid="stHeader"] { background-color:#0B0E14; }
.block-container { padding-top:2.8rem!important; padding-bottom:2rem!important; padding-left:0.8rem!important; padding-right:0.8rem!important; }
.kpi-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:6px; margin:6px 0 8px 0; }
.metric-card { background-color:#131722; border:1px solid #1E293B; border-radius:8px; padding:9px 10px; }
.metric-title { font-size:0.72rem; color:#94A3B8; margin-bottom:2px; }
.metric-value { font-size:1.05rem; font-weight:700; color:#F8FAFC; }
.badge-buy { background-color:rgba(16,185,129,0.15); color:#10B981; border:1px solid #10B981; padding:2px 6px; border-radius:5px; font-size:0.72rem; font-weight:600; display:inline-block; }
.badge-sell { background-color:rgba(239,68,68,0.15); color:#EF4444; border:1px solid #EF4444; padding:2px 6px; border-radius:5px; font-size:0.72rem; font-weight:600; display:inline-block; }
.badge-neutral { background-color:rgba(148,163,184,0.15); color:#94A3B8; border:1px solid #475569; padding:2px 6px; border-radius:5px; font-size:0.72rem; font-weight:600; display:inline-block; }
.ohlc-lbl { font-size:0.68rem; color:#64748B; margin-bottom:1px; }
.ohlc-val { font-size:0.82rem; font-weight:700; color:#F8FAFC; }
.tpsl-bar { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; background-color:#131722; border:1px solid #334155; border-radius:8px; padding:8px; margin-bottom:8px; text-align:center; }
.opt-box { background-color:#131722; border:1px solid #10B981; border-radius:8px; padding:10px; margin:6px 0 10px 0; }
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

with st.expander("⚙️ 종목 / 고승률 전략 / ATR 리스크 / 카톡 설정", expanded=False):
    ticker = st.text_input("메인 분석 티커 (예: QQQ, NVDA, TSLA)", value="QQQ").upper().strip()
    timeframe = st.radio("봉 기준", ["30분봉", "1시간봉", "일봉"], horizontal=True)
    strategy_filter = st.selectbox(
        "전략 필터",
        [
            "전체 전략 통합 (최고승률 엄선)",
            "VWAP+RSI 눌림",
            "EMA 정배열 회귀",
            "MACD+VWAP 반전",
            "볼린저 하단 반등",
            "ORB 돌파",
            "갭 채우기 이탈"
        ]
    )
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        hold_bars = st.slider("최대 보유봉", 2, 24, 8)
    with col_s2:
        sl_mult = st.number_input("손절 ATR 배수", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    with col_s3:
        tp_mult = st.number_input("익절 ATR 배수", min_value=1.0, max_value=10.0, value=2.5, step=0.5)

    use_adx_filter = st.checkbox("추세 강도(ADX>20) & EMA200 상승추세 필터 적용 (승률 극대화)", value=True)
    watchlist_str = st.text_input("스캐너 관심종목 (쉼표 구분)", value="QQQ, SPY, NVDA, TSLA, AAPL, MSFT, AMD, META")
    st.markdown("---")
    kakao_token = st.text_input("카카오톡 나에게 보내기 Access Token (선택)", type="password", placeholder="Kakao REST API Access Token")

    if st.button("🔄 데이터 새로고침", use_container_width=True):
        st.cache_data.clear()

def send_kakao_alert(access_token, msg):
    if not access_token:
        return False, "카카오톡 액세스 토큰이 입력되지 않았습니다."
    url = "https://kakao.com"
    headers = {"Authorization": f"Bearer {access_token}"}
    template = {
        "object_type": "text",
        "text": msg[:200],
        "link": {"web_url": "https://yahoo.com", "mobile_web_url": "https://yahoo.com"},
        "button_title": "차트 확인"
    }
    try:
        res = requests.post(url, headers=headers, data={"template_object": json.dumps(template)}, timeout=5)
        if res.status_code == 200:
            return True, "카카오톡 나에게 보내기 성공!"
        return False, f"카카오톡 전송 실패: {res.text}"
    except Exception as e:
        return False, f"오류 발생: {e}"
@st.cache_data(ttl=60)
def load_single(symbol, tf, strat_filter, adx_on):
    p, iv = ("1mo", "30m") if tf == "30분봉" else (("3mo", "1h") if tf == "1시간봉" else ("2y", "1d"))
    raw = yf.download(symbol, period=p, interval=iv, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna().copy()
    if len(df) < 25:
        return None

    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df_ny = df.tz_convert('America/New_York')
    df['Date_NY'], df['Time_NY'] = df_ny.index.date, df_ny.index.strftime('%H:%M')

    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    d = df['Close'].diff()
    g, l = d.where(d > 0, 0).rolling(14).mean(), (-d.where(d < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + g / l.replace(0, np.nan)))

    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)

    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean().bfill()

    atr_smooth = df['TR'].ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_smooth)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Sig'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Sig']

    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Low'] = bb_mid - (2.0 * bb_std)
    df['BB_Up'] = bb_mid + (2.0 * bb_std)

    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Vol'] = tp * df['Volume']
    df['VWAP'] = (
        (df.groupby('Date_NY')['TP_Vol'].cumsum() / df.groupby('Date_NY')['Volume'].cumsum())
        if tf in ["30분봉", "1시간봉"]
        else (df['TP_Vol'].rolling(20).sum() / df['Volume'].rolling(20).sum())
    )
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Buy_Signal'], df['Sell_Signal'], df['Signal_Reason'] = False, False, ""

    is_intraday = tf in ["30분봉", "1시간봉"]
    if is_intraday:
        fb, pc = df.groupby('Date_NY').first(), df.groupby('Date_NY')['Close'].last().shift(1)
        orb_h, d_open, f_close, p_close = fb['High'].to_dict(), fb['Open'].to_dict(), fb['Close'].to_dict(), pc.to_dict()

    for i in range(2, len(df)):
        idx, r, pv, pv2 = df.index[i], df.iloc[i], df.iloc[i - 1], df.iloc[i - 2]
        rb, rs = [], []

        if r['Close'] > r['VWAP'] and (pv['RSI'] <= 42 or pv2['RSI'] <= 40) and r['Close'] > r['Open']:
            rb.append("VWAP+RSI 눌림")

        if (r['EMA20'] > r['EMA50']) and ((r['Low'] <= r['EMA20'] * 1.003) or (r['Low'] <= r['EMA50'] * 1.003)) and (r['Close'] > r['Open']) and ((min(r['Open'], r['Close']) - r['Low']) >= abs(r['Close'] - r['Open']) * 0.4):
            rb.append("EMA 정배열 회귀")

        if (r['Close'] > r['VWAP']) and (r['MACD_Hist'] > 0) and (pv['MACD_Hist'] <= 0) and (r['Close'] > r['Open']):
            rb.append("MACD+VWAP 반전")

        if (pv['Low'] <= pv['BB_Low'] or r['Low'] <= r['BB_Low']) and (r['Close'] > r['BB_Low']) and (r['Close'] > r['Open']) and (r['RSI'] < 52):
            rb.append("볼린저 하단 반등")

        if is_intraday:
            dt, tm = r['Date_NY'], r['Time_NY']
            oh, do, pcv = orb_h.get(dt, np.nan), d_open.get(dt, np.nan), p_close.get(dt, np.nan)
            if tm > "09:30" and r['Close'] > oh and pv['Close'] <= oh and r['Volume'] > r['Vol_MA20'] * 1.4:
                rb.append("ORB 돌파")
            if pd.notna(pcv) and pcv > 0 and (0 < (do - pcv) / pcv <= 0.012) and (f_close.get(dt, np.nan) < do) and (r['Close'] < r['VWAP']) and (pv['Close'] >= pv['VWAP']):
                rs.append("갭 채우기 이탈")
        else:
            if (r['Open'] > pv['Close']) and (r['Close'] < pv['Close']) and (r['Close'] < r['VWAP']):
                rs.append("갭 채우기 이탈")

        if adx_on:
            trend_ok = (r['Close'] > r['EMA200']) and (r['ADX'] >= 20.0) and (r['RSI'] < 68.0)
        else:
            trend_ok = (r['Close'] > r['EMA50'] or r['EMA20'] > r['EMA50']) and (r['RSI'] < 70.0)

        if not trend_ok:
            rb = []
        if not (r['Close'] < r['EMA20'] and r['RSI'] > 30.0):
            rs = []

        if strat_filter != "전체 전략 통합 (최고승률 엄선)":
            rb = [x for x in rb if strat_filter in x]
            rs = [x for x in rs if strat_filter in x]
            if rb:
                df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, "매수: " + ", ".join(rb)
            if rs:
                df.at[idx, 'Sell_Signal'], df.at[idx, 'Signal_Reason'] = True, "매도: " + ", ".join(rs)
        else:
            if rb and (len(rb) >= 2 or r['Volume'] > r['Vol_MA20'] * 1.5):
                df.at[idx, 'Buy_Signal'], df.at[idx, 'Signal_Reason'] = True, "★최고승률 매수: " + ", ".join(rb)
            if rs:
                df.at[idx, 'Sell_Signal'], df.at[idx, 'Signal_Reason'] = True, "매도: " + ", ".join(rs)
    return df
def calc_backtest_metrics(df_in, n_bars, tp_m, sl_m):
    buy_indices = np.flatnonzero(df_in['Buy_Signal'].to_numpy())
    trades = []
    close_arr = df_in['Close'].to_numpy(dtype=float)
    high_arr = df_in['High'].to_numpy(dtype=float)
    low_arr = df_in['Low'].to_numpy(dtype=float)
    atr_arr = df_in['ATR'].to_numpy(dtype=float)
    idx_arr = df_in.index
    total_len = len(df_in)

    for pos in buy_indices:
        pos_int = int(pos)
        if pos_int + 1 >= total_len:
            continue

        entry_p = close_arr.item(pos_int)
        atr_v = atr_arr.item(pos_int)
        tp_target = entry_p + (atr_v * tp_m)
        sl_target = entry_p - (atr_v * sl_m)

        end_pos = min(pos_int + n_bars, total_len - 1)
        exit_p = close_arr.item(end_pos)
        exit_dt = idx_arr[end_pos]

        for step in range(pos_int + 1, end_pos + 1):
            bar_low = low_arr.item(step)
            bar_high = high_arr.item(step)
            if bar_low <= sl_target:
                exit_p = sl_target
                exit_dt = idx_arr[step]
                break
            if bar_high >= tp_target:
                exit_p = tp_target
                exit_dt = idx_arr[step]
                break

        ret = (exit_p - entry_p) / entry_p
        trades.append({"date": exit_dt, "ret": ret})

    if not trades:
        return 0, 0.0, 0.0, 0.0, 0.0, pd.DataFrame()

    t_df = pd.DataFrame(trades)
    t_df['cum_equity'] = (1.0 + t_df['ret']).cumprod()
    t_df['cum_ret_pct'] = (t_df['cum_equity'] - 1.0) * 100.0
    t_df['peak'] = t_df['cum_equity'].cummax()
    t_df['dd_pct'] = ((t_df['cum_equity'] - t_df['peak']) / t_df['peak']) * 100.0

    win_rate = float((t_df['ret'] > 0).mean() * 100.0)
    avg_ret = float(t_df['ret'].mean() * 100.0)
    total_ret = float(t_df['cum_ret_pct'].iloc[-1])
    mdd = float(t_df['dd_pct'].min())
    return len(t_df), win_rate, avg_ret, total_ret, mdd, t_df

def optimize_atr_params(df_in, n_bars):
    sl_candidates = (1.0, 1.2, 1.5, 2.0, 2.5)
    tp_candidates = (1.5, 2.0, 2.5, 3.0, 4.0)
    results = []
    for s_m in sl_candidates:
        for t_m in tp_candidates:
            if t_m <= s_m:
                continue
            cnt, win_r, avg_r, tot_r, mdd_r, _ = calc_backtest_metrics(df_in, n_bars, t_m, s_m)
            if cnt > 0:
                score = (tot_r * 0.6) + (win_r * 0.3) - (abs(mdd_r) * 0.4)
                results.append({
                    "sl": s_m, "tp": t_m, "rr": round(t_m / s_m, 2),
                    "cnt": cnt, "win": win_r, "tot": tot_r, "mdd": mdd_r, "score": score
                })
    if not results:
        return []
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def render_tv_chart(df_in, tf, tp_price, sl_price):
    pdf = df_in.tail(75).copy()
    ny_idx = pdf.index.tz_convert('America/New_York')
    x_lbl = ny_idx.strftime('%m-%d %H:%M').tolist() if tf != "일봉" else ny_idx.strftime('%Y-%m-%d').tolist()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.78, 0.22])
    fig.add_trace(go.Candlestick(
        x=x_lbl, open=pdf['Open'].tolist(), high=pdf['High'].tolist(), low=pdf['Low'].tolist(), close=pdf['Close'].tolist(),
        increasing_line_color='#EF4444', decreasing_line_color='#3B82F6',
        increasing_fillcolor='#EF4444', decreasing_fillcolor='#3B82F6', name="가격"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=x_lbl, y=pdf['EMA20'].tolist(), line=dict(color='#F59E0B', width=1.1), name="EMA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_lbl, y=pdf['EMA200'].tolist(), line=dict(color='#64748B', width=1.1, dash='dash'), name="EMA200"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_lbl, y=pdf['VWAP'].tolist(), line=dict(color='#A855F7', width=1.2, dash='dot'), name="VWAP"), row=1, col=1)

    fig.add_hline(y=tp_price, line_dash="dash", line_color="#10B981", line_width=1,
                  annotation_text=f"TP ${tp_price:.2f}", annotation_position="top left",
                  annotation_font=dict(color="#10B981", size=10), row=1, col=1)
    fig.add_hline(y=sl_price, line_dash="dash", line_color="#EF4444", line_width=1,
                  annotation_text=f"SL ${sl_price:.2f}", annotation_position="bottom left",
                  annotation_font=dict(color="#EF4444", size=10), row=1, col=1)

    buys = pdf[pdf['Buy_Signal']]
    if not buys.empty:
        b_idx = buys.index.tz_convert('America/New_York')
        b_x = b_idx.strftime('%m-%d %H:%M').tolist() if tf != "일봉" else b_idx.strftime('%Y-%m-%d').tolist()
        fig.add_trace(go.Scatter(
            x=b_x, y=(buys['Low'] * 0.997).tolist(), mode='markers',
            marker=dict(symbol='triangle-up', size=11, color='#10B981'),
            text=buys['Signal_Reason'].tolist(), hoverinfo='text+x+y', name="매수신호"
        ), row=1, col=1)

    sells = pdf[pdf['Sell_Signal']]
    if not sells.empty:
        s_idx = sells.index.tz_convert('America/New_York')
        s_x = s_idx.strftime('%m-%d %H:%M').tolist() if tf != "일봉" else s_idx.strftime('%Y-%m-%d').tolist()
        fig.add_trace(go.Scatter(
            x=s_x, y=(sells['High'] * 1.003).tolist(), mode='markers',
            marker=dict(symbol='triangle-down', size=11, color='#EF4444'),
            text=sells['Signal_Reason'].tolist(), hoverinfo='text+x+y', name="매도신호"
        ), row=1, col=1)

    vol_colors = ['#EF4444' if c >= o else '#3B82F6' for c, o in zip(pdf['Close'], pdf['Open'])]
    fig.add_trace(go.Bar(x=x_lbl, y=pdf['Volume'].tolist(), marker_color=vol_colors, name="거래량", showlegend=False), row=2, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0B0E14", plot_bgcolor="#131722",
        margin=dict(l=5, r=5, t=10, b=10), height=440, xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=10))
    )
    fig.update_xaxes(type='category', tickangle=-45, nticks=8, showgrid=True, gridcolor='#1E293B')
    fig.update_yaxes(showgrid=True, gridcolor='#1E293B', side='right')
    return fig

def render_equity_chart(t_df, tf):
    dt_series = pd.to_datetime(t_df['date'], utc=True).dt.tz_convert('America/New_York')
    x_lbl = dt_series.dt.strftime('%m-%d %H:%M').tolist() if tf != "일봉" else dt_series.dt.strftime('%Y-%m-%d').tolist()
    y_ret = t_df['cum_ret_pct'].astype(float).tolist()
    y_dd = t_df['dd_pct'].astype(float).tolist()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.65, 0.35])
    fig.add_trace(go.Scatter(
        x=x_lbl, y=y_ret, mode='lines+markers',
        line=dict(color='#10B981', width=2), fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.12)', name="누적 수익률(%)"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_lbl, y=y_dd, mode='lines',
        line=dict(color='#EF4444', width=1.5), fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.18)', name="낙폭 DD(%)"
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0B0E14", plot_bgcolor="#131722",
        margin=dict(l=5, r=5, t=10, b=10), height=280, showlegend=False
    )
    fig.update_xaxes(type='category', tickangle=-45, nticks=6, showgrid=True, gridcolor='#1E293B')
    fig.update_yaxes(showgrid=True, gridcolor='#1E293B', side='right')
    return fig
tab1, tab2 = st.tabs(["📊 종목 분석 & 자동 최적화", "🔍 멀티 스캐너 & 카톡 알림"])

with tab1:
    df = load_single(ticker, timeframe, strategy_filter, use_adx_filter)
    if df is None:
        st.error(f"⚠️ '{ticker}' 데이터를 불러올 수 없습니다. 티커명을 확인해주세요.")
    else:
        last, prev = df.iloc[-1], df.iloc[-2]
        chg = float(last['Close'] - prev['Close'])
        chg_pct = float((chg / prev['Close']) * 100.0)
        chg_color = "#EF4444" if chg >= 0 else "#3B82F6"
        chg_sign = "+" if chg >= 0 else ""

        atr_val = float(last['ATR'])
        tp_price = float(last['Close']) + (atr_val * tp_mult)
        sl_price = float(last['Close']) - (atr_val * sl_mult)
        tp_pct = ((tp_price - float(last['Close'])) / float(last['Close'])) * 100.0
        sl_pct = ((sl_price - float(last['Close'])) / float(last['Close'])) * 100.0
        rr_ratio = tp_mult / sl_mult

        if last['Buy_Signal']:
            status_html = f'<span class="badge-buy">{last["Signal_Reason"]}</span>'
        elif last['Sell_Signal']:
            status_html = f'<span class="badge-sell">{last["Signal_Reason"]}</span>'
        else:
            status_html = '<span class="badge-neutral">관망 / 대기</span>'

        st.markdown(f"### 📈 {ticker} <span style='font-size:0.85rem; color:#94A3B8;'>({timeframe})</span> {status_html}", unsafe_allow_html=True)

        bt_cnt, bt_win, bt_avg, bt_tot, bt_mdd, t_df = calc_backtest_metrics(df, hold_bars, tp_mult, sl_mult)
        tot_color = "#10B981" if bt_tot > 0 else ("#EF4444" if bt_tot < 0 else "#94A3B8")

        st.markdown(f"""
        <div class="kpi-grid">
            <div class="metric-card">
                <div class="metric-title">현재가 (USD)</div>
                <div class="metric-value">${last['Close']:,.2f} <span style="font-size:0.8rem; color:{chg_color};">({chg_sign}{chg_pct:.2f}%)</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-title">RSI (14) / ADX 추세강도</div>
                <div class="metric-value">{last['RSI']:.1f} <span style="font-size:0.8rem; color:#94A3B8;">(ADX {last['ADX']:.1f})</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-title">정밀 TP/SL 승률 (최대 {hold_bars}봉)</div>
                <div class="metric-value">{bt_win:.1f}% <span style="font-size:0.78rem; color:#94A3B8;">({bt_cnt}회 | 평균 {bt_avg:+.2f}%)</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-title">누적 수익률 / 최대낙폭(MDD)</div>
                <div class="metric-value" style="color:{tot_color};">{bt_tot:+.2f}% <span style="font-size:0.78rem; color:#EF4444;">({bt_mdd:.2f}%)</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="tpsl-bar">
            <div>
                <div class="ohlc-lbl">🎯 현재 익절가 (TP +{tp_mult}ATR)</div>
                <div class="ohlc-val" style="color:#10B981;">${tp_price:,.2f} (+{tp_pct:.2f}%)</div>
            </div>
            <div>
                <div class="ohlc-lbl">🛡️ 현재 손절가 (SL -{sl_mult}ATR)</div>
                <div class="ohlc-val" style="color:#EF4444;">${sl_price:,.2f} ({sl_pct:.2f}%)</div>
            </div>
            <div>
                <div class="ohlc-lbl">⚖️ 손익비 (R:R)</div>
                <div class="ohlc-val" style="color:#F59E0B;">1 : {rr_ratio:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 종목별 최적 손절/익절 배수 자동 탐색 기능
        if st.button(f"✨ {ticker} 최적 손절/익절 ATR 배수 자동 탐색하기", use_container_width=True):
            opt_list = optimize_atr_params(df, hold_bars)
            if not opt_list:
                st.warning("최적화를 수행할 매수 시그널 이력이 부족합니다. (설정에서 ADX 필터를 해제해 보세요.)")
            else:
                best = opt_list.pop(0)
                opt_tp_p = float(last['Close']) + (atr_val * best['tp'])
                opt_sl_p = float(last['Close']) - (atr_val * best['sl'])
                st.markdown(f"""
                <div class="opt-box">
                    <div style="font-size:0.85rem; font-weight:700; color:#10B981; margin-bottom:4px;">
                        🏆 {ticker} ({timeframe}) AI 추천 최적 비율: 손절 {best['sl']} ATR / 익절 {best['tp']} ATR (손익비 1:{best['rr']})
                    </div>
                    <div style="font-size:0.78rem; color:#E2E8F0;">
                        • 백테스트 성과: <strong>승률 {best['win']:.1f}%</strong> | 누적 수익률 <strong style="color:#10B981;">{best['tot']:+.2f}%</strong> | MDD {best['mdd']:.2f}%<br>
                        • 현재가 기준 적용 시: 목표 익절가 <strong>${opt_tp_p:,.2f}</strong> / 목표 손절가 <strong>${opt_sl_p:,.2f}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if opt_list:
                    st.caption("상위 차선 조합 비교 (설정창에서 배수를 변경해 차트에 반영할 수 있습니다)")
                    for rank_idx, cand in enumerate(opt_list[:2], start=2):
                        st.markdown(f"""
                        <div class="list-card">
                            <div class="list-sub">
                                <span><strong>{rank_idx}위 조합:</strong> 손절 {cand['sl']} ATR / 익절 {cand['tp']} ATR (1:{cand['rr']})</span>
                                <span>승률 {cand['win']:.1f}% | 수익 {cand['tot']:+.2f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        st.plotly_chart(render_tv_chart(df, timeframe, tp_price, sl_price), use_container_width=True, config={'displayModeBar': False})

        alert_text = (
            f"📊 [{ticker} - {timeframe}] 고승률 퀀트 리포트\n"
            f"• 현재가: ${last['Close']:,.2f} ({chg_sign}{chg_pct:.2f}%)\n"
            f"• 상태: {last['Signal_Reason'] if last['Signal_Reason'] else '관망'}\n"
            f"• 추천 TP: ${tp_price:,.2f} (+{tp_pct:.2f}%)\n"
            f"• 추천 SL: ${sl_price:,.2f} ({sl_pct:.2f}%)"
        )
        if st.button("💛 카카오톡 나에게 현재 분석 전송", use_container_width=True):
            ok, msg = send_kakao_alert(kakao_token, alert_text)
            st.toast(msg, icon="✅" if ok else "⚠️")

        st.markdown(f"#### 📈 정밀 백테스트 누적 수익 곡선 & MDD (TP/SL 적용)")
        if t_df.empty:
            st.info("백테스트를 그릴 매수 시그널이 부족합니다. (설정에서 ADX 필터를 끄면 더 많은 신호가 탐색됩니다.)")
        else:
            st.plotly_chart(render_equity_chart(t_df, timeframe), use_container_width=True, config={'displayModeBar': False})

        st.markdown(f"#### 🔔 최근 포착된 시그널 ({strategy_filter})")
        signals = df[df['Buy_Signal'] | df['Sell_Signal']].tail(10).iloc[::-1]
        if signals.empty:
            st.info("선택한 전략 조건을 만족한 최근 시그널이 없습니다.")
        else:
            for idx, row in signals.iterrows():
                ny_t = idx.tz_convert('America/New_York').strftime('%Y-%m-%d %H:%M') if timeframe != "일봉" else idx.tz_convert('America/New_York').strftime('%Y-%m-%d')
                badge = '<span class="badge-buy">BUY</span>' if row['Buy_Signal'] else '<span class="badge-sell">SELL</span>'
                sig_tp = float(row['Close']) + float(row['ATR']) * tp_mult
                sig_sl = float(row['Close']) - float(row['ATR']) * sl_mult
                st.markdown(f"""
                <div class="list-card">
                    <div class="list-top">
                        <div>{badge} <strong style="margin-left:6px; font-size:0.85rem;">{row['Signal_Reason']}</strong></div>
                        <div style="font-weight:700; color:#F8FAFC;">${row['Close']:,.2f}</div>
                    </div>
                    <div class="list-sub">
                        <span>NY: {ny_t} | RSI: {row['RSI']:.1f} | ADX: {row['ADX']:.1f}</span>
                        <span>TP: ${sig_tp:,.2f} / SL: ${sig_sl:,.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.markdown(f"#### 🔍 관심종목 실시간 스캔 ({timeframe})")
    symbols = [s.strip().upper() for s in watchlist_str.split(",") if s.strip()]
    auto_send_alert = st.checkbox("스캔 시 신호 포착 종목 카카오톡 자동 알림 발송", value=False)

    if st.button("🚀 전체 관심종목 지금 스캔하기", use_container_width=True):
        scan_results = []
        alert_lines = []
        progress = st.progress(0)

        for i, sym in enumerate(symbols):
            sdf = load_single(sym, timeframe, strategy_filter, use_adx_filter)
            if sdf is not None and len(sdf) >= 2:
                slast, sprev = sdf.iloc[-1], sdf.iloc[-2]
                schg_pct = ((float(slast['Close']) - float(sprev['Close'])) / float(sprev['Close'])) * 100.0
                s_tp = float(slast['Close']) + float(slast['ATR']) * tp_mult
                s_sl = float(slast['Close']) - float(slast['ATR']) * sl_mult

                recent_df = sdf.tail(3)
                sig_type, sig_msg, sig_time = "NEUTRAL", "특이 신호 없음", ""
                for r_idx, r_row in recent_df.iloc[::-1].iterrows():
                    if r_row['Buy_Signal']:
                        sig_type = "BUY"
                        sig_msg = r_row['Signal_Reason']
                        sig_time = r_idx.tz_convert('America/New_York').strftime('%m-%d %H:%M') if timeframe != "일봉" else r_idx.tz_convert('America/New_York').strftime('%Y-%m-%d')
                        alert_lines.append(f"🟢 [{sym}] {sig_msg} | ${slast['Close']:,.2f} (TP ${s_tp:,.2f} / SL ${s_sl:,.2f})")
                        break
                    elif r_row['Sell_Signal']:
                        sig_type = "SELL"
                        sig_msg = r_row['Signal_Reason']
                        sig_time = r_idx.tz_convert('America/New_York').strftime('%m-%d %H:%M') if timeframe != "일봉" else r_idx.tz_convert('America/New_York').strftime('%Y-%m-%d')
                        alert_lines.append(f"🔴 [{sym}] {sig_msg} | ${slast['Close']:,.2f}")
                        break

                scan_results.append({
                    "sym": sym, "price": float(slast['Close']), "chg_pct": schg_pct,
