with c1:
    color_p = "#10B981" if price_diff >= 0 else "#EF4444"
    sign = "+" if price_diff >= 0 else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{ticker} 현재가 ({timeframe})</div>
        <div class="metric-value">${latest['Close']:,.2f} <span style="font-size:0.95rem; color:{color_p};">({sign}{price_pct:.2f}%)</span></div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">현재 추세 및 시그널 판정</div>
        <div style="margin-top:6px;">{status_html}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">최근 매수 시그널 ({last_buy_str})</div>
        <div class="metric-value" style="font-size:1.15rem; color:#10B981;">{last_buy_reason}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    rsi_val = latest['RSI']
    rsi_col = "#EF4444" if rsi_val >= 70 else ("#10B981" if rsi_val <= 30 else "#F8FAFC")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">RSI (14) / 최근 매도({last_sell_str})</div>
        <div class="metric-value" style="color:{rsi_col};">{rsi_val:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

# 5. 메인 3단 인터랙티브 차트 (캔들+지표+시그널 / 거래량 / RSI)
plot_df = df.tail(120).copy()
x_labels = plot_df.index.strftime('%m-%d %H:%M') if timeframe != "일봉" else plot_df.index.strftime('%Y-%m-%d')

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[0.6, 0.2, 0.2],
    subplot_titles=(f"{ticker} 주가 & 퀀트 전략 시그널", "거래량 (Volume)", "RSI (14)")
)

fig.add_trace(go.Candlestick(
    x=x_labels,
    open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'],
    name="주가",
    increasing_line_color='#10B981', decreasing_line_color='#EF4444'
), row=1, col=1)

fig.add_trace(go.Scatter(x=x_labels, y=plot_df['EMA20'], name="EMA 20", line=dict(color='#38BDF8', width=1.2)), row=1, col=1)
fig.add_trace(go.Scatter(x=x_labels, y=plot_df['EMA50'], name="EMA 50", line=dict(color='#F59E0B', width=1.2)), row=1, col=1)
fig.add_trace(go.Scatter(x=x_labels, y=plot_df['EMA200'], name="EMA 200", line=dict(color='#A855F7', width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=x_labels, y=plot_df['VWAP'], name="VWAP", line=dict(color='#F43F5E', width=1.5, dash='dot')), row=1, col=1)

buy_pts = plot_df[plot_df['Buy_Signal']]
if not buy_pts.empty:
    buy_x = buy_pts.index.strftime('%m-%d %H:%M') if timeframe != "일봉" else buy_pts.index.strftime('%Y-%m-%d')
    fig.add_trace(go.Scatter(
        x=buy_x, y=buy_pts['Low'] * 0.996,
        mode='markers+text',
        name='매수 시그널',
        text=buy_pts['Signal_Reason'],
        textposition='bottom center',
        marker=dict(symbol='triangle-up', size=13, color='#10B981'),
        textfont=dict(color='#10B981', size=10)
    ), row=1, col=1)

sell_pts = plot_df[plot_df['Sell_Signal']]
if not sell_pts.empty:
    sell_x = sell_pts.index.strftime('%m-%d %H:%M') if timeframe != "일봉" else sell_pts.index.strftime('%Y-%m-%d')
    fig.add_trace(go.Scatter(
        x=sell_x, y=sell_pts['High'] * 1.004,
        mode='markers+text',
        name='매도 시그널',
        text=sell_pts['Signal_Reason'],
        textposition='top center',
        marker=dict(symbol='triangle-down', size=13, color='#EF4444'),
        textfont=dict(color='#EF4444', size=10)
    ), row=1, col=1)

vol_colors = ['#10B981' if c >= o else '#EF4444' for c, o in zip(plot_df['Close'], plot_df['Open'])]
fig.add_trace(go.Bar(x=x_labels, y=plot_df['Volume'], name="거래량", marker_color=vol_colors, opacity=0.7), row=2, col=1)
fig.add_trace(go.Scatter(x=x_labels, y=plot_df['Vol_MA20'], name="거래량 20이평", line=dict(color='#94A3B8', width=1)), row=2, col=1)

fig.add_trace(go.Scatter(x=x_labels, y=plot_df['RSI'], name="RSI (14)", line=dict(color='#818CF8', width=1.5)), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="#EF4444", opacity=0.5, row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="#10B981", opacity=0.5, row=3, col=1)

fig.update_layout(
    height=780,
    template="plotly_dark",
    paper_bgcolor="#0B0E14",
    plot_bgcolor="#131722",
    xaxis_rangeslider_visible=False,
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig.update_xaxes(type='category', nticks=12, gridcolor='#1E293B')
fig.update_yaxes(gridcolor='#1E293B')

st.plotly_chart(fig, use_container_width=True)

# 6. 하단 최근 시그널 발생 이력 테이블
st.subheader("📋 최근 포착된 퀀트 매매 시그널 이력")
signals_df = df[df['Buy_Signal'] | df['Sell_Signal']].copy().tail(15).iloc[::-1]

if signals_df.empty:
    st.info("선택한 기간 및 전략 조건에서 발생한 시그널이 없습니다.")
else:
    display_df = pd.DataFrame({
        "뉴욕 현지시간": signals_df.index.tz_convert('America/New_York').strftime('%Y-%m-%d %H:%M'),
        "구분": np.where(signals_df['Buy_Signal'], "🟢 매수 (BUY)", "🔴 매도 (SELL)"),
        "포착 전략": signals_df['Signal_Reason'],
        "종가 ($)": signals_df['Close'].map(lambda x: f"${x:,.2f}"),
        "VWAP ($)": signals_df['VWAP'].map(lambda x: f"${x:,.2f}"),
        "RSI(14)": signals_df['RSI'].map(lambda x: f"{x:.1f}")
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)
