with st.container(border=True):
    st.markdown(f"""
    <div class="ohlc-bar">
        <div><div class="ohlc-lbl">시가 (Open)</div><div class="ohlc-val">${latest['Open']:,.2f}</div></div>
        <div><div class="ohlc-lbl">고가 (High)</div><div class="ohlc-val" style="color:#10B981;">${latest['High']:,.2f}</div></div>
        <div><div class="ohlc-lbl">저가 (Low)</div><div class="ohlc-val" style="color:#EF4444;">${latest['Low']:,.2f}</div></div>
        <div><div class="ohlc-lbl">VWAP</div><div class="ohlc-val" style="color:#38BDF8;">${latest['VWAP']:,.2f}</div></div>
    </div>
    """, unsafe_allow_html=True)

    plot_df = df.tail(55).copy()
    plot_idx = plot_df.index
    plot_df = plot_df.reset_index(drop=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 7.2), dpi=160, gridspec_kw={'height_ratios': [3.3, 1, 1]}, sharex=True)
    fig.patch.set_facecolor('#131722')

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor('#0B0E14')
        ax.tick_params(colors='#94A3B8', labelsize=8)
        ax.grid(True, color='#1E293B', linestyle='-', linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color('#1E293B')

    for i, r in plot_df.iterrows():
        c = '#10B981' if r['Close'] >= r['Open'] else '#EF4444'
        ax1.plot([i, i], [r['Low'], r['High']], color=c, linewidth=1.0)
        low_b = min(r['Open'], r['Close'])
        h_b = max(abs(r['Close'] - r['Open']), (plot_df['High'].max() - plot_df['Low'].min()) * 0.004)
        ax1.add_patch(patches.Rectangle((i - 0.32, low_b), 0.64, h_b, facecolor=c, edgecolor=c))

    ax1.plot(plot_df.index, plot_df['EMA20'], color='#38BDF8', linewidth=1.2, label='EMA20')
    ax1.plot(plot_df.index, plot_df['EMA50'], color='#F59E0B', linewidth=1.2, label='EMA50')
    ax1.plot(plot_df.index, plot_df['EMA200'], color='#A855F7', linewidth=1.3, label='EMA200')
    ax1.plot(plot_df.index, plot_df['VWAP'], color='#F43F5E', linewidth=1.3, linestyle='--', label='VWAP')

    buy_pts = plot_df[plot_df['Buy_Signal']]
    if not buy_pts.empty:
        ax1.scatter(buy_pts.index, buy_pts['Low'] * 0.997, marker='^', color='#10B981', s=75, zorder=5, label='BUY')
    sell_pts = plot_df[plot_df['Sell_Signal']]
    if not sell_pts.empty:
        ax1.scatter(sell_pts.index, sell_pts['High'] * 1.003, marker='v', color='#EF4444', s=75, zorder=5, label='SELL')

    ax1.legend(loc='upper left', facecolor='#131722', edgecolor='#1E293B', labelcolor='#E2E8F0', fontsize=7.5, ncol=3)
    ax1.set_ylabel('Price ($)', color='#94A3B8', fontsize=8)

    vol_cols = ['#10B981' if c >= o else '#EF4444' for c, o in zip(plot_df['Close'], plot_df['Open'])]
    ax2.bar(plot_df.index, plot_df['Volume'], color=vol_cols, width=0.65, alpha=0.85)
    ax2.plot(plot_df.index, plot_df['Vol_MA20'], color='#94A3B8', linewidth=0.9)
    ax2.set_ylabel('Vol', color='#94A3B8', fontsize=8)

    ax3.plot(plot_df.index, plot_df['RSI'], color='#818CF8', linewidth=1.3)
    ax3.axhline(70, color='#EF4444', linestyle='--', linewidth=0.8, alpha=0.6)
    ax3.axhline(30, color='#10B981', linestyle='--', linewidth=0.8, alpha=0.6)
    ax3.set_ylim(10, 90)
    ax3.set_ylabel('RSI', color='#94A3B8', fontsize=8)

    step = max(1, len(plot_df) // 5)
    ticks = list(range(0, len(plot_df), step))
    labels = [plot_idx[i].strftime('%m/%d %H:%M') if timeframe != "일봉" else plot_idx[i].strftime('%y/%m/%d') for i in ticks]
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(labels, fontsize=7.5, color='#94A3B8')

    plt.tight_layout(pad=0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='#131722')
    plt.close(fig)
    buf.seek(0)
    st.image(buf, use_container_width=True)

st.markdown(f"#### 🧾 {ticker} 최근 {timeframe} 시세 데이터")
recent_ohlcv = df.tail(10).iloc[::-1].copy()
ohlcv_table = pd.DataFrame({
    "시간(NY)": recent_ohlcv.index.tz_convert('America/New_York').strftime('%m-%d %H:%M') if timeframe != "일봉" else recent_ohlcv.index.strftime('%Y-%m-%d'),
    "시가": recent_ohlcv['Open'].map(lambda x: f"${x:,.2f}"),
    "고가": recent_ohlcv['High'].map(lambda x: f"${x:,.2f}"),
    "저가": recent_ohlcv['Low'].map(lambda x: f"${x:,.2f}"),
    "종가": recent_ohlcv['Close'].map(lambda x: f"${x:,.2f}"),
    "거래량": recent_ohlcv['Volume'].map(lambda x: f"{int(x):,}")
})
st.dataframe(ohlcv_table, use_container_width=True, hide_index=True)

st.markdown("#### 📋 최근 매매 시그널 이력")
signals_df = df[df['Buy_Signal'] | df['Sell_Signal']].copy().tail(15).iloc[::-1]
if signals_df.empty:
    st.info("선택한 기간 및 전략 조건에서 발생한 시그널이 없습니다.")
else:
    display_df = pd.DataFrame({
        "뉴욕시간": signals_df.index.tz_convert('America/New_York').strftime('%m-%d %H:%M'),
        "구분": np.where(signals_df['Buy_Signal'], "🟢 매수", "🔴 매도"),
        "전략": signals_df['Signal_Reason'],
        "종가($)": signals_df['Close'].map(lambda x: f"${x:,.2f}"),
        "RSI": signals_df['RSI'].map(lambda x: f"{x:.1f}")
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)
# END
