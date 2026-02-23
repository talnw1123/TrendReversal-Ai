import json
import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def _extract_trade_points(signals, prices, stop_loss_pct=0.05, trailing_stop_pct=0.0, long_only=True, strategy_mode='active'):
    """Replicate the exact execution logic to find exact Buy/Sell coordinates"""
    trade_points = []
    position = 0
    entry_price = 0.0
    highest_price = 0.0

    for i in range(len(signals) - 1):
        signal = signals[i]
        price = prices[i]

        if position == 1 and price > highest_price:
            highest_price = price

        if position != 0 and entry_price > 0:
            if position == 1:
                pnl_pct = (price - entry_price) / entry_price
                if trailing_stop_pct > 0:
                    dd_from_peak = (highest_price - price) / highest_price
                    if dd_from_peak > trailing_stop_pct:
                        trade_points.append((i, 'sell'))
                        position = 0
                        entry_price = 0.0
                        highest_price = 0.0
                        continue
            else:
                pnl_pct = (entry_price - price) / entry_price

            if pnl_pct < -stop_loss_pct:
                trade_points.append((i, 'sell'))
                position = 0
                entry_price = 0.0
                highest_price = 0.0
                continue

        target_pos = 0
        if signal == 1:
            target_pos = 1
        elif signal == 0:
            target_pos = -1 if not long_only else 0
        elif signal == 99:
            target_pos = position if strategy_mode == 'smart_hold' else 0

        if position != target_pos:
            if position != 0:
                trade_points.append((i, 'sell'))
            if target_pos != 0:
                trade_points.append((i, 'buy' if target_pos == 1 else 'sell'))
                entry_price = price
                if target_pos == 1:
                    highest_price = price
            else:
                entry_price = 0.0

            position = target_pos

    # Close any open positions at the end
    if position != 0:
        trade_points.append((len(signals) - 1, 'sell'))

    return trade_points

def create_interactive_plot(json_file):
    if not os.path.exists(json_file):
        print(f"❌ File not found: {json_file}")
        return

    # User may pass BTC_history.json but we need BTC_plot_data.json
    if json_file.endswith('_history.json'):
        json_file = json_file.replace('_history.json', '_plot_data.json')
        if not os.path.exists(json_file):
            print(f"❌ Matching plot_data.json not found: {json_file}")
            print(f"   Please re-run `python workflows/trading_system.py --backtest --market <MKT>`")
            return

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dates = pd.to_datetime(data['dates'])
    prices = np.array(data['prices'])
    bnh_return = data['bnh_return']
    strategy_return = data['strategy_return']
    equity_curve = np.array(data['equity_curve'])
    signals = np.array(data['signals'])
    regime = np.array(data['regime'])
    market = data['market']
    sim_params = data['sim_params']

    # Adjust arrays (same alignment as plot_backtest)
    plot_dates = dates[1:]
    plot_prices = prices[1:]
    plot_bnh = (prices[1:] / prices[0]) * equity_curve[0] if len(equity_curve) > 0 else []
    plot_equity = equity_curve
    plot_signals = signals[:-1]
    plot_regime = regime[1:] if len(regime) == len(dates) else regime[:len(plot_dates)]

    # Extract actual trade points
    trade_points = _extract_trade_points(
        signals=signals,
        prices=prices,
        stop_loss_pct=sim_params.get('stop_loss', 0.05),
        trailing_stop_pct=sim_params.get('trailing', 0.0),
        long_only=sim_params.get('long_only', True),
        strategy_mode=sim_params.get('strategy', 'active')
    )

    buy_dates, buy_prices = [], []
    sell_dates, sell_prices = [], []

    for idx, action in trade_points:
        if 0 <= idx < len(dates):
            if action == 'buy':
                buy_dates.append(dates[idx])
                buy_prices.append(prices[idx])
            elif action == 'sell':
                sell_dates.append(dates[idx])
                sell_prices.append(prices[idx])

    # ---------------------------------------------------------
    # Create Subplots
    # ---------------------------------------------------------
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08,
                        subplot_titles=(f"{market} Combined - Price & Signals", 
                                        f"Equity Curve (Hybrid_MOO)"))

    # Panel 1: Price
    fig.add_trace(go.Scatter(
        x=plot_dates, y=plot_prices,
        mode='lines', name='Price',
        line=dict(color='black', width=1.5),
        hovertemplate='%{x|%Y-%m-%d}<br>Price: %{y:,.2f}<extra></extra>'
    ), row=1, col=1)

    # Markers
    if buy_dates:
        fig.add_trace(go.Scatter(
            x=buy_dates, y=buy_prices,
            mode='markers', name='Buy/Long',
            marker=dict(symbol='triangle-up', size=14, color='green', line=dict(width=1, color='darkgreen')),
            hovertemplate='%{x|%Y-%m-%d}<br><b>BUY</b> @ %{y:,.2f}<extra></extra>'
        ), row=1, col=1)

    if sell_dates:
        fig.add_trace(go.Scatter(
            x=sell_dates, y=sell_prices,
            mode='markers', name='Sell/Exit',
            marker=dict(symbol='triangle-down', size=14, color='red', line=dict(width=1, color='darkred')),
            hovertemplate='%{x|%Y-%m-%d}<br><b>SELL</b> @ %{y:,.2f}<extra></extra>'
        ), row=1, col=1)

    # Panel 2: Equity Curves
    fig.add_trace(go.Scatter(
        x=plot_dates, y=plot_bnh,
        mode='lines', name=f'Buy & Hold ({bnh_return:.2f}%)',
        line=dict(color='gray', width=2, dash='dash'),
        hovertemplate='Buy & Hold: $%{y:,.2f}<extra></extra>'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=plot_dates, y=plot_equity,
        mode='lines', name=f'Strategy ({strategy_return:.2f}%)',
        line=dict(color='blue', width=2.5),
        hovertemplate='Strategy: $%{y:,.2f}<extra></extra>'
    ), row=2, col=1)

    # Add Regime Background Shapes
    shapes = []
    trend_blocks = []
    if len(plot_regime) > 0:
        current_trend = plot_regime[0]
        start_idx = 0
        for i in range(1, len(plot_regime)):
            if plot_regime[i] != current_trend:
                trend_blocks.append((start_idx, i-1, current_trend))
                current_trend = plot_regime[i]
                start_idx = i
        trend_blocks.append((start_idx, len(plot_regime)-1, current_trend))

    for start, end, t in trend_blocks:
        color = 'rgba(0, 255, 0, 0.08)' if t == 1 else 'rgba(255, 0, 0, 0.08)'
        # Shape for top panel
        shapes.append(dict(
            type="rect", xref="x", yref="paper", 
            x0=plot_dates[start], y0=0.54, x1=plot_dates[end], y1=1,
            fillcolor=color, layer="below", line_width=0
        ))
        # Shape for bottom panel
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=plot_dates[start], y0=0, x1=plot_dates[end], y1=0.46,
            fillcolor=color, layer="below", line_width=0
        ))

    fig.update_layout(shapes=shapes)

    fig.update_layout(
        title=f"📈 Advanced Backtest History: {market} (Dual Panels)",
        height=900,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=2, col=1)

    out_file = json_file.replace('_plot_data.json', '_interactive_dual.html')
    fig.write_html(out_file)
    print(f"✅ Interactive Dual-Plot Generated: {out_file}")
    print(f"👉 Open this HTML file in your browser to view the identical PNG structure interactively!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot', 'BTC_plot_data.json')
        
    create_interactive_plot(target)
