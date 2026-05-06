"""Generate clean visual workflow diagram for p-e trading system (topic names only)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.family'] = 'DejaVu Sans'

fig, ax = plt.subplots(figsize=(22, 14))
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.axis('off')

ax.add_patch(plt.Rectangle((0, 0), 22, 14, facecolor='#FAFAFA', zorder=-1))

# Title
ax.text(11, 13.3, 'P-E Trading System', ha='center', va='center',
        fontsize=26, fontweight='bold', color='#1A237E')
ax.text(11, 12.7, 'Multi-Market Reversal Prediction Workflow',
        ha='center', va='center', fontsize=15, color='#3949AB')
ax.text(11, 12.25, 'US S&P500   |   UK FTSE100   |   Thai SET   |   Gold   |   BTC',
        ha='center', va='center', fontsize=11, style='italic', color='#666')

# Color palette
P1 = ('#1976D2', '#E3F2FD')
P2 = ('#F57C00', '#FFF3E0')
P3 = ('#7B1FA2', '#F3E5F5')
P4 = ('#388E3C', '#E8F5E9')
P5 = ('#C62828', '#FFEBEE')
RES = ('#F9A825', '#FFFDE7')

def box(x, y, w, h, text, colors, fontsize=10, weight='normal', radius=0.15):
    border, fill = colors
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={radius}",
                       facecolor=fill, edgecolor=border, linewidth=2)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color='#222')

def phase_label(x, y, w, h, num, title, colors):
    border, fill = colors
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                       facecolor=border, edgecolor=border, linewidth=2)
    ax.add_patch(b)
    ax.text(x + 0.5, y + h/2, num, ha='center', va='center',
            fontsize=18, fontweight='bold', color='white')
    ax.text(x + w/2 + 0.3, y + h/2, title, ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')

def arrow(x1, y1, x2, y2, color='#555', lw=2.2):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle='-|>', mutation_scale=22,
                        color=color, linewidth=lw,
                        connectionstyle="arc3,rad=0")
    ax.add_patch(a)

# ============================================================
# PHASE 1 — Data Preparation
# ============================================================
phase_label(0.5, 11.0, 4.0, 0.8, '1', 'DATA PREPARATION', P1)

box(0.7, 9.9, 3.6, 0.8, 'Fetch Market Data', P1, 11)
box(0.7, 8.9, 3.6, 0.8, 'Manual Trend Labeling', P1, 11)
box(0.7, 7.9, 3.6, 0.8, 'Train / Val / Test Split', P1, 11)
box(0.7, 6.9, 3.6, 0.8, 'Save Labeled Data', P1, 11)

arrow(2.5, 9.9, 2.5, 9.7)
arrow(2.5, 8.9, 2.5, 8.7)
arrow(2.5, 7.9, 2.5, 7.7)

# ============================================================
# PHASE 2 — Feature Engineering
# ============================================================
phase_label(5.0, 11.0, 4.0, 0.8, '2', 'FEATURE ENGINEERING', P2)

box(5.2, 9.9, 3.6, 0.8, 'Calculate Features', P2, 11)
box(5.2, 8.9, 3.6, 0.8, 'Technical Indicators', P2, 11)
box(5.2, 7.9, 3.6, 0.8, 'Feature Importance', P2, 11)
box(5.2, 6.9, 3.6, 0.8, 'Selected Features', P2, 11)

arrow(7.0, 9.9, 7.0, 9.7)
arrow(7.0, 8.9, 7.0, 8.7)
arrow(7.0, 7.9, 7.0, 7.7)

arrow(4.3, 7.3, 5.2, 7.3, '#888', 2.5)

# ============================================================
# PHASE 3 — Model Training
# ============================================================
phase_label(9.5, 11.0, 5.5, 0.8, '3', 'MODEL TRAINING', P3)

box(9.7, 9.9, 5.1, 0.8,
    'Train Separate Models per Market & Trend',
    P3, 11, 'bold')

box(9.7, 8.7, 2.4, 1.0,
    'Deep Learning\n\nLSTM · CNN\nMLP · Transformer',
    P3, 9.5)
box(12.4, 8.7, 2.4, 1.0,
    'Classical ML\n\nRandom Forest\nSVM · XGBoost',
    P3, 9.5)

box(9.7, 7.3, 2.4, 1.1,
    'Uptrend Models',
    P3, 11, 'bold')
box(12.4, 7.3, 2.4, 1.1,
    'Downtrend Models',
    P3, 11, 'bold')

arrow(10.9, 9.9, 10.9, 9.7)
arrow(13.6, 9.9, 13.6, 9.7)
arrow(10.9, 8.7, 10.9, 8.4)
arrow(13.6, 8.7, 13.6, 8.4)

arrow(8.8, 7.3, 9.7, 7.3, '#888', 2.5)

# ============================================================
# PHASE 4 — Regime Detection & MOO Selection
# ============================================================
phase_label(15.5, 11.0, 6.0, 0.8, '4', 'REGIME + MOO SELECTION', P4)

box(15.7, 9.9, 5.6, 0.8,
    'Super Regime Detection', P4, 11)

box(15.7, 8.9, 5.6, 0.8,
    'HMM · GMM · SMA200 · ADX · LR', P4, 11)

box(15.7, 7.7, 5.6, 1.0,
    'Best Regime per Market\n'
    'US/UK: HMM   |   Gold: SMA200\n'
    'Thai: LogisticReg   |   BTC: XGBoost',
    P4, 10, 'bold')

box(15.7, 6.4, 5.6, 1.1,
    'MOO + ASF (Pareto Front)\n\n'
    'Return ↑   MaxDD ↓   Sharpe ↑   WinRate ↑',
    P4, 10, 'bold')

arrow(18.5, 9.9, 18.5, 9.7)
arrow(18.5, 8.9, 18.5, 8.7)
arrow(18.5, 7.7, 18.5, 7.5)

# ============================================================
# CHAMPION MODELS BANNER
# ============================================================
champ = FancyBboxPatch((6.5, 5.1), 9.0, 1.2,
                       boxstyle="round,pad=0.03,rounding_size=0.2",
                       facecolor='#FFFDE7', edgecolor='#F57F17', linewidth=2.5)
ax.add_patch(champ)
ax.text(11.0, 5.95, 'CHAMPION MODELS PER MARKET',
        ha='center', va='center', fontsize=12, fontweight='bold', color='#E65100')
ax.text(11.0, 5.45,
        'BTC ↑ MLP / ↓ RF      Gold ↑ Transformer / ↓ XGB      Thai ↑ Transformer / ↓ CNN      UK ↑ LSTM / ↓ SVM      US ↑ RF / ↓ LSTM',
        ha='center', va='center', fontsize=9.5, color='#333')

arrow(13.6, 7.3, 13.0, 6.3, '#888', 2)
arrow(15.7, 6.5, 15.0, 6.0, '#888', 2)

# ============================================================
# PHASE 5 — Combined Backtest V2
# ============================================================
phase_label(0.5, 4.0, 21.0, 0.8, '5', 'COMBINED BACKTEST — Regime-Switching Multi-Strategy Optimization', P5)

box(0.7, 2.7, 4.0, 1.1,
    'Load Champion Models', P5, 11)

box(4.9, 2.7, 4.0, 1.1,
    'Per-Step Regime Switch\n\nUptrend → ↑ model\nDowntrend → ↓ model', P5, 10)

box(9.1, 2.7, 4.0, 1.1,
    'Multi-Strategy Search\n\nModes · Thresholds\nTrails · Vol-Targets', P5, 10)

box(13.3, 2.7, 4.0, 1.1,
    'Weighted Strategies\n\nConfidence · Asymmetric\nPyramid · Binary', P5, 10)

box(17.5, 2.7, 4.0, 1.1,
    'Composite Score\n\nAlpha · Sortino · Calmar\nWinRate · Trades − DD', P5, 10, 'bold')

arrow(4.7, 3.25, 4.9, 3.25, P5[0], 2)
arrow(8.9, 3.25, 9.1, 3.25, P5[0], 2)
arrow(13.1, 3.25, 13.3, 3.25, P5[0], 2)
arrow(17.3, 3.25, 17.5, 3.25, P5[0], 2)

arrow(11.0, 5.1, 11.0, 4.85, '#888', 2.5)

# ============================================================
# RESULTS
# ============================================================
phase_label(0.5, 1.6, 21.0, 0.7, 'OUT', 'RESULTS & OUTPUTS', RES)

box(0.7, 0.4, 6.8, 1.0,
    'Per-Market Backtest Plots',
    RES, 11)

box(7.7, 0.4, 6.8, 1.0,
    'Dashboard Summary',
    RES, 11)

box(14.7, 0.4, 6.8, 1.0,
    'Performance Report',
    RES, 11)

arrow(11.0, 2.7, 11.0, 2.4, '#888', 2.5)

plt.tight_layout()
output_path = '/Users/oattao/project/p-e/.agent/workflows/workflow_v2_pe_project.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.3)
plt.close()
print(f'Saved: {output_path}')
