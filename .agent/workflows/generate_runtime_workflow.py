"""Generate runtime workflow diagram for p-e trading system (workflows/ directory)."""
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
ax.text(11, 12.7, 'Live Trading Runtime & AI Assistant Workflow',
        ha='center', va='center', fontsize=15, color='#3949AB')
ax.text(11, 12.25, 'Daily Signal Generation   |   Mixture-of-Experts AI   |   API & Dashboard',
        ha='center', va='center', fontsize=11, style='italic', color='#666')

# Color palette
P1 = ('#1976D2', '#E3F2FD')   # blue   — Data
P2 = ('#F57C00', '#FFF3E0')   # orange — Trading Pipeline
P3 = ('#7B1FA2', '#F3E5F5')   # purple — Storage
P4 = ('#388E3C', '#E8F5E9')   # green  — AI Layer
P5 = ('#C62828', '#FFEBEE')   # red    — API
RES = ('#F9A825', '#FFFDE7')  # yellow — Output

def box(x, y, w, h, text, colors, fontsize=10, weight='normal', radius=0.15):
    border, fill = colors
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={radius}",
                       facecolor=fill, edgecolor=border, linewidth=2)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color='#222')

def phase_label(x, y, w, h, num, title, colors):
    border, _ = colors
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
# PHASE 1 — Data Source
# ============================================================
phase_label(0.5, 11.0, 4.0, 0.8, '1', 'DATA SOURCE', P1)

box(0.7, 9.9, 3.6, 0.8, 'Live Market Feed', P1, 11)
box(0.7, 8.9, 3.6, 0.8, 'Historical Database', P1, 11)
box(0.7, 7.9, 3.6, 0.8, 'Trained Models', P1, 11)
box(0.7, 6.9, 3.6, 0.8, 'Configuration', P1, 11)

arrow(2.5, 9.9, 2.5, 9.7)
arrow(2.5, 8.9, 2.5, 8.7)
arrow(2.5, 7.9, 2.5, 7.7)

# ============================================================
# PHASE 2 — Trading Pipeline
# ============================================================
phase_label(5.0, 11.0, 5.0, 0.8, '2', 'TRADING PIPELINE', P2)

box(5.2, 9.9, 4.6, 0.8,
    'Trading System (Daily Run)', P2, 11, 'bold')

box(5.2, 8.7, 4.6, 1.0,
    'Trend Detector\n\nHMM · GMM · ADX + Supertrend',
    P2, 10)

box(5.2, 7.5, 4.6, 1.0,
    'Signal Generator\n\nBUY · SELL · HOLD',
    P2, 10)

box(5.2, 6.3, 4.6, 1.0,
    'Trading Engine\n\nStop Loss · Trailing Stop',
    P2, 10)

arrow(7.5, 9.9, 7.5, 9.7)
arrow(7.5, 8.7, 7.5, 8.5)
arrow(7.5, 7.5, 7.5, 7.3)

arrow(4.3, 9.0, 5.2, 9.0, '#888', 2.5)
arrow(4.3, 8.0, 5.2, 8.0, '#888', 2.5)

# ============================================================
# PHASE 3 — Storage Layer
# ============================================================
phase_label(10.5, 11.0, 4.5, 0.8, '3', 'STORAGE LAYER', P3)

box(10.7, 9.9, 4.1, 0.8, 'Daily Signals Table', P3, 11)
box(10.7, 8.9, 4.1, 0.8, 'Performance Metrics', P3, 11)
box(10.7, 7.9, 4.1, 0.8, 'Trade History', P3, 11)
box(10.7, 6.9, 4.1, 0.8, 'Backtest Results', P3, 11)

arrow(9.8, 7.5, 10.7, 7.5, '#888', 2.5)
arrow(9.8, 6.8, 10.7, 7.0, '#888', 2.5)

arrow(12.75, 9.9, 12.75, 9.7)
arrow(12.75, 8.9, 12.75, 8.7)
arrow(12.75, 7.9, 12.75, 7.7)

# ============================================================
# PHASE 4 — AI Layer (Mixture of Experts)
# ============================================================
phase_label(15.5, 11.0, 6.0, 0.8, '4', 'AI LAYER — MIXTURE OF EXPERTS', P4)

box(15.7, 9.9, 5.6, 0.8,
    'LLM Hybrid Agent', P4, 11, 'bold')

box(15.7, 8.7, 2.6, 1.0,
    'Expert 1\n\nMarket Analyst',
    P4, 10)
box(18.7, 8.7, 2.6, 1.0,
    'Expert 2\n\nRisk Analyst',
    P4, 10)

box(15.7, 7.5, 5.6, 1.0,
    'Master Judge\n\n(Consensus & Final Verdict)',
    P4, 10, 'bold')

box(15.7, 6.3, 5.6, 1.0,
    'Casual Chat Detection\n+ Market News Context',
    P4, 10)

arrow(17.0, 9.9, 17.0, 9.7)
arrow(20.0, 9.9, 20.0, 9.7)
arrow(17.0, 8.7, 17.0, 8.5)
arrow(20.0, 8.7, 20.0, 8.5)
arrow(18.5, 7.5, 18.5, 7.3)

# Storage → AI
arrow(14.8, 8.0, 15.7, 8.0, '#888', 2.5)

# ============================================================
# PHASE 5 — API & Dashboard
# ============================================================
phase_label(0.5, 4.5, 21.0, 0.8, '5', 'API SERVER & DASHBOARD', P5)

box(0.7, 3.2, 4.0, 1.1,
    'Live Signals Endpoint\n\nGET /api/signals', P5, 10)

box(4.9, 3.2, 4.0, 1.1,
    'AI Chat Endpoint\n\nPOST /api/chat (MoE)\nPOST /api/chat/simple', P5, 10)

box(9.1, 3.2, 4.0, 1.1,
    'History Endpoint\n\nGET /api/history', P5, 10)

box(13.3, 3.2, 4.0, 1.1,
    'Performance Endpoint\n\nGET /api/performance', P5, 10)

box(17.5, 3.2, 4.0, 1.1,
    'Web Dashboard\n\nInteractive Charts', P5, 10, 'bold')

arrow(4.7, 3.75, 4.9, 3.75, P5[0], 2)
arrow(8.9, 3.75, 9.1, 3.75, P5[0], 2)
arrow(13.1, 3.75, 13.3, 3.75, P5[0], 2)
arrow(17.3, 3.75, 17.5, 3.75, P5[0], 2)

# Pipeline → API
arrow(7.5, 6.3, 7.5, 5.4, '#888', 2.5)
# Storage → API
arrow(12.75, 6.9, 12.75, 5.4, '#888', 2.5)
# AI → API
arrow(18.5, 6.3, 18.5, 5.4, '#888', 2.5)

# ============================================================
# OUTPUT — Users
# ============================================================
phase_label(0.5, 1.9, 21.0, 0.7, 'OUT', 'CONSUMERS & OUTPUTS', RES)

box(0.7, 0.5, 4.8, 1.1,
    'Daily Signal Reports\n\nBUY / SELL / HOLD',
    RES, 10)

box(5.7, 0.5, 4.8, 1.1,
    'Backtest Charts\n\nPer-Market Visualizations',
    RES, 10)

box(10.7, 0.5, 4.8, 1.1,
    'AI Trading Advisor\n\nNatural Language Q&A',
    RES, 10, 'bold')

box(15.7, 0.5, 5.8, 1.1,
    'External Clients\n\nWeb · Mobile · Notebooks',
    RES, 10)

arrow(11.0, 3.2, 11.0, 2.8, '#888', 2.5)

plt.tight_layout()
output_path = '/Users/oattao/project/p-e/.agent/workflows/runtime_workflow_pe_project.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.3)
plt.close()
print(f'Saved: {output_path}')
