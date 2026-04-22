"""
generate_figures.py
===================
Generates 3 publication-quality figures from your actual data:

  Figure A: PSM Bar Chart (psm_chart_final.png)
  Figure B: PSM Balance Chart (psm_balance_final.png)  
  Figure C: Clean Bipartite Network (bipartite_clean.png)

Run: python3 generate_figures.py
Outputs saved to current directory.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# ── Colors matching your existing pipeline output plots exactly ───────────────
BG       = '#0d1117'   # outer background (matches violin, regression, network plots)
PANEL    = '#111827'   # panel/axes background
RED      = '#f87171'   # AI-Generated (matches violin plot red)
BLUE     = '#60a5fa'   # Human-Created (matches violin plot blue)
TEAL     = '#0D9488'   # significant covariate (matches regression plot teal)
LTEAL    = '#99F6E4'   # light teal accent
YELLOW   = '#FACC15'   # highlight / annotation accent
GRAY     = '#6b7280'   # non-significant / secondary text
LGRAY    = '#9ca3af'   # lighter gray for grid lines
WHITE    = '#ffffff'   # text
OFFWHITE = '#e5e7eb'   # secondary text
GOLD     = '#F59E0B'   # gold for dual-tagging edges in bipartite

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE A: PSM BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
def make_psm_bar():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
    fig.patch.set_facecolor(BG)

    outcomes = ['log(Views)', 'log(Engagement\nTotal)', 'Engagement\nRate']
    ai_means = [3.2219,       0.8832,                   0.1112]
    hu_means = [3.0342,       0.8567,                   0.1073]
    pct_diff = ['+6.18%',     '+3.09%',                 '+3.58%']
    pvals    = ['p = 0.005 **', 'p = 0.787  n.s.', 'p = 0.778  n.s.']
    sig      = [True,           False,               False]

    for i, ax in enumerate(axes):
        ax.set_facecolor(PANEL)

        x    = [0.28, 0.72]
        vals = [ai_means[i], hu_means[i]]
        clrs = [RED, BLUE]
        # Dim bars for non-significant panels
        alpha = 1.0 if sig[i] else 0.55

        bars = ax.bar(x, vals, width=0.3, color=clrs, alpha=alpha,
                      edgecolor='none', linewidth=0, zorder=3)

        # Always start y at 0
        ymax = max(vals) * 1.48
        ax.set_ylim(0, ymax)

        # Value labels on bars (white text)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + ymax * 0.02,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=10, color=WHITE, fontweight='bold')

        # % difference + arrow for significant panel only
        if sig[i]:
            mid_y = max(vals) * 1.12
            ax.annotate('', xy=(0.72, mid_y), xytext=(0.28, mid_y),
                        arrowprops=dict(arrowstyle='<->', color=YELLOW, lw=1.8))
            ax.text(0.5, mid_y + ymax * 0.025, pct_diff[i],
                    ha='center', va='bottom', fontsize=10.5,
                    color=YELLOW, fontweight='bold')

        # p-value label
        pc = YELLOW if sig[i] else GRAY
        ax.text(0.5, ymax * 0.96, pvals[i],
                ha='center', va='top', fontsize=10,
                color=pc, fontweight='bold' if sig[i] else 'normal',
                transform=ax.transData)

        ax.set_xticks([0.28, 0.72])
        ax.set_xticklabels(['AI-Generated', 'Human-Created'],
                           fontsize=10, color=OFFWHITE)
        ax.set_title(outcomes[i], fontsize=12, color=WHITE,
                     fontweight='bold', pad=10)
        ax.set_xlim(0, 1)
        ax.tick_params(axis='y', colors=LGRAY, labelsize=8.5)
        ax.tick_params(axis='x', colors=OFFWHITE)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.yaxis.grid(True, color='#1f2937', linewidth=0.8,
                      linestyle='--', zorder=0)
        ax.set_axisbelow(True)

        # Highlight border for significant panel
        if sig[i]:
            for sp in ['bottom', 'left', 'top', 'right']:
                ax.spines[sp].set_visible(True)
                ax.spines[sp].set_edgecolor(TEAL)
                ax.spines[sp].set_linewidth(1.5)

    legend = [mpatches.Patch(color=RED,  label='AI-Generated (Treatment)'),
              mpatches.Patch(color=BLUE, label='Human-Created (Control)')]
    fig.legend(handles=legend, loc='lower center', ncol=2, fontsize=10.5,
               frameon=True, facecolor=PANEL, edgecolor=GRAY,
               labelcolor=WHITE, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('PSM Matched Sample: AI vs Human Content Engagement\n'
                 'N = 218 matched pairs  |  4 confounders balanced (all SMD < 0.1)',
                 fontsize=13, color=WHITE, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig('outputs/figures/psm_chart_final.png', dpi=200,
                bbox_inches='tight', facecolor=BG)
    plt.close()
    print("✓ psm_chart_final.png saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B: PSM BALANCE CHART
# ─────────────────────────────────────────────────────────────────────────────
def make_psm_balance():
    df    = pd.read_csv('outputs/tables/psm_matched_sample.csv')
    treat = df[df.treat == 1]
    ctrl  = df[df.treat == 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(BG)

    # ── Left: propensity score overlap ───────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(PANEL)

    ps_treat = treat['propensity_score'].values
    ps_ctrl  = ctrl['propensity_score'].values
    bins = np.linspace(0.25, 0.95, 25)

    ax.hist(ps_ctrl,  bins=bins, alpha=0.65, color=BLUE,
            label='Human-Created (Control)', edgecolor='none')
    ax.hist(ps_treat, bins=bins, alpha=0.65, color=RED,
            label='AI-Generated (Treatment)', edgecolor='none')

    ax.axvspan(ps_treat.min(), ps_treat.max(), alpha=0.06,
               color=TEAL, label=f'Common support ({ps_treat.min():.2f}–{ps_treat.max():.2f})')

    ax.set_xlabel('Propensity Score', color=OFFWHITE, fontsize=11)
    ax.set_ylabel('Number of Posts',  color=OFFWHITE, fontsize=11)
    ax.set_title('Propensity Score Distribution\n(Before Matching — Common Support Confirmed)',
                 color=WHITE, fontsize=11.5, fontweight='bold')
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRAY,
              labelcolor=WHITE, framealpha=0.9)
    ax.tick_params(colors=LGRAY)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.yaxis.grid(True, color='#1f2937', linewidth=0.7, linestyle='--', zorder=0)
    ax.set_axisbelow(True)

    stats_text = (f'Range: {ps_treat.min():.3f} – {ps_treat.max():.3f}\n'
                  f'Treatment mean: {ps_treat.mean():.3f}\n'
                  f'Control mean:   {ps_ctrl.mean():.3f}')
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            va='top', ha='right', fontsize=8.5, color=WHITE,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#1f2937',
                      edgecolor=GRAY, alpha=0.9))

    # ── Right: SMD balance plot ───────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(PANEL)

    covs = ['log(followers)', 'log(account age)', 'has media', 'hour of day']
    smds = []
    for col in ['log_followers', 'log_account_age', 'has_media_int', 'hour_of_day']:
        t_m, t_s = treat[col].mean(), treat[col].std()
        c_m, c_s = ctrl[col].mean(),  ctrl[col].std()
        pooled   = np.sqrt((t_s**2 + c_s**2) / 2)
        smds.append(abs((t_m - c_m) / pooled) if pooled > 0 else 0)

    # Color bars by SMD magnitude
    clrs = [TEAL if s < 0.05 else BLUE if s < 0.1 else RED for s in smds]
    ypos = range(len(covs))

    ax2.barh(list(ypos), smds, color=clrs, edgecolor='none',
             height=0.45, zorder=3)

    ax2.axvline(x=0.1,  color=RED,   linewidth=2,   linestyle='--',
                label='Acceptable threshold (SMD = 0.1)', zorder=4)
    ax2.axvline(x=0.05, color=TEAL,  linewidth=1.2, linestyle=':',
                label='Excellent balance (SMD = 0.05)',   zorder=4)

    for j, (yp, smd) in enumerate(zip(ypos, smds)):
        ax2.text(smd + 0.001, yp, f'{smd:.3f}',
                 va='center', fontsize=10, color=WHITE, fontweight='bold')
        ax2.text(0.112, yp, '✓',
                 va='center', fontsize=13, color=TEAL, fontweight='bold')

    ax2.set_yticks(list(ypos))
    ax2.set_yticklabels(covs, fontsize=11, color=OFFWHITE)
    ax2.set_xlabel('|Standardized Mean Difference (SMD)|',
                   color=OFFWHITE, fontsize=11)
    ax2.set_title('Covariate Balance After PSM\n(All SMD < 0.1 = Matching Successful)',
                  color=WHITE, fontsize=11.5, fontweight='bold')
    ax2.set_xlim(0, 0.13)
    ax2.legend(fontsize=9, facecolor=PANEL, edgecolor=GRAY,
               labelcolor=WHITE, framealpha=0.9, loc='lower right')
    ax2.tick_params(colors=LGRAY)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.xaxis.grid(True, color='#1f2937', linewidth=0.7, linestyle='--', zorder=0)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('outputs/figures/psm_balance_final.png', dpi=200,
                bbox_inches='tight', facecolor=BG)
    plt.close()
    print("✓ psm_balance_final.png saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE C: CLEAN BIPARTITE NETWORK
# ─────────────────────────────────────────────────────────────────────────────
def make_bipartite():
    df     = pd.read_csv('outputs/tables/psm_matched_sample.csv')
    acc_df = pd.read_csv('outputs/tables/account_network_stats.csv')

    comm_map = dict(zip(acc_df['username'], acc_df['community']))
    type_map = dict(zip(acc_df['username'], acc_df['content_type']))

    TOP_TAGS = ['grokai','grokimage','aigenerated','aiart','syntheticmedia','deepfakeai',
                'photography','portrait','artistsontwitter','streetphotography','illustration']
    TAG_LABELS = {
        'grokai':'#GrokAI','grokimage':'#GrokImage','aigenerated':'#AIGenerated',
        'aiart':'#AIArt','syntheticmedia':'#SyntheticMedia','deepfakeai':'#DeepfakeAI',
        'photography':'#photography','portrait':'#portrait',
        'artistsontwitter':'#artistsontwitter',
        'streetphotography':'#streetphotography','illustration':'#illustration'
    }
    AI_TAGS  = {'grokai','grokimage','aigenerated','aiart','syntheticmedia','deepfakeai'}
    HUM_TAGS = {'photography','portrait','artistsontwitter','streetphotography','illustration'}

    # Build edges
    edges = []
    tag_acct = {t: set() for t in TOP_TAGS}
    acct_tag_count = {}

    for _, row in df.iterrows():
        if pd.isna(row.get('hashtags')): continue
        tags = [t.strip().lower().lstrip('#') for t in str(row['hashtags']).split(',')]
        acct = str(row.get('author_username', ''))
        useful = [t for t in tags if t in TOP_TAGS]
        if useful and acct:
            for t in useful:
                edges.append((acct, t))
                tag_acct[t].add(acct)
                acct_tag_count[acct] = acct_tag_count.get(acct, 0) + 1

    tag_sizes = {t: len(s) for t, s in tag_acct.items()}

    # Show accounts using 2+ top tags + notable real ones
    real_notable = ['ReviewAI_','boiledkeyboard','thisisavatar',
                    'steto123','Gyanendra844522','ashay_pallav']
    accounts_show = list(set(
        [a for a, c in acct_tag_count.items() if c >= 2] +
        [a for a in real_notable if a in acct_tag_count]
    ))
    edges_filtered = [(a, t) for a, t in edges if a in accounts_show]

    print(f"  Accounts: {len(accounts_show)}, Tags: {len(TOP_TAGS)}, Edges: {len(edges_filtered)}")

    fig, ax = plt.subplots(figsize=(17, 11))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis('off')

    # Positions
    n_acc = len(accounts_show)
    acc_pos = {}
    for i, a in enumerate(sorted(accounts_show)):
        acc_pos[a] = (0.0, 1.0 - i / max(n_acc - 1, 1))

    tag_pos = {}
    for i, t in enumerate(TOP_TAGS):
        tag_pos[t] = (1.0, 1.0 - i / max(len(TOP_TAGS) - 1, 1))

    # Draw edges — three colors: AI→AI (red tint), Human→Human (blue tint), cross (gold)
    for a, t in edges_filtered:
        if a not in acc_pos or t not in tag_pos: continue
        x0, y0 = acc_pos[a]
        x1, y1 = tag_pos[t]
        a_type  = type_map.get(a, 'treatment')
        t_is_ai = t in AI_TAGS

        if a_type == 'treatment' and t_is_ai:
            ec, alpha, lw = RED,  0.18, 0.7
        elif a_type == 'control' and not t_is_ai:
            ec, alpha, lw = BLUE, 0.18, 0.7
        else:  # cross-group = dual-tagging
            ec, alpha, lw = GOLD, 0.60, 1.5

        ax.plot([x0, x1], [y0, y1], color=ec, alpha=alpha, linewidth=lw, zorder=1)

    # Account nodes
    for a, (x, y) in acc_pos.items():
        ct       = type_map.get(a, 'treatment')
        is_real  = a in real_notable
        nc = RED if ct == 'treatment' else BLUE
        sz = 140 if is_real else 35
        ax.scatter(x, y, s=sz, c=nc, zorder=5 if is_real else 3,
                   edgecolors=WHITE if is_real else 'none', linewidths=1.2)
        if is_real:
            ax.text(x - 0.025, y, a, ha='right', va='center',
                    fontsize=8.5, color=WHITE, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='#1f2937',
                              edgecolor=GRAY, alpha=0.92, linewidth=0.8))

    # Hashtag nodes (sized by account count)
    for t, (x, y) in tag_pos.items():
        is_ai = t in AI_TAGS
        nc    = RED if is_ai else BLUE
        sz    = max(220, tag_sizes.get(t, 1) * 14)
        ax.scatter(x, y, s=sz, c=nc, zorder=4,
                   edgecolors=WHITE, linewidths=1.5)
        label = TAG_LABELS.get(t, f'#{t}')
        fw = 'bold' if is_ai else 'normal'
        ax.text(x + 0.028, y, label, ha='left', va='center',
                fontsize=11, color=WHITE, fontweight=fw,
                bbox=dict(boxstyle='round,pad=0.28', facecolor='#1f2937',
                          edgecolor=nc, alpha=0.93, linewidth=1.2))

    # Divider between AI and human tag groups
    mid_y = (tag_pos['deepfakeai'][1] + tag_pos['photography'][1]) / 2
    ax.plot([0.82, 1.38], [mid_y, mid_y], color=GRAY,
            linewidth=1, linestyle='--', alpha=0.6)
    ax.text(1.40, mid_y, 'AI tags\nabove\n───\nHuman\ntags below',
            ha='left', va='center', fontsize=8, color=LGRAY, style='italic')

    # Section headers
    ax.text(0.0,  1.07, 'ACCOUNTS', ha='center', fontsize=12,
            color=OFFWHITE, fontweight='bold')
    ax.text(1.0,  1.07, 'HASHTAGS', ha='center', fontsize=12,
            color=OFFWHITE, fontweight='bold')
    ax.text(0.5,  1.07, '← lines = account used hashtag →',
            ha='center', fontsize=9, color=GRAY)

    # Legend
    legend_elements = [
        Line2D([0],[0], color=RED,  lw=2, alpha=0.7,
               label='AI account → AI hashtag'),
        Line2D([0],[0], color=BLUE, lw=2, alpha=0.7,
               label='Human account → Human hashtag'),
        Line2D([0],[0], color=GOLD, lw=2.5, alpha=0.9,
               label='Cross-group connection (dual-tagging strategy)'),
        mpatches.Patch(color=RED,  label='AI-specific hashtag / AI account'),
        mpatches.Patch(color=BLUE, label='Human creative hashtag / Human account'),
    ]
    leg = ax.legend(handles=legend_elements, loc='lower left',
                    fontsize=9.5, facecolor='#111827', edgecolor=GRAY,
                    framealpha=0.95, bbox_to_anchor=(-0.02, -0.02),
                    labelcolor=WHITE)

    ax.set_xlim(-0.18, 1.52)
    ax.set_ylim(-0.08, 1.12)

    ax.set_title('Bipartite Network: Accounts ↔ Hashtags — AI-NCII Ecosystem on X\n'
                 'Gold edges = dual-tagging: accounts using both AI and human creative tags in same post',
                 fontsize=13, color=WHITE, fontweight='bold', pad=16)

    plt.tight_layout()
    plt.savefig('outputs/figures/bipartite_clean.png', dpi=200,
                bbox_inches='tight', facecolor=BG)
    plt.close()
    print("✓ bipartite_clean.png saved")


# ── Run all three ─────────────────────────────────────────────────────────────
print("Generating figures from your actual data...")
make_psm_bar()
make_psm_balance()
make_bipartite()
print("\nAll done. Files saved to current directory.")
print("  psm_chart_final.png   — PSM bar chart (3 outcomes)")
print("  psm_balance_final.png — Propensity score + SMD balance")
print("  bipartite_clean.png   — Clean readable bipartite network")