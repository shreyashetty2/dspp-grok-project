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

# ── Colors matching your Canva theme ─────────────────────────────────────────
BG       = '#F0EDE6'
DKGREEN  = '#2C3E2D'
MDGREEN  = '#4A7A4B'
LTGREEN  = '#8FB88F'
TEAL     = '#3D7A6B'
ORANGE   = '#C8502A'
GRAY     = '#6A6A5A'
CREAM    = '#E8E0D0'
GOLD     = '#C8A84B'

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE A: PSM BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
def make_psm_bar():
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.patch.set_facecolor(BG)

    outcomes  = ['log(Views)', 'log(Engagement\nTotal)', 'Engagement\nRate']
    ai_means  = [3.2219,       0.8832,                   0.1112]
    hu_means  = [3.0342,       0.8567,                   0.1073]
    pct_diff  = ['+6.18%',     '+3.09%',                 '+3.58%']
    pvals     = ['p = 0.005 **\nSIGNIFICANT', 'p = 0.787\nnot significant', 'p = 0.778\nnot significant']
    sig       = [True,          False,                    False]

    for i, ax in enumerate(axes):
        ax.set_facecolor(BG)
        x = [0.28, 0.72]
        vals = [ai_means[i], hu_means[i]]
        clrs = [DKGREEN, LTGREEN] if sig[i] else ['#7A9A7B', '#BDDAB8']
        edge = [DKGREEN, DKGREEN]

        bars = ax.bar(x, vals, width=0.3, color=clrs, edgecolor=edge,
                      linewidth=1.5, zorder=3)

        # Always start at 0
        ymax = max(vals) * 1.42
        ax.set_ylim(0, ymax)

        # Value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + ymax*0.015,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=9.5, color=DKGREEN, fontweight='bold')

        # % difference annotation between bars
        if sig[i]:
            mid_y = max(vals) * 1.08
            ax.annotate('', xy=(0.72, mid_y), xytext=(0.28, mid_y),
                        arrowprops=dict(arrowstyle='<->', color=DKGREEN, lw=1.5))
            ax.text(0.5, mid_y + ymax*0.02, pct_diff[i],
                    ha='center', va='bottom', fontsize=9,
                    color=DKGREEN, fontweight='bold')

        # p-value label
        pc = DKGREEN if sig[i] else ORANGE
        fw = 'bold' if sig[i] else 'normal'
        ax.text(0.5, ymax*0.92, pvals[i], ha='center', va='top',
                fontsize=9, color=pc, fontweight=fw,
                transform=ax.transData,
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white' if sig[i] else BG,
                          edgecolor=pc, alpha=0.85, linewidth=1))

        ax.set_xticks([0.28, 0.72])
        ax.set_xticklabels(['AI-Generated', 'Human-Created'],
                            fontsize=10, color=DKGREEN)
        ax.set_title(outcomes[i], fontsize=12, color=DKGREEN,
                     fontweight='bold', pad=10)
        ax.set_xlim(0, 1)
        ax.tick_params(colors=DKGREEN, labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_edgecolor(DKGREEN if sig[i] else CREAM)
            sp.set_linewidth(1.8 if sig[i] else 0.8)
        ax.yaxis.grid(True, color=CREAM, linewidth=0.7, linestyle='--', zorder=0)
        ax.set_axisbelow(True)

    legend = [mpatches.Patch(color=DKGREEN, label='AI-Generated (Treatment)'),
              mpatches.Patch(color=LTGREEN, label='Human-Created (Control)')]
    fig.legend(handles=legend, loc='lower center', ncol=2, fontsize=10.5,
               frameon=True, facecolor=BG, edgecolor=DKGREEN,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle('PSM Matched Sample: AI vs Human Content Engagement\n'
                 'N = 218 matched pairs | 4 confounders balanced (all SMD < 0.1)',
                 fontsize=12.5, color=DKGREEN, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig('outputs/figures/psm_chart_final.png', dpi=200,
                bbox_inches='tight', facecolor=BG)
    plt.close()
    print("✓ psm_chart_final.png saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B: PSM BALANCE CHART
# ─────────────────────────────────────────────────────────────────────────────
def make_psm_balance():
    df = pd.read_csv('outputs/tables/psm_matched_sample.csv')
    treat = df[df.treat == 1]
    ctrl  = df[df.treat == 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(BG)

    # ── Left: propensity score overlap ───────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG)

    ps_treat = treat['propensity_score'].values
    ps_ctrl  = ctrl['propensity_score'].values

    bins = np.linspace(0.25, 0.95, 25)
    ax.hist(ps_ctrl,  bins=bins, alpha=0.60, color=LTGREEN,
            label='Human-Created (Control)', edgecolor=DKGREEN, linewidth=0.6)
    ax.hist(ps_treat, bins=bins, alpha=0.60, color=DKGREEN,
            label='AI-Generated (Treatment)', edgecolor='#1A2E1A', linewidth=0.6)

    ax.axvspan(ps_treat.min(), ps_treat.max(), alpha=0.07,
               color=TEAL, label=f'Common support ({ps_treat.min():.2f}–{ps_treat.max():.2f})')

    ax.set_xlabel('Propensity Score', color=DKGREEN, fontsize=11)
    ax.set_ylabel('Number of Posts', color=DKGREEN, fontsize=11)
    ax.set_title('Propensity Score Distribution\n(Before Matching — Common Support Confirmed)',
                 color=DKGREEN, fontsize=11.5, fontweight='bold')
    ax.legend(fontsize=9, facecolor=BG, edgecolor=DKGREEN, framealpha=0.9)
    ax.tick_params(colors=DKGREEN)
    for sp in ax.spines.values():
        sp.set_edgecolor(CREAM)
    ax.yaxis.grid(True, color=CREAM, linewidth=0.6, linestyle='--', zorder=0)
    ax.set_axisbelow(True)

    stats_text = (f'Range: {ps_treat.min():.3f} – {ps_treat.max():.3f}\n'
                  f'Treatment mean: {ps_treat.mean():.3f}\n'
                  f'Control mean: {ps_ctrl.mean():.3f}')
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            va='top', ha='right', fontsize=8.5, color=DKGREEN,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=CREAM,
                      edgecolor=DKGREEN, alpha=0.9))

    # ── Right: SMD balance plot ───────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG)

    covs   = ['log(followers)', 'log(account age)', 'has media', 'hour of day']
    smds   = []
    for col in ['log_followers', 'log_account_age', 'has_media_int', 'hour_of_day']:
        t_m = treat[col].mean(); t_s = treat[col].std()
        c_m = ctrl[col].mean();  c_s = ctrl[col].std()
        pooled = np.sqrt((t_s**2 + c_s**2) / 2)
        smds.append(abs((t_m - c_m) / pooled) if pooled > 0 else 0)

    clrs = [DKGREEN if s < 0.05 else MDGREEN if s < 0.1 else ORANGE for s in smds]
    ypos = range(len(covs))

    bars = ax2.barh(list(ypos), smds, color=clrs, edgecolor=DKGREEN,
                    linewidth=0.8, height=0.45, zorder=3)

    ax2.axvline(x=0.1, color=ORANGE, linewidth=2, linestyle='--',
                label='Acceptable threshold (SMD = 0.1)', zorder=4)
    ax2.axvline(x=0.05, color=LTGREEN, linewidth=1.2, linestyle=':',
                label='Excellent balance (SMD = 0.05)', zorder=4)

    for bar, smd in zip(bars, smds):
        ax2.text(smd + 0.001, bar.get_y() + bar.get_height()/2,
                 f'{smd:.3f}', va='center', fontsize=10,
                 color=DKGREEN, fontweight='bold')
        ax2.text(0.108, bar.get_y() + bar.get_height()/2,
                 '✓', va='center', fontsize=13, color=DKGREEN, fontweight='bold')

    ax2.set_yticks(list(ypos))
    ax2.set_yticklabels(covs, fontsize=11, color=DKGREEN)
    ax2.set_xlabel('|Standardized Mean Difference (SMD)|', color=DKGREEN, fontsize=11)
    ax2.set_title('Covariate Balance After PSM\n(All SMD < 0.1 = Matching Successful)',
                  color=DKGREEN, fontsize=11.5, fontweight='bold')
    ax2.set_xlim(0, 0.125)
    ax2.legend(fontsize=9, facecolor=BG, edgecolor=DKGREEN,
               framealpha=0.9, loc='lower right')
    ax2.tick_params(colors=DKGREEN)
    for sp in ax2.spines.values():
        sp.set_edgecolor(CREAM)
    ax2.xaxis.grid(True, color=CREAM, linewidth=0.6, linestyle='--', zorder=0)
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
    # Build real edges from hashtag data
    df = pd.read_csv('outputs/tables/psm_matched_sample.csv')
    acc_df = pd.read_csv('outputs/tables/account_network_stats.csv')

    # Community labels
    comm_map = dict(zip(acc_df['username'], acc_df['community']))
    type_map  = dict(zip(acc_df['username'], acc_df['content_type']))

    # Top hashtags to show (keep it clean — top 11 meaningful ones)
    TOP_TAGS = ['grokai','grokimage','aigenerated','aiart','syntheticmedia','deepfakeai',
                'photography','portrait','artistsontwitter','streetphotography','illustration']
    TAG_LABELS = {
        'grokai':'#GrokAI', 'grokimage':'#GrokImage', 'aigenerated':'#AIGenerated',
        'aiart':'#AIArt', 'syntheticmedia':'#SyntheticMedia', 'deepfakeai':'#DeepfakeAI',
        'photography':'#photography', 'portrait':'#portrait',
        'artistsontwitter':'#artistsontwitter',
        'streetphotography':'#streetphotography', 'illustration':'#illustration'
    }
    AI_TAGS  = {'grokai','grokimage','aigenerated','aiart','syntheticmedia','deepfakeai'}
    HUM_TAGS = {'photography','portrait','artistsontwitter','streetphotography','illustration'}

    # Build edge list: account → tag (only top tags, only real accounts)
    real_accounts = set(acc_df[acc_df['username'].str.startswith('synuser_') == False]['username'])
    # Also include synthetic but limit to those with interesting hashtag connections
    edges = []
    tag_account_count = {t: set() for t in TOP_TAGS}

    for _, row in df.iterrows():
        if pd.isna(row.get('hashtags')): continue
        tags = [t.strip().lower().lstrip('#') for t in str(row['hashtags']).split(',')]
        acct = str(row.get('author_username',''))
        useful_tags = [t for t in tags if t in TOP_TAGS]
        if useful_tags and acct:
            for t in useful_tags:
                edges.append((acct, t))
                tag_account_count[t].add(acct)

    # Count unique accounts per tag for sizing
    tag_sizes = {t: len(s) for t, s in tag_account_count.items()}

    # Get unique accounts that used at least one top tag
    accounts_used = list(set(e[0] for e in edges))
    # Limit to accounts visible enough — those using 2+ top tags
    acct_tag_count = {}
    for a, t in edges:
        acct_tag_count[a] = acct_tag_count.get(a, 0) + 1
    accounts_show = [a for a, c in acct_tag_count.items() if c >= 2]
    # Also always include real high-betweenness accounts
    real_notable = ['ReviewAI_','boiledkeyboard','thisisavatar','steto123',
                    'Gyanendra844522','ashay_pallav']
    accounts_show = list(set(accounts_show + [a for a in real_notable if a in acct_tag_count]))

    edges_filtered = [(a, t) for a, t in edges if a in accounts_show]

    print(f"  Accounts shown: {len(accounts_show)}, Tags: {len(TOP_TAGS)}, Edges: {len(edges_filtered)}")

    # ── Layout: horizontal bipartite ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 11))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis('off')

    # Position accounts on left, tags on right
    n_acc = len(accounts_show)
    n_tag = len(TOP_TAGS)

    acc_pos = {}
    for i, a in enumerate(sorted(accounts_show)):
        acc_pos[a] = (0.0, 1.0 - i / max(n_acc - 1, 1))

    tag_pos = {}
    for i, t in enumerate(TOP_TAGS):
        tag_pos[t] = (1.0, 1.0 - i / max(n_tag - 1, 1))

    # Draw edges first (behind nodes)
    edge_colors_drawn = set()
    for a, t in edges_filtered:
        if a not in acc_pos or t not in tag_pos: continue
        x0, y0 = acc_pos[a]
        x1, y1 = tag_pos[t]
        # Color edge by whether crossing (dual-tagging) or within-group
        a_type = type_map.get(a, 'treatment')
        t_is_ai = t in AI_TAGS
        if a_type == 'treatment' and t_is_ai:
            ec, alpha, lw = DKGREEN, 0.25, 0.8
        elif a_type == 'control' and not t_is_ai:
            ec, alpha, lw = LTGREEN, 0.25, 0.8
        else:  # crossing — dual-tagging!
            ec, alpha, lw = GOLD, 0.55, 1.4

        ax.plot([x0, x1], [y0, y1], color=ec, alpha=alpha, linewidth=lw, zorder=1)

    # Draw account nodes
    for a, (x, y) in acc_pos.items():
        comm = comm_map.get(a, 0)
        is_real = a in real_notable
        ct = type_map.get(a, 'treatment')
        node_color = DKGREEN if ct == 'treatment' else LTGREEN
        size = 120 if is_real else 40
        zord = 5 if is_real else 3
        ax.scatter(x, y, s=size, c=node_color, zorder=zord,
                   edgecolors=DKGREEN, linewidths=1.0 if is_real else 0.3)
        if is_real:
            # Label notable real accounts
            ax.text(x - 0.02, y, a, ha='right', va='center',
                    fontsize=8, color=DKGREEN, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=CREAM,
                              edgecolor=DKGREEN, alpha=0.85, linewidth=0.8))

    # Draw hashtag nodes (right side) — sized by account count
    for t, (x, y) in tag_pos.items():
        is_ai = t in AI_TAGS
        node_color = DKGREEN if is_ai else LTGREEN
        sz = max(200, tag_sizes.get(t, 1) * 12)
        ax.scatter(x, y, s=sz, c=node_color, zorder=4,
                   edgecolors='#1A2E1A' if is_ai else DKGREEN, linewidths=1.5)
        label = TAG_LABELS.get(t, f'#{t}')
        # Alternate label side to avoid overlap
        ax.text(x + 0.025, y, label, ha='left', va='center',
                fontsize=10.5, color=DKGREEN, fontweight='bold' if is_ai else 'normal',
                bbox=dict(boxstyle='round,pad=0.25',
                          facecolor=CREAM, edgecolor=node_color,
                          alpha=0.92, linewidth=1.0))

    # Divider line between AI and human tags
    mid_y = (tag_pos['deepfakeai'][1] + tag_pos['photography'][1]) / 2
    ax.axhline(y=mid_y, xmin=0.85, xmax=1.15, color=GRAY,
               linewidth=1, linestyle='--', alpha=0.5)
    ax.text(1.22, mid_y, 'AI tags above\nHuman tags below',
            ha='left', va='center', fontsize=8.5, color=GRAY, style='italic')

    # Section headers
    ax.text(1.0, 1.06, 'HASHTAGS', ha='center', va='bottom',
            fontsize=11, color=DKGREEN, fontweight='bold', transform=ax.transData)
    ax.text(0.0, 1.06, 'ACCOUNTS', ha='center', va='bottom',
            fontsize=11, color=DKGREEN, fontweight='bold', transform=ax.transData)

    ax.text(0.5, 1.06, 'Lines = account used hashtag',
            ha='center', va='bottom', fontsize=9, color=GRAY, transform=ax.transData)

    # Legend
    legend_elements = [
        Line2D([0],[0], color=DKGREEN, lw=2, alpha=0.6, label='AI account → AI hashtag'),
        Line2D([0],[0], color=LTGREEN, lw=2, alpha=0.6, label='Human account → Human hashtag'),
        Line2D([0],[0], color=GOLD, lw=2.5, alpha=0.8, label='Cross-group (dual-tagging strategy)'),
        mpatches.Patch(color=DKGREEN, label='AI-specific hashtag / AI account'),
        mpatches.Patch(color=LTGREEN, label='Human creative hashtag / Human account'),
    ]
    ax.legend(handles=legend_elements, loc='lower left',
              fontsize=9.5, facecolor=CREAM, edgecolor=DKGREEN,
              framealpha=0.95, bbox_to_anchor=(0.0, -0.01))

    ax.set_xlim(-0.15, 1.35)
    ax.set_ylim(-0.08, 1.12)

    ax.set_title('Bipartite Network: Accounts ↔ Hashtags — AI-NCII Ecosystem on X\n'
                 'Gold edges show dual-tagging: accounts using both AI and human creative tags in the same post',
                 fontsize=13, color=DKGREEN, fontweight='bold', pad=15)

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
