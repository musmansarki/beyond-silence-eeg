#!/usr/bin/env python3
"""
Figures for the thesis
=======================

"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # no display needed
import matplotlib.pyplot as plt
from scipy import stats
import os

FEATURES_CSV = ('/Users/momosarki/Documents/eeg_analysis/features/'
                'eeg_features_v4_withP12_20260802_101238.csv')
OUTPUT_FOLDER = '/Users/momosarki/Documents/eeg_analysis/figures'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CONDITION_ORDER = ['Silence', 'White noise', 'Lofi instrumental',
                   'Unfamiliar lyrical', 'Familiar lyrical']
SHORT_LABELS = ['Silence', 'White\nnoise', 'Lofi\ninstr.',
                'Unfam.\nlyrical', 'Familiar\nlyrical']

EEG_EXCLUDED = ['2']

# Colour-blind safe, and distinguishable in greyscale by marker shape
C_HIGH, C_LOW = '#0173B2', '#DE8F05'
M_HIGH, M_LOW = 'o', 's'

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})


# ── Load and filter ───────────────────────────────────────────────────────────
df = pd.read_csv(FEATURES_CSV, float_precision='round_trip')
df['participant_id'] = df['participant_id'].astype(str)

for col in ['recall_accuracy', 'tbr_sub', 'beta_db']:
    df[col] = pd.to_numeric(df[col], errors='coerce')


def subset(dv, flag_col, eeg=True):
    d = df[df['block_flags'] == 'OK'].copy()
    if eeg:
        d = d[~d['participant_id'].isin(EEG_EXCLUDED)]
        if flag_col:
            d = d[d[flag_col] == 'OK']
    d = d[d['condition'].isin(CONDITION_ORDER)]
    return d.dropna(subset=[dv])


def mean_ci(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan, 0
    if len(v) == 1:
        return v[0], np.nan, np.nan, 1
    m = v.mean()
    lo, hi = stats.t.interval(0.95, len(v) - 1, loc=m, scale=stats.sem(v))
    return m, lo, hi, len(v)


# ── Grouped means figure ──────────────────────────────────────────────────────
def plot_means(dv, flag, ylabel, title, filename, chance=None,
               zero_line=False):
    d = subset(dv, flag, eeg=(flag is not None))

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    x = np.arange(len(CONDITION_ORDER))
    offset = 0.09

    for group, colour, marker, dx in [('High', C_HIGH, M_HIGH, -offset),
                                      ('Low', C_LOW, M_LOW, +offset)]:
        means, los, his, ns = [], [], [], []
        for cond in CONDITION_ORDER:
            v = d[(d['condition'] == cond) &
                  (d['adhd_group'] == group)][dv]
            m, lo, hi, n = mean_ci(v)
            means.append(m)
            los.append(m - lo if not np.isnan(lo) else 0)
            his.append(hi - m if not np.isnan(hi) else 0)
            ns.append(n)

        ax.errorbar(x + dx, means, yerr=[los, his],
                    marker=marker, markersize=5, capsize=3, capthick=1,
                    linewidth=1.4, elinewidth=1, color=colour,
                    label=f'High ADHD (n={max(ns)})' if group == 'High'
                          else f'Low ADHD (n={max(ns)})',
                    zorder=3)

        # per-cell n beneath each point
        ylim_lo = ax.get_ylim()[0]
        for xi, n in zip(x + dx, ns):
            ax.annotate(str(n), (xi, ylim_lo), xytext=(0, 2),
                        textcoords='offset points', ha='center',
                        fontsize=6, color=colour, alpha=0.75)

    if chance is not None:
        ax.axhline(chance, color='grey', linestyle=':', linewidth=1,
                   zorder=1)
        ax.annotate('chance', (len(CONDITION_ORDER) - 0.55, chance),
                    xytext=(0, 3), textcoords='offset points',
                    fontsize=7, color='grey')
    if zero_line:
        ax.axhline(0, color='grey', linestyle=':', linewidth=1, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABELS)
    ax.set_xlabel('Auditory condition')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc='best')
    ax.set_xlim(-0.5, len(CONDITION_ORDER) - 0.5)

    fig.tight_layout()
    path = os.path.join(OUTPUT_FOLDER, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  {filename}")
    return d


# ── Individual trajectories ───────────────────────────────────────────────────
def plot_individual(filename='fig3_individual.pdf'):
    specs = [
        ('recall_accuracy', None, 'Recall accuracy', 'Recall accuracy', 0.25),
        ('tbr_sub', 'tbr_flags', 'TBR change from baseline',
         'Frontal TBR', None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.6), sharex=True)

    for row, (dv, flag, ylabel, title, chance) in enumerate(specs):
        d = subset(dv, flag, eeg=(flag is not None))
        for col, group in enumerate(['High', 'Low']):
            ax = axes[row, col]
            g = d[d['adhd_group'] == group]
            colour = C_HIGH if group == 'High' else C_LOW

            for pid, pdata in g.groupby('participant_id'):
                pdata = pdata.set_index('condition').reindex(CONDITION_ORDER)
                y = pdata[dv].values
                mask = ~np.isnan(y)
                ax.plot(np.arange(5)[mask], y[mask],
                        marker='o', markersize=2.5, linewidth=0.8,
                        color=colour, alpha=0.45, zorder=2)

            means = [g[g['condition'] == c][dv].mean()
                     for c in CONDITION_ORDER]
            ax.plot(np.arange(5), means, marker='o', markersize=5,
                    linewidth=2.2, color='black', zorder=4,
                    label='group mean')

            if chance is not None:
                ax.axhline(chance, color='grey', linestyle=':',
                           linewidth=1, zorder=1)
            if dv == 'tbr_sub':
                ax.axhline(0, color='grey', linestyle=':',
                           linewidth=1, zorder=1)

            n = g['participant_id'].nunique()
            ax.set_title(f'{title} — {group} ADHD (n={n})', fontsize=9)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xticks(np.arange(5))
                ax.set_xticklabels(SHORT_LABELS)
                ax.set_xlabel('Auditory condition')
            if row == 0 and col == 0:
                ax.legend(frameon=False, loc='best')

    # shared y-limits within each row for honest comparison
    for row in range(2):
        lo = min(axes[row, c].get_ylim()[0] for c in range(2))
        hi = max(axes[row, c].get_ylim()[1] for c in range(2))
        for c in range(2):
            axes[row, c].set_ylim(lo, hi)

    fig.tight_layout()
    path = os.path.join(OUTPUT_FOLDER, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  {filename}")


# ── Generate ──────────────────────────────────────────────────────────────────
print(f"Writing figures to {OUTPUT_FOLDER}")

plot_means('recall_accuracy', None,
           'Recall accuracy (proportion)',
           'Recall accuracy by condition and ADHD group',
           'fig1_recall.pdf', chance=0.25)

plot_means('tbr_sub', 'tbr_flags',
           'TBR change from baseline',
           'Baseline-corrected frontal TBR by condition and group',
           'fig2_tbr.pdf', zero_line=True)

plot_individual()

plot_means('beta_db', 'beta_flags',
           'Posterior beta change (dB re baseline)',
           'Posterior beta power by condition and ADHD group',
           'fig4_beta.pdf', zero_line=True)

print("\nError bars are 95% confidence intervals.")
print("Small numbers beneath each point are the per-cell block counts.")