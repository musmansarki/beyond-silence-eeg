#!/usr/bin/env python3
"""
Statistical Analysis — v3
==========================

"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import os
import sys

# ── Configuration ─────────────────────────────────────────────────────────────
FEATURES_CSV = ('/Users/momosarki/Documents/eeg_analysis/features/'
                'eeg_features_v4_withP12_20260802_101238.csv')
OUTPUT_FOLDER = '/Users/momosarki/Documents/eeg_analysis/results'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CONDITION_ORDER = [
    'Silence', 'White noise', 'Lofi instrumental',
    'Unfamiliar lyrical', 'Familiar lyrical',
]

QUAD_WEIGHTS = np.array([-2, 1, 2, 1, -2], dtype=float)

# Participants excluded from EEG analyses only. Behavioural data retained.
# P2: visual inspection of the independent component time courses revealed
# a large, regularly recurring transient artifact present across all
# components throughout the experimental blocks.
EEG_EXCLUDED_PARTICIPANTS = ['2']

# Dependent variables: (column, label, flag column)
DVS = [
    ('recall_accuracy', 'Recall accuracy', None),
    ('tbr_sub',         'Frontal TBR (baseline-corrected)', 'tbr_flags'),
    ('beta_db',         'Posterior beta (dB re baseline)',  'beta_flags'),
]

# For the baseline-correction comparison
DV_PAIRS = [
    ('tbr_encoding', 'tbr_sub', 'Frontal TBR', 'tbr_flags'),
    ('beta_encoding', 'beta_db', 'Posterior beta', 'beta_flags'),
]


def sep(title):
    print()
    print("=" * 68)
    print("  " + title)
    print("=" * 68)


# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading", os.path.basename(FEATURES_CSV))
df = pd.read_csv(FEATURES_CSV, float_precision='round_trip')
print(f"  {len(df)} rows, {df['participant_id'].nunique()} participants")

df['participant_id'] = df['participant_id'].astype(str)

for col in ['recall_accuracy', 'tbr_encoding', 'tbr_sub', 'tbr_ratio',
            'beta_encoding', 'beta_sub', 'beta_db', 'focus_rating',
            'enc_muscle_pct']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df['condition'] = pd.Categorical(df['condition'],
                                 categories=CONDITION_ORDER, ordered=True)


def subset(dv, flag_col, eeg_measure=True):
    """
    Return the analysable rows for one dependent variable.

    Always requires block_flags == 'OK'. EEG measures additionally require
    their own flag column to be 'OK' and exclude EEG-excluded participants.
    """
    d = df[df['block_flags'] == 'OK'].copy()
    if eeg_measure:
        d = d[~d['participant_id'].isin(EEG_EXCLUDED_PARTICIPANTS)]
        if flag_col:
            d = d[d[flag_col] == 'OK']
    d = d[d['condition'].notna()]
    d = d.dropna(subset=[dv])
    return d


# ── Sample sizes ──────────────────────────────────────────────────────────────
sep("ANALYSABLE SAMPLE PER DEPENDENT VARIABLE")

print(f"  EEG-excluded participants: "
      f"{', '.join(EEG_EXCLUDED_PARTICIPANTS) or 'none'}")
print()
print(f"  {'Measure':<38} {'blocks':>7} {'participants':>13} "
      f"{'High':>5} {'Low':>5}")
print("  " + "-" * 72)

for dv, label, flag in DVS:
    d = subset(dv, flag, eeg_measure=(flag is not None))
    hi = d[d['adhd_group'] == 'High']['participant_id'].nunique()
    lo = d[d['adhd_group'] == 'Low']['participant_id'].nunique()
    print(f"  {label:<38} {len(d):>7} {d['participant_id'].nunique():>13} "
          f"{hi:>5} {lo:>5}")

sep("BLOCKS PER GROUP x CONDITION")
for dv, label, flag in DVS:
    d = subset(dv, flag, eeg_measure=(flag is not None))
    print(f"\n  {label}")
    tab = (d.groupby(['adhd_group', 'condition'], observed=True)
             ['participant_id'].count().unstack(fill_value=0))
    print(tab.to_string().replace('\n', '\n    '))


# ── Linear mixed models ───────────────────────────────────────────────────────
def r_squared(model, data, dv):
    """
    Marginal and conditional R^2 following Nakagawa & Schielzeth (2013).

    Marginal R^2 is the proportion of total variance explained by the
    ENTIRE fixed-effects structure. It cannot be attributed to individual
    terms, and is not comparable across dependent variables measured on
    different scales.
    """
    var_random = float(model.cov_re.iloc[0, 0])
    var_resid = float(model.scale)
    re = {p: float(v.iloc[0]) for p, v in model.random_effects.items()}
    re_vals = np.array([re.get(p, 0.0) for p in data['participant_id'].values])
    fixed_pred = model.fittedvalues.values - re_vals
    var_fixed = float(np.var(fixed_pred, ddof=1))
    total = var_fixed + var_random + var_resid
    if total == 0:
        return np.nan, np.nan, (var_fixed, var_random, var_resid)
    return (var_fixed / total,
            (var_fixed + var_random) / total,
            (var_fixed, var_random, var_resid))


def fit_lmm(dv, label, flag, verbose=True):
    d = subset(dv, flag, eeg_measure=(flag is not None))
    d = d[['participant_id', 'adhd_group', 'condition', dv]].copy()
    d['cond'] = d['condition'].astype(str)

    if verbose:
        sep(f"LMM: {label}")
        print(f"  Blocks: {len(d)}   Participants: "
              f"{d['participant_id'].nunique()}")

    fits = {}
    for name, formula in [
        ('full',  f"{dv} ~ C(cond) * C(adhd_group)"),
        ('add',   f"{dv} ~ C(cond) + C(adhd_group)"),
        ('cond',  f"{dv} ~ C(cond)"),
        ('group', f"{dv} ~ C(adhd_group)"),
    ]:
        # Warnings are deliberately NOT suppressed. A ConvergenceWarning
        # here means the estimates below should not be trusted.
        fits[name] = smf.mixedlm(formula, d,
                                 groups=d['participant_id']).fit(
                                     reml=False, method='lbfgs')

    def lrt(reduced, full):
        chi2 = -2 * (fits[reduced].llf - fits[full].llf)
        ddf = fits[full].df_modelwc - fits[reduced].df_modelwc
        return chi2, ddf, stats.chi2.sf(chi2, ddf)

    res = {
        'Group':          lrt('cond', 'add'),
        'Condition':      lrt('group', 'add'),
        'Group x Cond':   lrt('add', 'full'),
    }

    if verbose:
        print()
        print(f"  {'Effect':<16} {'chi2':>8} {'df':>4} {'p':>8}")
        print("  " + "-" * 40)
        for name, (chi2, ddf, p) in res.items():
            print(f"  {name:<16} {chi2:>8.3f} {ddf:>4.0f} {p:>8.3f}")

    # R^2 from the REML fit of the full model
    m_reml = smf.mixedlm(f"{dv} ~ C(cond) * C(adhd_group)", d,
                         groups=d['participant_id']).fit(reml=True,
                                                         method='lbfgs')
    r2m, r2c, comps = r_squared(m_reml, d, dv)

    if verbose:
        print()
        print(f"  R2 marginal (all fixed effects jointly): {r2m:.3f}")
        print(f"  R2 conditional (fixed + random):         {r2c:.3f}")
        print(f"    var_fixed={comps[0]:.4g}  var_random={comps[1]:.4g}  "
              f"var_residual={comps[2]:.4g}")

    return {
        'dv': label, 'column': dv,
        'n_blocks': len(d), 'n_participants': d['participant_id'].nunique(),
        'group_chi2': res['Group'][0], 'group_df': res['Group'][1],
        'group_p': res['Group'][2],
        'cond_chi2': res['Condition'][0], 'cond_df': res['Condition'][1],
        'cond_p': res['Condition'][2],
        'int_chi2': res['Group x Cond'][0], 'int_df': res['Group x Cond'][1],
        'int_p': res['Group x Cond'][2],
        'r2_marginal': r2m, 'r2_conditional': r2c,
    }


results = []
for dv, label, flag in DVS:
    try:
        results.append(fit_lmm(dv, label, flag))
    except Exception as e:
        print(f"\n  MODEL FAILED for {label}: {e}")
        import traceback
        traceback.print_exc()


# ── Effect of baseline correction on variance decomposition ───────────────────
sep("EFFECT OF BASELINE CORRECTION ON R-SQUARED")
print("  Comparing un-baselined and baseline-corrected versions of the")
print("  same measure. Baseline correction should reduce conditional R2 by")
print("  removing stable between-participant differences in absolute power.")
print()
print(f"  {'Measure':<28} {'version':<16} {'R2m':>7} {'R2c':>7}")
print("  " + "-" * 62)

for raw_col, corr_col, label, flag in DV_PAIRS:
    for col, vname in [(raw_col, 'un-baselined'), (corr_col, 'corrected')]:
        d = subset(col, flag, eeg_measure=True)
        d = d[['participant_id', 'adhd_group', 'condition', col]].copy()
        d['cond'] = d['condition'].astype(str)
        try:
            m = smf.mixedlm(f"{col} ~ C(cond) * C(adhd_group)", d,
                            groups=d['participant_id']).fit(reml=True,
                                                            method='lbfgs')
            r2m, r2c, _ = r_squared(m, d, col)
            print(f"  {label:<28} {vname:<16} {r2m:>7.3f} {r2c:>7.3f}")
        except Exception as e:
            print(f"  {label:<28} {vname:<16}   FAILED: {e}")


# ── Quadratic trend ───────────────────────────────────────────────────────────
sep("QUADRATIC TREND (per-participant contrast scores)")
print("  Weights (-2, +1, +2, +1, -2) across the five ordered conditions.")
print("  Requires clean data in all five conditions, so N is smaller than")
print("  for the mixed models and differs between dependent variables.")

for dv, label, flag in DVS:
    d = subset(dv, flag, eeg_measure=(flag is not None))
    print(f"\n  {label}")
    for group in ['High', 'Low']:
        sub = d[d['adhd_group'] == group]
        piv = sub.pivot_table(index='participant_id', columns='condition',
                              values=dv, aggfunc='mean', observed=True)
        cols = [c for c in CONDITION_ORDER if c in piv.columns]
        if len(cols) < 5:
            print(f"    {group}: only {len(cols)} conditions present — skipped")
            continue
        piv = piv[cols].dropna()
        if len(piv) < 3:
            print(f"    {group}: only {len(piv)} complete participants — skipped")
            continue
        contrast = piv.values @ QUAD_WEIGHTS
        t, p = stats.ttest_1samp(contrast, 0)
        dz = contrast.mean() / contrast.std(ddof=1)
        ci = stats.t.interval(0.95, len(contrast) - 1,
                              loc=contrast.mean(),
                              scale=stats.sem(contrast))
        print(f"    {group}: t({len(piv)-1}) = {t:.3f}, p = {p:.3f}, "
              f"dz = {dz:.3f}, n = {len(piv)}")
        print(f"           mean contrast = {contrast.mean():.3f} "
              f"[{ci[0]:.3f}, {ci[1]:.3f}]")


# ── Descriptives with confidence intervals ────────────────────────────────────
sep("DESCRIPTIVE STATISTICS")
print("  Means with 95% confidence intervals. Wide intervals relative to")
print("  between-condition differences indicate the study cannot resolve")
print("  effects of the size hypothesised.")

desc_rows = []
for dv, label, flag in DVS:
    d = subset(dv, flag, eeg_measure=(flag is not None))
    print(f"\n  {label}")
    print(f"  {'Condition':<22} {'High: M [95% CI]':>28} "
          f"{'n':>3}   {'Low: M [95% CI]':>28} {'n':>3}")
    print("  " + "-" * 92)
    for cond in CONDITION_ORDER:
        cells = []
        for group in ['High', 'Low']:
            v = d[(d['condition'] == cond) &
                  (d['adhd_group'] == group)][dv].dropna()
            if len(v) >= 2:
                ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(),
                                      scale=stats.sem(v))
                cells.append((f"{v.mean():.3f} [{ci[0]:.3f}, {ci[1]:.3f}]",
                              len(v), v.mean(), v.std(), ci))
            elif len(v) == 1:
                cells.append((f"{v.mean():.3f} [--, --]", 1,
                              v.mean(), np.nan, (np.nan, np.nan)))
            else:
                cells.append(("n/a", 0, np.nan, np.nan, (np.nan, np.nan)))
        print(f"  {cond:<22} {cells[0][0]:>28} {cells[0][1]:>3}   "
              f"{cells[1][0]:>28} {cells[1][1]:>3}")
        for group, c in zip(['High', 'Low'], cells):
            desc_rows.append({
                'dv': label, 'condition': cond, 'group': group,
                'mean': c[2], 'sd': c[3], 'n': c[1],
                'ci_low': c[4][0], 'ci_high': c[4][1],
            })


# ── Exploratory: TBR and recall ───────────────────────────────────────────────
sep("EXPLORATORY: TBR x RECALL CORRELATIONS")
print("  Not pre-specified. Per-cell n is small; interpret with caution.")
print()

d_tbr = subset('tbr_sub', 'tbr_flags')
print(f"  {'Condition':<22} {'High: r (p) n':>26} {'Low: r (p) n':>26}")
print("  " + "-" * 76)
for cond in CONDITION_ORDER:
    out = []
    for group in ['High', 'Low']:
        s = d_tbr[(d_tbr['condition'] == cond) &
                  (d_tbr['adhd_group'] == group)][
                      ['tbr_sub', 'recall_accuracy']].dropna()
        if len(s) >= 4:
            r, p = stats.pearsonr(s['tbr_sub'], s['recall_accuracy'])
            out.append(f"{r:+.3f} ({p:.3f}) n={len(s)}")
        else:
            out.append(f"n={len(s)} — too few")
    print(f"  {cond:<22} {out[0]:>26} {out[1]:>26}")


# ── Muscle contamination by condition ─────────────────────────────────────────
sep("MUSCLE CONTAMINATION BY CONDITION")
print("  EMG inflates beta power, the denominator of TBR. If contamination")
print("  differed systematically across conditions it would act as a")
print("  confound on the primary dependent variable.")
print()

if 'enc_muscle_pct' in df.columns:
    d = df[(df['block_flags'] == 'OK') &
           (~df['participant_id'].isin(EEG_EXCLUDED_PARTICIPANTS))]
    tab = d.groupby('condition', observed=True)['enc_muscle_pct'].agg(
        ['mean', 'std', 'count'])
    print(tab.round(3).to_string().replace('\n', '\n  '))

    groups = [g['enc_muscle_pct'].dropna().values
              for _, g in d.groupby('condition', observed=True)]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) >= 2:
        F, p = stats.f_oneway(*groups)
        print(f"\n  One-way ANOVA across conditions: "
              f"F = {F:.3f}, p = {p:.3f}")
        if p < .05:
            print("  WARNING: muscle contamination differs across conditions.")
            print("  This is a potential confound on TBR and must be reported.")
        else:
            print("  No evidence that contamination differed across conditions.")


# ── Save ──────────────────────────────────────────────────────────────────────
if results:
    pd.DataFrame(results).to_csv(
        os.path.join(OUTPUT_FOLDER, 'lmm_results_v3.csv'), index=False)
    print(f"\n\nLMM results -> {OUTPUT_FOLDER}/lmm_results_v3.csv")

if desc_rows:
    pd.DataFrame(desc_rows).to_csv(
        os.path.join(OUTPUT_FOLDER, 'descriptives_v3.csv'), index=False)
    print(f"Descriptives -> {OUTPUT_FOLDER}/descriptives_v3.csv")

print("\nDone.")
print("\nNOTE: any ConvergenceWarning printed above means that model did not")
print("fit properly and its estimates should not be reported without")
print("investigating. Warnings are shown deliberately.")