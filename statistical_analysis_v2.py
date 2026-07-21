#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import warnings
import os
warnings.filterwarnings('ignore')

print("Script started")

FEATURES_CSV  = '/Users/momosarki/Documents/eeg_analysis/features/eeg_features_20260623_131800.csv'
OUTPUT_FOLDER = '/Users/momosarki/Documents/eeg_analysis/results'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CONDITION_ORDER = [
    'Silence',
    'White noise',
    'Lofi instrumental',
    'Unfamiliar lyrical',
    'Familiar lyrical',
]

QUAD_WEIGHTS = np.array([-2, 1, 2, 1, -2], dtype=float)

print("Loading data...")
df = pd.read_csv(FEATURES_CSV, float_precision='round_trip')
print("  Total rows loaded:", len(df))

df_clean = df[df['outlier_flags'] == 'OK'].copy()
print("  After outlier exclusion:", len(df_clean))

df_clean = df_clean[df_clean['adhd_group'] != 'Unknown'].copy()
print("  After removing Unknown group:", len(df_clean))

df_clean = df_clean[~df_clean['condition'].str.startswith('Block_')].copy()
print("  After removing unlabelled conditions:", len(df_clean))

for col in ['recall_accuracy', 'tbr_encoding', 'posterior_beta_encoding']:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

df_clean['log_beta'] = np.log10(
    df_clean['posterior_beta_encoding'].replace(0, np.nan)
)

df_clean['condition'] = pd.Categorical(
    df_clean['condition'], categories=CONDITION_ORDER, ordered=True
)

print()
print("Participants in clean data:", df_clean['participant_id'].nunique())
print("High ADHD:", df_clean[df_clean['adhd_group']=='High']['participant_id'].nunique())
print("Low ADHD: ", df_clean[df_clean['adhd_group']=='Low']['participant_id'].nunique())
print()
print("Block counts per group and condition:")
print(df_clean.groupby(['adhd_group','condition'])['participant_id'].count().to_string())

def sep(title):
    print()
    print("="*65)
    print("  " + title)
    print("="*65)

def run_lmm(dv, label):
    sep("LMM: " + label)
    sub = df_clean[['participant_id','adhd_group','condition', dv]].dropna().copy()
    sub['condition_cat'] = sub['condition'].astype(str)
    print("  N participants:", sub['participant_id'].nunique())
    print("  N blocks:", len(sub))
    print("  High:", sub[sub['adhd_group']=='High']['participant_id'].nunique())
    print("  Low: ", sub[sub['adhd_group']=='Low']['participant_id'].nunique())

    try:
        m_full = smf.mixedlm(
            dv + " ~ C(condition_cat) * C(adhd_group)",
            sub, groups=sub['participant_id']
        ).fit(reml=False, method='lbfgs')

        m_add = smf.mixedlm(
            dv + " ~ C(condition_cat) + C(adhd_group)",
            sub, groups=sub['participant_id']
        ).fit(reml=False, method='lbfgs')

        m_cond = smf.mixedlm(
            dv + " ~ C(condition_cat)",
            sub, groups=sub['participant_id']
        ).fit(reml=False, method='lbfgs')

        m_group = smf.mixedlm(
            dv + " ~ C(adhd_group)",
            sub, groups=sub['participant_id']
        ).fit(reml=False, method='lbfgs')

        def lrt(m_red, m_full):
            chi2 = -2 * (m_red.llf - m_full.llf)
            df_  = m_full.df_modelwc - m_red.df_modelwc
            p    = stats.chi2.sf(chi2, df_)
            return chi2, df_, p

        chi2_grp,  df_grp,  p_grp  = lrt(m_cond,  m_add)
        chi2_cond, df_cond, p_cond = lrt(m_group, m_add)
        chi2_int,  df_int,  p_int  = lrt(m_add,   m_full)

        print()
        print("  Effect              chi2      df     p       sig")
        print("  " + "-"*55)
        for name, chi2, df_, p in [
            ('Group',        chi2_grp,  df_grp,  p_grp),
            ('Condition',    chi2_cond, df_cond, p_cond),
            ('Group x Cond', chi2_int,  df_int,  p_int),
        ]:
            sig = '*' if p < .05 else ('+' if p < .10 else 'ns')
            print("  {:<20} {:>7.3f}  {:>4.0f}  {:>7.3f}  {}".format(
                name, chi2, df_, p, sig))

        return {
            'dv': label,
            'n_part': sub['participant_id'].nunique(),
            'n_blocks': len(sub),
            'grp_chi2': round(chi2_grp,3), 'grp_df': df_grp, 'grp_p': round(p_grp,3),
            'cond_chi2': round(chi2_cond,3), 'cond_df': df_cond, 'cond_p': round(p_cond,3),
            'int_chi2': round(chi2_int,3), 'int_df': df_int, 'int_p': round(p_int,3),
        }

    except Exception as e:
        print("  ERROR:", e)
        import traceback
        traceback.print_exc()
        return None

r1 = run_lmm('recall_accuracy', 'Recall Accuracy')
r2 = run_lmm('tbr_encoding',    'Frontal TBR')
r3 = run_lmm('log_beta',        'Log10 Posterior Beta')

sep("QUADRATIC TREND (CORRECTED: per-participant contrast scores)")
print("  Weights: [-2, +1, +2, +1, -2]")
print("  One contrast score per participant -> t(N-1)")
print()

for group in ['High', 'Low']:
    print("  --- " + group + " ADHD ---")
    for dv, label in [
        ('recall_accuracy', 'Recall'),
        ('tbr_encoding',    'TBR'),
        ('log_beta',        'Log Beta'),
    ]:
        sub = df_clean[df_clean['adhd_group'] == group]
        pivot = sub.pivot_table(
            index='participant_id',
            columns='condition',
            values=dv,
            aggfunc='mean'
        )
        cols = [c for c in CONDITION_ORDER if c in pivot.columns]
        if len(cols) < 5:
            print("    " + label + ": only " + str(len(cols)) + " conditions - skip")
            continue
        pivot = pivot[cols].dropna()
        if len(pivot) < 3:
            print("    " + label + ": only " + str(len(pivot)) + " participants - skip")
            continue
        contrast = pivot.values @ QUAD_WEIGHTS
        t, p = stats.ttest_1samp(contrast, 0)
        d = contrast.mean() / contrast.std(ddof=1)
        sig = '*' if p < .05 else ('+' if p < .10 else 'ns')
        print("    {}: t({}) = {:.3f}, p = {:.3f} {}, d = {:.3f}, n = {}".format(
            label, len(pivot)-1, t, p, sig, d, len(pivot)))
    print()

sep("EXPLORATORY: TBR x RECALL CORRELATIONS")
print()
print("  {:<22} {:>16} {:>16}".format("Condition", "High r (p)", "Low r (p)"))
print("  " + "-"*56)
for cond in CONDITION_ORDER:
    row = []
    for group in ['High', 'Low']:
        sub = df_clean[
            (df_clean['condition'] == cond) &
            (df_clean['adhd_group'] == group)
        ][['tbr_encoding', 'recall_accuracy']].dropna()
        if len(sub) >= 4:
            r, p = stats.pearsonr(sub['tbr_encoding'], sub['recall_accuracy'])
            sig = '*' if p < .05 else ('+' if p < .10 else '')
            row.append("r={:+.3f} p={:.3f}{}".format(r, p, sig))
        else:
            row.append('n<4')
    print("  {:<22} {:>16} {:>16}".format(cond, row[0], row[1]))

sep("DESCRIPTIVE MEANS")
for dv, label in [
    ('recall_accuracy', 'RECALL'),
    ('tbr_encoding',    'TBR'),
    ('log_beta',        'LOG BETA'),
]:
    print()
    print("  " + label)
    print("  {:<22} {:>22} {:>22}".format("Condition", "High M(SD) n", "Low M(SD) n"))
    print("  " + "-"*68)
    for cond in CONDITION_ORDER:
        hi = df_clean[
            (df_clean['condition']==cond) &
            (df_clean['adhd_group']=='High')
        ][dv].dropna()
        lo = df_clean[
            (df_clean['condition']==cond) &
            (df_clean['adhd_group']=='Low')
        ][dv].dropna()
        hs = "{:.3f}({:.3f}) n={}".format(hi.mean(), hi.std(), len(hi)) if len(hi) else 'N/A'
        ls = "{:.3f}({:.3f}) n={}".format(lo.mean(), lo.std(), len(lo)) if len(lo) else 'N/A'
        print("  {:<22} {:>22} {:>22}".format(cond, hs, ls))

results = [r for r in [r1, r2, r3] if r is not None]
if results:
    pd.DataFrame(results).to_csv(
        os.path.join(OUTPUT_FOLDER, 'lmm_results.csv'), index=False
    )
    print()
    print("LMM results saved to results/lmm_results.csv")

print()
print("Done.")

# ── R-squared for LMMs (Nakagawa & Schieleth 2013) ───────────────────────────
# Added at end to avoid modifying the main run_lmm function
# Marginal R2 = variance explained by fixed effects only
# Conditional R2 = variance explained by fixed + random effects

print()
print("="*65)
print("  R-SQUARED EFFECT SIZES (Nakagawa & Schieleth, 2013)")
print("="*65)
print("  Marginal R2 = fixed effects only")
print("  Conditional R2 = fixed + random effects")
print()

for dv, label in [
    ('recall_accuracy', 'Recall Accuracy'),
    ('tbr_encoding',    'Frontal TBR'),
    ('log_beta',        'Log10 Posterior Beta'),
]:
    sub = df_clean[['participant_id','adhd_group','condition', dv]].dropna().copy()
    sub['condition_cat'] = sub['condition'].astype(str)

    try:
        # Full model
        m_full = smf.mixedlm(
            dv + " ~ C(condition_cat) * C(adhd_group)",
            sub, groups=sub['participant_id']
        ).fit(reml=True, method='lbfgs')

        # Variance components
        var_random   = float(m_full.cov_re.iloc[0, 0])
        var_residual = float(m_full.scale)

        # Fixed-effect predictions: fitted - random intercept per participant
        # random_effects is dict {participant_id: array([intercept])}
        re_dict = {pid: float(vals.iloc[0])
                   for pid, vals in m_full.random_effects.items()}
        re_vals = np.array([re_dict.get(pid, 0.0)
                            for pid in sub['participant_id'].values])
        fixed_pred = m_full.fittedvalues.values - re_vals
        var_fixed = float(np.var(fixed_pred, ddof=1))

        total = var_fixed + var_random + var_residual
        if total == 0:
            print("  {}: total variance is zero - cannot compute R2".format(label))
            continue

        r2_marginal    = var_fixed / total
        r2_conditional = (var_fixed + var_random) / total

        print("  {:<25} R2m = {:.3f}   R2c = {:.3f}".format(
            label, r2_marginal, r2_conditional))
        print("    var_fixed={:.4f}  var_random={:.4f}  var_residual={:.4f}".format(
            var_fixed, var_random, var_residual))

    except Exception as e:
        import traceback
        print("  {}: ERROR - {}".format(label, e))
        traceback.print_exc()