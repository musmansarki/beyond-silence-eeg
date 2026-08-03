#!/usr/bin/env python3
"""
EEG Preprocessing and Feature Extraction — v4
==============================================


Reference note: an average reference is used rather than a mastoid
reference. EXG6 carries signal in only 6 of 20 recordings, and EXG5 is
inactive in two and abnormally noisy in two more, so no mastoid scheme
applies uniformly across participants.
"""

import mne
import numpy as np
import pandas as pd
import os
import csv
import glob
import json
from datetime import datetime
from mne.preprocessing import ICA, annotate_muscle_zscore

# ── Paths ─────────────────────────────────────────────────────────────────────
BDF_FOLDER    = '/Users/momosarki/Documents/eeg_analysis/eeg_data'
BLOCKS_CSV    = '/Users/momosarki/Documents/eeg_analysis/block_epochs/all_blocks_20260622_232517.csv'
BEHAV_FOLDER  = '/Users/momosarki/Documents/eeg_analysis/data'
OUTPUT_FOLDER = '/Users/momosarki/Documents/eeg_analysis/features'
ICA_FOLDER    = '/Users/momosarki/Documents/eeg_analysis/ica_solutions'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(ICA_FOLDER, exist_ok=True)

# ── Preprocessing settings ────────────────────────────────────────────────────
FILTER_LOW   = 1.0
FILTER_HIGH  = 40.0
REFERENCE    = 'average'

BAD_HIGH_RATIO   = 4.0
BAD_LOW_RATIO    = 0.25
MAX_BAD_FRACTION = 0.20

MUSCLE_THRESHOLD = 4.0
MUSCLE_FREQ      = (110, 140)
MUSCLE_MIN_GOOD  = 0.2

# ICA mode: 'off' | 'frontal_proxy' | 'apply_reviewed'
# 'apply_reviewed' reads ica_solutions/ica_exclusions.json, written by
# ica_review.py, and applies the manually selected components.
ICA_MODE = 'apply_reviewed'
ICA_N_COMPONENTS = 20

FRONTAL_CHANNELS   = ['F3', 'Fz', 'F4']
POSTERIOR_CHANNELS = ['P3', 'Pz', 'P4', 'PO3', 'PO4']
THETA_BAND = (4, 7)
BETA_BAND  = (13, 30)

# ── Exclusion thresholds ──────────────────────────────────────────────────────
# TBR
EXCL_TBR_MAX = 30.0
EXCL_TBR_MIN = 1.0          # beta > theta frontally is implausible

# Posterior beta. Set from the observed distribution (median 3.27e-13,
# 75th pct 3.84e-13, discontinuity between 2.4e-12 and 5.7e-12).
EXCL_BETA_MAX = 3e-12

# Block-level
EXCL_BASELINE_NOMINAL = 30.0
EXCL_BASELINE_TOL     = 2.0
EXCL_ENCODING_NOMINAL = 90.0
EXCL_ENCODING_TOL     = 3.0
EXCL_MUSCLE_PCT       = 25.0

NON_EEG = ['EXG1','EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8',
           'GSR1','GSR2','Erg1','Erg2','Resp','Plet','Temp']

ADHD_GROUPS = {
    '1':'High','2':'High','3':'Low','4':'High','5':'Low','6':'High',
    '7':'Low','8':'High','9':'High','10':'Low','11':'Low','12':'Low',
    '13':'High','14':'High','15':'High','16':'High','17':'Low',
    '18':'Low','19':'Low','20':'High',
}

CHANNEL_MAPPING = {
    'A1':'Fp1','A2':'AF3','A3':'F7','A4':'F3','A5':'FC1','A6':'FC5',
    'A7':'T7','A8':'C3','A9':'CP1','A10':'CP5','A11':'P7','A12':'P3',
    'A13':'Pz','A14':'PO3','A15':'O1','A16':'Oz','A17':'O2','A18':'PO4',
    'A19':'P4','A20':'P8','A21':'CP6','A22':'CP2','A23':'C4','A24':'T8',
    'A25':'FC6','A26':'FC2','A27':'F4','A28':'F8','A29':'AF4','A30':'Fp2',
    'A31':'Fz','A32':'Cz',
}


# ── Spectral helpers ──────────────────────────────────────────────────────────
def compute_band_power(seg, channels, fmin, fmax):
    """Mean PSD power in a band, averaged across available channels."""
    available = [ch for ch in channels if ch in seg.info['ch_names']]
    if not available:
        return np.nan
    sfreq = seg.info['sfreq']
    try:
        spectrum = seg.compute_psd(
            method='welch',
            fmin=1.0, fmax=FILTER_HIGH,
            n_fft=int(sfreq * 2),
            n_overlap=int(sfreq * 1),
            picks=available,
            reject_by_annotation=False,
            verbose=False,
        )
    except Exception:
        return np.nan
    freqs = spectrum.freqs
    data = spectrum.get_data()
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any() or data.size == 0:
        return np.nan
    return float(data[:, mask].mean())


def compute_tbr(seg):
    theta = compute_band_power(seg, FRONTAL_CHANNELS, *THETA_BAND)
    beta = compute_band_power(seg, FRONTAL_CHANNELS, *BETA_BAND)
    if np.isnan(theta) or np.isnan(beta) or beta == 0:
        return np.nan
    return theta / beta


def compute_posterior_beta(seg):
    return compute_band_power(seg, POSTERIOR_CHANNELS, *BETA_BAND)


# ── Baseline corrections ──────────────────────────────────────────────────────
def sub(a, b):
    return a - b if not (np.isnan(a) or np.isnan(b)) else np.nan


def ratio(a, b):
    if np.isnan(a) or np.isnan(b) or b == 0:
        return np.nan
    return a / b


def db(a, b):
    if np.isnan(a) or np.isnan(b) or b <= 0 or a <= 0:
        return np.nan
    return 10 * np.log10(a / b)


# ── Bad channel detection ─────────────────────────────────────────────────────
def detect_bad_channels(raw_filtered):
    """SD-ratio detection against the montage median. Filtered, pre-reference."""
    picks = mne.pick_types(raw_filtered.info, eeg=True)
    names = [raw_filtered.ch_names[i] for i in picks]
    sds = raw_filtered.get_data(picks=picks).std(axis=1)
    med = np.median(sds)
    bads, detail = [], {}
    for name, sd in zip(names, sds):
        r = sd / med if med > 0 else np.inf
        if r > BAD_HIGH_RATIO or r < BAD_LOW_RATIO:
            bads.append(name)
            detail[name] = round(float(r), 2)
    return bads, detail


def annotated_fraction(raw, prefix='BAD'):
    if raw.annotations is None or len(raw.annotations) == 0:
        return 0.0
    total = sum(d for desc, d in zip(raw.annotations.description,
                                     raw.annotations.duration)
                if desc.upper().startswith(prefix))
    return float(total / raw.times[-1]) if raw.times[-1] > 0 else 0.0


def annot_overlap_pct(raw, tmin, tmax):
    """Percentage of window [tmin, tmax] covered by BAD annotations."""
    if raw.annotations is None or len(raw.annotations) == 0:
        return 0.0
    dur = tmax - tmin
    if dur <= 0:
        return 0.0
    covered = 0.0
    for onset, d, desc in zip(raw.annotations.onset,
                              raw.annotations.duration,
                              raw.annotations.description):
        if not desc.upper().startswith('BAD'):
            continue
        a, b = max(onset, tmin), min(onset + d, tmax)
        if b > a:
            covered += b - a
    return 100.0 * covered / dur


# ── Behavioural data ──────────────────────────────────────────────────────────
def load_behavioural(participant_id, folder):
    result = {}
    for pattern in [f"P{participant_id}_group*_with_ratings.csv",
                    f"P{participant_id}_group*_responses.csv"]:
        matches = glob.glob(os.path.join(folder, pattern))
        if not matches:
            continue
        correct, total, rating = {}, {}, {}
        with open(matches[0], newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                bn = int(row['block_num'])
                correct[bn] = correct.get(bn, 0) + int(row['is_correct'])
                total[bn] = total.get(bn, 0) + 1
                if row.get('focus_rating'):
                    rating[bn] = row['focus_rating']
        for bn in total:
            result[bn] = {
                'recall_accuracy': round(correct.get(bn, 0) / total[bn], 4),
                'focus_rating': rating.get(bn, ''),
            }
        break
    return result


# ── Load block boundaries ─────────────────────────────────────────────────────
print("Loading block boundaries...")
blocks_by_participant = {}
with open(BLOCKS_CSV, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        blocks_by_participant.setdefault(row['participant_id'], []).append(row)
print(f"  {len(blocks_by_participant)} participants\n")

# ── Load manual ICA exclusions if applying them ───────────────────────────────
ica_exclusions = {}
if ICA_MODE == 'apply_reviewed':
    excl_path = os.path.join(ICA_FOLDER, 'ica_exclusions.json')
    if os.path.exists(excl_path):
        with open(excl_path) as f:
            ica_exclusions = json.load(f)
        print(f"Loaded manual ICA exclusions for "
              f"{len(ica_exclusions)} participants\n")
    else:
        raise FileNotFoundError(
            f"ICA_MODE='apply_reviewed' but {excl_path} not found. "
            f"Run ica_review.py first.")

all_features = []
qc_log = []

# ── Main loop ─────────────────────────────────────────────────────────────────
for pid in sorted(blocks_by_participant, key=int):
    blocks = blocks_by_participant[pid]
    bdf_path = os.path.join(BDF_FOLDER, f"eeg_participant{pid}.bdf")

    print("=" * 62)
    print(f"Participant {pid}")
    print("=" * 62)

    qc = {'participant_id': pid, 'status': 'ok'}

    if not os.path.exists(bdf_path):
        print("  BDF not found — skipping\n")
        qc['status'] = 'bdf_missing'
        qc_log.append(qc)
        continue

    group = ADHD_GROUPS.get(pid, 'Unknown')
    qc['adhd_group'] = group
    print(f"  Group: {group}")

    behav = load_behavioural(pid, BEHAV_FOLDER)
    print(f"  Behavioural blocks: {len(behav)}")

    # 1. Load
    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
    sfreq = raw.info['sfreq']
    qc['duration_min'] = round(raw.times[-1] / 60, 1)

    # 2. Drop non-EEG
    to_drop = [c for c in NON_EEG if c in raw.ch_names]
    raw.drop_channels(to_drop)
    qc['channels_dropped'] = len(to_drop)

    # 3. Rename and montage
    raw.rename_channels({o: n for o, n in CHANNEL_MAPPING.items()
                         if o in raw.ch_names})
    raw.set_montage(mne.channels.make_standard_montage('biosemi32'),
                    match_case=False, on_missing='ignore', verbose=False)

    # 4. Bad channels — on a filtered copy, leaving raw unfiltered so that
    #    muscle detection can still use the 110-140 Hz band
    tmp = raw.copy().filter(FILTER_LOW, FILTER_HIGH, verbose=False)
    bads, bad_detail = detect_bad_channels(tmp)
    del tmp

    n_eeg = len(mne.pick_types(raw.info, eeg=True))
    qc['bad_channels'] = ','.join(bads) if bads else ''
    qc['bad_detail'] = json.dumps(bad_detail)
    qc['n_bad'] = len(bads)
    print(f"  Bad channels: {bad_detail if bads else 'none'}")

    if len(bads) / n_eeg > MAX_BAD_FRACTION:
        print(f"  ABORT — {len(bads)}/{n_eeg} channels bad\n")
        qc['status'] = 'too_many_bad_channels'
        qc_log.append(qc)
        del raw
        continue

    raw.info['bads'] = bads

    # 5. Muscle detection — unfiltered
    try:
        annot_muscle, _ = annotate_muscle_zscore(
            raw, ch_type='eeg', threshold=MUSCLE_THRESHOLD,
            min_length_good=MUSCLE_MIN_GOOD, filter_freq=MUSCLE_FREQ,
            verbose=False)
        raw.set_annotations(raw.annotations + annot_muscle)
        pct = annotated_fraction(raw) * 100
        qc['pct_muscle'] = round(pct, 2)
        print(f"  Muscle annotated: {pct:.2f}% of recording")
    except Exception as e:
        qc['pct_muscle'] = ''
        print(f"  Muscle detection failed: {e}")

    # 6. Filter
    raw.filter(FILTER_LOW, FILTER_HIGH, verbose=False)
    print(f"  Filtered {FILTER_LOW}-{FILTER_HIGH} Hz")

    # 7. Interpolate
    if bads:
        raw.interpolate_bads(reset_bads=True, verbose=False)
        print(f"  Interpolated {len(bads)} channels")

    # 8. Reference
    raw.set_eeg_reference(REFERENCE, verbose=False)
    qc['reference'] = REFERENCE
    print(f"  Re-referenced: {REFERENCE}")

    # 9. ICA
    qc['ica_mode'] = ICA_MODE
    qc['ica_excluded'] = 0

    if ICA_MODE == 'apply_reviewed':
        ica_path = os.path.join(ICA_FOLDER, f'P{pid}-ica.fif')
        excl = ica_exclusions.get(pid, [])
        if os.path.exists(ica_path):
            ica = mne.preprocessing.read_ica(ica_path, verbose=False)
            ica.exclude = excl
            ica.apply(raw, verbose=False)
            qc['ica_excluded'] = len(excl)
            print(f"  ICA: applied, excluded {len(excl)} components {excl}")
        else:
            print(f"  ICA: no solution found for P{pid} — not applied")
            qc['ica_excluded'] = ''

    elif ICA_MODE == 'frontal_proxy':
        ica = ICA(n_components=ICA_N_COMPONENTS, random_state=42,
                  max_iter='auto', verbose=False)
        ica.fit(raw, reject_by_annotation=True, verbose=False)
        proxy = next((c for c in ['Fp1', 'Fp2'] if c in raw.ch_names), None)
        if proxy:
            idx, _ = ica.find_bads_eog(raw, ch_name=proxy, verbose=False)
            ica.exclude = idx
            qc['ica_excluded'] = len(idx)
            print(f"  ICA: excluded {len(idx)} components via {proxy}")
        ica.apply(raw, verbose=False)

    else:
        print("  ICA: disabled")

    # 10. Features per block
    max_t = raw.times[-1]
    for b in blocks:
        bn = int(b['block_num'])
        cond = b['condition']

        bs_t = int(b['baseline_start_sample']) / sfreq
        be_t = int(b['baseline_end_sample']) / sfreq
        es_t = int(b['encoding_start_sample']) / sfreq
        ee_t = int(b['encoding_end_sample']) / sfreq

        if bs_t >= max_t or ee_t >= max_t:
            print(f"    Block {bn}: outside recording — skipped")
            continue
        be_t = min(be_t, max_t - 0.1)
        ee_t = min(ee_t, max_t - 0.1)

        enc_muscle_pct = annot_overlap_pct(raw, es_t, ee_t)
        base_muscle_pct = annot_overlap_pct(raw, bs_t, be_t)

        try:
            base = raw.copy().crop(bs_t, be_t, include_tmax=False)
            b_tbr = compute_tbr(base)
            b_beta = compute_posterior_beta(base)
        except Exception:
            b_tbr = b_beta = np.nan

        try:
            enc = raw.copy().crop(es_t, ee_t, include_tmax=False)
            e_tbr = compute_tbr(enc)
            e_beta = compute_posterior_beta(enc)
        except Exception:
            e_tbr = e_beta = np.nan

        tbr_sub = sub(e_tbr, b_tbr)
        tbr_ratio = ratio(e_tbr, b_tbr)
        beta_sub = sub(e_beta, b_beta)
        beta_db = db(e_beta, b_beta)

        bdur = float(b['baseline_duration_s'])
        edur = float(b['encoding_duration_s'])

        # ── Block-level flags: affect every measure ───────────────────────
        block_flags = []
        if abs(bdur - EXCL_BASELINE_NOMINAL) > EXCL_BASELINE_TOL:
            block_flags.append(f'baseline_dur({bdur})')
        if abs(edur - EXCL_ENCODING_NOMINAL) > EXCL_ENCODING_TOL:
            block_flags.append(f'encoding_dur({edur})')
        if enc_muscle_pct > EXCL_MUSCLE_PCT:
            block_flags.append(f'muscle_pct({enc_muscle_pct:.1f})')
        if cond.startswith('Block_'):
            block_flags.append('condition_unmapped')
        block_str = ' | '.join(block_flags) if block_flags else 'OK'

        # ── TBR flags: frontal measures only ──────────────────────────────
        tbr_flags = []
        if np.isnan(e_tbr):
            tbr_flags.append('tbr_missing')
        else:
            if e_tbr > EXCL_TBR_MAX:
                tbr_flags.append(f'tbr_high({e_tbr:.1f})')
            if e_tbr < EXCL_TBR_MIN:
                tbr_flags.append(f'tbr_low({e_tbr:.2f})')
        if np.isnan(b_tbr):
            tbr_flags.append('tbr_baseline_missing')
        else:
            if b_tbr > EXCL_TBR_MAX:
                tbr_flags.append(f'tbr_baseline_high({b_tbr:.1f})')
            if b_tbr < EXCL_TBR_MIN:
                tbr_flags.append(f'tbr_baseline_low({b_tbr:.2f})')
        tbr_str = ' | '.join(tbr_flags) if tbr_flags else 'OK'

        # ── Beta flags: posterior measures only ───────────────────────────
        beta_flags = []
        if np.isnan(e_beta):
            beta_flags.append('beta_missing')
        elif e_beta > EXCL_BETA_MAX:
            beta_flags.append(f'beta_high({e_beta:.2e})')
        if np.isnan(b_beta):
            beta_flags.append('beta_baseline_missing')
        elif b_beta > EXCL_BETA_MAX:
            beta_flags.append(f'beta_baseline_high({b_beta:.2e})')
        beta_str = ' | '.join(beta_flags) if beta_flags else 'OK'

        bb = behav.get(bn, {})

        def mark(s):
            return '' if s == 'OK' else '  <<'

        print(f"    Block {bn} ({cond}): "
              f"TBR={e_tbr:.3f}{mark(tbr_str)} "
              f"beta={e_beta:.2e}{mark(beta_str)} "
              f"dB={beta_db:+.2f} acc={bb.get('recall_accuracy','')}"
              f"{mark(block_str)}")
        if block_str != 'OK':
            print(f"        block: {block_str}")
        if tbr_str != 'OK':
            print(f"        tbr:   {tbr_str}")
        if beta_str != 'OK':
            print(f"        beta:  {beta_str}")

        all_features.append({
            'participant_id': pid,
            'adhd_group': group,
            'block_num': bn,
            'condition': cond,
            'tbr_encoding': e_tbr,
            'tbr_baseline': b_tbr,
            'tbr_sub': tbr_sub,
            'tbr_ratio': tbr_ratio,
            'beta_encoding': e_beta,
            'beta_baseline': b_beta,
            'beta_sub': beta_sub,
            'beta_db': beta_db,
            'recall_accuracy': bb.get('recall_accuracy', ''),
            'focus_rating': bb.get('focus_rating', ''),
            'baseline_duration_s': bdur,
            'encoding_duration_s': edur,
            'n_bad_channels': len(bads),
            'enc_muscle_pct': round(enc_muscle_pct, 2),
            'base_muscle_pct': round(base_muscle_pct, 2),
            'block_flags': block_str,
            'tbr_flags': tbr_str,
            'beta_flags': beta_str,
        })

    qc_log.append(qc)
    del raw
    print()

# ── Save ──────────────────────────────────────────────────────────────────────
ts = datetime.now().strftime('%Y%m%d_%H%M%S')

if all_features:
    out = os.path.join(OUTPUT_FOLDER, f'eeg_features_v4_{ts}.csv')
    pd.DataFrame(all_features).to_csv(out, index=False, float_format='%.15e')
    print(f"Features: {out}  ({len(all_features)} rows)")

qc_path = os.path.join(OUTPUT_FOLDER, f'qc_log_v4_{ts}.csv')
pd.DataFrame(qc_log).to_csv(qc_path, index=False)
print(f"QC log:   {qc_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
if all_features:
    df = pd.DataFrame(all_features)
    ok_block = df['block_flags'] == 'OK'
    ok_tbr = ok_block & (df['tbr_flags'] == 'OK')
    ok_beta = ok_block & (df['beta_flags'] == 'OK')
    ok_recall = ok_block & df['recall_accuracy'].astype(str).ne('')

    print(f"\nBlocks total:            {len(df)}")
    print(f"Usable for recall:       {ok_recall.sum()}")
    print(f"Usable for frontal TBR:  {ok_tbr.sum()}")
    print(f"Usable for posterior beta: {ok_beta.sum()}")
    print(f"Participants:            {df['participant_id'].nunique()}")

    for name, col in [('Block-level', 'block_flags'),
                      ('TBR', 'tbr_flags'),
                      ('Beta', 'beta_flags')]:
        bad = df.loc[df[col] != 'OK', col]
        if len(bad):
            print(f"\n{name} exclusions:")
            for f_, n in bad.value_counts().items():
                print(f"  {n:3d}  {f_}")