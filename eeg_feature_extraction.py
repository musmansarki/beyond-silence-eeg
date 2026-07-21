#!/usr/bin/env python3
"""
EEG Preprocessing and Feature Extraction Pipeline
---------------------------------------------------
Reads the all_blocks CSV produced by eeg_block_isolator.py,
then for each participant:

1. Loads the BDF file
2. Renames channels from BioSemi A1-A32 to standard 10-20 names
3. Applies band-pass filter (1-35 Hz)
4. Runs ICA to remove eye blink artifacts
5. For each block extracts:
   - Baseline epoch (marker 30 → 31)
   - Encoding epoch (marker 40 → 43)
6. Computes Power Spectral Density using Welch's method
7. Computes frontal TBR and posterior beta power
8. Baseline-normalises the encoding features
9. Saves one row per block to a master features CSV

Output CSV columns:
  participant_id, adhd_group, block_num, condition,
  tbr_raw, tbr_baselined,
  posterior_beta_raw, posterior_beta_baselined,
  recall_accuracy, focus_rating
"""

import mne
import numpy as np
import os
import csv
import glob
from datetime import datetime
from mne.preprocessing import ICA

# ── Configuration ─────────────────────────────────────────────────────────────
BDF_FOLDER    = '/Users/momosarki/Documents/eeg_analysis/eeg_data'
BLOCKS_CSV    = '/Users/momosarki/Documents/eeg_analysis/block_epochs/all_blocks_20260622_232517.csv'
BEHAV_FOLDER  = '/Users/momosarki/Documents/eeg_analysis/data'
OUTPUT_FOLDER = '/Users/momosarki/Documents/eeg_analysis/features'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ADHD group assignments
ADHD_GROUPS = {
    '1':  'High',
    '2':  'High',
    '3':  'Low',
    '4':  'High',
    '5':  'Low',
    '6':  'High',       
    '7':  'Low',      
    '8':  'High',
    '9':  'High',   
    '10': 'Low',  
    '11': 'Low',       
    '12': 'Low', 
    '13': 'High',   
    '14': 'High',     
    '15': 'High',      
    '16': 'High',
    '17': 'Low',
    '18': 'Low',
    '19': 'Low',
    '20': 'High',
}

# BioSemi 32-channel A1-A32 to standard 10-20 mapping
CHANNEL_MAPPING = {
    'A1':  'Fp1', 'A2':  'AF3', 'A3':  'F7',  'A4':  'F3',
    'A5':  'FC1', 'A6':  'FC5', 'A7':  'T7',  'A8':  'C3',
    'A9':  'CP1', 'A10': 'CP5', 'A11': 'P7',  'A12': 'P3',
    'A13': 'Pz',  'A14': 'PO3', 'A15': 'O1',  'A16': 'Oz',
    'A17': 'O2',  'A18': 'PO4', 'A19': 'P4',  'A20': 'P8',
    'A21': 'CP6', 'A22': 'CP2', 'A23': 'C4',  'A24': 'T8',
    'A25': 'FC6', 'A26': 'FC2', 'A27': 'F4',  'A28': 'F8',
    'A29': 'AF4', 'A30': 'Fp2', 'A31': 'Fz',  'A32': 'Cz',
}

# EEG channel analysis targets
FRONTAL_CHANNELS   = ['F3', 'Fz', 'F4']
POSTERIOR_CHANNELS = ['P3', 'Pz', 'P4', 'PO3', 'PO4']
THETA_BAND         = (4, 7)
BETA_BAND          = (13, 30)

# Filter settings
FILTER_LOW  = 1.0
FILTER_HIGH = 35.0

# ICA settings
N_ICA_COMPONENTS = 20

# ── Helper: compute band power from a raw segment ────────────────────────────
def compute_band_power(raw_segment, channels, fmin, fmax):
    """
    Compute mean power in a frequency band across specified channels.
    Uses Welch's method with 2-second windows.
    Returns mean power value (float), or NaN if channels not found.
    """
    available = [ch for ch in channels if ch in raw_segment.info['ch_names']]
    if not available:
        return np.nan

    sfreq  = raw_segment.info['sfreq']
    n_fft  = int(sfreq * 2)       # 2-second windows
    n_over = int(sfreq * 1)       # 50% overlap

    spectrum = raw_segment.compute_psd(
        method='welch',
        fmin=1.0,
        fmax=35.0,
        n_fft=n_fft,
        n_overlap=n_over,
        picks=available,
        verbose=False,
    )

    freqs = spectrum.freqs
    data  = spectrum.get_data()  # shape: (n_channels, n_freqs)

    band_mask = (freqs >= fmin) & (freqs <= fmax)
    band_power = data[:, band_mask].mean()
    return float(band_power)


def compute_tbr(raw_segment):
    """Compute frontal theta/beta ratio."""
    theta = compute_band_power(
        raw_segment, FRONTAL_CHANNELS, THETA_BAND[0], THETA_BAND[1]
    )
    beta  = compute_band_power(
        raw_segment, FRONTAL_CHANNELS, BETA_BAND[0], BETA_BAND[1]
    )
    if np.isnan(theta) or np.isnan(beta) or beta == 0:
        return np.nan
    return theta / beta


def compute_posterior_beta(raw_segment):
    """Compute posterior beta power."""
    return compute_band_power(
        raw_segment, POSTERIOR_CHANNELS, BETA_BAND[0], BETA_BAND[1]
    )


# ── Helper: load behavioural data for a participant ───────────────────────────
def load_behavioural(participant_id, behav_folder):
    """
    Returns dict: block_num → {'recall_accuracy': float, 'focus_rating': int}
    """
    result = {}
    patterns = [
        f"P{participant_id}_group*_with_ratings.csv",
        f"P{participant_id}_group*_responses.csv",
    ]
    for pattern in patterns:
        matches = glob.glob(os.path.join(behav_folder, pattern))
        if matches:
            block_correct = {}
            block_total   = {}
            block_rating  = {}
            with open(matches[0], newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    bn = int(row['block_num'])
                    block_correct[bn] = block_correct.get(bn, 0) + int(row['is_correct'])
                    block_total[bn]   = block_total.get(bn, 0) + 1
                    if 'focus_rating' in row and row['focus_rating']:
                        block_rating[bn] = row['focus_rating']
            for bn in block_total:
                result[bn] = {
                    'recall_accuracy': round(
                        block_correct.get(bn, 0) / block_total[bn], 4
                    ),
                    'focus_rating': block_rating.get(bn, ''),
                }
            break
    return result


# ── Load block epoch boundaries ───────────────────────────────────────────────
print("Loading block epoch boundaries...")
blocks_by_participant = {}
with open(BLOCKS_CSV, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = row['participant_id']
        if pid not in blocks_by_participant:
            blocks_by_participant[pid] = []
        blocks_by_participant[pid].append(row)

print(f"Loaded blocks for {len(blocks_by_participant)} participants\n")

# ── Main pipeline loop ────────────────────────────────────────────────────────
all_features = []

for participant_id in sorted(blocks_by_participant.keys(), key=lambda x: int(x)):
    blocks = blocks_by_participant[participant_id]
    bdf_path = os.path.join(BDF_FOLDER, f"eeg_participant{participant_id}.bdf")

    print(f"{'='*60}")
    print(f"Participant {participant_id}")
    print(f"{'='*60}")

    if not os.path.exists(bdf_path):
        print(f"  BDF file not found: {bdf_path} — skipping\n")
        continue

    adhd_group = ADHD_GROUPS.get(participant_id, 'Unknown')
    print(f"  ADHD group: {adhd_group}")

    # Load behavioural data
    behav = load_behavioural(participant_id, BEHAV_FOLDER)
    print(f"  Behavioural data: {len(behav)} blocks loaded")

    # ── Step 1: Load BDF ──────────────────────────────────────────────────────
    print(f"  Loading BDF...")
    try:
        raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
    except Exception as e:
        print(f"  ERROR loading BDF: {e} — skipping\n")
        continue

    sfreq = raw.info['sfreq']
    print(f"  Loaded: {raw.times[-1]/60:.1f} min | {sfreq} Hz | "
          f"{len(raw.info['ch_names'])} channels")

    # ── Step 2: Rename channels ───────────────────────────────────────────────
    rename_map = {
        old: new for old, new in CHANNEL_MAPPING.items()
        if old in raw.info['ch_names']
    }
    if rename_map:
        raw.rename_channels(rename_map)
        print(f"  Renamed {len(rename_map)} channels to 10-20 standard")
    else:
        print(f"  WARNING: No A1-A32 channels found — channels may already "
              f"be named or mapping is wrong")
        print(f"  Current channels: {raw.info['ch_names'][:8]}...")

    # Set EOG channels
    eog_chs = [ch for ch in ['EXG1','EXG2','EXG3','EXG4']
               if ch in raw.info['ch_names']]
    if eog_chs:
        raw.set_channel_types({ch: 'eog' for ch in eog_chs})

    # Set montage for EEG channels only
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False, on_missing='ignore',
                        verbose=False)
    except Exception:
        pass

    # ── Step 3: Band-pass filter ──────────────────────────────────────────────
    print(f"  Filtering {FILTER_LOW}-{FILTER_HIGH} Hz...")
    raw.filter(l_freq=FILTER_LOW, h_freq=FILTER_HIGH, verbose=False)

    # ── Step 4: ICA ───────────────────────────────────────────────────────────
    print(f"  Running ICA...")
    try:
        # Use EEG channels only — excludes GSR, Resp, Plet, Temp, EXG5-8
        # which inflate variance and destabilise ICA decomposition
        eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False,
                                   exclude='bads')
        n_eeg = len(eeg_picks)

        # Use 0.999999 to let MNE auto-select stable number of components
        print(f"  ICA: fitting on {n_eeg} EEG channels, auto components")
        ica = ICA(n_components=0.999999, random_state=42, verbose=False)
        ica.fit(raw, picks=eeg_picks, verbose=False)

        # Auto-detect eye blink components using EOG channels
        if eog_chs:
            eog_idx, _ = ica.find_bads_eog(raw, verbose=False)
            if eog_idx:
                ica.exclude = eog_idx
                print(f"  ICA: excluding eye components {eog_idx}")
            else:
                print(f"  ICA: no eye components auto-detected")
        else:
            print(f"  ICA: no EOG channels — skipping auto-detection")

        # Apply ICA to EEG channels only
        ica.apply(raw, verbose=False)
        print(f"  ICA applied")
    except Exception as e:
        print(f"  WARNING: ICA failed ({e}) — continuing without ICA")

    # ── Step 5: Extract epochs and compute features ───────────────────────────
    print(f"  Extracting features per block...")

    for block_row in blocks:
        block_num = int(block_row['block_num'])
        condition = block_row['condition']

        # Get sample boundaries
        baseline_start = int(block_row['baseline_start_sample'])
        baseline_end   = int(block_row['baseline_end_sample'])
        enc_start      = int(block_row['encoding_start_sample'])
        enc_end        = int(block_row['encoding_end_sample'])

        # Convert samples to times
        baseline_start_t = baseline_start / sfreq
        baseline_end_t   = baseline_end   / sfreq
        enc_start_t      = enc_start      / sfreq
        enc_end_t        = enc_end        / sfreq

        # Validate boundaries are within recording
        max_time = raw.times[-1]
        if baseline_start_t >= max_time or enc_end_t >= max_time:
            print(f"    Block {block_num}: boundaries outside recording — skipping")
            continue

        baseline_end_t = min(baseline_end_t, max_time - 0.1)
        enc_end_t      = min(enc_end_t, max_time - 0.1)

        # ── Extract baseline epoch ────────────────────────────────────────────
        try:
            baseline_raw = raw.copy().crop(
                tmin=baseline_start_t,
                tmax=baseline_end_t,
                include_tmax=False
            )
            baseline_tbr  = compute_tbr(baseline_raw)
            baseline_beta = compute_posterior_beta(baseline_raw)
        except Exception as e:
            print(f"    Block {block_num}: baseline extraction failed ({e})")
            baseline_tbr  = np.nan
            baseline_beta = np.nan

        # ── Extract encoding epoch ────────────────────────────────────────────
        try:
            encoding_raw = raw.copy().crop(
                tmin=enc_start_t,
                tmax=enc_end_t,
                include_tmax=False
            )
            encoding_tbr  = compute_tbr(encoding_raw)
            encoding_beta = compute_posterior_beta(encoding_raw)
        except Exception as e:
            print(f"    Block {block_num}: encoding extraction failed ({e})")
            encoding_tbr  = np.nan
            encoding_beta = np.nan

        # ── Baseline normalisation ────────────────────────────────────────────
        # Subtract baseline from encoding to get change from rest
        tbr_baselined  = (
            encoding_tbr - baseline_tbr
            if not (np.isnan(encoding_tbr) or np.isnan(baseline_tbr))
            else np.nan
        )
        beta_baselined = (
            encoding_beta - baseline_beta
            if not (np.isnan(encoding_beta) or np.isnan(baseline_beta))
            else np.nan
        )

        # ── Get behavioural measures ──────────────────────────────────────────
        behav_block = behav.get(block_num, {})
        recall_acc  = behav_block.get('recall_accuracy', '')
        focus_rat   = behav_block.get('focus_rating', '')

        # ── Outlier flagging ──────────────────────────────────────────────────
        outlier_flags = []
        if not np.isnan(encoding_tbr) and encoding_tbr > 30:
            outlier_flags.append(f'TBR_high({encoding_tbr:.1f})')
        if not np.isnan(encoding_beta) and encoding_beta > 1e-10:
            outlier_flags.append(f'beta_high({encoding_beta:.2e})')
        if not np.isnan(tbr_baselined) and abs(tbr_baselined) > 20:
            outlier_flags.append(f'TBR_delta_large({tbr_baselined:.1f})')
        if str(block_row['baseline_duration_s']) != '30.0':
            outlier_flags.append(f"baseline_dur({block_row['baseline_duration_s']}s)")
        outlier_str = ' | '.join(outlier_flags) if outlier_flags else 'OK'

        flag_note = f' — FLAGGED: {outlier_str}' if outlier_flags else ''
        print(f"    Block {block_num} ({condition}): "
              f"TBR={encoding_tbr:.3f} (D{tbr_baselined:+.3f}), "
              f"b={encoding_beta:.2e}, acc={recall_acc}{flag_note}")

        # ── Store result ──────────────────────────────────────────────────────
        all_features.append({
            'participant_id':           participant_id,
            'adhd_group':               adhd_group,
            'block_num':                block_num,
            'condition':                condition,
            'tbr_encoding':             round(encoding_tbr, 6)   if not np.isnan(encoding_tbr)   else '',
            'tbr_baseline':             round(baseline_tbr, 6)   if not np.isnan(baseline_tbr)   else '',
            'tbr_baselined':            round(tbr_baselined, 6)  if not np.isnan(tbr_baselined)  else '',
            'posterior_beta_encoding':  float(encoding_beta) if not np.isnan(encoding_beta) else '',
            'posterior_beta_baseline':  float(baseline_beta) if not np.isnan(baseline_beta) else '',
            'posterior_beta_baselined': float(beta_baselined) if not np.isnan(beta_baselined) else '',
            'recall_accuracy':          recall_acc,
            'focus_rating':             focus_rat,
            'baseline_duration_s':      block_row['baseline_duration_s'],
            'encoding_duration_s':      block_row['encoding_duration_s'],
            'outlier_flags':            outlier_str,
        })
    del raw
    print()

# ── Save master features CSV ──────────────────────────────────────────────────
if all_features:
    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(OUTPUT_FOLDER, f'eeg_features_{timestamp}.csv')

    fieldnames = [
        'participant_id', 'adhd_group', 'block_num', 'condition',
        'tbr_encoding', 'tbr_baseline', 'tbr_baselined',
        'posterior_beta_encoding', 'posterior_beta_baseline', 'posterior_beta_baselined',
        'recall_accuracy', 'focus_rating',
        'baseline_duration_s', 'encoding_duration_s', 'outlier_flags',
    ]

    # Use pandas with float_format to preserve small values like 1e-13
    import pandas as pd
    df_out = pd.DataFrame(all_features, columns=fieldnames)
    df_out.to_csv(output_path, index=False, float_format='%.15e')

    print(f"{'='*60}")
    print(f"Features saved: {output_path}")
    print(f"Total rows: {len(all_features)}")
    print(f"Participants: {len(set(r['participant_id'] for r in all_features))}")
    print(f"{'='*60}")
else:
    print("No features extracted — check BDF paths and block CSV")