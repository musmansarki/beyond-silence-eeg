#!/usr/bin/env python3
"""
EEG Block Isolator
-------------------
For each BDF file, uses marker 43 (ENCODING_END — unique to my experiment)
as an anchor to find and extract exactly 5 clean block windows.

For each block it finds:
  - Baseline:  marker 30 → 31
  - Encoding:  marker 40 → 43
  - Recall:    marker 60 → 64

Condition identity is NOT read from EEG markers (21-25 are unreliable
due to partner overlap). Instead a CSV mapping of block→condition is
expected, read from the participant's behavioural data file.

Output:
  - Per-participant CSV: one row per block with sample boundaries
  - Console summary showing what was found for each file
"""

import mne
import numpy as np
import os
import csv
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────
BDF_FOLDER  = '/Users/momosarki/Documents/eeg_analysis/eeg_data'
CSV_FOLDER  = '/Users/momosarki/Documents/eeg_analysis/data'
OUTPUT_FOLDER = '/Users/momosarki/Documents/eeg_analysis/block_epochs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# My marker codes
M_BASELINE_START  = 30
M_BASELINE_END    = 31
M_ENCODING_START  = 40
M_ENCODING_END    = 43   # UNIQUE TO MY EXPERIMENT
M_RECALL_START    = 60
M_RECALL_END      = 64
M_RATING_START    = 70
M_RATING_RESPONSE = 71

# Partner's unique markers — used to detect boundary between experiments
PARTNER_UNIQUE = {91, 92, 93, 94, 95, 100, 254}

# How far to search around each anchor (in seconds)
SEARCH_WINDOW_BEFORE = 400   # look up to 400s before marker 43 for marker 40
SEARCH_WINDOW_AFTER  = 600   # look up to 600s after marker 43 for recall end

# ── Helper: find nearest event before a sample ────────────────────────────────
def find_last_before(events, code, before_sample, within_samples=None):
    """Find the last occurrence of code before before_sample."""
    mask = events[:, 2] == code
    mask &= events[:, 0] < before_sample
    if within_samples is not None:
        mask &= events[:, 0] >= (before_sample - within_samples)
    candidates = events[mask]
    return candidates[-1][0] if len(candidates) > 0 else None

def find_first_after(events, code, after_sample, within_samples=None):
    """Find the first occurrence of code after after_sample."""
    mask = events[:, 2] == code
    mask &= events[:, 0] > after_sample
    if within_samples is not None:
        mask &= events[:, 0] <= (after_sample + within_samples)
    candidates = events[mask]
    return candidates[0][0] if len(candidates) > 0 else None

def find_first_after_any(events, codes, after_sample):
    """Find first occurrence of any code in codes after after_sample."""
    mask = np.isin(events[:, 2], list(codes))
    mask &= events[:, 0] > after_sample
    candidates = events[mask]
    return candidates[0][0] if len(candidates) > 0 else None

# ── Load participant→group mapping from CSV filenames ─────────────────────────
def get_condition_map(participant_id, csv_folder):
    """
    Read the participant's behavioural CSV and return a dict
    mapping block_num (int) → condition (str).
    """
    condition_map = {}
    # Try both possible CSV filename formats
    patterns = [
        f"P{participant_id}_group*_with_ratings.csv",
        f"P{participant_id}_group*_responses.csv",
    ]
    import glob
    for pattern in patterns:
        matches = glob.glob(os.path.join(csv_folder, pattern))
        if matches:
            with open(matches[0], newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    bn = int(row['block_num'])
                    if bn not in condition_map:
                        condition_map[bn] = row['condition']
            break
    return condition_map

# ── Find BDF files ────────────────────────────────────────────────────────────
bdf_files = sorted([
    f for f in os.listdir(BDF_FOLDER)
    if f.lower().endswith('.bdf')
])

print(f"Found {len(bdf_files)} BDF files\n")

all_block_rows = []

# ── Process each file ─────────────────────────────────────────────────────────
for filename in bdf_files:
    filepath = os.path.join(BDF_FOLDER, filename)

    # Extract participant ID from filename
    # Expects format: eeg_participantN.bdf
    try:
        participant_id = filename.replace('eeg_participant', '').replace('.bdf', '')
        int(participant_id)  # validate it's a number
    except ValueError:
        print(f"Skipping {filename} — cannot parse participant ID")
        continue

    print(f"{'='*60}")
    print(f"Participant {participant_id}: {filename}")
    print(f"{'='*60}")

    # Load behavioural CSV for condition mapping
    condition_map = get_condition_map(participant_id, CSV_FOLDER)
    if condition_map:
        print(f"  Condition map loaded: {condition_map}")
    else:
        print(f"  WARNING: No behavioural CSV found for P{participant_id}")
        print(f"  Conditions will be labelled as 'Block_N'")

    # Load BDF
    try:
        raw = mne.io.read_raw_bdf(filepath, preload=False, verbose=False)
        sfreq        = raw.info['sfreq']
        duration_min = raw.times[-1] / 60
        print(f"  Duration: {duration_min:.1f} min | {sfreq} Hz")
        raw.load_data(verbose=False)
    except Exception as e:
        print(f"  LOAD ERROR: {e}\n")
        continue

    # Find all events
    try:
        events = mne.find_events(raw, stim_channel='Status', verbose=False)
    except Exception as e:
        print(f"  EVENT ERROR: {e}\n")
        del raw
        continue

    # Find all marker 43 events — these are my 5 encoding ends
    enc_ends = events[events[:, 2] == M_ENCODING_END]
    n_enc_ends = len(enc_ends)
    print(f"  Marker 43 count: {n_enc_ends}")

    if n_enc_ends == 0:
        print(f"  SKIP — no marker 43 found\n")
        del raw
        continue

    if n_enc_ends != 5:
        print(f"  WARNING — expected 5 marker 43s, got {n_enc_ends}")
        print(f"  Proceeding with {min(n_enc_ends, 5)} blocks")

    # ── Extract each block ────────────────────────────────────────────────────
    search_before = int(sfreq * SEARCH_WINDOW_BEFORE)
    search_after  = int(sfreq * SEARCH_WINDOW_AFTER)

    blocks_found = []

    for block_idx, enc_end_event in enumerate(enc_ends[:5], 1):
        enc_end_sample = enc_end_event[0]

        print(f"\n  Block {block_idx} — anchor: marker 43 at "
              f"{enc_end_sample/sfreq/60:.2f} min")

        # ── Find encoding start (marker 40) ───────────────────────────────────
        enc_start_sample = find_last_before(
            events, M_ENCODING_START, enc_end_sample, search_before
        )
        if enc_start_sample is None:
            print(f"    WARNING: No marker 40 found before this marker 43")
            enc_start_sample = enc_end_sample - int(sfreq * 90)
            print(f"    Estimating encoding start 90s before marker 43")

        enc_duration = (enc_end_sample - enc_start_sample) / sfreq
        print(f"    Encoding: {enc_start_sample/sfreq/60:.2f} → "
              f"{enc_end_sample/sfreq/60:.2f} min ({enc_duration:.0f}s)")

        # ── Find baseline start (marker 30) ───────────────────────────────────
        baseline_start_sample = find_last_before(
            events, M_BASELINE_START, enc_start_sample, search_before
        )
        baseline_end_sample = find_last_before(
            events, M_BASELINE_END, enc_start_sample, search_before
        )

        if baseline_start_sample is None:
            print(f"    WARNING: No marker 30 found — estimating baseline")
            baseline_start_sample = enc_start_sample - int(sfreq * 120)
        if baseline_end_sample is None:
            baseline_end_sample = enc_start_sample - int(sfreq * 5)

        baseline_duration = (baseline_end_sample - baseline_start_sample) / sfreq
        print(f"    Baseline: {baseline_start_sample/sfreq/60:.2f} → "
              f"{baseline_end_sample/sfreq/60:.2f} min ({baseline_duration:.0f}s)")

        # ── Find recall start (marker 60) ─────────────────────────────────────
        recall_start_sample = find_first_after(
            events, M_RECALL_START, enc_end_sample, search_after
        )
        if recall_start_sample is None:
            print(f"    WARNING: No marker 60 found after marker 43")
            recall_start_sample = enc_end_sample + int(sfreq * 35)

        # ── Find recall end (marker 64) ───────────────────────────────────────
        recall_end_sample = find_first_after(
            events, M_RECALL_END, recall_start_sample, search_after
        )
        if recall_end_sample is None:
            print(f"    WARNING: No marker 64 found — estimating recall end")
            recall_end_sample = recall_start_sample + int(sfreq * 300)

        recall_duration = (recall_end_sample - recall_start_sample) / sfreq
        print(f"    Recall:   {recall_start_sample/sfreq/60:.2f} → "
              f"{recall_end_sample/sfreq/60:.2f} min ({recall_duration:.0f}s)")

        # ── Find rating response (marker 71) ──────────────────────────────────
        rating_sample = find_first_after(
            events, M_RATING_RESPONSE, recall_end_sample, search_after
        )
        if rating_sample:
            print(f"    Rating:   {rating_sample/sfreq/60:.2f} min")

        # ── Get condition from CSV ────────────────────────────────────────────
        condition = condition_map.get(block_idx, f'Block_{block_idx}')
        print(f"    Condition: {condition}")

        blocks_found.append({
            'participant_id':        participant_id,
            'block_num':             block_idx,
            'condition':             condition,
            # Baseline window
            'baseline_start_sample': int(baseline_start_sample),
            'baseline_end_sample':   int(baseline_end_sample),
            'baseline_start_min':    round(baseline_start_sample/sfreq/60, 3),
            'baseline_end_min':      round(baseline_end_sample/sfreq/60, 3),
            'baseline_duration_s':   round(baseline_duration, 1),
            # Encoding window
            'encoding_start_sample': int(enc_start_sample),
            'encoding_end_sample':   int(enc_end_sample),
            'encoding_start_min':    round(enc_start_sample/sfreq/60, 3),
            'encoding_end_min':      round(enc_end_sample/sfreq/60, 3),
            'encoding_duration_s':   round(enc_duration, 1),
            # Recall window
            'recall_start_sample':   int(recall_start_sample),
            'recall_end_sample':     int(recall_end_sample),
            'recall_start_min':      round(recall_start_sample/sfreq/60, 3),
            'recall_end_min':        round(recall_end_sample/sfreq/60, 3),
            'recall_duration_s':     round(recall_duration, 1),
            # Rating
            'rating_sample':         int(rating_sample) if rating_sample else '',
            'rating_min':            round(rating_sample/sfreq/60, 3) if rating_sample else '',
        })

    # ── Save per-participant block epoch CSV ──────────────────────────────────
    if blocks_found:
        out_path = os.path.join(
            OUTPUT_FOLDER, f"P{participant_id}_blocks.csv"
        )
        fieldnames = list(blocks_found[0].keys())
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(blocks_found)
        print(f"\n  Saved {len(blocks_found)} blocks → {out_path}")
        all_block_rows.extend(blocks_found)
    else:
        print(f"\n  No blocks extracted")

    del raw
    print()

# ── Save master CSV with all participants ─────────────────────────────────────
if all_block_rows:
    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    master_path = os.path.join(OUTPUT_FOLDER, f'all_blocks_{timestamp}.csv')
    fieldnames  = list(all_block_rows[0].keys())
    with open(master_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_block_rows)
    print(f"{'='*60}")
    print(f"Master CSV saved: {master_path}")
    print(f"Total blocks extracted: {len(all_block_rows)}")
    print(f"Total participants: {len(set(r['participant_id'] for r in all_block_rows))}")