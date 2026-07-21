#!/usr/bin/env python3
"""
EEG File Scanner v2
--------------------
Correct isolation strategy:
1. Find all occurrences of marker 43 (unique to my experiment)
2. Work backwards from first 43 to find nearest code 1 (my experiment start)
3. Work forwards from last 43 to find nearest code 2 (my experiment end)
4. Everything between those two markers is my data
"""

import mne
import numpy as np
import os
import csv
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────
BDF_FOLDER = '/Users/momosarki/Documents/eeg_analysis/eeg_data'

# Expected counts within my isolated window
MY_EXPECTED = {
    30:  (5, 5),    # baseline rest start
    31:  (5, 5),    # baseline rest end
    40:  (4, 5),    # encoding start
    43:  (5, 5),    # encoding end
    50:  (4, 5),    # post encode rest start
    51:  (5, 5),    # post encode rest end
    60:  (5, 5),    # recall start
    61:  (120, 120),# question onset
    70:  (5, 5),    # rating start
    71:  (5, 5),    # rating response
    21:  (1, 1),    # silence
    22:  (1, 1),    # white noise
    23:  (1, 1),    # lofi
    24:  (1, 1),    # unfamiliar
    25:  (1, 1),    # familiar
}

MY_MARKER_NAMES = {
    1:  'EXP_START',        2:  'EXP_END',
    21: 'SILENCE',          22: 'WHITE_NOISE',
    23: 'LOFI',             24: 'UNFAMILIAR',
    25: 'FAMILIAR',         30: 'BASELINE_START',
    31: 'BASELINE_END',     40: 'ENCODING_START',
    43: 'ENCODING_END',     50: 'POST_REST_START',
    51: 'POST_REST_END',    60: 'RECALL_START',
    61: 'QUESTION_ONSET',   62: 'CORRECT',
    63: 'INCORRECT',        64: 'RECALL_END',
    70: 'RATING_START',     71: 'RATING_RESPONSE',
    72: 'RATING_END',       80: 'INTERBLOCK_START',
    81: 'INTERBLOCK_END',
}

PARTNER_UNIQUE = {91, 92, 93, 94, 95, 100, 254}
DUPLICATE_THRESHOLD_SEC = 10  # seconds — closer than this = duplicate

# ── Find BDF files ────────────────────────────────────────────────────────────
bdf_files = sorted([
    f for f in os.listdir(BDF_FOLDER)
    if f.lower().endswith('.bdf')
])

print(f"Found {len(bdf_files)} BDF files\n")
summary_rows = []

# ── Process each file ─────────────────────────────────────────────────────────
for i, filename in enumerate(bdf_files, 1):
    filepath = os.path.join(BDF_FOLDER, filename)

    print(f"{'='*65}")
    print(f"File {i}/{len(bdf_files)}: {filename}")
    print(f"{'='*65}")

    row = {
        'filename':          filename,
        'status':            '',
        'recommendation':    '',
        'duration_min':      '',
        'my_window_min':   '',
        'my_start_min':    '',
        'my_end_min':      '',
        'encoding_43_count': '',
        'question_61_count': '',
        'duplicate_markers': '',
        'partner_bleed':     '',
        'issues':            [],
    }

    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        raw = mne.io.read_raw_bdf(filepath, preload=False, verbose=False)
        sfreq        = raw.info['sfreq']
        duration_min = raw.times[-1] / 60
        row['duration_min'] = round(duration_min, 1)
        print(f"  Total duration: {duration_min:.1f} min | {sfreq} Hz")

        if duration_min < 10:
            row['status']         = 'DISCARD'
            row['recommendation'] = 'DISCARD — recording too short'
            row['issues'].append(f"Only {duration_min:.1f} min")
            summary_rows.append(row)
            print(f"  DISCARD\n")
            continue

        raw.load_data(verbose=False)

    except Exception as e:
        print(f"  LOAD ERROR: {e}")
        row['status']         = 'LOAD ERROR'
        row['recommendation'] = 'CHECK — cannot load file'
        row['issues'].append(str(e))
        summary_rows.append(row)
        print()
        continue

    # ── Get all events ────────────────────────────────────────────────────────
    try:
        events = mne.find_events(raw, stim_channel='Status', verbose=False)
    except Exception as e:
        print(f"  EVENT ERROR: {e}")
        row['status']         = 'EVENT ERROR'
        row['recommendation'] = 'CHECK — event detection failed'
        row['issues'].append(str(e))
        summary_rows.append(row)
        del raw
        print()
        continue

    # ── Step 1: Find marker 43 — unique anchor ───────────────────────────
    enc_end_43 = events[events[:, 2] == 43]
    n_43 = len(enc_end_43)
    row['encoding_43_count'] = n_43
    print(f"  Marker 43 (my encoding end): {n_43} occurrences")

    if n_43 == 0:
        print(f"  CANNOT ISOLATE — marker 43 not found at all")
        row['status']         = 'CANNOT ISOLATE'
        row['recommendation'] = 'INVESTIGATE — marker 43 absent'
        row['issues'].append('Marker 43 not found')
        summary_rows.append(row)
        del raw
        print()
        continue

    first_43_sample = enc_end_43[0][0]
    last_43_sample  = enc_end_43[-1][0]

    # ── Step 2: Work backwards from first 43 to find nearest code 1 ──────────
    # Get all code 1 events that appear BEFORE the first 43
    code1_events = events[
        (events[:, 2] == 1) &
        (events[:, 0] < first_43_sample)
    ]

    if len(code1_events) == 0:
        print(f"  WARNING: No code 1 found before first marker 43")
        print(f"  Using first marker 43 minus 5 minutes as start estimate")
        my_start_sample = max(0, first_43_sample - int(sfreq * 300))
        row['issues'].append('No code 1 found before first marker 43')
    else:
        # Take the LAST code 1 before the first 43
        # (handles case where partner's code 1 also appears before mine)
        my_start_sample = code1_events[-1][0]
        my_start_min    = my_start_sample / sfreq / 60
        print(f"  My code 1 (exp start) found at: {my_start_min:.1f} min")

    # ── Step 3: Work forwards from last 43 to find nearest code 2 ────────────
    # Get all code 2 events that appear AFTER the last 43
    code2_events = events[
        (events[:, 2] == 2) &
        (events[:, 0] > last_43_sample)
    ]

    if len(code2_events) == 0:
        print(f"  WARNING: No code 2 found after last marker 43")
        print(f"  Using last marker 43 plus 10 minutes as end estimate")
        my_end_sample = min(
            last_43_sample + int(sfreq * 600),
            int(raw.times[-1] * sfreq)
        )
        row['issues'].append('No code 2 found after last marker 43')
    else:
        # Take the FIRST code 2 after the last 43
        my_end_sample = code2_events[0][0]
        my_end_min    = my_end_sample / sfreq / 60
        print(f"  My code 2 (exp end) found at:   {my_end_min:.1f} min")

    # ── Step 4: Define and report my window ─────────────────────────────────
    my_start_time = my_start_sample / sfreq
    my_end_time   = my_end_sample / sfreq
    my_window_min = (my_end_time - my_start_time) / 60

    row['my_window_min'] = round(my_window_min, 1)
    row['my_start_min']  = round(my_start_time / 60, 1)
    row['my_end_min']    = round(my_end_time / 60, 1)

    print(f"  My window: {my_start_time/60:.1f} → "
          f"{my_end_time/60:.1f} min "
          f"({my_window_min:.1f} min total)")

    # ── Extract my events only ──────────────────────────────────────────────
    my_events = events[
        (events[:, 0] >= my_start_sample) &
        (events[:, 0] <= my_end_sample)
    ]

    my_unique_codes, my_counts = np.unique(
        my_events[:, 2], return_counts=True
    )
    my_marker_dict = dict(
        zip(my_unique_codes.tolist(), my_counts.tolist())
    )
    row['question_61_count'] = my_marker_dict.get(61, 0)

    # ── Check for partner marker bleed ────────────────────────────────────────
    partner_bleed = {
        c: cnt for c, cnt in my_marker_dict.items()
        if c in PARTNER_UNIQUE
    }
    if partner_bleed:
        bleed_str = ', '.join(f"code {c}×{n}" for c, n in sorted(partner_bleed.items()))
        row['partner_bleed'] = bleed_str
        print(f"  Partner bleed: {bleed_str}")
        row['issues'].append(f"Partner markers in window: {bleed_str}")
    else:
        row['partner_bleed'] = 'None'
        print(f"  Partner bleed: None ✓")

    # ── Detect duplicate markers ──────────────────────────────────────────────
    threshold_samples = int(sfreq * DUPLICATE_THRESHOLD_SEC)
    duplicates_found  = []

    for check_code in [43, 40, 30, 61]:
        code_events = my_events[my_events[:, 2] == check_code]
        if len(code_events) < 2:
            continue
        samples = code_events[:, 0]
        gaps    = np.diff(samples)
        close   = np.where(gaps < threshold_samples)[0]
        for idx in close:
            t1      = samples[idx] / sfreq / 60
            t2      = samples[idx + 1] / sfreq / 60
            gap_sec = gaps[idx] / sfreq
            duplicates_found.append(
                f"Code {check_code} at {t1:.2f}min + {t2:.2f}min "
                f"(gap={gap_sec:.1f}s)"
            )

    if duplicates_found:
        dup_str = ' | '.join(duplicates_found)
        row['duplicate_markers'] = dup_str
        print(f"  Duplicates: {dup_str}")
        row['issues'].append(f"Duplicates: {dup_str}")
    else:
        row['duplicate_markers'] = 'None'
        print(f"  Duplicates: None ✓")

    # ── Verify expected marker counts ─────────────────────────────────────────
    print(f"\n  Marker verification:")
    print(f"  {'Code':<6} {'Name':<22} {'Found':<8} {'Expected':<12} Status")
    print(f"  {'-'*56}")

    marker_issues = []
    for code, (exp_min, exp_max) in MY_EXPECTED.items():
        found  = my_marker_dict.get(code, 0)
        name   = MY_MARKER_NAMES.get(code, f'code {code}')
        ok     = exp_min <= found <= exp_max
        status = 'OK' if ok else f'GOT {found}'
        print(f"  {code:<6} {name:<22} {found:<8} {exp_min}-{exp_max:<8}   {status}")
        if not ok:
            marker_issues.append(
                f"Marker {code} ({name}): expected {exp_min}-{exp_max}, got {found}"
            )

    row['issues'].extend(marker_issues)

    # ── Overall status ────────────────────────────────────────────────────────
    critical_ok = (
        my_marker_dict.get(43, 0) == 5 and
        my_marker_dict.get(61, 0) == 120 and
        my_marker_dict.get(71, 0) == 5
    )
    conditions_ok = all(
        my_marker_dict.get(c, 0) == 1
        for c in [21, 22, 23, 24, 25]
    )
    no_bleed = len(partner_bleed) == 0

    if critical_ok and conditions_ok and no_bleed and not duplicates_found:
        status = 'COMPLETE'
        rec    = 'INCLUDE'
    elif critical_ok and conditions_ok and not duplicates_found:
        status = 'MINOR ISSUES'
        rec    = 'INCLUDE — minor partner bleed only'
    elif critical_ok and no_bleed and not duplicates_found:
        status = 'USABLE'
        rec    = 'INCLUDE WITH CAUTION — condition markers off'
    elif critical_ok:
        status = 'USABLE'
        rec    = 'INCLUDE WITH CAUTION — investigate duplicates/bleed'
    else:
        status = 'INCOMPLETE'
        rec    = 'INVESTIGATE — critical markers wrong'

    row['status']         = status
    row['recommendation'] = rec
    row['issues']         = ' | '.join(row['issues']) if row['issues'] else 'None'

    print(f"\n  STATUS: {status}")
    print(f"  REC:    {rec}\n")

    summary_rows.append(row)
    del raw

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print(f"FINAL SUMMARY")
print(f"{'='*90}")
print(f"{'File':<28} {'Dur':>6} {'Win':>6} {'43s':>4} "
      f"{'61s':>5}  {'Status':<20} Recommendation")
print(f"{'-'*95}")

status_counts = {}
for r in summary_rows:
    dur = f"{r['duration_min']}m"  if r['duration_min']      else 'N/A'
    win = f"{r['my_window_min']}m" if r['my_window_min']  else 'N/A'
    c43 = str(r['encoding_43_count']) if r['encoding_43_count'] != '' else '-'
    c61 = str(r['question_61_count']) if r['question_61_count'] != '' else '-'
    print(f"{r['filename']:<28} {dur:>6} {win:>6} {c43:>4} "
          f"{c61:>5}  {r['status']:<20} {r['recommendation']}")
    s = r['status']
    status_counts[s] = status_counts.get(s, 0) + 1

print()
for s, c in sorted(status_counts.items()):
    print(f"  {s:<22}: {c}")

# ── Save ──────────────────────────────────────────────────────────────────────
timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = os.path.join(BDF_FOLDER, f'scan_summary_v4_{timestamp}.csv')

with open(output_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'filename', 'status', 'recommendation',
        'duration_min', 'my_window_min',
        'my_start_min', 'my_end_min',
        'encoding_43_count', 'question_61_count',
        'duplicate_markers', 'partner_bleed', 'issues'
    ])
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"\nSaved to: {output_path}")