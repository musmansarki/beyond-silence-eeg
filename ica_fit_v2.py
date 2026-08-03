#!/usr/bin/env python3
"""
ICA fitting — stage 1 of 2
===========================

Preprocesses each participant exactly as eeg_feature_extraction_v3.py does,
fits an ICA solution, and saves both the preprocessed raw and the ICA.

Disk: each preprocessed raw is roughly 200 MB at 512 Hz, so ~3.7 GB for
19 participants. If space is short, set DOWNSAMPLE_REVIEW = 128. The ICA
is still fitted on full-rate data; only the copy saved for visual review
is downsampled. Blinks are slow events and remain clearly visible.

Delete the ica_solutions folder once the review and the final v3 run
are complete.
"""

import mne
import numpy as np
import os
import json
from mne.preprocessing import ICA, annotate_muscle_zscore

BDF_FOLDER = '/Users/momosarki/Documents/eeg_analysis/eeg_data'
ICA_FOLDER = '/Users/momosarki/Documents/eeg_analysis/ica_solutions'
os.makedirs(ICA_FOLDER, exist_ok=True)

MANIFEST_PATH = os.path.join(ICA_FOLDER, 'manifest.json')

# Set to an integer (e.g. 128) to downsample the saved review copy.
# None keeps the full 512 Hz. Does not affect the ICA fit.
DOWNSAMPLE_REVIEW = 128

# Must match eeg_feature_extraction_v3.py exactly
FILTER_LOW, FILTER_HIGH = 1.0, 40.0
REFERENCE = 'average'
BAD_HIGH_RATIO, BAD_LOW_RATIO = 4.0, 0.25
MAX_BAD_FRACTION = 0.20
MUSCLE_THRESHOLD = 4.0
MUSCLE_FREQ = (110, 140)
MUSCLE_MIN_GOOD = 0.2
N_COMPONENTS = 20

NON_EEG = ['EXG1','EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8',
           'GSR1','GSR2','Erg1','Erg2','Resp','Plet','Temp']

CHANNEL_MAPPING = {
    'A1':'Fp1','A2':'AF3','A3':'F7','A4':'F3','A5':'FC1','A6':'FC5',
    'A7':'T7','A8':'C3','A9':'CP1','A10':'CP5','A11':'P7','A12':'P3',
    'A13':'Pz','A14':'PO3','A15':'O1','A16':'Oz','A17':'O2','A18':'PO4',
    'A19':'P4','A20':'P8','A21':'CP6','A22':'CP2','A23':'C4','A24':'T8',
    'A25':'FC6','A26':'FC2','A27':'F4','A28':'F8','A29':'AF4','A30':'Fp2',
    'A31':'Fz','A32':'Cz',
}

PARTICIPANTS = ['1','2','3','4','5','6','7','8','9','10','11',
                '13','14','15','16','17','18','19','20']


def detect_bad_channels(raw_filtered):
    """SD-ratio detection against the montage median. Filtered, pre-reference."""
    picks = mne.pick_types(raw_filtered.info, eeg=True)
    names = [raw_filtered.ch_names[i] for i in picks]
    sds = raw_filtered.get_data(picks=picks).std(axis=1)
    med = np.median(sds)
    bads, detail = [], {}
    for name, sd in zip(names, sds):
        ratio = sd / med if med > 0 else np.inf
        if ratio > BAD_HIGH_RATIO or ratio < BAD_LOW_RATIO:
            bads.append(name)
            detail[name] = round(float(ratio), 2)
    return bads, detail


def save_manifest(manifest):
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)


# ── Resume from existing manifest ─────────────────────────────────────────────
if os.path.exists(MANIFEST_PATH):
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    print(f"Existing manifest found: {len(manifest)} participants already done")
else:
    manifest = {}

# ── Recover any orphaned files not in the manifest ────────────────────────────
recovered = 0
for pid in PARTICIPANTS:
    if pid in manifest:
        continue
    raw_p = os.path.join(ICA_FOLDER, f'P{pid}-prep-raw.fif')
    ica_p = os.path.join(ICA_FOLDER, f'P{pid}-ica.fif')
    if os.path.exists(raw_p) and os.path.exists(ica_p):
        # Verify the raw is readable — a truncated write will fail here
        try:
            mne.io.read_raw_fif(raw_p, preload=False, verbose=False)
        except Exception:
            print(f"P{pid}: found corrupt raw, deleting")
            os.remove(raw_p)
            continue
        manifest[pid] = {
            'raw': raw_p, 'ica': ica_p,
            'bads_interpolated': [], 'bad_detail': {},
            'n_components': N_COMPONENTS,
            'note': 'recovered — bad channel list not recorded',
        }
        recovered += 1

if recovered:
    save_manifest(manifest)
    print(f"Recovered {recovered} participants from existing files")

print()

# ── Main loop ─────────────────────────────────────────────────────────────────
for pid in PARTICIPANTS:
    if pid in manifest and os.path.exists(manifest[pid]['raw']):
        print(f"P{pid}: already done — skipping")
        continue

    path = os.path.join(BDF_FOLDER, f"eeg_participant{pid}.bdf")
    if not os.path.exists(path):
        print(f"P{pid}: BDF not found — skipping")
        continue

    print(f"P{pid}: loading...")
    raw = mne.io.read_raw_bdf(path, preload=True, verbose=False)
    raw.drop_channels([c for c in NON_EEG if c in raw.ch_names])
    raw.rename_channels({o: n for o, n in CHANNEL_MAPPING.items()
                         if o in raw.ch_names})
    raw.set_montage(mne.channels.make_standard_montage('biosemi32'),
                    match_case=False, on_missing='ignore', verbose=False)

    # Bad channels on a filtered copy, leaving raw unfiltered for muscle detect
    tmp = raw.copy().filter(FILTER_LOW, FILTER_HIGH, verbose=False)
    bads, bad_detail = detect_bad_channels(tmp)
    del tmp

    n_eeg = len(mne.pick_types(raw.info, eeg=True))
    if len(bads) / n_eeg > MAX_BAD_FRACTION:
        print(f"P{pid}: {len(bads)}/{n_eeg} channels bad — skipping\n")
        del raw
        continue

    raw.info['bads'] = bads

    # Muscle detection on unfiltered data
    try:
        annot_muscle, _ = annotate_muscle_zscore(
            raw, ch_type='eeg', threshold=MUSCLE_THRESHOLD,
            min_length_good=MUSCLE_MIN_GOOD, filter_freq=MUSCLE_FREQ,
            verbose=False)
        raw.set_annotations(raw.annotations + annot_muscle)
    except Exception as e:
        print(f"P{pid}: muscle detection failed ({e})")

    raw.filter(FILTER_LOW, FILTER_HIGH, verbose=False)
    if bads:
        raw.interpolate_bads(reset_bads=True, verbose=False)
    raw.set_eeg_reference(REFERENCE, verbose=False)

    # Fit ICA at full rate. Muscle-annotated spans are excluded from the
    # fit only — legitimate here, unlike excluding them from a PSD, since
    # ICA needs a representative sample rather than a contiguous one.
    print(f"P{pid}: fitting ICA ({N_COMPONENTS} components)...")
    ica = ICA(n_components=N_COMPONENTS, random_state=42,
              max_iter='auto', verbose=False)
    ica.fit(raw, reject_by_annotation=True, verbose=False)

    raw_path = os.path.join(ICA_FOLDER, f'P{pid}-prep-raw.fif')
    ica_path = os.path.join(ICA_FOLDER, f'P{pid}-ica.fif')

    try:
        ica.save(ica_path, overwrite=True, verbose=False)

        if DOWNSAMPLE_REVIEW:
            raw_out = raw.copy().resample(DOWNSAMPLE_REVIEW, verbose=False)
        else:
            raw_out = raw
        raw_out.save(raw_path, overwrite=True, verbose=False)
        if DOWNSAMPLE_REVIEW:
            del raw_out

    except OSError as e:
        print(f"\nP{pid}: SAVE FAILED — {e}")
        for p in (raw_path, ica_path):
            if os.path.exists(p):
                os.remove(p)
        print("Partial files removed. Free disk space, or set")
        print("DOWNSAMPLE_REVIEW = 128, then re-run. Progress is saved.\n")
        del raw, ica
        break

    manifest[pid] = {
        'raw': raw_path,
        'ica': ica_path,
        'bads_interpolated': bads,
        'bad_detail': bad_detail,
        'n_components': N_COMPONENTS,
        'sfreq_review': DOWNSAMPLE_REVIEW or int(raw.info['sfreq']),
    }
    save_manifest(manifest)

    print(f"P{pid}: saved  (interpolated: {bads if bads else 'none'})\n")
    del raw, ica

print(f"\nManifest: {len(manifest)} participants in {ICA_FOLDER}")
print("Next: python ica_review.py")