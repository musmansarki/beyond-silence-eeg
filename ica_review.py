#!/usr/bin/env python3
"""
ICA review — stage 2 of 2
==========================

Loads each saved ICA solution, displays component topographies and time
courses, and records which components marked for exclusion.


"""

import mne
import os
import sys
import json
import matplotlib
matplotlib.use('Qt5Agg')          # interactive backend
import matplotlib.pyplot as plt
from mne.preprocessing import read_ica

ICA_FOLDER = '/Users/momosarki/Documents/eeg_analysis/ica_solutions'
EXCL_PATH = os.path.join(ICA_FOLDER, 'ica_exclusions.json')

with open(os.path.join(ICA_FOLDER, 'manifest.json')) as f:
    manifest = json.load(f)

if os.path.exists(EXCL_PATH):
    with open(EXCL_PATH) as f:
        exclusions = json.load(f)
else:
    exclusions = {}

# Which participants to review
if len(sys.argv) > 1:
    todo = [p for p in sys.argv[1:] if p in manifest]
else:
    todo = [p for p in sorted(manifest, key=int) if p not in exclusions]

if not todo:
    print("Nothing to review. Delete entries from ica_exclusions.json to redo.")
    sys.exit(0)

print(f"Reviewing {len(todo)} participants: {', '.join(todo)}")
print("Read the header of this file before starting.\n")

for pid in todo:
    entry = manifest[pid]
    print("=" * 60)
    print(f"Participant {pid}")
    print(f"  interpolated: {entry['bads_interpolated'] or 'none'}")
    print("=" * 60)

    raw = mne.io.read_raw_fif(entry['raw'], preload=True, verbose=False)
    ica = read_ica(entry['ica'], verbose=False)

    print("  Window 1: topographies. Click a title to toggle exclusion.")
    ica.plot_components(inst=raw, show=True)
    plt.show(block=True)

    print("  Window 2: source time courses. Close when done.")
    ica.plot_sources(raw, show=True)
    plt.show(block=True)

    selected = sorted(ica.exclude)
    print(f"\n  Selected for exclusion: {selected if selected else 'none'}")

    resp = input("  Accept? [y]es / [r]edo / [s]kip / [q]uit: ").strip().lower()

    if resp == 'q':
        print("Quitting. Progress saved.")
        break
    if resp == 's':
        print("  Skipped.\n")
        del raw, ica
        continue
    if resp == 'r':
        print("  Re-run this participant with: python ica_review.py", pid, "\n")
        del raw, ica
        continue

    exclusions[pid] = selected
    with open(EXCL_PATH, 'w') as f:
        json.dump(exclusions, f, indent=2)
    print(f"  Saved. ({len(exclusions)}/{len(manifest)} done)\n")

    del raw, ica

print(f"\nExclusions written to {EXCL_PATH}")
if exclusions:
    counts = [len(v) for v in exclusions.values()]
    print(f"Components excluded: mean {sum(counts)/len(counts):.1f}, "
          f"range {min(counts)}-{max(counts)}")