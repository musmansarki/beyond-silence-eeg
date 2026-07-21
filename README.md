# Beyond Silence: Testing the Stimulation Hypothesis of ADHD Through Controlled Auditory Environments

BSc thesis project (Artificial Intelligence, University of Groningen) investigating
whether background auditory stimulation differentially affects cognitive encoding
performance and cortical arousal in individuals with high vs. low ADHD symptoms.

Nineteen participants completed a within-subjects paired-associate memorisation task
in an artificial language (**Veltric**) across five auditory conditions — silence,
white noise, lofi instrumental, unfamiliar lyrical, and familiar lyrical — with
simultaneous EEG recording (BioSemi ActiveTwo, 32 channels, 512 Hz).

## Research question

Is there an optimal level of background auditory stimulation that maximises cognitive
encoding performance in individuals with high ADHD symptoms, and does this optimal
level differ from that of individuals with low ADHD symptoms, as reflected in both
behavioural performance and EEG arousal markers (frontal theta/beta ratio, posterior
beta power)?

## Pipeline

Scripts are meant to be run in this order:

1. **`veltric_experiment.py`** — PsychoPy (Coder) task. Presents the 5-block
   paired-associate paradigm, plays audio conditions, logs recall responses and
   focus ratings, and sends EEG trigger codes via serial port.
2. **`eeg_scanner_v2.py`** — QC pass over raw BDF recordings. Flags short recordings,
   checks marker counts, reports load/event errors before committing to full
   processing.
3. **`eeg_block_isolator.py`** — Isolates each participant's 5 clean block windows
   from the shared EEG session (recorded concurrently with a partner researcher's
   experiment) using trigger marker 43 (`ENCODING_END`, unique to this experiment)
   as an anchor. Outputs sample-level boundaries for baseline, encoding, and recall
   windows per block.
4. **`eeg_feature_extraction.py`** — Loads each participant's BDF file, renames
   BioSemi A1–A32 channels to 10-20 standard names, band-pass filters (1–35 Hz),
   runs ICA for artifact removal, and computes frontal theta/beta ratio (TBR) and
   posterior beta power per block (baseline + encoding, baseline-normalised).
   Merges in behavioural recall accuracy and focus ratings.
5. **`statistical_analysis_v2.py`** — Runs linear mixed models (condition × ADHD group)
   on recall accuracy, frontal TBR, and log-transformed posterior beta power, plus
   quadratic trend analysis and exploratory TBR–recall correlations.

## Setup

```bash
pip install -r requirements.txt
```

Each script expects a local data directory structure:

```
eeg_data/         # raw .bdf recordings
data/             # per-participant behavioural CSVs from veltric_experiment.py
block_epochs/     # output of eeg_block_isolator.py
features/         # output of eeg_feature_extraction.py
results/          # output of statistical_analysis_v2.py
```

Update the path constants near the top of each script (e.g. `BDF_FOLDER`,
`CSV_FOLDER`, `OUTPUT_FOLDER`) to point at these directories on your machine.

## Data

Raw EEG recordings and participant behavioural data are **not included** in this
repository (participant privacy, file size). Folder structure is provided so the
pipeline can be run against your own data of the same format.

## Notes

- EEG was recorded in shared sessions with a concurrent experiment run by
  a partner researcher; `eeg_scanner_v2.py` and `eeg_block_isolator.py` handle
  separating the two trigger streams using a marker unique to this experiment.
- ADHD group assignment used ASRS-v1.1 Part A self-report screening, not clinical
  diagnosis.
- Sample size (N=19) was underpowered for the hypothesised effect sizes; results
  should be read as directional rather than confirmatory (see thesis Discussion).
