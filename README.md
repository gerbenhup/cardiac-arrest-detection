# Cardiac arrest detection from wrist PPG and accelerometry

A pipeline for detecting loss of pulse (as in out-of-hospital cardiac arrest) from a wrist-worn
photoplethysmography (PPG) and accelerometry device, and the code of the two studies that use it:

- **BECA-S1**: healthy participants wore the device for several days and repeatedly occluded the arm with a
  blood-pressure cuff (simulated loss of pulse).
- **BECA-S2 + BECA-S6**: patients wore the device during cathlab ICD testing or ablation, with annotated
  loss-of-pulse events (BECA-S2), pooled with event-free daily-life wear of healthy volunteers on both wrists
  (BECA-S6) for the false-alarm rate.

The pipeline computes per-sample features with a causal per-participant normalisation and trains a causal
CNN-LSTM that scores every second. It evaluates the classifier with nested cross-validation, at segment level
(ROC, precision-recall) and at event level (FROC, with a post-hoc debounce), and models performance against
participant characteristics. Where the event is induced by a procedure that leaves an artifact (BECA-S2:
defibrillator shocks), it masks the artifacts and trains with a gap-input-gradient penalty against target
leakage, and selects the penalty weight with equivalence tests on synthetic-gap probes.

The study data are not included: they cannot be shared. A generated example study shows how everything runs.

## Quick start

```bash
conda create -n ohca python=3.9
conda activate ohca
pip install -r requirements.txt
pip install -e .

python experiments/example/example.py --phase all
```

`requirements.txt` pins the dependencies; `pip install -e .` installs the `ohca` package itself. Scripts are
run from the repository root. The example generates a small study (8 participants, 1.5 hours each) and runs every phase on it:
features, training at two gap-penalty weights, leakage probes, the weight sweep, debounce selection and all
analyses. It takes about two hours on a CPU. Its signals come from a seeded random generator, so its numbers
say nothing about real performance.

## Layout

```
ohca/                    the pipeline
  dataset.py             H5Dataset: the shared data format and the base class of every study's dataset
  segmentation.py        recording + episode list -> labelled segments
  features.py            37 features per sample, causal EMA z-score
  models.py              CNN-LSTM, Tversky loss, gap-input-gradient penalty
  cv.py                  nested grouped cross-validation with a successive-halving search
  scoring.py             event-level scoring (numba kernels), ROC/PR, Kaplan-Meier
  debounce.py            post-hoc debounce selection
  analyses/              ROC/PR, FROC, operating point, participant characteristics, time of day,
                         debounce survival, leakage probes, penalty-weight sweep, the combined figure
  pipeline.py            RunSpec (a study's settings) and the phases
experiments/
  example/example.py     generated example study
  BECA-S1/BECA-S1.py     raw-data reader and settings of BECA-S1
  BECA-S2-S6/BECA-S2-S6.py  raw-data readers, settings and example-signal figure of BECA-S2 + BECA-S6
```

Each study writes its data and outputs next to its script; all of them are git-ignored.

## Using your own data

A study is a script with two parts:

1. **A dataset class** derived from `ohca.dataset.H5Dataset`. Its `build()` reads the raw files and writes
   `dataset.h5` with `H5Dataset.write`, one group per participant and segment:

   ```
   /{participant}/{segment:05d}/X     float32 [T, 7]  PPG gain, LED power, ambient, PPG, acc x, y, z (32 Hz)
   /{participant}/{segment:05d}/ANN   int8    [T]     +1 event, 0 monitored, -1 gapped (masked artifact)
   /{participant}/{segment:05d}/time  float64 [T]     epoch seconds
   ```

   Accelerometry is in counts at 512 LSB/g. `ohca.segmentation.segment` cuts a continuous recording and its
   episode list into segments that start with event-free signal and end at an event, or contain none.
   `read_metadata()` returns the participant characteristics, indexed by participant id.
2. **A `RunSpec`** with the study's settings, passed to `ohca.pipeline.run_phases`.

`experiments/example/example.py` is the smallest complete example: copy it and replace `build()`.

## Running

```bash
python <study>.py --phase {build,features,train,finalists,probes,sweep,debounce,aggregate,all} \
                  [--weight W|W1,W2|all] [--folds i/n|i,j,k] [--gpu k] [--gpu-mem MB] [--threads N]
```

Every phase is resumable: work already on disk is skipped (delete a file to recompute it). `--folds i/n`
runs every n-th fold starting at `i`, so `n` processes can split the folds. A training worker claims each
fold it starts (`cv/fold_XXX/claim`), so several workers can share one fold list; `--gpu-mem` caps a worker's
GPU memory so that several fit on one GPU. A claim left by a killed worker on another machine has to be
deleted before that fold can run again. The `finalists` phase is a helper next to the training workers: it
trains the finalists of folds whose search rungs are done, in parallel with the fold's own worker.

Studies with gap penalties keep one run folder per weight (`weights/w<weight>/`); `--weight` takes one weight,
a comma-separated list (run in that order) or `all`.

### BECA-S1

```bash
python experiments/BECA-S1/BECA-S1.py --phase features                    # builds dataset.h5 first
python experiments/BECA-S1/BECA-S1.py --phase train    --folds 0/4 --gpu 0   # one process per GPU, i = 0..3
python experiments/BECA-S1/BECA-S1.py --phase debounce --folds 0/16          # CPU only, i = 0..15
python experiments/BECA-S1/BECA-S1.py --phase aggregate
```

Leave-one-participant-out; participants with at least five occlusions. Raw data, under
`experiments/BECA-S1/`: `records.xlsx` (participant characteristics) and `raw_data/` with
`PDL/<pid>/FT_PPG_*.txt, FT_ACC_*.txt`, `Annotations/annotations/consensus/<pid>.json` (the participants are
those with a consensus annotation) and `soft labels.xlsx` (participant-reported occlusions).

### BECA-S2 + BECA-S6

Every BECA-S2 participant with at least one event is one fold; the BECA-S6 persons (both wrists together)
and the event-free BECA-S2 participants are dealt over the folds, and validation takes the next five folds.
The classifier is trained at nine gap-penalty weights (0, 1, 10, ..., 10^7). Weight 0 runs the nested
hyperparameter search; every other weight trains, per fold, the configuration that fold's weight-0 search
selected. The detector is armed once a participant has 2 minutes of history and each segment 10 s. Training
batches are half positive and half negative, negatives drawn in proportion to each record's negative time.
Sensitivity comes from the BECA-S2 events, false alarms from BECA-S6.

```bash
S=experiments/BECA-S2-S6/BECA-S2-S6.py
python $S --phase features                                     # builds dataset.h5 first
python $S --phase train     --weight 0 --folds 0/4 --gpu 0      # then the other weights: --weight all
python $S --phase finalists --weight 0 --folds 0/4 --gpu 1      # optional helpers for the weight-0 search
python $S --phase probes    --weight all
python $S --phase sweep
python $S --phase debounce  --weight all --folds 0/8
python $S --phase aggregate --weight all
python $S --phase example
```

The sweep selects the penalty weight: per weight, the arrest and arrest-free leakage effects of the
synthetic-gap probes are tested for equivalence to zero (within 5% of the discriminative range), by
fixed-sequence testing from the largest weight down; a weight whose discriminative range is below a quarter
of weight 0's has collapsed and is skipped. The selected weight (10^5) is compared with weight 0 in a paired
PR-AUC test. Every weight gets the same analyses; the paper reports the selected one.

Participant characteristics are modelled per cohort with a quasi-likelihood GLM and reported as average
marginal effects: a fractional logit of each BECA-S2 patient's PR-AUC (percentage points) and a Poisson GLM
of each BECA-S6 person's false alarms, both wrists together, with log monitored hours as offset (false alarms
per hour). The alarms are counted at each fold's selected threshold and debounce length. Confidence intervals
come from a bootstrap over the cross-validation folds, whose members share a threshold.

Raw data, under `experiments/BECA-S2-S6/raw_data/`:
- `BECA-S2/`: `records.xlsx` (sheets `records` and `wrist_side_selection`), `1_raw_and_alignment/data/`
  (`records.xlsx`, `elan_side_selection.xlsx`, `segment_offsets.csv`, and per patient
  `Elan_<L|R>/FT_PPG_*.txt, FT_ACC_*.txt` and `Cath/metadata.inf, data.txt, segment_timestamps_corrected.csv`)
  and `2_annotation_and_packaging/annotations/gerben/<pid>.json`. The BECA-S2 `dataset.h5` is built there
  first. `example_participant.txt` holds the pseudonym of the patient in the example-signals figure.
- `BECA-S6/`: `data/<pid><L|R>/FT_PPG_*.txt, FT_ACC_*.txt`, `annotations/alignment/<pid>.json`,
  `annotations/exp23/<pid><L|R>.json` and `beca-s6_records.xlsx`.

## Outputs

Figures are written as PNG and PDF.

| BECA-S1 | File (under `experiments/BECA-S1/`) |
|---|---|
| ROC and PR curves | `figures/roc_pr_curves` |
| Participant characteristics (PR-AUC) | `figures/characteristics_prauc`, `results/characteristics_prauc_coef.csv`, `results/characteristics_prauc_vif.csv` |
| FROC and selected debounce lengths | `figures/froc_debounce_stability`, `results/froc_far_table_censoring_{naive,corrected}.csv` |
| FAR versus debounce length | `figures/debounce_survival`, `results/debounce_survival.csv`, `results/km_table_{60,80,95}.csv` |
| False alarms by time of day | `figures/time_block`, `results/time_block_pairwise.csv`, `results/time_block_omnibus.csv` |

| BECA-S2 + BECA-S6 | File (under `experiments/BECA-S2-S6/`) |
|---|---|
| Example signals | `figures/example_event_signals` |
| Target leakage per penalty weight | `sweep/gap_penalty_effects`, `sweep/table.csv` |
| PR-AUC, selected weight versus weight 0 (paired bootstrap) | `sweep/prauc_comparison.csv` |
| ROC, PR and FROC (FAR on BECA-S6 and on BECA-S2) | `weights/w100000/figures/roc_pr_froc` |
| FAR at sensitivity | `weights/w100000/results/froc_far_table_censoring_{naive,corrected}[_far_BECA-S2].csv` |
| Operating point (sensitivity, FAR per cohort) | `weights/w100000/results/operating_point.csv` |
| Participant characteristics: BECA-S2 PR-AUC (percentage points), BECA-S6 false alarms (alarms/h) | `weights/w100000/figures/characteristics`, `weights/w100000/results/characteristics_prauc_{coef,vif}.csv`, `weights/w100000/results/characteristics_far_BECA-S6_{coef,vif}.csv` |

Per-participant AUC-ROC and PR-AUC, the curve tables and the selected debounce lengths are in each `results/`
folder as well.

## Reproducibility

`requirements.txt` pins the tested versions (TensorFlow 2.10 needs a matching CUDA/cuDNN for GPU training).
Training on a GPU is not bit-reproducible, so retraining gives slightly different models. Given the trained
models, debounce selection, the analyses and the leakage probes are deterministic. The probes run in
TensorFlow's deterministic mode; another GPU model can change single probe scores by about 1e-3.

## License

MIT, see `LICENSE`.
