from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import Parallel, delayed

from . import cli, debounce, environment, features, plotting, scoring
from .analyses import probes, summary_figure, sweep
from .analyses.characteristics import CharacteristicsAnalysis, FARCharacteristicsAnalysis
from .analyses.froc import FROCAnalysis
from .analyses.operating_point import OperatingPointAnalysis
from .analyses.roc_pr import PRCAnalysis, ROCAnalysis
from .analyses.survival import DebounceSurvivalAnalysis
from .analyses.time_of_day import TimeBlockFARAnalysis
from .cv import NestedCrossValidationManager, SearchSpec, arm
from .dataset import H5Dataset, test_by_fold, test_by_participant

SENS_TARGETS = (0.6, 0.8, 0.95)
TB_WORKERS   = 32
PHASES       = ("build", "features", "train", "debounce", "aggregate")
GAP_PHASES   = ("probes", "sweep")
METHODS      = ("censoring_naive", "censoring_corrected")


class RunSpec(NamedTuple):
    name:                str
    base_dir:            Path
    dataset_cls:         Type[H5Dataset]
    min_episodes:        int                 # participants with fewer events are excluded
    far_min:             float               # finest nonzero false-alarm rate on the FAR axis
    micro_average:       bool                # event-count (micro) instead of participant (macro) weights
    covariates:          Tuple[str, ...]
    classifier_label:    str
    panel_label_case:    str                 # "upper" -> (A), "lower" -> (a)
    search:              SearchSpec          # per-fold hyperparameter search
    gapped:              bool                # masked artifacts: zeroed feature rows plus a gap channel
    time_of_day:         bool                # false-alarm rate by time of day
    combined_figure:     bool                # one ROC/PR/FROC/subgroup figure instead of separate ones
    gap_penalties:       Tuple[int, ...] = ()    # gap-input-gradient penalty weights; () trains unpenalized
    pp_workers:          int             = 12
    n_bootstrap:         int             = 2000  # replicates behind every reported confidence interval
    far_covariates:      Tuple[str, ...] = ()    # false-alarm model of the event-free cohorts
    warm_start:          bool  = False           # normaliser: running average until the EMA's memory is reached
    steady_start:        bool  = False           # filters start in steady state at a segment's first sample
    available_history:   bool  = False           # change-over-d features use the history there is before d
    arming_s:            float = 0.0             # detector armed once a participant has this much history
    segment_warmup_s:    float = 0.0             # ... and each segment this much of its own
    crop_step_s:         int   = 1               # training crop lengths MIN_NS..MAX_NS in these steps
    debounce_evaluable:  float = 0.0             # fraction of validation events that must last a debounce length
    far_cohorts:         Optional[Tuple[str, ...]] = None  # debounce selection's FAR from these cohorts only
    negative_sampling:   str   = "record"        # negative crops uniform per record, or "time"-proportional
    config_weight:       Optional[int] = None    # every other weight trains this weight's per-fold configuration
    figure_width_in:     float = plotting.DOUBLE_COL_IN  # printed width of the full-width figures


def run_dir(spec: RunSpec, weight: Optional[int]) -> Path:
    return spec.base_dir if weight is None else spec.base_dir / "weights" / f"w{weight}"


def manager(spec: RunSpec, h5: H5Dataset, weight: Optional[int]) -> NestedCrossValidationManager:
    shared = spec.config_weight is not None and weight != spec.config_weight
    return NestedCrossValidationManager(spec.base_dir / "features", h5.fold_groups(), run_dir(spec, weight) / "cv",
                                        spec.search, float(weight or 0), cohorts(h5)[0],
                                        spec.negative_sampling, spec.arming_s, spec.segment_warmup_s,
                                        spec.crop_step_s, run_dir(spec, spec.config_weight) / "cv" if shared else None)


def cohorts(h5: H5Dataset) -> Tuple[Dict[str, str], Callable[[str], bool], List[str], List[str]]:
    # Cohort per participant, whether a participant is in a cohort with events, the cohorts with events and
    # the others (empty lists for a single cohort).
    cohort_of = {p: h5.cohort(p) for p in h5.participant_ids}
    names     = sorted(set(cohort_of.values()))
    with_events = ([] if len(names) < 2 else
                   [c for c in names if any(h5.n_episodes[p] > 0 for p in h5.participant_ids if cohort_of[p] == c)])

    def in_event(pid: str) -> bool:
        return not with_events or cohort_of[pid] in with_events

    return cohort_of, in_event, with_events, [] if len(names) < 2 else [c for c in names if c not in with_events]


def persons(h5: H5Dataset) -> Dict[str, str]:
    return {p: h5.person(p) for p in h5.participant_ids}


def load_results(spec: RunSpec, mgr: NestedCrossValidationManager, folds: Optional[List[int]] = None) -> None:
    # With arming, the scores before the detector is armed are cut.
    mgr.load_results(folds)
    for r in mgr.results:
        if mgr.armed:
            r["test_scores"] = arm(r["test_scores"], spec.arming_s, spec.segment_warmup_s)
            r["val_scores"]  = arm(r["val_scores"], spec.arming_s, spec.segment_warmup_s)


def extract_features(spec: RunSpec, h5: H5Dataset) -> None:
    out     = spec.base_dir / "features"
    out.mkdir(parents=True, exist_ok=True)
    missing = [p for p in h5.participant_ids if not (out / f"{p}.pkl").exists()]
    if not missing:
        return
    environment.log(f"Preprocessing {len(missing)} participants, {spec.pp_workers} workers ...")
    stream = Parallel(n_jobs=spec.pp_workers, backend="loky", max_nbytes=None, return_as="generator")(
        delayed(features.preprocess_participant)(pid, h5.get([pid]), out / f"{pid}.pkl", spec.gapped,
                                                 spec.warm_start, spec.steady_start, spec.available_history)
        for pid in missing)
    for k, (pid, n_segs) in enumerate(stream, 1):
        environment.log(f"  preprocessed {k}/{len(missing)}: {pid} ({n_segs} segments)")


def select_debounce(spec: RunSpec, mgr: NestedCrossValidationManager, folder: Path, folds: List[int]) -> None:
    # The validation units are participants, or the fold groups when a fold holds out several.
    out = folder / "debounce"
    out.mkdir(parents=True, exist_ok=True)
    far_grid = scoring.far_grid(spec.far_min)
    unit_of  = (None if all(len(g) == 1 for g in mgr.groups)
                else {p: f"group_{gi:03d}" for gi, g in enumerate(mgr.groups) for p in g})
    far_pids = None if spec.far_cohorts is None else {p for p, c in mgr.cohort_of.items() if c in spec.far_cohorts}
    load_results(spec, mgr, folds)
    if len(mgr.results) < len(folds):
        raise SystemExit("Not every fold is trained yet; run --phase train first.")
    for r in mgr.results:
        path = out / f"fold_{r['fold']:03d}.json"
        if path.exists():
            continue
        row = {"test": r["test"]}
        far, d, thr = debounce.censoring_naive(r["val_scores"], r["fold"], far_grid, spec.micro_average, unit_of,
                                               spec.debounce_evaluable, far_pids)
        row.update(debounce_censoring_naive=d, val_far_censoring_naive=far, threshold_censoring_naive=thr)
        far, d, thr = debounce.censoring_corrected(r["val_scores"], r["fold"], far_grid, spec.micro_average,
                                                   unit_of, spec.debounce_evaluable, far_pids)
        row.update(debounce_censoring_corrected=d, val_far_censoring_corrected=far, threshold_censoring_corrected=thr)
        path.write_text(json.dumps(row))
        environment.log(f"[fold {r['fold'] + 1}/{mgr.n_runs}] {row}")


Cohorts = Tuple[Dict[str, str], Callable[[str], bool], List[str], List[str]]


def _debounce_table(mgr: NestedCrossValidationManager, folder: Path) -> pd.DataFrame:
    paths = [folder / "debounce" / f"fold_{i:03d}.json" for i in range(mgr.n_runs)]
    if not all(p.exists() for p in paths):
        raise SystemExit("Debounce selection missing; run --phase debounce first.")
    return pd.DataFrame([json.loads(p.read_text()) for p in paths])


def _segment_level(res: List[Dict[str, Any]], in_event: Callable[[str], bool], n_boot: int, cache: Path,
                   results: Path) -> Tuple[ROCAnalysis, PRCAnalysis]:
    # With several cohorts, ROC and PR use the participants of the cohorts with events.
    by_pid = {p: s for p, s in test_by_participant(res).items() if in_event(p)}
    environment.log("ROC")
    roc = ROCAnalysis.cached(cache / "roc.pkl", lambda: ROCAnalysis.create(by_pid, n_boot))
    roc.curve_table().to_csv(results / "roc_curve.csv", index=False)
    roc.per_participant_table().to_csv(results / "auc_roc_per_participant.csv")
    environment.log("Precision-recall")
    prc = PRCAnalysis.cached(cache / "prc.pkl", lambda: PRCAnalysis.create(by_pid, n_boot))
    prc.curve_table().to_csv(results / "prc_curve.csv", index=False)
    prc.per_participant_table().to_csv(results / "prauc_per_participant.csv")
    return roc, prc


def _frocs(spec: RunSpec, res: List[Dict[str, Any]], deb: pd.DataFrame, groups: Cohorts, cache: Path, results: Path
           ) -> Tuple[Dict[str, FROCAnalysis], Dict[str, Optional[Tuple[str, ...]]]]:
    # Per-fold curves, averaged. With several cohorts, sensitivity comes from the event cohorts and the FAR
    # from far_cohorts (default: the event-free ones), plus one curve per event cohort on its own FAR.
    cohort_of, in_event, event_cohorts, free_cohorts = groups
    far_sets: Dict[str, Optional[Tuple[str, ...]]] = {"": None}
    if event_cohorts:
        far_sets = {"": tuple(spec.far_cohorts or free_cohorts)}
        far_sets.update({f"_far_{c}": (c,) for c in event_cohorts if (c,) != far_sets[""]})
    frocs: Dict[str, FROCAnalysis] = {}
    for method in METHODS:
        d_map = dict(enumerate(deb[f"debounce_{method}"]))
        for tag, far_c in far_sets.items():
            environment.log(f"FROC, {method}{tag}")
            key       = method + tag
            units     = test_by_fold(res) if far_c is None else test_by_fold(res, in_event)
            far_units = None if far_c is None else test_by_fold(res, lambda p, far_c=far_c: cohort_of[p] in far_c)
            frocs[key] = FROCAnalysis.cached(cache / f"froc_{key}.pkl", lambda: FROCAnalysis.create(
                units, d_map, method == "censoring_corrected", scoring.far_grid(spec.far_min), spec.micro_average,
                spec.n_bootstrap, far_units))
            frocs[key].curve_table().to_csv(results / f"froc_{key}.csv", index=False)
            frocs[key].far_table(spec.n_bootstrap).to_csv(results / f"froc_far_table_{key}.csv", index=False)
    return frocs, far_sets


def _false_alarm_characteristics(spec: RunSpec, h5: H5Dataset, res: List[Dict[str, Any]], deb: pd.DataFrame,
                                 groups: Cohorts, cache: Path, results: Path
                                 ) -> List[Tuple[str, FARCharacteristicsAnalysis]]:
    cohort_of, _, _, free_cohorts = groups
    environment.log("Operating point")
    op = OperatingPointAnalysis.cached(cache / "operating_point.pkl", lambda:
        OperatingPointAnalysis.create(res, deb, cohort_of, spec.n_bootstrap, persons(h5)))
    op.table.to_csv(results / "operating_point.csv", index=False)
    out = []
    for c in free_cohorts:
        environment.log(f"Participant characteristics (false alarms, {c})")
        counts, hours, fold_of = op.counts(c)
        fch = FARCharacteristicsAnalysis.cached(cache / f"characteristics_far_{c}.pkl", lambda:
            FARCharacteristicsAnalysis.create(counts, hours, fold_of, h5.metadata, spec.far_covariates,
                                              spec.n_bootstrap))
        fch.coefficients.to_csv(results / f"characteristics_far_{c}_coef.csv")
        pd.Series(fch.vif, name="vif").to_csv(results / f"characteristics_far_{c}_vif.csv")
        out.append((c, fch))
    return out


def _separate_figures(spec: RunSpec, roc: ROCAnalysis, prc: PRCAnalysis, ch: CharacteristicsAnalysis,
                      figures: Path) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(spec.figure_width_in, 3.4), dpi=plotting.DPI)
    roc.plot(ax_a, spec.classifier_label)
    prc.plot(ax_b, spec.classifier_label)
    for i, ax in enumerate((ax_a, ax_b)):
        ax.set_title(plotting.panel_letter(i, spec.panel_label_case), loc="left", fontsize=plotting.FONT_PANEL,
                     fontweight="bold")
        ax.set_aspect("equal")
    fig.tight_layout()
    plotting.save(fig, figures / "roc_pr_curves")

    fig, ax = plt.subplots(figsize=(plotting.SINGLE_COL_IN, 2.6), dpi=plotting.DPI)
    ch.plot(ax)
    fig.tight_layout()
    plotting.save(fig, figures / "characteristics_prauc")


def _debounce_figure(spec: RunSpec, frocs: Dict[str, FROCAnalysis], deb: pd.DataFrame, figures: Path) -> None:
    # FROC curves of both methods and the debounce lengths the folds selected.
    fig, (ax_f, ax_h) = plt.subplots(1, 2, figsize=(spec.figure_width_in, 2.8), dpi=plotting.DPI)
    bins = np.arange(0.5, debounce.DEBOUNCE_GRID[-1] + 1.5, 1.0)
    for method, color, ls in (("censoring_naive", "tab:blue", "-"), ("censoring_corrected", "tab:orange", "--")):
        frocs[method].plot(ax_f, color=color, ls=ls)
    for method, color in (("censoring_naive", "tab:blue"), ("censoring_corrected", "tab:orange")):
        ax_h.hist(deb[f"debounce_{method}"], bins=bins, color=color, edgecolor="k", lw=0.4, alpha=0.75,
                  label=frocs[method].label)
    for i, ax in enumerate((ax_f, ax_h)):
        ax.set_title(plotting.panel_letter(i, spec.panel_label_case), loc="left", fontsize=plotting.FONT_PANEL,
                     fontweight="bold")
    ax_h.set_xlabel("Debounce length (s)", fontsize=plotting.FONT_AXIS)
    ax_h.set_ylabel("Number of selected folds", fontsize=plotting.FONT_AXIS)
    ax_h.tick_params(labelsize=plotting.FONT_TICK)
    ax_h.grid(True, alpha=0.3, linewidth=0.4)
    ax_h.set_ylim(0, max(p.get_height() for p in ax_h.patches) * 1.35)
    ax_h.legend(fontsize=plotting.FONT_LEGEND, loc="upper center", framealpha=0.9)
    fig.tight_layout()
    plotting.save(fig, figures / "froc_debounce_stability")


def _debounce_survival(spec: RunSpec, res: List[Dict[str, Any]], cache: Path, results: Path, figures: Path) -> None:
    environment.log("Debounce survival (Kaplan-Meier)")
    ds = DebounceSurvivalAnalysis.cached(cache / "debounce_survival.pkl", lambda: DebounceSurvivalAnalysis.create(
        test_by_fold(res), scoring.far_grid(spec.far_min), SENS_TARGETS, spec.n_bootstrap))
    ds.table().to_csv(results / "debounce_survival.csv", index=False)
    for target in ds.sens_targets:
        ds.km_table(target, spec.n_bootstrap).to_csv(results / f"km_table_{int(target * 100)}.csv", index=False)
    plotting.save(ds.plot(spec.figure_width_in), figures / "debounce_survival")


def _time_of_day(spec: RunSpec, h5: H5Dataset, res: List[Dict[str, Any]], deb: pd.DataFrame, naive: FROCAnalysis,
                 cache: Path, results: Path, figures: Path) -> None:
    budget = dict(zip(SENS_TARGETS, naive.far_at_targets(np.array(SENS_TARGETS)).tolist()))
    environment.log(f"Time of day, cohort FAR budgets {budget}")
    tb = TimeBlockFARAnalysis.cached(cache / "time_block.pkl", lambda:
        TimeBlockFARAnalysis.create(res, h5, dict(enumerate(deb["debounce_censoring_naive"])),
                                    budget, SENS_TARGETS, spec.n_bootstrap, TB_WORKERS))
    tb.pairwise_table.to_csv(results / "time_block_pairwise.csv", index=False)
    tb.omnibus_table().to_csv(results / "time_block_omnibus.csv", index=False)
    plotting.save(tb.plot(spec.figure_width_in), figures / "time_block")


def aggregate(spec: RunSpec, h5: H5Dataset, mgr: NestedCrossValidationManager, folder: Path) -> None:

    cache, results, figures = folder / "cache", folder / "results", folder / "figures"
    results.mkdir(parents=True, exist_ok=True)
    load_results(spec, mgr)
    if len(mgr.results) < mgr.n_runs:
        raise SystemExit(f"Only {len(mgr.results)}/{mgr.n_runs} folds trained; run --phase train first.")
    deb = _debounce_table(mgr, folder)
    deb.to_csv(results / "debounce_selection.csv", index=False)

    res    = mgr.results
    groups = cohorts(h5)
    roc, prc = _segment_level(res, groups[1], spec.n_bootstrap, cache, results)
    frocs, far_sets = _frocs(spec, res, deb, groups, cache, results)

    environment.log("Participant characteristics (PR-AUC)")
    ch = CharacteristicsAnalysis.cached(cache / "characteristics_prauc.pkl", lambda:
        CharacteristicsAnalysis.create(prc.prauc_per_participant, h5.metadata, spec.covariates, spec.n_bootstrap))
    ch.coefficients.to_csv(results / "characteristics_prauc_coef.csv")
    pd.Series(ch.vif, name="vif").to_csv(results / "characteristics_prauc_vif.csv")

    curves = [(frocs["censoring_naive"], spec.classifier_label)]
    far_ch: List[Tuple[str, FARCharacteristicsAnalysis]] = []
    if groups[2]:
        far_ch = _false_alarm_characteristics(spec, h5, res, deb, groups, cache, results)
        # Labelled by the cohort the false alarms come from; sensitivity is always the event cohorts'.
        curves = ([(frocs["censoring_naive"], f"FAR evaluated on {', '.join(far_sets[''])}")]
                  + [(frocs[f"censoring_naive{t}"], f"FAR evaluated on {c[0]}") for t, c in far_sets.items() if t])
    if spec.combined_figure:
        plotting.save(summary_figure.create(roc, prc, curves, spec.classifier_label, spec.panel_label_case,
                                            spec.figure_width_in), figures / "roc_pr_froc")
        panels = ([(ch.coefficients, CharacteristicsAnalysis.PANEL_XLABEL)]
                  + [(f.coefficients, FARCharacteristicsAnalysis.XLABEL) for _, f in far_ch])
        plotting.save(summary_figure.characteristics(panels, spec.panel_label_case, spec.figure_width_in),
                      figures / "characteristics")
    else:
        _separate_figures(spec, roc, prc, ch, figures)
    _debounce_figure(spec, frocs, deb, figures)
    _debounce_survival(spec, res, cache, results, figures)
    if spec.time_of_day:
        _time_of_day(spec, h5, res, deb, frocs["censoring_naive"], cache, results, figures)
    environment.log(f"Done. AUC-ROC={roc.auc_mean:.4f} PR-AUC={prc.prauc_mean:.4f}")


def run_probes(mgr: NestedCrossValidationManager, folder: Path) -> None:
    out  = folder / "results"
    out.mkdir(parents=True, exist_ok=True)
    mgr.load_results()
    if len(mgr.results) < mgr.n_runs:
        raise SystemExit("Not every fold is trained yet; run --phase train first.")
    for name, fn in (("probes_arrest_free.csv", probes.arrest_free), ("probes_arrest.csv", probes.arrest)):
        if not (out / name).exists():
            fn(mgr).to_csv(out / name, index=False)


def run_sweep(spec: RunSpec, h5: H5Dataset) -> None:
    cohort_of, in_event, _, _ = cohorts(h5)
    inputs = {}
    for w in spec.gap_penalties:
        folder = run_dir(spec, w)
        mgr    = manager(spec, h5, w)
        load_results(spec, mgr)
        prc = PRCAnalysis.cached(folder / "cache" / "prc.pkl", lambda: PRCAnalysis.create(
            {p: s for p, s in test_by_participant(mgr.results).items() if in_event(p)}, spec.n_bootstrap))
        inputs[w] = (mgr.results, pd.read_csv(folder / "results" / "probes_arrest.csv"),
                     pd.read_csv(folder / "results" / "probes_arrest_free.csv"), prc)
    table, comparison = sweep.select(inputs, cohort_of, persons(h5))
    out = spec.base_dir / "sweep"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "table.csv", index=False)
    comparison.to_csv(out / "prauc_comparison.csv", index=False)
    plotting.save(sweep.plot(table, comparison, spec.figure_width_in), out / "gap_penalty_effects")


def run_phase(phase: str, spec: RunSpec, h5: H5Dataset, weights: List[Optional[int]], folds: Callable[[int], List[int]],
              extra: Dict[str, Callable[[RunSpec, H5Dataset], None]]) -> None:
    if phase in extra:
        extra[phase](spec, h5)
        return
    if phase in ("features", "train", "finalists", "probes"):
        extract_features(spec, h5)
    if phase == "sweep":
        run_sweep(spec, h5)
    if phase in ("build", "features", "sweep"):
        return
    for weight in weights:
        folder = run_dir(spec, weight)
        mgr    = manager(spec, h5, weight)
        if phase == "train":
            mgr.train(folds(mgr.n_runs))
        elif phase == "finalists":
            # Helper next to the fold workers: trains finalists in parallel with them.
            mgr.train_finalists(folds(mgr.n_runs))
        elif phase == "debounce":
            select_debounce(spec, mgr, folder, folds(mgr.n_runs))
        elif phase == "aggregate":
            aggregate(spec, h5, mgr, folder)
        elif phase == "probes":
            # Deterministic GPU kernels, so the probes (and the weight selection) are reproducible.
            tf.config.experimental.enable_op_determinism()
            run_probes(mgr, folder)


def run_phases(spec: RunSpec, extra_phases: Optional[Dict[str, Callable[[RunSpec, H5Dataset], None]]] = None) -> None:

    # "all" runs every phase in dependency order: the sweep needs every weight's probes.
    extra  = extra_phases or {}
    phases = PHASES[:3] + (GAP_PHASES if spec.gap_penalties else ()) + PHASES[3:] + tuple(extra)
    args   = cli.parse(phases + ("all", "finalists"), bool(spec.gap_penalties))
    todo   = list(phases) if args.phase == "all" else [args.phase]
    if args.phase == "debounce":
        environment.pin_threads(1)
    if any(p in ("train", "finalists", "probes") for p in todo):
        environment.set_gpu(args.gpu, args.gpu_mem, args.threads)

    h5 = spec.dataset_cls(spec.base_dir, spec.min_episodes, spec.pp_workers)
    environment.log(f"[{spec.name}] {len(h5.participant_ids)} participants included, "
                    f"{len(h5.excluded)} excluded {h5.excluded}")
    if not spec.gap_penalties:
        weights: List[Optional[int]] = [None]
    elif args.weight in (None, "all"):
        # Penalized training is several times slower: workers sharing a fold list take those jobs first,
        # after the weight whose configurations the others train.
        weights = sorted(spec.gap_penalties, key=lambda w: (w != spec.config_weight, -w))
    else:
        weights = [int(v) for v in str(args.weight).split(",")]   # in the order given
        unknown = [w for w in weights if w not in spec.gap_penalties]
        if unknown:
            raise SystemExit(f"--weight {unknown}: not among the gap penalties {spec.gap_penalties}")
    for phase in todo:
        run_phase(phase, spec, h5, weights, args.folds, extra)
