from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import environment, labels, plotting, resampling
from ..dataset import test_by_participant
from .roc_pr import PRCAnalysis

# Penalty-weight selection: a weight passes when its arrest and arrest-free probe effects are both
# equivalent to zero within +-TOL_PCT % of the discriminative range. Fixed-sequence testing from the largest
# weight down; a weight whose range is below DEGENERATE_BELOW times weight 0's has collapsed and is skipped.

TOL_PCT          = 5.0
ALPHA            = 0.05
N_BOOTSTRAP      = 20000
DEGENERATE_BELOW = 0.25
FIG_HEIGHT_IN    = 135.0 / 25.4
Y_PAD            = 0.05   # effect panels: margin beyond the error bars, as a fraction of their span


def score_sums(results: List[Dict[str, Any]]) -> Dict[str, Tuple[float, int, float, int]]:
    # Per participant: sum of per-arrest mean scores, n arrests, sum of arrest-free scores, n seconds.
    sums = {}
    for pid, segments in test_by_participant(results).items():
        ev, ng = [], []
        for _, _, z, ann in segments:
            z, ann = np.asarray(z), np.asarray(ann)
            ev += [float(np.mean(z[lo:hi])) for lo, hi in labels.regions(labels.positive(ann)) if hi > lo]
            ng.append(z[ann == labels.MONITOR])
        ng = np.concatenate(ng) if ng else np.array([])
        sums[pid] = (float(np.sum(ev)), len(ev), float(np.sum(ng)), len(ng))
    return sums


def equivalence(vals: Dict[str, float], sums: Dict[str, Tuple[float, int, float, int]], pids: List[str],
                tol: float, n_bootstrap: int, seed: int = 0) -> Dict[str, Any]:
    # Equivalence of the mean probe effect (vals, per participant) to zero within +-tol % of the
    # discriminative range (mean arrest minus arrest-free score, from sums); participant bootstrap.
    pids  = list(pids)
    probe = np.array([vals.get(p, np.nan) for p in pids], float)
    se, ne, sn, nn = (np.array([sums[p][k] for p in pids], float) for k in range(4))
    rng = np.random.default_rng(seed)
    n   = len(pids)
    eff = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        i = rng.integers(0, n, n)
        v = probe[i]
        v = v[~np.isnan(v)]
        range_b = se[i].sum() / max(ne[i].sum(), 1) - sn[i].sum() / max(nn[i].sum(), 1)
        eff[b] = 100.0 * v.mean() / range_b if len(v) and abs(range_b) > 1e-9 else np.nan
    good    = eff[~np.isnan(eff)]
    range_all = se.sum() / ne.sum() - sn.sum() / nn.sum()
    vv      = probe[~np.isnan(probe)]
    return dict(effect_pct=100.0 * vv.mean() / range_all,
                lo=float(np.percentile(good, 2.5)), hi=float(np.percentile(good, 97.5)),
                p_equiv=max(float((good >= tol).mean()), float((good <= -tol).mean())),
                n_probe_participants=int(len(vv)), discriminative_range=float(range_all))


def fixed_sequence_select(entries: List[Tuple[Any, float]], alpha: float) -> pd.DataFrame:
    # Fixed-sequence testing of a pre-declared order: each hypothesis at the full alpha, stopping at the
    # first failure; everything after it stays untested.
    rows, stop = [], False
    for label, p in entries:
        tested = not stop
        passed = tested and p < alpha
        rows.append(dict(label=label, tested=tested, passed=passed))
        if tested and not passed:
            stop = True
    return pd.DataFrame(rows)


def prauc_difference(base: PRCAnalysis, other: PRCAnalysis, n_bootstrap: int, seed: int = 0) -> Dict[str, Any]:
    # Paired participant-cluster bootstrap of the macro PR-AUC difference (other - base) over the
    # participants with a defined PR-AUC under both models; two-sided percentile p-value.
    pids = [p for p in base.pids if not (np.isnan(base.prauc_per_participant[p])
                                         or np.isnan(other.prauc_per_participant.get(p, np.nan)))]
    a    = np.array([base.prauc_per_participant[p] for p in pids])
    b    = np.array([other.prauc_per_participant[p] for p in pids])
    idx  = resampling.indices(len(pids), n_bootstrap, seed)
    d    = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    return dict(diff=float(b.mean() - a.mean()), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                p=min(1.0, 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))), n_participants=len(pids))


def by_person(sums: Dict[str, Tuple[float, int, float, int]], person_of: Dict[str, str]
              ) -> Dict[str, Tuple[float, int, float, int]]:
    # Score sums of a person's records (both wrists) added up: the person is the resampling unit.
    out: Dict[str, Tuple[float, int, float, int]] = {}
    for pid, v in sums.items():
        q = person_of.get(pid, pid)
        out[q] = tuple(a + b for a, b in zip(out[q], v)) if q in out else v
    return out


def select(weights: Dict[int, Tuple[List[Dict[str, Any]], pd.DataFrame, pd.DataFrame, PRCAnalysis]],
           cohort_of: Optional[Dict[str, str]] = None, person_of: Optional[Dict[str, str]] = None
           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # weights: {weight: (fold results, arrest probes, arrest-free probes, PR-AUC analysis)}. The arrest-free
    # effect is also reported per cohort; person_of makes a person one resampling unit.
    cohorts = sorted(set(cohort_of.values())) if cohort_of and len(set(cohort_of.values())) > 1 else []
    rows    = []
    for w, (results, arrest, arrest_free, prc) in sorted(weights.items()):
        sums   = by_person(score_sums(results), person_of or {})
        row    = {"weight": w}
        tests  = [("arrest", arrest), ("free", arrest_free)]
        tests += [(f"free_{c}", arrest_free[arrest_free["participant_id"].astype(str).map(cohort_of) == c])
                  for c in cohorts]
        for key, probes in tests:
            unit = probes["participant_id"].astype(str).map(lambda p: (person_of or {}).get(p, p))
            test = equivalence(probes.groupby(unit)["diff"].mean().to_dict(), sums, sorted(sums), TOL_PCT, N_BOOTSTRAP)
            row.update({f"{key}_{k}": v for k, v in test.items()})
        row.update(prauc=float(prc.prauc_mean), prauc_lo=float(prc.prauc_ci[0]), prauc_hi=float(prc.prauc_ci[1]))
        rows.append(row)
        environment.log(f"weight {w}: arrest {row['arrest_effect_pct']:+.1f}% p={row['arrest_p_equiv']:.4f} | "
                        f"arrest-free {row['free_effect_pct']:+.1f}% p={row['free_p_equiv']:.4f} | "
                        f"PR-AUC {row['prauc']:.3f}")

    t = pd.DataFrame(rows)
    t["ius_p_equiv"] = t[["arrest_p_equiv", "free_p_equiv"]].max(axis=1)
    order = t.sort_values("weight", ascending=False)
    base  = float(t.loc[t["weight"] == 0, "arrest_discriminative_range"].iloc[0])
    order = order[order["arrest_discriminative_range"] >= DEGENERATE_BELOW * base]
    fs = fixed_sequence_select(list(zip(order["weight"], order["ius_p_equiv"])), ALPHA)
    t  = t.merge(fs.rename(columns={"label": "weight", "passed": "passes_both"}), on="weight", how="left")
    # Weights left out of the order (collapsed) are untested.
    t["tested"]            = t["tested"].eq(True)
    t["passes_both"]       = t["passes_both"].eq(True)
    t["arrest_equivalent"] = t["tested"] & (t["arrest_p_equiv"] < ALPHA)
    t["free_equivalent"]   = t["tested"] & (t["free_p_equiv"] < ALPHA)
    selected               = t.loc[t["passes_both"], "weight"].min() if t["passes_both"].any() else None
    t["selected"]          = t["weight"] == selected
    environment.log(f"Selected weight: {selected}")

    # The price of mitigation: PR-AUC of the selected weight against the unpenalized model.
    comparison = pd.DataFrame()
    if selected is not None and selected != 0 and 0 in weights:
        test       = prauc_difference(weights[0][3], weights[selected][3], N_BOOTSTRAP)
        comparison = pd.DataFrame([dict(base_weight=0, weight=selected, **test)])
        environment.log(f"PR-AUC, weight {selected} - weight 0: {100 * test['diff']:+.1f} pp "
                        f"[{100 * test['lo']:+.1f}, {100 * test['hi']:+.1f}], p={test['p']:.4f} "
                        f"(n={test['n_participants']})")
    return t, comparison


def plot(t: pd.DataFrame, comparison: pd.DataFrame, width_in: float) -> Any:
    # (a) arrest and (b) arrest-free effects stacked on the left, (c) PR-AUC centred on the right.
    left, right, bottom, top, gap_v, gap_h = 0.75, 0.15, 0.60, 0.30, 0.60, 0.85
    width, height = width_in, FIG_HEIGHT_IN
    panel_h = (height - bottom - top - gap_v) / 2
    panel_w = (width - left - right - gap_h) / 2
    fig = plt.figure(figsize=(width, height), dpi=plotting.DPI)
    def box(x: float, y: float) -> List[float]:
        return [x / width, y / height, panel_w / width, panel_h / height]

    ax_a = fig.add_axes(box(left, bottom + panel_h + gap_v))
    ax_b = fig.add_axes(box(left, bottom))
    ax_c = fig.add_axes(box(left + panel_w + gap_h, bottom + (panel_h + gap_v) / 2))
    x    = np.arange(len(t))
    sel  = t["selected"].values
    star = dict(marker="*", s=55, facecolors="tab:green", edgecolors="black", linewidths=0.5, zorder=6)

    def effect_panel(ax: Any, key: str, ylabel: str) -> None:
        e, lo, hi = t[f"{key}_effect_pct"], t[f"{key}_lo"], t[f"{key}_hi"]
        ax.axhspan(-TOL_PCT, TOL_PCT, color="0.88", zorder=0, label=f"Tolerance (±{TOL_PCT:g}%)")
        ax.axhline(0, color="gray", ls="--", lw=1.0)
        # A near-zero range (collapsed weights) can put the estimate outside its percentile CI.
        ax.errorbar(x, e, yerr=[(e - lo).clip(lower=0), (hi - e).clip(lower=0)], fmt="o", color="tab:blue", ms=3.5,
                    capsize=2.2,
                    elinewidth=0.9, label="Estimate (95% CI)")
        if sel.any():
            ax.scatter(x[sel], e[t["selected"]], label="Selected", **star)
        ax.set_ylabel(ylabel, fontsize=plotting.FONT_AXIS)
        legend = ax.legend(fontsize=plotting.FONT_LEGEND, loc="upper right", framealpha=0.95)
        # Limits tight around the error bars and the tolerance band; the top is raised until no error bar
        # under the legend reaches it.
        y_lo, y_hi = min(lo.min(), e.min(), -TOL_PCT), max(hi.max(), e.max(), TOL_PCT)
        fig.canvas.draw()
        pad   = Y_PAD * (y_hi - y_lo)
        top   = y_hi + pad
        box   = legend.get_window_extent()
        reach = np.maximum(hi, e).values
        while True:
            ax.set_ylim(y_lo - pad, top)
            (x0, y0), (x1, _) = ax.transData.inverted().transform([[box.x0, box.y0], [box.x1, box.y1]])
            under = (x + 0.2 > x0) & (x - 0.2 < x1)
            if not under.any() or reach[under].max() + pad / 2 < y0:
                break
            top += pad / 2

    effect_panel(ax_a, "arrest", "Leakage effect, arrest\n(% of discriminative range)")
    effect_panel(ax_b, "free", "Leakage effect, arrest-free\n(% of discriminative range)")
    pr = t["prauc"] * 100
    ax_c.errorbar(x, pr, yerr=[pr - t["prauc_lo"] * 100, t["prauc_hi"] * 100 - pr], fmt="o", color="tab:blue",
                  ms=3.5, capsize=2.2, elinewidth=0.9, label="PR-AUC (95% CI)")
    if sel.any():
        ax_c.scatter(x[sel], pr[t["selected"]], label="Selected", **star)
    ax_c.set_ylim(-5, 105)
    ax_c.set_yticks([0, 20, 40, 60, 80, 100])
    ax_c.set_ylabel("PR-AUC (%)", fontsize=plotting.FONT_AXIS)
    ax_c.legend(fontsize=plotting.FONT_LEGEND, loc="lower left", framealpha=0.95)
    for _, c in comparison.iterrows():
        x0, x1 = (int(np.flatnonzero(t["weight"].values == w)[0]) for w in (c["base_weight"], c["weight"]))
        y      = 96
        ax_c.plot([x0, x0, x1, x1], [y - 2, y, y, y - 2], color="black", lw=0.7)
        ax_c.text((x0 + x1) / 2, y + 1, f"Δ = {100 * c['diff']:.1f} pp, p = {c['p']:.3f}".replace("-", "−"),
                  ha="center", va="bottom", fontsize=plotting.FONT_LEGEND)
    for i, ax in enumerate((ax_a, ax_b, ax_c)):
        ax.set_xticks(x)
        ax.set_xticklabels(["0" if w == 0 else f"$10^{{{int(round(np.log10(w)))}}}$" for w in t["weight"]],
                           fontsize=plotting.FONT_TICK, rotation=45)
        ax.set_xlim(x.min() - 0.6, x.max() + 0.6)
        ax.set_xlabel("Gradient-penalty weight", fontsize=plotting.FONT_AXIS)
        ax.set_title(plotting.panel_letter(i, "lower"), loc="left", fontsize=plotting.FONT_PANEL, fontweight="bold")
        ax.grid(True, alpha=0.2, lw=0.4)
        ax.tick_params(labelsize=plotting.FONT_TICK)
    return fig
