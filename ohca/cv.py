from __future__ import annotations
import json
import os
import pickle
import shutil
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import tensorflow as tf

from . import dataset, environment, models, scoring
from .dataset import Segment

SEED            = 0
SCALE_JITTER    = 0.05
FINALIST_POLL_S = 60    # seconds between checks for finalists trained by other workers
REPLACE_TRIES   = 10    # Windows may briefly hold a freshly saved model folder
MAX_REDRAWS     = 20    # random redraws of a duplicate search candidate

SEARCH_SPACE: Dict[str, List[Any]] = {
    "width":             ["small", "base", "large"],
    "encoder_dropout":   [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "tversky":           [(0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.3, 0.7)],
    "smoothness_lambda": [0.0, 0.1, 0.2],
}
WIDTH_TO_ARCH: Dict[str, Dict[str, Any]] = {
    "small": {"context_units": 64,  "encoder_filters": [8, 16, 32]},
    "base":  {"context_units": 128, "encoder_filters": [16, 32, 64]},
    "large": {"context_units": 256, "encoder_filters": [32, 64, 128]},
}


class SearchSpec(NamedTuple):
    screen_epochs:   int                  # epochs of the first screening rung (x rung_eta per rung)
    n_screen:        int           = 27   # Latin-hypercube candidates in the first rung
    n_finalists:     int           = 3    # trained to convergence; the best on validation PR-AUC wins
    rung_eta:        int           = 3    # each rung keeps 1/rung_eta of the candidates
    max_epochs:      int           = 1000
    patience:        Optional[int] = 20   # early stopping on the validation loss
    val_size:        int           = 8    # validation fold groups per fold
    steps_per_epoch: int           = 100  # training batches per epoch, and validation batches



def replace_dir(src: Path, dst: Path) -> None:
    for k in range(REPLACE_TRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if k == REPLACE_TRIES - 1:
                raise
            time.sleep(2.0 * (k + 1))


def armed_from(segments: List[Segment], arming: int, warmup: int) -> np.ndarray:
    # Per segment: the length of its unarmed head, until the participant has `arming` rows of history (its
    # segments in index order, as the normaliser runs) and the segment `warmup` rows of its own.
    head  = np.zeros(len(segments), dtype=np.int64)
    order = sorted(range(len(segments)), key=lambda k: (segments[k][0], segments[k][1]))
    pid, offset = None, 0
    for k in order:
        if segments[k][0] != pid:
            pid, offset = segments[k][0], 0
        head[k] = min(len(segments[k][2]), max(arming - offset, warmup, 0))
        offset += len(segments[k][2])
    return head


def arm(segments: List[Segment], arming_s: float, warmup_s: float) -> List[Segment]:
    # Scored segments without their unarmed head (not monitored time); a segment whose event begins in
    # the head is dropped.
    head = armed_from(segments, int(round(arming_s)), int(round(warmup_s)))
    out  = []
    for s, k in zip(segments, head):
        k   = int(k)
        pos = np.flatnonzero(np.asarray(s[3]) > 0)
        if k >= len(s[2]) or (k and len(pos) and pos[0] < k):
            continue
        out.append(s if not k else (s[0], s[1], s[2][k:], s[3][k:]))
    return out


class NestedCrossValidationManager:

    # Grouped CV: fold i holds out group i. The next val_size groups (cyclically) validate a
    # successive-halving search over SEARCH_SPACE on PR-AUC. With config_from (another weight's cv/), fold i
    # instead trains the configuration fold i's search there selected, once that fold has its model.

    def __init__(self, feature_dir: Path, groups: List[List[str]], cv_dir: Path, search: SearchSpec,
                 gap_penalty: float = 0.0, cohort_of: Optional[Dict[str, str]] = None,
                 negative_sampling: str = "record", arming_s: float = 0.0, segment_warmup_s: float = 0.0,
                 crop_step: int = 1, config_from: Optional[Path] = None) -> None:

        self.feature_dir  = Path(feature_dir)
        self.groups       = [list(g) for g in groups]
        self.ids          = [p for g in self.groups for p in g]
        self.cv_dir       = Path(cv_dir)
        self.search       = search
        self.gap_penalty  = gap_penalty
        self.cohort_of    = cohort_of or {}
        self.neg_sampling = negative_sampling
        self.arming_s     = arming_s
        self.warmup_s     = segment_warmup_s
        self.crop_step    = crop_step
        self.config_from  = None if config_from is None else Path(config_from)
        self.n_runs       = len(self.groups)
        self.results: List[Dict[str, Any]] = []

    @property
    def armed(self) -> bool:
        return self.arming_s > 0 or self.warmup_s > 0

    def fold_dir(self, i: int) -> Path:
        return self.cv_dir / f"fold_{i:03d}"

    def load_results(self, folds: Optional[List[int]] = None) -> None:
        self.results = []
        for i in (range(self.n_runs) if folds is None else folds):
            path = self.fold_dir(i) / "result.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    self.results.append({"fold": i, **pickle.load(f)})

    def load(self, pids: List[str]) -> List[Segment]:
        data: List[Segment] = []
        for pid in pids:
            with open(self.feature_dir / f"{pid}.pkl", "rb") as f:
                data.extend(pickle.load(f))
        return data

    def _split(self, i: int) -> Tuple[List[str], List[str], List[str]]:
        val  = [p for j in range(self.search.val_size) for p in self.groups[(i + 1 + j) % self.n_runs]]
        test = list(self.groups[i])
        held = set(test) | set(val)
        return [p for p in self.ids if p not in held], val, test

    def _sampler(self, data: List[Segment], seed: List[int], scale_jitter: float = 0.0) -> models.TrainingSampler:
        # Crops start once the detector is armed, so training sees what evaluation scores.
        armed = None if not self.armed else armed_from(data, int(round(self.arming_s * dataset.FS)),
                                                       int(round(self.warmup_s * dataset.FS)))
        return models.TrainingSampler(data, scale_jitter=scale_jitter, neg_weighting=self.neg_sampling,
                                      armed_from=armed, crop_step=self.crop_step, seed=seed)

    @staticmethod
    def _config(width: str, dropout: float, tversky: Tuple[float, float], smoothness: float) -> Dict[str, Any]:
        return {"width": width, "encoder_dropout": float(dropout), "tversky_alpha": float(tversky[0]),
                "tversky_beta": float(tversky[1]), "smoothness_lambda": float(smoothness)}

    @staticmethod
    def _sample_config(rng: np.random.Generator) -> Dict[str, Any]:
        space   = SEARCH_SPACE
        width   = space["width"][rng.integers(len(space["width"]))]
        tversky = space["tversky"][rng.integers(len(space["tversky"]))]
        dropout = space["encoder_dropout"][rng.integers(len(space["encoder_dropout"]))]
        return NestedCrossValidationManager._config(
            width, dropout, tversky, space["smoothness_lambda"][rng.integers(len(space["smoothness_lambda"]))])

    @staticmethod
    def _lhs_candidates(n: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
        # Discrete Latin hypercube: every level of every factor appears floor(n/k) or ceil(n/k) times,
        # each factor permuted independently. Duplicates are redrawn at random.
        space   = SEARCH_SPACE
        columns = {}
        for name in ["width", "encoder_dropout", "tversky", "smoothness_lambda"]:
            k   = len(space[name])
            idx = np.tile(np.arange(k), int(np.ceil(n / k)))[:n]
            rng.shuffle(idx)
            columns[name] = idx

        configs, seen = [], set()
        for i in range(n):
            cand = NestedCrossValidationManager._config(
                space["width"][columns["width"][i]], space["encoder_dropout"][columns["encoder_dropout"][i]],
                space["tversky"][columns["tversky"][i]], space["smoothness_lambda"][columns["smoothness_lambda"][i]])
            key, attempts = tuple(sorted(cand.items())), 0
            while key in seen and attempts < MAX_REDRAWS:
                cand      = NestedCrossValidationManager._sample_config(rng)
                key       = tuple(sorted(cand.items()))
                attempts += 1
            seen.add(key)
            configs.append(cand)
        return configs

    def rung_schedule(self) -> List[Tuple[int, int]]:
        # (n configs, epochs) per screening rung, keeping 1/eta per rung; the finalists come after.
        s        = self.search
        n, r     = s.n_screen, s.screen_epochs
        schedule = [(n, r)]
        while n > s.n_finalists * s.rung_eta:
            n = max(s.n_finalists, int(round(n / s.rung_eta)))
            r = r * s.rung_eta
            schedule.append((n, r))
        return schedule

    def _train_candidate(self, cand: Dict[str, Any], sampler_train: models.TrainingSampler,
                         sampler_val: models.TrainingSampler, max_epochs: int, patience: Optional[int]
                         ) -> models.RNNModel:
        arch = {**models.RNNModel.DEFAULT_ARCH_CONFIG, **WIDTH_TO_ARCH[cand["width"]]}
        arch["encoder_dropout"] = cand["encoder_dropout"]
        return models.RNNModel.train(arch, cand["tversky_alpha"], cand["tversky_beta"], cand["smoothness_lambda"],
                                     sampler_train, sampler_val, max_epochs, patience, self.gap_penalty,
                                     self.search.steps_per_epoch)

    def _val_prauc(self, model: models.RNNModel, val_data: List[Segment]) -> float:
        applied = model.apply(val_data)
        if self.armed:
            applied = arm(applied, self.arming_s, self.warmup_s)
        return float(scoring.pooled_prc(np.concatenate([s[2] for s in applied]),
                                        np.concatenate([s[3] for s in applied]).astype(float), scoring.PROB_GRID)[1])

    @staticmethod
    def _write_json(path: Path, obj: Any) -> None:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(obj, indent=1))
        os.replace(tmp, path)

    @staticmethod
    def _release() -> None:
        tf.keras.backend.clear_session()
        environment.free_memory()

    @staticmethod
    def search_winner(search: Dict[str, Any]) -> int:
        return max(search["finalists"], key=lambda f: f["val_prauc"])["cand"]

    def reference_config(self, i: int) -> Optional[Dict[str, Any]]:
        # None until that fold has its model (its search.json is final before model/ appears).
        ref = self.config_from / f"fold_{i:03d}"
        if not (ref / "model").exists():
            return None
        search = json.loads((ref / "search.json").read_text())
        best   = self.search_winner(search)
        return next(e["config"] for e in search["pool"] if e["cand"] == best)

    def train(self, folds: List[int]) -> None:

        # Per fold: search.json (resume state), then model/ and result.pkl. A worker claims a fold with an
        # exclusive claim file ("host:pid"), so workers can share a fold list; dead claims on this host are
        # taken over. With config_from, a fold whose reference is not ready yet is retried later.
        pending, waiting = list(folds), set()
        while pending:
            deferred = []
            for i in pending:
                fold_dir = self.fold_dir(i)
                if (fold_dir / "result.pkl").exists():
                    continue
                if self.config_from is not None and self.reference_config(i) is None:
                    if i not in waiting:
                        environment.log(f"=== Fold {i + 1}/{self.n_runs}: waiting for "
                                        f"{self.config_from.parent.name} ===")
                        waiting.add(i)
                    deferred.append(i)
                    continue
                fold_dir.mkdir(parents=True, exist_ok=True)
                if not self._claim(fold_dir):
                    environment.log(f"=== Fold {i + 1}/{self.n_runs}: claimed by {self._owner(fold_dir)}, skipped ===")
                    continue
                try:
                    self._train_fold(i, fold_dir)
                finally:
                    (fold_dir / "claim").unlink()
            pending = deferred
            if pending:
                time.sleep(FINALIST_POLL_S)

    @staticmethod
    def _owner(fold_dir: Path, name: str = "claim") -> str:
        try:
            return (fold_dir / name).read_text()
        except FileNotFoundError:
            return ""

    @staticmethod
    def _dead_here(owner: str) -> bool:
        # Whether a "host:pid" owner is a process on this host that no longer runs.
        host, _, pid = owner.partition(":")
        return host == socket.gethostname() and pid.isdigit() and not environment.process_alive(int(pid))

    @staticmethod
    def _claim(fold_dir: Path, name: str = "claim") -> bool:
        claim, me = fold_dir / name, f"{socket.gethostname()}:{os.getpid()}"
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, me.encode())
            os.close(fd)
            return True
        except FileExistsError:
            pass
        # A dead claim is taken over under a lock and replaced in place, so it never disappears.
        lock = fold_dir / f"{name}.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, me.encode())
            os.close(fd)
        except FileExistsError:
            # A lock of a dead process on this host is removed; the next attempt can take the claim.
            if NestedCrossValidationManager._dead_here(NestedCrossValidationManager._owner(fold_dir, lock.name)):
                lock.unlink(missing_ok=True)
            return False
        try:
            if not NestedCrossValidationManager._dead_here(NestedCrossValidationManager._owner(fold_dir, name)):
                return False
            tmp = fold_dir / f"{name}.{os.getpid()}"
            tmp.write_text(me)
            os.replace(tmp, claim)
            return True
        finally:
            lock.unlink()

    def _fold_data(self, i: int) -> Tuple[List[Segment], List[Segment], models.TrainingSampler, models.TrainingSampler]:
        train_ids, val_ids, _ = self._split(i)
        train_data, val_data  = self.load(train_ids), self.load(val_ids)
        return (train_data, val_data, self._sampler(train_data, [SEED, i, 0], SCALE_JITTER),
                self._sampler(val_data, [SEED, i, 1]))

    def _finalists(self, i: int, fold_dir: Path, survivors: List[int], config_of: Dict[int, Dict[str, Any]],
                   sampler_train: models.TrainingSampler, sampler_val: models.TrainingSampler,
                   val_data: List[Segment], recorded: List[Dict[str, Any]], wait: bool) -> List[Dict[str, Any]]:
        # Any process that claims a finalist trains it: finalist_CC/, then finalist_CC.json (validation
        # PR-AUC). With wait, returns once every finalist has a record.
        known = {f["cand"]: f for f in recorded}
        while True:
            for c in survivors:
                rec = fold_dir / f"finalist_{c:02d}.json"
                if c in known:
                    continue
                if rec.exists():
                    known[c] = json.loads(rec.read_text())
                    continue
                if not self._claim(fold_dir, f"finalist_{c:02d}.claim"):
                    continue
                try:
                    environment.log(f"  [fold {i + 1}] finalist cand {c}: full training")
                    model = self._train_candidate(config_of[c], sampler_train, sampler_val,
                                                  self.search.max_epochs, self.search.patience)
                    out, tmp = fold_dir / f"finalist_{c:02d}", fold_dir / f"finalist_{c:02d}.tmp"
                    for d in (out, tmp):
                        if d.exists():
                            shutil.rmtree(d)
                    model.save(tmp)
                    replace_dir(tmp, out)
                    known[c] = {"cand": c, "val_prauc": self._val_prauc(model, val_data)}
                    self._write_json(rec, known[c])
                    del model
                    self._release()
                finally:
                    (fold_dir / f"finalist_{c:02d}.claim").unlink(missing_ok=True)
            if not wait or all(c in known for c in survivors):
                return [known[c] for c in survivors if c in known]
            time.sleep(FINALIST_POLL_S)

    def train_finalists(self, folds: List[int]) -> None:
        # Trains unclaimed finalists of folds whose rungs are done, until every listed fold is scored.
        if self.config_from is not None:
            return
        schedule = self.rung_schedule()
        while not all((self.fold_dir(i) / "result.pkl").exists() for i in folds):
            for i in folds:
                fold_dir, search_path = self.fold_dir(i), self.fold_dir(i) / "search.json"
                if (fold_dir / "model").exists() or not search_path.exists():
                    continue
                search = json.loads(search_path.read_text())
                if len(search["rungs"][-1]) < schedule[-1][0]:
                    continue
                last      = sorted(search["rungs"][-1], key=lambda s: s["prauc"], reverse=True)
                survivors = [s["cand"] for s in last[:self.search.n_finalists]]
                done      = {f["cand"] for f in search["finalists"]}
                pending   = [c for c in survivors if c not in done
                             and not (fold_dir / f"finalist_{c:02d}.json").exists()
                             and not (fold_dir / f"finalist_{c:02d}.claim").exists()]
                if not pending:
                    continue
                environment.log(f"=== Fold {i + 1}/{self.n_runs}: helping with finalists {pending} ===")
                train_data, val_data, sampler_train, sampler_val = self._fold_data(i)
                config_of = {e["cand"]: e["config"] for e in search["pool"]}
                self._finalists(i, fold_dir, survivors, config_of, sampler_train, sampler_val, val_data,
                                search["finalists"], wait=False)
                del train_data, val_data, sampler_train, sampler_val
                self._release()
            time.sleep(FINALIST_POLL_S)

    def _train_fold(self, i: int, fold_dir: Path) -> None:
        train_ids, val_ids, test_ids = self._split(i)
        environment.log(f"=== Fold {i + 1}/{self.n_runs}  test={','.join(test_ids)} "
                        f"(train={len(train_ids)}, val={len(val_ids)}) ===")

        train_data, val_data, sampler_train, sampler_val = self._fold_data(i)

        search_path = fold_dir / "search.json"
        schedule    = [] if self.config_from is not None else self.rung_schedule()
        if search_path.exists():
            search = json.loads(search_path.read_text())
        elif self.config_from is not None:
            search = {"pool": [{"cand": 0, "config": self.reference_config(i)}], "rungs": [], "finalists": [],
                      "config_from": str(self.config_from / f"fold_{i:03d}")}
            self._write_json(search_path, search)
        else:
            cands  = self._lhs_candidates(schedule[0][0], np.random.default_rng(SEED + i))
            search = {"pool": [{"cand": c, "config": cand} for c, cand in enumerate(cands)],
                      "rungs": [[] for _ in schedule], "finalists": []}
            self._write_json(search_path, search)
        config_of = {e["cand"]: e["config"] for e in search["pool"]}

        survivors = [e["cand"] for e in search["pool"]]
        for r, (_, epochs) in enumerate(schedule):
            scored = search["rungs"][r]
            for c in [c for c in survivors if c not in {s["cand"] for s in scored}]:
                environment.log(f"  [fold {i + 1}] rung {r} ({epochs} ep) cand {c}: {config_of[c]}")
                model = self._train_candidate(config_of[c], sampler_train, sampler_val, epochs, None)
                scored.append({"cand": c, "prauc": self._val_prauc(model, val_data)})
                self._write_json(search_path, search)
                del model
                self._release()
            next_n    = schedule[r + 1][0] if r + 1 < len(schedule) else self.search.n_finalists
            survivors = [s["cand"] for s in sorted(scored, key=lambda s: s["prauc"], reverse=True)[:next_n]]

        model_dir = fold_dir / "model"
        if not model_dir.exists():
            search["finalists"] = self._finalists(i, fold_dir, survivors, config_of, sampler_train, sampler_val,
                                                  val_data, search["finalists"], wait=True)
            self._write_json(search_path, search)
            best = self.search_winner(search)
            replace_dir(fold_dir / f"finalist_{best:02d}", model_dir)
            for d in fold_dir.glob("finalist_*"):
                shutil.rmtree(d) if d.is_dir() else d.unlink()

        environment.log(f"  [fold {i + 1}] scoring test={','.join(test_ids)}")
        model  = models.RNNModel.load(model_dir)
        result = {"test": test_ids, "test_scores": model.apply(self.load(test_ids)),
                  "val_scores": model.apply(val_data)}
        tmp    = fold_dir / "result.pkl.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(result, f)
        os.replace(tmp, fold_dir / "result.pkl")
        del model, result

        del train_data, val_data, sampler_train, sampler_val
        self._release()
