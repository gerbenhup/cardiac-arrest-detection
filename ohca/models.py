from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

from . import dataset, labels
from .dataset import Segment

# Causal CNN encoder per 1 s window -> LSTM -> per-second logit, trained with a Tversky loss.

LEARNING_RATE     = 1e-3
MIN_NS            = 30      # training crop length range, seconds
MAX_NS            = 60
BATCH_SIZE        = 32
APPLY_BATCH_STEPS = 8192    # inference batch size in seconds (segments x padded length)
INNER_LOSS_SCALE  = 1024.0  # fixed loss scale of the gap penalty's input gradient under mixed_float16


class TrainingSampler:

    # Crops of ns seconds, pos_frac of them ending inside an event. A negative crop's record is uniform
    # ("record") or drawn by its number of negative crop ends ("time"). No crop starts before armed_from.

    NEG_WEIGHTINGS = ("record", "time")

    def __init__(self, data: List[Segment], pos_frac: float = 0.5,
                 scale_jitter: float = 0.0, neg_weighting: str = "record",
                 armed_from: Optional[np.ndarray] = None, crop_step: int = 1,
                 seed: Union[int, Sequence[int]] = 0) -> None:

        if neg_weighting not in self.NEG_WEIGHTINGS:
            raise ValueError(f"neg_weighting must be one of {self.NEG_WEIGHTINGS}")
        self.pos_frac      = pos_frac
        self.scale_jitter  = scale_jitter
        self.signals       = [sig for _, _, sig, _ in data]
        self.annotations   = [ann for _, _, _, ann in data]
        self.neg_weighting = neg_weighting
        self.rng           = np.random.default_rng(seed)
        self.crop_step     = crop_step

        neg_ends, pos_records = [], []
        for i, ann in enumerate(self.annotations):
            if labels.has_event(ann):
                pos_records.append(i)
                neg_ends.append(np.argwhere(ann == 1).ravel()[0].item())
            else:
                neg_ends.append(len(self.signals[i]))
        self.neg_ends    = np.array(neg_ends, dtype=np.int64)
        self.pos_ends    = np.array([len(s) for s in self.signals], dtype=np.int64)
        self.pos_records = np.array(pos_records, dtype=np.int64)
        self.armed_from  = (np.zeros(len(self.signals), dtype=np.int64) if armed_from is None
                            else np.asarray(armed_from, dtype=np.int64))

        self.C     = self.signals[0].shape[-1]
        self.dtype = self.signals[0].dtype

    def _negatives(self, eligible: np.ndarray, n_neg: int, length: int) -> np.ndarray:
        if n_neg <= 0:
            return np.array([], dtype=int)
        if self.neg_weighting == "time":
            pool = np.flatnonzero(eligible)
            ends = (self.neg_ends[pool] - self.armed_from[pool] - length + 1).astype(np.float64)
            return self.rng.choice(pool, size=n_neg, replace=True, p=ends / ends.sum())
        return self.rng.choice(np.flatnonzero(eligible), size=n_neg, replace=True)

    def _positives(self, valid_pos: np.ndarray, n_pos: int) -> np.ndarray:
        if n_pos <= 0:
            return np.array([], dtype=int)
        return self.rng.choice(valid_pos, size=n_pos, replace=True)

    def _raw_batch(self, batch_size: int, L: int, ns: int) -> Tuple[np.ndarray, np.ndarray]:

        length = ns * L
        n_pos  = int(np.clip(round(batch_size * self.pos_frac), 0, batch_size))
        n_neg  = batch_size - n_pos

        eligible  = self.neg_ends + 1 > self.armed_from + length
        valid_pos = self.pos_records[eligible[self.pos_records]]

        records_pos = self._positives(valid_pos, n_pos)
        records_neg = self._negatives(eligible, n_neg, length)

        if n_pos > 0:
            pos_neg_end      = self.neg_ends[records_pos]
            pos_end          = self.pos_ends[records_pos]
            first            = np.maximum(pos_neg_end + 1, self.armed_from[records_pos] + length)
            records_pos_ends = self.rng.integers(np.minimum(first, pos_end), pos_end + 1).astype(np.int64)
        else:
            records_pos_ends = np.array([], dtype=np.int64)
        if n_neg > 0:
            neg_end          = self.neg_ends[records_neg]
            records_neg_ends = self.rng.integers(self.armed_from[records_neg] + length, neg_end + 1).astype(np.int64)
        else:
            records_neg_ends = np.array([], dtype=np.int64)

        records = np.concatenate([records_pos, records_neg]).astype(np.int64)
        ends    = np.concatenate([records_pos_ends, records_neg_ends]).astype(np.int64)
        starts  = ends - length

        signals = np.stack([self.signals[i][a:b] for i, a, b in zip(records, starts, ends)], axis=0)
        targets = np.stack([self.annotations[i][a:b] for i, a, b in zip(records, starts, ends)], axis=0)
        signals = np.reshape(signals, (batch_size, ns, L, -1))
        if self.scale_jitter > 0.0:
            scale   = 1.0 + self.rng.uniform(-self.scale_jitter, self.scale_jitter,
                                              size=(signals.shape[0], 1, 1, 1)).astype(signals.dtype)
            signals = (signals * scale).astype(signals.dtype)
        return signals, np.reshape(targets, (batch_size, ns, L))

    def batch(self, batch_size: int, L: int, ns: int, with_raw: bool = False) -> Tuple[np.ndarray, ...]:
        signals, targets = self._raw_batch(batch_size, L, ns)
        raw     = targets.astype(np.float32)
        # A gapped sample is negative supervision: the model should stay quiet on a shock.
        targets = np.mean(np.maximum(targets, 0.0), axis=-1).astype(np.float32)
        return (signals, targets, raw) if with_raw else (signals, targets)

    def __call__(self, L: int, min_ns: int, max_ns: int, batch_size: int, with_raw: bool = False) -> tf.data.Dataset:

        dtype = tf.as_dtype(self.dtype)

        def py_batch(_: tf.Tensor) -> Tuple[np.ndarray, ...]:
            ns = min_ns + self.crop_step * int(self.rng.integers(0, (max_ns - min_ns) // self.crop_step + 1))
            return self.batch(batch_size, L, ns, with_raw=with_raw)

        def tf_batch(idx: tf.Tensor) -> Tuple[tf.Tensor, ...]:
            outs = tf.py_function(py_batch, [idx], [dtype, tf.float32] + ([tf.float32] if with_raw else []))
            outs[0].set_shape((batch_size, None, L, self.C))
            outs[1].set_shape((batch_size, None))
            if with_raw:
                outs[2].set_shape((batch_size, None, L))
            return tuple(outs)

        ds = tf.data.Dataset.range(2 ** 31 - 1).map(tf_batch, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)


class WeightedTverskyLoss:

    # 1 - Tversky index on soft counts: alpha weights false positives, beta false negatives; optionally
    # plus a penalty on step-to-step logit changes in event-free time.

    def __init__(self, alpha: float, beta: float, smoothness_lambda: float, smooth: float = 1.0) -> None:
        self.alpha             = alpha
        self.beta              = beta
        self.smoothness_lambda = smoothness_lambda
        self.smooth            = smooth

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # float32 regardless of the mixed_float16 policy: batch-wide sums overflow in float16.
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = tf.cast(y_pred, tf.float32)
        p       = tf.sigmoid(y_pred)
        TP      = tf.reduce_sum(y_true * p)
        FP      = tf.reduce_sum((1.0 - y_true) * p)
        FN      = tf.reduce_sum(y_true * (1.0 - p))
        loss    = 1.0 - (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        if self.smoothness_lambda > 0.0:
            dz       = y_pred[:, 1:] - y_pred[:, :-1]
            neg_pair = tf.cast((y_true[:, 1:] <= 0.0) & (y_true[:, :-1] <= 0.0), tf.float32)
            smooth_t = tf.reduce_sum(neg_pair * tf.square(dz)) / (tf.reduce_sum(neg_pair) + 1e-6)
            loss     = loss + self.smoothness_lambda * smooth_t
        return loss


class SequenceModel(tf.keras.Model):

    def __init__(self, L: int, C: int, config: Dict[str, Any], *args: Any, **kwargs: Any) -> None:

        super().__init__(*args, **kwargs)
        self.L      = L
        self.C      = C
        self.config = config
        reg         = tf.keras.regularizers.l2(config["l2_reg"]) if config["l2_reg"] > 0 else None
        self.encoder = self._make_encoder(config, reg)
        self.lstm    = tf.keras.layers.LSTM(config["context_units"], return_sequences=True,
                                            return_state=True, dropout=config["context_dropout"],
                                            kernel_regularizer=reg)
        self.decoder = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))

    @staticmethod
    def _make_encoder(config: Dict[str, Any], reg: Any) -> tf.keras.layers.TimeDistributed:
        filters, dropout = config["encoder_filters"], config["encoder_dropout"]
        layers = [
            tf.keras.layers.Conv1D(filters[0], 3, padding="same", activation="relu", kernel_regularizer=reg),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.MaxPool1D(),
            tf.keras.layers.Conv1D(filters[1], 3, padding="same", activation="relu", kernel_regularizer=reg),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.MaxPool1D(),
            tf.keras.layers.Conv1D(filters[2], 3, padding="same", activation="relu", kernel_regularizer=reg),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling1D(),
        ]
        return tf.keras.layers.TimeDistributed(tf.keras.Sequential(layers))

    def build(self, input_shape: Optional[tuple] = None) -> None:
        self.encoder.build((None, None, self.L, self.C))
        self.lstm.build((None, None, self.config["encoder_filters"][-1]))
        self.decoder.build((None, None, self.config["context_units"]))
        self.built = True

    def call(self, inputs: Any, training: Optional[bool] = None) -> tf.Tensor:
        x = self.encoder(inputs, training=training)
        lstm_out, _, _ = self.lstm(x, training=training)
        return tf.cast(self.decoder(lstm_out, training=training)[:, :, 0], tf.float32)

    @classmethod
    def built_model(cls, L: int, C: int, config: Dict[str, Any]) -> SequenceModel:
        model = cls(L, C, config)
        model.build()
        model(np.zeros((1, 1, L, C), dtype=np.float32), training=False)
        return model


class PenalizedSequenceModel(SequenceModel):

    # Generic LSTM cell instead of the cuDNN kernel, which has no second derivative. The weights have
    # the same layout, so a trained model is copied into a SequenceModel.

    def __init__(self, L: int, C: int, config: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(L, C, config, *args, **kwargs)
        reg = tf.keras.regularizers.l2(config["l2_reg"]) if config["l2_reg"] > 0 else None
        self.lstm = tf.keras.layers.RNN(
            tf.keras.layers.LSTMCell(config["context_units"], dropout=config["context_dropout"],
                                     kernel_regularizer=reg),
            return_sequences=True, return_state=True)

    def call(self, inputs: Any, training: Optional[bool] = None) -> tf.Tensor:
        # float32 input, so the input gradient is taken in full precision.
        return super().call(tf.cast(inputs, tf.float32), training=training)


def fit(model: tf.keras.Model, loss_fn: Any, sampler_train: TrainingSampler, sampler_val: TrainingSampler,
        max_epochs: int, patience: Optional[int], steps_per_epoch: int) -> None:

    # Without early stopping the validation loss is never used, so it is not computed.
    L = RNNModel.L
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=loss_fn)
    callbacks = [tf.keras.callbacks.TerminateOnNaN()]
    if patience is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True))
    model.fit(x                = sampler_train(L, MIN_NS, MAX_NS, BATCH_SIZE),
              steps_per_epoch  = steps_per_epoch,
              validation_data  = None if patience is None else sampler_val(L, MIN_NS, MAX_NS, BATCH_SIZE),
              validation_steps = None if patience is None else steps_per_epoch,
              epochs           = max_epochs,
              verbose          = 2,
              callbacks        = callbacks)


def fit_penalized(model: PenalizedSequenceModel, loss_fn: Any, sampler_train: TrainingSampler,
                  sampler_val: TrainingSampler, gap_penalty: float, max_epochs: int, patience: Optional[int],
                  steps_per_epoch: int) -> None:

    # fit() plus the gap-input-gradient penalty (Ross et al. 2017): gap_penalty * mean_b sum(gap_mask *
    # (dS_b/dx)^2), with S_b = sum_t w_t z_t and random +-1 weights w_t, which a plain sum could cancel.
    # The inner gradient has its own fixed loss scale: the optimizer's never reaches its tiny slopes.
    L           = RNNModel.L
    fp16        = tf.keras.mixed_precision.global_policy().name == "mixed_float16"
    inner_scale = INNER_LOSS_SCALE if fp16 else 1.0
    optimizer   = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    if fp16:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    C        = sampler_train.C
    in_dtype = tf.as_dtype(sampler_train.dtype)
    var      = model.trainable_variables

    @tf.function(input_signature=[tf.TensorSpec((BATCH_SIZE, None, L, C), in_dtype),
                                  tf.TensorSpec((BATCH_SIZE, None), tf.float32),
                                  tf.TensorSpec((BATCH_SIZE, None, L), tf.float32)])
    def train_step(x: tf.Tensor, y: tf.Tensor, raw: tf.Tensor) -> tf.Tensor:
        x = tf.cast(x, tf.float32)
        with tf.GradientTape() as outer:
            z, penalty = _penalty(x, raw)
            base   = loss_fn(y, z)
            loss   = base + gap_penalty * penalty
            scaled = optimizer.get_scaled_loss(loss) if fp16 else loss
        grads = outer.gradient(scaled, var)
        if fp16:
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, var))
        return base

    def _penalty(x: tf.Tensor, raw: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as inner:
            inner.watch(x)
            z = model(x, training=True)
            w = tf.where(tf.random.uniform(tf.shape(z)) < 0.5, -1.0, 1.0)
            s = tf.reduce_sum(w * z) * inner_scale
        g    = inner.gradient(s, x) / inner_scale
        mask = tf.cast(raw < 0.0, tf.float32)[..., None]
        return z, tf.reduce_sum(mask * tf.square(g)) / float(BATCH_SIZE)

    @tf.function(input_signature=[tf.TensorSpec((BATCH_SIZE, None, L, C), in_dtype),
                                  tf.TensorSpec((BATCH_SIZE, None), tf.float32)])
    def val_step(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        return loss_fn(y, model(x, training=False))

    # Losses stay on the device until the end of an epoch, so the steps are not synchronised one by one.
    it_train = iter(sampler_train(L, MIN_NS, MAX_NS, BATCH_SIZE, with_raw=True))
    it_val   = None if patience is None else iter(sampler_val(L, MIN_NS, MAX_NS, BATCH_SIZE))

    best, best_weights, wait = np.inf, None, 0
    for _ in range(max_epochs):
        train_loss = float(tf.reduce_mean(tf.stack([train_step(*next(it_train)) for _ in range(steps_per_epoch)])))
        if not np.isfinite(train_loss):
            break
        if it_val is None:
            continue
        val_loss = float(tf.reduce_mean(tf.stack([val_step(*next(it_val)) for _ in range(steps_per_epoch)])))
        if val_loss < best:
            best, best_weights, wait = val_loss, model.get_weights(), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_weights is not None:
        model.set_weights(best_weights)


class RNNModel:

    L = dataset.STEP
    DEFAULT_ARCH_CONFIG = {
        "encoder_filters": [16, 32, 64],
        "encoder_dropout": 0.5,
        "context_units":   128,
        "context_dropout": 0.0,
        "l2_reg":          0.0,
    }

    def __init__(self, network: SequenceModel) -> None:
        self.network = network

    @staticmethod
    def train(arch_config: Dict[str, Any], tversky_alpha: float, tversky_beta: float, smoothness_lambda: float,
              sampler_train: TrainingSampler, sampler_val: TrainingSampler, max_epochs: int,
              patience: Optional[int], gap_penalty: float, steps_per_epoch: int) -> RNNModel:

        # patience=None trains a fixed epoch budget (screening rungs); otherwise early stopping.
        L, C    = RNNModel.L, sampler_train.C
        config  = {**RNNModel.DEFAULT_ARCH_CONFIG, **arch_config}
        loss_fn = WeightedTverskyLoss(tversky_alpha, tversky_beta, smoothness_lambda)
        if gap_penalty > 0.0:
            trainer = PenalizedSequenceModel.built_model(L, C, config)
            fit_penalized(trainer, loss_fn, sampler_train, sampler_val, gap_penalty, max_epochs, patience,
                          steps_per_epoch)
            model = SequenceModel.built_model(L, C, config)
            for name in ("encoder", "lstm", "decoder"):
                getattr(model, name).set_weights(getattr(trainer, name).get_weights())
        else:
            model = SequenceModel(L, C, config)
            fit(model, loss_fn, sampler_train, sampler_val, max_epochs, patience, steps_per_epoch)
        return RNNModel(model)

    @staticmethod
    def step_labels(annotations: np.ndarray, n_steps: int) -> np.ndarray:
        # Per second: positive if any sample is, else gapped if any sample is, else monitored.
        ann_steps = np.asarray(annotations[:n_steps * RNNModel.L]).reshape(n_steps, RNNModel.L)
        return np.where((ann_steps == 1).any(axis=1), 1.0,
                        np.where((ann_steps == -1).any(axis=1), -1.0, 0.0)).astype(np.float32)

    def apply_segment(self, signals: np.ndarray, annotations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        L      = RNNModel.L
        N_crop = (len(signals) // L) * L
        if N_crop == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
        z = self.network(np.reshape(signals[:N_crop], (1, -1, L, signals.shape[-1])), training=False).numpy().ravel()
        return z, self.step_labels(annotations, N_crop // L)

    def apply(self, data: List[Segment]) -> List[Segment]:

        # Segments of similar length share a batch, right-padded: the encoder works per second and the
        # LSTM is causal, so the padding never reaches the scores that are kept.
        L       = RNNModel.L
        n_steps = [len(sig) // L for _, _, sig, _ in data]
        scores: Dict[int, np.ndarray] = {}
        batch: List[int] = []

        def flush() -> None:
            width = n_steps[batch[-1]]
            sig0  = data[batch[0]][2]
            x     = np.zeros((len(batch), width, L, sig0.shape[-1]), dtype=sig0.dtype)
            for b, k in enumerate(batch):
                x[b, :n_steps[k]] = np.reshape(data[k][2][:n_steps[k] * L], (n_steps[k], L, -1))
            z = self.network(x, training=False).numpy()
            for b, k in enumerate(batch):
                scores[k] = z[b, :n_steps[k]]

        for k in sorted((k for k in range(len(data)) if n_steps[k] > 0), key=lambda k: n_steps[k]):
            if batch and (len(batch) + 1) * n_steps[k] > APPLY_BATCH_STEPS:
                flush()
                batch = []
            batch.append(k)
        if batch:
            flush()
        return [(pid, si, scores[k], self.step_labels(ann, n_steps[k]))
                for k, (pid, si, _, ann) in enumerate(data) if k in scores]

    def save(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.network.save_weights(str(model_dir / "weights"), save_format="tf")
        (model_dir / "model.json").write_text(json.dumps(
            {"L": self.network.L, "C": self.network.C, "arch_config": self.network.config}, indent=1))

    @staticmethod
    def load(model_dir: Path) -> RNNModel:
        meta    = json.loads((model_dir / "model.json").read_text())
        network = SequenceModel.built_model(meta["L"], meta["C"], meta["arch_config"])
        network.load_weights(str(model_dir / "weights")).expect_partial()
        return RNNModel(network)

