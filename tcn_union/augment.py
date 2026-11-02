"""Training-only augmentation for raw gas-sensor response prefixes.

The augmentation is deliberately applied in the raw response domain before
``make_features``.  This keeps the derived ``raw_delta`` channels physically
consistent with the perturbed raw channels and never touches labels unless an
explicit mixup view is requested.

Nothing in this module is used for validation, test, or inference.  A view is
generated online once per training run per epoch, so the number of optimizer
updates does not change when augmentation is enabled.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import TIME_STEP_SECONDS, make_features

CHANNELS = ("H2", "NH3", "NO2")


@dataclass(frozen=True)
class AugmentConfig:
    """One named augmentation recipe.

    Parameters are intentionally expressed as interpretable fractions:
    ``noise`` is a fraction of the median absolute initial response, ``gain``
    and ``transient`` are relative standard deviations, ``offset`` is a
    fraction of the initial response, and ``warp`` is the maximum log-time
    distortion.
    """

    name: str
    probability: float = 1.0
    noise: float = 0.0
    gain: float = 0.0
    offset: float = 0.0
    warp: float = 0.0
    transient: float = 0.0
    transient_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mixup_alpha: float = 0.0
    mixup_probability: float = 0.0
    prefix_probability: float = 0.0
    prefix_min_seconds: float = 5.0
    cutout_probability: float = 0.0
    cutout_max_seconds: float = 2.0
    prefix_lengths: tuple[float, ...] = ()
    clip_z: float = 3.0
    extra: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("augmentation name must not be empty")
        for field_name in (
            "probability",
            "noise",
            "gain",
            "offset",
            "warp",
            "transient",
            "mixup_alpha",
            "mixup_probability",
            "prefix_probability",
            "cutout_probability",
            "cutout_max_seconds",
            "clip_z",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        if not 0.0 <= self.mixup_probability <= 1.0:
            raise ValueError("mixup_probability must be in [0, 1]")
        if not 0.0 <= self.prefix_probability <= 1.0:
            raise ValueError("prefix_probability must be in [0, 1]")
        if not 0.0 <= self.cutout_probability <= 1.0:
            raise ValueError("cutout_probability must be in [0, 1]")
        if self.prefix_probability > 0.0 and self.prefix_min_seconds <= 0.0:
            raise ValueError("prefix_min_seconds must be positive")
        if self.mixup_alpha > 0.0 and self.mixup_probability <= 0.0:
            raise ValueError("mixup requires a positive mixup_probability")
        if len(self.transient_weights) != 3:
            raise ValueError("transient_weights must contain three channels")
        if any(not math.isfinite(float(v)) or float(v) < 0.0 for v in self.transient_weights):
            raise ValueError("transient_weights must be finite and non-negative")
        if self.prefix_lengths:
            if any(not math.isfinite(float(v)) or float(v) <= 0.0 for v in self.prefix_lengths):
                raise ValueError("prefix_lengths must be positive and finite")
            if max(float(v) for v in self.prefix_lengths) > 10.0 + 1e-9:
                raise ValueError("prefix_lengths must not exceed 10 s")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "probability": self.probability,
            "noise": self.noise,
            "gain": self.gain,
            "offset": self.offset,
            "warp": self.warp,
            "transient": self.transient,
            "transient_weights": list(self.transient_weights),
            "mixup_alpha": self.mixup_alpha,
            "mixup_probability": self.mixup_probability,
            "prefix_probability": self.prefix_probability,
            "prefix_min_seconds": self.prefix_min_seconds,
            "cutout_probability": self.cutout_probability,
            "cutout_max_seconds": self.cutout_max_seconds,
            "prefix_lengths": list(self.prefix_lengths),
            "clip_z": self.clip_z,
        }
        payload.update(self.extra)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AugmentConfig:
        data = dict(payload)
        name = str(data.pop("name"))
        known = {
            "probability",
            "noise",
            "gain",
            "offset",
            "warp",
            "transient",
            "mixup_alpha",
            "mixup_probability",
            "prefix_probability",
            "prefix_min_seconds",
            "cutout_probability",
            "cutout_max_seconds",
            "prefix_lengths",
            "clip_z",
        }
        extra = {key: value for key, value in data.items() if key not in known}
        values = {key: data[key] for key in known if key in data}
        if "prefix_lengths" in values:
            values["prefix_lengths"] = tuple(float(v) for v in values["prefix_lengths"])
        if "transient_weights" in values:
            values["transient_weights"] = tuple(float(v) for v in values["transient_weights"])
        return cls(name=name, **values, extra=extra)


def prepend_constant_hold(responses: np.ndarray, cut: int) -> np.ndarray:
    """Return a copy whose suffix after ``cut`` is held at the last real value."""
    if responses.ndim != 2 or responses.shape[1] != 3:
        raise ValueError("responses must have shape [time, 3]")
    if not 1 <= cut < responses.shape[0]:
        raise ValueError("cut must leave at least one real sample and one held sample")
    result = responses.copy()
    result[cut:] = result[cut - 1]
    return result


def _interp_channels(time: np.ndarray, responses: np.ndarray, source_time: np.ndarray) -> np.ndarray:
    """Linearly interpolate all three response channels at ``source_time``."""
    result = np.empty_like(responses, dtype=np.float64)
    for channel in range(responses.shape[1]):
        result[:, channel] = np.interp(source_time, time, responses[:, channel])
    return result


def warp_time(
    time: np.ndarray, responses: np.ndarray, log_warp: float
) -> np.ndarray:
    """Warp the time axis inside the observed prefix without reading the future.

    ``t_source = T * (t/T) ** exp(log_warp)`` stays in ``[0, T]`` for every
    positive distortion, so a warped view never accesses data after the
    requested endpoint.
    """
    time = np.asarray(time, dtype=np.float64)
    responses = np.asarray(responses, dtype=np.float64)
    if time.ndim != 1 or responses.shape != (time.size, 3):
        raise ValueError("expected time[N] and responses[N,3]")
    if not math.isfinite(log_warp):
        raise ValueError("log_warp must be finite")
    if abs(float(time[0])) > 1e-6:
        raise ValueError("time must start at zero")
    end = float(time[-1])
    if end <= 0.0:
        raise ValueError("prefix must have a positive endpoint")
    gamma = math.exp(float(log_warp))
    source_time = end * np.power(time / end, gamma)
    return _interp_channels(time, responses, source_time)


def cutout(
    responses: np.ndarray, start: int, length: int
) -> np.ndarray:
    """Replace one interior segment by a linear bridge between its boundaries."""
    if responses.ndim != 2 or responses.shape[1] != 3:
        raise ValueError("responses must have shape [time, 3]")
    if start < 1 or length < 1 or start + length >= responses.shape[0]:
        raise ValueError("cutout must leave both boundary samples intact")
    result = responses.copy()
    left = result[start - 1]
    right = result[start + length]
    weights = (np.arange(1, length + 1, dtype=np.float64) / (length + 1))[:, None]
    result[start : start + length] = left[None, :] * (1.0 - weights) + right[None, :] * weights
    return result


@dataclass(frozen=True)
class AugmentContext:
    """Fold-local raw data and settings needed by the online dataset."""

    rows: Sequence[dict[str, Any]]
    seconds: float
    mode: str
    config: AugmentConfig
    base_seed: int
    noise_scale: np.ndarray

    def __post_init__(self) -> None:
        if self.seconds not in {5.0, 10.0}:
            raise ValueError("augmentation supports 5 s and 10 s prefixes")
        if self.mode not in {"raw_delta", "delta"}:
            raise ValueError("augmentation supports raw_delta and delta modes")
        if len(self.rows) == 0:
            raise ValueError("augmentation needs at least one training run")
        if self.noise_scale.shape != (3,):
            raise ValueError("noise_scale must contain three channels")
        if not np.isfinite(self.noise_scale).all() or np.any(self.noise_scale < 0.0):
            raise ValueError("noise_scale must be finite and non-negative")
        if self.config.prefix_lengths and max(self.config.prefix_lengths) > self.seconds + 1e-9:
            raise ValueError("prefix_lengths must not exceed the training window")


def estimate_initial_scale(rows: Sequence[dict[str, Any]], seconds: float) -> np.ndarray:
    """Median absolute initial response per channel for the current training set."""
    values = []
    for row in rows:
        time = np.asarray(row["_time"], dtype=np.float64)
        response = np.asarray(row["_response"], dtype=np.float64)
        mask = time <= seconds + 1e-9
        if not np.any(mask):
            raise ValueError(f"{row['file']}: no samples in requested prefix")
        values.append(np.abs(response[mask][0]))
    return np.median(np.stack(values), axis=0)


def _apply_recipe(
    time: np.ndarray,
    responses: np.ndarray,
    label: np.ndarray,
    partner: tuple[np.ndarray, np.ndarray] | None,
    context: AugmentContext,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one stochastic view to a raw prefix and return (features, label)."""
    cfg = context.config
    responses = np.asarray(responses, dtype=np.float64).copy()
    time = np.asarray(time, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64).copy()

    if partner is not None and cfg.mixup_probability > 0.0 and rng.random() < cfg.mixup_probability:
        partner_response, partner_label = partner
        alpha = float(cfg.mixup_alpha)
        if alpha <= 0.0:
            alpha = 0.2
        lam = float(rng.beta(alpha, alpha))
        lam = float(np.clip(lam, 0.2, 0.8))
        responses = lam * responses + (1.0 - lam) * np.asarray(partner_response, dtype=np.float64)
        label = lam * label + (1.0 - lam) * np.asarray(partner_label, dtype=np.float64)

    if cfg.prefix_probability > 0.0 and context.seconds == 10.0 and rng.random() < cfg.prefix_probability:
        min_samples = max(2, round(cfg.prefix_min_seconds / TIME_STEP_SECONDS) + 1)
        min_samples = min(min_samples, responses.shape[0] - 1)
        if min_samples < responses.shape[0]:
            cut = int(rng.integers(min_samples, responses.shape[0]))
            responses = prepend_constant_hold(responses, cut)

    if cfg.warp > 0.0 and rng.random() < 0.5:
        log_warp = float(rng.uniform(-cfg.warp, cfg.warp))
        responses = warp_time(time, responses, log_warp)

    if cfg.transient > 0.0:
        transient_std = cfg.transient * np.asarray(cfg.transient_weights, dtype=np.float64)
        scale = 1.0 + np.clip(
            rng.normal(0.0, 1.0, size=3) * transient_std,
            -cfg.clip_z * transient_std,
            cfg.clip_z * transient_std,
        )
        responses = responses[0][None, :] + scale[None, :] * (responses - responses[0][None, :])

    if cfg.gain > 0.0:
        gain = 1.0 + np.clip(
            rng.normal(0.0, cfg.gain, size=3), -cfg.clip_z * cfg.gain, cfg.clip_z * cfg.gain
        )
        responses = responses * gain[None, :]

    if cfg.offset > 0.0:
        offset = rng.normal(0.0, cfg.offset * np.maximum(np.abs(responses[0]), 1e-8), size=3)
        responses = responses + offset[None, :]

    if cfg.cutout_probability > 0.0 and rng.random() < cfg.cutout_probability:
        max_length = max(1, round(cfg.cutout_max_seconds / TIME_STEP_SECONDS))
        max_length = min(max_length, responses.shape[0] - 2)
        if max_length >= 1:
            length = int(rng.integers(1, max_length + 1))
            start = int(rng.integers(1, responses.shape[0] - length))
            responses = cutout(responses, start, length)

    if cfg.noise > 0.0:
        noise = rng.standard_normal(responses.shape)
        noise = np.clip(noise, -cfg.clip_z, cfg.clip_z)
        responses = responses + cfg.noise * context.noise_scale[None, :] * noise

    if not np.isfinite(responses).all() or not np.isfinite(label).all():
        raise FloatingPointError("augmentation produced a non-finite view")
    features = make_features(time, responses, context.seconds, context.mode)
    return features, label.astype(np.float32)


class AugmentedDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Online raw-prefix augmentation dataset for one fold and one epoch."""

    def __init__(self, context: AugmentContext) -> None:
        self.context = context
        self.epoch = 0
        self._times = [
            np.asarray(row["_time"], dtype=np.float64)[
                np.asarray(row["_time"], dtype=np.float64) <= context.seconds + 1e-9
            ]
            for row in context.rows
        ]
        self._responses = [
            np.asarray(row["_response"], dtype=np.float64)[
                np.asarray(row["_time"], dtype=np.float64) <= context.seconds + 1e-9
            ]
            for row in context.rows
        ]
        self._labels = [np.asarray(row["_labels"], dtype=np.float64) for row in context.rows]
        expected = round(context.seconds / TIME_STEP_SECONDS) + 1
        if any(len(t) != expected for t in self._times):
            raise ValueError("training prefixes do not match the requested window")

    def __len__(self) -> int:
        return len(self.context.rows)

    def _prefix_samples(self) -> int | None:
        lengths = self.context.config.prefix_lengths
        if not lengths:
            return None
        rng = np.random.default_rng(
            int(self.context.base_seed) + 99_991 * int(self.epoch) + 17
        )
        seconds = float(lengths[int(rng.integers(0, len(lengths)))])
        return round(seconds / TIME_STEP_SECONDS) + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.context.config
        prefix_samples = self._prefix_samples()
        seed = (
            int(self.context.base_seed)
            + 1_000_003 * int(self.epoch)
            + 7919 * int(index)
        ) % (2**32)
        rng = np.random.default_rng(seed)
        if cfg.probability < 1.0 and rng.random() >= cfg.probability:
            features = make_features(
                self._times[index], self._responses[index], self.context.seconds, self.context.mode
            )
            if prefix_samples is not None:
                features = np.ascontiguousarray(features[:, :prefix_samples])
            label = self._labels[index].astype(np.float32)
            return torch.from_numpy(features), torch.from_numpy(label)

        partner = None
        if cfg.mixup_probability > 0.0 and len(self) > 1:
            partner_index = int(rng.integers(0, len(self) - 1))
            if partner_index >= index:
                partner_index += 1
            partner = (self._responses[partner_index], self._labels[partner_index])
        features, label = _apply_recipe(
            self._times[index],
            self._responses[index],
            self._labels[index],
            partner,
            self.context,
            rng,
        )
        if prefix_samples is not None:
            features = np.ascontiguousarray(features[:, :prefix_samples])
        return torch.from_numpy(features), torch.from_numpy(label)


def make_context(
    rows: Sequence[dict[str, Any]],
    seconds: float,
    mode: str,
    config: AugmentConfig,
    base_seed: int,
) -> AugmentContext:
    """Build a context with a training-set-only initial-response scale."""
    return AugmentContext(
        rows=rows,
        seconds=seconds,
        mode=mode,
        config=config,
        base_seed=base_seed,
        noise_scale=estimate_initial_scale(rows, seconds),
    )
