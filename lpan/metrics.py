from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping

import torch

from .objectives import sample_nmse


def _db(value: float) -> float:
    return 10.0 * math.log10(max(value, 1e-30))


class MetricAccumulator:
    """Accumulates ratios in linear scale and converts only final means to dB."""

    def __init__(self) -> None:
        self.sums: defaultdict[str, float] = defaultdict(float)
        self.counts: defaultdict[str, int] = defaultdict(int)

    @staticmethod
    def _require_finite(key: str, values: torch.Tensor) -> torch.Tensor:
        detached = values.detach()
        finite_mask = torch.isfinite(detached)
        if not bool(finite_mask.all()):
            invalid = detached[~finite_mask]
            nan_count = int(torch.isnan(invalid).sum().item())
            positive_inf = int(torch.isposinf(invalid).sum().item())
            negative_inf = int(torch.isneginf(invalid).sum().item())
            raise FloatingPointError(
                f"Metric input {key!r} contains non-finite values: "
                f"NaN={nan_count}, +Inf={positive_inf}, -Inf={negative_inf}. "
                "The batch was rejected; no samples were accumulated."
            )
        return detached

    def _add(self, key: str, values: torch.Tensor) -> None:
        finite = self._require_finite(key, values).double().cpu()
        self.sums[key] += float(finite.sum())
        self.counts[key] += int(finite.numel())

    def update(
        self,
        prediction: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> None:
        target = batch["target_h"]
        self._require_finite("prediction", prediction)
        self._require_finite("target_h", target)
        self._add("overall", sample_nmse(prediction, target))
        for block in range(target.shape[1]):
            values = sample_nmse(prediction[:, block], target[:, block])
            self._add(
                f"block_{block + 1}",
                values,
            )
            self._add(f"q{block}", values)
        ris = batch["obs_ris_index"][0] if batch["obs_ris_index"].ndim == 2 else batch["obs_ris_index"]
        self._add(
            "observed_ris",
            sample_nmse(
                prediction.index_select(2, ris), target.index_select(2, ris)
            ),
        )
        mask = torch.ones(target.shape[2], dtype=torch.bool, device=target.device)
        mask[ris] = False
        unobserved = torch.where(mask)[0]
        self._add(
            "unobserved_ris",
            sample_nmse(
                prediction.index_select(2, unobserved),
                target.index_select(2, unobserved),
            ),
        )
        times = batch["obs_time_index"][0] if batch["obs_time_index"].ndim == 2 else batch["obs_time_index"]
        self._add(
            "pilot_blocks",
            sample_nmse(
                prediction.index_select(1, times), target.index_select(1, times)
            ),
        )
        time_mask = torch.ones(target.shape[1], dtype=torch.bool, device=target.device)
        time_mask[times] = False
        nonpilot = torch.where(time_mask)[0]
        if nonpilot.numel():
            self._add(
                "nonpilot_blocks",
                sample_nmse(
                    prediction.index_select(1, nonpilot),
                    target.index_select(1, nonpilot),
                ),
            )
        if target.shape[1] == 6 and tuple(int(v) for v in times.tolist()) == (0, 3):
            for key, indices in {
                "anchor_q0_q3": (0, 3),
                "nonpilot_q1_q2_q4_q5": (1, 2, 4, 5),
                "interpolation_q1_q2": (1, 2),
                "extrapolation_q4_q5": (4, 5),
            }.items():
                selected = torch.tensor(indices, device=target.device)
                self._add(
                    key,
                    sample_nmse(
                        prediction.index_select(1, selected),
                        target.index_select(1, selected),
                    ),
                )

    def compute(self) -> dict[str, float | int | dict[str, float]]:
        linear = {
            key: self.sums[key] / self.counts[key]
            for key in sorted(self.sums)
            if self.counts[key]
        }
        return {
            "sample_count": self.counts.get("overall", 0),
            "nmse_linear": linear,
            "nmse_db": {key: _db(value) for key, value in linear.items()},
        }
