from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


def sample_nmse(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    dims = tuple(range(1, prediction.ndim))
    error = (prediction - target).square().sum(dim=dims)
    power = target.square().sum(dim=dims)
    return error / power.clamp_min(eps)


def charbonnier(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3
) -> torch.Tensor:
    dims = tuple(range(1, prediction.ndim))
    target_rms = target.square().mean(dim=dims).sqrt().clamp_min(1e-12)
    robust_error = torch.sqrt(
        (prediction - target).square()
        + (eps * target_rms.reshape(-1, *([1] * (prediction.ndim - 1)))) ** 2
    ).mean(dim=dims)
    return (robust_error / target_rms).mean()


def observation_consistency(
    prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    obs = batch["obs_h"]
    ris_index = batch["obs_ris_index"]
    time_index = batch["obs_time_index"]
    ris = ris_index[0] if ris_index.ndim == 2 else ris_index
    times = time_index[0] if time_index.ndim == 2 else time_index
    selected = prediction.index_select(1, times).index_select(2, ris)
    mask = batch["observation_mask"].to(device=obs.device, dtype=torch.bool)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if tuple(mask.shape) != tuple(obs.shape[:3]):
        raise ValueError(
            "observation_mask must match [batch, observed_time, observed_RIS]."
        )
    valid_per_sample = mask.reshape(mask.shape[0], -1).any(dim=1)
    if not bool(valid_per_sample.all()):
        raise ValueError(
            "observation_consistency requires at least one valid observation "
            "for every sample."
        )
    expanded_mask = mask.unsqueeze(-1).unsqueeze(-1).to(obs.dtype)
    dims = tuple(range(1, obs.ndim))
    numerator = ((selected - obs).square() * expanded_mask).sum(dim=dims)
    denominator = (obs.square() * expanded_mask).sum(dim=dims)
    return (numerator / denominator.clamp_min(1e-12)).mean()


def delta_nmse(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    if prediction.shape[1] < 2:
        return prediction.new_zeros(())
    return sample_nmse(
        prediction[:, 1:] - prediction[:, :-1],
        target[:, 1:] - target[:, :-1],
        eps,
    ).mean()


@dataclass
class LossWeights:
    nmse: float = 1.0
    charbonnier: float = 0.1
    observation: float = 0.1
    delta: float = 0.05


def combined_loss(
    prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["target_h"]
    parts = {
        "nmse": sample_nmse(prediction, target).mean(),
        "charbonnier": charbonnier(prediction, target),
        "observation": observation_consistency(prediction, batch),
        "delta": delta_nmse(prediction, target),
    }
    total = (
        weights.nmse * parts["nmse"]
        + weights.charbonnier * parts["charbonnier"]
        + weights.observation * parts["observation"]
        + weights.delta * parts["delta"]
    )
    values = {key: float(value.detach()) for key, value in parts.items()}
    values["total"] = float(total.detach())
    return total, values
