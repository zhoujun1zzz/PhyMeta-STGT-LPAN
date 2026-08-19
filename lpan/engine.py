from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
from torch import nn

from .metrics import MetricAccumulator
from .objectives import LossWeights, combined_loss, progressive_charbonnier_loss
from .transfer import configure_adaptation, enforce_frozen_module_eval


def move_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def evaluate_model(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    metrics = MetricAccumulator()
    started = time.perf_counter()
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            prediction = model(batch)
            metrics.update(prediction, batch)
    result = metrics.compute()
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def train_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weights: LossWeights,
    *,
    grad_clip: float = 1.0,
    loss_profile: str = "combined",
) -> dict[str, float]:
    model.train()
    enforce_frozen_module_eval(model)
    totals: dict[str, float] = {}
    batches = 0
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        if loss_profile == "official_progressive_charbonnier":
            if not hasattr(model, "forward_multiscale"):
                raise ValueError(
                    "official_progressive_charbonnier requires forward_multiscale()."
                )
            predictions = model.forward_multiscale(batch)
            loss, parts = progressive_charbonnier_loss(predictions, batch)
        elif loss_profile == "combined":
            prediction = model(batch)
            loss, parts = combined_loss(prediction, batch, weights)
        else:
            raise ValueError(f"Unknown loss profile {loss_profile!r}.")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {parts}")
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


def train_balanced_joint_epoch(
    model: nn.Module,
    loaders: tuple[Iterable[Mapping[str, torch.Tensor]], ...],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weights: LossWeights,
    *,
    steps: int,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """Alternate homogeneous task batches, preventing invalid label padding."""
    model.train()
    enforce_frozen_module_eval(model)
    iterators = [iter(loader) for loader in loaders]
    totals: dict[str, float] = {}
    count = 0
    for step in range(steps):
        task = step % len(loaders)
        try:
            cpu_batch = next(iterators[task])
        except StopIteration:
            iterators[task] = iter(loaders[task])
            cpu_batch = next(iterators[task])
        batch = move_batch(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        loss, parts = combined_loss(prediction, batch, weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite joint loss: {parts}")
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / max(1, count) for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    best_nmse: float,
    model_name: str,
    model_config: dict[str, object],
    metadata: dict[str, object],
    rng_state: dict[str, object] | None = None,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_nmse": best_nmse,
        "model_name": model_name,
        "model_config": model_config,
        "metadata": metadata,
        "rng_state": rng_state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def capture_rng_state(
    loader_generators: Mapping[str, torch.Generator] | None = None,
) -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "loader_generators": {},
    }
    if loader_generators:
        state["loader_generators"] = {
            name: generator.get_state()
            for name, generator in loader_generators.items()
        }
    return state


def restore_rng_state(
    state: Mapping[str, object],
    loader_generators: Mapping[str, torch.Generator] | None = None,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_cpu_state = state["torch_cpu"]
    if not isinstance(torch_cpu_state, torch.Tensor):
        raise TypeError("torch_cpu RNG state must be a tensor.")
    torch.set_rng_state(
        torch_cpu_state.detach().cpu().to(dtype=torch.uint8).contiguous()
    )
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        if not isinstance(cuda_state, (list, tuple)):
            raise TypeError("torch_cuda RNG state must be a list or tuple.")
        normalized_cuda_state = []
        for index, item in enumerate(cuda_state):
            if not isinstance(item, torch.Tensor):
                raise TypeError(
                    f"torch_cuda RNG state at index {index} must be a tensor."
                )
            normalized_cuda_state.append(
                item.detach().cpu().to(dtype=torch.uint8).contiguous()
            )
        torch.cuda.set_rng_state_all(normalized_cuda_state)
    saved_generators = state.get("loader_generators", {})
    if loader_generators and isinstance(saved_generators, Mapping):
        for name, generator in loader_generators.items():
            if name in saved_generators:
                generator_state = saved_generators[name]
                if not isinstance(generator_state, torch.Tensor):
                    raise TypeError(
                        f"Generator RNG state for {name!r} must be a tensor."
                    )
                generator.set_state(
                    generator_state.detach()
                    .cpu()
                    .to(dtype=torch.uint8)
                    .contiguous()
                )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_history(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_history(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = {}
            for key, value in raw.items():
                if key == "epoch":
                    row[key] = int(value)
                else:
                    try:
                        row[key] = float(value)
                    except (TypeError, ValueError):
                        row[key] = value
            rows.append(row)
    return rows


def nmse_db_from_result(result: Mapping[str, object]) -> float:
    nmse = result["nmse_linear"]
    assert isinstance(nmse, Mapping)
    return 10 * math.log10(max(float(nmse["overall"]), 1e-30))
