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
from .objectives import LossWeights, combined_loss


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
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    batches = 0
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        loss, parts = combined_loss(prediction, batch, weights)
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


def configure_adaptation(model: nn.Module, policy: str) -> dict[str, object]:
    policy = policy.replace("-", "_")
    if policy != "full" and not hasattr(model, "domain_embedding"):
        raise ValueError(
            f"{policy} adaptation is only supported by PhyMetaSTGT; "
            "use policy='full' for baseline models."
        )
    for parameter in model.parameters():
        parameter.requires_grad = True
    if policy == "full":
        pass
    elif policy == "frozen_spatial":
        if not hasattr(model, "spatial_parameters"):
            raise ValueError("frozen_spatial requires PhyMetaSTGT.")
        for parameter in model.spatial_parameters():
            parameter.requires_grad = False
    elif policy in {"adapter_only", "selective"}:
        adapter_tokens = ("domain_embedding",)
        selective_tokens = adapter_tokens + (
            "time_encoder",
            "temporal_attention",
            "temporal_norm",
        )
        allowed = adapter_tokens if policy == "adapter_only" else selective_tokens
        for name, parameter in model.named_parameters():
            parameter.requires_grad = any(
                name == token or name.startswith(f"{token}.") for token in allowed
            )
    else:
        raise ValueError(
            "policy must be full, frozen_spatial, adapter_only, or selective."
        )
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable == 0:
        raise ValueError(f"Adaptation policy {policy!r} selected no parameters.")
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total,
        "trainable_parameter_names": trainable_names,
        "trainable_module_names": sorted(
            {name.split(".", 1)[0] for name in trainable_names}
        ),
    }


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
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    saved_generators = state.get("loader_generators", {})
    if loader_generators and isinstance(saved_generators, Mapping):
        for name, generator in loader_generators.items():
            if name in saved_generators:
                generator.set_state(saved_generators[name])


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
