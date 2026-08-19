from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from torch import nn


ADAPTATION_MODES = (
    "scratch",
    "full_finetune",
    "frozen_spatial",
    "domain_adapter_only",
    "adapter_head",
)

# Kept for old commands/checkpoints. Formal low-data runs use ADAPTATION_MODES.
LEGACY_ADAPTATION_MODES = ("full", "adapter_only", "selective")


def _matches_group(name: str, groups: Iterable[str]) -> bool:
    return any(name == group or name.startswith(f"{group}.") for group in groups)


def configure_adaptation(model: nn.Module, policy: str) -> dict[str, object]:
    """Apply one centralized V1 transfer parameter policy.

    ``adapter_head`` deliberately means the domain FiLM embedding plus the
    prediction decoder. The temporal attention stack remains part of the
    shared backbone. ``frozen_spatial`` is broader: it trains every non-spatial
    target-side module.
    """
    policy = policy.replace("-", "_")
    supported = set(ADAPTATION_MODES) | set(LEGACY_ADAPTATION_MODES)
    if policy not in supported:
        raise ValueError(f"policy must be one of {sorted(supported)}.")
    if policy not in {"scratch", "full", "full_finetune"} and not hasattr(
        model, "domain_embedding"
    ):
        raise ValueError(
            f"{policy} adaptation is only supported by PhyMetaSTGT; "
            "use policy='scratch' for baseline models."
        )

    for parameter in model.parameters():
        parameter.requires_grad = True

    if policy in {"scratch", "full", "full_finetune"}:
        pass
    elif policy == "frozen_spatial":
        if not hasattr(model, "spatial_parameters"):
            raise ValueError("frozen_spatial requires PhyMetaSTGT.")
        for parameter in model.spatial_parameters():
            parameter.requires_grad = False
    else:
        groups_by_policy = {
            "domain_adapter_only": ("domain_embedding",),
            "adapter_only": ("domain_embedding",),
            "adapter_head": ("domain_embedding", "decoder"),
            # Preserve the previous experimental policy outside formal Table 2.
            "selective": (
                "domain_embedding",
                "time_encoder",
                "temporal_attention",
                "temporal_norm",
            ),
        }
        allowed = groups_by_policy[policy]
        for name, parameter in model.named_parameters():
            parameter.requires_grad = _matches_group(name, allowed)

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
    frozen_names = [
        name for name, parameter in model.named_parameters() if not parameter.requires_grad
    ]
    frozen_modules = []
    for name, module in model.named_children():
        parameters = list(module.parameters())
        if parameters and not any(parameter.requires_grad for parameter in parameters):
            frozen_modules.append(name)
    setattr(model, "_adaptation_frozen_module_names", tuple(sorted(frozen_modules)))

    return {
        "policy": policy,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total,
        "trainable_ratio_percent": 100.0 * trainable / total,
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": frozen_names,
        "trainable_module_names": sorted(
            {name.split(".", 1)[0] for name in trainable_names}
        ),
        "frozen_module_names": sorted(
            {name.split(".", 1)[0] for name in frozen_names}
        ),
        "eval_locked_module_names": tuple(sorted(frozen_modules)),
    }


def enforce_frozen_module_eval(model: nn.Module) -> None:
    """Keep wholly frozen modules out of training mode after ``model.train()``."""
    names = getattr(model, "_adaptation_frozen_module_names", ())
    modules = dict(model.named_modules())
    for name in names:
        modules[name].eval()


def deterministic_subset_indices(
    total: int,
    fraction: float,
    seed: int,
    *,
    max_samples: int | None = None,
) -> np.ndarray:
    """Return a sorted prefix of one seeded permutation.

    Reusing the same ``total`` and ``seed`` makes all requested fractions
    nested and identical across methods.
    """
    if total <= 0:
        raise ValueError("total must be positive.")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    cap = total if max_samples is None else min(total, int(max_samples))
    if cap <= 0:
        raise ValueError("max_samples must be positive when provided.")
    count = max(1, int(np.floor(cap * fraction)))
    if count == total:
        return np.arange(total, dtype=np.int64)
    permutation = np.random.default_rng(seed).permutation(total)
    return np.sort(permutation[:count]).astype(np.int64)


def subset_index_hash(indices: np.ndarray) -> str:
    canonical = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
