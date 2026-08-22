from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, default_collate

from .data import (
    CANONICAL_MOBILITY_PROFILE,
    MOBILITY_BASELINE_CONTRACT_VERSION,
    LPANH5Dataset,
    default_observed_ris_indices,
    semantic_contract,
    semantic_fingerprint,
)
from .engine import evaluate_model
from .models import (
    build_model,
    lpan_grouped_raw_input,
    lpan_grouped_raw_output,
    lpan_raw_input,
    lpan_raw_output,
)
from .objectives import progressive_charbonnier_loss, sample_nmse
from .transfer import file_sha256


AUDIT_SEEDS = (123, 456, 789)
AUDIT_MODELS = ("lpan_progressive", "lpan_l_progressive")
LEGACY_PROFILE = "official_lpan"
FP32_REDUCTION_ATOL = 2e-6


def _maximum_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item()) if left.numel() else 0.0


def _check(
    left: torch.Tensor, right: torch.Tensor, *, atol: float = 0.0
) -> dict[str, object]:
    error = _maximum_error(left, right)
    return {
        "passed": bool(torch.allclose(left, right, rtol=0.0, atol=atol)),
        "max_abs_error": error,
        "absolute_tolerance": atol,
    }


def _batch(dataset: LPANH5Dataset, count: int) -> dict[str, torch.Tensor]:
    return default_collate([dataset[index] for index in range(min(count, len(dataset)))])


def paired_mobility_batches(
    path: str | Path,
    split: str,
    *,
    count: int = 64,
    seed: int = 123,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, object]]:
    """Decode identical sample IDs through legacy and canonical profiles."""
    legacy = LPANH5Dataset(
        path,
        "mobility",
        split,
        max_samples=count,
        subset_seed=seed,
        semantic_profile=LEGACY_PROFILE,
        complex_layout="interleaved",
        obs_time_index=(1, 4),
    )
    canonical = LPANH5Dataset(
        path,
        "mobility",
        split,
        max_samples=count,
        subset_seed=seed,
        semantic_profile=CANONICAL_MOBILITY_PROFILE,
        complex_layout="grouped",
        obs_time_index=(0, 3),
    )
    try:
        legacy_batch = _batch(legacy, count)
        canonical_batch = _batch(canonical, count)
        if not torch.equal(
            legacy_batch["sample_index"], canonical_batch["sample_index"]
        ):
            raise AssertionError("Legacy and canonical audit sample IDs differ.")
        provenance = {
            "path": str(Path(path).resolve()),
            "split": split,
            "sample_count": int(legacy_batch["sample_index"].numel()),
            "sample_indices": legacy_batch["sample_index"].tolist(),
            "subset_hash": legacy.subset_hash,
            "test_split_used": False,
        }
        return legacy_batch, canonical_batch, provenance
    finally:
        legacy.close()
        canonical.close()


def audit_batch_equivalence(
    legacy: Mapping[str, torch.Tensor],
    canonical: Mapping[str, torch.Tensor],
    *,
    seed: int = 2026,
) -> dict[str, dict[str, object]]:
    """Run the four tensor/loss/metric invariance checks on one paired batch."""
    effective_legacy_input = lpan_raw_input(legacy["obs_h"])
    effective_canonical_input = lpan_grouped_raw_input(canonical["obs_h"])
    effective_legacy_target = lpan_raw_input(legacy["target_h"])
    effective_canonical_target = lpan_grouped_raw_input(canonical["target_h"])

    generator = torch.Generator().manual_seed(seed)
    raw_predictions = tuple(
        torch.randn(
            canonical["target_h"].shape[0],
            12,
            64,
            width,
            generator=generator,
        )
        for width in (64, 128, 256)
    )
    legacy_predictions = tuple(lpan_raw_output(value, 6) for value in raw_predictions)
    canonical_predictions = tuple(
        lpan_grouped_raw_output(value, 6) for value in raw_predictions
    )
    legacy_loss, _ = progressive_charbonnier_loss(legacy_predictions, legacy)
    canonical_loss, _ = progressive_charbonnier_loss(
        canonical_predictions, canonical
    )
    legacy_nmse = sample_nmse(legacy_predictions[-1], legacy["target_h"])
    canonical_nmse = sample_nmse(
        canonical_predictions[-1], canonical["target_h"]
    )
    return {
        "effective_input_equal": _check(
            effective_legacy_input, effective_canonical_input
        ),
        "effective_target_equal": _check(
            effective_legacy_target, effective_canonical_target
        ),
        "progressive_loss_equal": _check(
            legacy_loss, canonical_loss, atol=FP32_REDUCTION_ATOL
        ),
        "nmse_permutation_invariant": _check(
            legacy_nmse, canonical_nmse, atol=FP32_REDUCTION_ATOL
        ),
    }


def _checkpoint_state(path: Path) -> dict[str, object]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {path} is not a dictionary.")
    return state


def checkpoint_compatibility(
    model_name: str, checkpoint: str | Path
) -> tuple[dict[str, object], torch.nn.Module | None]:
    path = Path(checkpoint)
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "missing_keys": [],
        "unexpected_keys": [],
        "strict_load": False,
    }
    if not path.is_file():
        result["error"] = "checkpoint not found"
        return result, None
    try:
        state = _checkpoint_state(path)
        if state.get("model_name") != model_name:
            raise ValueError(
                f"model_name={state.get('model_name')!r}, expected {model_name!r}"
            )
        config = state.get("model_config")
        if not isinstance(config, dict):
            config = {"domain": "mobility"}
        model = build_model(model_name, **config)
        weights = state.get("model_state")
        if not isinstance(weights, dict):
            raise ValueError("model_state is missing")
        incompatible = model.load_state_dict(weights, strict=False)
        result["missing_keys"] = list(incompatible.missing_keys)
        result["unexpected_keys"] = list(incompatible.unexpected_keys)
        model.load_state_dict(weights, strict=True)
        result.update(
            {
                "strict_load": True,
                "sha256": file_sha256(path),
                "passed": not incompatible.missing_keys
                and not incompatible.unexpected_keys,
            }
        )
        return result, model
    except Exception as exc:
        result.update({"passed": False, "error": f"{type(exc).__name__}: {exc}"})
        return result, None


def time_label_independence(
    model: torch.nn.Module, batch: Mapping[str, torch.Tensor]
) -> dict[str, object]:
    model.eval()
    original = {key: value[:1].clone() for key, value in batch.items()}
    relabeled = {key: value.clone() for key, value in original.items()}
    relabeled["obs_time_index"] = torch.tensor([[17, 41]])
    relabeled["query_time"] = torch.tensor([[9, 3, 22, 1, 8, 55]])
    with torch.inference_mode():
        first = model.forward_multiscale(original)
        second = model.forward_multiscale(relabeled)
    error = max(_maximum_error(left, right) for left, right in zip(first, second))
    return {"passed": error == 0.0, "max_abs_error": error}


def discover_checkpoints(
    root: str | Path, model_name: str, seeds: Sequence[int] = AUDIT_SEEDS
) -> dict[int, Path]:
    root = Path(root)
    found: dict[int, Path] = {}
    for seed in seeds:
        pattern = f"mobility_{model_name}_seed{seed}/checkpoints/best_checkpoint.pth"
        matches = sorted(root.rglob(pattern)) if root.exists() else []
        if matches:
            found[int(seed)] = matches[0]
    return found


def run_semantic_audit(
    train_path: str | Path,
    validation_path: str | Path,
    checkpoint_root: str | Path,
    *,
    count: int = 64,
) -> dict[str, object]:
    split_checks: dict[str, object] = {}
    canonical_validation: dict[str, torch.Tensor] | None = None
    provenance: list[dict[str, object]] = []
    for split, path in (("train", train_path), ("validation", validation_path)):
        legacy, canonical, split_provenance = paired_mobility_batches(
            path, split, count=count
        )
        split_checks[split] = audit_batch_equivalence(legacy, canonical)
        provenance.append(split_provenance)
        if split == "validation":
            canonical_validation = canonical
    assert canonical_validation is not None

    model_results: dict[str, object] = {}
    for model_name in AUDIT_MODELS:
        paths = discover_checkpoints(checkpoint_root, model_name)
        checkpoint_rows = []
        representative: torch.nn.Module | None = None
        for seed in AUDIT_SEEDS:
            row, loaded = checkpoint_compatibility(
                model_name, paths.get(seed, Path(f"missing-seed-{seed}.pth"))
            )
            row["seed"] = seed
            checkpoint_rows.append(row)
            representative = representative or loaded
        time_check = (
            time_label_independence(representative, canonical_validation)
            if representative is not None
            else {"passed": False, "error": "no compatible checkpoint available"}
        )
        aggregate_checks = {
            name: all(
                bool(split[name]["passed"]) for split in split_checks.values()
            )
            for name in (
                "effective_input_equal",
                "effective_target_equal",
                "progressive_loss_equal",
                "nmse_permutation_invariant",
            )
        }
        tensor_checks_pass = all(aggregate_checks.values())
        checkpoints_pass = len(paths) == len(AUDIT_SEEDS) and all(
            bool(row.get("passed")) for row in checkpoint_rows
        )
        status = (
            "REUSE_VERIFIED"
            if tensor_checks_pass and checkpoints_pass and time_check["passed"]
            else "RERUN_REQUIRED"
        )
        model_results[model_name] = {
            "checks": {
                **aggregate_checks,
                "time_label_independent": time_check,
                "checkpoint_strict_compatible": checkpoints_pass,
            },
            "checkpoint_audit": checkpoint_rows,
            "status": status,
        }
    contract = semantic_contract(
        domain="mobility",
        semantic_profile=CANONICAL_MOBILITY_PROFILE,
        complex_layout="grouped",
        obs_time_index=(0, 3),
        obs_ris_index=default_observed_ris_indices(),
    )
    return {
        "contract_version": MOBILITY_BASELINE_CONTRACT_VERSION,
        "semantic_contract": contract,
        "semantic_fingerprint": semantic_fingerprint(contract),
        "audit_sample_count_per_split": count,
        "split_checks": split_checks,
        "models": model_results,
        "formal_training_loss": {
            "profile": "official_progressive_charbonnier",
            "terms": ["hr2_charbonnier", "hr4_charbonnier", "hr8_charbonnier"],
            "observation_consistency": False,
            "temporal_delta": False,
            "time_labels_consumed": False,
        },
        "dataset_provenance": provenance,
        "test_split_used": False,
    }


def git_head(project: str | Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project, text=True
    ).strip()


def evaluate_canonical_checkpoints(
    checkpoint_root: str | Path,
    validation_path: str | Path,
    *,
    project: str | Path,
    device: torch.device,
    batch_size: int = 64,
    workers: int = 0,
    models: Sequence[str] = AUDIT_MODELS,
    seeds: Sequence[int] = AUDIT_SEEDS,
) -> dict[str, object]:
    """Re-evaluate reusable LPAN weights on canonical VALIDATION only."""
    contract = semantic_contract(
        domain="mobility",
        semantic_profile=CANONICAL_MOBILITY_PROFILE,
        complex_layout="grouped",
        obs_time_index=(0, 3),
        obs_ris_index=default_observed_ris_indices(),
    )
    rows = []
    for model_name in models:
        if model_name not in AUDIT_MODELS:
            raise ValueError(f"Canonical reuse evaluation rejects {model_name!r}.")
        for seed, checkpoint in discover_checkpoints(
            checkpoint_root, model_name, seeds
        ).items():
            compatibility, model = checkpoint_compatibility(model_name, checkpoint)
            if model is None:
                rows.append({"model": model_name, "seed": seed, **compatibility})
                continue
            dataset = LPANH5Dataset(
                validation_path,
                "mobility",
                "validation",
                semantic_profile=CANONICAL_MOBILITY_PROFILE,
                complex_layout="grouped",
                obs_time_index=(0, 3),
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=workers,
                shuffle=False,
            )
            try:
                metrics = evaluate_model(model.to(device), loader, device)
            finally:
                dataset.close()
            rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": compatibility.get("sha256"),
                    "metrics": metrics,
                }
            )
    return {
        "git_head": git_head(project),
        "contract_version": MOBILITY_BASELINE_CONTRACT_VERSION,
        "semantic_contract": contract,
        "semantic_fingerprint": semantic_fingerprint(contract),
        "dataset_provenance": {
            "path": str(Path(validation_path).resolve()),
            "split": "validation",
        },
        "results": rows,
        "test_split_used": False,
    }
