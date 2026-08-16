from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import shlex
import shutil
import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from lpan.complexity import (
    INTERPOLATION_POLICY,
    canonical_batch,
    profile_model_complexity,
)
from lpan.data import LPANH5Dataset
from lpan.engine import (
    capture_rng_state,
    configure_adaptation,
    evaluate_model,
    move_batch,
    nmse_db_from_result,
    read_history,
    restore_rng_state,
    save_checkpoint,
    train_balanced_joint_epoch,
    train_epoch,
    write_history,
    write_json,
)
from lpan.metrics import MetricAccumulator
from lpan.models import build_model, interpolation_baseline
from lpan.objectives import LossWeights, combined_loss
from lpan.paths import (
    dataset_candidates,
    default_data_root,
    resolve_dataset_path,
)
from lpan.ridge import EmpiricalRidge, RidgeStatistics
from lpan.studies import (
    ABLATION_VARIANTS,
    ARCHITECTURE_ABLATIONS,
    ablation_metadata,
    ablated_loss_weights,
    architectural_ablation,
    hyperparameter_candidates,
)


PROJECT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = default_data_root(PROJECT)
MOBILITY_EXPECTED_SAMPLES = {
    "train": 20000,
    "validation": 1800,
    "test": 9000,
}
MODEL_DISPLAY_NAMES = {
    "lpan_progressive": "LPAN",
    "lpan_l_progressive": "LPAN-L",
    "lpan_l_direct": "LPAN-L-Direct",
    "edsr_lite": "EDSR-lite",
    "spatial_gcn": "Spatial GCN",
    "cnn_gru": "CNN-GRU",
    "gcn_gru": "GCN-GRU",
    "phymeta_stgt": "PhyMeta-STGT",
}
PROGRESSIVE_LPAN_MODELS = {"lpan_progressive", "lpan_l_progressive"}


def infer_semantic_profile(
    domain: str,
    obs_times: object,
    obs_ris_indices: object,
    complex_layout: object,
) -> str:
    expected_times = (0, 1) if domain == "mobility" else (0,)
    try:
        times = tuple(obs_times)  # type: ignore[arg-type]
        indices = tuple(obs_ris_indices)  # type: ignore[arg-type]
    except TypeError:
        return "custom"
    return (
        "official_lpan"
        if times == expected_times
        and indices == tuple(range(0, 256, 8))
        and complex_layout == "grouped"
        else "custom"
    )


def ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def path_for(
    domain: str,
    split: str,
    explicit: str | None = None,
    data_root: str | Path | None = None,
) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return resolve_dataset_path(data_root or DEFAULT_DATA_ROOT, domain, split)


def make_loader(
    domain: str,
    split: str,
    *,
    path: str | Path | None = None,
    data_root: str | Path | None = None,
    batch_size: int,
    workers: int,
    max_samples: int | None = None,
    fraction: float = 1.0,
    seed: int = 123,
    obs_times: tuple[int, ...] | None = None,
    obs_ris_indices: tuple[int, ...] | None = None,
    complex_layout: str = "grouped",
    semantic_profile: str = "official_lpan",
    shuffle: bool = False,
) -> DataLoader:
    dataset = LPANH5Dataset(
        path_for(domain, split, str(path) if path else None, data_root),
        domain,
        split,
        max_samples=max_samples,
        fraction=fraction,
        subset_seed=seed,
        obs_time_index=obs_times,
        obs_ris_index=obs_ris_indices,
        complex_layout=complex_layout,
        semantic_profile=semantic_profile,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=generator,
    )


def device_from(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_config(args: argparse.Namespace, domain: str) -> dict[str, object]:
    config: dict[str, object] = {
        "domain": domain,
        "hidden": args.hidden,
        "graph_layers": args.graph_layers,
        "heads": args.heads,
        "dropout": args.dropout,
    }
    ablation = architectural_ablation(getattr(args, "ablation", "none"))
    if ablation != "none":
        config["ablation"] = ablation
    return config


def model_parameter_report(model: torch.nn.Module) -> dict[str, float | int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / max(1, total),
    }


def resolve_training_profile(args: argparse.Namespace) -> dict[str, object]:
    """Resolve model-specific optimization without changing unified evaluation."""
    requested = getattr(args, "training_profile", "auto")
    if requested == "auto":
        name = (
            "lpan_public_code"
            if args.model in PROGRESSIVE_LPAN_MODELS
            else "unified"
        )
    else:
        name = requested
    if name == "lpan_public_code" and args.model not in PROGRESSIVE_LPAN_MODELS:
        raise ValueError(
            "lpan_public_code training is only valid for progressive LPAN models."
        )
    if name == "lpan_public_code":
        return {
            "name": name,
            "loss_profile": "official_progressive_charbonnier",
            "optimizer": "AdamW",
            "betas": [0.9, 0.999],
            "initial_learning_rate": float(args.lpan_initial_learning_rate),
            "final_learning_rate": float(args.lpan_final_learning_rate),
            "weight_decay": float(args.lpan_weight_decay),
            "schedule": "cosine_epoch_deterministic",
            "schedule_max_epochs": int(args.epochs),
            "precision": "FP32",
            "source": "public-code-derived training profile",
            "charbonnier_constant": 1e-5,
            "scale_weights": [1.0, 1.0, 1.0],
            "official_public_code_used_fixed_100_epochs": True,
            "unified_framework_uses_validation_early_stopping": bool(
                args.early_stopping and args.mode == "full"
            ),
        }
    return {
        "name": "unified",
        "loss_profile": "combined",
        "optimizer": "AdamW",
        "betas": [0.9, 0.999],
        "initial_learning_rate": float(args.learning_rate),
        "final_learning_rate": None,
        "weight_decay": float(args.weight_decay),
        "schedule": "constant",
        "schedule_max_epochs": None,
        "precision": "FP32",
    }


def training_learning_rate(profile: dict[str, object], epoch: int) -> float:
    initial = float(profile["initial_learning_rate"])
    if profile["schedule"] == "constant":
        return initial
    final = float(profile["final_learning_rate"])
    maximum = int(profile["schedule_max_epochs"])
    progress = min(max(epoch - 1, 0), maximum) / maximum
    return final + 0.5 * (initial - final) * (1.0 + np.cos(np.pi * progress))


def audit_command(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).expanduser().resolve()
    report: dict[str, object] = {
        "data_root": str(data_root),
        "contract": {
            "quasi": {
                "obs_h": ["B", 1, 32, 64, 2],
                "target_h": ["B", 1, 256, 64, 2],
            },
            "mobility": {
                "obs_h": ["B", 2, 32, 64, 2],
                "target_h": ["B", 6, 256, 64, 2],
            },
        },
        "assumptions": {
            "semantic_profile": args.semantic_profile,
            "complex_layout": args.complex_layout,
            "obs_ris_index": list(args.obs_ris_indices),
            "mobility_obs_time_index": list(args.mobility_obs_times),
            "pilot_time_note": (
                "The task definition confirms that the two pilot blocks are "
                "the first two blocks of the six-block frame."
            ),
        },
        "files": [],
    }
    for domain in ("quasi", "mobility"):
        for split in ("train", "validation", "test"):
            candidates = dataset_candidates(data_root, domain, split)
            path = next(
                (candidate for candidate in candidates if candidate.is_file()),
                candidates[0],
            )
            entry: dict[str, object] = {
                "domain": domain,
                "split": split,
                "path": str(path),
                "exists": path.is_file(),
                "valid": False,
                "attempted_paths": [str(candidate) for candidate in candidates],
            }
            if path.is_file():
                try:
                    entry["file_size_bytes"] = path.stat().st_size
                    if not h5py.is_hdf5(path):
                        raise ValueError("Not an HDF5/MATLAB v7.3 file")
                    with h5py.File(path, "r") as handle:
                        entry["keys"] = sorted(handle.keys())
                        entry["datasets"] = {
                            key: {
                                "shape": list(handle[key].shape),
                                "dtype": str(handle[key].dtype),
                            }
                            for key in handle.keys()
                            if isinstance(handle[key], h5py.Dataset)
                        }
                    obs_times = (
                        args.mobility_obs_times if domain == "mobility" else (0,)
                    )
                    dataset = LPANH5Dataset(
                        path,
                        domain,
                        split,
                        max_samples=1,
                        obs_time_index=obs_times,
                        obs_ris_index=args.obs_ris_indices,
                        complex_layout=args.complex_layout,
                        semantic_profile=args.semantic_profile,
                    )
                    sample = dataset[0]
                    entry["input_key"] = dataset.input_key
                    entry["target_key"] = dataset.target_key
                    entry["total_samples_in_file"] = dataset.total_samples_in_file
                    dataset.close()
                    if (
                        args.semantic_profile == "official_lpan"
                        and domain == "mobility"
                    ):
                        expected = MOBILITY_EXPECTED_SAMPLES[split]
                        entry["expected_total_samples"] = expected
                        if dataset.total_samples_in_file != expected:
                            raise ValueError(
                                f"Official Mobility {split} split expects "
                                f"{expected} samples, found "
                                f"{dataset.total_samples_in_file}."
                            )
                    entry["unified_sample_shapes"] = {
                        key: list(value.shape)
                        for key, value in sample.items()
                        if key in {"obs_h", "target_h", "observation_mask"}
                    }
                    entry["valid"] = True
                except Exception as exc:
                    entry["error"] = f"{type(exc).__name__}: {exc}"
            report["files"].append(entry)
    output = Path(args.output)
    write_json(output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Audit written to {output}")


def profile_command(args: argparse.Namespace) -> dict[str, object]:
    """Report parameters and operation counts under one shared input contract."""

    if len(set(args.models)) != len(args.models):
        raise ValueError("--models must not contain duplicates.")
    device = device_from(args.device)
    batch = canonical_batch(
        args.domain, batch_size=args.batch_size, device=device
    )
    config = model_config(args, args.domain)
    rows: list[dict[str, object]] = []
    for name in args.models:
        if args.ablation != "none" and name != "phymeta_stgt":
            raise ValueError(
                "--ablation can only be used when profiling phymeta_stgt."
            )
        model = build_model(name, **config).to(device)
        profile = profile_model_complexity(model, batch)
        rows.append(
            {
                "model": name,
                "display_name": MODEL_DISPLAY_NAMES.get(name, name),
                **profile,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "domain": args.domain,
        "device": str(device),
        "shared_conditions": {
            "batch_size": args.batch_size,
            "input_shape": list(batch["obs_h"].shape),
            "dtype": "float32",
            "pass": "forward only",
            "test_data_read": False,
            "mac_flop_conversion": "1 MAC = 2 FLOPs",
            "interpolation_policy": INTERPOLATION_POLICY,
        },
        "results": rows,
    }
    output = Path(args.output)
    write_json(output, report)
    csv_output = output.with_suffix(".csv")
    with csv_output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model",
                "display_name",
                "total_parameters",
                "trainable_parameters",
                "gmacs",
                "gflops",
                "batch_size",
                "input_shape",
                "output_shape",
                "dtype",
            ),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"{'model':28s} {'parameters':>14s} {'GMACs':>12s} {'GFLOPs':>12s}"
    )
    for row in rows:
        print(
            f"{str(row['model']):28s} "
            f"{int(row['total_parameters']):14,d} "
            f"{float(row['gmacs']):12.6f} "
            f"{float(row['gflops']):12.6f}"
        )
    print(f"Complexity report written to {output} and {csv_output}")
    return report


def select_fastest_batch(rows: list[dict[str, object]]) -> dict[str, object]:
    completed = [row for row in rows if row.get("status") == "completed"]
    if not completed:
        raise RuntimeError("No batch-size candidate completed successfully.")
    return max(completed, key=lambda row: float(row["samples_per_second"]))


def benchmark_batch_command(args: argparse.Namespace) -> dict[str, object]:
    """Benchmark training throughput without producing reportable results."""

    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if not args.candidates or any(value <= 0 for value in args.candidates):
        raise ValueError("--candidates must contain positive batch sizes.")
    if len(set(args.candidates)) != len(args.candidates):
        raise ValueError("--candidates must not contain duplicates.")
    device = device_from(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable.")
    obs_times = args.obs_times if args.domain == "mobility" else (0,)
    rows: list[dict[str, object]] = []
    for batch_size in args.candidates:
        seed_everything(args.seed)
        row: dict[str, object] = {"batch_size": batch_size}
        model = optimizer = loader = None
        warmup = warmup_prediction = warmup_loss = None
        try:
            model = build_model(
                "phymeta_stgt",
                domain=args.domain,
                hidden=args.hidden,
                graph_layers=args.graph_layers,
                heads=args.heads,
                dropout=args.dropout,
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            loader = make_loader(
                args.domain,
                "train",
                path=args.train_path,
                data_root=args.data_root,
                batch_size=batch_size,
                workers=args.workers,
                max_samples=args.max_samples,
                seed=args.seed,
                obs_times=obs_times,
                obs_ris_indices=args.obs_ris_indices,
                complex_layout=args.complex_layout,
                semantic_profile=args.semantic_profile,
                shuffle=True,
            )
            # One untimed step initializes CUDA kernels and worker queues.
            warmup = move_batch(next(iter(loader)), device)
            optimizer.zero_grad(set_to_none=True)
            warmup_prediction = model(warmup)
            warmup_loss, _ = combined_loss(
                warmup_prediction, warmup, LossWeights()
            )
            warmup_loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            train_epoch(
                model,
                loader,
                optimizer,
                device,
                LossWeights(),
                grad_clip=args.grad_clip,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            row.update(
                {
                    "status": "completed",
                    "samples": len(loader.dataset),
                    "elapsed_seconds": elapsed,
                    "samples_per_second": len(loader.dataset) / elapsed,
                    "peak_memory_bytes": (
                        int(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda"
                        else None
                    ),
                }
            )
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message:
                status = "oom"
            elif (
                device.type == "cuda"
                and "invalid configuration argument" in message
            ):
                status = "cuda_configuration_error"
            else:
                raise
            row.update(
                {
                    "status": status,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            if loader is not None and hasattr(loader.dataset, "close"):
                loader.dataset.close()
            del warmup, warmup_prediction, warmup_loss
            del loader, optimizer, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows.append(row)
        write_json(Path(args.output), {"results": rows})
    best = select_fastest_batch(rows)
    report = {
        "status": "non_reportable_throughput_benchmark",
        "domain": args.domain,
        "device": str(device),
        "model": "phymeta_stgt",
        "max_samples": args.max_samples,
        "workers": args.workers,
        "test_split_used": False,
        "candidates": list(args.candidates),
        "results": rows,
        "selected_batch_size": int(best["batch_size"]),
        "selection_metric": "maximum training samples per second without OOM",
    }
    write_json(Path(args.output), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def validate_training_request(
    model_name: str,
    adaptation: str,
    pretrained: str | None,
    resume: str | None,
    ablation: str = "none",
) -> None:
    if pretrained and resume:
        raise ValueError("--pretrained and --resume cannot be used together.")
    if pretrained and model_name != "phymeta_stgt":
        raise ValueError(
            "--pretrained transfer is only supported for phymeta_stgt."
        )
    if adaptation != "full" and model_name != "phymeta_stgt":
        raise ValueError(
            "Non-full adaptation is only supported for phymeta_stgt."
        )
    if not pretrained and not resume and adaptation != "full":
        raise ValueError(
            "Non-full adaptation requires a pretrained checkpoint. "
            "Use --adaptation full for scratch training."
        )
    if ablation != "none" and model_name != "phymeta_stgt":
        raise ValueError("Ablations are only supported for phymeta_stgt.")
    if ablation != "none" and adaptation != "full":
        raise ValueError("Ablation runs require --adaptation full.")


def load_pretrained_checkpoint(
    model: torch.nn.Module,
    state: dict[str, object],
    current_config: dict[str, object],
    checkpoint: str | Path,
) -> dict[str, object]:
    if state.get("model_name") != "phymeta_stgt":
        raise ValueError(
            "Transfer learning requires a phymeta_stgt checkpoint, got "
            f"{state.get('model_name')!r}."
        )
    saved_config = state.get("model_config")
    if not isinstance(saved_config, dict):
        raise ValueError("Pretrained checkpoint is missing model_config.")
    architecture_keys = ("hidden", "graph_layers", "heads", "ablation")
    mismatches = {
        key: {
            "checkpoint": saved_config.get(key, "none" if key == "ablation" else None),
            "current": current_config.get(key, "none" if key == "ablation" else None),
        }
        for key in architecture_keys
        if saved_config.get(key, "none" if key == "ablation" else None)
        != current_config.get(key, "none" if key == "ablation" else None)
    }
    if mismatches:
        raise ValueError(
            "Pretrained model architecture does not match:\n"
            + json.dumps(mismatches, indent=2, ensure_ascii=False)
        )
    model_state = state.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError("Pretrained checkpoint is missing model_state.")
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"Pretrained checkpoint weights are incompatible:\n{exc}"
        ) from exc
    source_metadata = state.get("metadata")
    return {
        "path": str(Path(checkpoint).expanduser().resolve()),
        "model_name": state["model_name"],
        "model_config": saved_config,
        "source_domain": (
            source_metadata.get("domain")
            if isinstance(source_metadata, dict)
            else None
        ),
        "strict_load": True,
    }


def train_command(args: argparse.Namespace) -> dict[str, object]:
    validate_training_request(
        args.model,
        args.adaptation,
        args.pretrained,
        args.resume,
        getattr(args, "ablation", "none"),
    )

    domain = args.domain
    obs_times = args.obs_times if domain == "mobility" else (0,)
    smoke = args.mode == "smoke"
    max_train = args.max_train or (64 if smoke else None)
    max_val = args.max_val or (16 if smoke else None)
    epochs = (
        int(getattr(args, "smoke_epochs_override", 1))
        if smoke
        else args.epochs
    )
    stopping = early_stopping_config(args, smoke=smoke)
    train_loader = make_loader(
        domain,
        "train",
        path=args.train_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        max_samples=max_train,
        fraction=args.fraction,
        seed=args.seed,
        obs_times=obs_times,
        obs_ris_indices=args.obs_ris_indices,
        complex_layout=args.complex_layout,
        semantic_profile=args.semantic_profile,
        shuffle=True,
    )
    val_loader = make_loader(
        domain,
        "validation",
        path=args.val_path,
        data_root=args.data_root,
        batch_size=args.eval_batch_size,
        workers=args.workers,
        max_samples=max_val,
        seed=args.seed,
        obs_times=obs_times,
        obs_ris_indices=args.obs_ris_indices,
        complex_layout=args.complex_layout,
        semantic_profile=args.semantic_profile,
    )
    config = model_config(args, domain)
    model = build_model(args.model, **config)
    training_profile = resolve_training_profile(args)
    device = device_from(args.device)
    model.to(device)

    pretrained_metadata = None
    if args.pretrained:
        state = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        pretrained_metadata = load_pretrained_checkpoint(
            model,
            state,
            config,
            args.pretrained,
        )
    adaptation = configure_adaptation(model, args.adaptation)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(training_profile["initial_learning_rate"]),
        betas=tuple(training_profile["betas"]),
        weight_decay=float(training_profile["weight_decay"]),
    )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{domain}_{args.model}_{args.mode}_{stamp}"
    run_dir = (
        Path(args.resume).expanduser().resolve().parent.parent
        if args.resume
        else Path(args.output_root) / run_name
    )
    if args.resume and args.run_name and run_dir.name != args.run_name:
        raise ValueError(
            f"--run-name {args.run_name!r} does not match resumed run "
            f"directory {run_dir.name!r}."
        )
    if run_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Run directory already exists; choose another --run-name: {run_dir}"
        )
    checkpoints = run_dir / "checkpoints"
    results = run_dir / "results"
    checkpoints.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    command = getattr(args, "command_override", shlex.join(sys.argv))
    if not args.resume:
        (run_dir / "command.txt").write_text(command, encoding="utf-8")

    base_weights = LossWeights(
        args.nmse_weight,
        args.char_weight,
        args.obs_weight,
        args.delta_weight if domain == "mobility" else 0.0,
    )
    ablation = getattr(args, "ablation", "none")
    weights = ablated_loss_weights(base_weights, ablation, domain=domain)
    metadata = {
        "domain": domain,
        "model_display_name": MODEL_DISPLAY_NAMES.get(args.model, args.model),
        "mode": args.mode,
        "seed": args.seed,
        "obs_time_index": list(obs_times),
        "obs_ris_index": list(args.obs_ris_indices),
        "complex_layout": args.complex_layout,
        "semantic_profile": args.semantic_profile,
        "train_fraction": args.fraction,
        "adaptation": args.adaptation,
        "ablation": ablation,
        "architectural_ablation": architectural_ablation(ablation),
        "adaptation_parameters": adaptation,
        "pretrained": pretrained_metadata,
        "train_path": str(train_loader.dataset.path),
        "validation_path": str(val_loader.dataset.path),
        "max_train": max_train,
        "max_validation": max_val,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "workers": args.workers,
        "learning_rate": training_profile["initial_learning_rate"],
        "weight_decay": training_profile["weight_decay"],
        "training_profile": training_profile,
        "grad_clip": args.grad_clip,
        "early_stopping": stopping,
        "loss_weights": {
            "nmse": weights.nmse,
            "charbonnier": weights.charbonnier,
            "observation": weights.observation,
            "delta": weights.delta,
        },
    }
    if hasattr(model, "protocol_metadata"):
        metadata["model_protocol"] = model.protocol_metadata()
    start_epoch, best_nmse = 1, float("inf")
    stale_epochs = 0
    history: list[dict[str, object]] = []
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        saved_metadata = state.get("metadata", {})
        saved_semantic_profile = saved_metadata.get("semantic_profile") or (
            infer_semantic_profile(
                str(saved_metadata.get("domain")),
                saved_metadata.get("obs_time_index"),
                saved_metadata.get("obs_ris_index"),
                saved_metadata.get("complex_layout"),
            )
        )
        checks = {
            "model_name": (state.get("model_name"), args.model),
            "model_config": (state.get("model_config"), config),
            "domain": (saved_metadata.get("domain"), metadata["domain"]),
            "mode": (saved_metadata.get("mode"), metadata["mode"]),
            "seed": (saved_metadata.get("seed"), metadata["seed"]),
            "obs_time_index": (
                saved_metadata.get("obs_time_index"),
                metadata["obs_time_index"],
            ),
            "obs_ris_index": (
                saved_metadata.get("obs_ris_index"),
                metadata["obs_ris_index"],
            ),
            "complex_layout": (
                saved_metadata.get("complex_layout"),
                metadata["complex_layout"],
            ),
            "semantic_profile": (
                saved_semantic_profile,
                metadata["semantic_profile"],
            ),
            "train_fraction": (
                saved_metadata.get("train_fraction"),
                metadata["train_fraction"],
            ),
            "adaptation": (
                saved_metadata.get("adaptation"),
                metadata["adaptation"],
            ),
            "ablation": (
                saved_metadata.get("ablation", "none"),
                metadata["ablation"],
            ),
            "train_path": (
                saved_metadata.get("train_path"),
                metadata["train_path"],
            ),
            "validation_path": (
                saved_metadata.get("validation_path"),
                metadata["validation_path"],
            ),
            "max_train": (
                saved_metadata.get("max_train"),
                metadata["max_train"],
            ),
            "max_validation": (
                saved_metadata.get("max_validation"),
                metadata["max_validation"],
            ),
            "batch_size": (
                saved_metadata.get("batch_size"),
                metadata["batch_size"],
            ),
            "eval_batch_size": (
                saved_metadata.get("eval_batch_size"),
                metadata["eval_batch_size"],
            ),
            "workers": (
                saved_metadata.get("workers"),
                metadata["workers"],
            ),
            "learning_rate": (
                saved_metadata.get("learning_rate"),
                metadata["learning_rate"],
            ),
            "weight_decay": (
                saved_metadata.get("weight_decay"),
                metadata["weight_decay"],
            ),
            "training_profile": (
                saved_metadata.get("training_profile"),
                metadata["training_profile"],
            ),
            "grad_clip": (
                saved_metadata.get("grad_clip"),
                metadata["grad_clip"],
            ),
            "early_stopping": (
                saved_metadata.get("early_stopping"),
                metadata["early_stopping"],
            ),
            "loss_weights": (
                saved_metadata.get("loss_weights"),
                metadata["loss_weights"],
            ),
        }
        mismatches = {
            key: {"checkpoint": saved, "current": current}
            for key, (saved, current) in checks.items()
            if saved != current
        }
        if mismatches:
            raise ValueError(
                "Resume configuration does not match checkpoint:\n"
                + json.dumps(mismatches, indent=2, ensure_ascii=False)
            )
        if state.get("rng_state") is None:
            raise ValueError(
                "Checkpoint predates strict RNG-state saving and cannot be "
                "resumed reproducibly."
            )
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["epoch"]) + 1
        if epochs < start_epoch:
            raise ValueError(
                f"--epochs must be at least {start_epoch} when resuming "
                f"checkpoint epoch {state['epoch']}."
            )
        best_nmse = float(state["best_nmse"])
        history = read_history(results / "training_history.csv")
        checkpoint_epoch = int(state["epoch"])
        history_epochs = [int(row["epoch"]) for row in history]
        if not history or history_epochs != sorted(set(history_epochs)):
            raise ValueError(
                "training_history.csv is missing, duplicated, or not ordered."
            )
        if history_epochs[-1] < checkpoint_epoch:
            raise ValueError(
                "training_history.csv does not contain the checkpoint epoch; "
                "training metrics cannot be reconstructed safely."
            )
        if history_epochs[-1] > checkpoint_epoch:
            original_last_epoch = history_epochs[-1]
            history = [
                row for row in history if int(row["epoch"]) <= checkpoint_epoch
            ]
            if not history or int(history[-1]["epoch"]) != checkpoint_epoch:
                raise ValueError(
                    "training_history.csv is ahead of the checkpoint but has "
                    "no row matching the checkpoint epoch."
                )
            write_history(results / "training_history.csv", history)
            with (run_dir / "recovery.log").open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"truncated training_history.csv from epoch "
                    f"{original_last_epoch} to checkpoint epoch "
                    f"{checkpoint_epoch}\n"
                )
        history_best_nmse, stale_epochs = early_stopping_progress(
            history,
            min_epochs=int(stopping["min_epochs"]),
        )
        if not np.isclose(history_best_nmse, best_nmse, rtol=1e-9, atol=1e-12):
            raise ValueError(
                "Checkpoint best_nmse does not match training_history.csv: "
                f"checkpoint={best_nmse}, history={history_best_nmse}."
            )
        restore_rng_state(
            state["rng_state"],
            {
                "train": train_loader.generator,
                "validation": val_loader.generator,
            },
        )
        metadata = saved_metadata
        with (run_dir / "resume_commands.log").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {command}\n")

    stopped_early = False
    stop_reason = None
    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        current_learning_rate = training_learning_rate(training_profile, epoch)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_learning_rate
        train_result = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            weights,
            grad_clip=args.grad_clip,
            loss_profile=str(training_profile["loss_profile"]),
        )
        val_result = evaluate_model(model, val_loader, device)
        val_nmse = float(val_result["nmse_linear"]["overall"])
        best_nmse, stale_epochs, improved, should_stop = early_stopping_step(
            best_nmse=best_nmse,
            stale_epochs=stale_epochs,
            validation_nmse=val_nmse,
            epoch=epoch,
            enabled=bool(stopping["enabled"]),
            min_epochs=int(stopping["min_epochs"]),
            patience=int(stopping["patience"]),
        )
        row = {
            "epoch": epoch,
            "train_total": train_result["total"],
            "train_nmse": train_result.get("nmse"),
            "learning_rate": current_learning_rate,
            "validation_nmse_linear": val_nmse,
            "validation_nmse_db": nmse_db_from_result(val_result),
            "improved": improved,
            "early_stopping_stale_epochs": stale_epochs,
            "epoch_seconds": time.perf_counter() - started,
        }
        for key, value in train_result.items():
            if key not in {"total", "nmse"}:
                row[f"train_{key}"] = value
        history.append(row)
        write_history(results / "training_history.csv", history)
        rng_state = capture_rng_state(
            {
                "train": train_loader.generator,
                "validation": val_loader.generator,
            }
        )
        save_checkpoint(
            checkpoints / "last_checkpoint.pth",
            model,
            optimizer,
            epoch=epoch,
            best_nmse=best_nmse,
            model_name=args.model,
            model_config=config,
            metadata=metadata,
            rng_state=rng_state,
        )
        if improved:
            save_checkpoint(
                checkpoints / "best_checkpoint.pth",
                model,
                optimizer,
                epoch=epoch,
                best_nmse=best_nmse,
                model_name=args.model,
                model_config=config,
                metadata=metadata,
                rng_state=rng_state,
            )
        print(
            f"epoch={epoch} train={train_result['total']:.6g} "
            f"val={row['validation_nmse_db']:.4f} dB"
        )
        if should_stop:
            stopped_early = True
            stop_reason = (
                f"validation NMSE did not improve for {stale_epochs} epochs "
                f"after min_epochs={stopping['min_epochs']}"
            )
            print(f"Early stopping at epoch {epoch}: {stop_reason}")
            break
    complexity = profile_model_complexity(
        model, canonical_batch(domain, batch_size=1, device=device)
    )
    final = {
        "status": "smoke_test" if smoke else "validation",
        "best_validation_nmse_linear": best_nmse,
        "best_validation_nmse_db": 10 * torch.log10(torch.tensor(best_nmse)).item(),
        "epochs_completed": int(history[-1]["epoch"]) if history else 0,
        "max_epochs": epochs,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "history": history,
        "metadata": metadata,
        "parameters": model_parameter_report(model),
        "complexity": complexity,
    }
    write_json(results / "final_result.json", final)
    print(f"Run completed: {run_dir}")
    return {"run_dir": str(run_dir), "result": final}


def _write_study_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _study_trial_args(
    args: argparse.Namespace,
    study_dir: Path,
    run_name: str,
    **overrides: object,
) -> argparse.Namespace:
    values = dict(vars(args))
    values.update(
        {
            "model": "phymeta_stgt",
            "adaptation": "full",
            "pretrained": None,
            "resume": None,
            "run_name": run_name,
            "output_root": str(study_dir / "trials"),
            "ablation": "none",
        }
    )
    values.update(overrides)
    values["command_override"] = json.dumps(
        {
            "study": str(study_dir),
            "trial": run_name,
            "configuration": overrides,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return argparse.Namespace(**values)


def _prepare_resumable_study(
    study_dir: Path,
    plan_name: str,
    plan: dict[str, object],
) -> None:
    """Create a study or validate the plan of an interrupted study."""
    normalized_plan = json.loads(
        json.dumps(
            plan,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    )
    plan_path = study_dir / plan_name

    if study_dir.exists():
        if not plan_path.is_file():
            raise RuntimeError(
                f"Existing study directory has no {plan_name}: "
                f"{study_dir}"
            )

        existing = json.loads(
            plan_path.read_text(encoding="utf-8")
        )

        if existing != normalized_plan:
            raise RuntimeError(
                "Refusing to resume a study whose recorded plan "
                f"differs from the requested plan: {study_dir}"
            )
        return

    study_dir.mkdir(parents=True)
    write_json(plan_path, normalized_plan)


def _run_or_resume_study_trial(
    args: argparse.Namespace,
    study_dir: Path,
    run_name: str,
    **overrides: object,
) -> dict[str, object]:
    """Reuse a completed trial or resume its exact last checkpoint."""

    trial_args = _study_trial_args(
        args,
        study_dir,
        run_name,
        **overrides,
    )

    run_dir = Path(str(trial_args.output_root)) / run_name
    result_path = run_dir / "results" / "final_result.json"

    expected_epochs = (
        int(getattr(trial_args, "smoke_epochs_override", 1))
        if trial_args.mode == "smoke"
        else int(trial_args.epochs)
    )

    if result_path.is_file():
        result = json.loads(
            result_path.read_text(encoding="utf-8")
        )

        if int(result.get("max_epochs", -1)) == expected_epochs:
            print(f"[reuse] completed trial {run_name}")
            return {
                "run_dir": str(run_dir),
                "result": result,
                "recovery": "reused_completed",
            }

    last_checkpoint = (
        run_dir / "checkpoints" / "last_checkpoint.pth"
    )

    if last_checkpoint.is_file():
        print(
            f"[resume] trial {run_name} from "
            f"{last_checkpoint}"
        )
        trial_args.resume = str(last_checkpoint)
        outcome = train_command(trial_args)
        outcome["recovery"] = "resumed_checkpoint"
        return outcome

    if run_dir.exists():
        orphan = run_dir.with_name(
            f"{run_dir.name}.interrupted_no_checkpoint_"
            f"{time.strftime('%Y%m%d_%H%M%S')}"
        )
        run_dir.rename(orphan)

        print(
            f"[restart] preserved non-resumable partial run "
            f"as {orphan}"
        )

    outcome = train_command(trial_args)
    outcome["recovery"] = "started_from_scratch"
    return outcome


def two_round_tune_command(args: argparse.Namespace) -> dict[str, object]:
    """Run validation-only successive-halving style hyperparameter search."""

    candidates = hyperparameter_candidates(
        hidden_values=args.hidden_values,
        graph_layer_values=args.graph_layer_values,
        head_values=args.head_values,
        dropout_values=args.dropout_values,
        learning_rate_values=args.learning_rate_values,
        weight_decay_values=args.weight_decay_values,
        strategy=args.strategy,
        max_trials=args.max_trials,
        seed=args.search_seed,
    )
    smoke = args.mode == "smoke"
    round1_epochs = 1 if smoke else int(args.round1_epochs)
    final_epochs = 1 if smoke else int(args.epochs)
    promote_top_k = 0 if smoke else int(args.promote_top_k)
    if not smoke:
        if round1_epochs <= 0 or round1_epochs >= final_epochs:
            raise ValueError(
                "Full multi-fidelity search requires 0 < --round1-epochs "
                "< --epochs."
            )
        if promote_top_k <= 0 or promote_top_k > len(candidates):
            raise ValueError(
                "--promote-top-k must be positive and no larger than the "
                "number of candidates."
            )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    study_name = args.study_name or f"tune_{args.domain}_{stamp}"
    study_dir = Path(args.output_root) / study_name
    if study_dir.exists():
        raise FileExistsError(
            f"Study directory already exists; choose another --study-name: {study_dir}"
        )
    study_dir.mkdir(parents=True)
    write_json(
        study_dir / "search_plan.json",
        {
            "domain": args.domain,
            "model": "phymeta_stgt",
            "selection_metric": "minimum validation sample-level linear NMSE",
            "test_split_used": False,
            "strategy": args.strategy,
            "search_seed": args.search_seed,
            "training_seed_shared_by_all_trials": args.seed,
            "protocol": {
                "name": "two_round_validation_promotion",
                "round1_epochs": round1_epochs,
                "promote_top_k": promote_top_k,
                "final_max_epochs": final_epochs,
                "promotion_checkpoint": "epoch round1 last_checkpoint.pth",
                "resume_preserves_rng": True,
                "early_stopping": {
                    "enabled": bool(args.early_stopping and not smoke),
                    "min_epochs": args.min_epochs,
                    "patience": args.patience,
                },
            },
            "candidates": candidates,
        },
    )
    rows: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        run_name = f"trial_{index:03d}"
        seed_everything(args.seed)
        trial_args = _study_trial_args(
            args,
            study_dir,
            run_name,
            epochs=round1_epochs,
            **candidate,
        )
        row: dict[str, object] = {
            "trial": index,
            "run_name": run_name,
            **candidate,
        }
        try:
            outcome = train_command(trial_args)
            result = outcome["result"]
            assert isinstance(result, dict)
            row.update(
                {
                    "status": "round1_completed",
                    "round1_epochs_completed": result["epochs_completed"],
                    "round1_best_validation_nmse_linear": result[
                        "best_validation_nmse_linear"
                    ],
                    "round1_best_validation_nmse_db": result[
                        "best_validation_nmse_db"
                    ],
                    "run_dir": outcome["run_dir"],
                    "promoted": False,
                }
            )
        except Exception as exc:
            row.update(
                {
                    "status": "round1_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        rows.append(row)
        _write_study_rows(study_dir / "trials.csv", rows)
        write_json(study_dir / "trials.json", rows)
        if row["status"] == "round1_failed" and not smoke:
            raise RuntimeError(
                f"Round-1 trial {index} failed; inspect "
                f"{study_dir / 'trials.json'}."
            )
    round1_completed = [
        row for row in rows if row["status"] == "round1_completed"
    ]
    if not round1_completed:
        raise RuntimeError(
            f"All hyperparameter trials failed; inspect {study_dir / 'trials.json'}."
        )
    round1_ranking = sorted(
        round1_completed,
        key=lambda row: float(row["round1_best_validation_nmse_linear"]),
    )

    if smoke:
        final_ranking = round1_ranking
        for row in final_ranking:
            row["best_validation_nmse_linear"] = row[
                "round1_best_validation_nmse_linear"
            ]
            row["best_validation_nmse_db"] = row[
                "round1_best_validation_nmse_db"
            ]
    else:
        promoted = round1_ranking[:promote_top_k]
        for promotion_rank, row in enumerate(promoted, start=1):
            row["promoted"] = True
            row["promotion_rank"] = promotion_rank
            run_dir = Path(str(row["run_dir"]))
            resume_path = run_dir / "checkpoints" / "last_checkpoint.pth"
            if not resume_path.is_file():
                raise FileNotFoundError(
                    f"Promotion checkpoint is missing: {resume_path}"
                )
            candidate = {
                key: row[key]
                for key in (
                    "hidden",
                    "graph_layers",
                    "heads",
                    "dropout",
                    "learning_rate",
                    "weight_decay",
                )
            }
            seed_everything(args.seed)
            promoted_args = _study_trial_args(
                args,
                study_dir,
                str(row["run_name"]),
                epochs=final_epochs,
                resume=str(resume_path),
                **candidate,
            )
            try:
                outcome = train_command(promoted_args)
                result = outcome["result"]
                assert isinstance(result, dict)
                row.update(
                    {
                        "status": "promoted_completed",
                        "final_epochs_completed": result["epochs_completed"],
                        "stopped_early": result["stopped_early"],
                        "best_validation_nmse_linear": result[
                            "best_validation_nmse_linear"
                        ],
                        "best_validation_nmse_db": result[
                            "best_validation_nmse_db"
                        ],
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "status": "promotion_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
            _write_study_rows(study_dir / "trials.csv", rows)
            write_json(study_dir / "trials.json", rows)
            if row["status"] == "promotion_failed":
                raise RuntimeError(
                    f"Promotion for trial {row['trial']} failed; inspect "
                    f"{study_dir / 'trials.json'}."
                )
        final_ranking = sorted(
            (row for row in promoted if row["status"] == "promoted_completed"),
            key=lambda row: float(row["best_validation_nmse_linear"]),
        )
    best = final_ranking[0]
    summary = {
        "status": "smoke_search" if args.mode == "smoke" else "validation_search",
        "domain": args.domain,
        "study_dir": str(study_dir),
        "selection_metric": "minimum validation sample-level linear NMSE",
        "test_split_used": False,
        "protocol": "round1_all_candidates_then_top_k_resume",
        "round1_epochs": round1_epochs,
        "final_max_epochs": final_epochs,
        "promote_top_k": promote_top_k,
        "round1_completed_trials": len(round1_completed),
        "promoted_completed_trials": (
            0 if smoke else len(final_ranking)
        ),
        "failed_trials": len(
            [row for row in rows if str(row["status"]).endswith("failed")]
        ),
        "best_trial": best,
        "best_hyperparameters": {
            key: best[key]
            for key in (
                "hidden",
                "graph_layers",
                "heads",
                "dropout",
                "learning_rate",
                "weight_decay",
            )
        },
        "best_checkpoint": str(
            Path(str(best["run_dir"])) / "checkpoints" / "best_checkpoint.pth"
        ),
        "round1_ranking": [int(row["trial"]) for row in round1_ranking],
        "ranking": final_ranking,
    }
    write_json(study_dir / "best_result.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def late_window_nmse_metrics(
    history: list[dict[str, object]], start_epoch: int, end_epoch: int
) -> dict[str, float | int | list[int]]:
    """Summarize an inclusive epoch window in linear NMSE space."""
    selected = [
        row
        for row in history
        if start_epoch <= int(row["epoch"]) <= end_epoch
    ]
    expected = end_epoch - start_epoch + 1
    if len(selected) != expected:
        raise ValueError(
            f"Expected {expected} validation rows for epochs "
            f"{start_epoch}..{end_epoch}, found {len(selected)}."
        )
    values = np.asarray(
        [float(row["validation_nmse_linear"]) for row in selected],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise FloatingPointError("Late-window validation NMSE contains NaN/Inf.")
    return {
        "window_epochs": [start_epoch, end_epoch],
        "sample_count": int(values.size),
        "median_validation_nmse_linear": float(np.median(values)),
        "mean_validation_nmse_linear": float(np.mean(values)),
        "std_validation_nmse_linear": float(np.std(values, ddof=0)),
        "best_validation_nmse_linear": float(np.min(values)),
    }


def targeted_boundary_search_plan(args: argparse.Namespace) -> dict[str, object]:
    """Return the deterministic, validation-only targeted search plan."""
    smoke = args.mode == "smoke"
    capacity_epochs = 1 if smoke else 20
    learning_rate_epochs = 1 if smoke else 40
    final_epochs = 2 if smoke else 100
    return {
        "domain": args.domain,
        "model": "phymeta_stgt",
        "tuning_protocol": "targeted_boundary",
        "test_split_used": False,
        "training_seed_shared_by_all_trials": args.seed,
        "selection_space": "linear NMSE",
        "capacity": {
            "candidates": [96, 128, 160],
            "fixed": {
                "graph_layers": 2,
                "heads": 4,
                "dropout": 0.0,
                "learning_rate": 5e-4,
                "weight_decay": 0.0,
            },
            "budget_epochs": capacity_epochs,
            "late_window_epochs": (
                [1, 1] if smoke else [16, 20]
            ),
            "early_stopping_can_trigger": False,
            "primary_metric": "late_window_median_validation_nmse_linear",
            "secondary_metric": "best_validation_nmse_linear_up_to_budget",
        },
        "learning_rate": {
            "candidates": [5e-4, 8e-4, 1e-3],
            "fixed": {
                "graph_layers": 2,
                "heads": 4,
                "dropout": 0.0,
                "weight_decay": 0.0,
            },
            "from_scratch": True,
            "resume_from_capacity": False,
            "budget_epochs": learning_rate_epochs,
            "late_window_epochs": (
                [1, 1] if smoke else [31, 40]
            ),
            "early_stopping_can_trigger": False,
            "primary_metric": "late_window_median_validation_nmse_linear",
            "secondary_metric": "best_validation_nmse_linear_up_to_budget",
        },
        "final": {
            "max_epochs": final_epochs,
            "resume_checkpoint": "winning learning-rate trial last_checkpoint.pth",
            "resume_preserves_optimizer_rng_loader_and_history": True,
            "early_stopping": {
                "enabled": not smoke,
                "min_epochs": 40,
                "patience": 15,
            },
        },
    }


def targeted_boundary_tune_command(args: argparse.Namespace) -> dict[str, object]:
    """Run sequential capacity, learning-rate, and exact-resume tuning."""
    plan = targeted_boundary_search_plan(args)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    study_name = args.study_name or f"targeted_{args.domain}_{stamp}"
    study_dir = Path(args.output_root) / study_name
    _prepare_resumable_study(
        study_dir,
        "search_plan.json",
        plan,
    )
    capacity_dir = study_dir / "capacity"
    learning_rate_dir = study_dir / "learning_rate"
    capacity_dir.mkdir(parents=True, exist_ok=True)
    learning_rate_dir.mkdir(parents=True, exist_ok=True)
    study_started = time.perf_counter()

    capacity_plan = plan["capacity"]
    assert isinstance(capacity_plan, dict)
    capacity_fixed = capacity_plan["fixed"]
    assert isinstance(capacity_fixed, dict)
    capacity_budget = int(capacity_plan["budget_epochs"])
    capacity_window = capacity_plan["late_window_epochs"]
    assert isinstance(capacity_window, list)
    capacity_rows: list[dict[str, object]] = []
    for hidden in capacity_plan["candidates"]:
        run_name = f"trial_hidden_{int(hidden):03d}"
        trial_started = time.perf_counter()
        seed_everything(args.seed)
        outcome = _run_or_resume_study_trial(
            args,
            capacity_dir,
            run_name,
            epochs=capacity_budget,
            smoke_epochs_override=capacity_budget,
            early_stopping=True,
            min_epochs=40,
            patience=15,
            hidden=int(hidden),
            **capacity_fixed,
        )
        result = outcome["result"]
        assert isinstance(result, dict)
        history = result["history"]
        assert isinstance(history, list)
        window = late_window_nmse_metrics(
            history, int(capacity_window[0]), int(capacity_window[1])
        )
        row = {
            "hidden": int(hidden),
            **capacity_fixed,
            "status": "completed",
            "from_scratch": True,
            "resume": None,
            "recovery": outcome.get("recovery", "unknown"),
            "epochs_completed": result["epochs_completed"],
            "best_validation_nmse_linear_up_to_budget": result[
                "best_validation_nmse_linear"
            ],
            "late_window": window,
            "late_window_median_validation_nmse_linear": window[
                "median_validation_nmse_linear"
            ],
            "run_dir": outcome["run_dir"],
            "wall_clock_seconds": time.perf_counter() - trial_started,
        }
        capacity_rows.append(row)
        _write_study_rows(capacity_dir / "trials.csv", capacity_rows)
        write_json(capacity_dir / "trials.json", capacity_rows)
    capacity_ranking = sorted(
        capacity_rows,
        key=lambda row: (
            float(row["late_window_median_validation_nmse_linear"]),
            float(row["best_validation_nmse_linear_up_to_budget"]),
        ),
    )
    selected_hidden = int(capacity_ranking[0]["hidden"])
    write_json(capacity_dir / "capacity_ranking.json", capacity_ranking)

    lr_plan = plan["learning_rate"]
    assert isinstance(lr_plan, dict)
    lr_fixed = lr_plan["fixed"]
    assert isinstance(lr_fixed, dict)
    lr_budget = int(lr_plan["budget_epochs"])
    lr_window = lr_plan["late_window_epochs"]
    assert isinstance(lr_window, list)
    lr_rows: list[dict[str, object]] = []
    for learning_rate in lr_plan["candidates"]:
        label = f"{float(learning_rate):.0e}".replace("-0", "-")
        run_name = f"trial_lr_{label}"
        trial_started = time.perf_counter()
        seed_everything(args.seed)
        outcome = _run_or_resume_study_trial(
            args,
            learning_rate_dir,
            run_name,
            epochs=lr_budget,
            smoke_epochs_override=lr_budget,
            early_stopping=True,
            min_epochs=40,
            patience=15,
            hidden=selected_hidden,
            learning_rate=float(learning_rate),
            **lr_fixed,
        )
        result = outcome["result"]
        assert isinstance(result, dict)
        history = result["history"]
        assert isinstance(history, list)
        window = late_window_nmse_metrics(
            history, int(lr_window[0]), int(lr_window[1])
        )
        row = {
            "hidden": selected_hidden,
            "learning_rate": float(learning_rate),
            **lr_fixed,
            "status": "completed",
            "from_scratch": True,
            "resume": None,
            "recovery": outcome.get("recovery", "unknown"),
            "epochs_completed": result["epochs_completed"],
            "best_validation_nmse_linear_up_to_budget": result[
                "best_validation_nmse_linear"
            ],
            "late_window": window,
            "late_window_median_validation_nmse_linear": window[
                "median_validation_nmse_linear"
            ],
            "late_window_mean_validation_nmse_linear": window[
                "mean_validation_nmse_linear"
            ],
            "late_window_std_validation_nmse_linear": window[
                "std_validation_nmse_linear"
            ],
            "run_dir": outcome["run_dir"],
            "wall_clock_seconds": time.perf_counter() - trial_started,
        }
        lr_rows.append(row)
        _write_study_rows(learning_rate_dir / "trials.csv", lr_rows)
        write_json(learning_rate_dir / "trials.json", lr_rows)
    lr_ranking = sorted(
        lr_rows,
        key=lambda row: (
            float(row["late_window_median_validation_nmse_linear"]),
            float(row["best_validation_nmse_linear_up_to_budget"]),
        ),
    )
    write_json(learning_rate_dir / "lr_ranking.json", lr_ranking)
    winner = lr_ranking[0]

    source_run = Path(str(winner["run_dir"]))
    source_last = source_run / "checkpoints" / "last_checkpoint.pth"
    if not source_last.is_file():
        raise FileNotFoundError(f"Winning last checkpoint is missing: {source_last}")
    final_dir = study_dir / "final"

    if not final_dir.exists():
        shutil.copytree(source_run, final_dir)

    final_last = final_dir / "checkpoints" / "last_checkpoint.pth"
    final_plan = plan["final"]
    assert isinstance(final_plan, dict)
    final_max_epochs = int(final_plan["max_epochs"])

    final_result_path = (
        final_dir / "results" / "final_result.json"
    )
    final_result: dict[str, object] | None = None

    if final_result_path.is_file():
        candidate = json.loads(
            final_result_path.read_text(encoding="utf-8")
        )
        if int(candidate.get("max_epochs", -1)) == final_max_epochs:
            print("[reuse] completed targeted final continuation")
            final_result = candidate

    if final_result is None:
        if not final_last.is_file():
            raise FileNotFoundError(
                f"Final resume checkpoint is missing: {final_last}"
            )

        print(
            "[resume] targeted final continuation from "
            f"{final_last}"
        )

        seed_everything(args.seed)

        final_args = _study_trial_args(
            args,
            study_dir,
            "final",
            epochs=final_max_epochs,
            smoke_epochs_override=final_max_epochs,
            resume=str(final_last),
            early_stopping=True,
            min_epochs=40,
            patience=15,
            hidden=selected_hidden,
            graph_layers=2,
            heads=4,
            dropout=0.0,
            learning_rate=float(winner["learning_rate"]),
            weight_decay=0.0,
        )

        final_outcome = train_command(final_args)
        final_result = final_outcome["result"]
        assert isinstance(final_result, dict)
    boundary_hit = {
        "hidden_upper": selected_hidden == 160,
        "learning_rate_upper": float(winner["learning_rate"]) == 1e-3,
    }
    summary = {
        "status": "smoke_search" if args.mode == "smoke" else "validation_search",
        "domain": args.domain,
        "study_dir": str(study_dir),
        "tuning_protocol": "targeted_boundary",
        "test_split_used": False,
        "selection_metric": (
            "late-window median validation sample-level linear NMSE; "
            "best validation linear NMSE tie-break"
        ),
        "capacity_budget_epochs": capacity_budget,
        "lr_budget_epochs": lr_budget,
        "final_max_epochs": final_max_epochs,
        "selected_hidden": selected_hidden,
        "selected_learning_rate": float(winner["learning_rate"]),
        "best_hyperparameters": {
            "hidden": selected_hidden,
            "graph_layers": 2,
            "heads": 4,
            "dropout": 0.0,
            "learning_rate": float(winner["learning_rate"]),
            "weight_decay": 0.0,
        },
        "best_checkpoint": str(
            final_dir / "checkpoints" / "best_checkpoint.pth"
        ),
        "phase_c_resume_checkpoint": str(final_last),
        "phase_c_resume_source_epoch": lr_budget,
        "capacity_ranking": capacity_ranking,
        "learning_rate_ranking": lr_ranking,
        "final_result": final_result,
        "boundary_hit": boundary_hit,
        "boundary_recommendation": (
            "Consider one additional boundary point in a separate study."
            if any(boundary_hit.values())
            else None
        ),
        "wall_clock_seconds": time.perf_counter() - study_started,
    }
    write_json(study_dir / "best_result.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def tune_command(args: argparse.Namespace) -> dict[str, object]:
    if args.tuning_protocol == "targeted_boundary":
        return targeted_boundary_tune_command(args)
    return two_round_tune_command(args)


ABLATION_HYPERPARAMETER_KEYS = (
    "hidden",
    "graph_layers",
    "heads",
    "dropout",
    "learning_rate",
    "weight_decay",
)


def _load_best_hyperparameters(
    best_result_path: str | Path,
    *,
    expected_domain: str,
    require_full_search: bool,
) -> dict[str, int | float]:
    """Load and validate the Stage-B configuration used by an ablation."""

    path = Path(best_result_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if require_full_search and payload.get("status") != "validation_search":
        raise ValueError(
            "Full ablation requires best_result.json from a full "
            "validation_search, not a smoke search."
        )
    recorded_domain = payload.get("domain")
    search_plan = path.with_name("search_plan.json")
    if recorded_domain is None and search_plan.is_file():
        plan = json.loads(search_plan.read_text(encoding="utf-8"))
        recorded_domain = plan.get("domain")
    if recorded_domain is not None and recorded_domain != expected_domain:
        raise ValueError(
            f"Best-result domain {recorded_domain!r} does not match "
            f"ablation domain {expected_domain!r}."
        )
    raw = payload.get("best_hyperparameters")
    if not isinstance(raw, dict):
        raw = payload.get("best_trial")
    if not isinstance(raw, dict):
        raise ValueError("best_result.json has no best hyperparameter mapping.")
    missing = [key for key in ABLATION_HYPERPARAMETER_KEYS if key not in raw]
    if missing:
        raise ValueError(
            f"best_result.json is missing hyperparameters: {missing}."
        )
    config: dict[str, int | float] = {
        "hidden": int(raw["hidden"]),
        "graph_layers": int(raw["graph_layers"]),
        "heads": int(raw["heads"]),
        "dropout": float(raw["dropout"]),
        "learning_rate": float(raw["learning_rate"]),
        "weight_decay": float(raw["weight_decay"]),
    }
    if int(config["hidden"]) <= 0 or int(config["heads"]) <= 0:
        raise ValueError("Best-result hidden and heads must be positive.")
    if int(config["hidden"]) % int(config["heads"]):
        raise ValueError("Best-result hidden must be divisible by heads.")
    if int(config["graph_layers"]) < 0:
        raise ValueError("Best-result graph_layers must be non-negative.")
    if not 0.0 <= float(config["dropout"]) < 1.0:
        raise ValueError("Best-result dropout must be in [0, 1).")
    if float(config["learning_rate"]) <= 0.0:
        raise ValueError("Best-result learning_rate must be positive.")
    if float(config["weight_decay"]) < 0.0:
        raise ValueError("Best-result weight_decay must be non-negative.")
    return config


def early_stopping_progress(
    history: list[dict[str, object]],
    *,
    min_epochs: int,
) -> tuple[float, int]:
    """Reconstruct best NMSE and post-warmup stale epochs from history."""

    best_nmse = float("inf")
    stale_epochs = 0
    for row in history:
        epoch = int(row["epoch"])
        value = float(row["validation_nmse_linear"])
        if value < best_nmse:
            best_nmse = value
            stale_epochs = 0
        elif epoch > min_epochs:
            stale_epochs += 1
        else:
            # Patience starts only after the required minimum training budget.
            stale_epochs = 0
    return best_nmse, stale_epochs


def early_stopping_step(
    *,
    best_nmse: float,
    stale_epochs: int,
    validation_nmse: float,
    epoch: int,
    enabled: bool,
    min_epochs: int,
    patience: int,
) -> tuple[float, int, bool, bool]:
    improved = validation_nmse < best_nmse
    if improved:
        best_nmse = validation_nmse
        stale_epochs = 0
    elif epoch > min_epochs:
        stale_epochs += 1
    else:
        stale_epochs = 0
    should_stop = (
        enabled and epoch > min_epochs and stale_epochs >= patience
    )
    return best_nmse, stale_epochs, improved, should_stop


def early_stopping_config(
    args: argparse.Namespace,
    *,
    smoke: bool,
) -> dict[str, object]:
    min_epochs = int(args.min_epochs)
    patience = int(args.patience)
    if min_epochs < 0:
        raise ValueError("--min-epochs must be non-negative.")
    if patience <= 0:
        raise ValueError("--patience must be positive.")
    return {
        "enabled": bool(args.early_stopping and not smoke),
        "min_epochs": min_epochs,
        "patience": patience,
        "selection_metric": "validation sample-level linear NMSE",
        "test_split_used": False,
    }


def ablate_command(args: argparse.Namespace) -> dict[str, object]:
    """Run controlled one-factor ablations under a shared training protocol."""

    if args.mode == "full" and args.best_result is None:
        raise ValueError(
            "Full ablation requires --best-result from the matching Stage-B "
            "validation search so every variant inherits its hyperparameters."
        )
    inherited_hyperparameters = (
        _load_best_hyperparameters(
            args.best_result,
            expected_domain=args.domain,
            require_full_search=args.mode == "full",
        )
        if args.best_result is not None
        else {
            "hidden": args.hidden,
            "graph_layers": args.graph_layers,
            "heads": args.heads,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        }
    )
    variants = args.variants or ABLATION_VARIANTS
    if len(set(variants)) != len(variants):
        raise ValueError("--variants must not contain duplicates.")
    unknown = sorted(set(variants) - set(ABLATION_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown ablation variants: {unknown}.")
    if "none" not in variants:
        variants = ("none",) + tuple(variants)
    if args.domain == "quasi":
        variants = tuple(
            variant
            for variant in variants
            if variant != "no_temporal_delta_loss"
        )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    study_name = args.study_name or f"ablation_{args.domain}_{stamp}"
    study_dir = Path(args.output_root) / study_name

    ablation_plan = {
        "domain": args.domain,
        "mode": args.mode,
        "seed": args.seed,
        "variants": list(variants),
        "epochs": 1 if args.mode == "smoke" else args.epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "early_stopping": bool(args.early_stopping),
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "fraction": args.fraction,
        "best_result": (
            str(Path(args.best_result).resolve())
            if args.best_result is not None
            else None
        ),
        "inherited_hyperparameters": inherited_hyperparameters,
    }

    _prepare_resumable_study(
        study_dir,
        "ablation_plan.json",
        ablation_plan,
    )

    rows: list[dict[str, object]] = []
    for index, variant in enumerate(variants):
        seed_everything(args.seed)
        run_name = f"ablation_{index:02d}_{variant}"
        row: dict[str, object] = {
            "order": index,
            "variant": variant,
            **ablation_metadata(variant),
            "run_name": run_name,
        }
        try:
            outcome = _run_or_resume_study_trial(
                args,
                study_dir,
                run_name,
                **inherited_hyperparameters,
                ablation=variant,
            )
            result = outcome["result"]
            assert isinstance(result, dict)
            row.update(
                {
                    "status": "completed",
                    "best_validation_nmse_linear": result[
                        "best_validation_nmse_linear"
                    ],
                    "best_validation_nmse_db": result[
                        "best_validation_nmse_db"
                    ],
                    "run_dir": outcome["run_dir"],
                }
            )
        except Exception as exc:
            row.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        rows.append(row)
        _write_study_rows(study_dir / "ablation_results.csv", rows)
        write_json(study_dir / "ablation_results.json", rows)
    completed = [row for row in rows if row["status"] == "completed"]
    reference = next(
        (row for row in completed if row["variant"] == "none"), None
    )
    if reference is None:
        raise RuntimeError(
            f"The full-model reference failed; inspect {study_dir / 'ablation_results.json'}."
        )
    reference_db = float(reference["best_validation_nmse_db"])
    for row in completed:
        row["delta_vs_full_db"] = (
            float(row["best_validation_nmse_db"]) - reference_db
        )
    summary = {
        "status": "smoke_ablation" if args.mode == "smoke" else "validation_ablation",
        "study_dir": str(study_dir),
        "selection_metric": "best validation sample-level linear NMSE per run",
        "test_split_used": False,
        "control": {
            "shared_seed": args.seed,
            "shared_data_fraction": args.fraction,
            "shared_epochs": 1 if args.mode == "smoke" else args.epochs,
            "stage_b_best_result": (
                str(Path(args.best_result).resolve())
                if args.best_result is not None
                else None
            ),
            "shared_hyperparameters": inherited_hyperparameters,
            "one_factor_changed_per_variant": True,
        },
        "full_model": reference,
        "results": completed,
        "failed": [row for row in rows if row["status"] == "failed"],
    }
    _write_study_rows(study_dir / "ablation_results.csv", rows)
    write_json(study_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def load_checkpoint_model(
    checkpoint: str | Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, object]]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(state["model_name"], **state["model_config"])
    model.load_state_dict(state["model_state"])
    model.to(device)
    return model, state


def resolve_evaluation_semantics(
    args: argparse.Namespace,
    state: dict[str, object],
) -> tuple[str, tuple[int, ...], tuple[int, ...], str, str]:
    raw_metadata = state.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    saved_domain = metadata.get("domain")
    if saved_domain in {"quasi", "mobility"}:
        if args.domain is not None and args.domain != saved_domain:
            raise ValueError(
                f"Evaluation domain {args.domain!r} does not match checkpoint "
                f"domain {saved_domain!r}."
            )
        domain = args.domain or saved_domain
    else:
        domain = args.domain
    if domain not in {"quasi", "mobility"}:
        raise ValueError("Specify --domain because checkpoint metadata is missing.")

    mismatches: dict[str, dict[str, object]] = {}

    def resolve(
        key: str,
        requested: object,
        default: object,
        *,
        sequence: bool = False,
    ) -> object:
        saved = metadata.get(key)
        normalized_saved = tuple(saved) if sequence and saved is not None else saved
        normalized_requested = (
            tuple(requested)
            if sequence and requested is not None
            else requested
        )
        if (
            normalized_requested is not None
            and normalized_saved is not None
            and normalized_requested != normalized_saved
        ):
            mismatches[key] = {
                "checkpoint": normalized_saved,
                "evaluation": normalized_requested,
            }
        if normalized_requested is not None:
            return normalized_requested
        if normalized_saved is not None:
            return normalized_saved
        return default

    obs_times = resolve(
        "obs_time_index",
        args.obs_times,
        (0, 1) if domain == "mobility" else (0,),
        sequence=True,
    )
    obs_ris_indices = resolve(
        "obs_ris_index",
        args.obs_ris_indices,
        tuple(range(0, 256, 8)),
        sequence=True,
    )
    complex_layout = resolve(
        "complex_layout",
        args.complex_layout,
        "grouped",
    )
    legacy_profile = infer_semantic_profile(
        domain,
        obs_times,
        obs_ris_indices,
        complex_layout,
    )
    semantic_profile = resolve(
        "semantic_profile",
        args.semantic_profile,
        legacy_profile,
    )
    if mismatches and not args.allow_semantic_override:
        raise ValueError(
            "Evaluation data semantics do not match checkpoint:\n"
            + json.dumps(mismatches, indent=2, ensure_ascii=False)
            + "\nUse --allow-semantic-override only for an intentional override."
        )
    return (
        domain,
        tuple(obs_times),
        tuple(obs_ris_indices),
        str(complex_layout),
        str(semantic_profile),
    )


def evaluate_command(args: argparse.Namespace) -> None:
    device = device_from(args.device)
    model, state = load_checkpoint_model(args.checkpoint, device)
    domain, obs_times, obs_ris_indices, complex_layout, semantic_profile = (
        resolve_evaluation_semantics(args, state)
    )
    loader = make_loader(
        domain,
        args.split,
        path=args.data_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        max_samples=args.max_samples,
        obs_times=obs_times,
        obs_ris_indices=obs_ris_indices,
        complex_layout=complex_layout,
        semantic_profile=semantic_profile,
    )
    result = evaluate_model(model, loader, device)
    result.update(
        {
            "stage": (
                "evaluation_smoke_test"
                if args.max_samples
                else f"independent_{args.split}"
            ),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "domain": domain,
            "split": args.split,
            "data_path": str(loader.dataset.path),
            "parameters": model_parameter_report(model),
            "obs_time_index": list(obs_times),
            "obs_ris_index": list(obs_ris_indices),
            "complex_layout": complex_layout,
        }
    )
    output = Path(args.output)
    write_json(output, result)
    if args.per_snr:
        if domain != "mobility" or args.split != "test":
            raise ValueError("--per-snr is only defined for the mobility test split.")
        per_snr_evaluation(
            model,
            loader,
            device,
            output.with_suffix(".per_snr.csv"),
            args.snr_values,
            args.samples_per_snr,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def per_snr_evaluation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output: Path,
    snr_values: tuple[int, ...],
    samples_per_snr: int,
) -> None:
    if not snr_values or len(set(snr_values)) != len(snr_values):
        raise ValueError("--snr-values must contain unique values.")
    if samples_per_snr <= 0:
        raise ValueError("--samples-per-snr must be positive.")
    expected = len(snr_values) * samples_per_snr
    if len(loader.dataset) != expected:
        raise ValueError(
            f"Per-SNR evaluation expects {len(snr_values)} x "
            f"{samples_per_snr} = {expected} samples, but the loader has "
            f"{len(loader.dataset)}. Verify dataset ordering and grouping."
        )
    dataset_indices = getattr(loader.dataset, "indices", None)
    if dataset_indices is not None and not np.array_equal(
        np.asarray(dataset_indices), np.arange(expected)
    ):
        raise ValueError(
            "Per-SNR evaluation requires the complete, original sample order. "
            "Do not combine --per-snr with fraction or a truncating "
            "--max-samples setting."
        )
    accumulators = {snr: MetricAccumulator() for snr in snr_values}
    model.eval()
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            prediction = model(batch)
            groups = batch["sample_index"] // samples_per_snr
            for group in torch.unique(groups):
                selected = torch.where(groups == group)[0]
                group_int = int(group)
                if 0 <= group_int < len(snr_values):
                    sub_batch = {
                        key: (
                            value.index_select(0, selected)
                            if value.ndim and value.shape[0] == prediction.shape[0]
                            else value
                        )
                        for key, value in batch.items()
                    }
                    accumulators[snr_values[group_int]].update(
                        prediction.index_select(0, selected), sub_batch
                    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["snr_db", "sample_count", "nmse_linear", "nmse_db"])
        for snr, accumulator in accumulators.items():
            value = accumulator.compute()
            writer.writerow(
                [
                    snr,
                    value["sample_count"],
                    value["nmse_linear"].get("overall"),
                    value["nmse_db"].get("overall"),
                ]
            )


def interpolation_command(args: argparse.Namespace) -> None:
    obs_times = args.obs_times if args.domain == "mobility" else (0,)
    loader = make_loader(
        args.domain,
        args.split,
        path=args.data_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        max_samples=args.max_samples,
        obs_times=obs_times,
        obs_ris_indices=args.obs_ris_indices,
        complex_layout=args.complex_layout,
        semantic_profile=args.semantic_profile,
    )
    metrics = MetricAccumulator()
    for batch in loader:
        prediction = interpolation_baseline(
            batch, spatial=args.spatial, temporal=args.temporal
        )
        metrics.update(prediction, batch)
    result = metrics.compute()
    result.update(
        {
            "baseline": "LS coarse input + interpolation",
            "domain": args.domain,
            "split": args.split,
            "spatial": args.spatial,
            "temporal": args.temporal,
            "obs_time_index": list(obs_times),
            "obs_ris_index": list(args.obs_ris_indices),
            "complex_layout": args.complex_layout,
            "semantic_profile": args.semantic_profile,
        }
    )
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def ridge_command(args: argparse.Namespace) -> None:
    obs_times = args.obs_times if args.domain == "mobility" else (0,)
    train_loader = make_loader(
        args.domain,
        "train",
        path=args.train_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        max_samples=args.max_train,
        obs_times=obs_times,
        obs_ris_indices=args.obs_ris_indices,
        complex_layout=args.complex_layout,
        semantic_profile=args.semantic_profile,
    )
    val_loader = make_loader(
        args.domain,
        "validation",
        path=args.val_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        max_samples=args.max_val,
        obs_times=obs_times,
        obs_ris_indices=args.obs_ris_indices,
        complex_layout=args.complex_layout,
        semantic_profile=args.semantic_profile,
    )
    statistics = RidgeStatistics.accumulate(train_loader)
    candidates = []
    best_model, best_linear = None, float("inf")
    for regularization in args.lambdas:
        model = statistics.solve(regularization)
        result = model.evaluate(val_loader)
        candidates.append(result)
        linear = float(result["nmse_linear"]["overall"])
        if linear < best_linear:
            best_model, best_linear = model, linear
    assert best_model is not None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_model.save(output.with_suffix(".pth"))
    report: dict[str, object] = {
        "baseline": "Empirical Ridge",
        "domain": args.domain,
        "fit_rows": statistics.rows,
        "validation_candidates": candidates,
        "best_regularization": best_model.regularization,
        "model_path": str(output.with_suffix(".pth")),
    }
    if args.test:
        test_loader = make_loader(
            args.domain,
            "test",
            path=args.test_path,
            data_root=args.data_root,
            batch_size=args.batch_size,
            workers=args.workers,
            max_samples=args.max_test,
            obs_times=obs_times,
            obs_ris_indices=args.obs_ris_indices,
            complex_layout=args.complex_layout,
            semantic_profile=args.semantic_profile,
        )
        report["independent_test"] = best_model.evaluate(test_loader)
    write_json(output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def joint_command(args: argparse.Namespace) -> None:
    device = device_from(args.device)
    config = model_config(args, "mobility")
    model = build_model("phymeta_stgt", **config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    smoke = args.mode == "smoke"
    max_train, max_val = (32, 8) if smoke else (None, None)
    quasi_train = make_loader(
        "quasi",
        "train",
        data_root=args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        max_samples=max_train,
        shuffle=True,
        seed=args.seed,
        obs_ris_indices=args.obs_ris_indices,
        complex_layout=args.complex_layout,
        semantic_profile=args.semantic_profile,
    )
    mobility_train = make_loader(
        "mobility",
        "train",
        data_root=args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        max_samples=max_train,
        shuffle=True,
        seed=args.seed,
        obs_times=args.obs_times,
        obs_ris_indices=args.obs_ris_indices,
        complex_layout=args.complex_layout,
        semantic_profile=args.semantic_profile,
    )
    quasi_val = make_loader(
        "quasi",
        "validation",
        data_root=args.data_root,
        batch_size=args.eval_batch_size,
        workers=args.workers,
        max_samples=max_val,
        obs_ris_indices=args.obs_ris_indices,
        complex_layout=args.complex_layout,
        semantic_profile=args.semantic_profile,
    )
    mobility_val = make_loader(
        "mobility",
        "validation",
        data_root=args.data_root,
        batch_size=args.eval_batch_size,
        workers=args.workers,
        max_samples=max_val,
        obs_times=args.obs_times,
        obs_ris_indices=args.obs_ris_indices,
        complex_layout=args.complex_layout,
        semantic_profile=args.semantic_profile,
    )
    run_name = args.run_name or f"joint_phymeta_stgt_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_root) / run_name
    if run_dir.exists():
        raise FileExistsError(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "results").mkdir(parents=True)
    (run_dir / "command.txt").write_text(shlex.join(sys.argv), encoding="utf-8")
    weights = LossWeights()
    epochs = 1 if smoke else args.epochs
    steps = args.steps_per_epoch or 2 * max(len(quasi_train), len(mobility_train))
    history, best = [], float("inf")
    metadata = {
        "domain": "joint",
        "sampling": "balanced alternating homogeneous task batches",
        "mobility_obs_time_index": list(args.obs_times),
        "obs_ris_index": list(args.obs_ris_indices),
        "complex_layout": args.complex_layout,
        "semantic_profile": args.semantic_profile,
        "seed": args.seed,
    }
    for epoch in range(1, epochs + 1):
        train_result = train_balanced_joint_epoch(
            model,
            (quasi_train, mobility_train),
            optimizer,
            device,
            weights,
            steps=steps,
        )
        q_result = evaluate_model(model, quasi_val, device)
        m_result = evaluate_model(model, mobility_val, device)
        score = (
            float(q_result["nmse_linear"]["overall"])
            + float(m_result["nmse_linear"]["overall"])
        ) / 2
        history.append(
            {
                "epoch": epoch,
                "train_total": train_result["total"],
                "quasi_validation_nmse_db": nmse_db_from_result(q_result),
                "mobility_validation_nmse_db": nmse_db_from_result(m_result),
                "balanced_linear_score": score,
            }
        )
        save_checkpoint(
            run_dir / "checkpoints/last_checkpoint.pth",
            model,
            optimizer,
            epoch=epoch,
            best_nmse=min(best, score),
            model_name="phymeta_stgt",
            model_config=config,
            metadata=metadata,
        )
        if score < best:
            best = score
            save_checkpoint(
                run_dir / "checkpoints/best_checkpoint.pth",
                model,
                optimizer,
                epoch=epoch,
                best_nmse=best,
                model_name="phymeta_stgt",
                model_config=config,
                metadata=metadata,
            )
        write_history(run_dir / "results/training_history.csv", history)
        print(
            f"epoch={epoch} quasi={history[-1]['quasi_validation_nmse_db']:.4f} dB "
            f"mobility={history[-1]['mobility_validation_nmse_db']:.4f} dB"
        )
    write_json(
        run_dir / "results/final_result.json",
        {
            "status": "joint_smoke_test" if smoke else "joint_validation",
            "best_balanced_linear_score": best,
            "history": history,
            "metadata": metadata,
            "parameters": model_parameter_report(model),
            "complexity": profile_model_complexity(
                model,
                canonical_batch("mobility", batch_size=1, device=device),
            ),
        },
    )
    print(f"Joint run completed: {run_dir}")


def add_runtime(
    parser: argparse.ArgumentParser,
    *,
    batch_size: int = 32,
    workers: int = 8,
) -> None:
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=workers)
    parser.add_argument("--batch-size", type=int, default=batch_size)


def add_early_stopping(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use validation-NMSE early stopping in full mode. Smoke mode "
            "always runs its single epoch."
        ),
    )
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=15)


def add_training_profile(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--training-profile",
        choices=["auto", "unified", "lpan_public_code"],
        default="auto",
    )
    parser.add_argument("--lpan-initial-learning-rate", type=float, default=1e-3)
    parser.add_argument("--lpan-final-learning-rate", type=float, default=5e-6)
    parser.add_argument(
        "--lpan-weight-decay",
        type=float,
        default=0.01,
        help="Explicit AdamW value matching the current PyTorch default.",
    )


def add_data_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help=(
            "Dataset root. Defaults to LPAN_DATA_ROOT or <project>/data. "
            "Explicit --train-path/--val-path/--data-path values take priority."
        ),
    )


def add_data_semantics(
    parser: argparse.ArgumentParser, *, optional: bool = False
) -> None:
    parser.add_argument(
        "--obs-ris-indices",
        type=ints,
        default=None if optional else tuple(range(0, 256, 8)),
        help="32 comma-separated full-grid indices in Yd column order.",
    )
    parser.add_argument(
        "--complex-layout",
        choices=["grouped", "interleaved"],
        default=None if optional else "grouped",
        help="Raw real/imag channel ordering in Yd and Hd.",
    )
    parser.add_argument(
        "--semantic-profile",
        choices=["official_lpan", "custom"],
        default=None if optional else "official_lpan",
        help=(
            "official_lpan locks the verified RIS/time/layout semantics; "
            "custom is only for independently rearranged datasets."
        ),
    )


def add_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)


def add_study_training_protocol(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--domain", choices=["quasi", "mobility"], required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--train-path")
    parser.add_argument("--val-path")
    parser.add_argument("--obs-times", type=ints, default=(0, 1))
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--nmse-weight", type=float, default=1.0)
    parser.add_argument("--char-weight", type=float, default=0.1)
    parser.add_argument("--obs-weight", type=float, default=0.1)
    parser.add_argument("--delta-weight", type=float, default=0.05)
    parser.add_argument("--study-name")
    parser.add_argument("--output-root", default="runs")
    add_data_root(parser)
    add_data_semantics(parser)
    add_runtime(parser)
    add_model(parser)
    add_early_stopping(parser)
    add_training_profile(parser)
    parser.set_defaults(
        model="phymeta_stgt",
        adaptation="full",
        pretrained=None,
        resume=None,
        run_name=None,
        ablation="none",
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="LPAN two-dataset channel completion experiments."
    )
    commands = root.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit")
    audit.add_argument("--mobility-obs-times", type=ints, default=(0, 1))
    audit.add_argument("--output", default="runs/data_audit.json")
    add_data_root(audit)
    add_data_semantics(audit)
    audit.set_defaults(func=audit_command)

    profile = commands.add_parser("profile")
    profile.add_argument("--domain", choices=["quasi", "mobility"], required=True)
    profile.add_argument(
        "--models",
        type=strings,
        default=(
            "lpan_progressive",
            "lpan_l_progressive",
            "lpan_l_direct",
            "edsr_lite",
            "spatial_gcn",
            "cnn_gru",
            "gcn_gru",
            "phymeta_stgt",
        ),
        help="Comma-separated registered model names.",
    )
    profile.add_argument("--batch-size", type=int, default=1)
    profile.add_argument("--device", default="cpu")
    profile.add_argument("--output", default="runs/complexity_profile.json")
    profile.add_argument(
        "--ablation",
        choices=("none",) + ARCHITECTURE_ABLATIONS,
        default="none",
    )
    add_model(profile)
    profile.set_defaults(func=profile_command)

    benchmark = commands.add_parser("benchmark-batch")
    benchmark.add_argument(
        "--domain", choices=["quasi", "mobility"], required=True
    )
    benchmark.add_argument("--train-path")
    benchmark.add_argument("--obs-times", type=ints, default=(0, 1))
    benchmark.add_argument(
        "--candidates", type=ints, default=(16, 32, 64, 128)
    )
    benchmark.add_argument("--max-samples", type=int, default=1024)
    benchmark.add_argument("--workers", type=int, default=8)
    benchmark.add_argument("--device", default="cuda")
    benchmark.add_argument("--seed", type=int, default=123)
    benchmark.add_argument("--learning-rate", type=float, default=2e-4)
    benchmark.add_argument("--weight-decay", type=float, default=1e-5)
    benchmark.add_argument("--grad-clip", type=float, default=1.0)
    benchmark.add_argument(
        "--output", default="runs/batch_benchmark.json"
    )
    add_data_root(benchmark)
    add_data_semantics(benchmark)
    add_model(benchmark)
    benchmark.set_defaults(func=benchmark_batch_command)

    train = commands.add_parser("train")
    train.add_argument("--domain", choices=["quasi", "mobility"], required=True)
    train.add_argument(
        "--model",
        choices=[
            "lpan_progressive",
            "lpan_l_progressive",
            "lpan_l_direct",
            "edsr_lite",
            "spatial_gcn",
            "cnn_gru",
            "gcn_gru",
            "phymeta_stgt",
        ],
        required=True,
    )
    train.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    train.add_argument("--train-path")
    train.add_argument("--val-path")
    train.add_argument("--obs-times", type=ints, default=(0, 1))
    train.add_argument("--fraction", type=float, default=1.0)
    train.add_argument("--max-train", type=int)
    train.add_argument("--max-val", type=int)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--eval-batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=123)
    train.add_argument("--nmse-weight", type=float, default=1.0)
    train.add_argument("--char-weight", type=float, default=0.1)
    train.add_argument("--obs-weight", type=float, default=0.1)
    train.add_argument("--delta-weight", type=float, default=0.05)
    train.add_argument(
        "--adaptation",
        choices=["full", "frozen_spatial", "adapter_only", "selective"],
        default="full",
    )
    train.add_argument("--pretrained")
    train.add_argument("--resume")
    train.add_argument(
        "--ablation",
        choices=ABLATION_VARIANTS,
        default="none",
        help="Run one controlled PhyMeta-STGT architecture or loss ablation.",
    )
    train.add_argument("--run-name")
    train.add_argument("--output-root", default="runs")
    add_data_root(train)
    add_data_semantics(train)
    add_runtime(train)
    add_model(train)
    add_early_stopping(train)
    add_training_profile(train)
    train.set_defaults(func=train_command)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--domain", choices=["quasi", "mobility"])
    evaluate.add_argument("--split", choices=["validation", "test"], default="test")
    evaluate.add_argument("--data-path")
    evaluate.add_argument("--obs-times", type=ints)
    evaluate.add_argument("--max-samples", type=int)
    evaluate.add_argument("--per-snr", action="store_true")
    evaluate.add_argument(
        "--snr-values",
        type=ints,
        default=tuple(range(-10, 31, 5)),
        help="Comma-separated SNR labels in test-file order.",
    )
    evaluate.add_argument(
        "--samples-per-snr",
        type=int,
        default=1000,
        help="Expected contiguous test samples for each SNR label.",
    )
    evaluate.add_argument("--output", default="runs/evaluation_result.json")
    evaluate.add_argument(
        "--allow-semantic-override",
        action="store_true",
        help=(
            "Allow explicitly supplied observation times, RIS indices, or "
            "complex layout to differ from checkpoint metadata."
        ),
    )
    add_data_root(evaluate)
    add_data_semantics(evaluate, optional=True)
    add_runtime(evaluate, batch_size=64)
    evaluate.set_defaults(func=evaluate_command)

    interpolation = commands.add_parser("interpolate")
    interpolation.add_argument("--domain", choices=["quasi", "mobility"], required=True)
    interpolation.add_argument("--split", choices=["validation", "test"], default="validation")
    interpolation.add_argument("--data-path")
    interpolation.add_argument("--obs-times", type=ints, default=(0, 1))
    interpolation.add_argument("--spatial", choices=["linear", "nearest"], default="linear")
    interpolation.add_argument("--temporal", choices=["linear", "nearest"], default="linear")
    interpolation.add_argument("--max-samples", type=int)
    interpolation.add_argument("--output", default="runs/interpolation_result.json")
    interpolation.add_argument("--workers", type=int, default=8)
    interpolation.add_argument("--batch-size", type=int, default=64)
    add_data_root(interpolation)
    add_data_semantics(interpolation)
    interpolation.set_defaults(func=interpolation_command)

    ridge = commands.add_parser("ridge")
    ridge.add_argument("--domain", choices=["quasi", "mobility"], required=True)
    ridge.add_argument("--train-path")
    ridge.add_argument("--val-path")
    ridge.add_argument("--test-path")
    ridge.add_argument("--obs-times", type=ints, default=(0, 1))
    ridge.add_argument("--lambdas", type=floats, default=(1e-6, 1e-4, 1e-2, 1.0))
    ridge.add_argument("--max-train", type=int)
    ridge.add_argument("--max-val", type=int)
    ridge.add_argument("--max-test", type=int)
    ridge.add_argument("--test", action="store_true")
    ridge.add_argument("--output", default="runs/ridge_result.json")
    ridge.add_argument("--workers", type=int, default=8)
    ridge.add_argument("--batch-size", type=int, default=64)
    add_data_root(ridge)
    add_data_semantics(ridge)
    ridge.set_defaults(func=ridge_command)

    joint = commands.add_parser("joint")
    joint.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    joint.add_argument("--obs-times", type=ints, default=(0, 1))
    joint.add_argument("--epochs", type=int, default=100)
    joint.add_argument("--steps-per-epoch", type=int)
    joint.add_argument("--eval-batch-size", type=int, default=64)
    joint.add_argument("--learning-rate", type=float, default=2e-4)
    joint.add_argument("--weight-decay", type=float, default=1e-5)
    joint.add_argument("--seed", type=int, default=123)
    joint.add_argument("--run-name")
    joint.add_argument("--output-root", default="runs")
    add_data_root(joint)
    add_data_semantics(joint)
    add_runtime(joint)
    add_model(joint)
    joint.set_defaults(func=joint_command)

    tune = commands.add_parser("tune")
    add_study_training_protocol(tune)
    tune.add_argument(
        "--tuning-protocol",
        choices=["two_round_validation_promotion", "targeted_boundary"],
        default="two_round_validation_promotion",
        help="Preserve the historical protocol or run sequential boundary tuning.",
    )
    tune.add_argument("--strategy", choices=["grid", "random"], default="random")
    tune.add_argument("--max-trials", type=int, default=12)
    tune.add_argument("--search-seed", type=int, default=2026)
    tune.add_argument(
        "--round1-epochs",
        type=int,
        default=25,
        help="Epoch budget for every candidate before validation ranking.",
    )
    tune.add_argument(
        "--promote-top-k",
        type=int,
        default=3,
        help="Number of round-1 candidates resumed to --epochs.",
    )
    tune.add_argument("--hidden-values", type=ints, default=(48, 64, 96))
    tune.add_argument("--graph-layer-values", type=ints, default=(1, 2, 3))
    tune.add_argument("--head-values", type=ints, default=(4, 8))
    tune.add_argument("--dropout-values", type=floats, default=(0.0, 0.1))
    tune.add_argument(
        "--learning-rate-values",
        type=floats,
        default=(1e-4, 2e-4, 5e-4),
    )
    tune.add_argument(
        "--weight-decay-values", type=floats, default=(0.0, 1e-5)
    )
    tune.set_defaults(func=tune_command)

    ablate = commands.add_parser("ablate")
    add_study_training_protocol(ablate)
    ablate.add_argument(
        "--best-result",
        type=Path,
        help=(
            "Stage-B best_result.json whose hyperparameters are inherited by "
            "every ablation. Required in full mode."
        ),
    )
    ablate.add_argument(
        "--variants",
        type=strings,
        help=(
            "Comma-separated variants. The full model is always included. "
            f"Available: {','.join(ABLATION_VARIANTS)}"
        ),
    )
    ablate.set_defaults(func=ablate_command)
    return root


def main() -> None:
    args = parser().parse_args()
    seed = getattr(args, "seed", 123)
    seed_everything(seed)
    args.func(args)


if __name__ == "__main__":
    main()
