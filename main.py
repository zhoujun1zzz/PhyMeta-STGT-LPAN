from __future__ import annotations

import argparse
import csv
import json
import random
import shlex
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

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
from lpan.objectives import LossWeights
from lpan.paths import (
    dataset_candidates,
    default_data_root,
    resolve_dataset_path,
)
from lpan.ridge import EmpiricalRidge, RidgeStatistics


PROJECT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = default_data_root(PROJECT)


def ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


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


def model_config(args: argparse.Namespace, domain: str) -> dict[str, object]:
    return {
        "domain": domain,
        "hidden": args.hidden,
        "graph_layers": args.graph_layers,
        "heads": args.heads,
        "dropout": args.dropout,
    }


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
                    )
                    sample = dataset[0]
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


def validate_training_request(
    model_name: str,
    adaptation: str,
    pretrained: str | None,
    resume: str | None,
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
    architecture_keys = ("hidden", "graph_layers", "heads")
    mismatches = {
        key: {
            "checkpoint": saved_config.get(key),
            "current": current_config.get(key),
        }
        for key in architecture_keys
        if saved_config.get(key) != current_config.get(key)
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


def train_command(args: argparse.Namespace) -> None:
    validate_training_request(
        args.model, args.adaptation, args.pretrained, args.resume
    )

    domain = args.domain
    obs_times = args.obs_times if domain == "mobility" else (0,)
    smoke = args.mode == "smoke"
    max_train = args.max_train or (64 if smoke else None)
    max_val = args.max_val or (16 if smoke else None)
    epochs = 1 if smoke else args.epochs
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
    )
    config = model_config(args, domain)
    model = build_model(args.model, **config)
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
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
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
    command = shlex.join(sys.argv)
    if not args.resume:
        (run_dir / "command.txt").write_text(command, encoding="utf-8")

    weights = LossWeights(
        args.nmse_weight,
        args.char_weight,
        args.obs_weight,
        args.delta_weight if domain == "mobility" else 0.0,
    )
    metadata = {
        "domain": domain,
        "mode": args.mode,
        "seed": args.seed,
        "obs_time_index": list(obs_times),
        "obs_ris_index": list(args.obs_ris_indices),
        "complex_layout": args.complex_layout,
        "train_fraction": args.fraction,
        "adaptation": args.adaptation,
        "adaptation_parameters": adaptation,
        "pretrained": pretrained_metadata,
        "train_path": str(train_loader.dataset.path),
        "validation_path": str(val_loader.dataset.path),
        "max_train": max_train,
        "max_validation": max_val,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "workers": args.workers,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "loss_weights": {
            "nmse": weights.nmse,
            "charbonnier": weights.charbonnier,
            "observation": weights.observation,
            "delta": weights.delta,
        },
    }
    start_epoch, best_nmse = 1, float("inf")
    history: list[dict[str, object]] = []
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        saved_metadata = state.get("metadata", {})
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
            "train_fraction": (
                saved_metadata.get("train_fraction"),
                metadata["train_fraction"],
            ),
            "adaptation": (
                saved_metadata.get("adaptation"),
                metadata["adaptation"],
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
            "grad_clip": (
                saved_metadata.get("grad_clip"),
                metadata["grad_clip"],
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
        if not history or int(history[-1]["epoch"]) != int(state["epoch"]):
            raise ValueError(
                "training_history.csv is missing or out of sync with checkpoint."
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

    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        train_result = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            weights,
            grad_clip=args.grad_clip,
        )
        val_result = evaluate_model(model, val_loader, device)
        val_nmse = float(val_result["nmse_linear"]["overall"])
        row = {
            "epoch": epoch,
            "train_total": train_result["total"],
            "train_nmse": train_result["nmse"],
            "validation_nmse_linear": val_nmse,
            "validation_nmse_db": nmse_db_from_result(val_result),
            "epoch_seconds": time.perf_counter() - started,
        }
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
            best_nmse=min(best_nmse, val_nmse),
            model_name=args.model,
            model_config=config,
            metadata=metadata,
            rng_state=rng_state,
        )
        if val_nmse < best_nmse:
            best_nmse = val_nmse
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
    final = {
        "status": "smoke_test" if smoke else "validation",
        "best_validation_nmse_linear": best_nmse,
        "best_validation_nmse_db": 10 * torch.log10(torch.tensor(best_nmse)).item(),
        "history": history,
        "metadata": metadata,
        "parameters": model_parameter_report(model),
    }
    write_json(results / "final_result.json", final)
    print(f"Run completed: {run_dir}")


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
) -> tuple[str, tuple[int, ...], tuple[int, ...], str]:
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
    )


def evaluate_command(args: argparse.Namespace) -> None:
    device = device_from(args.device)
    model, state = load_checkpoint_model(args.checkpoint, device)
    domain, obs_times, obs_ris_indices, complex_layout = (
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
        },
    )
    print(f"Joint run completed: {run_dir}")


def add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)


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


def add_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)


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

    train = commands.add_parser("train")
    train.add_argument("--domain", choices=["quasi", "mobility"], required=True)
    train.add_argument(
        "--model",
        choices=["edsr_lite", "spatial_gcn", "cnn_gru", "gcn_gru", "phymeta_stgt"],
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
    train.add_argument("--eval-batch-size", type=int, default=2)
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
    train.add_argument("--run-name")
    train.add_argument("--output-root", default="runs")
    add_data_root(train)
    add_data_semantics(train)
    add_runtime(train)
    add_model(train)
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
    add_runtime(evaluate)
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
    interpolation.add_argument("--workers", type=int, default=0)
    interpolation.add_argument("--batch-size", type=int, default=8)
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
    ridge.add_argument("--workers", type=int, default=0)
    ridge.add_argument("--batch-size", type=int, default=2)
    add_data_root(ridge)
    add_data_semantics(ridge)
    ridge.set_defaults(func=ridge_command)

    joint = commands.add_parser("joint")
    joint.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    joint.add_argument("--obs-times", type=ints, default=(0, 1))
    joint.add_argument("--epochs", type=int, default=100)
    joint.add_argument("--steps-per-epoch", type=int)
    joint.add_argument("--eval-batch-size", type=int, default=2)
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
    return root


def main() -> None:
    args = parser().parse_args()
    seed = getattr(args, "seed", 123)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    args.func(args)


if __name__ == "__main__":
    main()
