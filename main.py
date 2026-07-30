from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
import time
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader

from lpan.data import LPANH5Dataset
from lpan.engine import (
    configure_adaptation,
    evaluate_model,
    move_batch,
    nmse_db_from_result,
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
            "complex_layout": "grouped real blocks followed by imaginary blocks",
            "obs_ris_index": list(range(0, 256, 8)),
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
                "attempted_paths": [str(candidate) for candidate in candidates],
            }
            if path.is_file():
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
                obs_times = args.mobility_obs_times if domain == "mobility" else (0,)
                dataset = LPANH5Dataset(
                    path, domain, split, max_samples=1, obs_time_index=obs_times
                )
                sample = dataset[0]
                entry["unified_sample_shapes"] = {
                    key: list(value.shape)
                    for key, value in sample.items()
                    if key in {"obs_h", "target_h", "observation_mask"}
                }
            report["files"].append(entry)
    output = Path(args.output)
    write_json(output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Audit written to {output}")


def train_command(args: argparse.Namespace) -> None:
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
    )
    config = model_config(args, domain)
    model = build_model(args.model, **config)
    device = device_from(args.device)
    model.to(device)

    pretrained_metadata = None
    if args.pretrained:
        state = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(
            state["model_state"], strict=False
        )
        pretrained_metadata = {
            "path": str(Path(args.pretrained).resolve()),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }
    adaptation = configure_adaptation(model, args.adaptation)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_epoch, best_nmse = 1, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["epoch"]) + 1
        best_nmse = float(state["best_nmse"])

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{domain}_{args.model}_{args.mode}_{stamp}"
    run_dir = Path(args.output_root) / run_name
    if run_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Run directory already exists; choose another --run-name: {run_dir}"
        )
    checkpoints = run_dir / "checkpoints"
    results = run_dir / "results"
    checkpoints.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(
        shlex.join(sys.argv), encoding="utf-8"
    )

    weights = LossWeights(
        args.nmse_weight,
        args.char_weight,
        args.obs_weight,
        args.delta_weight if domain == "mobility" else 0.0,
    )
    history: list[dict[str, object]] = []
    metadata = {
        "domain": domain,
        "mode": args.mode,
        "seed": args.seed,
        "obs_time_index": list(obs_times),
        "obs_ris_index": list(range(0, 256, 8)),
        "train_fraction": args.fraction,
        "adaptation": args.adaptation,
        "adaptation_parameters": adaptation,
        "pretrained": pretrained_metadata,
        "train_path": str(train_loader.dataset.path),
        "validation_path": str(val_loader.dataset.path),
    }
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
        save_checkpoint(
            checkpoints / "last_checkpoint.pth",
            model,
            optimizer,
            epoch=epoch,
            best_nmse=min(best_nmse, val_nmse),
            model_name=args.model,
            model_config=config,
            metadata=metadata,
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
            )
        write_history(results / "training_history.csv", history)
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


def evaluate_command(args: argparse.Namespace) -> None:
    device = device_from(args.device)
    model, state = load_checkpoint_model(args.checkpoint, device)
    domain = args.domain or state.get("metadata", {}).get("domain")
    if domain not in {"quasi", "mobility"}:
        raise ValueError("Specify --domain because checkpoint metadata is missing.")
    obs_times = args.obs_times if domain == "mobility" else (0,)
    loader = make_loader(
        domain,
        args.split,
        path=args.data_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        max_samples=args.max_samples,
        obs_times=obs_times,
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
        }
    )
    output = Path(args.output)
    write_json(output, result)
    if args.per_snr:
        if domain != "mobility" or args.split != "test":
            raise ValueError("--per-snr is only defined for the mobility test split.")
        per_snr_evaluation(model, loader, device, output.with_suffix(".per_snr.csv"))
    print(json.dumps(result, indent=2, ensure_ascii=False))


def per_snr_evaluation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output: Path,
) -> None:
    accumulators = {snr: MetricAccumulator() for snr in range(-10, 31, 5)}
    model.eval()
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            prediction = model(batch)
            groups = batch["sample_index"] // 1000
            for group in torch.unique(groups):
                selected = torch.where(groups == group)[0]
                group_int = int(group)
                if 0 <= group_int <= 8:
                    sub_batch = {
                        key: (
                            value.index_select(0, selected)
                            if value.ndim and value.shape[0] == prediction.shape[0]
                            else value
                        )
                        for key, value in batch.items()
                    }
                    accumulators[-10 + 5 * group_int].update(
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
    )
    quasi_val = make_loader(
        "quasi",
        "validation",
        data_root=args.data_root,
        batch_size=args.eval_batch_size,
        workers=args.workers,
        max_samples=max_val,
    )
    mobility_val = make_loader(
        "mobility",
        "validation",
        data_root=args.data_root,
        batch_size=args.eval_batch_size,
        workers=args.workers,
        max_samples=max_val,
        obs_times=args.obs_times,
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
    add_runtime(train)
    add_model(train)
    train.set_defaults(func=train_command)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--domain", choices=["quasi", "mobility"])
    evaluate.add_argument("--split", choices=["validation", "test"], default="test")
    evaluate.add_argument("--data-path")
    evaluate.add_argument("--obs-times", type=ints, default=(0, 1))
    evaluate.add_argument("--max-samples", type=int)
    evaluate.add_argument("--per-snr", action="store_true")
    evaluate.add_argument("--output", default="runs/evaluation_result.json")
    add_data_root(evaluate)
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
    add_runtime(joint)
    add_model(joint)
    joint.set_defaults(func=joint_command)
    return root


def main() -> None:
    args = parser().parse_args()
    torch.manual_seed(getattr(args, "seed", 123))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(getattr(args, "seed", 123))
    args.func(args)


if __name__ == "__main__":
    main()
