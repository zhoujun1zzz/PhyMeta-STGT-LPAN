from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import h5py


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lpan.paths import resolve_dataset_path  # noqa: E402
from lpan.transfer import (  # noqa: E402
    ADAPTATION_MODES,
    deterministic_subset_indices,
    file_sha256,
    subset_index_hash,
)


METHODS = ADAPTATION_MODES
FRACTIONS = (0.01, 0.05, 0.10, 0.20, 1.0)
METHOD_PRIORITY = (
    "adapter_head",
    "full_finetune",
    "scratch",
    "domain_adapter_only",
    "frozen_spatial",
)
REUSE_FIELDS = (
    "method",
    "fraction",
    "seed",
    "git.commit",
    "source.domain",
    "source.checkpoint_sha256",
    "target.domain",
    "target.train_identity.size_bytes",
    "target.validation_identity.size_bytes",
    "subset.size",
    "subset.sha256",
    "model",
    "model_config",
    "optimizer",
    "scheduler",
    "training_budget",
    "validation_protocol",
    "metric_definition",
    "test_split_used",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def fraction_label(fraction: float) -> str:
    return f"f{round(100 * fraction):02d}"


def run_name(method: str, fraction: float, seed: int) -> str:
    return f"{method}_{fraction_label(fraction)}_seed{seed}"


def file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def sample_count(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        for key in ("Yd", "input_da"):
            if key in handle:
                return int(handle[key].shape[3])
    raise KeyError(f"No supported input array found in {path}")


def source_record(
    path: Path | None,
    *,
    dry_run: bool,
    explicit_git_commit: str | None = None,
) -> dict[str, object]:
    if path is None:
        if dry_run:
            return {
                "domain": "quasi",
                "seed": 123,
                "checkpoint": None,
                "checkpoint_sha256": None,
                "git_commit": None,
                "validation": "not supplied for dry-run",
            }
        raise ValueError("--source-checkpoint is required for transfer methods.")
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    metadata = state.get("metadata") if isinstance(state, dict) else None
    if not isinstance(metadata, dict):
        raise ValueError("Source checkpoint is missing metadata.")
    checks = {
        "model_name": (state.get("model_name"), "phymeta_stgt"),
        "domain": (metadata.get("domain"), "quasi"),
        "seed": (metadata.get("seed"), 123),
        "test_split_used": (metadata.get("test_split_used", False), False),
    }
    mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in checks.items()
        if actual != expected
    }
    if mismatches:
        raise ValueError(f"Invalid source checkpoint provenance: {mismatches}")
    source_commit = (
        metadata.get("git_commit")
        or metadata.get("source_commit")
        or explicit_git_commit
    )
    if not source_commit:
        raise ValueError(
            "Source checkpoint git commit is missing; provide "
            "--source-git-commit from its formal run provenance."
        )
    return {
        "domain": "quasi",
        "seed": 123,
        "checkpoint": str(path),
        "checkpoint_sha256": file_sha256(path),
        "git_commit": source_commit,
        "model_config": state.get("model_config"),
        "validation": "metadata verified",
    }


def nested(payload: dict[str, Any], dotted: str) -> Any:
    value: Any = payload
    for token in dotted.split("."):
        if not isinstance(value, dict) or token not in value:
            return None
        value = value[token]
    return value


def validate_reuse(
    expected: dict[str, Any], candidate: dict[str, Any]
) -> tuple[bool, list[str]]:
    reasons = [
        field
        for field in REUSE_FIELDS
        if nested(expected, field) != nested(candidate, field)
    ]
    if candidate.get("status") != "completed":
        reasons.append("status")
    if candidate.get("test_split_used") is not False:
        reasons.append("test_split_used")
    for field in ("best_checkpoint", "final_result"):
        value = candidate.get(field)
        if not value or not Path(str(value)).is_file():
            reasons.append(field)
    return not reasons, sorted(set(reasons))


def history_manifests(roots: Iterable[Path]) -> list[tuple[Path, dict[str, Any]]]:
    manifests = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("manifest.json"):
            try:
                manifests.append((path, read_json(path)))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
    return manifests


def completed_manifest(
    run_dir: Path, expected: dict[str, Any]
) -> tuple[bool, list[str], dict[str, Any] | None]:
    path = run_dir / "manifest.json"
    if not path.is_file():
        return False, ["manifest missing"], None
    candidate = read_json(path)
    reusable, reasons = validate_reuse(expected, candidate)
    if not reusable:
        return False, reasons, candidate
    return True, [], candidate


def expected_manifest(
    args: argparse.Namespace,
    method: str,
    fraction: float,
    train_path: Path,
    val_path: Path,
    source: dict[str, object],
) -> dict[str, Any]:
    total = sample_count(train_path)
    indices = deterministic_subset_indices(
        total,
        fraction,
        args.seed,
        max_samples=args.max_train if args.smoke else None,
    )
    commit = git_value("rev-parse", "HEAD")
    dirty = bool(git_value("status", "--porcelain"))
    return {
        "schema_version": 1,
        "status": "planned",
        "git": {"commit": commit, "dirty": dirty},
        "source": source,
        "target": {
            "domain": "mobility",
            "train_identity": file_identity(train_path),
            "validation_identity": file_identity(val_path),
        },
        "method": method,
        "fraction": fraction,
        "seed": args.seed,
        "subset": {
            "size": len(indices),
            "total": total,
            "sha256": subset_index_hash(indices),
            "indices_record": "subsets/<run>.json",
        },
        "model": "phymeta_stgt",
        "model_config": {
            "hidden": args.hidden,
            "graph_layers": args.graph_layers,
            "heads": args.heads,
            "dropout": args.dropout,
            "ablation": "none",
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "betas": [0.9, 0.999],
            "weight_decay": args.weight_decay,
            "parameter_source": "requires_grad_only",
        },
        "scheduler": {"name": "constant"},
        "training_budget": {
            "epochs": 1 if args.smoke else args.epochs,
            "early_stopping": not args.no_early_stopping,
            "min_epochs": args.min_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "max_train": args.max_train if args.smoke else None,
            "max_validation": args.max_val if args.smoke else None,
            "precision": "FP32",
            "gradient_clip": args.grad_clip,
        },
        "validation_protocol": {
            "split": "validation",
            "fraction": 1.0,
            "checkpoint_selection": "minimum sample-level linear NMSE",
        },
        "metric_definition": "sample-level linear NMSE, reported in dB",
        "test_split_used": False,
        "runtime": {
            "python": sys.version.split()[0],
            "device_request": args.device,
        },
    }


def command_for(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    run_root: Path,
    train_path: Path,
    val_path: Path,
    resume_checkpoint: Path | None,
) -> list[str]:
    method = str(manifest["method"])
    command = [
        args.python,
        str(PROJECT / "main.py"),
        "train",
        "--domain",
        "mobility",
        "--model",
        "phymeta_stgt",
        "--mode",
        "smoke" if args.smoke else "full",
        "--train-path",
        str(train_path),
        "--val-path",
        str(val_path),
        "--fraction",
        str(manifest["fraction"]),
        "--seed",
        str(args.seed),
        "--adaptation",
        method,
        "--run-name",
        run_name(method, float(manifest["fraction"]), args.seed),
        "--output-root",
        str(run_root),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--workers",
        str(args.workers),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--grad-clip",
        str(args.grad_clip),
        "--min-epochs",
        str(args.min_epochs),
        "--patience",
        str(args.patience),
        "--device",
        args.device,
        "--hidden",
        str(args.hidden),
        "--graph-layers",
        str(args.graph_layers),
        "--heads",
        str(args.heads),
        "--dropout",
        str(args.dropout),
    ]
    if args.no_early_stopping:
        command.append("--no-early-stopping")
    if args.smoke:
        command.extend(["--max-train", str(args.max_train), "--max-val", str(args.max_val)])
    if resume_checkpoint is not None:
        command.extend(["--resume", str(resume_checkpoint)])
    elif method != "scratch":
        command.extend(["--pretrained", str(manifest["source"]["checkpoint"])])
    return command


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, command)


def select_cells(args: argparse.Namespace) -> list[tuple[str, float]]:
    methods = tuple(args.methods or METHODS)
    fractions = tuple(args.fractions or FRACTIONS)
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    if args.seed != 123:
        raise ValueError("The formal low-data protocol is frozen to seed 123.")
    if any(fraction not in FRACTIONS for fraction in fractions):
        raise ValueError(f"Fractions must be selected from {FRACTIONS}.")
    if args.stage == 1:
        fractions = (0.05,)
    elif args.stage == 2:
        fractions = tuple(value for value in fractions if value != 0.05)
    order = [method for method in METHOD_PRIORITY if method in methods]
    return [(method, fraction) for method in order for fraction in fractions]


def complete_manifest(
    manifest: dict[str, Any], run_dir: Path, run_root: Path
) -> dict[str, Any]:
    result_path = run_dir / "results" / "final_result.json"
    checkpoint = run_dir / "checkpoints" / "best_checkpoint.pth"
    result = read_json(result_path)
    if result.get("status") not in {"validation", "smoke_test"}:
        raise ValueError(f"Training result is incomplete: {result_path}")
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Training result metadata is missing.")
    actual_subset = metadata.get("train_subset")
    if not isinstance(actual_subset, dict) or actual_subset.get("sha256") != manifest["subset"]["sha256"]:
        raise ValueError("Training subset hash does not match the planned manifest.")
    adaptation = metadata.get("adaptation_parameters")
    if not isinstance(adaptation, dict):
        raise ValueError("Adaptation parameter report is missing.")
    manifest.update(
        {
            "status": "completed",
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "best_checkpoint": str(checkpoint.resolve()),
            "final_result": str(result_path.resolve()),
            "best_epoch": min(
                (row for row in result.get("history", []) if row.get("improved")),
                key=lambda row: float(row["validation_nmse_linear"]),
                default={},
            ).get("epoch"),
            "best_validation_nmse_db": result.get("best_validation_nmse_db"),
            "best_validation_nmse_linear": result.get("best_validation_nmse_linear"),
            "adaptation_seconds": result.get("adaptation_seconds"),
            "adaptation_minutes": result.get("adaptation_minutes"),
            "trainable_params": adaptation.get("trainable_parameters"),
            "total_params": adaptation.get("total_parameters"),
            "trainable_ratio_percent": adaptation.get("trainable_ratio_percent"),
            "trainable_module_names": adaptation.get("trainable_module_names"),
            "frozen_module_names": adaptation.get("frozen_module_names"),
            "runtime": metadata.get("runtime", manifest.get("runtime")),
            "reused_or_new": "NEW",
        }
    )
    write_json_atomic(run_dir / "manifest.json", manifest)
    write_json_atomic(run_root / "manifests" / f"{run_dir.name}.json", manifest)
    return manifest


def execute(args: argparse.Namespace) -> int:
    cells = select_cells(args)
    train_path = Path(args.train_path).resolve() if args.train_path else resolve_dataset_path(args.data_root, "mobility", "train")
    val_path = Path(args.val_path).resolve() if args.val_path else resolve_dataset_path(args.data_root, "mobility", "validation")
    needs_source = any(method != "scratch" for method, _ in cells)
    source = source_record(
        Path(args.source_checkpoint) if args.source_checkpoint else None,
        dry_run=args.dry_run or not needs_source,
        explicit_git_commit=args.source_git_commit,
    )
    histories = history_manifests(Path(value).resolve() for value in args.history_root)
    short_commit = git_value("rev-parse", "--short", "HEAD")
    root = Path(args.run_root).resolve() if args.run_root else PROJECT / "runs" / f"v1_lowdata_transfer_seed123_{short_commit}_{time.strftime('%Y%m%d_%H%M%S')}"
    plans: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []
    for method, fraction in cells:
        cell_source = (
            {
                "domain": None,
                "seed": None,
                "checkpoint": None,
                "checkpoint_sha256": None,
                "git_commit": None,
                "validation": "scratch uses random initialization",
            }
            if method == "scratch"
            else source
        )
        expected = expected_manifest(
            args, method, fraction, train_path, val_path, cell_source
        )
        status = "NEW"
        reusable_manifest = None
        for _, candidate in histories:
            reusable, _ = validate_reuse(expected, candidate)
            if reusable:
                status, reusable_manifest = "REUSE", candidate
                break
        run_dir = root / run_name(method, fraction, args.seed)
        if run_dir.exists():
            complete, reasons, previous = completed_manifest(run_dir, expected)
            if complete:
                status, reusable_manifest = "SKIP_COMPLETED", previous
            elif not args.resume:
                raise FileExistsError(
                    f"INCOMPLETE existing run {run_dir}: {reasons}; pass --resume only after review."
                )
        plans.append((status, expected, reusable_manifest))

    print(f"Theoretical cells: {len(cells)}")
    for status, manifest, _ in plans:
        print(f"{status:14} {run_name(manifest['method'], manifest['fraction'], args.seed)} subset={manifest['subset']['size']} hash={manifest['subset']['sha256'][:12]}")
    print(
        "Plan counts: "
        + ", ".join(
            f"{status}={sum(item[0] == status for item in plans)}"
            for status in ("NEW", "REUSE", "SKIP_COMPLETED")
        )
    )
    if args.dry_run:
        return 0

    if not args.smoke and git_value("status", "--porcelain"):
        raise RuntimeError("Formal runs require a clean committed worktree.")
    root.mkdir(parents=True, exist_ok=True)
    for folder in ("logs", "manifests", "results", "subsets"):
        (root / folder).mkdir(exist_ok=True)
    for status, manifest, reused in plans:
        name = run_name(manifest["method"], manifest["fraction"], args.seed)
        run_dir = root / name
        write_json_atomic(root / "subsets" / f"{name}.json", manifest["subset"])
        if status == "SKIP_COMPLETED":
            continue
        if status == "REUSE":
            assert reused is not None
            registered = dict(reused)
            registered.update(
                {
                    "reused_or_new": "REUSED",
                    "reused_from": reused.get("run_dir") or reused.get("final_result"),
                    "reuse_validation_status": "exact metadata match",
                }
            )
            write_json_atomic(root / "manifests" / f"{name}.json", registered)
            continue
        manifest["status"] = "running"
        manifest["run_dir"] = str(run_dir.resolve())
        resume_checkpoint = None
        if args.resume:
            candidate = run_dir / "checkpoints" / "last_checkpoint.pth"
            if candidate.is_file():
                resume_checkpoint = candidate
            elif run_dir.exists():
                raise RuntimeError(
                    f"INCOMPLETE run has no resumable last checkpoint: {run_dir}"
                )
        command = command_for(args, manifest, root, train_path, val_path, resume_checkpoint)
        manifest["command"] = command
        write_json_atomic(root / "manifests" / f"{name}.json", manifest)
        run_logged(command, root / "logs" / f"{name}.log")
        complete_manifest(manifest, run_dir, root)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="V1 single-seed low-data target transfer runner.")
    root.add_argument("--methods", nargs="+", choices=METHODS)
    root.add_argument("--fractions", nargs="+", type=float)
    root.add_argument("--seed", type=int, default=123)
    root.add_argument("--source-checkpoint")
    root.add_argument("--source-git-commit")
    root.add_argument("--run-root")
    root.add_argument("--history-root", action="append", default=[])
    root.add_argument("--resume", action="store_true")
    root.add_argument("--dry-run", action="store_true")
    root.add_argument("--stage", choices=(1, 2), type=int)
    root.add_argument("--smoke", action="store_true")
    root.add_argument("--python", default=sys.executable)
    root.add_argument("--data-root", default=str(PROJECT.parent))
    root.add_argument("--train-path")
    root.add_argument("--val-path")
    root.add_argument("--device", default="auto")
    root.add_argument("--workers", type=int, default=8)
    root.add_argument("--batch-size", type=int, default=32)
    root.add_argument("--eval-batch-size", type=int, default=64)
    root.add_argument("--epochs", type=int, default=100)
    root.add_argument("--min-epochs", type=int, default=40)
    root.add_argument("--patience", type=int, default=15)
    root.add_argument("--no-early-stopping", action="store_true")
    root.add_argument("--learning-rate", type=float, default=2e-4)
    root.add_argument("--weight-decay", type=float, default=1e-5)
    root.add_argument("--grad-clip", type=float, default=1.0)
    root.add_argument("--hidden", type=int, default=64)
    root.add_argument("--graph-layers", type=int, default=2)
    root.add_argument("--heads", type=int, default=4)
    root.add_argument("--dropout", type=float, default=0.0)
    root.add_argument("--max-train", type=int, default=8)
    root.add_argument("--max-val", type=int, default=4)
    return root


if __name__ == "__main__":
    try:
        raise SystemExit(execute(parser().parse_args()))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
