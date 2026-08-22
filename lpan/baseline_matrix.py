from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


FORMAL_BASELINE_MODELS = (
    "lpan_progressive",
    "lpan_l_progressive",
    "edsr_lite",
    "cnn_gru",
)
FORMAL_SEEDS = (123, 456, 789)
EXCLUDED_FORMAL_MODELS = {
    "gcn_gru": "excluded from the formal paper baseline matrix",
}


def _read_audit(path: str | Path | None) -> Mapping[str, object]:
    if path is None or not Path(path).is_file():
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def audit_status(audit: Mapping[str, object], model: str) -> str:
    models = audit.get("models")
    if isinstance(models, Mapping):
        row = models.get(model)
        if isinstance(row, Mapping):
            return str(row.get("status", "RERUN_REQUIRED"))
    return "RERUN_REQUIRED"


def build_baseline_plan(
    models: Sequence[str],
    seeds: Sequence[int],
    *,
    audit: Mapping[str, object] | None = None,
    checkpoint_root: str | Path = "runs",
) -> dict[str, object]:
    requested = tuple(models)
    requested_seeds = tuple(int(seed) for seed in seeds)
    unknown = sorted(set(requested) - set(FORMAL_BASELINE_MODELS))
    if unknown:
        raise ValueError(
            "Formal baseline matrix accepts only "
            f"{FORMAL_BASELINE_MODELS}; rejected {unknown}."
        )
    if requested_seeds != FORMAL_SEEDS:
        raise ValueError(f"Formal seeds must be exactly {FORMAL_SEEDS}.")
    audit = audit or {}
    tasks = []
    for seed in requested_seeds:
        for model in requested:
            if model in {"lpan_progressive", "lpan_l_progressive"}:
                status = audit_status(audit, model)
                action = "validate_existing" if status == "REUSE_VERIFIED" else "train"
                reason = (
                    "semantic audit verified exact reuse; canonical validation required"
                    if action == "validate_existing"
                    else "semantic audit did not verify all six reuse gates"
                )
            else:
                status = "CANONICAL_RETRAIN_REQUIRED"
                action = "train"
                reason = (
                    "old checkpoint semantics are not canonical grouped q0/q3"
                    if model == "edsr_lite"
                    else "formal CNN-GRU requires canonical grouped q0/q3 three-seed training"
                )
            tasks.append(
                {
                    "model": model,
                    "seed": seed,
                    "worker_gpu": requested_seeds.index(seed),
                    "action": action,
                    "audit_status": status,
                    "reason": reason,
                    "data_splits": ["train", "validation"],
                    "test_split_used": False,
                }
            )
    return {
        "formal_models": list(requested),
        "seeds": list(requested_seeds),
        "excluded_models": EXCLUDED_FORMAL_MODELS,
        "semantic_profile": "v3_mobility_q0_q3",
        "complex_layout": "grouped",
        "obs_time_index": [0, 3],
        "query_time": list(range(6)),
        "obs_ris_index": list(range(0, 256, 8)),
        "metric_aggregation": "per-sample linear NMSE -> sample mean -> one dB",
        "checkpoint_root": str(checkpoint_root),
        "tasks": tasks,
        "test_split_used": False,
    }


def launcher_text(
    *,
    plan_path: str | Path,
    output_root: str | Path,
    data_root: str | Path,
) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for gpu, seed in enumerate(FORMAL_SEEDS):
        command = (
            f"CUDA_VISIBLE_DEVICES={gpu} python main.py baseline-matrix "
            f"--action run-worker --seed {seed} --device cuda "
            f"--plan-file '{plan_path}' --data-root '{data_root}' "
            f"--output-root '{output_root}' > '{output_root}/worker_seed{seed}.log' 2>&1 &"
        )
        lines.append(command)
    lines.extend(["", "wait", ""])
    return "\n".join(lines)


def write_plan_and_launcher(
    plan: Mapping[str, object],
    *,
    plan_path: str | Path,
    launcher_path: str | Path,
    output_root: str | Path,
    data_root: str | Path,
) -> None:
    plan_file = Path(plan_path)
    launcher_file = Path(launcher_path)
    plan_file.parent.mkdir(parents=True, exist_ok=True)
    launcher_file.parent.mkdir(parents=True, exist_ok=True)
    plan_file.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    launcher_file.write_text(
        launcher_text(
            plan_path=plan_file,
            output_root=output_root,
            data_root=data_root,
        ),
        encoding="utf-8",
        newline="\n",
    )


def run_worker(
    plan: Mapping[str, object],
    *,
    seed: int,
    device: str,
    data_root: str | Path,
    output_root: str | Path,
    main_path: str | Path,
    epochs: int,
    batch_size: int,
    workers: int,
) -> list[dict[str, object]]:
    if seed not in FORMAL_SEEDS:
        raise ValueError(f"Worker seed must be one of {FORMAL_SEEDS}.")
    raw_tasks = plan.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("Plan is missing tasks.")
    rows = []
    for task in raw_tasks:
        if not isinstance(task, Mapping) or int(task.get("seed", -1)) != seed:
            continue
        model = str(task["model"])
        if task.get("action") == "validate_existing":
            validation_output = (
                Path(output_root)
                / f"mobility_{model}_canonical_seed{seed}"
                / "results"
                / "canonical_validation.json"
            )
            command = [
                sys.executable,
                str(main_path),
                "evaluate-v3-baselines",
                "--checkpoint-root",
                str(plan.get("checkpoint_root", "runs")),
                "--models",
                model,
                "--seeds",
                str(seed),
                "--device",
                device,
                "--data-root",
                str(data_root),
                "--output",
                str(validation_output),
            ]
            subprocess.run(command, check=True)
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "status": "REUSED_AND_CANONICALLY_VALIDATED",
                    "result": str(validation_output),
                }
            )
            continue
        command = [
            sys.executable,
            str(main_path),
            "train",
            "--domain",
            "mobility",
            "--model",
            model,
            "--mode",
            "full",
            "--semantic-profile",
            "v3_mobility_q0_q3",
            "--complex-layout",
            "grouped",
            "--obs-times",
            "0,3",
            "--seed",
            str(seed),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--eval-batch-size",
            str(batch_size),
            "--workers",
            str(workers),
            "--device",
            device,
            "--data-root",
            str(data_root),
            "--output-root",
            str(output_root),
            "--run-name",
            f"mobility_{model}_canonical_seed{seed}",
        ]
        subprocess.run(command, check=True)
        rows.append({"model": model, "seed": seed, "status": "COMPLETED"})
    return rows
