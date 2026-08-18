from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "d51be59183456af81591993ae2458f46153718ca"
SEEDS = (123, 456, 789)
REPAIRED_MAIN_BASELINES = {
    "quasi": ("spatial_gcn",),
    "mobility": ("cnn_gru", "gcn_gru"),
}
TRUSTED_STAGE_D_MODELS = (
    "lpan_progressive",
    "lpan_l_progressive",
    "edsr_lite",
)
INVALIDATED_BASELINES = ("spatial_gcn", "cnn_gru", "gcn_gru")
COMPACT_ABLATIONS = (
    "no_spatial_cross_attention",
    "no_graph",
    "no_temporal_attention",
    "no_domain_adapter",
    "no_coordinate_encoding",
    "no_charbonnier_loss",
    "no_observation_loss",
    "no_temporal_delta_loss",
)
OFFICIAL_RIS = tuple(range(0, 256, 8))


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def current_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _finite_metric(result: dict[str, object]) -> tuple[float, float]:
    linear = float(result["best_validation_nmse_linear"])
    db = float(result["best_validation_nmse_db"])
    if not math.isfinite(linear) or linear <= 0 or not math.isfinite(db):
        raise ValueError("Validation NMSE must be positive and finite.")
    return linear, db


def _load_checkpoint_metadata(path: Path) -> tuple[str, dict[str, object]]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment-specific diagnostic
        raise RuntimeError(
            "PyTorch is required to validate checkpoint model and semantic metadata."
        ) from exc
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint is not a state dictionary: {path}")
    metadata = state.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Checkpoint metadata is missing: {path}")
    return str(state.get("model_name")), metadata


def _validate_semantics(
    metadata: dict[str, object], *, domain: str, seed: int, checkpoint: Path
) -> None:
    expected_obs_time = [0, 1] if domain == "mobility" else [0]
    checks = {
        "domain": (metadata.get("domain"), domain),
        "seed": (metadata.get("seed"), seed),
        "semantic_profile": (metadata.get("semantic_profile"), "official_lpan"),
        "complex_layout": (metadata.get("complex_layout"), "grouped"),
        "obs_time_index": (metadata.get("obs_time_index"), expected_obs_time),
        "obs_ris_index": (metadata.get("obs_ris_index"), list(OFFICIAL_RIS)),
    }
    mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in checks.items()
        if actual != expected
    }
    if mismatches:
        raise ValueError(
            f"Checkpoint semantic metadata mismatch in {checkpoint}: {mismatches}"
        )


def validate_training_run(
    run_dir: Path,
    *,
    model: str,
    domain: str,
    seed: int,
    source_commit: str,
    stage: str,
) -> dict[str, object]:
    final_path = run_dir / "results" / "final_result.json"
    checkpoint = run_dir / "checkpoints" / "best_checkpoint.pth"
    if not final_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(
            f"Completed run requires final_result.json and best checkpoint: {run_dir}"
        )
    final = read_json(final_path)
    if final.get("status") != "validation" or int(final.get("epochs_completed", 0)) <= 0:
        raise ValueError(f"Run is not a completed validation training run: {run_dir}")
    linear, db = _finite_metric(final)
    checkpoint_model, metadata = _load_checkpoint_metadata(checkpoint)
    if checkpoint_model != model:
        raise ValueError(
            f"Checkpoint model {checkpoint_model!r} does not match {model!r}: {checkpoint}"
        )
    _validate_semantics(metadata, domain=domain, seed=seed, checkpoint=checkpoint)
    return {
        "id": f"{domain}_{model}_seed{seed}",
        "status": "reused",
        "stage": stage,
        "source_commit": source_commit,
        "source_run": str(run_dir.resolve()),
        "model": model,
        "domain": domain,
        "seed": seed,
        "checkpoint": str(checkpoint.resolve()),
        "final_result": str(final_path.resolve()),
        "validation_metric": {
            "sample_level_linear_nmse": linear,
            "nmse_db": db,
        },
        "reason": "trusted_unaffected",
        "test_split_used": False,
    }


def validate_stage_b_reference(
    best_result: Path, *, domain: str, source_commit: str
) -> dict[str, object]:
    payload = read_json(best_result)
    if payload.get("status") != "validation_search":
        raise ValueError(f"Stage-B search is not complete: {best_result}")
    if payload.get("domain") != domain or payload.get("test_split_used") is not False:
        raise ValueError(f"Stage-B domain/test-isolation metadata mismatch: {best_result}")
    final = payload.get("final_result")
    if not isinstance(final, dict):
        raise ValueError(f"Stage-B final_result is missing: {best_result}")
    linear, db = _finite_metric(final)
    checkpoint = Path(str(payload.get("best_checkpoint"))).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Stage-B best checkpoint is missing: {checkpoint}")
    checkpoint_model, metadata = _load_checkpoint_metadata(checkpoint)
    if checkpoint_model != "phymeta_stgt":
        raise ValueError(f"Unexpected Stage-B model {checkpoint_model!r}: {checkpoint}")
    _validate_semantics(metadata, domain=domain, seed=123, checkpoint=checkpoint)
    return {
        "id": f"{domain}_phymeta_stgt_seed123",
        "status": "reused",
        "stage": "B/C",
        "source_commit": source_commit,
        "source_run": str(best_result.parent.resolve()),
        "model": "phymeta_stgt",
        "domain": domain,
        "seed": 123,
        "checkpoint": str(checkpoint),
        "final_result": str(best_result.resolve()),
        "validation_metric": {
            "sample_level_linear_nmse": linear,
            "nmse_db": db,
        },
        "reason": "trusted_unaffected",
        "test_split_used": False,
    }


def validate_stage_a_artifact(
    artifact: Path, *, baseline: str, domain: str, source_commit: str
) -> dict[str, object]:
    payload = read_json(artifact)
    expected_name = (
        "LS coarse input + interpolation" if baseline == "interpolation" else "Empirical Ridge"
    )
    if payload.get("baseline") != expected_name or payload.get("domain") != domain:
        raise ValueError(f"Stage-A model/domain mismatch: {artifact}")
    checkpoint: str | None = None
    if baseline == "interpolation":
        checks = {
            "split": (payload.get("split"), "validation"),
            "semantic_profile": (payload.get("semantic_profile"), "official_lpan"),
            "complex_layout": (payload.get("complex_layout"), "grouped"),
            "obs_ris_index": (payload.get("obs_ris_index"), list(OFFICIAL_RIS)),
            "obs_time_index": (
                payload.get("obs_time_index"),
                [0, 1] if domain == "mobility" else [0],
            ),
        }
        if any(actual != expected for actual, expected in checks.values()):
            raise ValueError(f"Stage-A interpolation semantics mismatch: {artifact}")
        linear = float(payload["nmse_linear"]["overall"])  # type: ignore[index]
        db = float(payload["nmse_db"]["overall"])  # type: ignore[index]
    else:
        if "independent_test" in payload:
            raise ValueError(f"Stage-A Ridge artifact unexpectedly contains test: {artifact}")
        candidates = payload.get("validation_candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Stage-A Ridge validation candidates are missing: {artifact}")
        best = min(
            (row for row in candidates if isinstance(row, dict)),
            key=lambda row: float(row["nmse_linear"]["overall"]),  # type: ignore[index]
        )
        linear = float(best["nmse_linear"]["overall"])  # type: ignore[index]
        db = float(best["nmse_db"]["overall"])  # type: ignore[index]
        ridge_path = Path(str(payload.get("model_path"))).expanduser().resolve()
        if not ridge_path.is_file():
            raise FileNotFoundError(f"Stage-A Ridge model is missing: {ridge_path}")
        checkpoint = str(ridge_path)
    if not math.isfinite(linear) or linear <= 0 or not math.isfinite(db):
        raise ValueError(f"Stage-A validation metric is invalid: {artifact}")
    return {
        "id": f"{domain}_{baseline}_validation",
        "status": "reused",
        "stage": "A",
        "source_commit": source_commit,
        "source_run": str(artifact.parent.parent.resolve()),
        "model": baseline,
        "domain": domain,
        "seed": None,
        "checkpoint": checkpoint,
        "final_result": str(artifact.resolve()),
        "validation_metric": {
            "sample_level_linear_nmse": linear,
            "nmse_db": db,
        },
        "reason": "trusted_unaffected",
        "test_split_used": False,
    }


def validate_reuse_root(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    state_path = root / "pipeline_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Formal pipeline_state.json is missing: {state_path}")
    state = read_json(state_path)
    protocol = state.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("commit") != SOURCE_COMMIT:
        raise ValueError(
            f"Reuse root must belong to formal commit {SOURCE_COMMIT}: {state_path}"
        )

    reused: list[dict[str, object]] = []
    for domain in ("quasi", "mobility"):
        for baseline in ("interpolation", "ridge"):
            artifact = root / "stage_a" / f"{domain}_{baseline}_validation.json"
            if not artifact.is_file():
                raise FileNotFoundError(f"Trusted Stage-A artifact is missing: {artifact}")
            reused.append(
                validate_stage_a_artifact(
                    artifact,
                    baseline=baseline,
                    domain=domain,
                    source_commit=SOURCE_COMMIT,
                )
            )

        best_result = (
            root
            / "stage_b"
            / f"{domain}_targeted_boundary_search"
            / "best_result.json"
        )
        reused.append(
            validate_stage_b_reference(
                best_result, domain=domain, source_commit=SOURCE_COMMIT
            )
        )
        for seed in (456, 789):
            reused.append(
                validate_training_run(
                    root / "stage_c" / f"{domain}_phymeta_stgt_seed{seed}",
                    model="phymeta_stgt",
                    domain=domain,
                    seed=seed,
                    source_commit=SOURCE_COMMIT,
                    stage="C",
                )
            )
        for model in TRUSTED_STAGE_D_MODELS:
            for seed in SEEDS:
                reused.append(
                    validate_training_run(
                        root / "stage_d" / f"{domain}_{model}_seed{seed}",
                        model=model,
                        domain=domain,
                        seed=seed,
                        source_commit=SOURCE_COMMIT,
                        stage="D",
                    )
                )

    historical: list[dict[str, object]] = []
    for domain in ("quasi", "mobility"):
        for model in INVALIDATED_BASELINES:
            for seed in SEEDS:
                run_dir = root / "stage_d" / f"{domain}_{model}_seed{seed}"
                if run_dir.exists():
                    historical.append(
                        {
                            "id": f"legacy_{domain}_{model}_seed{seed}",
                            "status": "legacy_invalid_for_final_comparison",
                            "stage": "D",
                            "source_commit": SOURCE_COMMIT,
                            "source_run": str(run_dir.resolve()),
                            "model": model,
                            "domain": domain,
                            "seed": seed,
                            "reason": "baseline_semantics_invalidated",
                            "test_split_used": False,
                        }
                    )
    return reused, historical


class RepairPipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.python = str(Path(args.python).expanduser().resolve())
        self.data_root = str(Path(args.data_root).expanduser().resolve())
        self.reuse_root = Path(args.reuse_formal_root).expanduser().resolve()
        self.root = Path(args.output_root).expanduser().resolve()
        if self.root == self.reuse_root or self.reuse_root in self.root.parents:
            raise ValueError(
                "--output-root must not be the reused formal root or one of its descendants."
            )
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs = self.root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "pipeline_state.json"
        self.state = read_json(self.state_path) if self.state_path.is_file() else {}
        self.state.setdefault("steps", {})
        self.reused, self.historical = validate_reuse_root(self.reuse_root)
        self.source_best = {
            domain: self.reuse_root
            / "stage_b"
            / f"{domain}_targeted_boundary_search"
            / "best_result.json"
            for domain in ("quasi", "mobility")
        }
        self.commit = current_commit()
        tracked_dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--"], cwd=PROJECT, check=False
        ).returncode
        if tracked_dirty and not args.dry_run:
            raise RuntimeError(
                "The repair protocol requires committed tracked changes so manifest "
                "provenance identifies the exact implementation."
            )
        current_protocol = {
            "name": "v1_repair_compact",
            "commit": self.commit,
            "source_commit": SOURCE_COMMIT,
            "reuse_formal_root": str(self.reuse_root),
            "test_split_used": False,
            "stage_f_enabled": False,
            "formal_seeds": list(SEEDS),
            "compact_ablation_seed": 123,
            "compact_ablations": list(COMPACT_ABLATIONS),
            "reference_retrained": False,
        }
        previous_protocol = self.state.get("protocol")
        if isinstance(previous_protocol, dict) and previous_protocol != current_protocol:
            raise RuntimeError(
                "Existing repair output belongs to a different commit or source root; "
                "use a new --output-root."
            )
        self.state.setdefault(
            "protocol",
            current_protocol,
        )
        write_json_atomic(self.state_path, self.state)
        self.write_manifest()

    def save_state(self) -> None:
        write_json_atomic(self.state_path, self.state)

    def run_process(
        self,
        name: str,
        command: list[str],
        expected_outputs: Iterable[Path] = (),
    ) -> None:
        steps = self.state["steps"]
        assert isinstance(steps, dict)
        outputs = [path.resolve() for path in expected_outputs]
        if outputs and all(path.is_file() for path in outputs):
            steps[name] = {
                "status": "completed_existing",
                "command": command,
                "outputs": [str(path) for path in outputs],
            }
            self.save_state()
            return
        if self.args.dry_run:
            print("DRY-RUN:", subprocess.list2cmdline(command), flush=True)
            return
        log_path = self.logs / f"{name}.log"
        steps[name] = {
            "status": "running",
            "command": command,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "log": str(log_path),
        }
        self.save_state()
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        with log_path.open("a", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=PROJECT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            steps[name] = {**steps[name], "status": "failed", "returncode": completed.returncode}
            self.save_state()
            raise RuntimeError(f"Step {name!r} failed; inspect {log_path}.")
        missing = [str(path) for path in outputs if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Step {name!r} did not create: {missing}")
        steps[name] = {
            **steps[name],
            "status": "completed",
            "outputs": [str(path) for path in outputs],
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.save_state()

    def run_main(
        self, name: str, arguments: Iterable[object], outputs: Iterable[Path]
    ) -> None:
        command = [self.python, str(PROJECT / "main.py"), *(str(v) for v in arguments)]
        self.run_process(name, command, outputs)

    def run_pytest(self) -> None:
        marker = self.root / "phase_0_pytest.passed"
        if marker.is_file():
            return
        self.run_process(
            "phase_0_pytest",
            [self.python, "-m", "pytest", "-q"],
        )
        if not self.args.dry_run:
            marker.write_text("pytest passed\n", encoding="utf-8")

    def _models(self) -> dict[str, tuple[str, ...]]:
        models = {key: value for key, value in REPAIRED_MAIN_BASELINES.items()}
        if self.args.include_mobility_spatial_control:
            models["mobility"] = ("spatial_gcn", *models["mobility"])
        return models

    def training_arguments(
        self, *, domain: str, model: str, seed: int, mode: str, run_name: str, root: Path
    ) -> list[object]:
        batch = self.args.quasi_batch_size if domain == "quasi" else 32
        arguments: list[object] = [
            "train",
            "--domain", domain,
            "--model", model,
            "--mode", mode,
            "--seed", seed,
            "--batch-size", batch,
            "--eval-batch-size", 64,
            "--workers", 8,
            "--hidden", 64,
            "--graph-layers", 2,
            "--learning-rate", 2e-4,
            "--weight-decay", 1e-5,
            "--grad-clip", 1.0,
            "--data-root", self.data_root,
            "--device", self.args.device,
            "--run-name", run_name,
            "--output-root", root,
        ]
        if mode == "full":
            arguments.extend(
                [
                    "--epochs", 100,
                    "--min-epochs", 40,
                    "--patience", 15,
                    "--early-stopping",
                ]
            )
        return arguments

    def run_smoke(self) -> None:
        root = self.root / "smoke"
        for domain, models in self._models().items():
            for model in models:
                name = f"{domain}_{model}_fixed_smoke"
                final = root / name / "results" / "final_result.json"
                self.run_main(
                    f"phase_1_{name}",
                    self.training_arguments(
                        domain=domain, model=model, seed=123, mode="smoke", run_name=name, root=root
                    ),
                    [final, root / name / "checkpoints" / "best_checkpoint.pth"],
                )
                if not self.args.dry_run:
                    result = read_json(final)
                    if result.get("status") != "smoke_test":
                        raise ValueError(f"Smoke run did not complete: {final}")
                    _finite_metric(result)

    def _require_smoke(self, domain: str, model: str) -> None:
        if self.args.dry_run:
            return
        final = (
            self.root
            / "smoke"
            / f"{domain}_{model}_fixed_smoke"
            / "results"
            / "final_result.json"
        )
        if not final.is_file():
            raise RuntimeError(
                f"Run --phase smoke before full seed 123; missing {domain}/{model}."
            )
        result = read_json(final)
        if result.get("status") != "smoke_test":
            raise ValueError(f"Smoke result is incomplete: {final}")
        _finite_metric(result)

    def _validate_gate(self, domain: str, model: str) -> None:
        run_dir = self.root / "stage_d_repaired" / f"{domain}_{model}_fixed_seed123"
        result = read_json(run_dir / "results" / "final_result.json")
        _, db = _finite_metric(result)
        if db > self.args.seed123_max_nmse_db:
            raise RuntimeError(
                f"Seed-123 gate failed for {domain}/{model}: {db:.4f} dB exceeds "
                f"{self.args.seed123_max_nmse_db:.4f} dB. Stop and audit before other seeds."
            )

    def run_full_seeds(self, seeds: tuple[int, ...]) -> None:
        root = self.root / "stage_d_repaired"
        for domain, models in self._models().items():
            for model in models:
                if seeds == (123,):
                    self._require_smoke(domain, model)
                if seeds != (123,):
                    self._validate_gate(domain, model)
                for seed in seeds:
                    name = f"{domain}_{model}_fixed_seed{seed}"
                    run_dir = root / name
                    self.run_main(
                        f"phase_{'2' if seed == 123 else '3'}_{name}",
                        self.training_arguments(
                            domain=domain, model=model, seed=seed, mode="full", run_name=name, root=root
                        ),
                        [
                            run_dir / "results" / "final_result.json",
                            run_dir / "checkpoints" / "best_checkpoint.pth",
                        ],
                    )
                    if not self.args.dry_run and seed == 123:
                        self._validate_gate(domain, model)
        self.write_manifest()

    def run_compact_ablation(self) -> None:
        stage = self.root / "stage_e_compact"
        study = "mobility_compact_ablation_seed123"
        summary = stage / study / "summary.json"
        self.run_main(
            "stage_e_compact_seed123",
            [
                "ablate",
                "--domain", "mobility",
                "--mode", "full",
                "--best-result", self.source_best["mobility"],
                "--reuse-full-reference",
                "--variants", ",".join(COMPACT_ABLATIONS),
                "--epochs", 100,
                "--min-epochs", 40,
                "--patience", 15,
                "--early-stopping",
                "--seed", 123,
                "--batch-size", 32,
                "--eval-batch-size", 64,
                "--workers", 8,
                "--data-root", self.data_root,
                "--device", self.args.device,
                "--study-name", study,
                "--output-root", stage,
            ],
            [summary],
        )
        self.write_manifest()

    def _rerun_entries(self) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        root = self.root / "stage_d_repaired"
        for domain, models in self._models().items():
            for model in models:
                for seed in SEEDS:
                    run_dir = root / f"{domain}_{model}_fixed_seed{seed}"
                    if not (run_dir / "results" / "final_result.json").is_file():
                        continue
                    entry = validate_training_run(
                        run_dir,
                        model=model,
                        domain=domain,
                        seed=seed,
                        source_commit=self.commit,
                        stage="D-repaired",
                    )
                    entry["status"] = "rerun"
                    entry["reason"] = "baseline_semantics_fixed"
                    entries.append(entry)
        summary = (
            self.root
            / "stage_e_compact"
            / "mobility_compact_ablation_seed123"
            / "summary.json"
        )
        if summary.is_file():
            payload = read_json(summary)
            if payload.get("reference_retrained") is not False:
                raise ValueError("Compact ablation reference must not be retrained.")
            rows = payload.get("results")
            if not isinstance(rows, list) or {str(row.get("variant")) for row in rows if isinstance(row, dict)} != set(COMPACT_ABLATIONS):
                raise ValueError("Compact ablation summary must contain exactly eight variants.")
            for row in rows:
                assert isinstance(row, dict)
                run_dir = Path(str(row["run_dir"])).resolve()
                entries.append(
                    {
                        "id": f"mobility_ablation_{row['variant']}_seed123",
                        "status": "rerun",
                        "stage": "E-compact",
                        "source_commit": self.commit,
                        "source_run": str(run_dir),
                        "model": "phymeta_stgt",
                        "domain": "mobility",
                        "seed": 123,
                        "variant": row["variant"],
                        "checkpoint": str((run_dir / "checkpoints" / "best_checkpoint.pth").resolve()),
                        "final_result": str(summary.resolve()),
                        "validation_metric": {
                            "sample_level_linear_nmse": row["best_validation_nmse_linear"],
                            "nmse_db": row["best_validation_nmse_db"],
                        },
                        "reason": "compact_one_factor_ablation",
                        "test_split_used": False,
                    }
                )
        return entries

    def write_manifest(self) -> None:
        rerun = self._rerun_entries() if hasattr(self, "commit") else []
        manifest = [*self.reused, *rerun, *self.historical]
        manifest_path = self.root / "result_manifest.json"
        write_json_atomic(
            manifest_path,
            {
                "protocol": "v1_repair_compact",
                "source_formal_root": str(self.reuse_root),
                "source_commit": SOURCE_COMMIT,
                "repair_commit": getattr(self, "commit", None),
                "test_split_used": False,
                "stage_f_executed": False,
                "results": manifest,
            },
        )
        jsonl = self.root / "result_manifest.jsonl"
        temporary = jsonl.with_suffix(".jsonl.tmp")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
            encoding="utf-8",
        )
        os.replace(temporary, jsonl)
        write_json_atomic(
            self.root / "result_classification.json",
            {
                "reused_trusted_formal_result": [row["id"] for row in self.reused],
                "repaired_and_rerun_result": [row["id"] for row in rerun],
                "historical_invalidated_result": [row["id"] for row in self.historical],
            },
        )

    def run(self) -> None:
        phase = self.args.phase
        if phase in {"pytest", "all"}:
            self.run_pytest()
        if phase in {"smoke", "all"}:
            self.run_smoke()
        if phase in {"seed123", "all"}:
            self.run_full_seeds((123,))
        if phase in {"remaining", "all"}:
            self.run_full_seeds((456, 789))
        if phase in {"ablation", "all"}:
            self.run_compact_ablation()
        self.write_manifest()
        self.state["last_phase"] = phase
        self.state["test_split_used"] = False
        self.save_state()


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description=(
            "Run the validation-only V1 repaired-baseline and compact-ablation "
            "protocol while referencing trusted formal artifacts read-only."
        )
    )
    command.add_argument("--reuse-formal-root", required=True, type=Path)
    command.add_argument("--output-root", default="runs/v1_repair_compact")
    command.add_argument("--data-root", default=os.environ.get("LPAN_DATA_ROOT", str(PROJECT / "data")))
    command.add_argument("--device", default="cuda")
    command.add_argument("--python", default=sys.executable)
    command.add_argument("--quasi-batch-size", type=int, default=32)
    command.add_argument("--seed123-max-nmse-db", type=float, default=0.0)
    command.add_argument("--include-mobility-spatial-control", action="store_true")
    command.add_argument("--dry-run", action="store_true")
    command.add_argument(
        "--phase",
        choices=("manifest", "pytest", "smoke", "seed123", "remaining", "ablation", "all"),
        default="manifest",
    )
    return command


def main() -> None:
    RepairPipeline(parser().parse_args()).run()


if __name__ == "__main__":
    main()
