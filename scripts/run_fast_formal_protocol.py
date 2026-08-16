from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable


# Freeze CPU-side thread fan-out before any child process imports NumPy/PyTorch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

PROJECT = Path(__file__).resolve().parents[1]
SEEDS = (123, 456, 789)
MOBILITY_BATCH = 32
EVAL_BATCH = 64
WORKERS = 8
MAX_EPOCHS = 100
MIN_EPOCHS = 40
PATIENCE = 15
QUASI_BATCH_CANDIDATES = (16, 32, 64, 128)
QUASI_MODELS = (
    "lpan_progressive",
    "lpan_l_progressive",
    "edsr_lite",
    "spatial_gcn",
)
MOBILITY_MODELS = QUASI_MODELS + ("cnn_gru", "gcn_gru")
ALL_ABLATIONS = (
    "none",
    "no_spatial_cross_attention",
    "no_graph",
    "no_temporal_attention",
    "no_domain_adapter",
    "no_coordinate_encoding",
    "nmse_only",
    "no_charbonnier_loss",
    "no_observation_loss",
    "no_temporal_delta_loss",
)
STRUCTURAL_ABLATIONS = ALL_ABLATIONS[:6]


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


class FormalPipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.python = str(Path(args.python).expanduser().resolve())
        self.data_root = str(Path(args.data_root).expanduser().resolve())
        self.root = Path(args.output_root).expanduser().resolve()
        self.logs = self.root / "logs"
        self.state_path = self.root / "pipeline_state.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.state: dict[str, object] = (
            read_json(self.state_path) if self.state_path.is_file() else {}
        )
        self.state.setdefault("steps", {})
        current_protocol = self.protocol_record()
        previous_protocol = self.state.get("protocol")
        if isinstance(previous_protocol, dict):
            if previous_protocol.get("commit") != current_protocol["commit"]:
                raise RuntimeError(
                    "Pipeline state belongs to a different commit; use a new "
                    "--output-root rather than mixing formal protocols."
                )
        else:
            self.state["protocol"] = current_protocol
        write_json_atomic(self.state_path, self.state)

    def protocol_record(self) -> dict[str, object]:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if dirty:
            raise RuntimeError(
                "Formal pipeline requires a clean committed worktree."
            )
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {
            "commit": commit,
            "fp32": True,
            "amp_enabled": False,
            "test_used_before_stage_f": False,
            "mobility_train_batch": MOBILITY_BATCH,
            "eval_batch": EVAL_BATCH,
            "workers": WORKERS,
            "omp_num_threads": 1,
            "mkl_num_threads": 1,
            "max_epochs": MAX_EPOCHS,
            "min_epochs": MIN_EPOCHS,
            "patience": PATIENCE,
            "stage_b": {
                "protocol": "targeted_boundary",
                "capacity_candidates": [96, 128, 160],
                "capacity_epochs": 20,
                "learning_rate_candidates": [5e-4, 8e-4, 1e-3],
                "learning_rate_epochs": 40,
                "final_max_epochs": MAX_EPOCHS,
                "training_seed": 123,
            },
            "formal_seeds": list(SEEDS),
            "joint_training_in_main_protocol": False,
        }

    def save_state(self) -> None:
        write_json_atomic(self.state_path, self.state)

    def run_main(
        self,
        name: str,
        arguments: Iterable[object],
        expected_outputs: Iterable[Path],
    ) -> None:
        steps = self.state["steps"]
        assert isinstance(steps, dict)
        outputs = [Path(path).resolve() for path in expected_outputs]
        previous = steps.get(name)
        if (
            isinstance(previous, dict)
            and previous.get("status") == "completed"
            and all(path.is_file() for path in outputs)
        ):
            print(f"[skip] {name}", flush=True)
            return
        command = [
            self.python,
            "-u",
            str(PROJECT / "main.py"),
            *(str(value) for value in arguments),
        ]
        log_path = self.logs / f"{name}.log"
        steps[name] = {
            "status": "running",
            "command": command,
            "log": str(log_path),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.save_state()
        print(f"[run] {name}", flush=True)
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\nCOMMAND: " + " ".join(command) + "\n")
            handle.flush()
            completed = subprocess.run(
                command,
                cwd=PROJECT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            steps[name] = {
                **steps[name],
                "status": "failed",
                "returncode": completed.returncode,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.save_state()
            raise RuntimeError(
                f"Pipeline step {name!r} failed; inspect {log_path}."
            )
        missing = [str(path) for path in outputs if not path.is_file()]
        if missing:
            steps[name] = {
                **steps[name],
                "status": "failed",
                "error": f"missing expected outputs: {missing}",
            }
            self.save_state()
            raise FileNotFoundError(
                f"Step {name!r} completed without expected outputs: {missing}"
            )
        steps[name] = {
            **steps[name],
            "status": "completed",
            "outputs": [str(path) for path in outputs],
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.save_state()

    def common(self) -> list[object]:
        return [
            "--data-root",
            self.data_root,
            "--device",
            self.args.device,
            "--workers",
            WORKERS,
        ]

    def benchmark_batches(self) -> int:
        output = self.root / "benchmark" / "quasi_batch.json"
        self.run_main(
            "benchmark_quasi_batch",
            [
                "benchmark-batch",
                "--domain",
                "quasi",
                "--candidates",
                ",".join(map(str, QUASI_BATCH_CANDIDATES)),
                "--max-samples",
                1024,
                "--output",
                output,
                *self.common(),
            ],
            [output],
        )
        selected = int(read_json(output)["selected_batch_size"])
        protocol = self.state["protocol"]
        assert isinstance(protocol, dict)
        protocol["quasi_train_batch"] = selected
        self.save_state()
        return selected

    def stage_a(self) -> None:
        stage = self.root / "stage_a"
        for domain in ("quasi", "mobility"):
            complexity = stage / f"{domain}_complexity.json"
            models = (
                "lpan_progressive,lpan_l_progressive,edsr_lite,"
                "spatial_gcn,phymeta_stgt"
                if domain == "quasi"
                else (
                    "lpan_progressive,lpan_l_progressive,edsr_lite,"
                    "spatial_gcn,cnn_gru,gcn_gru,phymeta_stgt"
                )
            )
            self.run_main(
                f"stage_a_{domain}_complexity",
                [
                    "profile",
                    "--domain",
                    domain,
                    "--models",
                    models,
                    "--batch-size",
                    1,
                    "--device",
                    "cpu",
                    "--output",
                    complexity,
                ],
                [complexity, complexity.with_suffix(".csv")],
            )
            interpolation = stage / f"{domain}_interpolation_validation.json"
            self.run_main(
                f"stage_a_{domain}_interpolation_validation",
                [
                    "interpolate",
                    "--domain",
                    domain,
                    "--split",
                    "validation",
                    "--output",
                    interpolation,
                    "--data-root",
                    self.data_root,
                    "--workers",
                    WORKERS,
                    "--batch-size",
                    EVAL_BATCH,
                ],
                [interpolation],
            )
            ridge = stage / f"{domain}_ridge_validation.json"
            self.run_main(
                f"stage_a_{domain}_ridge_validation",
                [
                    "ridge",
                    "--domain",
                    domain,
                    "--output",
                    ridge,
                    "--data-root",
                    self.data_root,
                    "--workers",
                    WORKERS,
                    "--batch-size",
                    EVAL_BATCH,
                ],
                [ridge],
            )

    def stage_b(self, quasi_batch: int) -> dict[str, Path]:
        results: dict[str, Path] = {}
        for domain, batch in (("quasi", quasi_batch), ("mobility", MOBILITY_BATCH)):
            study_root = self.root / "stage_b"
            study_name = f"{domain}_targeted_boundary_search"
            best = study_root / study_name / "best_result.json"
            self.run_main(
                f"stage_b_{domain}",
                [
                    "tune",
                    "--domain",
                    domain,
                    "--mode",
                    "full",
                    "--tuning-protocol",
                    "targeted_boundary",
                    "--epochs",
                    MAX_EPOCHS,
                    "--min-epochs",
                    MIN_EPOCHS,
                    "--patience",
                    PATIENCE,
                    "--early-stopping",
                    "--seed",
                    123,
                    "--batch-size",
                    batch,
                    "--eval-batch-size",
                    EVAL_BATCH,
                    "--study-name",
                    study_name,
                    "--output-root",
                    study_root,
                    *self.common(),
                ],
                [best],
            )
            results[domain] = best
        return results

    @staticmethod
    def hyperparameter_arguments(best_result: Path) -> list[object]:
        values = read_json(best_result)["best_hyperparameters"]
        assert isinstance(values, dict)
        return [
            "--hidden",
            values["hidden"],
            "--graph-layers",
            values["graph_layers"],
            "--heads",
            values["heads"],
            "--dropout",
            values["dropout"],
            "--learning-rate",
            values["learning_rate"],
            "--weight-decay",
            values["weight_decay"],
        ]

    def training_arguments(
        self,
        *,
        domain: str,
        model: str,
        seed: int,
        batch: int,
        output_root: Path,
        run_name: str,
    ) -> list[object]:
        return [
            "train",
            "--domain",
            domain,
            "--model",
            model,
            "--mode",
            "full",
            "--epochs",
            MAX_EPOCHS,
            "--min-epochs",
            MIN_EPOCHS,
            "--patience",
            PATIENCE,
            "--early-stopping",
            "--seed",
            seed,
            "--batch-size",
            batch,
            "--eval-batch-size",
            EVAL_BATCH,
            "--run-name",
            run_name,
            "--output-root",
            output_root,
            *self.common(),
        ]

    def stage_c(
        self,
        best_results: dict[str, Path],
        quasi_batch: int,
    ) -> list[dict[str, object]]:
        manifest: list[dict[str, object]] = []
        for domain, batch in (("quasi", quasi_batch), ("mobility", MOBILITY_BATCH)):
            best_payload = read_json(best_results[domain])
            manifest.append(
                {
                    "id": f"{domain}_phymeta_stgt_seed123",
                    "stage": "C",
                    "source_stage": "B",
                    "domain": domain,
                    "model": "phymeta_stgt",
                    "variant": "none",
                    "seed": 123,
                    "checkpoint": best_payload["best_checkpoint"],
                }
            )
            for seed in (456, 789):
                run_name = f"{domain}_phymeta_stgt_seed{seed}"
                run_root = self.root / "stage_c"
                final = run_root / run_name / "results" / "final_result.json"
                self.run_main(
                    f"stage_c_{run_name}",
                    [
                        *self.training_arguments(
                            domain=domain,
                            model="phymeta_stgt",
                            seed=seed,
                            batch=batch,
                            output_root=run_root,
                            run_name=run_name,
                        ),
                        *self.hyperparameter_arguments(best_results[domain]),
                    ],
                    [
                        final,
                        run_root
                        / run_name
                        / "checkpoints"
                        / "best_checkpoint.pth",
                    ],
                )
                manifest.append(
                    {
                        "id": run_name,
                        "stage": "C",
                        "domain": domain,
                        "model": "phymeta_stgt",
                        "variant": "none",
                        "seed": seed,
                        "checkpoint": str(
                            run_root
                            / run_name
                            / "checkpoints"
                            / "best_checkpoint.pth"
                        ),
                    }
                )
        return manifest

    def stage_d(self, quasi_batch: int) -> list[dict[str, object]]:
        manifest: list[dict[str, object]] = []
        mobility_models = MOBILITY_MODELS
        if self.args.exclude_mobility_adapted_lpan:
            mobility_models = tuple(
                model for model in mobility_models if model != "lpan_progressive"
            )
        for domain, models, batch in (
            ("quasi", QUASI_MODELS, quasi_batch),
            ("mobility", mobility_models, MOBILITY_BATCH),
        ):
            for model in models:
                for seed in SEEDS:
                    run_name = f"{domain}_{model}_seed{seed}"
                    run_root = self.root / "stage_d"
                    final = run_root / run_name / "results" / "final_result.json"
                    self.run_main(
                        f"stage_d_{run_name}",
                        self.training_arguments(
                            domain=domain,
                            model=model,
                            seed=seed,
                            batch=batch,
                            output_root=run_root,
                            run_name=run_name,
                        ),
                        [
                            final,
                            run_root
                            / run_name
                            / "checkpoints"
                            / "best_checkpoint.pth",
                        ],
                    )
                    manifest.append(
                        {
                            "id": run_name,
                            "stage": "D",
                            "domain": domain,
                            "model": model,
                            "variant": "none",
                            "seed": seed,
                            "checkpoint": str(
                                run_root
                                / run_name
                                / "checkpoints"
                                / "best_checkpoint.pth"
                            ),
                        }
                    )
        return manifest

    def stage_e(self, mobility_best: Path) -> list[dict[str, object]]:
        manifest: list[dict[str, object]] = []
        for seed, variants in (
            (123, ALL_ABLATIONS),
            (456, STRUCTURAL_ABLATIONS),
            (789, STRUCTURAL_ABLATIONS),
        ):
            study_name = f"mobility_ablation_seed{seed}"
            stage_root = self.root / "stage_e"
            summary = stage_root / study_name / "summary.json"
            self.run_main(
                f"stage_e_{study_name}",
                [
                    "ablate",
                    "--domain",
                    "mobility",
                    "--mode",
                    "full",
                    "--best-result",
                    mobility_best,
                    "--variants",
                    ",".join(variants),
                    "--epochs",
                    MAX_EPOCHS,
                    "--min-epochs",
                    MIN_EPOCHS,
                    "--patience",
                    PATIENCE,
                    "--early-stopping",
                    "--seed",
                    seed,
                    "--batch-size",
                    MOBILITY_BATCH,
                    "--eval-batch-size",
                    EVAL_BATCH,
                    "--study-name",
                    study_name,
                    "--output-root",
                    stage_root,
                    *self.common(),
                ],
                [summary],
            )
            payload = read_json(summary)
            rows = payload["results"]
            assert isinstance(rows, list)
            for row in rows:
                assert isinstance(row, dict)
                variant = str(row["variant"])
                manifest.append(
                    {
                        "id": f"mobility_ablation_{variant}_seed{seed}",
                        "stage": "E",
                        "domain": "mobility",
                        "model": "phymeta_stgt",
                        "variant": variant,
                        "seed": seed,
                        "checkpoint": str(
                            Path(str(row["run_dir"]))
                            / "checkpoints"
                            / "best_checkpoint.pth"
                        ),
                    }
                )
        return manifest

    def stage_f(self, manifest: list[dict[str, object]]) -> None:
        stage = self.root / "stage_f"
        frozen = stage / "frozen_model_manifest.json"
        ridge_regularization = {
            domain: float(
                read_json(
                    self.root
                    / "stage_a"
                    / f"{domain}_ridge_validation.json"
                )["best_regularization"]
            )
            for domain in ("quasi", "mobility")
        }
        write_json_atomic(
            frozen,
            {
                "frozen_before_test": True,
                "test_used_for_selection": False,
                "models": manifest,
                "interpolation": {"spatial": "linear", "temporal": "linear"},
                "ridge_regularization": ridge_regularization,
                "complexity_reports": {
                    domain: str(
                        self.root / "stage_a" / f"{domain}_complexity.json"
                    )
                    for domain in ("quasi", "mobility")
                },
            },
        )
        evaluation_rows: list[dict[str, object]] = []
        for entry in manifest:
            identifier = str(entry["id"])
            domain = str(entry["domain"])
            output = stage / "evaluations" / f"{identifier}.json"
            arguments: list[object] = [
                "evaluate",
                "--checkpoint",
                entry["checkpoint"],
                "--domain",
                domain,
                "--split",
                "test",
                "--data-root",
                self.data_root,
                "--device",
                self.args.device,
                "--workers",
                WORKERS,
                "--batch-size",
                EVAL_BATCH,
                "--output",
                output,
            ]
            expected = [output]
            if domain == "mobility":
                arguments.append("--per-snr")
                expected.append(output.with_suffix(".per_snr.csv"))
            self.run_main(f"stage_f_eval_{identifier}", arguments, expected)
            result = read_json(output)
            nmse_db = result["nmse_db"]
            nmse_linear = result["nmse_linear"]
            assert isinstance(nmse_db, dict) and isinstance(nmse_linear, dict)
            evaluation_rows.append(
                {
                    **entry,
                    "result": str(output),
                    "nmse_linear": float(nmse_linear["overall"]),
                    "nmse_db": float(nmse_db["overall"]),
                }
            )
        for domain in ("quasi", "mobility"):
            interpolation = stage / f"{domain}_interpolation_test.json"
            self.run_main(
                f"stage_f_{domain}_interpolation_test",
                [
                    "interpolate",
                    "--domain",
                    domain,
                    "--split",
                    "test",
                    "--data-root",
                    self.data_root,
                    "--workers",
                    WORKERS,
                    "--batch-size",
                    EVAL_BATCH,
                    "--output",
                    interpolation,
                ],
                [interpolation],
            )
            ridge = stage / f"{domain}_ridge_test.json"
            self.run_main(
                f"stage_f_{domain}_ridge_test",
                [
                    "ridge",
                    "--domain",
                    domain,
                    "--test",
                    "--lambdas",
                    ridge_regularization[domain],
                    "--data-root",
                    self.data_root,
                    "--workers",
                    WORKERS,
                    "--batch-size",
                    EVAL_BATCH,
                    "--output",
                    ridge,
                ],
                [ridge],
            )
        groups: defaultdict[
            tuple[str, str, str, str], list[dict[str, object]]
        ] = (
            defaultdict(list)
        )
        for row in evaluation_rows:
            groups[
                (
                    str(row["stage"]),
                    str(row["domain"]),
                    str(row["model"]),
                    str(row["variant"]),
                )
            ].append(row)
        aggregate = []
        for key, rows in sorted(groups.items()):
            linear = [float(row["nmse_linear"]) for row in rows]
            db = [float(row["nmse_db"]) for row in rows]
            mean_linear = statistics.fmean(linear)
            aggregate.append(
                {
                    "stage": key[0],
                    "domain": key[1],
                    "model": key[2],
                    "variant": key[3],
                    "seeds": [int(row["seed"]) for row in rows],
                    "runs": len(rows),
                    "mean_nmse_linear": mean_linear,
                    "std_nmse_linear": (
                        statistics.stdev(linear) if len(linear) > 1 else 0.0
                    ),
                    "db_of_mean_linear_nmse": 10 * math.log10(mean_linear),
                    "mean_per_seed_nmse_db": statistics.fmean(db),
                    "std_per_seed_nmse_db": (
                        statistics.stdev(db) if len(db) > 1 else 0.0
                    ),
                }
            )
        write_json_atomic(stage / "independent_test_runs.json", evaluation_rows)
        write_json_atomic(stage / "mean_std_summary.json", aggregate)

    def run(self) -> None:
        quasi_batch = self.benchmark_batches()
        self.stage_a()
        best_results = self.stage_b(quasi_batch)
        manifest = self.stage_c(best_results, quasi_batch)
        manifest.extend(self.stage_d(quasi_batch))
        manifest.extend(self.stage_e(best_results["mobility"]))
        self.stage_f(manifest)
        self.state["status"] = "completed"
        self.state["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save_state()
        print(f"Formal pipeline completed: {self.root}", flush=True)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description=(
            "Run the frozen fast formal LPAN protocol. Audit and semantic "
            "verification are intentionally not repeated."
        )
    )
    command.add_argument(
        "--data-root",
        default=os.environ.get("LPAN_DATA_ROOT", str(PROJECT / "data")),
    )
    command.add_argument("--output-root", default="runs/formal_fast")
    command.add_argument("--device", default="cuda")
    command.add_argument("--python", default=sys.executable)
    command.add_argument(
        "--exclude-mobility-adapted-lpan",
        action="store_true",
        help="Skip adapted Mobility LPAN; official Mobility LPAN-L remains enabled.",
    )
    return command


def main() -> None:
    FormalPipeline(parser().parse_args()).run()


if __name__ == "__main__":
    main()
