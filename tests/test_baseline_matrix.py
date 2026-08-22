from __future__ import annotations

import pytest

from lpan.baseline_matrix import (
    FORMAL_BASELINE_MODELS,
    FORMAL_SEEDS,
    build_baseline_plan,
    launcher_text,
)


def test_formal_plan_has_four_models_three_seeds_and_no_test() -> None:
    audit = {
        "models": {
            "lpan_progressive": {"status": "REUSE_VERIFIED"},
            "lpan_l_progressive": {"status": "RERUN_REQUIRED"},
        }
    }
    plan = build_baseline_plan(FORMAL_BASELINE_MODELS, FORMAL_SEEDS, audit=audit)
    assert plan["formal_models"] == list(FORMAL_BASELINE_MODELS)
    assert plan["seeds"] == [123, 456, 789]
    assert "gcn_gru" not in plan["formal_models"]
    assert "gcn_gru" in plan["excluded_models"]
    assert plan["test_split_used"] is False
    tasks = plan["tasks"]
    assert len(tasks) == 12
    assert all(task["test_split_used"] is False for task in tasks)
    assert all(task["data_splits"] == ["train", "validation"] for task in tasks)
    reused = [task for task in tasks if task["model"] == "lpan_progressive"]
    rerun = [task for task in tasks if task["model"] == "lpan_l_progressive"]
    assert all(task["action"] == "validate_existing" for task in reused)
    assert all(task["action"] == "train" for task in rerun)


def test_formal_plan_rejects_gcn_or_noncanonical_seeds() -> None:
    with pytest.raises(ValueError, match="rejected"):
        build_baseline_plan((*FORMAL_BASELINE_MODELS, "gcn_gru"), FORMAL_SEEDS)
    with pytest.raises(ValueError, match="exactly"):
        build_baseline_plan(FORMAL_BASELINE_MODELS, (123,))


def test_launcher_assigns_one_seed_per_gpu_and_serial_worker() -> None:
    text = launcher_text(
        plan_path="runs/plan.json", output_root="runs/formal", data_root="data"
    )
    for gpu, seed in enumerate(FORMAL_SEEDS):
        assert f"CUDA_VISIBLE_DEVICES={gpu}" in text
        assert f"--seed {seed} --device cuda" in text
    assert text.count("baseline-matrix --action run-worker") == 3
    assert text.rstrip().endswith("wait")
