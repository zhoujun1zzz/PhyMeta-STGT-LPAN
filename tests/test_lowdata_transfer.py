from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from lpan.complexity import canonical_batch
from lpan.engine import configure_adaptation, train_epoch
from lpan.models import build_model
from lpan.objectives import LossWeights
from lpan.transfer import (
    ADAPTATION_MODES,
    deterministic_subset_indices,
    subset_index_hash,
)
from main import validate_training_request
from scripts.run_v1_lowdata_transfer import command_for
from scripts.run_v1_lowdata_transfer import parser as transfer_parser
from scripts.run_v1_lowdata_transfer import select_cells, validate_reuse
from scripts.summarize_lowdata_transfer import manifest_rows


def small_model() -> torch.nn.Module:
    return build_model(
        "phymeta_stgt",
        domain="mobility",
        hidden=16,
        graph_layers=1,
        heads=4,
        dropout=0.2,
    )


def test_scratch_and_transfer_checkpoint_guards() -> None:
    validate_training_request("phymeta_stgt", "scratch", None, None)
    scratch = small_model()
    report = configure_adaptation(scratch, "scratch")
    assert report["trainable_parameters"] == report["total_parameters"]
    assert all(parameter.requires_grad for parameter in scratch.parameters())
    with pytest.raises(ValueError, match="random initialization"):
        validate_training_request(
            "phymeta_stgt", "scratch", "source.pth", None
        )
    for method in ADAPTATION_MODES[1:]:
        with pytest.raises(ValueError, match="pretrained checkpoint"):
            validate_training_request("phymeta_stgt", method, None, None)


@pytest.mark.parametrize(
    ("method", "expected_modules"),
    [
        ("domain_adapter_only", {"domain_embedding"}),
        ("adapter_head", {"domain_embedding", "decoder"}),
    ],
)
def test_narrow_policies_define_exact_optimizer_membership(
    method: str, expected_modules: set[str]
) -> None:
    model = small_model()
    report = configure_adaptation(model, method)
    assert set(report["trainable_module_names"]) == expected_modules
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad)
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    assert optimizer_ids == expected_ids


def test_frozen_spatial_weights_do_not_change_and_modules_stay_eval() -> None:
    model = small_model()
    report = configure_adaptation(model, "frozen_spatial")
    assert "channel_encoder" in report["frozen_module_names"]
    assert "decoder" in report["trainable_module_names"]
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    batch = canonical_batch("mobility", batch_size=1, device=torch.device("cpu"))
    train_epoch(
        model,
        [batch],
        optimizer,
        torch.device("cpu"),
        LossWeights(),
    )
    after = dict(model.named_parameters())
    assert all(torch.equal(value, after[name]) for name, value in before.items())
    assert model.training is True
    assert model.channel_encoder.training is False
    assert model.graph_layers.training is False


def test_fraction_indices_are_nested_and_method_independent() -> None:
    fractions = (0.01, 0.05, 0.10, 0.20, 1.0)
    indices = {
        fraction: deterministic_subset_indices(1000, fraction, 123)
        for fraction in fractions
    }
    for smaller, larger in zip(fractions, fractions[1:]):
        assert set(indices[smaller]).issubset(set(indices[larger]))
    hashes_by_method = {
        method: subset_index_hash(deterministic_subset_indices(1000, 0.05, 123))
        for method in ADAPTATION_MODES
    }
    assert len(set(hashes_by_method.values())) == 1


def exact_manifest(tmp_path: Path) -> dict[str, object]:
    checkpoint = tmp_path / "best.pth"
    result = tmp_path / "result.json"
    checkpoint.write_bytes(b"checkpoint")
    result.write_text("{}", encoding="utf-8")
    return {
        "status": "completed",
        "method": "adapter_head",
        "fraction": 0.05,
        "seed": 123,
        "git": {"commit": "abc"},
        "source": {
            "domain": "quasi",
            "checkpoint_sha256": "source-hash",
        },
        "target": {
            "domain": "mobility",
            "train_identity": {"size_bytes": 100},
            "validation_identity": {"size_bytes": 20},
            "semantic_contract": {
                "domain": "mobility",
                "semantic_profile": "official_lpan",
                "complex_layout": "interleaved",
                "obs_time_index": [1, 4],
                "query_time": list(range(6)),
                "obs_ris_index": list(range(0, 256, 8)),
            },
            "semantic_fingerprint": "canonical-semantic-hash",
        },
        "subset": {"size": 10, "sha256": "subset-hash"},
        "model": "phymeta_stgt",
        "model_config": {"hidden": 64},
        "optimizer": {"name": "AdamW"},
        "scheduler": {"name": "constant"},
        "training_budget": {"epochs": 100},
        "validation_protocol": {"split": "validation"},
        "metric_definition": "sample-level linear NMSE, reported in dB",
        "test_split_used": False,
        "best_checkpoint": str(checkpoint),
        "final_result": str(result),
    }


def test_history_reuse_requires_exact_metadata_and_provenance(tmp_path: Path) -> None:
    expected = exact_manifest(tmp_path)
    candidate = copy.deepcopy(expected)
    reusable, reasons = validate_reuse(expected, candidate)
    assert reusable and not reasons
    candidate["subset"]["sha256"] = "different"  # type: ignore[index]
    reusable, reasons = validate_reuse(expected, candidate)
    assert reusable is False
    assert "subset.sha256" in reasons
    legacy = copy.deepcopy(expected)
    del legacy["target"]["semantic_contract"]  # type: ignore[index]
    del legacy["target"]["semantic_fingerprint"]  # type: ignore[index]
    reusable, reasons = validate_reuse(expected, legacy)
    assert reusable is False
    assert "target.semantic_contract" in reasons
    assert "target.semantic_fingerprint" in reasons


def test_runner_stage_counts_and_never_builds_test_cells() -> None:
    parser = transfer_parser()
    all_args = parser.parse_args(["--dry-run"])
    stage1 = parser.parse_args(["--dry-run", "--stage", "1"])
    stage2 = parser.parse_args(["--dry-run", "--stage", "2"])
    assert len(select_cells(all_args)) == 25
    assert len(select_cells(stage1)) == 5
    assert len(select_cells(stage2)) == 20
    serialized = json.dumps(select_cells(all_args))
    assert "test" not in serialized.lower()
    command = command_for(
        all_args,
        {
            "method": "adapter_head",
            "fraction": 0.05,
            "source": {"checkpoint": "source.pth"},
        },
        Path("runs"),
        Path("mobility_train.mat"),
        Path("mobility_validation.mat"),
        None,
    )
    assert "evaluate" not in command
    assert "--split" not in command


def test_summary_rejects_trainable_ratio_drift(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    base = {
        "status": "completed",
        "method": "adapter_head",
        "seed": 123,
        "best_validation_nmse_db": -1.0,
        "adaptation_minutes": 1.0,
        "reused_or_new": "NEW",
    }
    for index, (fraction, ratio) in enumerate(((0.01, 13.0), (0.05, 14.0))):
        (manifests / f"run{index}.json").write_text(
            json.dumps(
                {
                    **base,
                    "fraction": fraction,
                    "trainable_ratio_percent": ratio,
                }
            ),
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="Trainable ratio changed"):
        manifest_rows(tmp_path)
