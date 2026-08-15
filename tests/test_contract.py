from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as experiment_main

from lpan.data import LPANH5Dataset, default_observed_ris_indices
from lpan.complexity import canonical_batch, profile_model_complexity
from lpan.engine import (
    capture_rng_state,
    configure_adaptation,
    read_history,
    restore_rng_state,
    save_checkpoint,
    write_history,
)
from lpan.metrics import MetricAccumulator
from lpan.models import (
    build_model,
    expand_observations_to_grid,
    interpolation_baseline,
)
from lpan.objectives import (
    LossWeights,
    charbonnier,
    combined_loss,
    observation_consistency,
)
from lpan.paths import dataset_candidates, resolve_dataset_path
from lpan.ridge import RidgeStatistics
from lpan.studies import (
    ARCHITECTURE_ABLATIONS,
    ablation_metadata,
    ablated_loss_weights,
    hyperparameter_candidates,
)
from main import (
    _load_best_hyperparameters,
    ablate_command,
    audit_command,
    early_stopping_progress,
    early_stopping_step,
    main as cli_main,
    load_pretrained_checkpoint,
    parser as experiment_parser,
    per_snr_evaluation,
    resolve_evaluation_semantics,
    select_fastest_batch,
    tune_command,
    validate_training_request,
)
from lpan.graph import ris_index_to_grid
from scripts.verify_data_semantics import verify_mobility


def write_mat(path: Path, domain: str, samples: int = 3) -> None:
    channels, targets = ((2, 2) if domain == "quasi" else (4, 12))
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "Yd",
            data=np.random.default_rng(1).normal(
                size=(channels, 32, 64, samples)
            ).astype(np.float32),
        )
        handle.create_dataset(
            "Hd",
            data=np.random.default_rng(2).normal(
                size=(targets, 256, 64, samples)
            ).astype(np.float32),
        )


def collate_sample(domain: str) -> dict[str, torch.Tensor]:
    t, q, domain_id = ((1, 1, 0) if domain == "quasi" else (2, 6, 1))
    obs_times = [0] if domain == "quasi" else [0, 1]
    return {
        "obs_h": torch.randn(1, t, 32, 64, 2),
        "target_h": torch.randn(1, q, 256, 64, 2),
        "obs_ris_index": torch.arange(0, 256, 8).unsqueeze(0),
        "obs_time_index": torch.tensor(obs_times).unsqueeze(0),
        "query_time": torch.arange(q).unsqueeze(0),
        "domain_id": torch.tensor([domain_id]),
        "observation_mask": torch.ones(1, t, 32, dtype=torch.bool),
        "sample_index": torch.tensor([0]),
    }


def test_hdf5_contract(tmp_path: Path) -> None:
    for domain, expected in (
        ("quasi", ((1, 32, 64, 2), (1, 256, 64, 2))),
        ("mobility", ((2, 32, 64, 2), (6, 256, 64, 2))),
    ):
        path = tmp_path / f"{domain}.mat"
        write_mat(path, domain)
        dataset = LPANH5Dataset(path, domain, "validation")
        sample = dataset[0]
        assert tuple(sample["obs_h"].shape) == expected[0]
        assert tuple(sample["target_h"].shape) == expected[1]


def test_partially_constructed_dataset_can_close_without_warning(
    tmp_path: Path,
) -> None:
    dataset = LPANH5Dataset.__new__(LPANH5Dataset)
    with pytest.raises(ValueError, match="domain must be one of"):
        dataset.__init__(tmp_path / "missing.mat", "invalid", "train")
    dataset.close()


def test_verified_lpan_ris_mapping() -> None:
    indices = default_observed_ris_indices()
    assert indices == tuple(range(0, 256, 8))
    coordinates = [ris_index_to_grid(index) for index in indices]
    assert coordinates == [
        (row, col) for row in range(16) for col in (0, 8)
    ]


def test_portable_data_root_resolution(tmp_path: Path) -> None:
    candidate = dataset_candidates(tmp_path, "quasi", "validation")[0]
    candidate.parent.mkdir(parents=True)
    write_mat(candidate, "quasi")
    assert resolve_dataset_path(tmp_path, "quasi", "validation") == candidate


def test_fraction_subsets_are_nested(tmp_path: Path) -> None:
    path = tmp_path / "nested.mat"
    write_mat(path, "quasi", samples=40)
    small = LPANH5Dataset(
        path, "quasi", "validation", fraction=0.1, subset_seed=7
    )
    large = LPANH5Dataset(
        path, "quasi", "validation", fraction=0.5, subset_seed=7
    )
    assert set(small.indices) < set(large.indices)


def test_interleaved_complex_layout(tmp_path: Path) -> None:
    path = tmp_path / "interleaved.mat"
    y = np.zeros((4, 32, 64, 1), dtype=np.float32)
    h = np.zeros((12, 256, 64, 1), dtype=np.float32)
    y[:, 0, 0, 0] = [10, 20, 30, 40]
    h[:, 0, 0, 0] = np.arange(12)
    with h5py.File(path, "w") as handle:
        handle["Yd"] = y
        handle["Hd"] = h
    sample = LPANH5Dataset(
        path,
        "mobility",
        "validation",
        complex_layout="interleaved",
        semantic_profile="custom",
    )[0]
    assert sample["obs_h"][0, 0, 0].tolist() == [10, 20]
    assert sample["obs_h"][1, 0, 0].tolist() == [30, 40]
    assert sample["target_h"][0, 0, 0].tolist() == [0, 1]
    assert sample["target_h"][1, 0, 0].tolist() == [2, 3]


def test_all_model_shapes_and_losses() -> None:
    for domain in ("quasi", "mobility"):
        batch = collate_sample(domain)
        for name in (
            "lpan_l_direct",
            "edsr_lite",
            "spatial_gcn",
            "cnn_gru",
            "gcn_gru",
            "phymeta_stgt",
        ):
            model = build_model(
                name, domain=domain, hidden=16, graph_layers=1, heads=4
            )
            prediction = model(batch)
            assert prediction.shape == batch["target_h"].shape, name
            assert torch.isfinite(prediction).all(), name
            loss, parts = combined_loss(prediction, batch, LossWeights())
            assert torch.isfinite(loss)
            assert all(np.isfinite(value) for value in parts.values())


def test_lpan_l_direct_is_single_stage_32_to_256() -> None:
    model = build_model("lpan_l_direct", domain="mobility")
    batch = collate_sample("mobility")
    prediction = model(batch)
    assert prediction.shape == (1, 6, 256, 64, 2)
    assert model.target_nodes == 256
    assert not any(isinstance(module, torch.nn.Upsample) for module in model.modules())


def test_lpan_l_direct_rejects_nonofficial_geometry_and_ambiguous_alias() -> None:
    model = build_model("lpan_l_direct", domain="mobility")
    batch = collate_sample("mobility")
    batch["obs_ris_index"] = torch.arange(1, 256, 8).unsqueeze(0)
    with pytest.raises(ValueError, match="verified official LPAN ordering"):
        model(batch)
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("lpan_l", domain="mobility")


def test_complexity_profile_uses_one_explicit_convention() -> None:
    batch = canonical_batch("mobility", batch_size=1)
    for name in ("lpan_l_direct", "phymeta_stgt"):
        model = build_model(
            name,
            domain="mobility",
            hidden=16,
            graph_layers=1,
            heads=4,
        )
        result = profile_model_complexity(model, batch)
        assert result["total_parameters"] == sum(
            parameter.numel() for parameter in model.parameters()
        )
        assert result["macs"] > 0
        assert result["flops"] == 2 * result["macs"]
        assert result["gflops"] == pytest.approx(2 * result["gmacs"])
        assert result["input_shape"] == [1, 2, 32, 64, 2]
        assert result["output_shape"] == [1, 6, 256, 64, 2]
        assert result["interpolation_policy"].startswith("excluded in full")
        assert "interpolation (all spatial and temporal paths)" in result[
            "excluded_operations"
        ]


def test_complexity_excludes_interpolation_paths() -> None:
    class InterpolationOnly(torch.nn.Module):
        def forward(
            self, batch: dict[str, torch.Tensor]
        ) -> torch.Tensor:
            return expand_observations_to_grid(batch)

    result = profile_model_complexity(
        InterpolationOnly(), canonical_batch("mobility", batch_size=1)
    )
    assert result["macs"] == 0
    assert result["flops"] == 0


def test_profile_cli_defaults_to_batch_one() -> None:
    args = experiment_parser().parse_args(
        [
            "profile",
            "--domain",
            "mobility",
            "--models",
            "lpan_l_direct,phymeta_stgt",
        ]
    )
    assert args.batch_size == 1
    assert args.device == "cpu"
    assert args.models == ("lpan_l_direct", "phymeta_stgt")


def test_phymeta_architecture_and_loss_ablations() -> None:
    batch = collate_sample("mobility")
    for variant in ARCHITECTURE_ABLATIONS:
        model = build_model(
            "phymeta_stgt",
            domain="mobility",
            hidden=16,
            graph_layers=1,
            heads=4,
            ablation=variant,
        )
        prediction = model(batch)
        assert prediction.shape == batch["target_h"].shape, variant
        assert torch.isfinite(prediction).all(), variant
    base = LossWeights(1.0, 0.1, 0.2, 0.3)
    nmse_only = ablated_loss_weights(base, "nmse_only", domain="mobility")
    assert nmse_only == LossWeights(1.0, 0.0, 0.0, 0.0)
    no_delta = ablated_loss_weights(
        base, "no_temporal_delta_loss", domain="mobility"
    )
    assert no_delta.delta == 0.0
    spatial = ablation_metadata("no_spatial_cross_attention")
    temporal = ablation_metadata("no_temporal_attention")
    assert "row-wise grid-aware" in str(spatial["replacement_mechanism"])
    assert "temporal interpolation" in str(temporal["replacement_mechanism"])


def test_phymeta_attention_disables_unused_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_model(
        "phymeta_stgt",
        domain="mobility",
        hidden=16,
        graph_layers=1,
        heads=4,
    )
    calls: list[bool | None] = []
    for attention in (model.spatial_cross_attention, model.temporal_attention):
        assert attention is not None
        original = attention.forward

        def wrapped(*args, _original=original, **kwargs):
            calls.append(kwargs.get("need_weights"))
            return _original(*args, **kwargs)

        monkeypatch.setattr(attention, "forward", wrapped)
    prediction = model(collate_sample("mobility"))
    assert torch.isfinite(prediction).all()
    assert calls == [False, False]


def test_hyperparameter_search_plan_is_valid_and_deterministic() -> None:
    values = {
        "hidden_values": (16, 18),
        "graph_layer_values": (1, 2),
        "head_values": (4,),
        "dropout_values": (0.0, 0.1),
        "learning_rate_values": (1e-4, 2e-4),
        "weight_decay_values": (0.0,),
        "strategy": "random",
        "max_trials": 3,
        "seed": 9,
    }
    first = hyperparameter_candidates(**values)
    second = hyperparameter_candidates(**values)
    assert first == second
    assert len(first) == 3
    assert all(item["hidden"] % item["heads"] == 0 for item in first)


def test_tune_and_ablate_cli_contracts() -> None:
    tune = experiment_parser().parse_args(
        ["tune", "--domain", "mobility", "--max-trials", "2"]
    )
    assert tune.model == "phymeta_stgt"
    assert tune.max_trials == 2
    assert tune.round1_epochs == 25
    assert tune.promote_top_k == 3
    assert tune.batch_size == 32
    assert tune.eval_batch_size == 64
    assert tune.workers == 8
    assert tune.min_epochs == 40
    assert tune.patience == 15
    ablate = experiment_parser().parse_args(
        [
            "ablate",
            "--domain",
            "mobility",
            "--variants",
            "none,no_graph,nmse_only",
            "--best-result",
            "runs/stage_b/best_result.json",
        ]
    )
    assert ablate.variants == ("none", "no_graph", "nmse_only")
    assert ablate.best_result == Path("runs/stage_b/best_result.json")

    missing_stage_b = experiment_parser().parse_args(
        ["ablate", "--domain", "mobility", "--mode", "full"]
    )
    with pytest.raises(ValueError, match="requires --best-result"):
        ablate_command(missing_stage_b)


def test_early_stopping_starts_patience_after_minimum_epoch() -> None:
    best = float("inf")
    stale = 0
    stopped_at = None
    for epoch in range(1, 80):
        value = 1.0 if epoch == 1 else 2.0
        best, stale, improved, should_stop = early_stopping_step(
            best_nmse=best,
            stale_epochs=stale,
            validation_nmse=value,
            epoch=epoch,
            enabled=True,
            min_epochs=40,
            patience=15,
        )
        assert improved is (epoch == 1)
        if should_stop:
            stopped_at = epoch
            break
    assert stopped_at == 55
    history = [
        {"epoch": epoch, "validation_nmse_linear": 1.0 if epoch == 1 else 2.0}
        for epoch in range(1, 56)
    ]
    assert early_stopping_progress(history, min_epochs=40) == (1.0, 15)


def test_select_fastest_batch_rejects_oom_rows() -> None:
    selected = select_fastest_batch(
        [
            {"batch_size": 16, "status": "completed", "samples_per_second": 10.0},
            {"batch_size": 32, "status": "completed", "samples_per_second": 20.0},
            {"batch_size": 64, "status": "oom"},
        ]
    )
    assert selected["batch_size"] == 32


def test_tune_promotes_top_trial_from_last_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[argparse.Namespace] = []

    def fake_train(args: argparse.Namespace) -> dict[str, object]:
        calls.append(args)
        if args.resume:
            run_dir = Path(args.resume).parent.parent
            metric = 0.05
        else:
            run_dir = Path(args.output_root) / args.run_name
            metric = 0.2 if args.run_name == "trial_000" else 0.1
        checkpoints = run_dir / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        (checkpoints / "last_checkpoint.pth").touch()
        (checkpoints / "best_checkpoint.pth").touch()
        return {
            "run_dir": str(run_dir),
            "result": {
                "best_validation_nmse_linear": metric,
                "best_validation_nmse_db": 10 * np.log10(metric),
                "epochs_completed": args.epochs,
                "stopped_early": False,
            },
        }

    monkeypatch.setattr(experiment_main, "train_command", fake_train)
    args = experiment_parser().parse_args(
        [
            "tune",
            "--domain",
            "mobility",
            "--mode",
            "full",
            "--max-trials",
            "2",
            "--round1-epochs",
            "2",
            "--promote-top-k",
            "1",
            "--epochs",
            "5",
            "--study-name",
            "promotion_test",
            "--output-root",
            str(tmp_path),
        ]
    )
    result = tune_command(args)
    assert len(calls) == 3
    assert calls[0].epochs == calls[1].epochs == 2
    assert calls[0].resume is None and calls[1].resume is None
    assert calls[2].epochs == 5
    assert Path(calls[2].resume).name == "last_checkpoint.pth"
    assert calls[2].run_name == "trial_001"
    assert result["round1_completed_trials"] == 2
    assert result["promoted_completed_trials"] == 1
    assert result["best_trial"]["trial"] == 1


def test_ablation_loads_stage_b_best_hyperparameters(tmp_path: Path) -> None:
    best_result = tmp_path / "best_result.json"
    best_result.write_text(
        __import__("json").dumps(
            {
                "status": "validation_search",
                "domain": "mobility",
                "best_hyperparameters": {
                    "hidden": 48,
                    "graph_layers": 3,
                    "heads": 4,
                    "dropout": 0.1,
                    "learning_rate": 1e-4,
                    "weight_decay": 1e-5,
                },
            }
        ),
        encoding="utf-8",
    )
    config = _load_best_hyperparameters(
        best_result,
        expected_domain="mobility",
        require_full_search=True,
    )
    assert config == {
        "hidden": 48,
        "graph_layers": 3,
        "heads": 4,
        "dropout": 0.1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
    }
    with pytest.raises(ValueError, match="does not match"):
        _load_best_hyperparameters(
            best_result,
            expected_domain="quasi",
            require_full_search=True,
        )


def test_backward_optimizer_and_future_gru_outputs() -> None:
    batch = collate_sample("mobility")
    for name in ("cnn_gru", "gcn_gru"):
        model = build_model(
            name, domain="mobility", hidden=16, graph_layers=1, heads=4
        )
        prediction = model(batch)
        assert all(
            not torch.allclose(left, right)
            for left, right in zip(
                prediction[:, 1:-1].unbind(1),
                prediction[:, 2:].unbind(1),
            )
        )
        loss, _ = combined_loss(prediction, batch, LossWeights())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        optimizer.step()


def test_cnn_grid_expansion_respects_observed_indices() -> None:
    batch = collate_sample("mobility")
    full = expand_observations_to_grid(batch)
    indices = batch["obs_ris_index"][0]
    assert torch.equal(full.index_select(2, indices), batch["obs_h"])


def test_grid_aware_interpolation_never_crosses_ris_rows() -> None:
    batch = collate_sample("quasi")
    obs = torch.zeros_like(batch["obs_h"])
    for row in range(16):
        obs[:, :, 2 * row] = float(100 * row)
        obs[:, :, 2 * row + 1] = float(100 * row + 8)
    batch["obs_h"] = obs
    linear = expand_observations_to_grid(batch)
    nearest = expand_observations_to_grid(batch, nearest=True)
    assert linear.shape == nearest.shape == (1, 1, 256, 64, 2)
    assert torch.isfinite(linear).all()
    assert torch.isfinite(nearest).all()
    assert torch.all(linear[:, :, 15] == 8.0)
    assert torch.all(linear[:, :, 16] == 100.0)
    assert torch.all(nearest[:, :, 15] == 8.0)
    assert torch.all(nearest[:, :, 16] == 100.0)
    batch["observation_mask"][:, :, 0] = False
    masked = expand_observations_to_grid(batch)
    assert torch.all(masked[:, :, :16] == 8.0)


@pytest.mark.parametrize(
    ("model_name", "ablation"),
    [
        ("edsr_lite", "none"),
        ("cnn_gru", "none"),
        ("phymeta_stgt", "no_spatial_cross_attention"),
    ],
)
def test_grid_dependent_models_support_backward(
    model_name: str, ablation: str
) -> None:
    batch = collate_sample("mobility")
    model = build_model(
        model_name,
        domain="mobility",
        hidden=16,
        graph_layers=1,
        heads=4,
        ablation=ablation,
    )
    prediction = model(batch)
    loss, _ = combined_loss(prediction, batch, LossWeights())
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_adaptation_guards_and_parameter_freezing() -> None:
    validate_training_request("phymeta_stgt", "full", None, None)
    with pytest.raises(ValueError, match="pretrained checkpoint"):
        validate_training_request("phymeta_stgt", "selective", None, None)
    with pytest.raises(ValueError, match="only supported"):
        validate_training_request("cnn_gru", "selective", "source.pth", None)
    baseline = build_model("cnn_gru", domain="mobility", hidden=16)
    with pytest.raises(ValueError, match="only supported"):
        configure_adaptation(baseline, "adapter_only")
    trainable_sets = {}
    reports = {}
    for policy in ("full", "frozen_spatial", "selective", "adapter_only"):
        proposed = build_model(
            "phymeta_stgt",
            domain="mobility",
            hidden=16,
            graph_layers=1,
            heads=4,
        )
        reports[policy] = configure_adaptation(proposed, policy)
        expected_names = [
            name
            for name, parameter in proposed.named_parameters()
            if parameter.requires_grad
        ]
        trainable_sets[policy] = set(expected_names)
        assert reports[policy]["trainable_parameter_names"] == expected_names
    assert (
        trainable_sets["adapter_only"]
        < trainable_sets["selective"]
        < trainable_sets["frozen_spatial"]
        < trainable_sets["full"]
    )
    assert reports["adapter_only"]["trainable_module_names"] == [
        "domain_embedding"
    ]


def test_pretrained_checkpoint_requires_exact_phymeta_architecture() -> None:
    source_config = {
        "domain": "quasi",
        "hidden": 16,
        "graph_layers": 1,
        "heads": 4,
        "dropout": 0.0,
    }
    target_config = {**source_config, "domain": "mobility"}
    source = build_model("phymeta_stgt", **source_config)
    state = {
        "model_name": "phymeta_stgt",
        "model_config": source_config,
        "model_state": source.state_dict(),
        "metadata": {"domain": "quasi"},
    }
    target = build_model("phymeta_stgt", **target_config)
    report = load_pretrained_checkpoint(
        target, state, target_config, "source.pth"
    )
    assert report["strict_load"] is True
    assert report["source_domain"] == "quasi"

    with pytest.raises(ValueError, match="requires a phymeta_stgt"):
        load_pretrained_checkpoint(
            target,
            {**state, "model_name": "cnn_gru"},
            target_config,
            "wrong-model.pth",
        )
    with pytest.raises(ValueError, match="architecture does not match"):
        load_pretrained_checkpoint(
            target,
            {
                **state,
                "model_config": {**source_config, "heads": 2},
            },
            target_config,
            "wrong-heads.pth",
        )
    incomplete_state = dict(source.state_dict())
    incomplete_state.pop(next(iter(incomplete_state)))
    with pytest.raises(ValueError, match="weights are incompatible"):
        load_pretrained_checkpoint(
            target,
            {**state, "model_state": incomplete_state},
            target_config,
            "missing-key.pth",
        )


def test_evaluation_semantics_inherit_reject_and_override() -> None:
    saved_indices = tuple(range(1, 256, 8))
    state = {
        "metadata": {
            "domain": "mobility",
            "obs_time_index": [0, 1],
            "obs_ris_index": list(saved_indices),
            "complex_layout": "interleaved",
        }
    }
    defaults = {
        "domain": None,
        "obs_times": None,
        "obs_ris_indices": None,
        "complex_layout": None,
        "semantic_profile": None,
        "allow_semantic_override": False,
    }
    resolved = resolve_evaluation_semantics(
        argparse.Namespace(**defaults), state
    )
    assert resolved == (
        "mobility",
        (0, 1),
        saved_indices,
        "interleaved",
        "custom",
    )

    conflicting = argparse.Namespace(**{**defaults, "complex_layout": "grouped"})
    with pytest.raises(ValueError, match="do not match checkpoint"):
        resolve_evaluation_semantics(conflicting, state)

    override = argparse.Namespace(
        **{
            **defaults,
            "complex_layout": "grouped",
            "allow_semantic_override": True,
        }
    )
    assert resolve_evaluation_semantics(override, state)[3] == "grouped"


def test_evaluate_cli_semantics_default_to_checkpoint() -> None:
    args = experiment_parser().parse_args(
        ["evaluate", "--checkpoint", "model.pth"]
    )
    assert args.obs_times is None
    assert args.obs_ris_indices is None
    assert args.complex_layout is None
    assert args.semantic_profile is None
    assert args.allow_semantic_override is False


def test_ci_uses_existing_action_major_versions() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
    assert "actions/checkout@v6" in workflow
    assert "actions/setup-python@v6" in workflow
    assert "@v7" not in workflow


def test_rng_state_round_trip() -> None:
    generator = torch.Generator().manual_seed(9)
    state = capture_rng_state({"train": generator})
    expected_global = torch.rand(4)
    expected_loader = torch.randperm(10, generator=generator)
    restore_rng_state(state, {"train": generator})
    assert torch.equal(torch.rand(4), expected_global)
    assert torch.equal(torch.randperm(10, generator=generator), expected_loader)


def test_rng_restore_normalizes_cpu_and_generator_states_to_byte() -> None:
    generator = torch.Generator().manual_seed(17)
    state = capture_rng_state({"train": generator})
    expected_global = torch.rand(4)
    expected_loader = torch.randperm(10, generator=generator)
    state["torch_cpu"] = state["torch_cpu"].to(torch.int16)
    loader_states = state["loader_generators"]
    assert isinstance(loader_states, dict)
    loader_states["train"] = loader_states["train"].to(torch.int16)

    restored_generator = torch.Generator()
    restore_rng_state(state, {"train": restored_generator})

    assert torch.equal(torch.rand(4), expected_global)
    assert torch.equal(
        torch.randperm(10, generator=restored_generator), expected_loader
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for RNG map-location regression test",
)
def test_restore_rng_state_after_cuda_checkpoint_load(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(123)
    state = capture_rng_state({"train": generator})
    path = tmp_path / "rng_checkpoint.pth"
    torch.save({"rng_state": state}, path)
    loaded = torch.load(path, map_location="cuda", weights_only=False)
    loaded_state = loaded["rng_state"]
    assert loaded_state["torch_cpu"].is_cuda
    assert loaded_state["loader_generators"]["train"].is_cuda

    restored_generator = torch.Generator()
    restore_rng_state(loaded_state, {"train": restored_generator})

    assert torch.get_rng_state().device.type == "cpu"
    assert restored_generator.get_state().device.type == "cpu"


def test_rng_restore_rejects_nontensor_state() -> None:
    state = capture_rng_state()
    state["torch_cpu"] = "not-a-tensor"
    with pytest.raises(TypeError, match="torch_cpu RNG state must be a tensor"):
        restore_rng_state(state)


def test_cli_resume_keeps_history_and_rng_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_path = tmp_path / "train.mat"
    val_path = tmp_path / "validation.mat"
    write_mat(train_path, "mobility", samples=2)
    write_mat(val_path, "mobility", samples=1)
    output_root = tmp_path / "runs"
    common = [
        "main.py",
        "train",
        "--domain",
        "mobility",
        "--model",
        "cnn_gru",
        "--mode",
        "full",
        "--train-path",
        str(train_path),
        "--val-path",
        str(val_path),
        "--max-train",
        "1",
        "--max-val",
        "1",
        "--batch-size",
        "1",
        "--eval-batch-size",
        "1",
        "--hidden",
        "4",
        "--workers",
        "0",
        "--device",
        "cpu",
        "--seed",
        "77",
        "--output-root",
        str(output_root),
        "--run-name",
        "resume_test",
    ]
    monkeypatch.setattr(sys, "argv", common + ["--epochs", "1"])
    cli_main()
    run_dir = output_root / "resume_test"
    checkpoint = run_dir / "checkpoints" / "last_checkpoint.pth"
    original_command = (run_dir / "command.txt").read_text(encoding="utf-8")
    history_path = run_dir / "results" / "training_history.csv"
    history = read_history(history_path)
    synthetic_ahead_row = dict(history[-1])
    synthetic_ahead_row["epoch"] = 2
    write_history(history_path, history + [synthetic_ahead_row])

    monkeypatch.setattr(
        sys,
        "argv",
        common
        + [
            "--epochs",
            "2",
            "--resume",
            str(checkpoint),
        ],
    )
    cli_main()

    history_lines = (
        run_dir / "results" / "training_history.csv"
    ).read_text(encoding="utf-8-sig").splitlines()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert len(history_lines) == 3
    assert state["epoch"] == 2
    assert state["rng_state"] is not None
    assert (run_dir / "command.txt").read_text(encoding="utf-8") == original_command
    assert len(
        (run_dir / "resume_commands.log").read_text(encoding="utf-8").splitlines()
    ) == 1
    recovery = (run_dir / "recovery.log").read_text(encoding="utf-8")
    assert "truncated training_history.csv from epoch 2 to checkpoint epoch 1" in recovery


def test_auxiliary_losses_are_scale_normalized() -> None:
    batch = collate_sample("mobility")
    prediction = torch.randn_like(batch["target_h"])
    base_char = charbonnier(prediction, batch["target_h"])
    base_observation = observation_consistency(prediction, batch)
    scaled_batch = {
        key: value * 10 if key in {"obs_h", "target_h"} else value
        for key, value in batch.items()
    }
    assert torch.allclose(
        charbonnier(prediction * 10, scaled_batch["target_h"]),
        base_char,
        rtol=1e-5,
        atol=1e-7,
    )
    assert torch.allclose(
        observation_consistency(prediction * 10, scaled_batch),
        base_observation,
        rtol=1e-5,
        atol=1e-7,
    )


def test_observation_consistency_uses_mask_and_rejects_empty_sample() -> None:
    batch = collate_sample("mobility")
    prediction = batch["target_h"].clone()
    ris = batch["obs_ris_index"][0]
    times = batch["obs_time_index"][0]
    prediction[:, times[0], ris[0]] = 1e6
    batch["observation_mask"][:, 0, 0] = False
    masked = observation_consistency(prediction, batch)
    prediction[:, times[0], ris[0]] = -1e6
    assert torch.equal(observation_consistency(prediction, batch), masked)

    batch["observation_mask"].zero_()
    with pytest.raises(ValueError, match="at least one valid observation"):
        observation_consistency(prediction, batch)
    model = build_model(
        "phymeta_stgt",
        domain="mobility",
        hidden=16,
        graph_layers=1,
        heads=4,
    )
    with pytest.raises(ValueError, match="at least one valid observation token"):
        model(batch)


def test_audit_reports_corrupt_file_without_stopping(tmp_path: Path) -> None:
    corrupt = dataset_candidates(tmp_path, "quasi", "train")[0]
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"not an hdf5 file")
    output = tmp_path / "audit.json"
    args = type(
        "Args",
        (),
        {
            "data_root": str(tmp_path),
            "mobility_obs_times": (0, 1),
            "obs_ris_indices": tuple(range(0, 256, 8)),
            "complex_layout": "grouped",
            "semantic_profile": "official_lpan",
            "output": str(output),
        },
    )()
    audit_command(args)
    report = __import__("json").loads(output.read_text(encoding="utf-8"))
    entry = report["files"][0]
    assert entry["exists"] is True
    assert entry["valid"] is False
    assert "Not an HDF5" in entry["error"]
    assert len(report["files"]) == 6


def test_official_semantic_profile_and_audit_reject_wrong_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mobility.mat"
    write_mat(path, "mobility", samples=3)
    with pytest.raises(ValueError, match="official_lpan semantic profile rejects"):
        LPANH5Dataset(
            path,
            "mobility",
            "validation",
            complex_layout="interleaved",
        )
    with pytest.raises(ValueError, match="official_lpan semantic profile rejects"):
        LPANH5Dataset(
            path,
            "mobility",
            "validation",
            obs_ris_index=tuple(range(1, 256, 8)),
        )
    with pytest.raises(ValueError, match="official_lpan semantic profile rejects"):
        LPANH5Dataset(
            path,
            "mobility",
            "validation",
            obs_time_index=(1, 2),
        )
    custom = LPANH5Dataset(
        path,
        "mobility",
        "validation",
        obs_ris_index=tuple(range(1, 256, 8)),
        obs_time_index=(1, 2),
        complex_layout="interleaved",
        semantic_profile="custom",
    )
    assert custom.semantic_profile == "custom"
    custom.close()

    candidate = dataset_candidates(tmp_path, "mobility", "validation")[0]
    candidate.parent.mkdir(parents=True, exist_ok=True)
    write_mat(candidate, "mobility", samples=3)
    output = tmp_path / "audit.json"
    args = argparse.Namespace(
        data_root=str(tmp_path),
        mobility_obs_times=(0, 1),
        obs_ris_indices=tuple(range(0, 256, 8)),
        complex_layout="grouped",
        semantic_profile="official_lpan",
        output=str(output),
    )
    audit_command(args)
    report = __import__("json").loads(output.read_text(encoding="utf-8"))
    entry = next(
        row
        for row in report["files"]
        if row["domain"] == "mobility" and row["split"] == "validation"
    )
    assert entry["valid"] is False
    assert entry["expected_total_samples"] == 1800
    assert "expects 1800 samples, found 3" in entry["error"]


def test_atomic_checkpoint_replaces_final_file(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "checkpoint.pth"
    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=4,
        best_nmse=0.5,
        model_name="unit",
        model_config={},
        metadata={},
    )
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["epoch"] == 4
    assert not path.with_name(f"{path.name}.tmp").exists()


def test_per_snr_requires_explicit_total_size(tmp_path: Path) -> None:
    loader = type("Loader", (), {"dataset": [0, 1, 2]})()
    with pytest.raises(ValueError, match="expects"):
        per_snr_evaluation(
            torch.nn.Identity(),
            loader,
            torch.device("cpu"),
            tmp_path / "snr.csv",
            (-10, -5),
            1000,
        )


def test_per_snr_rejects_reordered_subset(tmp_path: Path) -> None:
    dataset = type(
        "Dataset",
        (),
        {
            "__len__": lambda self: 4,
            "indices": np.array([0, 2, 3, 5]),
        },
    )()
    loader = type("Loader", (), {"dataset": dataset})()
    with pytest.raises(ValueError, match="original sample order"):
        per_snr_evaluation(
            torch.nn.Identity(),
            loader,
            torch.device("cpu"),
            tmp_path / "snr.csv",
            (-10, -5),
            2,
        )


def test_interpolation_metrics_and_ridge() -> None:
    batch = collate_sample("mobility")
    prediction = interpolation_baseline(batch)
    assert prediction.shape == batch["target_h"].shape
    metrics = MetricAccumulator()
    metrics.update(prediction, batch)
    result = metrics.compute()
    assert result["sample_count"] == 1
    statistics = RidgeStatistics.accumulate([batch])
    ridge = statistics.solve(1e-3)
    ridge_prediction = ridge.predict(batch)
    assert ridge_prediction.shape == batch["target_h"].shape


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_metric_accumulator_rejects_non_finite_samples(invalid: float) -> None:
    batch = collate_sample("mobility")
    prediction = torch.randn_like(batch["target_h"])
    prediction[0, 0, 0, 0, 0] = invalid
    metrics = MetricAccumulator()
    with pytest.raises(FloatingPointError, match="non-finite"):
        metrics.update(prediction, batch)
    assert metrics.compute()["sample_count"] == 0


def _encode_complex_layout(values: np.ndarray, layout: str) -> np.ndarray:
    # [S,T,RIS,BS] -> [2T,RIS,BS,S]
    samples, blocks, nodes, antennas = values.shape
    if layout == "grouped":
        channels = np.concatenate((values.real, values.imag), axis=1)
    else:
        channels = np.empty(
            (samples, 2 * blocks, nodes, antennas), dtype=np.float64
        )
        channels[:, 0::2] = values.real
        channels[:, 1::2] = values.imag
    return channels.transpose(1, 2, 3, 0)


@pytest.mark.parametrize("layout", ["grouped", "interleaved"])
def test_mobility_layout_is_inferred_from_correlation_evidence(
    tmp_path: Path, layout: str
) -> None:
    rng = np.random.default_rng(42)
    target = rng.normal(size=(8, 6, 256, 64)) + 1j * rng.normal(
        size=(8, 6, 256, 64)
    )
    observed = target[:, :2, np.arange(0, 256, 8)]
    path = tmp_path / f"{layout}.mat"
    with h5py.File(path, "w") as handle:
        handle["Yd"] = _encode_complex_layout(observed, layout)
        handle["Hd"] = _encode_complex_layout(target, layout)
    result = verify_mobility(path, samples=8)
    assert result["verified_layout"] == layout
    assert result["layout_inference"]["status"] == "verified"
    expected_second_channel = "Re(t2)" if layout == "grouped" else "Im(t1)"
    assert result["raw_Yd_channels"][1] == expected_second_channel
