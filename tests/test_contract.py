from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lpan.data import LPANH5Dataset
from lpan.engine import (
    capture_rng_state,
    configure_adaptation,
    restore_rng_state,
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
from main import (
    audit_command,
    main as cli_main,
    per_snr_evaluation,
    validate_training_request,
)


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
    )[0]
    assert sample["obs_h"][0, 0, 0].tolist() == [10, 20]
    assert sample["obs_h"][1, 0, 0].tolist() == [30, 40]
    assert sample["target_h"][0, 0, 0].tolist() == [0, 1]
    assert sample["target_h"][1, 0, 0].tolist() == [2, 3]


def test_all_model_shapes_and_losses() -> None:
    for domain in ("quasi", "mobility"):
        batch = collate_sample(domain)
        for name in (
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


def test_adaptation_guards_and_parameter_freezing() -> None:
    validate_training_request("phymeta_stgt", "full", None, None)
    with pytest.raises(ValueError, match="pretrained checkpoint"):
        validate_training_request("phymeta_stgt", "selective", None, None)
    with pytest.raises(ValueError, match="only supported"):
        validate_training_request("cnn_gru", "selective", "source.pth", None)
    baseline = build_model("cnn_gru", domain="mobility", hidden=16)
    with pytest.raises(ValueError, match="only supported"):
        configure_adaptation(baseline, "adapter_only")
    proposed = build_model(
        "phymeta_stgt",
        domain="mobility",
        hidden=16,
        graph_layers=1,
        heads=4,
    )
    report = configure_adaptation(proposed, "selective")
    assert 0 < report["trainable_parameters"] < report["total_parameters"]


def test_rng_state_round_trip() -> None:
    generator = torch.Generator().manual_seed(9)
    state = capture_rng_state({"train": generator})
    expected_global = torch.rand(4)
    expected_loader = torch.randperm(10, generator=generator)
    restore_rng_state(state, {"train": generator})
    assert torch.equal(torch.rand(4), expected_global)
    assert torch.equal(torch.randperm(10, generator=generator), expected_loader)


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
