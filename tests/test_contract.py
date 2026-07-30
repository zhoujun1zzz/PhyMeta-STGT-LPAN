from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lpan.data import LPANH5Dataset
from lpan.metrics import MetricAccumulator
from lpan.models import build_model, interpolation_baseline
from lpan.objectives import LossWeights, combined_loss
from lpan.paths import dataset_candidates, resolve_dataset_path
from lpan.ridge import RidgeStatistics


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
