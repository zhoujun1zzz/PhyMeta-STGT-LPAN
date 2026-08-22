from __future__ import annotations

import numpy as np
import pytest
import torch

from lpan.complexity import canonical_batch
from lpan.data import (
    CANONICAL_MOBILITY_PROFILE,
    MOBILITY_BASELINE_CONTRACT_VERSION,
    LPANH5Dataset,
    default_observed_ris_indices,
    semantic_contract,
)
from lpan.models import (
    ProgressiveLPAN,
    lpan_grouped_raw_input,
    lpan_grouped_raw_output,
    lpan_raw_input,
    lpan_raw_output,
)
from lpan.semantic_audit import audit_batch_equivalence, time_label_independence


def _paired_batches(batch_size: int = 2) -> tuple[dict[str, torch.Tensor], ...]:
    generator = np.random.default_rng(2026)
    raw_obs = generator.normal(size=(batch_size, 4, 32, 64)).astype(np.float32)
    raw_target = generator.normal(size=(batch_size, 12, 256, 64)).astype(
        np.float32
    )

    def decode(raw: np.ndarray, blocks: int, layout: str) -> torch.Tensor:
        values = [LPANH5Dataset._to_complex_last(item, blocks, layout) for item in raw]
        return torch.from_numpy(np.stack(values))

    shared = {
        "obs_ris_index": torch.arange(0, 256, 8).expand(batch_size, -1),
        "query_time": torch.arange(6).expand(batch_size, -1),
        "domain_id": torch.ones(batch_size, dtype=torch.long),
        "observation_mask": torch.ones(batch_size, 2, 32, dtype=torch.bool),
        "sample_index": torch.arange(batch_size),
    }
    legacy = {
        **shared,
        "obs_h": decode(raw_obs, 2, "interleaved"),
        "target_h": decode(raw_target, 6, "interleaved"),
        "obs_time_index": torch.tensor([1, 4]).expand(batch_size, -1),
        "complex_layout_id": torch.ones(batch_size, dtype=torch.long),
    }
    canonical = {
        **shared,
        "obs_h": decode(raw_obs, 2, "grouped"),
        "target_h": decode(raw_target, 6, "grouped"),
        "obs_time_index": torch.tensor([0, 3]).expand(batch_size, -1),
        "complex_layout_id": torch.zeros(batch_size, dtype=torch.long),
    }
    return legacy, canonical


def test_canonical_contract_locks_grouped_q0_q3() -> None:
    contract = semantic_contract(
        domain="mobility",
        semantic_profile=CANONICAL_MOBILITY_PROFILE,
        complex_layout="grouped",
        obs_time_index=(0, 3),
        obs_ris_index=default_observed_ris_indices(),
    )
    assert contract["contract_version"] == MOBILITY_BASELINE_CONTRACT_VERSION
    assert contract["obs_time_index"] == [0, 3]
    assert contract["query_time"] == list(range(6))
    assert contract["obs_ris_index"] == list(range(0, 256, 8))
    with pytest.raises(ValueError, match="locked columns"):
        semantic_contract(
            domain="mobility",
            semantic_profile=CANONICAL_MOBILITY_PROFILE,
            complex_layout="interleaved",
            obs_time_index=(1, 4),
            obs_ris_index=default_observed_ris_indices(),
        )


def test_grouped_y_h_decoder_and_raw_adapters_are_exact() -> None:
    legacy, canonical = _paired_batches(1)
    assert torch.equal(
        lpan_raw_input(legacy["obs_h"]),
        lpan_grouped_raw_input(canonical["obs_h"]),
    )
    assert torch.equal(
        lpan_raw_input(legacy["target_h"]),
        lpan_grouped_raw_input(canonical["target_h"]),
    )
    raw = torch.arange(12.0).reshape(1, 12, 1, 1)
    grouped = lpan_grouped_raw_output(raw, 6)
    interleaved = lpan_raw_output(raw, 6)
    assert grouped[0, :, 0, 0, 0].tolist() == list(range(6))
    assert grouped[0, :, 0, 0, 1].tolist() == list(range(6, 12))
    assert interleaved[0, :, 0, 0, 0].tolist() == list(range(0, 12, 2))
    assert torch.equal(lpan_grouped_raw_input(grouped), raw)


def test_input_target_loss_and_nmse_cancellation() -> None:
    legacy, canonical = _paired_batches()
    checks = audit_batch_equivalence(legacy, canonical)
    assert set(checks) == {
        "effective_input_equal",
        "effective_target_equal",
        "progressive_loss_equal",
        "nmse_permutation_invariant",
    }
    assert all(check["passed"] for check in checks.values())
    assert all(check["max_abs_error"] <= check["absolute_tolerance"] for check in checks.values())


@pytest.mark.parametrize("lightweight", [False, True])
def test_progressive_models_ignore_time_labels(lightweight: bool) -> None:
    model = ProgressiveLPAN(
        2, 6, lightweight=lightweight, domain="mobility", channels=4
    )
    batch = canonical_batch("mobility")
    result = time_label_independence(model, batch)
    assert result == {"passed": True, "max_abs_error": 0.0}


def test_semantic_adapter_adds_no_checkpoint_parameters() -> None:
    model = ProgressiveLPAN(2, 6, lightweight=True, domain="mobility", channels=4)
    before = tuple(model.state_dict())
    legacy, canonical = _paired_batches(1)
    with torch.inference_mode():
        model(legacy)
        model(canonical)
    assert tuple(model.state_dict()) == before
    clone = ProgressiveLPAN(2, 6, lightweight=True, domain="mobility", channels=4)
    incompatible = clone.load_state_dict(model.state_dict(), strict=False)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
