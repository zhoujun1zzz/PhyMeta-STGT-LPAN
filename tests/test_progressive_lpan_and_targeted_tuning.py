from __future__ import annotations

from pathlib import Path

import pytest
import torch

import main as experiment_main
from lpan.complexity import canonical_batch, profile_model_complexity
from lpan.engine import train_epoch
from lpan.models import (
    LPANFeatureStage,
    LPANLFeatureStage,
    LPANLFinalFeatureStage,
    LPANLResidualBlock,
    LPANResidualBlock,
    ProgressiveLPAN,
    build_model,
    lpan_grouped_input,
    lpan_grouped_output,
)
from lpan.objectives import (
    LossWeights,
    progressive_charbonnier_loss,
    progressive_targets,
)



def test_progressive_models_are_registered_without_relabeling_direct() -> None:
    assert isinstance(build_model("lpan_progressive", domain="quasi"), ProgressiveLPAN)
    assert isinstance(
        build_model("lpan_l_progressive", domain="mobility"), ProgressiveLPAN
    )
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("lpan_l", domain="mobility")


@pytest.mark.parametrize(
    ("domain", "obs_blocks", "query_blocks", "expected"),
    [
        ("quasi", 1, 1, ((1, 1, 64, 64, 2), (1, 1, 128, 64, 2), (1, 1, 256, 64, 2))),
        ("mobility", 2, 6, ((1, 6, 64, 64, 2), (1, 6, 128, 64, 2), (1, 6, 256, 64, 2))),
    ],
)
@pytest.mark.parametrize("lightweight", [False, True])
def test_progressive_lpan_shapes(
    domain: str,
    obs_blocks: int,
    query_blocks: int,
    expected: tuple[tuple[int, ...], ...],
    lightweight: bool,
) -> None:
    model = ProgressiveLPAN(
        obs_blocks,
        query_blocks,
        lightweight=lightweight,
        domain=domain,
        channels=4,
    ).eval()
    with torch.inference_mode():
        outputs = model.forward_multiscale(canonical_batch(domain))
        final = model(canonical_batch(domain))
    assert tuple(tuple(item.shape) for item in outputs) == expected
    assert torch.equal(final, outputs[-1])


def test_progressive_target_indices_are_exact() -> None:
    target = torch.arange(256, dtype=torch.float32).view(1, 1, 256, 1, 1)
    hr2, hr4, hr8 = progressive_targets(target)
    assert torch.equal(hr2, target[:, :, 1::4])
    assert torch.equal(hr4, target[:, :, 1::2])
    assert torch.equal(hr8, target)
    assert hr2.flatten().tolist() == list(range(1, 256, 4))
    assert hr4.flatten().tolist() == list(range(1, 256, 2))


def test_grouped_channel_order_round_trip() -> None:
    obs = torch.zeros(1, 2, 3, 1, 2)
    obs[0, 0, :, 0, 0] = 10
    obs[0, 1, :, 0, 0] = 20
    obs[0, 0, :, 0, 1] = 30
    obs[0, 1, :, 0, 1] = 40
    grouped = lpan_grouped_input(obs)
    assert grouped[:, :, 0, 0].tolist() == [[10, 20, 30, 40]]

    output = torch.zeros(1, 12, 1, 3)
    for channel in range(12):
        output[:, channel] = channel
    unified = lpan_grouped_output(output, query_blocks=6)
    assert unified[0, :, 0, 0, 0].tolist() == list(range(6))
    assert unified[0, :, 0, 0, 1].tolist() == list(range(6, 12))
    assert torch.equal(lpan_grouped_input(unified), output)


def test_block_fidelity_and_public_loop_equivalence() -> None:
    lpan_stage = LPANFeatureStage(16)
    assert len(lpan_stage.blocks) == 4
    assert all(isinstance(block, LPANResidualBlock) for block in lpan_stage.blocks)
    assert all(
        isinstance(block.activation2, torch.nn.LeakyReLU)
        for block in lpan_stage.blocks
    )

    lpan_l_stage = LPANLFeatureStage(16)
    assert lpan_l_stage.grouped_block.grouped
    assert len(lpan_l_stage.ordinary_blocks) == 2
    assert all(not block.grouped for block in lpan_l_stage.ordinary_blocks)
    final_stage = LPANLFinalFeatureStage(16)
    assert len(final_stage.ordinary_blocks) == 0
    assert not hasattr(final_stage.refine[0], "weight_g")

    ordinary = LPANLResidualBlock(16, grouped=False)
    grouped = LPANLResidualBlock(16, grouped=True)
    assert isinstance(ordinary.project, torch.nn.Identity)
    assert not hasattr(ordinary, "activation2")
    assert not isinstance(grouped.project, torch.nn.Identity)
    x = torch.randn(1, 16, 4, 4)
    for repeats in (2, 4):
        literal = None
        for _ in range(repeats):
            literal = grouped(x)
        optimized = grouped(x)
        assert literal is not None
        assert torch.equal(literal, optimized)

    quasi = ProgressiveLPAN(1, 1, lightweight=True, domain="quasi", channels=16)
    mobility = ProgressiveLPAN(2, 6, lightweight=True, domain="mobility", channels=16)
    assert quasi.reconstruction_stages[0] is quasi.reconstruction_stages[1]
    assert quasi.reconstruction_stages[1] is not quasi.reconstruction_stages[2]
    assert mobility.reconstruction_stages[0] is not mobility.reconstruction_stages[1]
    assert mobility.reconstruction_stages[1] is mobility.reconstruction_stages[2]
    assert sum(parameter.numel() for parameter in mobility.parameters()) == 1_112_904


@pytest.mark.parametrize("domain,obs_blocks,query_blocks", [("quasi", 1, 1), ("mobility", 2, 6)])
@pytest.mark.parametrize("lightweight", [False, True])
def test_progressive_loss_backward_is_finite(
    domain: str, obs_blocks: int, query_blocks: int, lightweight: bool
) -> None:
    batch = canonical_batch(domain)
    batch["target_h"].normal_()
    model = ProgressiveLPAN(
        obs_blocks,
        query_blocks,
        lightweight=lightweight,
        domain=domain,
        channels=4,
    )
    predictions = model.forward_multiscale(batch)
    loss, parts = progressive_charbonnier_loss(predictions, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    assert set(parts) == {"hr2_charbonnier", "hr4_charbonnier", "hr8_charbonnier", "total"}


def test_progressive_model_uses_unified_evaluator_and_profiler() -> None:
    model = ProgressiveLPAN(1, 1, lightweight=True, domain="quasi", channels=4)
    report = profile_model_complexity(model, canonical_batch("quasi"))
    assert report["output_shape"] == [1, 1, 256, 64, 2]
    assert report["macs"] > 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = train_epoch(
        model,
        [canonical_batch("quasi")],
        optimizer,
        torch.device("cpu"),
        LossWeights(),
        loss_profile="official_progressive_charbonnier",
    )
    assert result["total"] > 0


def test_targeted_search_plan_and_linear_late_window() -> None:
    args = experiment_main.parser().parse_args(
        ["tune", "--domain", "mobility", "--mode", "full", "--tuning-protocol", "targeted_boundary"]
    )
    first = experiment_main.targeted_boundary_search_plan(args)
    second = experiment_main.targeted_boundary_search_plan(args)
    assert first == second
    assert first["capacity"]["candidates"] == [96, 128, 160]
    assert first["learning_rate"]["candidates"] == [5e-4, 8e-4, 1e-3]
    assert first["learning_rate"]["from_scratch"] is True
    history = [
        {"epoch": epoch, "validation_nmse_linear": value}
        for epoch, value in enumerate([1.0, 0.25, 4.0], start=1)
    ]
    metrics = experiment_main.late_window_nmse_metrics(history, 1, 3)
    assert metrics["median_validation_nmse_linear"] == 1.0
    assert metrics["mean_validation_nmse_linear"] == pytest.approx(1.75)
    assert metrics["sample_count"] == 3


def test_targeted_protocol_scratch_trials_and_exact_final_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_train(args):
        calls.append(args)
        if args.resume:
            run_dir = Path(args.resume).parent.parent
        else:
            run_dir = Path(args.output_root) / args.run_name
            (run_dir / "checkpoints").mkdir(parents=True)
            (run_dir / "checkpoints" / "last_checkpoint.pth").write_bytes(b"checkpoint")
            (run_dir / "checkpoints" / "best_checkpoint.pth").write_bytes(b"best")
            (run_dir / "results").mkdir()
        epochs = int(args.epochs)
        if args.resume:
            score = 0.01
        elif args.run_name.startswith("trial_hidden"):
            score = 1.0 / int(args.hidden)
        else:
            score = abs(float(args.learning_rate) - 8e-4) + 0.1
        history = [
            {"epoch": epoch, "validation_nmse_linear": score}
            for epoch in range(1, epochs + 1)
        ]
        return {
            "run_dir": str(run_dir),
            "result": {
                "epochs_completed": epochs,
                "best_validation_nmse_linear": score,
                "history": history,
            },
        }

    monkeypatch.setattr(experiment_main, "train_command", fake_train)
    args = experiment_main.parser().parse_args(
        [
            "tune",
            "--domain",
            "mobility",
            "--mode",
            "full",
            "--tuning-protocol",
            "targeted_boundary",
            "--study-name",
            "study",
            "--output-root",
            str(tmp_path),
        ]
    )
    result = experiment_main.targeted_boundary_tune_command(args)
    capacity_calls = calls[:3]
    learning_rate_calls = calls[3:6]
    final_call = calls[6]
    assert [call.epochs for call in capacity_calls] == [20, 20, 20]
    assert all(call.min_epochs == 40 for call in capacity_calls)
    assert [call.epochs for call in learning_rate_calls] == [40, 40, 40]
    assert all(call.resume is None for call in learning_rate_calls)
    assert all(call.seed == 123 for call in learning_rate_calls)
    assert final_call.resume.endswith("final/checkpoints/last_checkpoint.pth")
    assert final_call.epochs == 100
    assert result["phase_c_resume_source_epoch"] == 40
    assert result["test_split_used"] is False
    assert result["boundary_hit"] == {
        "hidden_upper": True,
        "learning_rate_upper": False,
    }
    loaded = experiment_main._load_best_hyperparameters(
        tmp_path / "study" / "best_result.json",
        expected_domain="mobility",
        require_full_search=True,
    )
    assert loaded == result["best_hyperparameters"]


def test_old_and_targeted_tuning_protocols_remain_parseable() -> None:
    parser = experiment_main.parser()
    old = parser.parse_args(["tune", "--domain", "quasi"])
    targeted = parser.parse_args(
        ["tune", "--domain", "quasi", "--tuning-protocol", "targeted_boundary"]
    )
    assert old.tuning_protocol == "two_round_validation_promotion"
    assert targeted.tuning_protocol == "targeted_boundary"
