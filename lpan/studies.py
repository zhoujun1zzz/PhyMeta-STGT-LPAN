from __future__ import annotations

import itertools
import random
from collections.abc import Sequence

from .objectives import LossWeights


ARCHITECTURE_ABLATIONS = (
    "no_spatial_cross_attention",
    "no_graph",
    "no_temporal_attention",
    "no_domain_adapter",
    "no_coordinate_encoding",
)
LOSS_ABLATIONS = (
    "nmse_only",
    "no_charbonnier_loss",
    "no_observation_loss",
    "no_temporal_delta_loss",
)
ABLATION_VARIANTS = ("none",) + ARCHITECTURE_ABLATIONS + LOSS_ABLATIONS
COMPACT_ABLATIONS = ARCHITECTURE_ABLATIONS + (
    "no_charbonnier_loss",
    "no_observation_loss",
    "no_temporal_delta_loss",
)
ABLATION_METADATA = {
    "none": {
        "display_name": "Full model",
        "replacement_mechanism": None,
    },
    "no_spatial_cross_attention": {
        "display_name": (
            "w/o spatial cross-attention "
            "(row-wise grid-aware interpolation replacement)"
        ),
        "replacement_mechanism": "row-wise grid-aware spatial interpolation",
    },
    "no_graph": {
        "display_name": "w/o local graph refinement",
        "replacement_mechanism": "identity (graph layers removed)",
    },
    "no_temporal_attention": {
        "display_name": (
            "w/o temporal attention "
            "(deterministic temporal interpolation replacement)"
        ),
        "replacement_mechanism": (
            "linear temporal interpolation with nearest-value extension"
        ),
    },
    "no_domain_adapter": {
        "display_name": "w/o domain-conditioned FiLM adapter",
        "replacement_mechanism": "identity (domain FiLM disabled)",
    },
    "no_coordinate_encoding": {
        "display_name": "w/o RIS coordinate encoding",
        "replacement_mechanism": "coordinate features omitted",
    },
    "nmse_only": {
        "display_name": "NMSE loss only",
        "replacement_mechanism": "all auxiliary losses assigned zero weight",
    },
    "no_charbonnier_loss": {
        "display_name": "w/o Charbonnier loss",
        "replacement_mechanism": "Charbonnier weight set to zero",
    },
    "no_observation_loss": {
        "display_name": "w/o observation-consistency loss",
        "replacement_mechanism": "observation-consistency weight set to zero",
    },
    "no_temporal_delta_loss": {
        "display_name": "w/o temporal-difference loss",
        "replacement_mechanism": "temporal-difference weight set to zero",
    },
}


def ablation_metadata(variant: str) -> dict[str, str | None]:
    if variant not in ABLATION_METADATA:
        raise ValueError(
            f"Unknown ablation {variant!r}; choose from {ABLATION_VARIANTS}."
        )
    return {"variant_id": variant, **ABLATION_METADATA[variant]}


def architectural_ablation(variant: str) -> str:
    if variant not in ABLATION_VARIANTS:
        raise ValueError(
            f"Unknown ablation {variant!r}; choose from {ABLATION_VARIANTS}."
        )
    return variant if variant in ARCHITECTURE_ABLATIONS else "none"


def ablated_loss_weights(
    base: LossWeights,
    variant: str,
    *,
    domain: str,
) -> LossWeights:
    if variant not in ABLATION_VARIANTS:
        raise ValueError(
            f"Unknown ablation {variant!r}; choose from {ABLATION_VARIANTS}."
        )
    weights = LossWeights(
        base.nmse,
        base.charbonnier,
        base.observation,
        base.delta if domain == "mobility" else 0.0,
    )
    if variant == "nmse_only":
        return LossWeights(weights.nmse, 0.0, 0.0, 0.0)
    if variant == "no_charbonnier_loss":
        weights.charbonnier = 0.0
    elif variant == "no_observation_loss":
        weights.observation = 0.0
    elif variant == "no_temporal_delta_loss":
        weights.delta = 0.0
    return weights


def hyperparameter_candidates(
    *,
    hidden_values: Sequence[int],
    graph_layer_values: Sequence[int],
    head_values: Sequence[int],
    dropout_values: Sequence[float],
    learning_rate_values: Sequence[float],
    weight_decay_values: Sequence[float],
    strategy: str = "random",
    max_trials: int | None = 12,
    seed: int = 123,
) -> list[dict[str, int | float]]:
    """Build a deterministic, validity-filtered search plan."""

    if strategy not in {"grid", "random"}:
        raise ValueError("strategy must be 'grid' or 'random'.")
    sequences = (
        hidden_values,
        graph_layer_values,
        head_values,
        dropout_values,
        learning_rate_values,
        weight_decay_values,
    )
    if any(not values for values in sequences):
        raise ValueError("Every hyperparameter value list must be non-empty.")
    candidates: list[dict[str, int | float]] = []
    for hidden, layers, heads, dropout, learning_rate, weight_decay in itertools.product(
        *sequences
    ):
        if hidden <= 0 or layers < 0 or heads <= 0 or hidden % heads:
            continue
        if not 0.0 <= dropout < 1.0:
            continue
        if learning_rate <= 0.0 or weight_decay < 0.0:
            continue
        candidates.append(
            {
                "hidden": int(hidden),
                "graph_layers": int(layers),
                "heads": int(heads),
                "dropout": float(dropout),
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
            }
        )
    if not candidates:
        raise ValueError("The search space contains no valid configurations.")
    if strategy == "random":
        random.Random(seed).shuffle(candidates)
    if max_trials is not None:
        if max_trials <= 0:
            raise ValueError("max_trials must be positive when supplied.")
        candidates = candidates[:max_trials]
    return candidates
