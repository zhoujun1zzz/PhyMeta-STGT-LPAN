from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from .graph import grid_coordinates, grid_edge_index, normalized_adjacency


def _shared_vector(value: torch.Tensor) -> torch.Tensor:
    return value[0] if value.ndim > 1 else value


def linear_query_weights(
    obs_time: torch.Tensor, query_time: torch.Tensor
) -> torch.Tensor:
    """Piecewise-linear weights, with nearest-value extrapolation."""
    obs = obs_time.to(dtype=torch.float32)
    query = query_time.to(dtype=torch.float32)
    if obs.numel() == 1:
        return torch.ones(query.numel(), 1, device=query.device)
    order = torch.argsort(obs)
    obs = obs[order]
    weights = torch.zeros(query.numel(), obs.numel(), device=query.device)
    for qi, value in enumerate(query):
        if value <= obs[0]:
            weights[qi, 0] = 1
        elif value >= obs[-1]:
            weights[qi, -1] = 1
        else:
            right = int(torch.searchsorted(obs, value).item())
            left = right - 1
            alpha = (value - obs[left]) / (obs[right] - obs[left])
            weights[qi, left] = 1 - alpha
            weights[qi, right] = alpha
    restored = torch.zeros_like(weights)
    restored[:, order] = weights
    return restored


def grid_aware_spatial_interpolation_weights(
    obs_index: torch.Tensor,
    nodes: int = 256,
    nearest: bool = False,
    grid_width: int = 16,
) -> torch.Tensor:
    """Build row-wise interpolation weights on the physical RIS grid.

    A query node may use observations from its own row only. Linear mode uses
    piecewise interpolation between observed columns and constant nearest-value
    extension outside their range. Nearest mode also stays within the row.
    """

    if nodes <= 0 or grid_width <= 0 or nodes % grid_width:
        raise ValueError("nodes must be positive and divisible by grid_width.")
    index = obs_index.to(dtype=torch.long)
    if index.ndim != 1 or index.numel() == 0:
        raise ValueError("obs_index must be a non-empty one-dimensional tensor.")
    if int(index.min()) < 0 or int(index.max()) >= nodes:
        raise ValueError(f"Observed RIS indices must be in [0, {nodes - 1}].")
    if torch.unique(index).numel() != index.numel():
        raise ValueError("Observed RIS indices must be unique.")
    weights = torch.zeros(
        nodes, index.numel(), device=index.device, dtype=torch.float32
    )
    observed_rows = torch.div(index, grid_width, rounding_mode="floor")
    observed_cols = index % grid_width
    for row in range(nodes // grid_width):
        row_positions = torch.where(observed_rows == row)[0]
        if row_positions.numel() == 0:
            raise ValueError(
                f"Grid-aware interpolation requires at least one observation "
                f"in RIS row {row}."
            )
        order = torch.argsort(observed_cols[row_positions])
        positions = row_positions[order]
        columns = observed_cols[positions].to(torch.float32)
        for column in range(grid_width):
            node = row * grid_width + column
            value = torch.tensor(float(column), device=index.device)
            distances = torch.abs(columns - value)
            if nearest or positions.numel() == 1:
                weights[node, positions[torch.argmin(distances)]] = 1
            elif value <= columns[0]:
                weights[node, positions[0]] = 1
            elif value >= columns[-1]:
                weights[node, positions[-1]] = 1
            else:
                right = int(torch.searchsorted(columns, value).item())
                if columns[right] == value:
                    weights[node, positions[right]] = 1
                else:
                    left = right - 1
                    alpha = (value - columns[left]) / (
                        columns[right] - columns[left]
                    )
                    weights[node, positions[left]] = 1 - alpha
                    weights[node, positions[right]] = alpha
    return weights


def spatial_interpolation_weights(
    obs_index: torch.Tensor, nodes: int = 256, nearest: bool = False
) -> torch.Tensor:
    """Compatibility name for the grid-aware interpolation implementation."""

    return grid_aware_spatial_interpolation_weights(
        obs_index, nodes=nodes, nearest=nearest
    )


def expand_observations_to_grid(
    batch: Mapping[str, torch.Tensor],
    *,
    nearest: bool = False,
) -> torch.Tensor:
    """Expand sparse observations using row-wise physical-grid interpolation."""

    obs = batch["obs_h"]
    obs_index = _shared_vector(batch["obs_ris_index"]).to(obs.device)
    weights = grid_aware_spatial_interpolation_weights(
        obs_index, nearest=nearest
    ).to(obs.dtype)
    raw_mask = batch.get("observation_mask")
    if raw_mask is None:
        return torch.einsum("np,btpmc->btnmc", weights, obs)
    mask = raw_mask.to(device=obs.device, dtype=torch.bool)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if tuple(mask.shape) != tuple(obs.shape[:3]):
        raise ValueError(
            "observation_mask must match [batch, observed_time, observed_RIS]."
        )
    # Official LPAN batches carry an explicit all-true mask. Reuse the shared
    # interpolation matrix instead of rebuilding it for every sample/time.
    if bool(mask.all()):
        return torch.einsum("np,btpmc->btnmc", weights, obs)
    batches: list[torch.Tensor] = []
    for batch_index in range(obs.shape[0]):
        times: list[torch.Tensor] = []
        for time_index in range(obs.shape[1]):
            valid_positions = torch.where(mask[batch_index, time_index])[0]
            if valid_positions.numel() == 0:
                raise ValueError(
                    "Every (sample, observed_time) must contain at least one "
                    "valid observation."
                )
            valid_indices = obs_index.index_select(0, valid_positions)
            local_weights = grid_aware_spatial_interpolation_weights(
                valid_indices, nearest=nearest
            ).to(obs.dtype)
            valid_observations = obs[batch_index, time_index].index_select(
                0, valid_positions
            )
            times.append(
                torch.einsum("np,pmc->nmc", local_weights, valid_observations)
            )
        batches.append(torch.stack(times, dim=0))
    return torch.stack(batches, dim=0)


@torch.no_grad()
def interpolation_baseline(
    batch: Mapping[str, torch.Tensor],
    *,
    spatial: str = "linear",
    temporal: str = "linear",
) -> torch.Tensor:
    obs = batch["obs_h"]
    obs_time = _shared_vector(batch["obs_time_index"]).to(obs.device)
    query_time = _shared_vector(batch["query_time"]).to(obs.device)
    if spatial == "linear":
        spatial_full = expand_observations_to_grid(batch)
    else:
        spatial_full = expand_observations_to_grid(batch, nearest=True)
    if temporal == "nearest":
        distances = torch.abs(
            query_time[:, None].float() - obs_time[None, :].float()
        )
        tw = F.one_hot(distances.argmin(dim=1), obs_time.numel()).to(obs.dtype)
    else:
        tw = linear_query_weights(obs_time, query_time).to(obs.dtype)
    return torch.einsum("qt,btnmc->bqnmc", tw, spatial_full)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.body(x)


class LPANLChannelAttention(nn.Module):
    """Tanh channel attention used by the LPAN-family baselines."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(nn.Linear(channels, channels), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.gate(self.pool(x).flatten(1)).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class LPANLResidualBlock(nn.Module):
    """LPAN-L-style dilated residual block with channel attention."""

    def __init__(self, channels: int, *, grouped: bool = False) -> None:
        super().__init__()
        self.grouped = grouped
        first_groups = 16 if grouped and channels % 16 == 0 else 1
        second_groups = 4 if grouped and channels % 4 == 0 else 1
        self.conv1 = weight_norm(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=2,
                dilation=2,
                groups=first_groups,
            )
        )
        self.conv2 = weight_norm(
            nn.Conv2d(
                channels, channels, 3, padding=1, groups=second_groups
            )
        )
        self.project = (
            weight_norm(nn.Conv2d(channels, channels, 1))
            if grouped
            else nn.Identity()
        )
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.attention = LPANLChannelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.activation(self.conv1(x))
        residual = self.conv2(residual)
        if self.grouped:
            residual = self.activation(residual)
        residual = self.project(residual)
        return x + self.attention(residual)


class LPANLDirect(nn.Module):
    """Single-stage LPAN-L-derived baseline for the repository task contract.

    Unlike the original progressive 2x/4x/8x reconstruction, this variant
    extracts features on the 32-column input and resizes them directly to the
    256-column target exactly once.  It returns only the final dense channel.
    """

    def __init__(
        self,
        obs_blocks: int,
        query_blocks: int,
        channels: int = 96,
        body_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.obs_blocks = obs_blocks
        self.query_blocks = query_blocks
        self.target_nodes = 256
        input_channels = 2 * obs_blocks
        output_channels = 2 * query_blocks
        self.head = weight_norm(nn.Conv2d(input_channels, channels, 3, padding=1))
        blocks = [LPANLResidualBlock(channels, grouped=True)]
        blocks.extend(
            LPANLResidualBlock(channels) for _ in range(max(1, body_blocks - 1))
        )
        self.body = nn.Sequential(*blocks)
        self.feature_refine = nn.Sequential(
            weight_norm(nn.Conv2d(channels, channels, 3, padding=1)),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.residual_head = weight_norm(
            nn.Conv2d(channels, output_channels, 3, padding=1)
        )
        self.skip_head = nn.Conv2d(input_channels, output_channels, 3, padding=1)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        obs = batch["obs_h"]
        b, t, p, m, _ = obs.shape
        if p != 32:
            raise ValueError(f"LPANLDirect expects 32 observed RIS columns, got {p}.")
        index = batch["obs_ris_index"].to(obs.device)
        expected = torch.arange(0, 256, 8, device=obs.device)
        if index.ndim == 1:
            compatible = index.shape == expected.shape and torch.equal(index, expected)
        else:
            compatible = (
                index.shape[-1] == expected.numel()
                and bool((index == expected.view(1, -1)).all())
            )
        if not compatible:
            raise ValueError(
                "LPAN-L-Direct supports only the verified official LPAN "
                "ordering obs_ris_index=(0,8,...,248)."
            )
        x = obs.permute(0, 4, 1, 3, 2).reshape(b, 2 * t, m, p)
        features = self.body(self.head(x))
        features = F.interpolate(
            features,
            size=(m, self.target_nodes),
            mode="nearest",
        )
        residual = self.residual_head(self.feature_refine(features))
        skip = self.skip_head(
            F.interpolate(x, size=(m, self.target_nodes), mode="nearest")
        )
        output = residual + skip
        output = output.reshape(
            b, 2, self.query_blocks, m, self.target_nodes
        )
        return output.permute(0, 2, 4, 3, 1).contiguous()


def _validate_progressive_lpan_input(
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Validate and return the unified LPAN observation tensor."""
    obs = batch["obs_h"]
    if obs.ndim != 5 or obs.shape[-1] != 2:
        raise ValueError("obs_h must have shape [B,T,32,64,2].")
    if obs.shape[2:4] != (32, 64):
        raise ValueError(
            "Progressive LPAN expects 32 observed RIS columns and 64 antennas."
        )
    index = batch["obs_ris_index"].to(obs.device)
    expected = torch.arange(0, 256, 8, device=obs.device)
    if index.ndim == 1:
        compatible = index.shape == expected.shape and torch.equal(index, expected)
    else:
        compatible = (
            index.shape[-1] == expected.numel()
            and bool((index == expected.view(1, -1)).all())
        )
    if not compatible:
        raise ValueError(
            "Progressive LPAN supports only obs_ris_index=(0,8,...,248)."
        )
    return obs


def lpan_raw_input(obs: torch.Tensor) -> torch.Tensor:
    """Restore official MAT raw order [Re0,Im0,Re1,Im1,...]."""
    if obs.ndim != 5 or obs.shape[-1] != 2:
        raise ValueError("obs must have shape [B,T,N,M,2].")
    b, t, n, m, _ = obs.shape
    return obs.permute(0, 1, 4, 3, 2).reshape(b, 2 * t, m, n).contiguous()


def lpan_grouped_raw_input(obs: torch.Tensor) -> torch.Tensor:
    """Restore grouped MAT order [Re0,Re1,...,Im0,Im1,...]."""
    if obs.ndim != 5 or obs.shape[-1] != 2:
        raise ValueError("obs must have shape [B,T,N,M,2].")
    b, t, n, m, _ = obs.shape
    return obs.permute(0, 4, 1, 3, 2).reshape(b, 2 * t, m, n).contiguous()


def lpan_raw_output(
    image: torch.Tensor, query_blocks: int
) -> torch.Tensor:
    """Decode official MAT raw order into unified complex-last tensors."""
    if image.ndim != 4 or image.shape[1] != 2 * query_blocks:
        raise ValueError(
            f"image must have shape [B,{2 * query_blocks},M,N]."
        )
    b, _, m, n = image.shape
    return (
        image.reshape(b, query_blocks, 2, m, n)
        .permute(0, 1, 4, 3, 2)
        .contiguous()
    )


def lpan_grouped_raw_output(
    image: torch.Tensor, query_blocks: int
) -> torch.Tensor:
    """Decode grouped raw channels into [B,Q,N,M,2]."""
    if image.ndim != 4 or image.shape[1] != 2 * query_blocks:
        raise ValueError(
            f"image must have shape [B,{2 * query_blocks},M,N]."
        )
    b, _, m, n = image.shape
    return (
        image.reshape(b, 2, query_blocks, m, n)
        .permute(0, 2, 4, 3, 1)
        .contiguous()
    )


def _batch_uses_grouped_layout(batch: Mapping[str, torch.Tensor]) -> bool:
    """Resolve a homogeneous batch layout without adding model parameters."""
    layout = batch.get("complex_layout_id")
    if layout is None:
        # Backwards compatibility for pre-profile callers and historical tests.
        return False
    values = layout.reshape(-1)
    if values.numel() == 0 or not bool((values == values[0]).all()):
        raise ValueError("Every item in a batch must use the same complex layout.")
    value = int(values[0].item())
    if value not in (0, 1):
        raise ValueError("complex_layout_id must be 0 (grouped) or 1 (interleaved).")
    return value == 0


class LPANResidualBlock(nn.Module):
    """Ordinary residual block from the public LPAN implementation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = weight_norm(nn.Conv2d(channels, channels, 3, padding=1))
        self.conv2 = weight_norm(nn.Conv2d(channels, channels, 3, padding=1))
        self.activation1 = nn.LeakyReLU(negative_slope=0.2)
        self.activation2 = nn.LeakyReLU(negative_slope=0.2)
        self.attention = LPANLChannelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.activation1(self.conv1(x))
        residual = self.activation2(self.conv2(residual))
        return x + self.attention(residual)


class LPANFeatureStage(nn.Module):
    """Four ordinary LPAN blocks followed by 2x feature refinement."""

    def __init__(self, channels: int = 96) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(LPANResidualBlock(channels) for _ in range(4))
        self.refine = nn.Sequential(
            weight_norm(nn.Conv2d(channels, channels, 3, padding=1)),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.refine(
            F.interpolate(x, scale_factor=(1, 2), mode="nearest")
        )


class LPANLFeatureStage(nn.Module):
    """Public LPAN-L first/second stage: one grouped plus two ordinary blocks."""

    def __init__(self, channels: int = 96) -> None:
        super().__init__()
        self.grouped_block = LPANLResidualBlock(channels, grouped=True)
        self.ordinary_blocks = nn.ModuleList(
            LPANLResidualBlock(channels) for _ in range(2)
        )
        self.refine = nn.Sequential(
            weight_norm(nn.Conv2d(channels, channels, 3, padding=1)),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Public code overwrites repeated evaluations of the same block on x.
        x = self.grouped_block(x)
        for block in self.ordinary_blocks:
            x = block(x)
        return self.refine(
            F.interpolate(x, scale_factor=(1, 2), mode="nearest")
        )


class LPANLFinalFeatureStage(nn.Module):
    """Public LPAN-L third stage with the same-input loop elided."""

    def __init__(self, channels: int = 96) -> None:
        super().__init__()
        self.grouped_block = LPANLResidualBlock(channels, grouped=True)
        self.ordinary_blocks = nn.ModuleList()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.grouped_block(x)
        return self.refine(
            F.interpolate(x, scale_factor=(1, 2), mode="nearest")
        )


class ProgressiveReconstruction(nn.Module):
    """Public LPAN feature and nearest-neighbour skip reconstruction."""

    def __init__(self, feature_channels: int, input_channels: int, output_channels: int):
        super().__init__()
        self.feature = weight_norm(
            nn.Conv2d(feature_channels, output_channels, 3, padding=1)
        )
        self.skip = nn.Conv2d(input_channels, output_channels, 3, padding=1)

    def forward(self, features: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        previous = F.interpolate(
            previous, scale_factor=(1, 2), mode="nearest"
        )
        return self.feature(features) + self.skip(previous)


class ProgressiveLPAN(nn.Module):
    """True 32->64->128->256 LPAN or LPAN-L reconstruction baseline."""

    def __init__(
        self,
        obs_blocks: int,
        query_blocks: int,
        *,
        lightweight: bool,
        domain: str,
        channels: int = 96,
    ) -> None:
        super().__init__()
        self.obs_blocks = obs_blocks
        self.query_blocks = query_blocks
        self.domain = domain
        self.lightweight = lightweight
        input_channels = 2 * obs_blocks
        output_channels = 2 * query_blocks
        self.head = weight_norm(
            nn.Conv2d(input_channels, channels, 3, padding=1)
        )
        if lightweight:
            self.feature_stages = nn.ModuleList(
                [
                    LPANLFeatureStage(channels),
                    LPANLFeatureStage(channels),
                    LPANLFinalFeatureStage(channels),
                ]
            )
        else:
            self.feature_stages = nn.ModuleList(
                LPANFeatureStage(channels) for _ in range(3)
            )
        first_reconstruction = ProgressiveReconstruction(
            channels, input_channels, output_channels
        )
        if lightweight and input_channels == output_channels:
            # Public Quasi LPAN-L reuses ImageReconstruction1 for HR2 and HR4.
            final_reconstruction = ProgressiveReconstruction(
                channels, output_channels, output_channels
            )
            reconstruction_stages = [
                first_reconstruction,
                first_reconstruction,
                final_reconstruction,
            ]
        elif lightweight:
            # Public Mobility LPAN-L reuses ImageReconstruction2 for HR4 and HR8.
            later_reconstruction = ProgressiveReconstruction(
                channels, output_channels, output_channels
            )
            reconstruction_stages = [
                first_reconstruction,
                later_reconstruction,
                later_reconstruction,
            ]
        else:
            reconstruction_stages = [
                first_reconstruction,
                ProgressiveReconstruction(
                    channels, output_channels, output_channels
                ),
                ProgressiveReconstruction(
                    channels, output_channels, output_channels
                ),
            ]
        self.reconstruction_stages = nn.ModuleList(reconstruction_stages)

    def forward_multiscale(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = _validate_progressive_lpan_input(batch)
        if obs.shape[1] != self.obs_blocks:
            raise ValueError(
                f"Expected {self.obs_blocks} observation blocks, got {obs.shape[1]}."
            )
        grouped = _batch_uses_grouped_layout(batch)
        previous = (
            lpan_grouped_raw_input(obs) if grouped else lpan_raw_input(obs)
        )
        features = self.head(previous)
        outputs: list[torch.Tensor] = []
        for feature_stage, reconstruction_stage in zip(
            self.feature_stages, self.reconstruction_stages
        ):
            features = feature_stage(features)
            previous = reconstruction_stage(features, previous)
            outputs.append(
                lpan_grouped_raw_output(previous, self.query_blocks)
                if grouped
                else lpan_raw_output(previous, self.query_blocks)
            )
        return outputs[0], outputs[1], outputs[2]

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.forward_multiscale(batch)[-1]

    def protocol_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "progressive_reconstruction": True,
            "progressive_scales": [64, 128, 256],
            "source_repository": "WiCi-Lab/LPAN",
            "domain": self.domain,
            "feature_channels": 96,
            "raw_channel_order": "profile_selected_grouped_or_interleaved",
            "semantic_adapter_parameters": 0,
        }
        if self.lightweight:
            metadata.update(
                {
                    "public_code_semantics_preserved": True,
                    "redundant_same_input_loop_elided": True,
                    "public_reconstruction_weight_sharing_preserved": True,
                    "official_mobility_model": self.domain == "mobility",
                    "source_fidelity": (
                        "public Mobility_LPAN_L1.py"
                        if self.domain == "mobility"
                        else "public LPAN_L.py"
                    ),
                }
            )
        elif self.domain == "mobility":
            metadata.update(
                {
                    "source_fidelity": (
                        "LPAN architecture with mobility channel adaptation"
                    ),
                    "official_mobility_model": False,
                }
            )
        else:
            metadata.update(
                {
                    "official_mobility_model": False,
                    "source_fidelity": "public LPAN.py",
                }
            )
        return metadata


class EDSRLite(nn.Module):
    def __init__(
        self, obs_blocks: int, query_blocks: int, hidden: int = 48, layers: int = 4
    ) -> None:
        super().__init__()
        self.obs_blocks = obs_blocks
        self.query_blocks = query_blocks
        self.head = nn.Conv2d(2 * obs_blocks, hidden, 3, padding=1)
        self.body = nn.Sequential(*(ResidualBlock(hidden) for _ in range(layers)))
        self.tail = nn.Conv2d(hidden, 2 * query_blocks, 3, padding=1)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        full = expand_observations_to_grid(batch)
        b, t, _, m, _ = full.shape
        x = full.permute(0, 1, 4, 3, 2).reshape(b, 2 * t, m, 256)
        x = self.tail(self.body(self.head(x)))
        x = x.reshape(b, self.query_blocks, 2, m, 256)
        return x.permute(0, 1, 4, 3, 2).contiguous()


class GridGraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer("adjacency", normalized_adjacency(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., N, F]. Sparse multiplication avoids a torch-geometric dependency.
        shape = x.shape
        n = shape[-2]
        projected = self.linear(x).reshape(-1, n, self.linear.out_features)
        columns = projected.permute(1, 0, 2).reshape(n, -1)
        mixed = torch.sparse.mm(self.adjacency, columns)
        return mixed.reshape(n, projected.shape[0], -1).permute(1, 0, 2).reshape(
            *shape[:-2], n, -1
        )


class SpatialGCN(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 3) -> None:
        super().__init__()
        self.input = nn.Linear(64 * 2 + 3, hidden)
        self.layers = nn.ModuleList(
            GridGraphConv(hidden, hidden) for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.output = nn.Linear(hidden, 64 * 2)
        self.register_buffer("node_xy", grid_coordinates(), persistent=False)

    def protocol_metadata(self) -> dict[str, object]:
        return {
            "spatial_initialization": "row_wise_physical_grid_interpolation",
            "node_features": [
                "interpolated_complex_channel",
                "sparse_observation_mask",
                "normalized_physical_coordinate",
            ],
            "graph_role": "residual_refinement",
            "mobility_temporal_policy": (
                "q1_q4_piecewise_linear_with_nearest_extension"
            ),
            "mobility_positioning": "spatial_only_control",
        }

    def encode(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        obs = batch["obs_h"]
        b, t, p, _, _ = obs.shape
        index = _shared_vector(batch["obs_ris_index"]).to(obs.device)
        full = expand_observations_to_grid(batch).reshape(b, t, 256, 128)
        raw_mask = batch.get("observation_mask")
        if raw_mask is None:
            observed_mask = torch.ones(b, t, p, device=obs.device, dtype=obs.dtype)
        else:
            observed_mask = raw_mask.to(device=obs.device, dtype=obs.dtype)
            if observed_mask.ndim == 2:
                observed_mask = observed_mask.unsqueeze(0)
            if tuple(observed_mask.shape) != (b, t, p):
                raise ValueError(
                    "observation_mask must match [batch, observed_time, observed_RIS]."
                )
        observed_mask = observed_mask.unsqueeze(-1)
        mask = obs.new_zeros(b, t, 256, 1)
        mask[:, :, index] = observed_mask
        xy = self.node_xy.to(obs).view(1, 1, 256, 2).expand(b, t, -1, -1)
        x = self.input(torch.cat((full, mask, xy), dim=-1))
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + F.gelu(layer(x)))
        return x

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        x = self.encode(batch)
        obs_time = _shared_vector(batch["obs_time_index"]).to(x.device)
        query_time = _shared_vector(batch["query_time"]).to(x.device)
        weights = linear_query_weights(obs_time, query_time).to(x.dtype)
        x = torch.einsum("qt,btnh->bqnh", weights, x)
        b, q, n, _ = x.shape
        return self.output(x).reshape(b, q, n, 64, 2)


class CNNGRU(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.Sequential(
            nn.Conv2d(2, hidden, 3, padding=1),
            nn.GELU(),
            ResidualBlock(hidden),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.time_decoder = AlignedTemporalDecoder(hidden)
        self.head = nn.Linear(hidden, 2)

    def protocol_metadata(self) -> dict[str, object]:
        return {
            "temporal_alignment": "anchor_interpolation_plus_time_conditioned_gru",
            "observed_query_policy": "q1_h0_q4_h1_exact",
            "unobserved_query_policy": (
                "piecewise_linear_anchor_hidden_with_nearest_extrapolation_"
                "then_one_time_conditioned_gru_cell"
            ),
            "mobility_future_recurrent_steps": 0,
            "hidden_width_policy": "registry_hidden_used_without_scaling",
        }

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        full = expand_observations_to_grid(batch)
        b, t, _, m, _ = full.shape
        x = full.permute(0, 1, 4, 3, 2).reshape(b * t, 2, m, 256)
        x = self.encoder(x).reshape(b, t, self.hidden, m, 256)
        x = x.permute(0, 3, 4, 1, 2).reshape(b * m * 256, t, self.hidden)
        observed_states, _ = self.gru(x)
        obs_time = _shared_vector(batch["obs_time_index"]).to(x.device)
        query_time = _shared_vector(batch["query_time"]).to(x.device)
        x = self.time_decoder(observed_states, obs_time, query_time)
        q = x.shape[1]
        x = self.head(x).reshape(b, m, 256, q, 2)
        return x.permute(0, 3, 2, 1, 4).contiguous()


class GCNGRU(SpatialGCN):
    def __init__(self, hidden: int = 64, layers: int = 3) -> None:
        super().__init__(hidden, layers)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.time_decoder = AlignedTemporalDecoder(hidden)

    def protocol_metadata(self) -> dict[str, object]:
        return {
            **super().protocol_metadata(),
            "mobility_positioning": "spatiotemporal_baseline",
            "temporal_alignment": "anchor_interpolation_plus_time_conditioned_gru",
            "observed_query_policy": "q1_h0_q4_h1_exact",
            "unobserved_query_policy": (
                "piecewise_linear_anchor_hidden_with_nearest_extrapolation_"
                "then_one_time_conditioned_gru_cell"
            ),
            "mobility_future_recurrent_steps": 0,
        }

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        x = self.encode(batch)
        b, t, n, h = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * n, t, h)
        observed_states, _ = self.gru(x)
        obs_time = _shared_vector(batch["obs_time_index"]).to(x.device)
        query_time = _shared_vector(batch["query_time"]).to(x.device)
        x = self.time_decoder(observed_states, obs_time, query_time)
        q = x.shape[1]
        x = self.output(x).reshape(b, n, q, 64, 2)
        return x.permute(0, 2, 1, 3, 4).contiguous()


class AlignedTemporalDecoder(nn.Module):
    """Decode arbitrary queries from sparse temporal anchor states.

    Observed-time queries return their exact anchor states. Other queries use
    piecewise-linear anchor interpolation (nearest extension outside the
    anchors) followed by one time-conditioned GRUCell update. Queries are not
    assumed to be future-only.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.cell = nn.GRUCell(hidden, hidden)

    def forward(
        self,
        observed_states: torch.Tensor,
        obs_time: torch.Tensor,
        query_time: torch.Tensor,
    ) -> torch.Tensor:
        if observed_states.ndim != 3:
            raise ValueError("observed_states must have shape [items, observed_time, hidden].")
        obs_time = obs_time.reshape(-1).to(observed_states.device)
        query_time = query_time.reshape(-1).to(observed_states.device)
        if observed_states.shape[1] != obs_time.numel():
            raise ValueError("observed_states and obs_time lengths must match.")
        if obs_time.numel() == 0 or query_time.numel() == 0:
            raise ValueError("obs_time and query_time must be non-empty.")
        if obs_time.numel() > 1 and not bool(torch.all(obs_time[1:] > obs_time[:-1])):
            raise ValueError("obs_time must be strictly increasing.")

        matches = query_time[:, None] == obs_time[None, :]
        matched = matches.any(dim=1)
        weights = linear_query_weights(obs_time, query_time).to(observed_states)
        output = torch.einsum("qt,ith->iqh", weights, observed_states)
        if bool(matched.any()):
            observed_indices = matches[matched].to(torch.int64).argmax(dim=1)
            output[:, matched] = observed_states.index_select(1, observed_indices)

        unobserved = ~matched
        if bool(unobserved.any()):
            items, _, hidden = output.shape
            scale = max(
                1,
                int(torch.max(torch.abs(torch.cat((obs_time, query_time)))).item()),
            )
            times = query_time[unobserved].to(observed_states)
            features = self.time_encoder(times.reshape(-1, 1) / scale)
            anchors = output[:, unobserved]
            expanded = features.unsqueeze(0).expand(items, -1, -1)
            decoded = self.cell(
                expanded.reshape(-1, hidden), anchors.reshape(-1, hidden)
            )
            output[:, unobserved] = decoded.reshape(items, -1, hidden)
        return output


class SparseGraphAttention(nn.Module):
    def __init__(self, hidden: int, heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads.")
        self.hidden, self.heads, self.head_dim = hidden, heads, hidden // heads
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.edge_bias = nn.Sequential(
            nn.Linear(3, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, heads),
        )
        self.out = nn.Linear(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden),
        )
        self.dropout = nn.Dropout(dropout)
        edge_index = grid_edge_index()
        coordinates = grid_coordinates()
        delta = coordinates[edge_index[0]] - coordinates[edge_index[1]]
        edge_features = torch.cat(
            (delta, torch.linalg.vector_norm(delta, dim=-1, keepdim=True)), dim=-1
        )
        self.register_buffer("edge_index", edge_index, persistent=False)
        self.register_buffer("edge_features", edge_features, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x2 = x.reshape(-1, original_shape[-2], self.hidden)
        bt, n, _ = x2.shape
        q, k, v = self.qkv(self.norm1(x2)).chunk(3, dim=-1)
        q = q.reshape(bt, n, self.heads, self.head_dim)
        k = k.reshape(bt, n, self.heads, self.head_dim)
        v = v.reshape(bt, n, self.heads, self.head_dim)
        src, dst = self.edge_index
        score = (q[:, dst] * k[:, src]).sum(-1) / math.sqrt(self.head_dim)
        score = score + self.edge_bias(self.edge_features).unsqueeze(0)
        scatter_index = dst.view(1, -1, 1).expand(bt, -1, self.heads)
        maxima = score.new_full((bt, n, self.heads), -torch.inf)
        maxima.scatter_reduce_(
            1, scatter_index, score, reduce="amax", include_self=True
        )
        weight = torch.exp(score - maxima.gather(1, scatter_index))
        denom = score.new_zeros(bt, n, self.heads)
        denom.scatter_add_(1, scatter_index, weight)
        weight = weight / denom.gather(1, scatter_index).clamp_min(1e-9)
        messages = weight[..., None] * v[:, src]
        out = v.new_zeros(bt, n, self.heads, self.head_dim)
        out.scatter_add_(
            1,
            dst.view(1, -1, 1, 1).expand(bt, -1, self.heads, self.head_dim),
            messages,
        )
        x2 = x2 + self.dropout(self.out(out.reshape(bt, n, self.hidden)))
        x2 = x2 + self.dropout(self.ffn(self.norm2(x2)))
        return x2.reshape(original_shape)


class PhyMetaSTGT(nn.Module):
    """LPAN-compatible cross-attention + local graph + time-query model."""

    def __init__(
        self,
        hidden: int = 64,
        heads: int = 4,
        graph_layers: int = 2,
        dropout: float = 0.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        valid_ablations = {
            "none",
            "no_spatial_cross_attention",
            "no_graph",
            "no_temporal_attention",
            "no_domain_adapter",
            "no_coordinate_encoding",
        }
        if ablation not in valid_ablations:
            raise ValueError(
                f"Unknown architectural ablation {ablation!r}; choose from "
                f"{sorted(valid_ablations)}."
            )
        self.ablation = ablation
        self.hidden = hidden
        self.channel_encoder = nn.Sequential(
            nn.Linear(128, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.coord_encoder = (
            None
            if ablation == "no_coordinate_encoding"
            else nn.Sequential(nn.Linear(2, hidden), nn.GELU())
        )
        self.node_query = (
            None
            if ablation == "no_spatial_cross_attention"
            else nn.Parameter(torch.randn(256, hidden) * 0.02)
        )
        self.spatial_cross_attention = (
            None
            if ablation == "no_spatial_cross_attention"
            else nn.MultiheadAttention(
                hidden, heads, dropout=dropout, batch_first=True
            )
        )
        self.graph_layers = nn.ModuleList(
            SparseGraphAttention(hidden, heads, dropout)
            for _ in range(0 if ablation == "no_graph" else graph_layers)
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.temporal_attention = (
            None
            if ablation == "no_temporal_attention"
            else nn.MultiheadAttention(
                hidden, heads, dropout=dropout, batch_first=True
            )
        )
        self.temporal_norm = nn.LayerNorm(hidden)
        self.domain_embedding = (
            None
            if ablation == "no_domain_adapter"
            else nn.Embedding(2, 2 * hidden)
        )
        if self.domain_embedding is not None:
            nn.init.zeros_(self.domain_embedding.weight)
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, 128),
        )
        self.register_buffer("node_xy", grid_coordinates(), persistent=False)

    def spatial_parameters(self):
        modules = [self.channel_encoder, self.coord_encoder, self.spatial_cross_attention]
        modules.append(self.graph_layers)
        for module in modules:
            if module is not None:
                yield from module.parameters()
        if self.node_query is not None:
            yield self.node_query

    def encode_space(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        obs = batch["obs_h"]
        b, t, p, _, _ = obs.shape
        index = _shared_vector(batch["obs_ris_index"]).to(obs.device)
        xy = self.node_xy.to(obs)
        observed = self.channel_encoder(obs.reshape(b * t, p, 128))
        if self.coord_encoder is not None:
            observed = observed + self.coord_encoder(xy[index]).unsqueeze(0)
        observation_mask = batch["observation_mask"].reshape(b * t, p).bool()
        if not bool(observation_mask.any(dim=1).all()):
            raise ValueError(
                "Every (sample, observed_time) must contain at least one "
                "valid observation token."
            )
        observed = observed * observation_mask.unsqueeze(-1)
        if self.spatial_cross_attention is None:
            dense = expand_observations_to_grid(batch).reshape(b * t, 256, 128)
            x = self.channel_encoder(dense)
            if self.coord_encoder is not None:
                x = x + self.coord_encoder(xy).unsqueeze(0)
        else:
            assert self.node_query is not None
            query = self.node_query.to(obs).unsqueeze(0)
            if self.coord_encoder is not None:
                query = query + self.coord_encoder(xy).unsqueeze(0)
            query = query.expand(b * t, -1, -1)
            x, _ = self.spatial_cross_attention(
                query,
                observed,
                observed,
                key_padding_mask=~observation_mask,
                need_weights=False,
            )
        x = x.reshape(b, t, 256, self.hidden)
        for layer in self.graph_layers:
            x = layer(x)
        return x

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        spatial = self.encode_space(batch)
        b, t, n, h = spatial.shape
        obs_time = _shared_vector(batch["obs_time_index"]).to(spatial.device)
        query_time = _shared_vector(batch["query_time"]).to(spatial.device)
        scale = max(1, int(query_time.max().item()))
        obs_pos = self.time_encoder((obs_time.float() / scale).view(t, 1))
        query_pos = self.time_encoder(
            (query_time.float() / scale).view(query_time.numel(), 1)
        )
        query = query_pos.unsqueeze(0).expand(b * n, -1, -1)
        if self.temporal_attention is None:
            weights = linear_query_weights(obs_time, query_time).to(spatial.dtype)
            temporal = torch.einsum("qt,btnh->bnqh", weights, spatial)
            q = temporal.shape[2]
            temporal = temporal.reshape(b * n, q, h)
            temporal = self.temporal_norm(temporal + query)
        else:
            memory = spatial.permute(0, 2, 1, 3).reshape(b * n, t, h)
            memory = memory + obs_pos.unsqueeze(0)
            temporal, _ = self.temporal_attention(
                query,
                memory,
                memory,
                need_weights=False,
            )
            temporal = self.temporal_norm(temporal + query)
            q = temporal.shape[1]
        temporal = temporal.reshape(b, n, q, h)
        if self.domain_embedding is not None:
            domain = batch["domain_id"].reshape(b).to(spatial.device)
            gamma, beta = self.domain_embedding(domain).chunk(2, dim=-1)
            temporal = temporal * (1 + gamma[:, None, None]) + beta[:, None, None]
        output = self.decoder(temporal).reshape(b, n, q, 64, 2)
        return output.permute(0, 2, 1, 3, 4).contiguous()


def build_model(
    name: str,
    *,
    domain: str = "mobility",
    hidden: int = 64,
    graph_layers: int = 2,
    heads: int = 4,
    dropout: float = 0.0,
    ablation: str = "none",
) -> nn.Module:
    name = name.lower().replace("-", "_")
    blocks = (1, 1) if domain == "quasi" else (2, 6)
    if name in {"edsr", "edsr_lite"}:
        return EDSRLite(*blocks, hidden=max(32, hidden), layers=graph_layers + 2)
    if name in {"lpan_l_direct", "lpanl_direct"}:
        return LPANLDirect(*blocks)
    if name in {"lpan_progressive", "lpan"}:
        return ProgressiveLPAN(*blocks, lightweight=False, domain=domain)
    if name == "lpan_l_progressive":
        return ProgressiveLPAN(*blocks, lightweight=True, domain=domain)
    if name in {"spatial_gcn", "gcn"}:
        return SpatialGCN(hidden, graph_layers)
    if name in {"cnn_gru", "cnngru"}:
        return CNNGRU(hidden)
    if name in {"gcn_gru", "gcngru"}:
        return GCNGRU(hidden, graph_layers)
    if name in {"phymeta_stgt", "stgt", "ours"}:
        return PhyMetaSTGT(hidden, heads, graph_layers, dropout, ablation)
    raise ValueError(
        f"Unknown model {name!r}; choose lpan_progressive, "
        "lpan_l_progressive, lpan_l_direct, edsr_lite, spatial_gcn, "
        "cnn_gru, gcn_gru, or phymeta_stgt."
    )
