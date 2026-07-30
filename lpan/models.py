from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

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


def spatial_interpolation_weights(
    obs_index: torch.Tensor, nodes: int = 256, nearest: bool = False
) -> torch.Tensor:
    index = obs_index.to(dtype=torch.float32)
    query = torch.arange(nodes, device=index.device, dtype=torch.float32)
    weights = torch.zeros(nodes, index.numel(), device=index.device)
    for ni, value in enumerate(query):
        distances = torch.abs(index - value)
        if nearest or value <= index.min() or value >= index.max():
            weights[ni, torch.argmin(distances)] = 1
            continue
        right = int(torch.searchsorted(index, value).item())
        if index[right] == value:
            weights[ni, right] = 1
        else:
            left = right - 1
            alpha = (value - index[left]) / (index[right] - index[left])
            weights[ni, left] = 1 - alpha
            weights[ni, right] = alpha
    return weights


@torch.no_grad()
def interpolation_baseline(
    batch: Mapping[str, torch.Tensor],
    *,
    spatial: str = "linear",
    temporal: str = "linear",
) -> torch.Tensor:
    obs = batch["obs_h"]
    obs_index = _shared_vector(batch["obs_ris_index"]).to(obs.device)
    obs_time = _shared_vector(batch["obs_time_index"]).to(obs.device)
    query_time = _shared_vector(batch["query_time"]).to(obs.device)
    sw = spatial_interpolation_weights(
        obs_index, nearest=(spatial == "nearest")
    ).to(obs.dtype)
    spatial_full = torch.einsum("np,btpmc->btnmc", sw, obs)
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
        obs = batch["obs_h"]
        b, t, _, m, _ = obs.shape
        x = obs.permute(0, 1, 4, 3, 2).reshape(b, 2 * t, m, -1)
        x = F.interpolate(x, size=(m, 256), mode="bilinear", align_corners=False)
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

    def encode(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        obs = batch["obs_h"]
        b, t, p, _, _ = obs.shape
        index = _shared_vector(batch["obs_ris_index"]).to(obs.device)
        full = obs.new_zeros(b, t, 256, 128)
        observed_mask = batch["observation_mask"].to(obs).unsqueeze(-1)
        full[:, :, index] = obs.reshape(b, t, p, 128) * observed_mask
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
        self.head = nn.Linear(hidden, 2)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        obs = batch["obs_h"]
        b, t, _, m, _ = obs.shape
        x = obs.permute(0, 1, 4, 3, 2).reshape(b * t, 2, m, -1)
        x = F.interpolate(x, size=(m, 256), mode="bilinear", align_corners=False)
        x = self.encoder(x).reshape(b, t, self.hidden, m, 256)
        x = x.permute(0, 3, 4, 1, 2).reshape(b * m * 256, t, self.hidden)
        x, _ = self.gru(x)
        obs_time = _shared_vector(batch["obs_time_index"]).to(x.device)
        query_time = _shared_vector(batch["query_time"]).to(x.device)
        weights = linear_query_weights(obs_time, query_time).to(x.dtype)
        x = torch.einsum("qt,bth->bqh", weights, x)
        q = x.shape[1]
        x = self.head(x).reshape(b, m, 256, q, 2)
        return x.permute(0, 3, 2, 1, 4).contiguous()


class GCNGRU(SpatialGCN):
    def __init__(self, hidden: int = 64, layers: int = 3) -> None:
        super().__init__(hidden, layers)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        x = self.encode(batch)
        b, t, n, h = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * n, t, h)
        x, _ = self.gru(x)
        obs_time = _shared_vector(batch["obs_time_index"]).to(x.device)
        query_time = _shared_vector(batch["query_time"]).to(x.device)
        weights = linear_query_weights(obs_time, query_time).to(x.dtype)
        x = torch.einsum("qt,bth->bqh", weights, x)
        q = x.shape[1]
        x = self.output(x).reshape(b, n, q, 64, 2)
        return x.permute(0, 2, 1, 3, 4).contiguous()


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
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.channel_encoder = nn.Sequential(
            nn.Linear(128, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.coord_encoder = nn.Sequential(nn.Linear(2, hidden), nn.GELU())
        self.node_query = nn.Parameter(torch.randn(256, hidden) * 0.02)
        self.spatial_cross_attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.graph_layers = nn.ModuleList(
            SparseGraphAttention(hidden, heads, dropout)
            for _ in range(graph_layers)
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.temporal_attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(hidden)
        self.domain_embedding = nn.Embedding(2, 2 * hidden)
        nn.init.zeros_(self.domain_embedding.weight)
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, 128),
        )
        self.register_buffer("node_xy", grid_coordinates(), persistent=False)

    def spatial_parameters(self):
        modules = [
            self.channel_encoder,
            self.coord_encoder,
            self.spatial_cross_attention,
            self.graph_layers,
        ]
        for module in modules:
            yield from module.parameters()
        yield self.node_query

    def encode_space(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        obs = batch["obs_h"]
        b, t, p, _, _ = obs.shape
        index = _shared_vector(batch["obs_ris_index"]).to(obs.device)
        xy = self.node_xy.to(obs)
        observed = self.channel_encoder(obs.reshape(b * t, p, 128))
        observed = observed + self.coord_encoder(xy[index]).unsqueeze(0)
        observation_mask = batch["observation_mask"].reshape(b * t, p).bool()
        observed = observed * observation_mask.unsqueeze(-1)
        query = self.node_query.to(obs).unsqueeze(0) + self.coord_encoder(xy).unsqueeze(0)
        query = query.expand(b * t, -1, -1)
        x, _ = self.spatial_cross_attention(
            query,
            observed,
            observed,
            key_padding_mask=~observation_mask,
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
        memory = spatial.permute(0, 2, 1, 3).reshape(b * n, t, h)
        memory = memory + obs_pos.unsqueeze(0)
        query = query_pos.unsqueeze(0).expand(b * n, -1, -1)
        temporal, _ = self.temporal_attention(query, memory, memory)
        temporal = self.temporal_norm(temporal + query)
        domain = batch["domain_id"].reshape(b).to(spatial.device)
        gamma, beta = self.domain_embedding(domain).chunk(2, dim=-1)
        q = temporal.shape[1]
        temporal = temporal.reshape(b, n, q, h)
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
) -> nn.Module:
    name = name.lower().replace("-", "_")
    blocks = (1, 1) if domain == "quasi" else (2, 6)
    if name in {"edsr", "edsr_lite"}:
        return EDSRLite(*blocks, hidden=max(32, hidden), layers=graph_layers + 2)
    if name in {"spatial_gcn", "gcn"}:
        return SpatialGCN(hidden, graph_layers)
    if name in {"cnn_gru", "cnngru"}:
        return CNNGRU(max(16, hidden // 2))
    if name in {"gcn_gru", "gcngru"}:
        return GCNGRU(hidden, graph_layers)
    if name in {"phymeta_stgt", "stgt", "ours"}:
        return PhyMetaSTGT(hidden, heads, graph_layers, dropout)
    raise ValueError(
        f"Unknown model {name!r}; choose edsr_lite, spatial_gcn, cnn_gru, "
        "gcn_gru, or phymeta_stgt."
    )
