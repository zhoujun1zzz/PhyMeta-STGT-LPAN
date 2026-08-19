from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .models import (
    GridGraphConv,
    SparseGraphAttention,
)


INTERPOLATION_POLICY = (
    "excluded in full: spatial/temporal interpolation, sparse-to-dense "
    "expansion einsums, and framework interpolation kernels are not counted"
)


def canonical_batch(
    domain: str,
    *,
    batch_size: int = 1,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Create the shared profiling input without reading train/val/test data."""

    if domain not in {"quasi", "mobility"}:
        raise ValueError("domain must be 'quasi' or 'mobility'.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    observed_blocks, query_blocks, domain_id = (
        (1, 1, 0) if domain == "quasi" else (2, 6, 1)
    )
    target_device = torch.device(device)
    return {
        "obs_h": torch.zeros(
            batch_size,
            observed_blocks,
            32,
            64,
            2,
            dtype=torch.float32,
            device=target_device,
        ),
        "target_h": torch.zeros(
            batch_size,
            query_blocks,
            256,
            64,
            2,
            dtype=torch.float32,
            device=target_device,
        ),
        "obs_ris_index": torch.arange(
            0, 256, 8, dtype=torch.long, device=target_device
        ).unsqueeze(0).expand(batch_size, -1),
        "obs_time_index": torch.tensor(
            [0] if domain == "quasi" else [1, 4],
            dtype=torch.long,
            device=target_device,
        ).unsqueeze(0).expand(batch_size, -1),
        "query_time": torch.arange(
            query_blocks, dtype=torch.long, device=target_device
        ).unsqueeze(0).expand(batch_size, -1),
        "domain_id": torch.full(
            (batch_size,), domain_id, dtype=torch.long, device=target_device
        ),
        "observation_mask": torch.ones(
            batch_size,
            observed_blocks,
            32,
            dtype=torch.bool,
            device=target_device,
        ),
        "sample_index": torch.arange(
            batch_size, dtype=torch.long, device=target_device
        ),
    }


class _MacCounter:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.macs = 0
        self.handles: list[Any] = []

    def _add_conv(
        self,
        module: nn.Conv2d,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        kernel_macs = (
            module.kernel_size[0]
            * module.kernel_size[1]
            * module.in_channels
            // module.groups
        )
        self.macs += int(output.numel()) * kernel_macs

    def _add_linear(
        self,
        module: nn.Linear,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.macs += int(output.numel()) * module.in_features

    def _add_multihead_attention(
        self,
        module: nn.MultiheadAttention,
        inputs: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, torch.Tensor | None],
    ) -> None:
        query, key, value = inputs[:3]
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("Complexity profiling expects batched 3D attention inputs.")
        if module.batch_first:
            batch, query_length, embed_dim = query.shape
            key_length = key.shape[1]
            value_length = value.shape[1]
        else:
            query_length, batch, embed_dim = query.shape
            key_length = key.shape[0]
            value_length = value.shape[0]
        projection = batch * (
            query_length * embed_dim * embed_dim
            + key_length * embed_dim * embed_dim
            + value_length * embed_dim * embed_dim
            + query_length * embed_dim * embed_dim
        )
        attention = 2 * batch * query_length * key_length * embed_dim
        self.macs += int(projection + attention)

    def _add_gru(
        self,
        module: nn.GRU,
        inputs: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x = inputs[0]
        if module.batch_first:
            batch, sequence_length, _ = x.shape
        else:
            sequence_length, batch, _ = x.shape
        directions = 2 if module.bidirectional else 1
        layer_input = module.input_size
        total = 0
        for _ in range(module.num_layers):
            total += (
                batch
                * sequence_length
                * directions
                * 3
                * (
                    layer_input * module.hidden_size
                    + module.hidden_size * module.hidden_size
                )
            )
            layer_input = directions * module.hidden_size
        self.macs += int(total)

    def _add_gru_cell(
        self,
        module: nn.GRUCell,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        batch = inputs[0].numel() // module.input_size
        self.macs += int(
            batch
            * 3
            * (
                module.input_size * module.hidden_size
                + module.hidden_size * module.hidden_size
            )
        )

    def _add_grid_graph(
        self,
        module: GridGraphConv,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        x = inputs[0]
        nodes = x.shape[-2]
        prefix = x.numel() // (nodes * x.shape[-1])
        self.macs += int(
            prefix * module.adjacency._nnz() * module.linear.out_features
        )

    def _add_sparse_graph_attention(
        self,
        module: SparseGraphAttention,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        x = inputs[0]
        nodes = x.shape[-2]
        prefix = x.numel() // (nodes * module.hidden)
        edges = module.edge_index.shape[1]
        self.macs += int(2 * prefix * edges * module.hidden)

    def __enter__(self) -> _MacCounter:
        attention_children: set[int] = set()
        for module in self.model.modules():
            if isinstance(module, nn.MultiheadAttention):
                attention_children.update(id(child) for child in module.modules())
                attention_children.discard(id(module))
        for module in self.model.modules():
            if isinstance(module, nn.MultiheadAttention):
                self.handles.append(
                    module.register_forward_hook(self._add_multihead_attention)
                )
            elif id(module) in attention_children:
                continue
            elif isinstance(module, nn.Conv2d):
                self.handles.append(module.register_forward_hook(self._add_conv))
            elif isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self._add_linear))
            elif isinstance(module, nn.GRU):
                self.handles.append(module.register_forward_hook(self._add_gru))
            elif isinstance(module, nn.GRUCell):
                self.handles.append(module.register_forward_hook(self._add_gru_cell))
            elif isinstance(module, GridGraphConv):
                self.handles.append(module.register_forward_hook(self._add_grid_graph))
            elif isinstance(module, SparseGraphAttention):
                self.handles.append(
                    module.register_forward_hook(self._add_sparse_graph_attention)
                )
        return self

    def __exit__(self, *args: object) -> None:
        for handle in self.handles:
            handle.remove()


def profile_model_complexity(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    """Profile one forward pass under an explicit MAC/FLOP convention.

    Multiply-accumulate operations from convolutions, linear projections,
    recurrent cells, attention products and graph aggregation are counted.
    Bias, normalization, activation, softmax, indexing and every interpolation
    path are excluded. FLOPs use the transparent conversion 1 MAC = 2 FLOPs.
    """

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    was_training = model.training
    model.eval()
    counter = _MacCounter(model)
    with torch.inference_mode(), counter:
        output = model(batch)
    if was_training:
        model.train()
    obs = batch["obs_h"]
    batch_size = obs.shape[0]
    macs = int(counter.macs)
    flops = 2 * macs
    return {
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "macs": macs,
        "gmacs": macs / 1e9,
        "flops": flops,
        "gflops": flops / 1e9,
        "batch_size": int(batch_size),
        "input_shape": list(obs.shape),
        "output_shape": list(output.shape),
        "dtype": str(obs.dtype).replace("torch.", ""),
        "scope": "single forward pass; model operations only; no backward",
        "convention": "1 MAC = 2 FLOPs",
        "interpolation_policy": INTERPOLATION_POLICY,
        "excluded_operations": [
            "bias",
            "normalization",
            "activation",
            "softmax",
            "indexing",
            "interpolation (all spatial and temporal paths)",
        ],
    }
