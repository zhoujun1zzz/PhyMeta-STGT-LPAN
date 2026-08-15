from __future__ import annotations

import torch


def ris_index_to_grid(index: int, size: int = 16) -> tuple[int, int]:
    """Map a zero-based LPAN/SimRIS element index to UPA grid coordinates.

    SimRIS enumerates the first lattice coordinate in the outer loop and the
    second coordinate in the inner loop.  Therefore ``index = row*size+col``.
    The LPAN observations ``0,8,...,248`` occupy columns 0 and 8 in every row.
    """
    if not 0 <= index < size * size:
        raise ValueError(f"index must be in [0, {size * size - 1}]")
    return divmod(index, size)


def grid_coordinates(size: int = 16) -> torch.Tensor:
    """Normalized coordinates in the verified SimRIS row-major order."""
    axis = torch.linspace(-1.0, 1.0, size)
    row, col = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((row.reshape(-1), col.reshape(-1)), dim=-1)


def grid_edge_index(size: int = 16, eight_neighbour: bool = False) -> torch.Tensor:
    offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    if eight_neighbour:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    src, dst = [], []
    for row in range(size):
        for col in range(size):
            target = row * size + col
            for dr, dc in offsets:
                rr, cc = row + dr, col + dc
                if 0 <= rr < size and 0 <= cc < size:
                    src.append(rr * size + cc)
                    dst.append(target)
    return torch.tensor([src, dst], dtype=torch.long)


def normalized_adjacency(size: int = 16, eight_neighbour: bool = False) -> torch.Tensor:
    edge_index = grid_edge_index(size, eight_neighbour)
    n = size * size
    values = torch.ones(edge_index.shape[1])
    adjacency = torch.sparse_coo_tensor(edge_index, values, (n, n)).coalesce()
    degree = torch.sparse.sum(adjacency, dim=1).to_dense().clamp_min(1)
    src, dst = adjacency.indices()
    norm = degree[src].rsqrt() * degree[dst].rsqrt()
    return torch.sparse_coo_tensor(
        adjacency.indices(), norm, adjacency.shape
    ).coalesce()
