from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch

from .metrics import MetricAccumulator


def _complex_design(
    batch: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    obs = torch.view_as_complex(batch["obs_h"].contiguous())
    target = torch.view_as_complex(batch["target_h"].contiguous())
    # Treat each BS antenna as a regression row, shared across antennas.
    x = obs.permute(0, 3, 1, 2).reshape(-1, obs.shape[1] * obs.shape[2])
    y = target.permute(0, 3, 1, 2).reshape(
        -1, target.shape[1] * target.shape[2]
    )
    return x.to(torch.complex128), y.to(torch.complex128)


@dataclass
class RidgeStatistics:
    gram: torch.Tensor
    cross: torch.Tensor
    rows: int
    query_blocks: int

    @classmethod
    def accumulate(
        cls, loader: Iterable[Mapping[str, torch.Tensor]]
    ) -> "RidgeStatistics":
        gram = cross = None
        rows = query_blocks = 0
        for batch in loader:
            x, y = _complex_design(batch)
            if gram is None:
                gram = torch.zeros(
                    x.shape[1], x.shape[1], dtype=torch.complex128
                )
                cross = torch.zeros(
                    x.shape[1], y.shape[1], dtype=torch.complex128
                )
                query_blocks = int(batch["target_h"].shape[1])
            gram += x.mH @ x
            cross += x.mH @ y
            rows += x.shape[0]
        if gram is None or cross is None:
            raise ValueError("Cannot fit ridge on an empty loader.")
        return cls(gram, cross, rows, query_blocks)

    def solve(self, regularization: float) -> "EmpiricalRidge":
        scale = self.gram.diag().real.mean().clamp_min(1e-12)
        identity = torch.eye(self.gram.shape[0], dtype=self.gram.dtype)
        coefficient = torch.linalg.solve(
            self.gram + regularization * scale * identity, self.cross
        )
        return EmpiricalRidge(coefficient, self.query_blocks, regularization)


class EmpiricalRidge:
    def __init__(
        self,
        coefficient: torch.Tensor,
        query_blocks: int,
        regularization: float,
    ) -> None:
        self.coefficient = coefficient.cpu()
        self.query_blocks = query_blocks
        self.regularization = regularization

    def predict(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        x, _ = _complex_design(batch)
        y = x @ self.coefficient
        b = batch["obs_h"].shape[0]
        m = batch["obs_h"].shape[3]
        y = y.reshape(b, m, self.query_blocks, 256).permute(0, 2, 3, 1)
        return torch.view_as_real(y.to(torch.complex64)).contiguous()

    def evaluate(
        self, loader: Iterable[Mapping[str, torch.Tensor]]
    ) -> dict[str, object]:
        metrics = MetricAccumulator()
        for batch in loader:
            metrics.update(self.predict(batch), batch)
        result = metrics.compute()
        result["regularization"] = self.regularization
        return result

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "coefficient": self.coefficient,
                "query_blocks": self.query_blocks,
                "regularization": self.regularization,
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EmpiricalRidge":
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        return cls(**state)
