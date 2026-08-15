from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lpan.complexity import canonical_batch, profile_model_complexity


class OfficialMobilityLPANLAdapter(nn.Module):
    """Adapt the external official model to the unified batch contract."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, batch: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        obs = batch["obs_h"]
        b, observed_blocks, observed_nodes, antennas, _ = obs.shape
        x = obs.permute(0, 4, 1, 3, 2).reshape(
            b, 2 * observed_blocks, antennas, observed_nodes
        )
        outputs = self.model(x)
        final = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
        query_blocks = final.shape[1] // 2
        final = final.reshape(b, 2, query_blocks, antennas, 256)
        return final.permute(0, 2, 4, 3, 1).contiguous()


def load_official_model(source_root: Path) -> nn.Module:
    source = source_root / "Mobility_LPAN_L1.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location("official_mobility_lpan_l", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LPAN_L()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile the external official progressive Mobility LPAN-L."
    )
    parser.add_argument(
        "--source-root",
        default=r"D:\数据1\LPAN_mobility_baseline",
    )
    parser.add_argument(
        "--output",
        default="reports/official_mobility_lpan_l_complexity.json",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = OfficialMobilityLPANLAdapter(
        load_official_model(Path(args.source_root))
    ).to(device)
    batch = canonical_batch("mobility", batch_size=1, device=device)
    result = {
        "model": "official_progressive_mobility_lpan_l",
        "display_name": "Official progressive Mobility LPAN-L",
        "source_root": str(Path(args.source_root).resolve()),
        **profile_model_complexity(model, batch),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
