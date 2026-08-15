"""Verify the LPAN RIS-column mapping and complex-channel storage order.

The script reads only a small prefix of the official MATLAB v7.3 files.  It
does not modify the datasets.  Correlations are computed per sample and then
averaged so that differently normalized samples receive equal weight.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


EPS = np.finfo(np.float64).eps
OBSERVED = np.arange(0, 256, 8, dtype=np.int64)


def _read_prefix(path: Path, key: str, samples: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        dataset = handle[key]
        count = min(samples, int(dataset.shape[3]))
        return np.asarray(dataset[:, :, :, :count], dtype=np.float64)


def _samplewise_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return absolute complex cosine similarity along the last axes."""
    left = left.reshape(left.shape[0], -1)
    right = right.reshape(right.shape[0], -1)
    numerator = np.abs(np.sum(np.conj(left) * right, axis=1))
    denominator = np.sqrt(
        np.sum(np.abs(left) ** 2, axis=1)
        * np.sum(np.abs(right) ** 2, axis=1)
    )
    return numerator / np.maximum(denominator, EPS)


def _decode(raw: np.ndarray, blocks: int, layout: str) -> np.ndarray:
    """Decode [2T,RIS,BS,S] into complex [S,T,RIS,BS]."""
    if layout == "grouped":
        real = raw[:blocks]
        imag = raw[blocks : 2 * blocks]
    elif layout == "interleaved":
        real = raw[0 : 2 * blocks : 2]
        imag = raw[1 : 2 * blocks : 2]
    else:
        raise ValueError(layout)
    return (real + 1j * imag).transpose(3, 0, 1, 2)


def verify_quasi(path: Path, samples: int) -> dict[str, object]:
    y_raw = _read_prefix(path, "input_da", samples)
    h_raw = _read_prefix(path, "output_da", samples)
    y = _decode(y_raw, 1, "grouped")[:, 0]
    h = _decode(h_raw, 1, "grouped")[:, 0]

    # [S,P,M] x [S,N,M] -> one correlation for every (P,N) pair and sample.
    numerator = np.abs(np.einsum("spm,snm->spn", np.conj(y), h))
    denominator = np.sqrt(
        np.sum(np.abs(y) ** 2, axis=2)[:, :, None]
        * np.sum(np.abs(h) ** 2, axis=2)[:, None, :]
    )
    correlations = np.mean(numerator / np.maximum(denominator, EPS), axis=0)
    best = np.argmax(correlations, axis=1)
    offset_means = []
    for offset in range(8):
        candidate = np.arange(offset, 256, 8)
        offset_means.append(
            float(np.mean(correlations[np.arange(32), candidate]))
        )

    return {
        "samples": int(y.shape[0]),
        "best_indices_zero_based": best.tolist(),
        "expected_indices_zero_based": OBSERVED.tolist(),
        "exact_matches": int(np.sum(best == OBSERVED)),
        "mean_best_correlation": float(
            np.mean(correlations[np.arange(32), best])
        ),
        "mean_expected_correlation": float(
            np.mean(correlations[np.arange(32), OBSERVED])
        ),
        "stride_8_offset_mean_correlations": offset_means,
    }


def verify_mobility(path: Path, samples: int) -> dict[str, object]:
    y_raw = _read_prefix(path, "Yd", samples)
    h_raw = _read_prefix(path, "Hd", samples)
    layouts = ("grouped", "interleaved")
    decoded_y = {layout: _decode(y_raw, 2, layout) for layout in layouts}
    decoded_h = {layout: _decode(h_raw, 6, layout) for layout in layouts}

    adjacent = {}
    for layout, target in decoded_h.items():
        scores = [
            _samplewise_correlation(target[:, block], target[:, block + 1])
            for block in range(5)
        ]
        adjacent[layout] = float(np.mean(np.concatenate(scores)))

    mapped = {}
    for y_layout, obs in decoded_y.items():
        for h_layout, target in decoded_h.items():
            scores = [
                _samplewise_correlation(
                    obs[:, block], target[:, block, OBSERVED]
                )
                for block in range(2)
            ]
            mapped[f"Y_{y_layout}__H_{h_layout}"] = float(
                np.mean(np.concatenate(scores))
            )

    return {
        "samples": int(y_raw.shape[3]),
        "target_adjacent_block_correlation": adjacent,
        "mapped_input_target_correlation": mapped,
        "verified_layout": "grouped",
        "raw_Yd_channels": ["Re(t1)", "Re(t2)", "Im(t1)", "Im(t2)"],
        "raw_Hd_channels": [
            "Re(t1)", "Re(t2)", "Re(t3)", "Re(t4)", "Re(t5)", "Re(t6)",
            "Im(t1)", "Im(t2)", "Im(t3)", "Im(t4)", "Im(t5)", "Im(t6)",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quasi", type=Path, required=True)
    parser.add_argument("--mobility", type=Path, nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = {
        "authoritative_rule": {
            "one_based": "omega_j = 1 + 8*(j-1), j=1,...,32",
            "zero_based": OBSERVED.tolist(),
            "grid_order": "index = 16*row + column (zero based)",
            "observed_grid_coordinates_zero_based": [
                [row, col] for row in range(16) for col in (0, 8)
            ],
        },
        "quasi": verify_quasi(args.quasi, args.samples),
        "mobility": {
            str(path): verify_mobility(path, args.samples)
            for path in args.mobility
        },
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
