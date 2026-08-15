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
LAYOUTS = ("grouped", "interleaved")


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


def _channel_labels(blocks: int, layout: str) -> list[str]:
    if layout == "grouped":
        return [f"Re(t{block})" for block in range(1, blocks + 1)] + [
            f"Im(t{block})" for block in range(1, blocks + 1)
        ]
    if layout == "interleaved":
        return [
            component
            for block in range(1, blocks + 1)
            for component in (f"Re(t{block})", f"Im(t{block})")
        ]
    raise ValueError(layout)


def _infer_layout(
    pair_scores: dict[tuple[str, str], float],
    minimum_margin: float,
) -> dict[str, object]:
    """Infer Yd/Hd layouts from mapped input-to-target correlations."""

    if minimum_margin < 0:
        raise ValueError("minimum_margin must be non-negative.")
    ranking = sorted(pair_scores.items(), key=lambda item: item[1], reverse=True)
    (best_y, best_h), best_score = ranking[0]
    runner_score = ranking[1][1]
    margin = best_score - runner_score
    if margin < minimum_margin:
        status = "ambiguous"
        verified_layout = None
    elif best_y != best_h:
        status = "inconsistent"
        verified_layout = None
    else:
        status = "verified"
        verified_layout = best_y
    return {
        "status": status,
        "verified_layout": verified_layout,
        "inferred_Yd_layout": best_y,
        "inferred_Hd_layout": best_h,
        "best_pair_score": float(best_score),
        "runner_up_score": float(runner_score),
        "margin": float(margin),
        "minimum_required_margin": float(minimum_margin),
        "criterion": (
            "highest correlation between Yd pilot blocks and the matching "
            "Hd blocks at the authoritative observed RIS indices"
        ),
        "ranking": [
            {
                "Yd_layout": y_layout,
                "Hd_layout": h_layout,
                "score": float(score),
            }
            for (y_layout, h_layout), score in ranking
        ],
    }


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


def verify_mobility(
    path: Path,
    samples: int,
    minimum_layout_margin: float = 1e-3,
) -> dict[str, object]:
    y_raw = _read_prefix(path, "Yd", samples)
    h_raw = _read_prefix(path, "Hd", samples)
    decoded_y = {layout: _decode(y_raw, 2, layout) for layout in LAYOUTS}
    decoded_h = {layout: _decode(h_raw, 6, layout) for layout in LAYOUTS}

    adjacent = {}
    for layout, target in decoded_h.items():
        scores = [
            _samplewise_correlation(target[:, block], target[:, block + 1])
            for block in range(5)
        ]
        adjacent[layout] = float(np.mean(np.concatenate(scores)))

    pair_scores: dict[tuple[str, str], float] = {}
    for y_layout, obs in decoded_y.items():
        for h_layout, target in decoded_h.items():
            scores = [
                _samplewise_correlation(
                    obs[:, block], target[:, block, OBSERVED]
                )
                for block in range(2)
            ]
            pair_scores[(y_layout, h_layout)] = float(
                np.mean(np.concatenate(scores))
            )

    inference = _infer_layout(pair_scores, minimum_layout_margin)
    verified_layout = inference["verified_layout"]
    mapped = {
        f"Y_{y_layout}__H_{h_layout}": score
        for (y_layout, h_layout), score in pair_scores.items()
    }

    return {
        "samples": int(y_raw.shape[3]),
        "target_adjacent_block_correlation": adjacent,
        "mapped_input_target_correlation": mapped,
        "layout_inference": inference,
        "verified_layout": verified_layout,
        "raw_Yd_channels": (
            _channel_labels(2, str(verified_layout)) if verified_layout else None
        ),
        "raw_Hd_channels": (
            _channel_labels(6, str(verified_layout)) if verified_layout else None
        ),
        "candidate_channel_orders": {
            layout: {
                "Yd": _channel_labels(2, layout),
                "Hd": _channel_labels(6, layout),
            }
            for layout in LAYOUTS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quasi", type=Path, required=True)
    parser.add_argument("--mobility", type=Path, nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument(
        "--minimum-layout-margin",
        type=float,
        default=1e-3,
        help=(
            "Minimum absolute correlation lead over the runner-up layout "
            "combination required for a verified decision."
        ),
    )
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
            str(path): verify_mobility(
                path,
                args.samples,
                minimum_layout_margin=args.minimum_layout_margin,
            )
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
