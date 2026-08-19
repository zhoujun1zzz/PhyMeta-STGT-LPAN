"""Verify the LPAN RIS-column mapping and complex-channel storage order.

The script reads only a small prefix of the official MATLAB v7.3 files.  It
does not modify the datasets.  Correlations are computed per sample and then
averaged so that differently normalized samples receive equal weight.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path

import h5py
import numpy as np


EPS = np.finfo(np.float64).eps
OBSERVED = np.arange(0, 256, 8, dtype=np.int64)
LAYOUTS = ("grouped", "interleaved")


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


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


def _samplewise_nmse(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return sample-level complex normalized error along the last axes."""
    left = left.reshape(left.shape[0], -1)
    right = right.reshape(right.shape[0], -1)
    numerator = np.sum(np.abs(left - right) ** 2, axis=1)
    denominator = np.sum(np.abs(right) ** 2, axis=1)
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


def _infer_mobility_contract(
    candidates: list[dict[str, object]],
    minimum_margin: float,
) -> dict[str, object]:
    """Rank layouts and pilot positions without assuming a time mapping."""

    if minimum_margin < 0:
        raise ValueError("minimum_margin must be non-negative.")
    ranking = sorted(candidates, key=lambda item: float(item["score"]), reverse=True)
    best = ranking[0]
    runner = ranking[1]
    best_score = float(best["score"])
    runner_score = float(runner["score"])
    margin = best_score - runner_score
    if margin < minimum_margin:
        status = "ambiguous"
        verified_layout = None
        verified_positions = None
    elif best["Yd_layout"] != best["Hd_layout"]:
        status = "inconsistent"
        verified_layout = None
        verified_positions = None
    else:
        status = "verified"
        verified_layout = best["Yd_layout"]
        verified_positions = best["pilot_positions"]
    return {
        "status": status,
        "verified_layout": verified_layout,
        "verified_pilot_positions": verified_positions,
        "inferred_Yd_layout": best["Yd_layout"],
        "inferred_Hd_layout": best["Hd_layout"],
        "inferred_pilot_positions": best["pilot_positions"],
        "best_pair_score": float(best_score),
        "runner_up_score": float(runner_score),
        "runner_up": runner,
        "margin": float(margin),
        "minimum_required_margin": float(minimum_margin),
        "criterion": (
            "highest mapping correlation/(1+NMSE), with a fixed 0.01 guardrail "
            "for each exact duplicate complex target-block pair, over the "
            "joint search of Yd layout, Hd layout, and all ordered two-of-six "
            "pilot positions at the authoritative observed RIS indices"
        ),
        "ranking": ranking,
    }


def verify_quasi(path: Path, samples: int) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        if {"input_da", "output_da"} <= set(handle):
            input_key, target_key = "input_da", "output_da"
        elif {"input_da_test", "output_da_test"} <= set(handle):
            input_key, target_key = "input_da_test", "output_da_test"
        elif {"Yd", "Hd"} <= set(handle):
            input_key, target_key = "Yd", "Hd"
        else:
            raise KeyError(f"Unsupported Quasi keys in {path}: {sorted(handle.keys())}")
    y_raw = _read_prefix(path, input_key, samples)
    h_raw = _read_prefix(path, target_key, samples)
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
        "input_key": input_key,
        "target_key": target_key,
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
    chunk_size: int = 64,
) -> dict[str, object]:
    if samples <= 0 or chunk_size <= 0:
        raise ValueError("samples and chunk_size must be positive.")
    with h5py.File(path, "r") as handle:
        y_dataset = handle["Yd"]
        h_dataset = handle["Hd"]
        count = min(samples, int(y_dataset.shape[3]))
        y_shape = [int(value) for value in y_dataset.shape]
        h_shape = [int(value) for value in h_dataset.shape]
        pair_sums: dict[tuple[str, str, int, int], tuple[float, float]] = {}
        adjacent_sums = {layout: 0.0 for layout in LAYOUTS}
        target_pair_max_abs = {
            (layout, left, right): 0.0
            for layout in LAYOUTS
            for left, right in combinations(range(6), 2)
        }
        duplicate_diagnostics = {
            "max_abs_H_raw_0_minus_2": 0.0,
            "max_abs_H_raw_6_minus_8": 0.0,
        }
        for start in range(0, count, chunk_size):
            stop = min(count, start + chunk_size)
            y_raw = np.asarray(y_dataset[:, :, :, start:stop], dtype=np.float64)
            h_raw = np.asarray(h_dataset[:, :, :, start:stop], dtype=np.float64)
            decoded_y = {layout: _decode(y_raw, 2, layout) for layout in LAYOUTS}
            decoded_h = {layout: _decode(h_raw, 6, layout) for layout in LAYOUTS}
            for layout, target in decoded_h.items():
                adjacent_sums[layout] += sum(
                    float(
                        np.sum(
                            _samplewise_correlation(
                                target[:, block], target[:, block + 1]
                            )
                        )
                    )
                    for block in range(5)
                )
                for left, right in combinations(range(6), 2):
                    key = (layout, left, right)
                    target_pair_max_abs[key] = max(
                        target_pair_max_abs[key],
                        float(np.max(np.abs(target[:, left] - target[:, right]))),
                    )
            for y_layout, obs in decoded_y.items():
                for h_layout, target in decoded_h.items():
                    for pilot in range(2):
                        for target_position in range(6):
                            expected = target[:, target_position, OBSERVED]
                            key = (y_layout, h_layout, pilot, target_position)
                            previous_correlation, previous_nmse = pair_sums.get(
                                key, (0.0, 0.0)
                            )
                            pair_sums[key] = (
                                previous_correlation
                                + float(
                                    np.sum(
                                        _samplewise_correlation(
                                            obs[:, pilot], expected
                                        )
                                    )
                                ),
                                previous_nmse
                                + float(
                                    np.sum(_samplewise_nmse(obs[:, pilot], expected))
                                ),
                            )
            duplicate_diagnostics["max_abs_H_raw_0_minus_2"] = max(
                duplicate_diagnostics["max_abs_H_raw_0_minus_2"],
                float(np.max(np.abs(h_raw[0] - h_raw[2]))),
            )
            duplicate_diagnostics["max_abs_H_raw_6_minus_8"] = max(
                duplicate_diagnostics["max_abs_H_raw_6_minus_8"],
                float(np.max(np.abs(h_raw[6] - h_raw[8]))),
            )

    adjacent = {
        layout: adjacent_sums[layout] / (5 * count) for layout in LAYOUTS
    }
    exact_duplicate_pairs = {
        layout: [
            [left, right]
            for left, right in combinations(range(6), 2)
            if target_pair_max_abs[(layout, left, right)] <= 1e-12
        ]
        for layout in LAYOUTS
    }

    candidates: list[dict[str, object]] = []
    for y_layout in LAYOUTS:
        for h_layout in LAYOUTS:
            for pilot_positions in combinations(range(6), 2):
                totals = [
                    pair_sums[(y_layout, h_layout, pilot, target_position)]
                    for pilot, target_position in enumerate(pilot_positions)
                ]
                mean_correlation = sum(value[0] for value in totals) / (2 * count)
                mean_nmse = sum(value[1] for value in totals) / (2 * count)
                mapping_score = mean_correlation / (1.0 + mean_nmse)
                temporal_coherence = adjacent[h_layout]
                duplicate_penalty = 0.01 * len(exact_duplicate_pairs[h_layout])
                candidates.append(
                    {
                        "Yd_layout": y_layout,
                        "Hd_layout": h_layout,
                        "pilot_positions": list(pilot_positions),
                        "mean_complex_correlation": mean_correlation,
                        "mean_complex_nmse": mean_nmse,
                        "mapping_score": mapping_score,
                        "target_temporal_coherence": temporal_coherence,
                        "exact_duplicate_target_pairs": exact_duplicate_pairs[
                            h_layout
                        ],
                        "duplicate_degeneracy_penalty": duplicate_penalty,
                        "score": mapping_score - duplicate_penalty,
                    }
                )

    inference = _infer_mobility_contract(candidates, minimum_layout_margin)
    verified_layout = inference["verified_layout"]
    digest = _file_sha256(path)

    return {
        "dataset_sha256": digest,
        "Yd_shape": y_shape,
        "Hd_shape": h_shape,
        "samples": count,
        "chunk_size": chunk_size,
        "observed_ris_indices_zero_based": OBSERVED.tolist(),
        "target_adjacent_block_correlation": adjacent,
        "target_exact_duplicate_block_pairs": exact_duplicate_pairs,
        "raw_duplicate_diagnostics": duplicate_diagnostics,
        "layout_inference": inference,
        "verified_layout": verified_layout,
        "verified_pilot_positions": inference["verified_pilot_positions"],
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
    parser.add_argument("--quasi", type=Path)
    parser.add_argument("--mobility", type=Path, nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
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
        "quasi": verify_quasi(args.quasi, args.samples) if args.quasi else None,
        "mobility": {
            str(path): verify_mobility(
                path,
                args.samples,
                minimum_layout_margin=args.minimum_layout_margin,
                chunk_size=args.chunk_size,
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
