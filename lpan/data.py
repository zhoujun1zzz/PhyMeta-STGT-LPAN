from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetSpec:
    domain: str
    obs_blocks: int
    query_blocks: int
    domain_id: int
    obs_time_index: tuple[int, ...]


SPECS = {
    "quasi": DatasetSpec("quasi", 1, 1, 0, (0,)),
    # The two pilot blocks are the first two blocks in the six-block frame.
    "mobility": DatasetSpec("mobility", 2, 6, 1, (0, 1)),
}


def default_observed_ris_indices() -> tuple[int, ...]:
    """Return the verified LPAN 32-pilot linear indices (zero based).

    The LPAN paper defines the one-based subset as
    ``{1, 1 + Gamma, ..., 1 + (P - 1) * Gamma}``.  Here ``P=32`` and
    ``Gamma=8``, so the stored Yd columns map to ``0, 8, ..., 248``.
    """
    return tuple(range(0, 256, 8))


class LPANH5Dataset(Dataset):
    """Worker-safe lazy reader returning the unified complex-last contract."""

    def __init__(
        self,
        mat_path: str | Path,
        domain: str,
        split: str,
        *,
        obs_ris_index: Sequence[int] | None = None,
        obs_time_index: Sequence[int] | None = None,
        # Official LPAN mobility files store all real time blocks first,
        # followed by all imaginary time blocks.
        complex_layout: str = "grouped",
        max_samples: int | None = None,
        fraction: float = 1.0,
        subset_seed: int = 123,
    ) -> None:
        if domain not in SPECS:
            raise ValueError(f"domain must be one of {sorted(SPECS)}, got {domain!r}")
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1].")
        self.path = Path(mat_path)
        self.domain = domain
        self.split = split
        self.spec = SPECS[domain]
        if complex_layout not in {"grouped", "interleaved"}:
            raise ValueError("complex_layout must be grouped or interleaved.")
        self.complex_layout = complex_layout
        self.obs_ris_index = tuple(
            default_observed_ris_indices()
            if obs_ris_index is None
            else (int(x) for x in obs_ris_index)
        )
        self.obs_time_index = tuple(
            self.spec.obs_time_index
            if obs_time_index is None
            else (int(x) for x in obs_time_index)
        )
        if len(self.obs_ris_index) != 32 or len(set(self.obs_ris_index)) != 32:
            raise ValueError("obs_ris_index must contain 32 unique indices.")
        if min(self.obs_ris_index) < 0 or max(self.obs_ris_index) >= 256:
            raise ValueError("RIS indices must be in [0, 255].")
        if any(
            left >= right
            for left, right in zip(self.obs_ris_index, self.obs_ris_index[1:])
        ):
            raise ValueError(
                "obs_ris_index must be strictly increasing and ordered exactly "
                "like the 32 Yd columns."
            )
        if len(self.obs_time_index) != self.spec.obs_blocks:
            raise ValueError(
                f"{domain} requires {self.spec.obs_blocks} observation times."
            )
        if min(self.obs_time_index) < 0 or max(self.obs_time_index) >= self.spec.query_blocks:
            raise ValueError("Observation time indices fall outside the target frame.")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if not h5py.is_hdf5(self.path):
            raise ValueError(f"Not an HDF5/MATLAB v7.3 file: {self.path}")

        with h5py.File(self.path, "r") as handle:
            self.input_key, self.target_key = self._resolve_keys(handle)
            y_shape = tuple(int(v) for v in handle[self.input_key].shape)
            h_shape = tuple(int(v) for v in handle[self.target_key].shape)
            self._validate_shapes(y_shape, h_shape)
            total = y_shape[3]

        count = total if max_samples is None else min(total, int(max_samples))
        count = max(1, int(np.floor(count * fraction)))
        if fraction < 1.0 or max_samples is not None:
            rng = np.random.default_rng(subset_seed)
            # One seeded permutation makes fractions and max-sample caps nested.
            indices = rng.permutation(total)[:count]
            self.indices = np.sort(indices).astype(np.int64)
        else:
            self.indices = np.arange(count, dtype=np.int64)

        self.total_samples_in_file = total
        self._file: h5py.File | None = None
        self._input: h5py.Dataset | None = None
        self._target: h5py.Dataset | None = None

    def _resolve_keys(self, handle: h5py.File) -> tuple[str, str]:
        keys = set(handle.keys())
        if {"Yd", "Hd"} <= keys:
            return "Yd", "Hd"
        if self.domain == "quasi":
            if self.split == "train" and {"input_da", "output_da"} <= keys:
                return "input_da", "output_da"
            if self.split in {"validation", "val", "internal_validation"} and {
                "input_da_test",
                "output_da_test",
            } <= keys:
                return "input_da_test", "output_da_test"
        raise KeyError(
            f"{self.path} has keys {sorted(keys)}, but no supported pair for "
            f"domain={self.domain}, split={self.split}."
        )

    def _validate_shapes(
        self, y_shape: tuple[int, ...], h_shape: tuple[int, ...]
    ) -> None:
        expected_y = (2, 32, 64) if self.domain == "quasi" else (4, 32, 64)
        expected_h = (2, 256, 64) if self.domain == "quasi" else (12, 256, 64)
        if len(y_shape) != 4 or y_shape[:3] != expected_y:
            raise ValueError(f"Unexpected {self.input_key} shape: {y_shape}")
        if len(h_shape) != 4 or h_shape[:3] != expected_h:
            raise ValueError(f"Unexpected {self.target_key} shape: {h_shape}")
        if y_shape[3] != h_shape[3]:
            raise ValueError("Yd/Hd sample counts differ.")

    def _ensure_open(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
            self._input = self._file[self.input_key]
            self._target = self._file[self.target_key]

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def _to_complex_last(
        array: np.ndarray, blocks: int, layout: str
    ) -> np.ndarray:
        # Raw sample: [2*T, RIS, BS].
        if layout == "grouped":
            real = array[:blocks]
            imag = array[blocks : 2 * blocks]
        else:
            real = array[0 : 2 * blocks : 2]
            imag = array[1 : 2 * blocks : 2]
        return np.stack((real, imag), axis=-1).transpose(0, 1, 2, 3)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        self._ensure_open()
        assert self._input is not None and self._target is not None
        index = int(self.indices[item])
        y = np.asarray(self._input[:, :, :, index], dtype=np.float32)
        h = np.asarray(self._target[:, :, :, index], dtype=np.float32)
        # [T, P/N, M, 2] is already obtained because raw axes are
        # [channel, RIS, BS]. No speculative row/column remapping is applied.
        obs = self._to_complex_last(
            y, self.spec.obs_blocks, self.complex_layout
        )
        target = self._to_complex_last(
            h, self.spec.query_blocks, self.complex_layout
        )
        return {
            "obs_h": torch.from_numpy(np.ascontiguousarray(obs)),
            "target_h": torch.from_numpy(np.ascontiguousarray(target)),
            "obs_ris_index": torch.tensor(self.obs_ris_index, dtype=torch.long),
            "obs_time_index": torch.tensor(self.obs_time_index, dtype=torch.long),
            "query_time": torch.arange(self.spec.query_blocks, dtype=torch.long),
            "domain_id": torch.tensor(self.spec.domain_id, dtype=torch.long),
            "observation_mask": torch.ones(
                self.spec.obs_blocks, len(self.obs_ris_index), dtype=torch.bool
            ),
            "sample_index": torch.tensor(index, dtype=torch.long),
        }

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = self._input = self._target = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = state["_input"] = state["_target"] = None
        return state

    def __del__(self) -> None:
        self.close()


def make_dataset(
    path: str | Path,
    domain: str,
    split: str,
    **kwargs,
) -> LPANH5Dataset:
    return LPANH5Dataset(path, domain, split, **kwargs)
