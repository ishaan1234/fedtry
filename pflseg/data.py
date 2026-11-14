"""Clients, folds, missing-modality assignment and patch loading.

Preprocessed layout (written by scripts/prepare_fets.py):

    <data_dir>/index.csv               case_id,D,H,W
    <data_dir>/<case_id>_img.npy       float16, (4, D, H, W), channels t1, t1ce, t2, flair
    <data_dir>/<case_id>_seg.npy       uint8, (D, H, W), BraTS labels 0, 1, 2, 4
"""
from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MODALITIES = ("t1", "t1ce", "t2", "flair")
REGIONS = ("WT", "TC", "ET")
HETEROGENEITY = ("site", "modality", "site_modality")


# ----------------------------------------------------------------- labels
def labels_to_regions(lab: torch.Tensor) -> torch.Tensor:
    """BraTS labels (B, D, H, W) -> nested regions (B, 3, D, H, W): WT, TC, ET.

    WT = any tumour; TC = necrosis + enhancing; ET = enhancing (label 4, or 3).
    """
    wt = lab > 0
    et = (lab == 4) | (lab == 3)
    tc = (lab == 1) | et
    return torch.stack([wt, tc, et], dim=1)


# ----------------------------------------------------------------- index
def load_index(data_dir: str | Path) -> dict[str, tuple[int, int, int]]:
    out = {}
    with open(Path(data_dir) / "index.csv") as f:
        for r in csv.DictReader(f):
            out[r["case_id"]] = (int(r["D"]), int(r["H"]), int(r["W"]))
    if not out:
        raise RuntimeError(f"no cases in {data_dir}/index.csv")
    return out


# ----------------------------------------------------------------- clients
def read_partition(partition_csv: str | Path, available) -> dict[str, list[str]]:
    """Client name -> sorted case ids, keeping only cases present in the data."""
    available = set(available)
    clients: dict[str, list[str]] = {}
    with open(partition_csv) as f:
        for r in csv.DictReader(f):
            cid = r["Subject_ID"].strip()
            if cid in available:
                clients.setdefault(f"site{int(r['Partition_ID']):02d}", []).append(cid)
    if not clients:
        raise RuntimeError("partition file matched no preprocessed cases")
    return {k: sorted(v) for k, v in sorted(clients.items())}


def make_clients(partition_csv, available, het: str, seed: int = 0) -> dict[str, list[str]]:
    """Clients for a heterogeneity setting.

    site, site_modality: the real institutions.
    modality: the same client sizes, but cases dealt out at random, so the only
    difference between clients is which MRI sequences they hold.
    """
    if het not in HETEROGENEITY:
        raise ValueError(f"het must be one of {HETEROGENEITY}")
    real = read_partition(partition_csv, available)
    if het != "modality":
        return real
    pool = sorted(c for v in real.values() for c in v)
    np.random.default_rng(seed).shuffle(pool)
    out, i = {}, 0
    for name, cases in real.items():
        out[name] = sorted(pool[i:i + len(cases)])
        i += len(cases)
    return out


@dataclass
class ClientSplit:
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)


def split_folds(clients: dict[str, list[str]], fold: int, n_folds: int = 5,
                val_frac: float = 0.2, seed: int = 0) -> dict[str, ClientSplit]:
    """Five-fold cross-validation drawn inside each client, as in Manthe et al.

    The test fold is one fifth of each client's cases; of the rest, ``val_frac``
    goes to validation (at least one case when two or more remain).
    """
    if not 0 <= fold < n_folds:
        raise ValueError("fold out of range")
    out = {}
    for k, (name, cases) in enumerate(clients.items()):
        rng = np.random.default_rng([seed, k])
        perm = list(rng.permutation(cases))
        test = [c for i, c in enumerate(perm) if i % n_folds == fold]
        rest = [c for i, c in enumerate(perm) if i % n_folds != fold]
        n_val = max(1, round(val_frac * len(rest))) if len(rest) >= 2 else 0
        out[name] = ClientSplit(train=sorted(rest[n_val:]), val=sorted(rest[:n_val]),
                                test=sorted(test))
    return out


def cap_training(splits: dict[str, ClientSplit], cap: int, seed: int = 0) -> None:
    """Subsample each client's training cases to at most ``cap`` (site-size factor)."""
    if cap <= 0:
        return
    for k, (name, s) in enumerate(splits.items()):
        if len(s.train) > cap:
            rng = np.random.default_rng([seed, k, 7])
            s.train = sorted(rng.choice(s.train, size=cap, replace=False).tolist())


def assign_modalities(client_names, het: str, seed: int = 0) -> dict[str, tuple[int, ...]]:
    """Which MRI sequences each client holds.

    With missing modalities, clients are ranked at random and hold 4, 3, 2, 1
    sequences in turn, so each count covers about a quarter of the clients;
    the subset of a given size is drawn at random. Fixed by ``seed`` so every
    split in the grid sees the same assignment.
    """
    names = list(client_names)
    if het == "site":
        return {n: tuple(range(4)) for n in names}
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(names))
    out = {}
    for rank, i in enumerate(order):
        count = 4 - (rank % 4)
        combos = list(itertools.combinations(range(4), count))
        out[names[i]] = combos[int(rng.integers(len(combos)))]
    return out


def channel_mask(mods: tuple[int, ...], device) -> torch.Tensor:
    m = torch.zeros(1, 4, 1, 1, 1, device=device)
    m[:, list(mods)] = 1.0
    return m


# ----------------------------------------------------------------- loading
def load_case(data_dir: str | Path, case_id: str, mmap: bool = False):
    mode = "r" if mmap else None
    d = Path(data_dir)
    return (np.load(d / f"{case_id}_img.npy", mmap_mode=mode),
            np.load(d / f"{case_id}_seg.npy", mmap_mode=mode))


class VolumeCache:
    """Whole volumes for validation and testing, read once and kept in RAM."""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self._c: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def get(self, case_id):
        if case_id not in self._c:
            img, seg = load_case(self.data_dir, case_id)
            self._c[case_id] = (np.ascontiguousarray(img), np.ascontiguousarray(seg))
        return self._c[case_id]


class PatchDataset(Dataset):
    """Random cubic patch from one case; indexed by position in ``case_ids``."""

    def __init__(self, data_dir, case_ids: list[str], patch: int):
        self.data_dir, self.case_ids, self.patch = data_dir, list(case_ids), patch

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, i):
        img, seg = load_case(self.data_dir, self.case_ids[i], mmap=True)
        p = self.patch
        starts = [int(torch.randint(0, s - p + 1, (1,))) for s in seg.shape]
        sl = tuple(slice(s, s + p) for s in starts)
        x = np.array(img[(slice(None),) + sl], copy=True)  # copy out of the memmap
        y = np.array(seg[sl], copy=True)
        return torch.from_numpy(x), torch.from_numpy(y)


class RoundBatches:
    """Batch sampler whose batches are replaced before each pass.

    The DataLoader reads it in the main process, so persistent workers can be
    reused across rounds while the schedule changes.
    """

    def __init__(self):
        self.batches: list[list[int]] = []

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# ----------------------------------------------------------------- augmentation
def augment(x: torch.Tensor, y: torch.Tensor):
    """On-GPU augmentation following Manthe et al.: flips, noise, scaling, contrast."""
    B = x.shape[0]
    for b in range(B):
        for ax in (1, 2, 3):
            if torch.rand(()) < 0.5:
                x[b] = x[b].flip(ax)
                y[b] = y[b].flip(ax)
    dev = x.device
    noise = (torch.rand(B, 1, 1, 1, 1, device=dev) < 0.15).float()
    x = x + noise * 0.1 * torch.randn_like(x)
    scale_on = torch.rand(B, 1, 1, 1, 1, device=dev) < 0.15
    scale = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(0.7, 1.3)
    x = torch.where(scale_on, x * scale, x)
    con_on = torch.rand(B, 1, 1, 1, 1, device=dev) < 0.15
    con = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(0.65, 1.5)
    mean = x.mean(dim=(2, 3, 4), keepdim=True)
    x = torch.where(con_on, (x - mean) * con + mean, x)
    return x, y
