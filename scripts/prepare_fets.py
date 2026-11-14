"""Preprocess BraTS-style NIfTI cases into the arrays the trainer reads.

Per case, following Manthe et al.: crop to the brain (non-zero voxels in any
sequence), z-score each sequence inside the brain, zero outside, pad every side
to at least --min_size, save float16 image (4, D, H, W) in the order
t1, t1ce, t2, flair and uint8 labels (BraTS: 0, 1 necrosis, 2 oedema, 4 enhancing).

FeTS 2022 layout (default):
    <src>/**/FeTS2022_00000/FeTS2022_00000_{t1,t1ce,t2,flair,seg}.nii.gz
MSD Task01 layout (--format msd): imagesTr/*.nii.gz (4D) and labelsTr/*.nii.gz.

    python scripts/prepare_fets.py --src data/fets2022 --out data/fets2022_prep --workers 16
"""
from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

ORDER = ("t1", "t1ce", "t2", "flair")
MSD_CHANNELS = (1, 2, 3, 0)          # MSD stores FLAIR, T1w, T1gd, T2w
MSD_TO_BRATS = np.array([0, 2, 1, 4], dtype=np.uint8)  # MSD 1 oedema, 2 non-enh., 3 enh.


def _read(p) -> np.ndarray:
    return np.asanyarray(nib.load(str(p)).dataobj).astype(np.float32)


def find_cases(src: Path, fmt: str) -> list[tuple[str, dict]]:
    if fmt == "msd":
        out = []
        for lab in sorted((src).rglob("labelsTr/*.nii.gz")):
            if lab.name.startswith("."):
                continue
            img = lab.parent.parent / "imagesTr" / lab.name
            out.append((lab.name.replace(".nii.gz", ""), dict(img=img, seg=lab)))
        return out
    out = []
    for seg in sorted(src.rglob("*_seg.nii.gz")):
        cid = seg.name[: -len("_seg.nii.gz")]
        files = {m: seg.parent / f"{cid}_{m}.nii.gz" for m in ORDER}
        if all(f.exists() for f in files.values()):
            out.append((cid, dict(seg=seg, **files)))
    return out


def process(job):
    cid, files, out, fmt, min_size, stride = job
    img_path, seg_path = out / f"{cid}_img.npy", out / f"{cid}_seg.npy"
    if img_path.exists() and seg_path.exists():
        seg = np.load(seg_path, mmap_mode="r")
        return cid, seg.shape
    if fmt == "msd":
        vol = _read(files["img"])
        img = np.stack([vol[..., c] for c in MSD_CHANNELS])
        seg = MSD_TO_BRATS[_read(files["seg"]).astype(np.uint8)]
    else:
        img = np.stack([_read(files[m]) for m in ORDER])
        seg = _read(files["seg"]).astype(np.uint8)
    if stride > 1:  # only for quick tests
        img, seg = img[:, ::stride, ::stride, ::stride], seg[::stride, ::stride, ::stride]

    brain = (img != 0).any(0)
    nz = np.argwhere(brain)
    lo, hi = nz.min(0), nz.max(0) + 1
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    img, seg, brain = img[(slice(None),) + sl], seg[sl], brain[sl]
    for c in range(img.shape[0]):
        v = img[c][brain]
        img[c] = np.where(brain, (img[c] - v.mean()) / (v.std() + 1e-8), 0.0)
    pad = [(max(0, (min_size - s + 1) // 2), max(0, (min_size - s) // 2)) for s in seg.shape]
    img = np.pad(img, [(0, 0)] + pad)
    seg = np.pad(seg, pad)
    np.save(img_path, img.astype(np.float16))
    np.save(seg_path, seg.astype(np.uint8))
    return cid, seg.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", choices=["fets", "msd"], default="fets")
    ap.add_argument("--min_size", type=int, default=128)
    ap.add_argument("--stride", type=int, default=1, help="downsampling, for quick tests only")
    ap.add_argument("--limit", type=int, default=0, help="first N cases only, for tests")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = find_cases(src, a.format)
    if a.limit:
        cases = cases[: a.limit]
    if not cases:
        raise SystemExit(f"no cases found under {src}")
    print(f"{len(cases)} cases -> {out}")
    jobs = [(cid, f, out, a.format, a.min_size, a.stride) for cid, f in cases]
    rows = []
    with Pool(a.workers) as pool:
        for i, (cid, shape) in enumerate(pool.imap_unordered(process, jobs), 1):
            rows.append((cid, *shape))
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "D", "H", "W"])
        w.writerows(sorted(rows))
    print("done")


if __name__ == "__main__":
    main()
