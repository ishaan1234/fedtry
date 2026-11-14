import csv
import json
from collections import Counter

import numpy as np
import pytest
import torch

from pflseg.data import (assign_modalities, cap_training, labels_to_regions, make_clients,
                         split_folds)


def _partition(tmp_path, sizes):
    p = tmp_path / "part.csv"
    ids = []
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Partition_ID", "Subject_ID"])
        n = 0
        for site, s in enumerate(sizes, 1):
            for _ in range(s):
                cid = f"CASE_{n:04d}"
                w.writerow([site, cid])
                ids.append(cid)
                n += 1
    return p, ids


def test_regions_follow_brats_nesting():
    lab = torch.tensor([[0, 1, 2, 4]], dtype=torch.uint8).view(1, 4, 1, 1)
    r = labels_to_regions(lab)[0, :, :, 0, 0]
    assert r[0].tolist() == [False, True, True, True]   # WT
    assert r[1].tolist() == [False, True, False, True]  # TC
    assert r[2].tolist() == [False, False, False, True]  # ET


def test_folds_are_disjoint_and_cover(tmp_path):
    p, ids = _partition(tmp_path, [40, 12, 4])
    clients = make_clients(p, ids, "site")
    seen_test = Counter()
    for f in range(5):
        s = split_folds(clients, f)
        for name, cs in s.items():
            assert not set(cs.train) & set(cs.val) and not set(cs.train) & set(cs.test)
            assert not set(cs.val) & set(cs.test)
            assert sorted(cs.train + cs.val + cs.test) == clients[name]
            seen_test.update(cs.test)
    assert all(seen_test[c] == 1 for c in ids)  # each case tested exactly once
    assert split_folds(clients, 0)["site01"].test == split_folds(clients, 0)["site01"].test


def test_iid_partition_keeps_sizes_and_mixes(tmp_path):
    p, ids = _partition(tmp_path, [40, 12, 4])
    real = make_clients(p, ids, "site")
    iid = make_clients(p, ids, "modality")
    assert [len(v) for v in iid.values()] == [len(v) for v in real.values()]
    assert sorted(c for v in iid.values() for c in v) == sorted(ids)
    assert iid["site01"] != real["site01"]


def test_modality_assignment_balanced_and_fixed():
    names = [f"site{i:02d}" for i in range(1, 24)]
    a = assign_modalities(names, "site_modality", seed=0)
    assert a == assign_modalities(names, "site_modality", seed=0)
    counts = Counter(len(v) for v in a.values())
    assert set(counts) == {1, 2, 3, 4} and max(counts.values()) - min(counts.values()) <= 1
    assert all(len(v) == 4 for v in assign_modalities(names, "site").values())


def test_cap_training(tmp_path):
    p, ids = _partition(tmp_path, [40, 12])
    s = split_folds(make_clients(p, ids, "site"), 0)
    cap_training(s, 10)
    assert len(s["site01"].train) == 10 and len(s["site02"].train) <= 10


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """Six 32^3 cases with a bright cube as the tumour, two sites."""
    d = tmp_path_factory.mktemp("syn")
    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):
        img = rng.normal(size=(4, 32, 32, 32)).astype(np.float16)
        seg = np.zeros((32, 32, 32), np.uint8)
        seg[10:20, 10:20, 10:20] = 2
        seg[13:17, 13:17, 13:17] = 4
        img[:, seg > 0] += 2
        np.save(d / f"C{i}_img.npy", img)
        np.save(d / f"C{i}_seg.npy", seg)
        rows.append([f"C{i}", 32, 32, 32])
    with open(d / "index.csv", "w", newline="") as f:
        csv.writer(f).writerows([["case_id", "D", "H", "W"]] + rows)
    with open(d / "part.csv", "w", newline="") as f:
        csv.writer(f).writerows([["Partition_ID", "Subject_ID"]]
                                + [[1 if i < 3 else 2, f"C{i}"] for i in range(6)])
    return d


@pytest.mark.parametrize("split,het,ft", [("decoder_local", "site_modality", 1),
                                          ("fedavg", "site", 0)])
def test_train_end_to_end(synthetic, tmp_path, split, het, ft):
    from pflseg.train import main
    run = main(["--data_dir", str(synthetic), "--partition", str(synthetic / "part.csv"),
                "--split", split, "--het", het, "--rounds", "2", "--val_every", "1",
                "--patch", "32", "--batch", "2", "--workers", "0",
                "--finetune_epochs", str(ft), "--out", str(tmp_path)])
    s = json.loads((run / "summary.json").read_text())
    rows = list(csv.DictReader(open(run / "test_cases.csv")))
    assert s["n_test_cases"] == len(rows) == 2
    assert all(0.0 <= float(r["dice_mean"]) <= 1.0 for r in rows)
    assert not (run / "ckpt.pt").exists()
    assert main(["--data_dir", str(synthetic), "--partition", str(synthetic / "part.csv"),
                 "--split", split, "--het", het, "--rounds", "2", "--val_every", "1",
                 "--patch", "32", "--batch", "2", "--workers", "0",
                 "--finetune_epochs", str(ft), "--out", str(tmp_path)]) == run  # skipped
