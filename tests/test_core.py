import pytest
import torch

from pflseg.fed import aggregate, start_round
from pflseg.model import BLOCKS, block_of, build_unet, get_params, is_norm_param, set_params
from pflseg.splits import all_splits


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return build_unet()


def test_size_matches_reference(net):
    # Manthe et al. report about 22.5M parameters for this architecture.
    n = sum(p.numel() for p in net.parameters())
    assert 22.4e6 < n < 22.7e6


def test_every_param_has_one_block(net):
    names = [n for n, _ in net.named_parameters()]
    assert {block_of(n) for n in names} == set(BLOCKS)


def test_norm_params_detected(net):
    names = [n for n, _ in net.named_parameters()]
    norm = [n for n in names if is_norm_param(n)]
    n_norm_modules = sum(isinstance(m, torch.nn.InstanceNorm3d) for m in
                         {id(m): m for m in net.modules()}.values())
    assert len(norm) == 2 * n_norm_modules and n_norm_modules > 0


def test_forward_shape(net):
    with torch.no_grad():
        y = net(torch.zeros(1, 4, 32, 32, 32))
    assert y.shape == (1, 3, 32, 32, 32)


def test_splits_partition_parameters(net):
    names = [n for n, _ in net.named_parameters()]
    splits = all_splits()
    for s in splits.values():
        shared, local = s.shared_names(names), s.local_names(names)
        assert shared | local == set(names) and not shared & local
    assert not splits["fedavg"].local_names(names)
    assert not splits["local"].shared_names(names)
    assert splits["out_local_1"].local_names(names) == {n for n in names if block_of(n) == "head"}
    assert splits["decoder_local"].local_names(names) == splits["out_local_5"].local_names(names)
    assert splits["encoder_local"].local_names(names) == splits["in_local_5"].local_names(names)
    assert splits["norm_local"].local_names(names) == {n for n in names if is_norm_param(n)}


def test_aggregate_averages_shared_and_keeps_local():
    torch.manual_seed(1)
    nets = [build_unet() for _ in range(2)]
    params = [get_params(n) for n in nets]
    names = list(params[0])
    split = all_splits()["decoder_local"]
    shared = split.shared_names(names)
    g = aggregate(params, [1.0, 3.0], shared)
    assert set(g) == shared
    some = next(iter(shared))
    assert torch.allclose(g[some], 0.25 * params[0][some] + 0.75 * params[1][some], atol=1e-6)
    merged = start_round(params[0], g)
    for n in names:
        if n in shared:
            assert torch.equal(merged[n], g[n])
        else:
            assert torch.equal(merged[n], params[0][n])
    set_params(nets[0], merged)
    back = get_params(nets[0])
    assert all(torch.equal(back[n], merged[n]) for n in names)
