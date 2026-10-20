import math

import numpy as np
import pytest
import torch

from craton.bench import csep
from craton.bench.dists import LogisticMixture
from craton.config import CratonConfig, ModelShape, TrainSettings, trunk_params
from craton.model.craton import CratonModel, Tokens
from craton.model.encoders import (
    ENCODERS,
    PAD,
    VALUE_WIDTH,
    EncoderSpec,
    instrument_id,
    magnitude_type_id,
    register_encoder,
)
from craton.model.heads import (
    EVENT_BINS,
    FLOOR,
    MAX_BITS,
    RATE_SCALES,
    Bins,
    GnssHead,
    MaskedHead,
    NextDayHead,
    floored,
)
from craton.model.positional import Rope4D, Sinusoidal4D, default_waves, rotate

ALL_INPUTS = ("seismic_catalog", "gnss_daily", "solid_earth_tide")


def tiny_config(
    positional: str = "rope4d", inputs: tuple[str, ...] = ALL_INPUTS, length: int = 16
) -> CratonConfig:
    shape = ModelShape(
        d_model=32,
        n_layers=2,
        n_heads=2,
        context_tokens=length,
        positional=positional,  # type: ignore[arg-type]
        inputs=inputs,
    )
    n = trunk_params(shape)
    return CratonConfig(
        name="tiny",
        status="active",
        description="test",
        target_hardware="cpu",
        params_range=(n, n),
        model=shape,
        train=TrainSettings(
            precision="fp32", micro_batch=4, lr=1e-3, seeds=(0, 1, 2), max_steps=5, eval_every=5
        ),
    )


def tokens(b: int = 2, t: int = 16, seed: int = 0) -> Tokens:
    g = torch.Generator().manual_seed(seed)
    values = torch.stack(
        [
            42 + torch.rand(b, t, generator=g),
            13 + torch.rand(b, t, generator=g),
            10 * torch.rand(b, t, generator=g),
            1 + 2 * torch.rand(b, t, generator=g),
        ],
        dim=-1,
    )
    coords = torch.cat(
        [
            torch.cumsum(torch.rand(b, t, 1, generator=g, dtype=torch.float64) * 3600, 1),
            values[..., :3].double(),
        ],
        dim=-1,
    )
    padded = torch.zeros(b, t, VALUE_WIDTH)
    padded[..., :4] = values
    return Tokens(
        instrument=torch.full((b, t), instrument_id("seismic_catalog"), dtype=torch.long),
        values=padded,
        hidden=torch.zeros(b, t, VALUE_WIDTH, dtype=torch.bool),
        label=torch.zeros(b, t, dtype=torch.long),
        coords=coords,
        known=torch.ones(b, t, 4, dtype=torch.bool),
        clock=torch.rand(b, t, 2, generator=g),
    )


# ---------------------------------------------------------------- positions


def _scores(rope: Rope4D, coords: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    q = torch.randn(1, 1, coords.shape[1], 32, generator=g)
    k = torch.randn(1, 1, coords.shape[1], 32, generator=g)
    cos, sin = rope.tables(coords, known, torch.float32)
    return rotate(q, cos, sin) @ rotate(k, cos, sin).transpose(-2, -1)


def test_rope_sees_only_differences() -> None:
    rope = Rope4D(32)
    coords = tokens(1).coords
    known = torch.ones(1, 16, 4, dtype=torch.bool)
    shift = torch.tensor([1.0e9, 0.5, -0.7, 3.0], dtype=torch.float64)
    a, b = _scores(rope, coords, known), _scores(rope, coords + shift, known)
    assert torch.allclose(a, b, atol=1e-4)


def test_rope_changes_with_distance() -> None:
    rope = Rope4D(32)
    coords = tokens(1).coords
    known = torch.ones(1, 16, 4, dtype=torch.bool)
    moved = coords.clone()
    moved[0, 3, 1] += 0.3  # one quake 33 km north
    assert not torch.allclose(_scores(rope, coords, known), _scores(rope, moved, known))


def test_unknown_axis_zeroes_its_block_and_hides_the_value() -> None:
    rope = Rope4D(32)
    coords = tokens(1).coords
    known = torch.ones(1, 16, 4, dtype=torch.bool)
    known[0, 5, 1:] = False
    cos, sin = rope.tables(coords, known, torch.float32)
    time_pairs = default_waves(32)[0].pairs
    assert torch.all(cos[0, 0, 5, time_pairs:] == 0)
    assert torch.all(sin[0, 0, 5, time_pairs:] == 0)
    moved = coords.clone()
    moved[0, 5, 1:] += 1.0  # the hidden place changes; nothing downstream may notice
    assert torch.equal(_scores(rope, coords, known), _scores(rope, moved, known))


def test_waves_fill_the_head_and_refuse_odd_widths() -> None:
    assert 2 * sum(w.pairs for w in default_waves(32)) == 32
    assert 2 * sum(w.pairs for w in default_waves(64)) == 64
    with pytest.raises(ValueError, match="divisible by 16"):
        default_waves(24)


def test_sinusoidal_ignores_unknown_axes() -> None:
    sin = Sinusoidal4D(32, 16)
    coords = tokens(1).coords
    known = torch.zeros(1, 16, 4, dtype=torch.bool)
    assert torch.equal(sin(coords, known), torch.zeros(1, 16, 32))


# ---------------------------------------------------------------- heads


def test_bins_match_cb01() -> None:
    from craton.bench.tasks import CB01_BINS

    for head_bin, bench_bin in zip(EVENT_BINS, CB01_BINS["seismic_catalog"], strict=True):
        assert head_bin.channel == bench_bin.channel
        assert head_bin.width == pytest.approx(bench_bin.width)
        x = np.array([head_bin.lo + 0.37 * head_bin.width, head_bin.lo + 17.5 * head_bin.width])
        lo, _ = bench_bin.edges_of(x)
        k = head_bin.index(torch.tensor(x)).numpy()
        assert np.allclose(head_bin.edges[k], lo)


def test_bins_clamp() -> None:
    b = Bins("x", 0.0, 1.0, 3)
    assert b.index(torch.tensor([-5.0, 0.5, 2.999, 9.0])).tolist() == [0, 0, 2, 2]


def test_floor_keeps_every_bin_possible() -> None:
    logits = torch.tensor([[100.0, -100.0, -100.0, -100.0]])
    p = floored(logits).exp()
    assert p.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert p[0, 1].item() == pytest.approx(FLOOR / 4, rel=1e-3)


def test_masked_loss_is_bits_on_the_true_bins() -> None:
    torch.manual_seed(0)
    head = MaskedHead(8)
    h = torch.randn(3, 8)
    values = torch.tensor([[42.0, 13.0, 10.0, 2.0], [41.5, 12.0, 3.0, 4.1], [43.9, 14.9, 0.0, 0.0]])
    manual = 0.0
    for c, (lp, b) in enumerate(zip(head(h), head.bins, strict=True)):
        idx = b.index(values[:, c])
        manual += -lp[torch.arange(3), idx].mean().item() / math.log(2)
    assert head.loss_bits(h, values).item() == pytest.approx(manual, rel=1e-5)


def test_next_day_loss_is_the_joint_poisson_likelihood() -> None:
    grid = csep.Grid(42.0, 42.2, 13.0, 13.2, cell_deg=0.1, m_min=1.5, m_max=2.0)
    torch.manual_seed(0)
    head = NextDayHead(8, grid)
    h = torch.randn(2, 8)
    counts = torch.tensor([3, 0])
    cells = torch.tensor([0, 0, 3])
    mags = torch.tensor([1, 2, 0])
    day_of = torch.tensor([0, 0, 0])
    recent = torch.randn(2, len(RATE_SCALES))
    bits, n = head.loss_bits(h, recent, counts, cells, mags, day_of)
    log_rate, lc, lm = head(h, recent)
    lam = (log_rate[:, None, None] + lc[:, :, None] + lm[:, None, :]).exp()  # (day, cell, mag)
    obs = torch.zeros_like(lam)
    for c, m in zip(cells.tolist(), mags.tolist(), strict=True):
        obs[0, c, m] += 1
    ll = (obs * lam.log() - lam - torch.lgamma(obs + 1)).sum()
    assert n.item() == 3
    assert bits.item() == pytest.approx(-ll.item() / math.log(2), rel=1e-5)


# ---------------------------------------------------------------- the model


@pytest.mark.parametrize("positional", ["rope4d", "sinusoidal"])
def test_model_runs_and_counts_its_parts(positional: str) -> None:
    cfg = tiny_config(positional)
    model = CratonModel(cfg)
    h = model(tokens())
    assert h.shape == (2, 16, 32)
    parts = model.count()
    assert parts["trunk"] == cfg.params
    assert parts["total"] == sum(v for k, v in parts.items() if k != "total")
    att = model.attention(tokens(), layer=1)
    assert att.shape == (2, 2, 16, 16)
    assert torch.allclose(att.sum(-1), torch.ones(2, 2, 16), atol=1e-5)


def test_hidden_values_never_reach_the_model() -> None:
    model = CratonModel(tiny_config()).eval()
    a = tokens()
    a.hidden[0, 4] = True
    a.known[0, 4, 1:] = False
    b = Tokens(**{f: getattr(a, f).clone() for f in a.__dataclass_fields__ if f != "padded"})
    b.values[0, 4, :4] += torch.tensor([1.0, -1.0, 30.0, 3.0])
    b.coords[0, 4, 1:] += 1.0
    with torch.no_grad():
        assert torch.equal(model(a), model(b))


def test_encoders_register_once() -> None:
    assert list(ENCODERS)[:3] == ["seismic_catalog", "gnss_daily", "solid_earth_tide"]
    assert [instrument_id(n) for n in ALL_INPUTS] == [1, 2, 3]  # ids never move
    with pytest.raises(ValueError, match="already registered"):
        register_encoder(EncoderSpec("seismic_catalog", ("x",), lambda d: torch.nn.Identity()))
    with pytest.raises(ValueError, match="at most"):
        register_encoder(EncoderSpec("too_wide", ("x",) * 13, lambda d: torch.nn.Identity()))
    assert magnitude_type_id("ML") == 0
    assert magnitude_type_id("Mx?") == magnitude_type_id("other")


def test_a_model_refuses_an_instrument_it_does_not_read() -> None:
    model = CratonModel(tiny_config(inputs=("seismic_catalog",))).eval()
    tok = tokens()
    tok.instrument[0, 3] = instrument_id("gnss_daily")
    with torch.no_grad(), pytest.raises(ValueError, match="not one this model reads"):
        model(tok)
    with pytest.raises(ValueError, match="no encoder"):
        CratonModel(tiny_config(inputs=("seismic_catalog", "nope")))


def test_each_token_gets_its_own_instruments_encoder() -> None:
    model = CratonModel(tiny_config()).eval()
    a = tokens()
    b = Tokens(**{f: getattr(a, f).clone() for f in a.__dataclass_fields__ if f != "padded"})
    b.instrument[0, 2] = instrument_id("gnss_daily")
    with torch.no_grad():
        xa, _ = model.embed(a)
        xb, _ = model.embed(b)
    assert not torch.allclose(xa[0, 2], xb[0, 2])
    assert torch.equal(xa[0, 3], xb[0, 3])


def test_padding_changes_nothing_for_real_tokens() -> None:
    model = CratonModel(tiny_config()).eval()
    a = tokens(1, 12)
    b = Tokens(**{f: getattr(a, f).clone() for f in a.__dataclass_fields__ if f != "padded"})
    pad = tokens(1, 4, seed=3)
    pad.instrument[:] = PAD
    joined = Tokens(
        **{
            f: torch.cat([getattr(b, f), getattr(pad, f)], dim=1)
            for f in a.__dataclass_fields__
            if f != "padded"
        },
        padded=True,
    )
    with torch.no_grad():
        assert torch.allclose(model(a), model(joined)[:, :12], atol=1e-5)


# ---------------------------------------------------------------- the GNSS head


def test_gnss_bits_match_the_benchmark_distribution() -> None:
    torch.manual_seed(0)
    head = GnssHead(8)
    head._start_small()
    h = torch.randn(5, 8)
    years = torch.tensor([-3.0, 0.0, 1.5, 8.0, 11.0])
    lo = torch.tensor([[-2.0, 0.0, 3.0]] * 5) + torch.arange(5.0)[:, None]
    hi = lo + 1.0
    bits = head.bin_bits(h, years, lo, hi).detach().double()
    logw, mu, s = (x.detach().double().numpy() for x in head(h, years))
    base = np.array([4_700_000.123, -12_345.6, 0.0, 1.0, 2.0])
    for c in range(3):
        dist = LogisticMixture(base, logw[:, c], mu[:, c], s[:, c])
        lo_m = base + lo[:, c].double().numpy() / 1000.0
        mass = dist.mass(lo_m, lo_m + 0.001)
        assert np.allclose(-np.log2(mass), bits[:, c].numpy(), atol=1e-3)


def test_gnss_head_never_says_impossible() -> None:
    head = GnssHead(8)
    h = torch.zeros(1, 8)
    far = torch.tensor([[5_000.0, -80_000.0, 1e9]])
    bits = head.bin_bits(h, torch.zeros(1), far, far + 1.0)
    assert torch.all(torch.isfinite(bits))
    assert bits.max().item() <= MAX_BITS
    near = head.bin_bits(h, torch.zeros(1), torch.full((1, 3), -0.5), torch.full((1, 3), 0.5))
    assert near.max().item() < 4.0  # a fresh head starts a few millimetres wide


def test_gnss_ramp_moves_every_component_with_time() -> None:
    torch.manual_seed(1)
    head = GnssHead(8)
    h = torch.randn(1, 8)
    m0 = head.mean_mm(h, torch.zeros(1))
    m1 = head.mean_mm(h, torch.ones(1))
    m2 = head.mean_mm(h, torch.full((1,), 2.0))
    assert torch.allclose(m2 - m1, m1 - m0, atol=1e-4)  # a straight line in time
