import math

import numpy as np
import pytest
import torch

from craton.bench import csep
from craton.config import CratonConfig, ModelShape, TrainSettings, trunk_params
from craton.model.craton import CratonModel, Tokens
from craton.model.encoders import ENCODERS, EncoderSpec, magnitude_type_id, register_encoder
from craton.model.heads import (
    EVENT_BINS,
    FLOOR,
    RATE_SCALES,
    Bins,
    MaskedHead,
    NextDayHead,
    floored,
)
from craton.model.positional import Rope4D, Sinusoidal4D, default_waves, rotate


def tiny_config(positional: str = "rope4d") -> CratonConfig:
    shape = ModelShape(d_model=32, n_layers=2, n_heads=2, context_tokens=16, positional=positional)  # type: ignore[arg-type]
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
    return Tokens(
        values=values,
        hidden=torch.zeros(b, t, 4, dtype=torch.bool),
        mag_type=torch.zeros(b, t, dtype=torch.long),
        coords=coords,
        known=torch.ones(b, t, 4, dtype=torch.bool),
        query=torch.zeros(b, t, dtype=torch.bool),
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
    b = Tokens(**{f: getattr(a, f).clone() for f in a.__dataclass_fields__})
    b.values[0, 4] += torch.tensor([1.0, -1.0, 30.0, 3.0])
    b.coords[0, 4, 1:] += 1.0
    with torch.no_grad():
        assert torch.equal(model(a), model(b))


def test_encoders_register_once() -> None:
    assert "seismic_catalog" in ENCODERS
    with pytest.raises(ValueError, match="already registered"):
        register_encoder(EncoderSpec("seismic_catalog", ("x",), lambda d: torch.nn.Identity()))
    assert magnitude_type_id("ML") == 0
    assert magnitude_type_id("Mx?") == magnitude_type_id("other")
