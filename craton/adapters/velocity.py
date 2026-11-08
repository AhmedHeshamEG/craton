"""Seismic velocity change (dv/v) from ambient-noise correlations, on the GPU.

Correlate the hum recorded at two sensors for long enough and the result looks like a seismogram
of a wave that went from one to the other (Shapiro & Campillo 2004). Its late part, the coda, has
bounced around the crust for tens of seconds, so when the rock gets a little slower, cracked or
wetter, every late wiggle arrives a little later: the coda is *stretched*. Compare today's coda
with a reference and the stretch factor is the velocity change (Sens-Schönfelder & Wegler 2006).
A big quake shakes the crust and it drops by a fraction of a percent, then heals over months.

The steps, every one in ``torch`` on the GPU:

    1. hourly windows, each sign-only ("1-bit"), then whitened inside the band, so storms,
       quakes and gaps weigh the same as quiet hours
    2. cross-correlate two sensors per hour (two stations: a pair; one station's two
       components: Z-N, Z-E, N-E), average the hours into a daily correlation
    3. stack a trailing window of days (only the past: these are inputs to a forecaster)
    4. stretch the reference over the coda window, find the factor with the best match: that
       factor is dv/v, the match is the coherence

The error of one estimate follows Weaver, Hadziioannou, Larose & Campillo (2011).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

BAND = (0.1, 0.45)
"""Hz. The top sits below the Nyquist of 1 Hz data (0.5 Hz)."""
MAX_LAG_S = 120.0
EPS_MAX = 0.01
"""Stretch factors searched: ±1%. Coseismic drops at these periods are ~0.1%."""
EPS_STEP = 2e-5


def whiten_onebit(
    hours: torch.Tensor, rate: float = 1.0, band: tuple[float, float] = BAND, taper: float = 0.02
) -> torch.Tensor:
    """Sign-only, then flat amplitude spectrum inside ``band`` (cosine edges of ``taper`` Hz),
    zero outside. Returns the spectra, shape (n, n_fft // 2 + 1), padded to twice the length
    so correlations do not wrap around."""
    x = torch.sign(hours - hours.mean(dim=-1, keepdim=True))
    n = x.shape[-1]
    spec = torch.fft.rfft(x, n=2 * n, dim=-1)
    f = torch.fft.rfftfreq(2 * n, d=1.0 / rate).to(spec.device, torch.float64)
    lo, hi = band
    w = torch.zeros_like(f)
    w[(f >= lo) & (f <= hi)] = 1.0
    rise = (f >= lo - taper) & (f < lo)
    w[rise] = 0.5 * (1 - torch.cos(math.pi * (f[rise] - (lo - taper)) / taper))
    fall = (f > hi) & (f <= hi + taper)
    w[fall] = 0.5 * (1 + torch.cos(math.pi * (f[fall] - hi) / taper))
    out: torch.Tensor = spec / spec.abs().clamp_min(1e-12) * w
    return out


def correlate(
    a: torch.Tensor, b: torch.Tensor, n: int, rate: float = 1.0, max_lag_s: float = MAX_LAG_S
) -> torch.Tensor:
    """Cross-correlations of whitened spectra ``a`` and ``b`` (same shape), lags -L..+L.

    Positive lag = the wave reached ``b`` after ``a``. Each row is normalised by the two
    traces' energy, so a perfect copy gives 1 at zero lag."""
    cc = torch.fft.irfft(b * a.conj(), n=2 * n, dim=-1)
    k = round(max_lag_s * rate)
    cc = torch.cat([cc[..., -k:], cc[..., : k + 1]], dim=-1)
    ea = torch.fft.irfft(a * a.conj(), n=2 * n, dim=-1)[..., :1]
    eb = torch.fft.irfft(b * b.conj(), n=2 * n, dim=-1)[..., :1]
    return cc / torch.sqrt(ea * eb).clamp_min(1e-12)


def stretch(
    ref: torch.Tensor,
    cur: torch.Tensor,
    rate: float,
    window_s: tuple[float, float],
    eps_max: float = EPS_MAX,
    eps_step: float = EPS_STEP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """dv/v and coherence of each row of ``cur`` against ``ref`` (lags -L..+L, both sides).

    A velocity change δ makes every arrival come at t·(1 - δ), so today's trace looks like the
    reference sampled at t·(1 + δ). We try every δ on the grid, keep the one that matches best
    over |t| in ``window_s``. ``ref``: (L,) or (m, L); ``cur``: (m, L)."""
    dev = cur.device
    n_lag = cur.shape[-1]
    k = (n_lag - 1) // 2
    t = torch.arange(-k, k + 1, dtype=torch.float64, device=dev) / rate
    sel = (t.abs() >= window_s[0]) & (t.abs() <= window_s[1])
    eps = torch.arange(-eps_max, eps_max + eps_step / 2, eps_step, dtype=torch.float64, device=dev)
    # reference sampled at t (1 + eps), linear interpolation, for every eps: (E, Lsel)
    ts = t[sel][None, :] * (1 + eps[:, None])
    pos = (ts * rate + k).clamp(0, n_lag - 1.000001)
    i0 = pos.floor().long()
    frac = pos - i0
    r = ref if ref.dim() == 2 else ref[None, :].expand(cur.shape[0], -1)
    left = r[:, i0]  # (m, E, Lsel)
    right = r[:, i0 + 1]
    stretched = left + (right - left) * frac[None]
    c = cur[:, sel]
    c = c - c.mean(dim=-1, keepdim=True)
    s = stretched - stretched.mean(dim=-1, keepdim=True)
    num = (s * c[:, None, :]).sum(-1)
    den = torch.sqrt((s * s).sum(-1) * (c * c).sum(-1)[:, None]).clamp_min(1e-12)
    coh = num / den  # (m, E)
    best = coh.argmax(dim=-1)
    return eps[best], coh.gather(-1, best[:, None])[:, 0]


def stretch_error(
    coherence: FloatArray,
    window_s: tuple[float, float],
    band: tuple[float, float] = BAND,
) -> FloatArray:
    """1σ of dv/v from the coherence (Weaver et al. 2011, eq. for the stretching method)."""
    x = np.clip(coherence, 1e-3, 0.999)
    t1, t2 = window_s
    big_t = 1.0 / (band[1] - band[0])
    wc = 2 * math.pi * (band[0] + band[1]) / 2
    out: FloatArray = (
        np.sqrt(1 - x**2)
        / (2 * x)
        * np.sqrt(6 * math.sqrt(math.pi / 2) * big_t / (wc**2 * (t2**3 - t1**3)))
    )
    return out


HOURS = 24
HOUR_S = 3_600


def hour_spectra(
    grid: FloatArray, rate: float = 1.0, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """One channel-day on a regular grid (NaN = missing) → whitened spectra of its 24 hours and
    which hours were complete. Incomplete hours get zero spectra and ``False``. A stack of
    channel-days (C, day) gives (C * 24, F) and (C * 24,), channel by channel."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per = round(HOUR_S * rate)
    hours = torch.as_tensor(
        grid[..., : per * HOURS].reshape(-1, per), dtype=torch.float64, device=dev
    )
    ok = ~torch.isnan(hours).any(dim=1)
    spec = whiten_onebit(torch.nan_to_num(hours), rate)
    return spec * ok[:, None], ok


def daily_correlation(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    rate: float = 1.0,
    max_lag_s: float = MAX_LAG_S,
) -> tuple[FloatArray, int]:
    """The day's correlation of two channels: the mean of the hours both recorded whole."""
    both = a[1] & b[1]
    n = int(both.sum())
    k = round(max_lag_s * rate)
    if n == 0:
        return np.full(2 * k + 1, np.nan), 0
    per = round(HOUR_S * rate)
    cc = correlate(a[0][both], b[0][both], per, rate, max_lag_s)
    out: FloatArray = cc.mean(dim=0).cpu().numpy()
    return out, n


def coda_window(
    distance_km: float, length_s: float = 50.0, v_km_s: float = 1.5
) -> tuple[float, float]:
    """Where the coda starts: after the slowest direct surface wave (``distance / v``) and never
    before 8 s (two periods at 0.25 Hz)."""
    t1 = max(8.0, distance_km / v_km_s)
    return t1, min(t1 + length_s, MAX_LAG_S - 1)


def pair_correlations(
    spectra: torch.Tensor,
    ok: torch.Tensor,
    pairs: torch.Tensor,
    rate: float = 1.0,
    max_lag_s: float = MAX_LAG_S,
    chunk: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One day's correlations for many station pairs and all nine component pairs at once.

    ``spectra``: (S, 3, 24, F) whitened hour spectra, components Z, N, E. ``ok``: (S, 3, 24)
    complete hours. ``pairs``: (P, 2) station indices. Returns the day's mean correlation over
    the hours both channels recorded whole, (P, 9, 2k+1) with NaN where no hour was shared, and
    the hour counts (P, 9). Component order: Z-Z, Z-N, Z-E, N-Z, … E-E (first station first)."""
    per = round(HOUR_S * rate)
    k = round(max_lag_s * rate)
    out = torch.full((len(pairs), 9, 2 * k + 1), float("nan"), dtype=torch.float64)
    counts = torch.zeros((len(pairs), 9), dtype=torch.int64)
    ca = torch.arange(3).repeat_interleave(3)
    cb = torch.arange(3).repeat(3)
    for i in range(0, len(pairs), chunk):
        p = pairs[i : i + chunk]
        a = spectra[p[:, 0]][:, ca]  # (c, 9, 24, F)
        b = spectra[p[:, 1]][:, cb]
        both = ok[p[:, 0]][:, ca] & ok[p[:, 1]][:, cb]  # (c, 9, 24)
        cc = correlate(a, b, per, rate, max_lag_s) * both[..., None]
        n = both.sum(-1)
        mean = cc.sum(-2) / n.clamp_min(1)[..., None]
        mean[n == 0] = float("nan")
        out[i : i + len(p)] = mean.cpu()
        counts[i : i + len(p)] = n.cpu()
    return out, counts


def trailing_stack(daily: FloatArray, counts: FloatArray, days: int) -> FloatArray:
    """Each day's stack = the hour-weighted mean of that day and the ``days - 1`` before it.
    Days with no hours contribute nothing; a stack with no hours at all is NaN."""
    w = counts.astype(np.float64)
    weighted = np.nan_to_num(daily) * w[:, None]
    cs = np.cumsum(np.vstack([np.zeros_like(weighted[:1]), weighted]), axis=0)
    cw = np.cumsum(np.concatenate([[0.0], w]))
    hi = np.arange(1, len(w) + 1)
    lo = np.maximum(hi - days, 0)
    tot = (cw[hi] - cw[lo])[:, None]
    out: FloatArray = np.where(tot > 0, (cs[hi] - cs[lo]) / np.maximum(tot, 1e-12), np.nan)
    return out


MIN_REF_DAYS = 30
"""A component needs this many reference days with hours, or it is left out of the pair."""


def pair_series(
    cc: FloatArray,
    hours: FloatArray,
    ref: np.ndarray[Any, np.dtype[np.bool_]],
    stack: int,
    window_s: tuple[float, float],
    device: torch.device | None = None,
    rate: float = 1.0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """One station pair's dv/v per day from its nine daily correlations.

    ``cc`` (9, D, L) and ``hours`` (9, D); ``ref`` (D,) marks the reference days. Each
    component is stretched against its own reference (the hour-weighted mean of its reference
    days), on a trailing stack of ``stack`` days; the nine are then averaged with weights
    1/σ² (σ from :func:`stretch_error`). Returns dv/v, its σ and the weighted coherence, NaN
    on days no component could be measured."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_days = cc.shape[1]
    num = np.zeros(n_days)
    wsum = np.zeros(n_days)
    csum = np.zeros(n_days)
    for k in range(cc.shape[0]):
        h = hours[k].astype(np.float64)
        use = ref & (h > 0)
        if use.sum() < MIN_REF_DAYS:
            continue
        reference = np.average(np.nan_to_num(cc[k][use]), axis=0, weights=h[use])
        cur = trailing_stack(cc[k].astype(np.float64), h, stack)
        good = ~np.isnan(cur).any(axis=1)
        if not good.any():
            continue
        r = torch.as_tensor(reference, device=dev)
        c = torch.as_tensor(cur[good], device=dev)
        eps, coh = [], []
        for i in range(0, c.shape[0], 256):
            e, q = stretch(r, c[i : i + 256], rate, window_s)
            eps.append(e.cpu().numpy())
            coh.append(q.cpu().numpy())
        e_all, q_all = np.concatenate(eps), np.concatenate(coh)
        sig = stretch_error(q_all, window_s)
        w = 1.0 / np.maximum(sig, 1e-6) ** 2
        w[q_all <= 0] = 0.0
        num[good] += w * e_all
        wsum[good] += w
        csum[good] += w * q_all
    with np.errstate(invalid="ignore", divide="ignore"):
        dvv = np.where(wsum > 0, num / wsum, np.nan)
        sigma = np.where(wsum > 0, 1.0 / np.sqrt(wsum), np.nan)
        coherence = np.where(wsum > 0, csum / wsum, np.nan)
    return dvv, sigma, coherence


YEAR_S = 365.2425 * 86_400


def design(
    days: FloatArray, terms: tuple[str, ...], noise: dict[str, FloatArray], stack: int
) -> FloatArray:
    """The regression's columns per day: an intercept, then each named term.

    ``annual``: sine and cosine of the day of the year. ``microseism``: the network's daily
    secondary-microseism level (dB). ``balance``: the short band minus the secondary band
    (dB), how the hum's frequency content tilts (Zhan, Tsai & Clayton 2013 show a shift in it
    alone fakes dv/v under stretching). Noise terms are averaged over the same trailing
    ``stack`` days as the correlations, NaN where no day had a value."""
    cols = [np.ones(len(days))]
    for term in terms:
        if term == "annual":
            ph = 2 * math.pi * days / YEAR_S
            cols += [np.sin(ph), np.cos(ph)]
        else:
            x = noise[term]
            ok = np.isfinite(x)
            w = ok.astype(np.float64)
            cols.append(trailing_stack(np.where(ok, x, 0.0)[:, None], w, stack)[:, 0])
    return np.stack(cols, axis=1)


def fit_correction(
    dvv: FloatArray, sigma: FloatArray, x: FloatArray, fit: np.ndarray[Any, np.dtype[np.bool_]]
) -> FloatArray | None:
    """Weighted least squares of dv/v on the columns ``x`` over the ``fit`` days only.
    Returns the coefficients, or None if fewer than ten days per column can be used."""
    use = fit & np.isfinite(dvv) & np.isfinite(sigma) & np.isfinite(x).all(axis=1)
    if use.sum() < 10 * x.shape[1]:
        return None
    w = 1.0 / sigma[use]
    beta, *_ = np.linalg.lstsq(x[use] * w[:, None], dvv[use] * w, rcond=None)
    out: FloatArray = np.asarray(beta, dtype=np.float64)
    return out
