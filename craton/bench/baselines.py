"""The simple methods every Craton model has to beat, registered by name.

    climatology   CB-01   each masked value drawn from how that instrument usually reads
                  CB-17   the same, from a split that never saw the region: context ignored
    context_kde   CB-17   the region's own first N days, smoothed: what counting gets you
    poisson       CB-02/3 quakes at their long-run rate, in their long-run places
    etas          CB-02/3 the standard aftershock model (``craton.bench.etas``)
    idw_trend     CB-04   a hidden station rebuilt from its neighbours: their trend + their day

None of them learns from a held-out token: fits read only the split shards, and forecasts read
only quakes from before the day they forecast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from craton.bench import csep, etas
from craton.bench.data import DAY, Events
from craton.bench.dists import ChannelDist, Gaussian, Histogram, LogisticMixture
from craton.bench.splits import stable_stations, station_sites
from craton.bench.tasks import (
    BenchContext,
    ForecastSpec,
    GapQueries,
    GriddedForecast,
    MaskedQueries,
    RegionQueries,
)
from craton.spec.timeaxis import to_seconds

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

#: The modern INGV catalogue starts in 2005 (D-027); forecasts fit from there.
FIT_FROM = "2005-01-01T00:00:00Z"
#: Forecast models are fitted on quakes at or above this magnitude and scaled down by GR.
FIT_MC = 2.0


# ---------------------------------------------------------------- climatology (CB-01)


def _edges(lo: float, hi: float, width: float, offset: float = 0.0) -> FloatArray:
    n = round((hi - lo) / width)
    return np.asarray(np.round(lo - offset + width * np.arange(n + 1), 6), dtype=np.float64)


def _smoothed_hist(values: FloatArray, edges: FloatArray) -> FloatArray:
    """Bin counts plus one (Laplace), normalised: nothing inside the range is impossible."""
    counts, _ = np.histogram(values, bins=edges)
    p = counts + 1.0
    return np.asarray(p / p.sum(), dtype=np.float64)


LAT_EDGES = _edges(41, 44, 0.01)
LON_EDGES = _edges(11.5, 15, 0.01)
DEPTH_EDGES = _edges(-5, 700, 1.0)
MAG_EDGES = _edges(-1.0, 8.1, 0.1, 0.05)


def quake_climatology(ev: Events, types: list[str]) -> dict[str, ChannelDist]:
    """Histograms of latitude, longitude, depth and (per magnitude type) magnitude."""
    all_m = _smoothed_hist(ev.mag, MAG_EDGES)
    by_type = {t: _smoothed_hist(ev.mag[ev.mag_type == t], MAG_EDGES) for t in sorted(set(types))}
    m_probs = np.stack([by_type.get(t, all_m) for t in types]) if types else all_m
    return {
        "lat": Histogram(LAT_EDGES, _smoothed_hist(ev.lat, LAT_EDGES)),
        "lon": Histogram(LON_EDGES, _smoothed_hist(ev.lon, LON_EDGES)),
        "depth_km": Histogram(DEPTH_EDGES, _smoothed_hist(ev.depth_km, DEPTH_EDGES)),
        "magnitude": Histogram(MAG_EDGES, m_probs),
    }


@dataclass
class Climatology:
    """Per-token climatology: every masked value is drawn from that instrument's history.

    Quakes: separate histograms of latitude, longitude, depth and magnitude (one per magnitude
    type) over the training split. GNSS: a Gaussian per station and component, with the
    station's training mean and spread. It never looks at the context.
    """

    name: str = "climatology"
    kind: str = "baseline"

    def predict_masked(
        self, queries: MaskedQueries, ctx: BenchContext, seed: int
    ) -> dict[str, dict[str, ChannelDist]]:
        out: dict[str, dict[str, ChannelDist]] = {}
        ev = ctx.train_events("pretrain")
        idx = queries.where("seismic_catalog")
        types = [queries.qualifiers[i]["magnitude_type"] for i in idx]
        out["seismic_catalog"] = quake_climatology(ev, types)

        g = ctx.train_gnss("pretrain")
        gi = queries.where("gnss_daily")
        stations = [queries.qualifiers[i]["station"] for i in gi]
        stats: dict[str, tuple[FloatArray, FloatArray]] = {}
        for s in sorted(set(stations)):
            enu = g.enu[g.station == s]
            stats[s] = (enu.mean(axis=0), np.maximum(enu.std(axis=0), 0.001))
        mean = np.array([stats[s][0] for s in stations]).reshape(-1, 3)
        std = np.array([stats[s][1] for s in stations]).reshape(-1, 3)
        out["gnss_daily"] = {
            ch: Gaussian(mean[:, i], std[:, i]) for i, ch in enumerate(("east", "north", "up"))
        }
        return out

    def predict_in_context(
        self, queries: RegionQueries, days: int, ctx: BenchContext, seed: int
    ) -> dict[str, ChannelDist]:
        """The whole region's climatology from a split that never saw the CB-17 region; the
        context is ignored, so every N gets the same answer."""
        return quake_climatology(ctx.train_events("cb17_fit"), queries.magnitude_type)


#: Interpolation's answer: two logistics around the line between a day's neighbours, the wide
#: one for the days a station jumps. Widths are per station and component, from training.
INTERP_WIDE = 6.0
INTERP_WIDE_WEIGHT = 0.08


def _neighbour_line(
    t: FloatArray, vt: FloatArray, venu: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """For each query time ``t``, the straight line between the station's nearest visible day
    before and after it (or the nearest one, when only one side has any). Returns the
    positions (n, 3) and the gap in days between the two days used (at least 1)."""
    pos = np.searchsorted(vt, t)
    lo = np.clip(pos - 1, 0, len(vt) - 1)
    hi = np.clip(pos, 0, len(vt) - 1)
    t_lo, t_hi = vt[lo], vt[hi]
    span = t_hi - t_lo
    frac = np.where(span > 0, (t - t_lo) / np.where(span > 0, span, 1.0), 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    enu = venu[lo] + frac[:, None] * (venu[hi] - venu[lo])
    gap = np.maximum(np.maximum(span, np.abs(t - t_lo)), np.abs(t_hi - t)) / DAY
    return enu, np.maximum(gap, 1.0)


@dataclass
class Interpolation:
    """CB-01's honest GNSS floor: a hidden day is the line between the same station's
    nearest visible days. Quakes get the climatology answer.

    How wide the answer is comes from the training split alone: every interior training day
    is hidden in turn and rebuilt from its two neighbours, and the residuals set one logistic
    width per station and component (median absolute residual / ln 3), growing with the
    square root of the gap. A second logistic, ``INTERP_WIDE`` times wider, holds
    ``INTERP_WIDE_WEIGHT`` of the probability for the days a station jumps.
    """

    name: str = "interpolation"
    kind: str = "baseline"

    def widths(self, ctx: BenchContext) -> dict[str, FloatArray]:
        """Per station: the logistic width (mm) of a one-day-gap rebuild, per component."""
        g = ctx.train_gnss("pretrain")
        out: dict[str, FloatArray] = {}
        for s in np.unique(g.station):
            sel = g.station == s
            t, enu = g.time[sel], g.enu[sel]
            if len(t) < 3:
                continue
            frac = (t[1:-1] - t[:-2]) / (t[2:] - t[:-2])
            line = enu[:-2] + frac[:, None] * (enu[2:] - enu[:-2])
            gap = (t[2:] - t[:-2]) / DAY
            resid = (enu[1:-1] - line) * 1000.0 / np.sqrt(gap / 2.0)[:, None]
            out[str(s)] = np.maximum(np.median(np.abs(resid), axis=0) / math.log(3.0), 0.2)
        return out

    def predict_masked(
        self, queries: MaskedQueries, ctx: BenchContext, seed: int
    ) -> dict[str, dict[str, ChannelDist]]:
        out = Climatology().predict_masked(queries, ctx, seed)
        gi = queries.where("gnss_daily")
        if not len(gi):
            return out
        from craton.bench.tasks import TASKS, GeoPerplexity

        task = TASKS["CB-01"]
        assert isinstance(task, GeoPerplexity)
        hidden = task.hidden(ctx)
        g = ctx.gnss
        g = g.select(~np.isin(g.token_hash, sorted(hidden)) & ~np.isin(g.station, ctx.heldout))
        widths = self.widths(ctx)
        fallback = np.median(np.stack(list(widths.values())), axis=0)
        stations = np.array([queries.qualifiers[i]["station"] for i in gi])
        t = queries.time[gi]
        base = np.zeros((len(gi), 3))
        scale = np.zeros((len(gi), 3))
        for s in np.unique(stations):
            rows = np.nonzero(stations == s)[0]
            sel = g.station == s
            enu, gap = _neighbour_line(t[rows], g.time[sel], g.enu[sel])
            base[rows] = enu
            scale[rows] = widths.get(str(s), fallback)[None, :] * np.sqrt(gap / 2.0)[:, None]
        logw = np.log(np.array([1 - INTERP_WIDE_WEIGHT, INTERP_WIDE_WEIGHT]))
        out["gnss_daily"] = {
            ch: LogisticMixture(
                base[:, c],
                np.broadcast_to(logw, (len(gi), 2)).copy(),
                np.zeros((len(gi), 2)),
                np.stack([scale[:, c], INTERP_WIDE * scale[:, c]], axis=1),
            )
            for c, ch in enumerate(("east", "north", "up"))
        }
        return out


def _kernel_hist(x: FloatArray, edges: FloatArray, sigma: float) -> FloatArray:
    """Probability per bin of a Gaussian kernel density over ``x`` (unnormalised mass kept
    inside the edges)."""
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    out = np.zeros(len(centres))
    for chunk in np.array_split(x, max(1, len(x) // 512)):
        z = (centres[None, :] - chunk[:, None]) / sigma
        out += (np.exp(-0.5 * z**2) * width / (sigma * np.sqrt(2 * np.pi))).sum(0)
    return out


@dataclass
class ContextKde:
    """CB-17's counting baseline: a kernel density of the region's first N days of quakes,
    blended with the climatology of a split that never saw the region.

    Each channel is (n · kde + prior · climatology) / (n + prior), with n the number of
    context quakes: an empty context is pure climatology, a full year is mostly the region's
    own quakes. Kernels: 0.02° in latitude and longitude, 2 km in depth, 0.2 in magnitude.
    It knows nothing about earthquakes; it is what remembering the context gets you.
    """

    name: str = "context_kde"
    kind: str = "baseline"
    prior: float = 20.0

    def predict_in_context(
        self, queries: RegionQueries, days: int, ctx: BenchContext, seed: int
    ) -> dict[str, ChannelDist]:
        clim = quake_climatology(ctx.train_events("cb17_fit"), queries.magnitude_type)
        _, values, _ = queries.context(days)
        n = len(values)
        out: dict[str, ChannelDist] = {}
        for c, (ch, edges, sigma) in enumerate(
            (
                ("lat", LAT_EDGES, 0.02),
                ("lon", LON_EDGES, 0.02),
                ("depth_km", DEPTH_EDGES, 2.0),
                ("magnitude", MAG_EDGES, 0.2),
            )
        ):
            base = clim[ch]
            assert isinstance(base, Histogram)
            if not n:
                out[ch] = base
                continue
            kde = _kernel_hist(values[:, c], edges, sigma)
            kde = kde / kde.sum()
            probs = (n * kde + self.prior * base.probs) / (n + self.prior)
            out[ch] = Histogram(edges, probs)
        return out


# ---------------------------------------------------------------- forecasts (CB-02, CB-03)


def _fit_catalog(spec: ForecastSpec, ctx: BenchContext) -> tuple[Events, float, float]:
    """Quakes to fit on (from the split), the fit start and the fit end (seconds)."""
    t0 = to_seconds(spec.fit_from)
    split = ctx.train_events(spec.fit_split)
    t1 = to_seconds(spec.fit_end)
    return split.window(t0, t1), t0, t1


def ctx_split_end(spec: ForecastSpec) -> str:
    return spec.fit_end


def _shares(b: float, mc: float, grid: csep.Grid) -> tuple[float, FloatArray]:
    raw = csep.gutenberg_richter_bins(b, mc, grid)
    return float(raw.sum()), raw / raw.sum()


@dataclass
class PoissonClimatology:
    """Time-independent: the long-run rate of quakes, spread like the long-run catalogue."""

    name: str = "poisson"
    kind: str = "baseline"
    #: Forecasts periods longer than a day too (CB-19).
    long_periods: ClassVar[bool] = True

    def forecast(self, spec: ForecastSpec, ctx: BenchContext, seed: int) -> GriddedForecast:
        grid = spec.grid
        ev, t0, t1 = _fit_catalog(spec, ctx)
        ev = ev.above(spec.mc)
        proj = etas.Projection.for_grid(grid)
        x, y = proj.to_km(ev.lat, ev.lon)
        bg = etas.Background(x, y, np.ones(len(ev)), etas.box_km(grid, proj))
        per_day = len(ev) / ((t1 - t0) / DAY)
        b = csep.b_value(ev.mag, spec.mc)
        scale, shares = _shares(b, spec.mc, grid)
        cells = per_day * spec.period_days * scale * bg.cell_mass(grid, proj)
        rates = np.tile(cells, (len(spec.day_starts), 1))
        return GriddedForecast(rates, shares)


@dataclass
class Etas:
    """Space-time ETAS fitted by maximum likelihood on the split, then run day by day."""

    name: str = "etas"
    kind: str = "baseline"
    anisotropic: bool = False
    #: Forecasts periods longer than a day too (CB-19), the cascade stretched to the period.
    long_periods: ClassVar[bool] = True

    def fit(self, spec: ForecastSpec, ctx: BenchContext, seed: int) -> etas.EtasFit:
        key = f"etas-fit:{spec.task_id}:{seed}"
        if key not in ctx._cache:
            ev, t0, t1 = _fit_catalog(spec, ctx)
            proj = etas.Projection.for_grid(spec.grid)
            cat = etas.to_catalog(ev.time, ev.lat, ev.lon, ev.mag, proj, t0)
            window = etas.FitWindow(etas.BURN_IN_DAYS, (t1 - t0) / DAY)
            ctx._cache[key] = etas.fit(
                cat, spec.mc, window, etas.box_km(spec.grid, proj), t0, seed=seed
            )
        out: etas.EtasFit = ctx._cache[key]
        return out

    def forecast(self, spec: ForecastSpec, ctx: BenchContext, seed: int) -> GriddedForecast:
        f = self.fit(spec, ctx, seed)
        proj = etas.Projection.for_grid(spec.grid)
        # Every quake known before the end of the test window; each day reads only its past.
        hist = ctx.events.window(f.t_ref, spec.window.t1).above(f.mc)
        cat = etas.to_catalog(hist.time, hist.lat, hist.lon, hist.mag, proj, f.t_ref)
        days = (spec.day_starts - f.t_ref) / DAY
        daily = etas.forecast_days(
            f, cat, days, spec.grid, proj, self.anisotropic, span=spec.period_days
        )
        scale, shares = _shares(f.b, f.mc, spec.grid)
        return GriddedForecast(daily.rates * scale, shares)


#: ETAS-incomplete searches these blind times (seconds) and productivity factors.
BLIND_SECONDS = (0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0)
PRODUCTIVITY_FACTORS = tuple(round(0.7 + 0.1 * i, 1) for i in range(19))  # 0.7 … 2.5


@dataclass
class EtasIncomplete(Etas):
    """ETAS that knows the network goes partly blind after a big quake.

    Short-term aftershock incompleteness (Hainzl 2016, 2022): right after a large quake,
    smaller ones hide in its shaking and the catalogue misses them. Under Hainzl's blind-time
    assumption a quake is missed if a larger one came less than Tb before it
    (``etas.detection_factors``). ETAS fitted on such a catalogue under-counts how productive
    big quakes are, so the triggered rate gets a factor κ, and the forecast is of the quakes
    the network will *see*, which is what the catalogue scores.

    A simplification of Hainzl (2022), which refits every parameter with the blind time
    inside the likelihood: here ETAS keeps its fit, and only (Tb, κ) are chosen, by grid
    search, to maximise the likelihood of each training day's count per magnitude bin.
    """

    name: str = "etas_incomplete"
    #: Its blind-time correction is minute by minute within one day.
    long_periods: ClassVar[bool] = False

    @staticmethod
    def _daily_box(
        f: etas.EtasFit, cat: etas.Catalog, in_box: FloatArray, starts: FloatArray, bg: float
    ) -> FloatArray:
        """Triggered quakes expected per day in the box, M ≥ mc, cascade in; ``bg`` is the
        background's daily total."""
        pr = f.params
        factor = etas.cascade_factor(pr, f.b, f.mc, 1.0)
        prod = pr.k * np.exp(pr.a * (cat.m - f.mc)) * in_box
        trig = np.zeros(len(starts))
        for i in range(0, len(starts), 256):
            t0 = starts[i : i + 256, None]
            w = etas.omori_integral(t0 + 1 - cat.t[None, :], pr.c, pr.p) - etas.omori_integral(
                t0 - cat.t[None, :], pr.c, pr.p
            )
            trig[i : i + 256] = np.where(cat.t[None, :] < t0, w, 0.0) @ prod
        return trig * factor

    def forecast(self, spec: ForecastSpec, ctx: BenchContext, seed: int) -> GriddedForecast:
        f = self.fit(spec, ctx, seed)
        grid = spec.grid
        proj = etas.Projection.for_grid(grid)
        factor = etas.cascade_factor(f.params, f.b, f.mc, 1.0)
        bg_cell = f.params.mu * f.background.cell_mass(grid, proj) * factor
        bg_box = float(bg_cell.sum())
        hist = ctx.events.window(f.t_ref, spec.window.t1).above(f.mc)
        cat = etas.to_catalog(hist.time, hist.lat, hist.lon, hist.mag, proj, f.t_ref)
        smat = etas.spatial_matrix(f.params, cat, f.mc, grid, proj)
        in_box = np.asarray(smat.sum(axis=1)).ravel()
        half = grid.m_step / 2
        lo = grid.magnitudes - half
        hi = np.append(grid.magnitudes[1:] - half, np.inf)
        gr = csep.gutenberg_richter_bins(f.b, f.mc, grid)

        # Choose (Tb, κ) on the training days: each day's count per magnitude bin, M ≥ mc.
        train = ctx.train_events(spec.fit_split).window(f.t_ref, to_seconds(ctx_split_end(spec)))
        tr = etas.to_catalog(train.time, train.lat, train.lon, train.mag, proj, f.t_ref)
        t_end = (to_seconds(ctx_split_end(spec)) - f.t_ref) / DAY
        starts = np.arange(etas.BURN_IN_DAYS, np.floor(t_end))
        past = tr.select(tr.m >= f.mc - 1e-9)
        past_in_box = np.asarray(
            etas.spatial_matrix(f.params, past, f.mc, grid, proj).sum(axis=1)
        ).ravel()
        trig = self._daily_box(f, past, past_in_box, starts, bg_box)
        bg = np.full(len(starts), bg_box)
        fine1 = etas.fine_box_rate(f, past, starts, bg + trig, past_in_box)
        first = int(grid.mag_bin_of(np.array([f.mc]))[0])
        inside = grid.cell_of(train.lat, train.lon) >= 0
        keep = inside & (train.mag >= f.mc - 1e-9)
        day = np.floor(tr.t[keep] - etas.BURN_IN_DAYS).astype(np.int64)
        ok = (day >= 0) & (day < len(starts))
        counts = np.zeros((len(starts), grid.n_mags))
        np.add.at(counts, (day[ok], grid.mag_bin_of(train.mag[keep][ok])), 1.0)
        best = (-math.inf, 0.0, 1.0)
        for kappa in PRODUCTIVITY_FACTORS:
            fine = bg[:, None] + kappa * (fine1 - bg[:, None])
            for tb in BLIND_SECONDS:
                det = etas.detection_factors(fine, f.b, f.mc, lo[first:], hi[first:], tb / DAY)
                lam = fine.mean(axis=1)[:, None] * gr[None, first:] * det
                ll = float(np.sum(counts[:, first:] * np.log(np.maximum(lam, 1e-300)) - lam))
                if ll > best[0]:
                    best = (ll, tb, kappa)
        _, tb, kappa = best
        ctx._cache[f"etas-incomplete:{spec.task_id}:{seed}"] = {"blind_s": tb, "kappa": kappa}

        # The test days: ETAS's map with the triggered part scaled, then what will be seen.
        days = (spec.day_starts - f.t_ref) / DAY
        daily = etas.forecast_days(f, cat, days, grid, proj)
        true = bg_cell[None, :] + kappa * (daily.rates - bg_cell[None, :])
        fine_t = etas.fine_box_rate(f, cat, days, daily.rates.sum(axis=1), in_box)
        fine_t = bg_box + kappa * (fine_t - bg_box)
        det = etas.detection_factors(fine_t, f.b, f.mc, lo, hi, tb / DAY)
        seen = gr[None, :] * det  # (days, mags) per quake of the true rate at M ≥ mc
        total = seen.sum(axis=1)
        return GriddedForecast(true * total[:, None], (seen / total[:, None])[:, None, :])


@dataclass
class EtasAniso(Etas):
    """ETAS with stretched aftershock zones: the same fit, but every M4+ quake's zone becomes
    an ellipse along its first hours of aftershocks (``etas.learn_shapes``), the way a
    rupture drags its aftershocks along the fault."""

    name: str = "etas_aniso"
    anisotropic: bool = True


# ---------------------------------------------------------------- IDW + trend (CB-04)


def _haversine(lat1: float, lon1: float, lat2: FloatArray, lon2: FloatArray) -> FloatArray:
    p1, p2 = math.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    h = np.sin(dp / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return np.asarray(2 * 6371.0 * np.arcsin(np.sqrt(h)), dtype=np.float64)


@dataclass
class IdwTrend:
    """Inverse-distance weighting of the nearest stations: their velocity, and their day.

    Each neighbour's series is split into a straight line (its velocity) and what is left
    (seasons, earthquakes, noise). The hidden station gets the distance-weighted velocity plus
    the distance-weighted leftovers of whichever neighbours have that day. Only stable sites
    serve as neighbours: a code whose site jumps by kilometres would drag everything with it.
    """

    name: str = "idw_trend"
    kind: str = "baseline"
    neighbours: int = 6
    max_km: float = 100.0
    power: float = 2.0

    def fill(self, queries: GapQueries, ctx: BenchContext, seed: int) -> FloatArray:
        g = ctx.gnss
        visible = g.select(~np.isin(g.station, ctx.heldout))
        visible = visible.select(np.isin(visible.station, sorted(stable_stations(visible))))
        sites = station_sites(visible)
        codes = sorted(sites)
        slat = np.array([sites[c][0] for c in codes])
        slon = np.array([sites[c][1] for c in codes])
        series: dict[str, tuple[FloatArray, FloatArray, FloatArray]] = {}

        def fitted(code: str) -> tuple[FloatArray, FloatArray, FloatArray]:
            if code not in series:
                sel = visible.station == code
                t = visible.time[sel] / (365.25 * DAY)
                enu = visible.enu[sel]
                design = np.stack([np.ones_like(t), t - t.mean()], axis=1)
                coef, *_ = np.linalg.lstsq(design, enu, rcond=None)
                series[code] = (visible.time[sel], coef[1], enu - design @ coef)
            return series[code]

        q_station = np.asarray(queries.station)
        out = np.zeros((len(q_station), 3))
        for s in sorted(set(queries.station)):
            rows = np.nonzero(q_station == s)[0]
            lat, lon = queries.site_lat[rows[0]], queries.site_lon[rows[0]]
            dist = _haversine(lat, lon, slat, slon)
            order = [i for i in np.argsort(dist) if dist[i] <= self.max_km][: self.neighbours]
            if not order:
                continue
            w = 1.0 / np.maximum(dist[order], 1.0) ** self.power
            vel = sum(wi * fitted(codes[i])[1] for wi, i in zip(w, order, strict=True)) / w.sum()
            t = queries.time[rows]
            pred = np.outer((t - t.mean()) / (365.25 * DAY), vel)
            num = np.zeros((len(rows), 3))
            den = np.zeros(len(rows))
            for wi, i in zip(w, order, strict=True):
                nt, _, resid = fitted(codes[i])
                pos = np.searchsorted(nt, t)
                hit = (pos < len(nt)) & (nt[np.minimum(pos, len(nt) - 1)] == t)
                num[hit] += wi * resid[pos[hit]]
                den[hit] += wi
            have = den > 0
            pred[have] += num[have] / den[have, None]
            out[rows] = pred
        return out


METHODS: dict[str, Any] = {}


def register_method(method: Any) -> Any:
    """Make a method runnable by ``craton bench run NAME``. Names are unique."""
    if method.name in METHODS:
        raise ValueError(f"a method called {method.name!r} is already registered")
    METHODS[method.name] = method
    return method


for _m in (
    Climatology(),
    ContextKde(),
    Interpolation(),
    PoissonClimatology(),
    Etas(),
    EtasAniso(),
    EtasIncomplete(),
    IdwTrend(),
):
    register_method(_m)

# CB-06's floors live with the detector they are the floor of.
from craton.trust.ood import FingerprintKnn, NoDetector  # noqa: E402

register_method(NoDetector())
register_method(FingerprintKnn())
