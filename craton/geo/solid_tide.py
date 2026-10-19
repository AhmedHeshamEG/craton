"""The solid-Earth body tide: how far the Moon and the Sun stretch the ground under a point.

The Moon pulls the near side of the Earth harder than the centre, and the centre harder than the
far side, so the whole planet is stretched along the line to the Moon, and a bit less along the
line to the Sun. Rock is elastic: the ground in Rome rises and falls by up to ~30 cm twice a day,
and nobody feels it because everything around rises with it. GNSS stations see it, and
earthquakes may care about it, which is why Craton carries it as a forcing.

The physics is the IERS Conventions (2010) model, chapter 7.1.1 (Mathews, Dehant & Gipson 1997):

* **Step 1**, in the time domain: degree-2 and degree-3 tides with Love number h (up) and Shida
  number l (sideways), h₂ and l₂ depending on latitude, plus small corrections for the
  out-of-phase (anelastic) part of the diurnal and semidiurnal bands and the latitude dependence.
* **Step 2**, in the frequency domain: corrections for 31 diurnal and 5 long-period tidal lines
  whose Love numbers depart from the nominal ones, mostly near the free-core nutation resonance.
* No Step 3: the permanent tide stays in, as in the IERS convention ("conventional tide free").

This module is a derived work of the IERS Conventions software (see ``IERS_NOTICE.txt`` next
to it, reproduced as its licence asks). It is not IERS Conventions software and is neither
distributed nor endorsed by the IERS Conventions Center. **How it differs from the original:**

1. NumPy, vectorised over many points and times; the routines are renamed
   (``_degree_2_3``, ``_out_of_phase_diurnal``, ``_out_of_phase_semidiurnal``,
   ``_latitude_dependence``, ``_frequency_diurnal``, ``_frequency_long_period``).
2. Time comes as Craton seconds (UTC); the leap-second table is ``craton.geo.clock``.
3. Sun and Moon positions come from ``craton.geo.ephemeris`` (Montenbruck & Gill 2000), where
   the original expects them as inputs.
4. :func:`solid_tide_enu` adds the station position and the rotation to east, north, up.

The formulas, coefficients and tables are the original's. With the original's own Sun and Moon
inputs, this code reproduces its four published test cases to 1e-12 m.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from craton.geo.clock import DAYS_PER_CENTURY, JD_J2000, julian_date, tt_offset_s
from craton.geo.ephemeris import moon_ecef, sun_ecef
from craton.geo.frames import geodetic_to_ecef, to_enu
from craton.spec.timeaxis import SECONDS_PER_DAY

Vec = NDArray[np.float64]

MODEL = "IERS2010"

H20, L20, H3, L3 = 0.6078, 0.0847, 0.292, 0.015
"""Nominal degree-2 and degree-3 Love (h) and Shida (l) numbers."""
MASS_RATIO_SUN = 332_946.0482
MASS_RATIO_MOON = 0.0123000371
RE = 6_378_136.6
"""Earth's equatorial radius as the IERS model uses it, metres."""

# Step 2, diurnal band (IERS 2010 Table 7.3a). Columns: multipliers of the Doodson-like
# arguments s, h, p, N', ps, then the in-phase and out-of-phase radial and transverse
# corrections in millimetres.
_DIURNAL = np.array(
    [
        [-3, 0, 2, 0, 0, -0.01, 0, 0, 0],
        [-3, 2, 0, 0, 0, -0.01, 0, 0, 0],
        [-2, 0, 1, -1, 0, -0.02, 0, 0, 0],
        [-2, 0, 1, 0, 0, -0.08, 0, -0.01, 0.01],
        [-2, 2, -1, 0, 0, -0.02, 0, 0, 0],
        [-1, 0, 0, -1, 0, -0.10, 0, 0, 0],
        [-1, 0, 0, 0, 0, -0.51, 0, -0.02, 0.03],
        [-1, 2, 0, 0, 0, 0.01, 0, 0, 0],
        [0, -2, 1, 0, 0, 0.01, 0, 0, 0],
        [0, 0, -1, 0, 0, 0.02, 0, 0, 0],
        [0, 0, 1, 0, 0, 0.06, 0, 0, 0],
        [0, 0, 1, 1, 0, 0.01, 0, 0, 0],
        [0, 2, -1, 0, 0, 0.01, 0, 0, 0],
        [1, -3, 0, 0, 1, -0.06, 0, 0, 0],
        [1, -2, 0, -1, 0, 0.01, 0, 0, 0],
        [1, -2, 0, 0, 0, -1.23, -0.07, 0.06, 0.01],
        [1, -1, 0, 0, -1, 0.02, 0, 0, 0],
        [1, -1, 0, 0, 1, 0.04, 0, 0, 0],
        [1, 0, 0, -1, 0, -0.22, 0.01, 0.01, 0],
        [1, 0, 0, 0, 0, 12.00, -0.80, -0.67, -0.03],
        [1, 0, 0, 1, 0, 1.73, -0.12, -0.10, 0],
        [1, 0, 0, 2, 0, -0.04, 0, 0, 0],
        [1, 1, 0, 0, -1, -0.50, -0.01, 0.03, 0],
        [1, 1, 0, 0, 1, 0.01, 0, 0, 0],
        [0, 1, 0, 1, -1, -0.01, 0, 0, 0],
        [1, 2, -2, 0, 0, -0.01, 0, 0, 0],
        [1, 2, 0, 0, 0, -0.11, 0.01, 0.01, 0],
        [2, -2, 1, 0, 0, -0.01, 0, 0, 0],
        [2, 0, -1, 0, 0, -0.02, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 0, 0, 1, 0, 0, 0, 0, 0],
    ]
)

# Step 2, long-period band (IERS 2010 Table 7.3b), same layout.
_LONG_PERIOD = np.array(
    [
        [0, 0, 0, 1, 0, 0.47, 0.23, 0.16, 0.07],
        [0, 2, 0, 0, 0, -0.20, -0.12, -0.11, -0.05],
        [1, 0, -1, 0, 0, -0.11, -0.08, -0.09, -0.04],
        [2, 0, 0, 0, 0, -0.13, -0.11, -0.15, -0.07],
        [2, 0, 0, 1, 0, -0.05, -0.05, -0.06, -0.03],
    ]
)


class _Geometry:
    """Station and body vectors with the trigonometry every step shares. Shapes (n, 3)."""

    def __init__(self, xsta: Vec, xsun: Vec, xmon: Vec) -> None:
        self.xsta, self.xsun, self.xmon = xsta, xsun, xmon
        self.rsta = np.linalg.norm(xsta, axis=-1)
        self.rsun = np.linalg.norm(xsun, axis=-1)
        self.rmon = np.linalg.norm(xmon, axis=-1)
        self.sinphi = xsta[..., 2] / self.rsta
        self.cosphi = np.hypot(xsta[..., 0], xsta[..., 1]) / self.rsta
        self.sinla = xsta[..., 1] / self.cosphi / self.rsta
        self.cosla = xsta[..., 0] / self.cosphi / self.rsta
        self.fac2sun = MASS_RATIO_SUN * RE * (RE / self.rsun) ** 3
        self.fac2mon = MASS_RATIO_MOON * RE * (RE / self.rmon) ** 3

    def to_xyz(self, dr: Vec, de: Vec, dn: Vec) -> Vec:
        """Radial, east and north pieces back into ECEF (the original's closing lines)."""
        x = dr * self.cosla * self.cosphi - de * self.sinla - dn * self.sinphi * self.cosla
        y = dr * self.sinla * self.cosphi + de * self.cosla - dn * self.sinphi * self.sinla
        z = dr * self.sinphi + dn * self.cosphi
        out: Vec = np.stack([x, y, z], axis=-1)
        return out

    def bodies(self) -> tuple[tuple[Vec, Vec, Vec], tuple[Vec, Vec, Vec]]:
        """(position, distance, factor) for the Sun, then the Moon."""
        return (self.xsun, self.rsun, self.fac2sun), (self.xmon, self.rmon, self.fac2mon)


def _degree_2_3(g: _Geometry) -> Vec:
    """Step 1, main part: degree-2 and degree-3 tides (DEHANTTIDEINEL)."""
    h2 = H20 - 0.0006 * (1 - 1.5 * g.cosphi**2)
    l2 = L20 + 0.0002 * (1 - 1.5 * g.cosphi**2)
    total = np.zeros_like(g.xsta)
    for xb, rb, fac2 in g.bodies():
        sc = np.einsum("...i,...i->...", g.xsta, xb) / g.rsta / rb
        p2 = 3 * (h2 / 2 - l2) * sc**2 - h2 / 2
        p3 = 2.5 * (H3 - 3 * L3) * sc**3 + 1.5 * (L3 - H3) * sc
        x2 = 3 * l2 * sc
        x3 = 1.5 * L3 * (5 * sc**2 - 1)
        fac3 = fac2 * (RE / rb)
        ub = xb / rb[..., None]
        us = g.xsta / g.rsta[..., None]
        total += fac2[..., None] * (x2[..., None] * ub + p2[..., None] * us)
        total += fac3[..., None] * (x3[..., None] * ub + p3[..., None] * us)
    return total


def _out_of_phase_diurnal(g: _Geometry) -> Vec:
    """Step 1: out-of-phase part of the Love numbers, diurnal band (ST1IDIU)."""
    dhi, dli = -0.0025, -0.0007
    cos2phi = g.cosphi**2 - g.sinphi**2
    dr = de = dn = 0.0
    for xb, rb, fac in g.bodies():
        a = fac * xb[..., 2] * (xb[..., 0] * g.sinla - xb[..., 1] * g.cosla) / rb**2
        b = fac * xb[..., 2] * (xb[..., 0] * g.cosla + xb[..., 1] * g.sinla) / rb**2
        dr = dr - 3 * dhi * g.sinphi * g.cosphi * a
        dn = dn - 3 * dli * cos2phi * a
        de = de - 3 * dli * g.sinphi * b
    return g.to_xyz(np.asarray(dr), np.asarray(de), np.asarray(dn))


def _out_of_phase_semidiurnal(g: _Geometry) -> Vec:
    """Step 1: out-of-phase part of the Love numbers, semidiurnal band (ST1ISEM)."""
    dhi, dli = -0.0022, -0.0007
    cos2la = g.cosla**2 - g.sinla**2
    sin2la = 2 * g.cosla * g.sinla
    dr = de = dn = 0.0
    for xb, rb, fac in g.bodies():
        p = xb[..., 0] ** 2 - xb[..., 1] ** 2
        q = 2 * xb[..., 0] * xb[..., 1]
        a = fac * (p * sin2la - q * cos2la) / rb**2
        b = fac * (p * cos2la + q * sin2la) / rb**2
        dr = dr - 0.75 * dhi * g.cosphi**2 * a
        dn = dn + 1.5 * dli * g.sinphi * g.cosphi * a
        de = de - 1.5 * dli * g.cosphi * b
    return g.to_xyz(np.asarray(dr), np.asarray(de), np.asarray(dn))


def _latitude_dependence(g: _Geometry) -> Vec:
    """Step 1: latitude dependence of the Shida number, l^(1) (ST1L1). No radial part."""
    l1d, l1sd = 0.0012, 0.0024
    cos2la = g.cosla**2 - g.sinla**2
    sin2la = 2 * g.cosla * g.sinla
    zero = np.zeros_like(g.rsta)
    de_d = dn_d = de_s = dn_s = 0.0
    for xb, rb, fac in g.bodies():
        b = fac * xb[..., 2] * (xb[..., 0] * g.cosla + xb[..., 1] * g.sinla) / rb**2
        a = fac * xb[..., 2] * (xb[..., 0] * g.sinla - xb[..., 1] * g.cosla) / rb**2
        dn_d = dn_d - l1d * g.sinphi**2 * b
        de_d = de_d + l1d * g.sinphi * (g.cosphi**2 - g.sinphi**2) * a
        p = xb[..., 0] ** 2 - xb[..., 1] ** 2
        q = 2 * xb[..., 0] * xb[..., 1]
        dn_s = dn_s - l1sd / 2 * g.sinphi * g.cosphi * fac * (p * cos2la + q * sin2la) / rb**2
        de_s = de_s - l1sd / 2 * g.sinphi**2 * g.cosphi * fac * (p * sin2la - q * cos2la) / rb**2
    diurnal = g.to_xyz(zero, 3 * np.asarray(de_d), 3 * np.asarray(dn_d))
    semi = g.to_xyz(zero, 3 * np.asarray(de_s), 3 * np.asarray(dn_s))
    out: Vec = diurnal + semi
    return out


def _arguments(t_cent: Vec) -> tuple[Vec, Vec, Vec, Vec, Vec]:
    """Mean longitudes s, h, p, N', ps in degrees (the original's polynomials)."""
    tc = t_cent
    s = 218.31664563 + (481267.88194 + (-0.0014663889 + 0.00000185139 * tc) * tc) * tc
    s = s + (1.396971278 + (0.000308889 + (0.000000021 + 0.000000007 * tc) * tc) * tc) * tc
    h = (
        280.46645
        + (36000.7697489 + (0.00030322222 + (0.000000020 - 0.00000000654 * tc) * tc) * tc) * tc
    )
    p = (
        83.35324312
        + (4069.01363525 + (-0.01032172222 + (-0.0000124991 + 0.00000005263 * tc) * tc) * tc) * tc
    )
    zns = (
        234.95544499
        + (1934.13626197 + (-0.00207561111 + (-0.00000213944 + 0.00000001650 * tc) * tc) * tc) * tc
    )
    ps = (
        282.93734098
        + (1.71945766667 + (0.00045688889 + (-0.00000001778 - 0.00000000334 * tc) * tc) * tc) * tc
    )
    return tuple(np.mod(a, 360.0) for a in (s, h, p, zns, ps))  # type: ignore[return-value]


def _tau(fhr: Vec, t_cent: Vec) -> Vec:
    """Mean lunar time in degrees (needs the unreduced, un-precessed s)."""
    tc = t_cent
    s = 218.31664563 + (481267.88194 + (-0.0014663889 + 0.00000185139 * tc) * tc) * tc
    tau = fhr * 15 + 280.4606184 + (36000.7700536 + (0.00038793 - 0.0000000258 * tc) * tc) * tc - s
    out: Vec = np.mod(tau, 360.0)
    return out


def _frequency_diurnal(g: _Geometry, fhr: Vec, t_cent: Vec) -> Vec:
    """Step 2: in- and out-of-phase corrections, diurnal band (STEP2DIU)."""
    s, h, p, zns, ps = _arguments(t_cent)
    tau = _tau(fhr, t_cent)
    zla = np.arctan2(g.xsta[..., 1], g.xsta[..., 0])
    args = np.stack([s, h, p, zns, ps], axis=-1)  # (n, 5)
    theta = np.radians(tau[..., None] + args @ _DIURNAL[:, :5].T) + zla[..., None]  # (n, 31)
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    a = _DIURNAL[:, 5:]
    sc = (2 * g.sinphi * g.cosphi)[..., None]
    c2 = (g.cosphi**2 - g.sinphi**2)[..., None]
    dr = (a[:, 0] * sc * sin_t + a[:, 1] * sc * cos_t).sum(-1)
    dn = (a[:, 2] * c2 * sin_t + a[:, 3] * c2 * cos_t).sum(-1)
    de = (g.sinphi[..., None] * (a[:, 2] * cos_t - a[:, 3] * sin_t)).sum(-1)
    out: Vec = g.to_xyz(dr, de, dn) / 1000.0
    return out


def _frequency_long_period(g: _Geometry, t_cent: Vec) -> Vec:
    """Step 2: in- and out-of-phase corrections, long-period band (STEP2LON)."""
    s, h, p, zns, ps = _arguments(t_cent)
    args = np.stack([s, h, p, zns, ps], axis=-1)
    theta = np.radians(args @ _LONG_PERIOD[:, :5].T)  # (n, 5)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    a = _LONG_PERIOD[:, 5:]
    p2 = ((3 * g.sinphi**2 - 1) / 2)[..., None]
    sc = (2 * g.cosphi * g.sinphi)[..., None]
    dr = (a[:, 0] * p2 * cos_t + a[:, 2] * p2 * sin_t).sum(-1)
    dn = (a[:, 1] * sc * cos_t + a[:, 3] * sc * sin_t).sum(-1)
    out: Vec = g.to_xyz(dr, np.zeros_like(dr), dn) / 1000.0
    return out


def tide_ecef(xsta: ArrayLike, xsun: ArrayLike, xmon: ArrayLike, t: ArrayLike) -> Vec:
    """Tidal displacement (n, 3) in ECEF metres, with the Sun and the Moon given.

    The direct counterpart of the original routine: ``xsta``, ``xsun``, ``xmon`` are ECEF
    positions in metres, ``t`` the UTC time in Craton seconds.
    """
    xs = np.atleast_2d(np.asarray(xsta, dtype=float))
    su = np.atleast_2d(np.asarray(xsun, dtype=float))
    mo = np.atleast_2d(np.asarray(xmon, dtype=float))
    xs, su, mo = np.broadcast_arrays(xs, su, mo)
    tt = np.broadcast_to(np.asarray(t, dtype=float), xs.shape[:-1])
    g = _Geometry(xs, su, mo)
    # Julian centuries since J2000 in TT for step 2; hour of the UTC day for the lunar time.
    days = julian_date(tt) - JD_J2000
    t_cent = (days + tt_offset_s(tt) / SECONDS_PER_DAY) / DAYS_PER_CENTURY
    fhr = np.mod(tt, SECONDS_PER_DAY) / 3600.0
    out: Vec = (
        _degree_2_3(g)
        + _out_of_phase_diurnal(g)
        + _out_of_phase_semidiurnal(g)
        + _latitude_dependence(g)
        + _frequency_diurnal(g, fhr, t_cent)
        + _frequency_long_period(g, t_cent)
    )
    return out


def solid_tide_enu(lat: ArrayLike, lon: ArrayLike, t: ArrayLike, height_m: ArrayLike = 0.0) -> Vec:
    """East, north, up body-tide displacement (n, 3), metres, at geodetic points and times.

    ``lat``, ``lon`` in degrees (WGS84), ``t`` in Craton seconds (UTC); all broadcast together.
    """
    lat_a, lon_a, t_a, h_a = np.broadcast_arrays(
        np.asarray(lat, dtype=float),
        np.asarray(lon, dtype=float),
        np.asarray(t, dtype=float),
        np.asarray(height_m, dtype=float),
    )
    lat_a, lon_a, t_a, h_a = (np.atleast_1d(a).ravel() for a in (lat_a, lon_a, t_a, h_a))
    xsta = geodetic_to_ecef(lat_a, lon_a, h_a)
    d = tide_ecef(xsta, sun_ecef(t_a), moon_ecef(t_a), t_a)
    return to_enu(d, lat_a, lon_a)
