"""CB-24, migration: lightings, the top-3 score with ties, and the floors."""

from __future__ import annotations

import numpy as np
import pytest

from craton.bench import csep
from craton.bench.data import DAY, Events
from craton.bench.migration import (
    TOP,
    DistanceOnly,
    MigrationCase,
    MigrationQuery,
    QuakeHistory,
    km_matrix,
    lightings_of,
    top_hit,
)
from craton.spec.timeaxis import to_seconds

CUT = "2016-08-24T00:00:00Z"
GRID = csep.Grid(42.0, 43.0, 13.0, 14.0, cell_deg=0.1, m_min=1.5, m_max=8.0)
CASE = MigrationCase("test", "Test", "central_italy", "seen", CUT, GRID, 2.0, 3.0)


def events(rows: list[tuple[float, float, float, float]]) -> Events:
    """(days after the cut, lat, lon, magnitude) → Events, sorted by time."""
    t0 = to_seconds(CUT)
    a = np.array(rows, dtype=np.float64)
    order = np.argsort(a[:, 0])
    a = a[order]
    n = len(a)
    return Events(
        t0 + a[:, 0] * DAY,
        a[:, 1],
        a[:, 2],
        np.full(n, 10.0),
        a[:, 3],
        np.array(["ML"] * n),
        np.array([f"h{i}" for i in range(n)]),
    )


def test_lightings_keep_each_cells_first_quake_in_order() -> None:
    ev = events(
        [
            (-3.0, 42.55, 13.55, 5.0),  # before the cut: never a lighting
            (0.1, 42.55, 13.55, 6.0),  # first: lights its cell
            (0.2, 42.55, 13.55, 4.0),  # same cell again: ignored
            (0.3, 42.75, 13.25, 2.9),  # below the threshold: ignored
            (0.4, 42.75, 13.25, 3.0),  # lights at exactly the threshold
            (400.0, 42.15, 13.15, 4.0),  # after the window: ignored
            (2.0, 41.5, 13.5, 4.0),  # outside the grid: ignored
        ]
    )
    lit = lightings_of(ev, CASE)
    assert len(lit) == 2
    assert (
        lit.cell.tolist()
        == GRID.cell_of(np.array([42.55, 42.75]), np.array([13.55, 13.25])).tolist()
    )
    assert lit.quakes.mag.tolist() == [6.0, 3.0]


def test_top_hit_counts_ties_in_expectation() -> None:
    unlit = np.ones(10, dtype=bool)
    s = np.arange(10, dtype=np.float64)
    assert top_hit(s, unlit, 9) == (1.0, 1.0)
    assert top_hit(s, unlit, 7)[0] == 1.0  # third best
    assert top_hit(s, unlit, 6)[0] == 0.0  # fourth
    flat = np.zeros(10)
    hit, rr = top_hit(flat, unlit, 4)
    assert hit == pytest.approx(TOP / 10)
    assert rr == pytest.approx(np.mean(1 / np.arange(1, 11)))
    # Lit cells don't compete, NaN ranks last.
    unlit[[8, 9]] = False
    assert top_hit(s, unlit, 5)[0] == 1.0
    s2 = s.copy()
    s2[5] = np.nan
    assert top_hit(s2, np.ones(10, bool), 5)[0] == 0.0


def query(lit_cells: list[int], history: Events) -> MigrationQuery:
    la, lo = GRID.cell_centres()
    c = np.array(lit_cells, dtype=np.int64)
    rows = [(0.1 * (i + 1), float(la[k]), float(lo[k]), 4.0) for i, k in enumerate(c)]
    return MigrationQuery(
        "test", "central_italy", "seen", GRID, CUT, 2.0, 3.0, history, c, events(rows)
    )


def test_distance_only_ranks_the_neighbours_of_the_last_lit_cell() -> None:
    ev = events([(-1.0, 42.05, 13.05, 3.5)])
    q = query([0, 55], ev)
    s = DistanceOnly().next_cell_scores(q, 0)
    assert s.shape == (GRID.n_cells,)
    assert int(np.argmax(s)) == 55
    neighbours = {45, 54, 56, 65}
    order = [int(i) for i in np.argsort(-s) if q.unlit[i]]
    assert set(order[:4]) == neighbours


def test_quake_history_counts_scored_quakes_before_the_cut() -> None:
    ev = events([(-5.0, 42.05, 13.05, 3.5), (-4.0, 42.05, 13.05, 3.1), (-3.0, 42.95, 13.95, 2.5)])
    s = QuakeHistory().next_cell_scores(query([50], ev), 0)
    assert s[0] == 2.0
    assert s.sum() == 2.0


def test_km_matrix_is_symmetric_and_right_sized() -> None:
    la, lo = np.array([42.0, 43.0]), np.array([13.0, 13.0])
    d = km_matrix(la, lo, la, lo)
    assert d[0, 1] == pytest.approx(111.2, abs=0.2)
    assert d[0, 0] == 0.0
    assert d[1, 0] == pytest.approx(d[0, 1])
