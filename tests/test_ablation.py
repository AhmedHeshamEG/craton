from datetime import UTC, datetime

import pytest

from craton.bench import ablation
from craton.bench.results import RunResults, TaskResult


def _run(variant: str, scores: dict[str, tuple[float, ...]], gnss_bits: float = 10.0) -> RunResults:
    results = []
    for task, values in scores.items():
        direction = "higher" if task in ("CB-02", "CB-03") else "lower"
        extras = {}
        if task == "CB-01":
            extras = {
                "seismic_catalog_bits_per_token": tuple(v + 1 for v in values),
                "gnss_daily_bits_per_token": (gnss_bits,) * len(values),
            }
        results.append(
            TaskResult(
                task_id=task,
                metric="m",
                direction=direction,  # type: ignore[arg-type]
                fingerprint="f" * 16,
                seeds=tuple(range(len(values))),
                values=values,
                extras=extras,
            )
        )
    return RunResults(
        run_id=f"run-{variant}",
        model=variant,
        config_hash="c",
        git_commit="g",
        created=datetime(2026, 10, 20, tzinfo=UTC),
        results=tuple(results),
    )


RUNS = {
    "all": _run("all", {"CB-01": (15.0, 15.1, 14.9), "CB-02": (-10.0, -10.2, -9.8)}),
    "no_gnss": _run("no_gnss", {"CB-01": (21.0, 21.1, 20.9), "CB-02": (-10.1, -9.9, -10.0)}, 22.8),
    "no_tides": _run("no_tides", {"CB-01": (15.05, 15.0, 15.1), "CB-02": (-10.0, -10.1, -9.9)}),
}


def test_a_share_is_what_leaving_an_instrument_out_costs() -> None:
    gnss = ablation.share(RUNS, 0, ablation.GNSS)
    assert gnss is not None
    assert gnss.value == pytest.approx(6.0)  # lower is better: 21 without, 15 with
    assert gnss.beyond_noise
    tides = ablation.share(RUNS, 0, ablation.TIDES)
    assert tides is not None
    assert not tides.beyond_noise
    forecast = ablation.share(RUNS, 3, ablation.GNSS)  # CB-02: higher is better
    assert forecast is not None
    assert forecast.value == pytest.approx(0.0, abs=1e-9)
    assert ablation.share(RUNS, 0, ablation.QUAKES) is None  # no run without quakes
    part = ablation.share(RUNS, 2, ablation.GNSS)  # the GNSS half of CB-01
    assert part is not None
    assert part.value == pytest.approx(12.8)


def test_the_table_is_generated_from_the_runs() -> None:
    text = ablation.render(RUNS)
    assert text.startswith("<!-- Generated")
    assert "| CB-01 geo-perplexity | 15.00 ± 0.10 bits | · | **+6.00** |" in text
    assert "(noise ±" in text
    assert "`run-all`" in text


def test_variants_answer_only_what_they_can() -> None:
    assert "CB-04" not in ablation.VARIANTS["no_gnss"][1]
    assert "CB-02" not in ablation.VARIANTS["no_quakes"][1]
    for inst in ablation.ALL:
        inputs, _ = ablation.VARIANTS[ablation.LEFT_OUT[inst]]
        assert inst not in inputs
        assert len(inputs) == 2
