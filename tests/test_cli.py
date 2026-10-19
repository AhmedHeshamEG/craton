import pytest

from craton import __version__
from craton.cli import main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_doctor_runs() -> None:
    assert main(["doctor"]) == 0


def test_doctor_smoke_learns() -> None:
    pytest.importorskip("torch")
    assert main(["doctor", "--smoke"]) == 0


def test_config_mini_runs() -> None:
    assert main(["config", "mini"]) == 0


def test_ratchet_with_nothing_pending_passes() -> None:
    assert main(["ratchet"]) == 0


def test_version_matches_pyproject() -> None:
    import tomllib
    from pathlib import Path

    data = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text("utf-8"))
    assert data["project"]["version"] == __version__
