"""The examples are documentation, so they have to be true."""

import json
from pathlib import Path

import pytest

from craton.cli import main
from craton.spec.examples import (
    ILLUSTRATIVE,
    RAW_ARQT,
    RAW_MONTECASSINO,
    RAW_NORCIA,
    all_examples,
    sha256_text,
)


def test_one_example_per_kind_at_least() -> None:
    kinds = {t.kind for t in all_examples().values()}
    assert kinds == {"observation", "forcing", "property", "event", "derived"}


def test_the_monk_is_a_token() -> None:
    monk = all_examples()["montecassino_1349"]
    assert monk.time_iso == "1349-09-17T12:00:00Z"  # 9 September in his calendar
    assert monk.value == [9.0]
    assert monk.qualifiers["scale"] == "MCS"
    assert monk.resolution.temporal_s == 86_400  # dated to the day, not the hour


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("norcia_2016", RAW_NORCIA),
        ("arquata_gnss", RAW_ARQT),
        ("montecassino_1349", RAW_MONTECASSINO),
    ],
)
def test_real_examples_hash_their_raw_record(name: str, raw: str) -> None:
    tok = all_examples()[name]
    assert tok.provenance.content_hash == sha256_text(raw)
    assert tok.provenance.source != ILLUSTRATIVE


def test_illustrative_examples_say_so() -> None:
    real = {"norcia_2016", "arquata_gnss", "montecassino_1349"}
    for name, tok in all_examples().items():
        if name not in real:
            assert tok.provenance.source == ILLUSTRATIVE, name


def test_arquata_reads_ngl_longitude_and_height() -> None:
    tok = all_examples()["arquata_gnss"]
    assert tok.lon == pytest.approx(13.19872, abs=1e-5)
    assert tok.depth_km == pytest.approx(-1.65945, abs=1e-5)
    assert tok.uncertainty_basis == "reported"


def test_cli_spec_show_and_example(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["spec"]) == 0
    assert main(["spec", "example", "montecassino_1349"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out[out.index("{") :])["qualifiers"]["place"] == "Abbazia di Montecassino"
    assert main(["spec", "example", "nope"]) == 1


def test_cli_validate_good_and_bad_files(tmp_path: Path) -> None:
    ordered = sorted(all_examples().values(), key=lambda t: t.time)
    tokens = [t.to_dict() for t in ordered]
    good = tmp_path / "good.jsonl"
    good.write_text("\n".join(json.dumps(t) for t in tokens), "utf-8")
    assert main(["spec", "validate", str(good)]) == 0
    shuffled = tmp_path / "shuffled.jsonl"
    shuffled.write_text("\n".join(json.dumps(t) for t in reversed(tokens)), "utf-8")
    assert main(["spec", "validate", str(shuffled)]) == 1  # files are in time order
    tokens[0]["lat"] = 123.0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(tokens), "utf-8")
    assert main(["spec", "validate", str(bad)]) == 1
    assert main(["spec", "validate"]) == 2
