import copy
from pathlib import Path

import pytest

from craton.spec import Token, TokenError, validate_token
from craton.spec.examples import EXAMPLES, all_examples
from craton.store import encode, fingerprint_problem, read_tokens, write_tokens


def _tokens() -> list[Token]:
    return list(all_examples().values())


def test_round_trip_gives_back_the_same_tokens(tmp_path: Path) -> None:
    tokens = _tokens()
    files = write_tokens(tokens, tmp_path)
    back = [t for f in files for t in read_tokens(f.path)]
    assert sorted(back, key=Token.token_hash) == sorted(tokens, key=Token.token_hash)


def test_same_tokens_same_bytes(tmp_path: Path) -> None:
    a = write_tokens(_tokens(), tmp_path / "a")
    b = write_tokens(list(reversed(_tokens())), tmp_path / "b")
    assert [f.sha256 for f in a] == [f.sha256 for f in b]
    for fa, fb in zip(a, b, strict=True):
        assert fa.path.read_bytes() == fb.path.read_bytes()


def test_files_are_partitioned_and_named_by_their_hash(tmp_path: Path) -> None:
    files = write_tokens(_tokens(), tmp_path)
    rel = {f.path.relative_to(tmp_path).parent.as_posix() for f in files}
    assert "instrument=macroseismic_report/year=1349" in rel
    assert "instrument=seismic_catalog/year=2016" in rel
    for f in files:
        assert f.sha256.startswith(f.path.stem)
        assert fingerprint_problem(f.path) is None


def test_writing_again_is_a_no_op(tmp_path: Path) -> None:
    first = write_tokens(_tokens(), tmp_path)
    stamps = {f.path: f.path.stat().st_mtime_ns for f in first}
    second = write_tokens(_tokens(), tmp_path)
    assert first == second
    assert all(f.path.stat().st_mtime_ns == stamps[f.path] for f in second)


def test_a_tampered_file_is_refused(tmp_path: Path) -> None:
    f = write_tokens(_tokens(), tmp_path)[0]
    data = bytearray(f.path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    f.path.write_bytes(bytes(data))
    assert fingerprint_problem(f.path) is not None
    with pytest.raises(ValueError, match="does not match"):
        read_tokens(f.path)


def test_duplicates_are_refused() -> None:
    tok = all_examples()["norcia_2016"]
    with pytest.raises(TokenError) as exc:
        encode([tok, tok])
    assert "duplicate" in exc.value.codes


def test_deep_time_gets_a_partition(tmp_path: Path) -> None:
    data = copy.deepcopy(EXAMPLES["limestone_porosity"]())
    data["time"] = -1e6 * 365.2425 * 86_400
    del data["time_iso"]
    files = write_tokens([validate_token(data)], tmp_path)
    assert files[0].year == 1970 - 1_000_000
    assert read_tokens(files[0].path)[0].time == data["time"]
