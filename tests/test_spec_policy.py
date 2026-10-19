import json
from pathlib import Path

import pytest

from craton.paths import find_root
from craton.spec import docs as spec_docs
from craton.spec.policy import FROZEN_DIR, freeze, spec_problems


def test_the_live_spec_keeps_every_frozen_promise() -> None:
    assert spec_problems() == []


def test_spec_1_0_0_is_frozen() -> None:
    assert (FROZEN_DIR / "1.0.0.json").is_file()


def test_freeze_refuses_to_overwrite() -> None:
    with pytest.raises(FileExistsError):
        freeze(FROZEN_DIR / "1.0.0.json")


def _copy_frozen(tmp_path: Path) -> tuple[Path, dict]:  # type: ignore[type-arg]
    data = json.loads((FROZEN_DIR / "1.0.0.json").read_text("utf-8"))
    return tmp_path, data


def _write(folder: Path, data: dict) -> None:  # type: ignore[type-arg]
    (folder / "1.0.0.json").write_text(json.dumps(data), "utf-8")


def test_a_field_the_code_lost_is_caught(tmp_path: Path) -> None:
    folder, data = _copy_frozen(tmp_path)
    data["token"]["properties"]["salinity"] = {"type": "number"}
    _write(folder, data)
    assert any("salinity" in p and "removed" in p for p in spec_problems(folder))


def test_a_redefined_field_is_caught(tmp_path: Path) -> None:
    folder, data = _copy_frozen(tmp_path)
    data["token"]["properties"]["lat"]["description"] = "something else"
    _write(folder, data)
    assert any("'lat'" in p for p in spec_problems(folder))


def test_a_newly_required_field_is_caught(tmp_path: Path) -> None:
    folder, data = _copy_frozen(tmp_path)
    data["token"]["required"].remove("kind")
    _write(folder, data)
    assert any("required" in p for p in spec_problems(folder))


def test_an_old_snapshot_with_fewer_instruments_still_passes(tmp_path: Path) -> None:
    folder, data = _copy_frozen(tmp_path)
    del data["instruments"]["lithology"]
    _write(folder, data)
    assert spec_problems(folder) == []


def test_no_snapshot_is_a_problem(tmp_path: Path) -> None:
    assert spec_problems(tmp_path) != []


def test_the_spec_document_is_generated() -> None:
    assert spec_docs.drift_problem(find_root()) is None


def test_hand_edits_to_the_spec_document_are_caught(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "token-spec.md").write_text(spec_docs.render() + "edit\n", "utf-8")
    assert spec_docs.drift_problem(tmp_path) is not None
