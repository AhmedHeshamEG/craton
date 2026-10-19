from pathlib import Path

from craton.bench import scoreboard
from craton.bench.catalog import CATALOG
from craton.bench.ratchet import accept, judge
from craton.bench.results import Scoreboard
from craton.paths import find_root, scoreboard_json
from tests.test_ratchet import run, task


def test_empty_board_lists_every_planned_task() -> None:
    md = scoreboard.render(Scoreboard())
    for t in CATALOG:
        assert t.task_id in md
    assert "No runs yet" in md


def test_render_is_deterministic() -> None:
    r = run(task())
    board = accept(Scoreboard(), r, judge(Scoreboard(), r))
    assert scoreboard.render(board) == scoreboard.render(board)
    assert "3 ± 0.1 ↓" in scoreboard.render(board)


def test_hand_edit_is_detected(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    Scoreboard().dump(scoreboard_json(tmp_path))
    scoreboard.write(tmp_path)
    assert scoreboard.integrity_problem(tmp_path) is None

    md = tmp_path / "SCOREBOARD.md"
    md.write_text(md.read_text(encoding="utf-8").replace("not measured yet", "1.23", 1), "utf-8")
    assert scoreboard.integrity_problem(tmp_path) is not None


def test_committed_scoreboard_matches_its_json() -> None:
    assert scoreboard.integrity_problem(find_root()) is None


def test_json_round_trip(tmp_path: Path) -> None:
    r = run(task())
    board = accept(Scoreboard(), r, judge(Scoreboard(), r))
    board.dump(tmp_path / "s.json")
    assert Scoreboard.load(tmp_path / "s.json") == board
