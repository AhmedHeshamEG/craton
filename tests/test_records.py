import subprocess
from pathlib import Path

import pytest

from craton.records import append_only_problems


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    (tmp_path / "LOGBOOK.md").write_text("# Logbook\n\n## entry one\n", encoding="utf-8")
    (tmp_path / "DECISIONS.md").write_text("# Decisions\n\nD-001\n", encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_untouched_records_pass(repo: Path) -> None:
    assert append_only_problems(repo) == []


def test_appending_passes(repo: Path) -> None:
    with (repo / "LOGBOOK.md").open("a", encoding="utf-8") as f:
        f.write("\n## entry two\n")
    assert append_only_problems(repo) == []


def test_editing_an_old_line_fails(repo: Path) -> None:
    (repo / "DECISIONS.md").write_text("# Decisions\n\nD-001 (edited)\n", encoding="utf-8")
    problems = append_only_problems(repo)
    assert len(problems) == 1
    assert "DECISIONS.md: line 3" in problems[0]


def test_deleting_a_record_fails(repo: Path) -> None:
    (repo / "LOGBOOK.md").unlink()
    assert "deleted" in append_only_problems(repo)[0]


def test_crlf_checkout_is_not_an_edit(repo: Path) -> None:
    (repo / "LOGBOOK.md").write_bytes(b"# Logbook\r\n\r\n## entry one\r\n")
    assert append_only_problems(repo) == []


def test_unknown_ref_is_skipped(repo: Path) -> None:
    assert append_only_problems(repo, "no-such-ref") == []
