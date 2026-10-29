"""Read the first sheet of an Excel workbook (.xlsx) with the standard library alone.

An .xlsx file is a zip of XML parts: the cells of each sheet in ``xl/worksheets/sheetN.xml``,
the text they share in ``xl/sharedStrings.xml``. Regulators publish their tables this way, and
reading the cells takes a page of code, so no spreadsheet library joins the dependencies.

Only what a data table needs: numbers, shared and inline text, booleans and empty cells.
Formulas come back as their cached value; dates come back as Excel's day numbers (convert
with :func:`excel_day`), because a cell does not say it is a date without its style.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_REF = re.compile(r"^([A-Z]+)(\d+)$")
#: Excel's day 0 (with the 1900 leap-year bug folded in, as every reader does).
_EXCEL_EPOCH = datetime(1899, 12, 30, tzinfo=UTC)

Cell = str | float | bool | None


def column_index(letters: str) -> int:
    """``A`` → 0, ``Z`` → 25, ``AA`` → 26."""
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n - 1


def excel_day(value: float) -> datetime:
    """An Excel day number (1900 system) as a UTC datetime."""
    return _EXCEL_EPOCH + timedelta(days=value)


def _text(el: ET.Element) -> str:
    return "".join(t.text or "" for t in el.iter(f"{_NS}t"))


def _shared(z: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    return [_text(si) for si in root.iter(f"{_NS}si")]


def _first_sheet(z: zipfile.ZipFile) -> str:
    """The path of the workbook's first sheet, through its relationships."""
    book = ET.fromstring(z.read("xl/workbook.xml"))
    sheet = next(book.iter(f"{_NS}sheet"))
    rid = sheet.get(f"{_REL}id")
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    for rel in rels:
        if rel.get("Id") == rid:
            target = rel.get("Target", "")
            return target.lstrip("/") if target.startswith("/") else "xl/" + target
    raise ValueError("workbook names no first sheet")


def rows(data: bytes) -> Iterator[list[Cell]]:
    """Every row of the first sheet, as a list of cells (empty cells are None)."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        strings = _shared(z)
        sheet = z.read(_first_sheet(z))
    for _, el in ET.iterparse(io.BytesIO(sheet)):
        if el.tag != f"{_NS}row":
            continue
        out: list[Cell] = []
        for c in el.iter(f"{_NS}c"):
            m = _REF.match(c.get("r", ""))
            col = column_index(m[1]) if m else len(out)
            out.extend([None] * (col - len(out)))
            kind = c.get("t", "n")
            v = c.find(f"{_NS}v")
            value: Cell
            if kind == "inlineStr":
                value = _text(c)
            elif v is None or v.text is None:
                value = None
            elif kind == "s":
                value = strings[int(v.text)]
            elif kind == "b":
                value = v.text == "1"
            elif kind in ("str", "e"):
                value = v.text
            else:
                value = float(v.text)
            out.append(value)
        el.clear()
        yield out


def records(data: bytes) -> Iterator[dict[str, Cell]]:
    """The first sheet as dictionaries, keyed by the header row's text."""
    it = rows(data)
    header = [str(h).strip() if h is not None else "" for h in next(it)]
    for row in it:
        if not any(v not in (None, "") for v in row):
            continue
        row = row + [None] * (len(header) - len(row))
        yield {h: v for h, v in zip(header, row, strict=False) if h}
