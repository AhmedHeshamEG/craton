"""One look for every Craton figure.

Near-black ground, one accent (magma orange, for the shallow crust where earthquakes live),
greys for everything else, Inter for type. Each figure renders a dark and a light variant so
the README matches GitHub's theme.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs" / "assets"
CACHE = ROOT / "data" / "cache"

ACCENT = "#ff6b35"


@dataclass(frozen=True)
class Theme:
    name: str
    ground: str
    ink: str  # primary text
    muted: str  # secondary text
    faint: str  # rules, deep quakes
    mid: str  # intermediate-depth quakes


DARK = Theme(
    "dark", ground="#0b0b0d", ink="#f5f5f7", muted="#86868b", faint="#3a3a3f", mid="#8e8e93"
)
LIGHT = Theme(
    "light", ground="#f5f5f7", ink="#1d1d1f", muted="#6e6e73", faint="#c7c7cc", mid="#636366"
)
THEMES = (DARK, LIGHT)

_FONT_URL = "https://cdn.jsdelivr.net/fontsource/fonts/inter@5.1.0/latin-{w}-normal.ttf"


def download(url: str, timeout: float = 120, tries: int = 5) -> bytes:
    """GET with polite retries: servers hiccup, and we never hammer them."""
    import time
    import urllib.error

    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "craton-figures"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data: bytes = r.read()
                return data
        except (urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
            time.sleep(5 * 2**attempt)
    raise RuntimeError("unreachable")


def inter(weight: int = 400) -> font_manager.FontProperties:
    """Inter (SIL Open Font Licence), fetched once into the cache."""
    path = CACHE / "fonts" / f"inter-{weight}.ttf"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(download(_FONT_URL.format(w=weight)))
    font_manager.fontManager.addfont(str(path))
    return font_manager.FontProperties(fname=str(path))
