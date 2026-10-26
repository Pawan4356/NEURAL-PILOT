"""Fetching and reading documentation pages (offline ingestion only, Phase 8).

Pages are fetched by the ingestion CLI, never at request time. Only hosts on
the allow-list below may be fetched, and a response is capped in size, so a
mistyped or hostile source entry cannot pull arbitrary content into the index.
The page text is taken from its main content region when the page has one,
so navigation and sidebars do not end up in the passages.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
from html.parser import HTMLParser

from .chunks import Section

ALLOWED_HOSTS = (
    "scikit-learn.org",
    "xgboost.readthedocs.io",
    "optuna.readthedocs.io",
    "shap.readthedocs.io",
    "catboost.ai",
)
MAX_BYTES = 5_000_000
USER_AGENT = "neuralpilot-docs-ingest/1.0"

_SKIPPED_TAGS = {"script", "style", "nav", "header", "footer", "aside", "form", "button", "svg", "noscript", "head"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4"}
_MAIN_CLASSES = {"body", "document", "bd-article", "rst-content", "bd-article-container", "content"}


def host_allowed(url: str) -> bool:
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    return any(host == allowed or host.endswith("." + allowed) for allowed in ALLOWED_HOSTS)


def fetch_html(url: str, timeout: float = 30.0) -> str:
    if not host_allowed(url):
        raise ValueError(f"host of {url} is not on the documentation allow-list")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - allow-listed above
        body = response.read(MAX_BYTES + 1)
    if len(body) > MAX_BYTES:
        raise ValueError(f"{url} is larger than {MAX_BYTES} bytes")
    return body.decode("utf-8", errors="replace")


class _PageParser(HTMLParser):
    """Collects text per heading section, noting which text sits in main content."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._stack: list[tuple[str, bool, bool]] = []  # (tag, skipped, main)
        self._skipped = 0
        self._main = 0
        self._heading: list[str] | None = None
        self._sections: list[dict] = [{"heading": "", "main": [], "all": []}]

    def handle_starttag(self, tag, attrs):
        skipped = tag in _SKIPPED_TAGS
        main = tag in ("main", "article") or _is_main_region(attrs)
        self._stack.append((tag, skipped, main))
        self._skipped += skipped
        self._main += main
        if tag in _HEADING_TAGS and not self._skipped:
            self._heading = []
            self._sections.append({"heading": "", "main": [], "all": []})
        elif tag in ("p", "li", "pre", "dd", "dt", "td", "br", "tr", "div", "section"):
            self._add_text("\n")

    def handle_endtag(self, tag):
        for index in range(len(self._stack) - 1, -1, -1):
            open_tag, skipped, main = self._stack[index]
            if open_tag == tag:
                del self._stack[index:]
                self._skipped -= skipped
                self._main -= main
                break
        if tag in _HEADING_TAGS and self._heading is not None:
            self._sections[-1]["heading"] = " ".join("".join(self._heading).split())
            self._heading = None

    def handle_data(self, data):
        if self._skipped:
            return
        if self._heading is not None:
            self._heading.append(data)
        else:
            self._add_text(data)

    def _add_text(self, text: str) -> None:
        section = self._sections[-1]
        section["all"].append(text)
        if self._main:
            section["main"].append(text)

    def sections(self) -> list[Section]:
        has_main = any(s["main"] for s in self._sections)
        result = []
        for section in self._sections:
            parts = section["main"] if has_main else section["all"]
            text = " ".join("".join(parts).split())
            if text:
                result.append(Section(heading=section["heading"] or "Overview", text=text))
        return result


def _is_main_region(attrs) -> bool:
    values = dict(attrs)
    if values.get("role") == "main":
        return True
    classes = set((values.get("class") or "").split())
    return bool(classes & _MAIN_CLASSES)


def html_sections(html: str) -> list[Section]:
    """Heading-delimited text sections of one page, main content only when present."""
    parser = _PageParser()
    parser.feed(html)
    parser.close()
    return parser.sections()
