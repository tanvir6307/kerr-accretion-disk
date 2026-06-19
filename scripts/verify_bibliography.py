"""Verify BibTeX entries against Crossref and write a review report."""

from __future__ import annotations

import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

BIB_PATH = Path("literature/references.bib")
OUTPUT = Path("literature/reference_verification.csv")
SUMMARY = Path("literature/reference_verification_summary.md")
CONTACT = "tanvir6307@gmail.com"
USER_AGENT = f"kerrdisk-uq bibliography verifier (mailto:{CONTACT})"


@dataclass(frozen=True)
class BibEntry:
    key: str
    title: str
    year: int | None
    doi: str


@dataclass(frozen=True)
class VerificationRow:
    key: str
    status: str
    source: str
    original_year: str
    matched_year: str
    doi: str
    title_similarity: str
    matched_title: str
    notes: str


def main() -> None:
    """Verify references and write CSV/Markdown summaries."""

    entries = _parse_bib(BIB_PATH.read_text(encoding="utf-8"))
    rows = [_verify_entry(entry) for entry in entries]
    _write_csv(rows)
    _write_summary(rows)
    print(OUTPUT)
    print(SUMMARY)


def _parse_bib(text: str) -> list[BibEntry]:
    entries: list[BibEntry] = []
    for match in re.finditer(r"@\w+\{([^,]+),(.*?)(?=\n\})\n\}", text, re.S):
        key = match.group(1).strip()
        body = match.group(2)
        title = _field(body, "title")
        year_text = _field(body, "year")
        doi = _field(body, "doi")
        year = int(year_text) if year_text.isdigit() else None
        entries.append(BibEntry(key=key, title=title, year=year, doi=doi))
    return entries


def _field(body: str, name: str) -> str:
    match = re.search(
        rf"\n\s*{re.escape(name)}\s*=\s*\{{(.*?)\}}\s*,?",
        body,
        re.S | re.I,
    )
    if match is None:
        return ""
    return " ".join(match.group(1).split())


def _verify_entry(entry: BibEntry) -> VerificationRow:
    if not entry.title:
        return _row(entry, "NEEDS_REVIEW", "", "", "", 0.0, "Missing title field.")
    try:
        if entry.doi:
            item = _crossref_by_doi(entry.doi)
            if item is not None:
                return _row_from_item(entry, item, "VERIFIED_DOI")
        item = _crossref_by_title(entry.title, entry.year)
    except (TimeoutError, urllib.error.URLError) as exc:
        return _row(entry, "ERROR", "", "", "", 0.0, str(exc))
    if item is None:
        return _row(entry, "NO_MATCH", "", "", "", 0.0, "No Crossref match.")
    title = _item_title(item)
    matched_year = _item_year(item)
    similarity = _similarity(entry.title, title)
    year_ok = (
        entry.year is None
        or matched_year is None
        or abs(entry.year - matched_year) <= 1
    )
    status = (
        "VERIFIED_CROSSREF_TITLE" if similarity >= 0.82 and year_ok else "NEEDS_REVIEW"
    )
    notes = (
        "High-confidence title/year match."
        if status.startswith("VERIFIED")
        else ("Crossref candidate requires manual review.")
    )
    return _row(
        entry,
        status,
        str(item.get("DOI", "")),
        str(matched_year or ""),
        title,
        similarity,
        notes,
    )


def _row_from_item(
    entry: BibEntry,
    item: dict[str, Any],
    status: str,
) -> VerificationRow:
    title = _item_title(item)
    matched_year = _item_year(item)
    similarity = _similarity(entry.title, title)
    year_ok = (
        entry.year is None
        or matched_year is None
        or abs(entry.year - matched_year) <= 1
    )
    final_status = status if similarity >= 0.70 and year_ok else "NEEDS_REVIEW"
    notes = (
        "DOI resolved and title/year are consistent."
        if final_status == status
        else "DOI resolved but title/year require manual review."
    )
    return _row(
        entry,
        final_status,
        str(item.get("DOI", entry.doi)),
        str(matched_year or ""),
        title,
        similarity,
        notes,
    )


def _row(
    entry: BibEntry,
    status: str,
    doi: str,
    matched_year: str,
    matched_title: str,
    similarity: float,
    notes: str,
) -> VerificationRow:
    return VerificationRow(
        key=entry.key,
        status=status,
        source="Crossref",
        original_year="" if entry.year is None else str(entry.year),
        matched_year=matched_year,
        doi=doi,
        title_similarity=f"{similarity:.3f}",
        matched_title=matched_title,
        notes=notes,
    )


def _crossref_by_doi(doi: str) -> dict[str, Any] | None:
    quoted = urllib.parse.quote(doi)
    data = _get_json(f"https://api.crossref.org/works/{quoted}")
    return data.get("message") if data else None


def _crossref_by_title(title: str, year: int | None) -> dict[str, Any] | None:
    query = title if year is None else f"{title} {year}"
    params = urllib.parse.urlencode(
        {
            "query.bibliographic": query,
            "rows": "1",
            "select": "DOI,title,published-print,published-online,issued",
            "mailto": CONTACT,
        }
    )
    data = _get_json(f"https://api.crossref.org/works?{params}")
    items = data.get("message", {}).get("items", []) if data else []
    if not items:
        return None
    return max(items, key=lambda item: _similarity(title, _item_title(item)))


def _get_json(url: str) -> dict[str, Any] | None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=8) as response:
        time.sleep(0.02)
        return json.loads(response.read().decode("utf-8"))


def _item_title(item: dict[str, Any]) -> str:
    title = item.get("title", [""])
    if isinstance(title, list):
        return " ".join(str(part) for part in title)
    return str(title)


def _item_year(item: dict[str, Any]) -> int | None:
    for key in ("published-print", "published-online", "issued"):
        parts = item.get(key, {}).get("date-parts", [])
        if parts and parts[0] and parts[0][0] is not None:
            return int(parts[0][0])
    return None


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _normalize(left), _normalize(right)).ratio()


def _normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(text.split())


def _write_csv(rows: list[VerificationRow]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].__dict__.keys()))
        writer.writeheader()
        writer.writerows(row.__dict__ for row in rows)


def _write_summary(rows: list[VerificationRow]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    lines = [
        "# Reference Verification Summary",
        "",
        "Source: Crossref API.",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "Entries marked `VERIFIED_DOI` or `VERIFIED_CROSSREF_TITLE` are "
            "machine-verified against Crossref title/year metadata. Entries "
            "marked `NEEDS_REVIEW`, `NO_MATCH`, or `ERROR` must be checked "
            "against ADS, arXiv, or publisher pages before manuscript use.",
            "",
        ]
    )
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
