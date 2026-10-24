# Chunking package role: shared-interface - Common page, chunk, and strategy definitions.
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import NotRequired, TypedDict


class Heading(TypedDict):
    line: int
    text: str


class Page(TypedDict):
    page: int
    text: str
    headings: NotRequired[list[Heading]]


class Chunk(TypedDict):
    content: str
    page_start: int
    page_end: int
    section_path: list[str]
    chunk_index: int
    parent_id: str | None
    chunk_id: NotRequired[str]
    chunk_type: NotRequired[str]
    strategy: NotRequired[str]


class TextUnit(TypedDict):
    content: str
    page: int
    section_path: list[str]


_NUMBERED_HEADING = re.compile(
    r"^(?:(?:chapter|section|appendix)\s+[\w.-]+|"
    r"(\d+(?:\.\d+)*)[.)]?)\s+\S",
    re.IGNORECASE,
)
_LIST_LINE = re.compile(r"^\s*(?:[-*•▪◦‣]|\d+[.)]|[A-Za-z][.)])\s+\S")


def is_heading(line: str) -> bool:
    text = line.strip()
    if not text or len(text) > 120:
        return False
    if _LIST_LINE.match(text):
        return False
    if _NUMBERED_HEADING.match(text) or re.match(r"^page\s+\d+\s*[:.-]\s*\S", text, re.I):
        return True

    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    if len(words) >= 2 and text.upper() == text:
        return True
    return (
        1 <= len(words) <= 12
        and len(text) <= 100
        and not text.endswith((".", "?", "!", ";", ","))
        and sum(word[0].isupper() for word in words) / len(words) >= 0.7
    )


def pages_from_analysis_report(report: Mapping[str, object]) -> list[Page]:
    """Convert document_analyzer.py output pages to the shared page input."""
    raw_pages = report.get("pages")
    if not isinstance(raw_pages, list):
        raise ValueError("Analysis report must contain a 'pages' list.")

    pages: list[Page] = []
    for index, raw_page in enumerate(raw_pages, start=1):
        if not isinstance(raw_page, dict):
            raise ValueError(f"Analysis report page {index} must be an object.")
        page_number = raw_page.get("page_number")
        text = raw_page.get("extracted_text")
        raw_headings = raw_page.get("headings", [])
        if not isinstance(page_number, int) or page_number < 1:
            raise ValueError(f"Analysis report page {index} has an invalid page number.")
        if not isinstance(text, str):
            raise ValueError(f"Analysis report page {index} has no extracted text.")
        if not isinstance(raw_headings, list):
            raise ValueError(f"Analysis report page {index} has invalid headings.")

        headings: list[Heading] = []
        for heading in raw_headings:
            if (
                not isinstance(heading, dict)
                or not isinstance(heading.get("line"), int)
                or not isinstance(heading.get("text"), str)
            ):
                raise ValueError(f"Analysis report page {index} has an invalid heading.")
            headings.append({
                "line": heading["line"],
                "text": heading["text"],
            })
        pages.append({
            "page": page_number,
            "text": text,
            "headings": headings,
        })
    return pages


def iter_text_units(pages: Sequence[Page]) -> list[TextUnit]:
    units: list[TextUnit] = []
    section_stack: list[tuple[int, str]] = []
    last_page = 0

    for page_data in pages:
        page_number = page_data["page"]
        text = page_data["text"]
        if not isinstance(page_number, int) or page_number < 1:
            raise ValueError("Each page must have a positive integer 'page' value.")
        if not isinstance(text, str):
            raise TypeError("Each page must have string 'text'.")
        if last_page and page_number < last_page:
            raise ValueError("Pages must be supplied in ascending page order.")
        last_page = page_number

        page_headings = {
            heading["line"]: heading["text"]
            for heading in page_data.get("headings", [])
        }
        for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
            line_text = line.strip()
            heading_text = page_headings.get(line_number)
            if line_text and (heading_text or is_heading(line_text)):
                section_title = heading_text or line_text
                match = _NUMBERED_HEADING.match(section_title)
                level = (
                    match.group(1).count(".") + 1
                    if match and match.group(1)
                    else 1
                )
                section_stack = [
                    (current_level, title)
                    for current_level, title in section_stack
                    if current_level < level
                ]
                section_stack.append((level, section_title))
            units.append({
                "content": line,
                "page": page_number,
                "section_path": [title for _, title in section_stack],
            })

    return units


def make_chunk(
    content: str,
    page_start: int,
    page_end: int,
    section_path: Sequence[str],
    chunk_index: int,
    parent_id: str | None = None,
    chunk_type: str | None = None,
) -> Chunk:
    chunk: Chunk = {
        "content": content.strip(),
        "page_start": page_start,
        "page_end": page_end,
        "section_path": list(section_path),
        "chunk_index": chunk_index,
        "parent_id": parent_id,
    }
    if chunk_type is not None:
        chunk["chunk_type"] = chunk_type
    return chunk


class ChunkingStrategy(ABC):
    """Common interface for turning ordered PDF pages into metadata-rich chunks."""

    @abstractmethod
    def chunk(self, pages: Sequence[Page]) -> list[Chunk]:
        """Return chunks using the shared Chunk format."""
