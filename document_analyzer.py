from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from pypdf import PdfReader

from config import CHUNK_OVERLAP, CHUNK_SIZE
from tracing import trace_operation


CHUNKING_DIR = Path(__file__).with_name("chunking")
DEFAULT_TABLE_ROW_RATIO = 0.25
DEFAULT_TEXT_HEAVY_CHARACTERS = 750
DEFAULT_SCANNED_TEXT_CHARACTERS = 80

HEADING_PATTERNS = (
    re.compile(
        r"^(?:(?:chapter|section|appendix)\s+[\w.-]+|"
        r"(?:\d+(?:\.\d+)*[.)]?)\s+)\S",
        re.IGNORECASE,
    ),
    re.compile(r"^page\s+\d+\s*[:.-]\s*\S", re.IGNORECASE),
)
LIST_PATTERN = re.compile(r"^\s*(?:[-*•▪◦‣]|\d+[.)]|[A-Za-z][.)])\s+\S")
TABLE_DELIMITERS = re.compile(r"\t+| {2,}|\|")
CHUNKING_OPTION_PATTERN = re.compile(
    r"^#\s*Chunking strategy option:\s*([a-z][a-z0-9_]*)\s*-\s*(.+?)\s*$"
)


class ChunkingStrategyOption(TypedDict):
    id: str
    description: str
    source_file: str


def _is_title_case_heading(line: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", line)
    return (
        1 <= len(words) <= 12
        and len(line) <= 100
        and not line.endswith((".", "?", "!", ";", ","))
        and sum(word[0].isupper() for word in words) / len(words) >= 0.7
    )


def _detect_headings(lines: list[str]) -> list[dict[str, Any]]:
    headings = []
    for line_number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text or len(text) > 120:
            continue

        explicit = any(pattern.search(text) for pattern in HEADING_PATTERNS)
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
        all_caps = len(words) >= 2 and text.upper() == text
        if explicit or all_caps or _is_title_case_heading(text):
            headings.append({
                "line": line_number,
                "text": text,
                "detection": (
                    "numbered_or_labeled"
                    if explicit
                    else "all_caps"
                    if all_caps
                    else "title_case_heuristic"
                ),
            })
    return headings


def _detect_lists(lines: list[str]) -> list[dict[str, Any]]:
    return [
        {"line": line_number, "text": line.strip()}
        for line_number, line in enumerate(lines, start=1)
        if LIST_PATTERN.match(line)
    ]


def _table_row_candidates(layout_text: str) -> list[dict[str, Any]]:
    candidates = []
    for line_number, line in enumerate(layout_text.splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        cells = [cell.strip() for cell in TABLE_DELIMITERS.split(text) if cell.strip()]
        if len(cells) >= 2:
            candidates.append({"line": line_number, "cells": cells})
    return candidates


def discover_chunking_strategies(
    chunking_dir: Path = CHUNKING_DIR,
) -> list[ChunkingStrategyOption]:
    if not chunking_dir.is_dir():
        raise FileNotFoundError(f"Chunking strategies folder not found: {chunking_dir}")

    options: list[ChunkingStrategyOption] = []
    seen: set[str] = set()
    for source_file in sorted(chunking_dir.glob("*.py")):
        with source_file.open(encoding="utf-8") as file:
            for line in file:
                match = CHUNKING_OPTION_PATTERN.match(line.strip())
                if not match:
                    continue
                strategy_id, description = match.groups()
                if strategy_id in seen:
                    raise ValueError(
                        f"Duplicate chunking strategy option '{strategy_id}' "
                        f"in {source_file}."
                    )
                seen.add(strategy_id)
                options.append({
                    "id": strategy_id,
                    "description": description,
                    "source_file": source_file.name,
                })

    if not options:
        raise ValueError(
            f"No chunking strategy option comments found in {chunking_dir}."
        )
    return options


def _classify_page(text: str, image_count: int, table_rows: list[dict[str, Any]]) -> list[str]:
    labels = []
    character_count = len(text.strip())
    word_count = len(text.split())

    if character_count >= DEFAULT_TEXT_HEAVY_CHARACTERS or word_count >= 120:
        labels.append("text_heavy")

    non_empty_lines = [
        line for line in text.splitlines() if line.strip()
    ]
    if (
        len(table_rows) >= 3
        and len(table_rows) / max(len(non_empty_lines), 1) >= DEFAULT_TABLE_ROW_RATIO
    ):
        labels.append("table_heavy")

    if character_count < DEFAULT_SCANNED_TEXT_CHARACTERS and image_count > 0:
        labels.append("scanned")
    elif character_count < DEFAULT_SCANNED_TEXT_CHARACTERS:
        labels.append("little_or_no_extractable_text")

    if not labels:
        labels.append("mixed_or_standard_text")
    return labels


def _recommend_chunking(
    pages: list[dict[str, Any]],
    available_strategies: list[ChunkingStrategyOption] | None = None,
) -> dict[str, Any]:
    available = (
        available_strategies
        if available_strategies is not None
        else discover_chunking_strategies()
    )
    strategy_descriptions = {
        option["id"]: option["description"] for option in available
    }
    page_count = len(pages)
    denominator = max(page_count, 1)
    scanned_count = sum("scanned" in page["classification"] for page in pages)
    table_count = sum("table_heavy" in page["classification"] for page in pages)
    text_heavy_count = sum("text_heavy" in page["classification"] for page in pages)
    heading_pages = sum(bool(page["headings"]) for page in pages)

    if scanned_count / denominator >= 0.2:
        strategy = "recursive"
        rationale = (
            f"{scanned_count} of {page_count} pages appear image-based with little "
            "extractable text. Perform OCR first, then recursively split the "
            "resulting text at natural boundaries. OCR is a prerequisite, not "
            "part of the chunker."
        )
        parameters = {"ocr_required": True}
    elif table_count / denominator >= 0.2:
        strategy = "structure_aware"
        rationale = (
            f"{table_count} of {page_count} pages appear table-heavy. Use the "
            "structure-aware strategy to keep table lines together when they fit "
            "within a chunk; very long lines may still be split."
        )
        parameters = {
            "ocr_required": False,
            "preserve_table_lines_when_they_fit": True,
        }
    elif heading_pages / denominator >= 0.3 and text_heavy_count > 0:
        strategy = "structure_aware"
        rationale = (
            f"{heading_pages} of {page_count} pages contain heading-like lines "
            "and the document is primarily extractable text. Split at detected "
            "headings and page boundaries."
        )
        parameters = {"ocr_required": False}
    else:
        strategy = "fixed_size"
        rationale = (
            "The document has no strong scanned-page, table-heavy, or consistent "
            "heading structure signal. Fixed-size overlapping text chunks provide "
            "a reasonable default."
        )
        parameters = {
            "ocr_required": False,
            "target_chunk_size_characters": CHUNK_SIZE,
            "chunk_overlap_characters": CHUNK_OVERLAP,
        }

    if strategy not in strategy_descriptions:
        raise ValueError(
            f"Recommended strategy '{strategy}' is not among the discovered "
            "chunking strategy options."
        )

    return {
        "strategy": strategy,
        "description": strategy_descriptions[strategy],
        "rationale": rationale,
        "parameters": parameters,
        "available_strategies": available,
        "counts": {
            "scanned_pages": scanned_count,
            "table_heavy_pages": table_count,
            "text_heavy_pages": text_heavy_count,
            "pages_with_headings": heading_pages,
        },
        "heuristics": {
            "text_heavy_min_characters": DEFAULT_TEXT_HEAVY_CHARACTERS,
            "scanned_max_characters": DEFAULT_SCANNED_TEXT_CHARACTERS,
            "table_heavy_min_rows": 3,
            "table_heavy_min_row_ratio": DEFAULT_TABLE_ROW_RATIO,
        },
    }


def analyze_document(pdf_path: Path) -> dict[str, Any]:
    with trace_operation(f"Document profiling: {pdf_path}"):
        reader = PdfReader(str(pdf_path))
        pages = []
        total_characters = 0
        total_words = 0

        for page_number, page in enumerate(reader.pages, start=1):
            with trace_operation(f"Extract page {page_number}"):
                text = page.extract_text() or ""
                layout_text = page.extract_text(extraction_mode="layout") or text
                image_count = len(page.images)

            lines = text.splitlines()
            headings = _detect_headings(lines)
            list_items = _detect_lists(lines)
            table_rows = _table_row_candidates(layout_text)
            classification = _classify_page(text, image_count, table_rows)
            character_count = len(text.strip())
            word_count = len(text.split())
            total_characters += character_count
            total_words += word_count

            pages.append({
                "page_number": page_number,
                "classification": classification,
                "metrics": {
                    "characters": character_count,
                    "words": word_count,
                    "non_empty_lines": sum(bool(line.strip()) for line in lines),
                    "embedded_images": image_count,
                    "table_like_rows": len(table_rows),
                },
                "headings": headings,
                "lists": list_items,
                "table_row_candidates": table_rows,
                "extracted_text": text,
            })

        classification_counts = Counter(
            label
            for page in pages
            for label in page["classification"]
        )
        available_strategies = discover_chunking_strategies()
        return {
            "report_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "document": {
                "file_name": pdf_path.name,
                "file_path": str(pdf_path.resolve()),
                "file_size_bytes": pdf_path.stat().st_size,
                "page_count": len(pages),
                "total_extracted_characters": total_characters,
                "total_extracted_words": total_words,
            },
            "classification_summary": dict(classification_counts),
            "document_structure": {
                "headings": [
                    {"page": page["page_number"], **heading}
                    for page in pages
                    for heading in page["headings"]
                ],
                "list_item_count": sum(len(page["lists"]) for page in pages),
                "table_like_row_count": sum(
                    page["metrics"]["table_like_rows"] for page in pages
                ),
            },
            "pages": pages,
            "chunking_recommendation": _recommend_chunking(
                pages,
                available_strategies,
            ),
            "limitations": [
                "Page classifications and structure detection use text-layout heuristics; "
                "they are indicators, not semantic guarantees.",
                "A scanned-page label requires both very little extracted text and at "
                "least one embedded image. OCR is not performed.",
                "Table-like rows are inferred from aligned whitespace, tabs, or pipe "
                "delimiters; visually drawn table borders may not be detected.",
            ],
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile a PDF and recommend an appropriate RAG chunking strategy."
    )
    parser.add_argument("pdf", type=Path, help="PDF document to analyze")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="JSON report path (default: <pdf-name>_analysis.json beside the PDF)",
    )
    args = parser.parse_args()
    pdf_path = args.pdf.expanduser().resolve()

    if not pdf_path.is_file():
        parser.error(f"PDF file does not exist: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        parser.error(f"Expected a PDF file: {pdf_path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else pdf_path.with_name(f"{pdf_path.stem}_analysis.json")
    )

    try:
        report = analyze_document(pdf_path)
        with trace_operation(f"Write analysis report: {output_path}"):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    except Exception as exc:
        print(
            f"Document analysis failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"Pages analyzed: {report['document']['page_count']}")
    print(f"JSON report: {output_path}")
    print(
        "Recommended chunking strategy: "
        f"{report['chunking_recommendation']['strategy']}"
    )
    print(f"Reason: {report['chunking_recommendation']['rationale']}")
    print("Available chunking strategies:")
    for option in report["chunking_recommendation"]["available_strategies"]:
        print(f"- {option['id']}: {option['description']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
