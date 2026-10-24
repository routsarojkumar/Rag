from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import chromadb
import ollama
from pypdf import PdfReader

from chunking import (
    Chunk,
    ChunkingStrategy,
    FixedSizeChunker,
    Page,
    ParentChildChunker,
    RecursiveChunker,
    SemanticChunker,
    StructureAwareChunker,
    pages_from_analysis_report,
)
from config import (
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)
from document_analyzer import (
    ChunkingStrategyOption,
    analyze_document,
    discover_chunking_strategies,
)
from tracing import OperationTrace, trace_operation


DOCUMENTS_DIR = "./documents"
ANALYSIS_REPORT_SUFFIX = "_analysis.json"
InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


class IngestionCancelled(Exception):
    """Raised when the user declines to choose a chunking strategy."""


def load_pdf(file_path: str) -> list[Page]:
    """Extract text from a PDF, retaining each page's original number."""
    with trace_operation(f"PDF extraction: {file_path}"):
        reader = PdfReader(file_path)
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": page_number, "text": text})
        return pages


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Legacy fixed-size helper retained for callers and existing test fixtures."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1.")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def get_embedding(text: str, trace: OperationTrace | None = None) -> list[float]:
    """Generate an embedding using the configured Ollama model."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=text)
    if trace is not None:
        trace.record_ollama_usage(response)
    return response["embeddings"][0]


def _semantic_embedding(text: str) -> Sequence[float]:
    with trace_operation("Semantic chunk embedding") as trace:
        return get_embedding(text, trace=trace)


def _make_chunker(strategy: str) -> ChunkingStrategy:
    chunkers = {
        "fixed_size": lambda: FixedSizeChunker(
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        ),
        "recursive": lambda: RecursiveChunker(
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        ),
        "semantic": lambda: SemanticChunker(
            embedder=_semantic_embedding,
            chunk_size=CHUNK_SIZE,
        ),
        "structure_aware": lambda: StructureAwareChunker(
            chunk_size=CHUNK_SIZE,
        ),
        "parent_child": lambda: ParentChildChunker(
            parent_size=CHUNK_SIZE * 2,
            child_size=CHUNK_SIZE,
            child_overlap=CHUNK_OVERLAP,
        ),
    }
    try:
        return chunkers[strategy]()
    except KeyError as exc:
        raise ValueError(f"Unsupported chunking strategy: {strategy}") from exc


def _choose_strategy(
    recommendation: dict[str, Any],
    options: list[ChunkingStrategyOption],
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> str:
    recommended = recommendation["strategy"]
    available = {option["id"] for option in options}
    if recommended not in available:
        raise ValueError(
            f"Analyzer recommended unavailable strategy '{recommended}'."
        )

    output_fn(f"\nSuggested chunking strategy: {recommended}")
    output_fn(recommendation["description"])
    output_fn(f"Reason: {recommendation['rationale']}")
    output_fn("Available strategies:")
    for option in options:
        marker = " (recommended)" if option["id"] == recommended else ""
        output_fn(f"  {option['id']}: {option['description']}{marker}")
    output_fn(
        "Press Enter to confirm the recommendation, enter another strategy ID "
        "to select it, or enter 'q' to cancel."
    )

    while True:
        selected = input_fn(f"Chunking strategy [{recommended}]: ").strip()
        if not selected:
            return recommended
        if selected.casefold() in {"q", "quit", "cancel"}:
            raise IngestionCancelled("Chunking strategy selection was cancelled.")
        if selected in available:
            return selected
        output_fn(
            f"Unknown strategy '{selected}'. Choose one of: "
            f"{', '.join(sorted(available))}, or 'q' to cancel."
        )


def _analyze_pdf(file_path: Path) -> tuple[dict[str, Any], list[Page], Path]:
    report = analyze_document(file_path)
    report_path = file_path.with_name(
        f"{file_path.stem}{ANALYSIS_REPORT_SUFFIX}"
    )
    with trace_operation(f"Write analysis report: {report_path}"):
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return report, pages_from_analysis_report(report), report_path


def _prepare_chunks(
    pages: Sequence[Page],
    strategy: str,
) -> list[Chunk]:
    chunker = _make_chunker(strategy)
    with trace_operation(f"Chunking document using {strategy}"):
        chunks = chunker.chunk(pages)
    for index, chunk in enumerate(chunks):
        chunk["chunk_index"] = index
    return chunks


def _store_chunks(
    filename: str,
    chunks: Sequence[Chunk],
    collection: Any,
) -> int:
    document_ids_by_chunk_id: dict[str, str] = {}
    for chunk in chunks:
        if chunk.get("chunk_type") == "parent":
            document_ids_by_chunk_id[chunk["chunk_id"]] = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                "\0".join((
                    filename,
                    chunk.get("strategy", ""),
                    "parent",
                    str(chunk["chunk_index"]),
                    chunk["content"],
                )),
            ))

    existing_records = collection.get(
        where={"source": filename},
        include=["metadatas"],
    )
    existing_ids = set(existing_records.get("ids") or [])

    total_chunks = 0
    indexed_ids = set()
    chunk_index_by_page: dict[int, int] = {}
    for chunk in chunks:
        chunk_type = chunk.get("chunk_type", "standard")
        page_chunk_index = chunk_index_by_page.get(chunk["page_start"], 0)
        chunk_index_by_page[chunk["page_start"]] = page_chunk_index + 1
        operation = (
            f"{filename}, pages {chunk['page_start']}-{chunk['page_end']}, "
            f"{chunk_type} chunk {chunk['chunk_index'] + 1}"
        )
        with trace_operation(f"Embedding: {operation}") as trace:
            embedding = get_embedding(chunk["content"], trace=trace)

        document_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            "\0".join((
                filename,
                chunk.get("strategy", ""),
                chunk_type,
                str(chunk["chunk_index"]),
                chunk["content"],
            )),
        ))
        indexed_ids.add(document_id)
        parent_id = chunk.get("parent_id")
        metadata: dict[str, str | int] = {
            "source": filename,
            "page": chunk["page_start"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "chunk": page_chunk_index,
            "chunk_index": chunk["chunk_index"],
            "section_path": json.dumps(chunk["section_path"], ensure_ascii=False),
            "chunking_strategy": chunk.get("strategy", ""),
            "chunk_type": chunk_type,
        }
        if parent_id is not None:
            try:
                metadata["parent_id"] = document_ids_by_chunk_id[parent_id]
            except KeyError as exc:
                raise ValueError(
                    f"Child chunk references unknown parent ID: {parent_id}"
                ) from exc
        with trace_operation(f"Chroma write: {operation}"):
            collection.upsert(
                ids=[document_id],
                embeddings=[embedding],
                documents=[chunk["content"]],
                metadatas=[metadata],
            )
        total_chunks += 1

    stale_ids = existing_ids - indexed_ids
    if stale_ids:
        with trace_operation(
            f"Remove {len(stale_ids)} stale chunks for {filename}"
        ):
            collection.delete(ids=sorted(stale_ids))
    return total_chunks


def ingest_pdf(
    file_path: str,
    collection: Any,
    strategy: str | None = None,
    *,
    confirm_strategy: bool = True,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> int:
    """Analyze, confirm/select a strategy, chunk, embed, and index one PDF."""
    pdf_path = Path(file_path).expanduser().resolve()
    filename = pdf_path.name
    with trace_operation(f"PDF ingestion: {filename}"):
        with trace_operation(f"Document analysis: {filename}"):
            report, pages, report_path = _analyze_pdf(pdf_path)
        recommendation = report["chunking_recommendation"]
        available = recommendation["available_strategies"]

        if strategy is None:
            selected_strategy = (
                _choose_strategy(
                    recommendation,
                    available,
                    input_fn=input_fn,
                    output_fn=output_fn,
                )
                if confirm_strategy
                else recommendation["strategy"]
            )
        else:
            valid_strategy_ids = {option["id"] for option in available}
            if strategy not in valid_strategy_ids:
                raise ValueError(
                    f"Unknown strategy '{strategy}'. Available strategies: "
                    f"{', '.join(sorted(valid_strategy_ids))}"
                )
            selected_strategy = strategy

        if not any(page["text"].strip() for page in pages):
            raise ValueError(
                f"No extractable text found in {filename}. Run OCR before ingestion."
            )
        if recommendation["parameters"].get("ocr_required"):
            output_fn(
                "Warning: analysis detected scanned pages. OCR is required for "
                "those pages to be searchable; this application does not run OCR."
            )

        output_fn(f"Confirmed chunking strategy: {selected_strategy}")
        output_fn(f"Analysis report: {report_path}")
        chunks = _prepare_chunks(pages, selected_strategy)
        for chunk in chunks:
            chunk["strategy"] = selected_strategy
        output_fn(f"Chunking complete: {filename}, {len(chunks)} chunk(s)")
        return _store_chunks(filename, chunks, collection)


def _parse_args() -> argparse.Namespace:
    available = discover_chunking_strategies()
    parser = argparse.ArgumentParser(
        description=(
            "Analyze each PDF, confirm a chunking strategy, and store its chunks "
            "in ChromaDB."
        )
    )
    parser.add_argument(
        "--strategy",
        choices=[option["id"] for option in available],
        help="Use this strategy for every PDF instead of prompting per document.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Automatically accept each analyzer recommendation.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.strategy and args.yes:
        print("--strategy and --yes cannot be used together.", file=sys.stderr)
        return 2

    print("Starting ingestion...")
    with trace_operation("Chroma client and collection initialization"):
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)

    total_chunks = 0
    try:
        with trace_operation("Ingest all PDFs"):
            for filename in sorted(os.listdir(DOCUMENTS_DIR)):
                if not filename.lower().endswith(".pdf"):
                    continue
                file_path = os.path.join(DOCUMENTS_DIR, filename)
                print(f"\nProcessing: {filename}")
                total_chunks += ingest_pdf(
                    file_path,
                    collection,
                    strategy=args.strategy,
                    confirm_strategy=not args.yes,
                )
    except IngestionCancelled as exc:
        print(str(exc))
        return 0
    except Exception as exc:
        print(
            f"Ingestion failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print("\n-----------------------------")
    print("Ingestion completed")
    print(f"Total chunks: {total_chunks}")
    print("-----------------------------")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
