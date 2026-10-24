# Chunking strategy option: structure_aware - Splits on detected headings and page boundaries while keeping short lines together.
from __future__ import annotations

from collections.abc import Sequence

from config import CHUNK_SIZE
from chunking.base import Chunk, ChunkingStrategy, Page, iter_text_units


class StructureAwareChunker(ChunkingStrategy):
    """Keep heading sections and list items together where the size permits."""

    def __init__(self, chunk_size: int = CHUNK_SIZE) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")
        self.chunk_size = chunk_size

    def chunk(self, pages: Sequence[Page]) -> list[Chunk]:
        units = iter_text_units(pages)
        chunks: list[Chunk] = []
        current = []
        current_size = 0

        def flush() -> None:
            nonlocal current, current_size
            if not current:
                return
            content = "".join(unit["content"] for unit in current).strip()
            if content:
                chunks.append({
                    "content": content,
                    "page_start": current[0]["page"],
                    "page_end": current[-1]["page"],
                    "section_path": current[0]["section_path"],
                    "chunk_index": len(chunks),
                    "parent_id": None,
                })
            current = []
            current_size = 0

        for unit in units:
            text = unit["content"]
            section_changed = (
                current and current[-1]["section_path"] != unit["section_path"]
            )
            page_changed = current and current[-1]["page"] != unit["page"]
            would_overflow = current_size + len(text) > self.chunk_size
            if current and (section_changed or page_changed or would_overflow):
                flush()

            if len(text) > self.chunk_size:
                for start in range(0, len(text), self.chunk_size):
                    part = dict(unit)
                    part["content"] = text[start:start + self.chunk_size]
                    current.append(part)
                    current_size += len(part["content"])
                    flush()
                continue
            current.append(unit)
            current_size += len(text)
        flush()
        return chunks
