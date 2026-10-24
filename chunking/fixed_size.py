# Chunking strategy option: fixed_size - Fixed-length character windows with configurable overlap.
from __future__ import annotations

from collections.abc import Sequence

from config import CHUNK_OVERLAP, CHUNK_SIZE
from chunking.base import Chunk, ChunkingStrategy, Page, iter_text_units, make_chunk


class FixedSizeChunker(ChunkingStrategy):
    """Create character-bounded chunks with an optional character overlap."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be between 0 and chunk_size - 1.")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, pages: Sequence[Page]) -> list[Chunk]:
        units = iter_text_units(pages)
        chunks: list[Chunk] = []
        page_numbers = dict.fromkeys(unit["page"] for unit in units)
        for page_number in page_numbers:
            page_units = [unit for unit in units if unit["page"] == page_number]
            source = "".join(unit["content"] for unit in page_units)
            start = 0
            while start < len(source):
                end = min(start + self.chunk_size, len(source))
                raw_content = source[start:end]
                leading_trim = len(raw_content) - len(raw_content.lstrip())
                trailing_trim = len(raw_content) - len(raw_content.rstrip())
                content_start = start + leading_trim
                content_end = end - trailing_trim
                content = source[content_start:content_end]
                if content:
                    chunks.append(make_chunk(
                        content=content,
                        page_start=page_number,
                        page_end=page_number,
                        section_path=page_units[0]["section_path"],
                        chunk_index=len(chunks),
                    ))
                if end == len(source):
                    break
                start = end - self.overlap
        return chunks
