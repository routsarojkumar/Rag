# Chunking strategy option: recursive - Recursively splits at paragraph, line, sentence, and word boundaries.
from __future__ import annotations

from collections.abc import Sequence

from config import CHUNK_OVERLAP, CHUNK_SIZE
from chunking.base import Chunk, ChunkingStrategy, Page, TextUnit, iter_text_units


class RecursiveChunker(ChunkingStrategy):
    """Split by paragraphs, lines, sentences, words, then characters."""

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

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        for separator in ("\n\n", "\n", ". ", " ", ""):
            if separator:
                pieces = text.split(separator)
                if len(pieces) == 1:
                    continue
                parts = [
                    piece + (separator if index < len(pieces) - 1 else "")
                    for index, piece in enumerate(pieces)
                    if piece
                ]
            else:
                parts = list(text)

            if len(parts) > 1:
                result: list[str] = []
                for part in parts:
                    result.extend(self._split_text(part))
                return result
        return [
            text[index:index + self.chunk_size]
            for index in range(0, len(text), self.chunk_size)
        ]

    def chunk(self, pages: Sequence[Page]) -> list[Chunk]:
        units = iter_text_units(pages)
        split_units: list[TextUnit] = []
        for unit in units:
            for part in self._split_text(unit["content"]):
                split_units.append({
                    "content": part,
                    "page": unit["page"],
                    "section_path": unit["section_path"],
                })

        chunks: list[Chunk] = []
        current: list[TextUnit] = []
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
            if self.overlap and current:
                retained: list[TextUnit] = []
                retained_size = 0
                for unit in reversed(current):
                    if retained_size + len(unit["content"]) > self.overlap:
                        break
                    retained.insert(0, unit)
                    retained_size += len(unit["content"])
                current = retained
                current_size = retained_size
            else:
                current = []
                current_size = 0

        for unit in split_units:
            length = len(unit["content"])
            if current and current_size + length > self.chunk_size:
                flush()
                if current_size + length > self.chunk_size:
                    current = []
                    current_size = 0
            if length > self.chunk_size:
                for index in range(0, length, self.chunk_size):
                    piece = dict(unit)
                    piece["content"] = unit["content"][index:index + self.chunk_size]
                    current.append(piece)
                    current_size += len(piece["content"])
                    flush()
                continue
            current.append(unit)
            current_size += length
        if current:
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
        return chunks
