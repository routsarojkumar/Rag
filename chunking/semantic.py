# Chunking strategy option: semantic - Groups embedding-similar sentences and needs a caller-supplied embedder.
from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence

from config import CHUNK_SIZE
from chunking.base import Chunk, ChunkingStrategy, Page, TextUnit, iter_text_units


Embedder = Callable[[str], Sequence[float]]
_SENTENCE_PATTERN = re.compile(r".+?(?:[.!?](?=\s|$)|$)", re.DOTALL)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Embedder must return non-empty vectors of matching dimensions.")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _sentence_units(units: Sequence[TextUnit]) -> list[TextUnit]:
    sentences: list[TextUnit] = []
    for unit in units:
        text = unit["content"]
        for match in _SENTENCE_PATTERN.finditer(text):
            sentence = match.group().strip()
            if sentence:
                sentences.append({
                    "content": sentence,
                    "page": unit["page"],
                    "section_path": unit["section_path"],
                })
    return sentences


class SemanticChunker(ChunkingStrategy):
    """Split between sentences when their embedding similarity is low."""

    def __init__(
        self,
        embedder: Embedder,
        chunk_size: int = CHUNK_SIZE,
        similarity_threshold: float = 0.5,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1 and 1.")
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold

    def chunk(self, pages: Sequence[Page]) -> list[Chunk]:
        sentences = _sentence_units(iter_text_units(pages))
        if not sentences:
            return []
        vectors = [self.embedder(sentence["content"]) for sentence in sentences]

        groups: list[list[TextUnit]] = []
        current = [sentences[0]]
        current_size = len(sentences[0]["content"])
        for index in range(1, len(sentences)):
            sentence = sentences[index]
            similarity = _cosine_similarity(vectors[index - 1], vectors[index])
            if (
                similarity < self.similarity_threshold
                or current_size + len(sentence["content"]) + 1 > self.chunk_size
                or sentence["section_path"] != current[-1]["section_path"]
                or sentence["page"] != current[-1]["page"]
            ):
                groups.append(current)
                current = [sentence]
                current_size = len(sentence["content"])
            else:
                current.append(sentence)
                current_size += len(sentence["content"]) + 1
        groups.append(current)

        chunks: list[Chunk] = []
        for group in groups:
            content = " ".join(unit["content"] for unit in group)
            if len(content) <= self.chunk_size:
                chunks.append({
                    "content": content,
                    "page_start": group[0]["page"],
                    "page_end": group[-1]["page"],
                    "section_path": group[0]["section_path"],
                    "chunk_index": len(chunks),
                    "parent_id": None,
                })
                continue

            for start in range(0, len(content), self.chunk_size):
                chunks.append({
                    "content": content[start:start + self.chunk_size].strip(),
                    "page_start": group[0]["page"],
                    "page_end": group[-1]["page"],
                    "section_path": group[0]["section_path"],
                    "chunk_index": len(chunks),
                    "parent_id": None,
                })
        return chunks
