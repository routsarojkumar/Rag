from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def _section_path(metadata: dict[str, Any]) -> list[str]:
    value = metadata.get("section_path")
    if isinstance(value, str) and value:
        parts = json.loads(value)
        if not isinstance(parts, list) or not all(
            isinstance(part, str) for part in parts
        ):
            raise ValueError("Chunk metadata contains an invalid section path.")
        return parts
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        return value
    return []


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    matched_text: str
    source: str
    page_start: int | str
    page_end: int | str
    section_path: list[str]
    distance: float
    chunk_id: str
    matched_metadata: dict[str, Any] = field(default_factory=dict)
    context_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_match(
        cls,
        chunk_id: str,
        matched_text: str,
        matched_metadata: dict[str, Any],
        text: str,
        context_metadata: dict[str, Any],
        distance: float,
    ) -> RetrievedChunk:
        page_start = context_metadata.get(
            "page_start",
            context_metadata.get("page", "unknown"),
        )
        return cls(
            text=text,
            matched_text=matched_text,
            source=context_metadata.get("source", "Unknown document"),
            page_start=page_start,
            page_end=context_metadata.get("page_end", page_start),
            section_path=_section_path(context_metadata),
            distance=distance,
            chunk_id=chunk_id,
            matched_metadata=matched_metadata,
            context_metadata=context_metadata,
        )

    def source_summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section_path": self.section_path,
            "distance": self.distance,
        }


@dataclass(frozen=True)
class RetrievalResult:
    search_query: str
    chunks: list[RetrievedChunk]

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "search_query": self.search_query,
            "ids": [[chunk.chunk_id for chunk in self.chunks]],
            "documents": [[chunk.matched_text for chunk in self.chunks]],
            "metadatas": [[chunk.matched_metadata for chunk in self.chunks]],
            "distances": [[chunk.distance for chunk in self.chunks]],
            "context_documents": [[chunk.text for chunk in self.chunks]],
            "context_metadatas": [[chunk.context_metadata for chunk in self.chunks]],
            "context_distances": [[chunk.distance for chunk in self.chunks]],
            "sources": [chunk.source_summary() for chunk in self.chunks],
        }
