# Chunking strategy option: parent_child - Produces larger parent chunks and linked overlapping child chunks.
from __future__ import annotations

from collections.abc import Sequence

from chunking.base import Chunk, ChunkingStrategy, Page, make_chunk
from chunking.fixed_size import FixedSizeChunker
from chunking.structure_aware import StructureAwareChunker


class ParentChildChunker(ChunkingStrategy):
    """Return larger parent chunks and smaller linked child chunks."""

    def __init__(
        self,
        parent_size: int = 1600,
        child_size: int = 400,
        child_overlap: int = 40,
    ) -> None:
        if parent_size < 1 or child_size < 1:
            raise ValueError("parent_size and child_size must be at least 1.")
        if child_size > parent_size:
            raise ValueError("child_size cannot exceed parent_size.")
        if child_overlap < 0 or child_overlap >= child_size:
            raise ValueError("child_overlap must be between 0 and child_size - 1.")
        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap

    def chunk(self, pages: Sequence[Page]) -> list[Chunk]:
        parents = StructureAwareChunker(chunk_size=self.parent_size).chunk(pages)
        output: list[Chunk] = []
        child_chunker = FixedSizeChunker(
            chunk_size=self.child_size,
            overlap=self.child_overlap,
        )

        for parent in parents:
            parent_id = f"parent-{parent['chunk_index']:05d}"
            output.append(make_chunk(
                content=parent["content"],
                page_start=parent["page_start"],
                page_end=parent["page_end"],
                section_path=parent["section_path"],
                chunk_index=len(output),
                parent_id=None,
                chunk_type="parent",
            ))
            output[-1]["chunk_id"] = parent_id
            child_pages: list[Page] = [{
                "page": parent["page_start"],
                "text": parent["content"],
            }]
            for child in child_chunker.chunk(child_pages):
                child["page_start"] = parent["page_start"]
                child["page_end"] = parent["page_end"]
                child["section_path"] = list(parent["section_path"])
                child["chunk_index"] = len(output)
                child["parent_id"] = parent_id
                child["chunk_type"] = "child"
                output.append(child)
        return output
