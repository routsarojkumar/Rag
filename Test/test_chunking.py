from __future__ import annotations

import unittest

from chunking import (
    FixedSizeChunker,
    ParentChildChunker,
    RecursiveChunker,
    SemanticChunker,
    StructureAwareChunker,
    pages_from_analysis_report,
)


REQUIRED_FIELDS = {
    "content",
    "page_start",
    "page_end",
    "section_path",
    "chunk_index",
    "parent_id",
}


class ChunkingStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pages = [
            {
                "page": 1,
                "text": (
                    "1 Introduction\n"
                    "First sentence discusses routing. Second sentence discusses routes.\n"
                    "- Route A\n"
                    "- Route B\n"
                ),
            },
            {
                "page": 2,
                "text": "2 Security\nSecurity procedures are documented here.",
            },
        ]

    def assert_common_chunks(self, chunks: list[dict]) -> None:
        self.assertTrue(chunks)
        for index, chunk in enumerate(chunks):
            self.assertTrue(REQUIRED_FIELDS.issubset(chunk))
            self.assertTrue(chunk["content"])
            self.assertGreaterEqual(chunk["page_start"], 1)
            self.assertGreaterEqual(chunk["page_end"], chunk["page_start"])
            self.assertEqual(chunk["chunk_index"], index)
            self.assertIsInstance(chunk["section_path"], list)

    def test_fixed_chunks_respect_size_overlap_and_page_range(self) -> None:
        chunks = FixedSizeChunker(chunk_size=36, overlap=6).chunk(self.pages)
        self.assert_common_chunks(chunks)
        self.assertTrue(all(len(chunk["content"]) <= 36 for chunk in chunks))
        self.assertEqual(chunks[-1]["page_end"], 2)

    def test_recursive_chunks_are_bounded_and_keep_section_path(self) -> None:
        chunks = RecursiveChunker(chunk_size=45, overlap=0).chunk(self.pages)
        self.assert_common_chunks(chunks)
        self.assertTrue(all(len(chunk["content"]) <= 45 for chunk in chunks))
        self.assertIn("1 Introduction", chunks[0]["section_path"])

    def test_recursive_chunks_remain_bounded_when_overlap_cannot_fit_next_unit(self) -> None:
        pages = [{"page": 1, "text": f"{'A' * 35}\n{'B' * 35}\n{'C' * 35}"}]
        chunks = RecursiveChunker(chunk_size=40, overlap=10).chunk(pages)
        self.assert_common_chunks(chunks)
        self.assertTrue(all(len(chunk["content"]) <= 40 for chunk in chunks))

    def test_structure_aware_splits_on_section_and_page(self) -> None:
        chunks = StructureAwareChunker(chunk_size=200).chunk(self.pages)
        self.assert_common_chunks(chunks)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["section_path"], ["1 Introduction"])
        self.assertEqual(chunks[-1]["page_start"], 2)

    def test_semantic_chunks_use_injected_embedder_and_split_dissimilar_text(self) -> None:
        vectors = {
            "Cats purr softly.": [1.0, 0.0],
            "Kittens like naps.": [0.99, 0.01],
            "Engines need fuel.": [0.0, 1.0],
        }
        pages = [{"page": 1, "text": "Cats purr softly. Kittens like naps. Engines need fuel."}]
        chunker = SemanticChunker(
            embedder=lambda text: vectors[text],
            chunk_size=100,
            similarity_threshold=0.8,
        )
        chunks = chunker.chunk(pages)
        self.assert_common_chunks(chunks)
        self.assertEqual(len(chunks), 2)
        self.assertIn("Kittens like naps.", chunks[0]["content"])
        self.assertIn("Engines need fuel.", chunks[1]["content"])

    def test_parent_child_chunks_link_back_to_parent(self) -> None:
        chunks = ParentChildChunker(
            parent_size=100,
            child_size=35,
            child_overlap=5,
        ).chunk(self.pages)
        self.assert_common_chunks(chunks)
        parents = [chunk for chunk in chunks if chunk["chunk_type"] == "parent"]
        children = [chunk for chunk in chunks if chunk["chunk_type"] == "child"]
        self.assertTrue(parents)
        self.assertTrue(children)
        parent_ids = {
            chunk["chunk_id"] for chunk in parents
        }
        self.assertTrue(all(chunk["parent_id"] in parent_ids for chunk in children))
        for child in children:
            parent = next(
                chunk for chunk in parents if chunk["chunk_id"] == child["parent_id"]
            )
            self.assertEqual(child["page_start"], parent["page_start"])
            self.assertEqual(child["page_end"], parent["page_end"])

    def test_analysis_report_pages_are_supported_with_reported_headings(self) -> None:
        pages = pages_from_analysis_report({
            "pages": [{
                "page_number": 3,
                "extracted_text": (
                    "The table of contents appears here.\n"
                    "3.2 Routing policy\n"
                    "Details follow."
                ),
                "headings": [{"line": 2, "text": "3.2 Routing policy"}],
            }],
        })
        chunks = StructureAwareChunker().chunk(pages)
        self.assert_common_chunks(chunks)
        self.assertEqual(chunks[0]["section_path"], [])
        self.assertEqual(chunks[1]["section_path"], ["3.2 Routing policy"])
        self.assertEqual(chunks[1]["page_start"], 3)

    def test_invalid_chunk_sizes_and_bad_pages_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FixedSizeChunker(chunk_size=10, overlap=10)
        with self.assertRaises(ValueError):
            ParentChildChunker(parent_size=10, child_size=11)
        with self.assertRaises(ValueError):
            FixedSizeChunker(chunk_size=10, overlap=0).chunk([
                {"page": 0, "text": "Invalid page"}
            ])


if __name__ == "__main__":
    unittest.main()
