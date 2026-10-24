from __future__ import annotations

import unittest
from pathlib import Path

from document_analyzer import (
    _recommend_chunking,
    discover_chunking_strategies,
)


class AnalyzerChunkingRecommendationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.options = discover_chunking_strategies()
        self.option_ids = {option["id"] for option in self.options}

    def _page(
        self,
        classification: list[str],
        headings: list[dict] | None = None,
    ) -> dict:
        return {
            "classification": classification,
            "headings": headings or [],
        }

    def test_discovers_all_implemented_chunkers_from_comments(self) -> None:
        expected = {
            "fixed_size",
            "recursive",
            "semantic",
            "structure_aware",
            "parent_child",
        }
        self.assertEqual(self.option_ids, expected)
        for option in self.options:
            source_path = Path(__file__).resolve().parents[1] / "chunking" / option["source_file"]
            self.assertTrue(source_path.is_file())
            self.assertTrue(option["description"])

    def test_each_chunking_module_has_a_strategy_or_role_comment(self) -> None:
        chunking_dir = Path(__file__).resolve().parents[1] / "chunking"
        for source_path in chunking_dir.glob("*.py"):
            with self.subTest(source=source_path.name):
                first_lines = source_path.read_text(encoding="utf-8").splitlines()[:3]
                self.assertTrue(
                    any(
                        "Chunking strategy option:" in line
                        or "Chunking package role:" in line
                        for line in first_lines
                    )
                )

    def test_recommendations_always_select_a_discovered_option(self) -> None:
        cases = [
            (
                [self._page(["scanned"])],
                "recursive",
            ),
            (
                [self._page(["table_heavy"])],
                "structure_aware",
            ),
            (
                [self._page(["text_heavy"], [{"text": "Introduction"}])],
                "structure_aware",
            ),
            (
                [self._page(["mixed_or_standard_text"])],
                "fixed_size",
            ),
        ]
        for pages, expected in cases:
            with self.subTest(expected=expected):
                recommendation = _recommend_chunking(pages, self.options)
                self.assertIn(recommendation["strategy"], self.option_ids)
                self.assertEqual(recommendation["strategy"], expected)
                self.assertEqual(
                    len(recommendation["available_strategies"]),
                    len(self.options),
                )
                self.assertTrue(recommendation["description"])


if __name__ == "__main__":
    unittest.main()
