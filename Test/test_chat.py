from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import chat


class FakeRAG:
    def __init__(self) -> None:
        self.histories = []

    def ask(self, query: str, history: list[dict[str, str]]) -> tuple[str, dict]:
        self.histories.append(list(history))
        return "A cited answer.", {
            "sources": [{
                "source": "guide.pdf",
                "page_start": 2,
                "page_end": 3,
                "section_path": ["Operations"],
                "distance": 0.25,
            }],
        }


class ChatTests(unittest.TestCase):
    def test_displays_sources_and_carries_history_to_followups(self) -> None:
        fake_rag = FakeRAG()
        inputs = iter(["first question", "follow-up", "exit"])
        output = io.StringIO()
        with (
            patch("chat.SimpleRAG", return_value=fake_rag),
            patch("builtins.input", side_effect=lambda _prompt: next(inputs)),
            redirect_stdout(output),
        ):
            chat.main()

        rendered = output.getvalue()
        self.assertIn("guide.pdf (pages 2-3, section Operations; distance 0.250)", rendered)
        self.assertEqual(fake_rag.histories[0], [])
        self.assertEqual(
            fake_rag.histories[1],
            [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "A cited answer."},
            ],
        )


if __name__ == "__main__":
    unittest.main()
