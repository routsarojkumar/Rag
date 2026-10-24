from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ingest


class FakeCollection:
    def __init__(self) -> None:
        self.records = {}

    def upsert(self, **batch) -> None:
        for index, record_id in enumerate(batch["ids"]):
            self.records[record_id] = {
                "ids": [record_id],
                "embeddings": [batch["embeddings"][index]],
                "documents": [batch["documents"][index]],
                "metadatas": [batch["metadatas"][index]],
            }

    def get(self, where: dict, include: list[str]) -> dict:
        matching = [
            record
            for record in self.records.values()
            if record["metadatas"][0].get("source") == where["source"]
        ]
        return {
            "ids": [record["ids"][0] for record in matching],
            "metadatas": [record["metadatas"][0] for record in matching],
        }

    def delete(self, ids: list[str]) -> None:
        for record_id in ids:
            self.records.pop(record_id, None)


def analysis_report() -> dict:
    return {
        "document": {"page_count": 1},
        "pages": [{
            "page_number": 1,
            "extracted_text": "1 Introduction\nA short test document for ingestion.",
            "headings": [{
                "line": 1,
                "text": "1 Introduction",
                "detection": "numbered_or_labeled",
            }],
            "classification": ["mixed_or_standard_text"],
            "metrics": {
                "table_like_rows": 0,
            },
            "lists": [],
            "table_row_candidates": [],
        }],
        "chunking_recommendation": {
            "strategy": "structure_aware",
            "description": "Splits at detected headings.",
            "rationale": "The page contains a heading.",
            "parameters": {"ocr_required": False},
            "available_strategies": [
                {"id": "fixed_size", "description": "Fixed chunks.", "source_file": "fixed_size.py"},
                {"id": "structure_aware", "description": "Heading chunks.", "source_file": "structure_aware.py"},
            ],
        },
    }


class IngestionPipelineTests(unittest.TestCase):
    def test_analyze_confirm_chunk_and_store_in_selected_strategy(self) -> None:
        collection = FakeCollection()
        with tempfile.TemporaryDirectory() as temp_directory:
            pdf_path = Path(temp_directory) / "document.pdf"
            pdf_path.write_bytes(b"%PDF test fixture")
            with (
                patch("ingest.analyze_document", return_value=analysis_report()) as analyzer,
                patch("ingest.get_embedding", return_value=[0.1, 0.2]),
            ):
                count = ingest.ingest_pdf(
                    str(pdf_path),
                    collection,
                    input_fn=lambda _prompt: "fixed_size",
                    output_fn=lambda _message: None,
                )

            analyzer.assert_called_once_with(pdf_path)
            self.assertTrue((Path(temp_directory) / "document_analysis.json").exists())
            self.assertEqual(count, len(collection.records))
            self.assertGreater(count, 0)
            for record in collection.records.values():
                metadata = record["metadatas"][0]
                self.assertEqual(metadata["chunking_strategy"], "fixed_size")
                self.assertEqual(metadata["source"], "document.pdf")
                self.assertEqual(metadata["page_start"], 1)
                self.assertEqual(metadata["page_end"], 1)
                self.assertEqual(metadata["chunk"], 0)
                self.assertEqual(metadata["chunk_index"], 0)
                self.assertIn("section_path", metadata)

    def test_cancelling_selection_does_not_embed_or_write(self) -> None:
        collection = FakeCollection()
        with tempfile.TemporaryDirectory() as temp_directory:
            pdf_path = Path(temp_directory) / "document.pdf"
            pdf_path.write_bytes(b"%PDF test fixture")
            with (
                patch("ingest.analyze_document", return_value=analysis_report()),
                patch("ingest.get_embedding") as embedder,
            ):
                with self.assertRaises(ingest.IngestionCancelled):
                    ingest.ingest_pdf(
                        str(pdf_path),
                        collection,
                        input_fn=lambda _prompt: "q",
                        output_fn=lambda _message: None,
                    )

            embedder.assert_not_called()
            self.assertFalse(collection.records)

    def test_accepting_recommendation_uses_enter_default(self) -> None:
        prompts = []
        selected = ingest._choose_strategy(
            analysis_report()["chunking_recommendation"],
            analysis_report()["chunking_recommendation"]["available_strategies"],
            input_fn=lambda prompt: prompts.append(prompt) or "",
            output_fn=lambda _message: None,
        )
        self.assertEqual(selected, "structure_aware")
        self.assertEqual(len(prompts), 1)

    def test_explicit_strategy_is_validated(self) -> None:
        collection = FakeCollection()
        with tempfile.TemporaryDirectory() as temp_directory:
            pdf_path = Path(temp_directory) / "document.pdf"
            pdf_path.write_bytes(b"%PDF test fixture")
            with patch("ingest.analyze_document", return_value=analysis_report()):
                with self.assertRaisesRegex(ValueError, "Unknown strategy"):
                    ingest.ingest_pdf(
                        str(pdf_path),
                        collection,
                        strategy="semantic",
                        output_fn=lambda _message: None,
                    )

    def test_reingesting_same_chunks_upserts_without_duplicates(self) -> None:
        collection = FakeCollection()
        chunks = [{
            "content": "Stable content.",
            "page_start": 1,
            "page_end": 1,
            "section_path": ["Introduction"],
            "chunk_index": 0,
            "parent_id": None,
            "strategy": "fixed_size",
        }]
        with patch("ingest.get_embedding", return_value=[0.1, 0.2]):
            first_count = ingest._store_chunks("document.pdf", chunks, collection)
            first_ids = set(collection.records)
            second_count = ingest._store_chunks("document.pdf", chunks, collection)

        self.assertEqual(first_count, 1)
        self.assertEqual(second_count, 1)
        self.assertEqual(set(collection.records), first_ids)
        self.assertEqual(len(collection.records), 1)

    def test_reingesting_removes_stale_chunks_after_success(self) -> None:
        collection = FakeCollection()
        chunks = [{
            "content": "Current content.",
            "page_start": 1,
            "page_end": 1,
            "section_path": [],
            "chunk_index": 0,
            "parent_id": None,
            "strategy": "fixed_size",
        }]
        with patch("ingest.get_embedding", return_value=[0.1, 0.2]):
            ingest._store_chunks("document.pdf", [dict(chunks[0], chunk_index=1)], collection)
            ingest._store_chunks("document.pdf", chunks, collection)

        self.assertEqual(len(collection.records), 1)
        self.assertEqual(
            next(iter(collection.records.values()))["documents"][0],
            "Current content.",
        )


if __name__ == "__main__":
    unittest.main()
