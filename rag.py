from __future__ import annotations

from typing import Any

import chromadb

from config import CHROMA_PATH, COLLECTION_NAME
from generator import NO_ANSWER, AnswerGenerator
from rag_models import RetrievalResult
from retriever import Retriever
from tracing import trace_operation


class SimpleRAG:
    def __init__(self) -> None:
        with trace_operation("Chroma client and collection initialization"):
            self.client = chromadb.PersistentClient(path=CHROMA_PATH)
            self.collection = self.client.get_collection(name=COLLECTION_NAME)

        self.retriever = Retriever(self.collection)
        self.generator = AnswerGenerator()

    def embed_query(self, query: str) -> list[float]:
        return self.retriever.embed_query(query)

    def _standalone_query(
        self,
        query: str,
        history: list[dict[str, str]] | None,
    ) -> str:
        return self.retriever._standalone_query(query, history)

    def retrieve(
        self,
        query: str,
        where: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        result = self.retriever.retrieve(query, where=where, history=history)
        return result.to_trace_dict()

    def build_prompt(
        self,
        query: str,
        results: RetrievalResult,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        return self.generator.build_prompt(query, results, history=history)

    def generate_answer(self, prompt: str) -> str:
        return self.generator.generate_answer(prompt)

    def ask(
        self,
        query: str,
        where: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        with trace_operation("RAG question"):
            result = self.retriever.retrieve(query, where=where, history=history)
            if not result.chunks:
                return NO_ANSWER, result.to_trace_dict()

            prompt = self.build_prompt(
                query,
                result,
                history=history,
            )
            answer = self.generate_answer(prompt)
            return answer, result.to_trace_dict()
