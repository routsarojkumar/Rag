from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Generator


logger = logging.getLogger("simple_rag")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


@dataclass
class OperationTrace:
    input_tokens: int | None = None
    output_tokens: int | None = None
    retrieved_chunks: int | None = None

    def record_retrieved_chunks(self, results: dict) -> None:
        documents = results.get("documents")
        self.retrieved_chunks = len(documents[0]) if documents else 0

    def record_ollama_usage(self, response: Any) -> None:
        if hasattr(response, "model_dump"):
            response_data = response.model_dump()
        elif isinstance(response, dict):
            response_data = response
        else:
            return

        input_tokens = response_data.get("prompt_eval_count")
        output_tokens = response_data.get("eval_count")

        if isinstance(input_tokens, int):
            self.input_tokens = input_tokens
        if isinstance(output_tokens, int):
            self.output_tokens = output_tokens

    def format_token_usage(self) -> str:
        details = ""
        if self.retrieved_chunks is not None:
            details += f" retrieved_chunks={self.retrieved_chunks}"

        if self.input_tokens is None and self.output_tokens is None:
            return details

        input_count = self.input_tokens or 0
        output_count = self.output_tokens or 0
        total_count = input_count + output_count
        return details + (
            f" input_tokens={input_count}"
            f" output_tokens={output_count}"
            f" total_tokens={total_count}"
        )


@contextmanager
def trace_operation(operation: str) -> Generator[OperationTrace, None, None]:
    started = perf_counter()
    trace = OperationTrace()
    logger.info("START %s", operation)
    try:
        yield trace
    except Exception as exc:
        elapsed = perf_counter() - started
        logger.exception(
            "FAILED %s after %.3f sec (%s: %s)",
            operation,
            elapsed,
            type(exc).__name__,
            exc,
        )
        raise
    else:
        elapsed = perf_counter() - started
        logger.info(
            "DONE %s in %.3f sec%s",
            operation,
            elapsed,
            trace.format_token_usage(),
        )
